#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        out = np.zeros_like(x)
        return out
    m = float(np.mean(x[valid]))
    s = float(np.std(x[valid]))
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


@dataclass(frozen=True)
class SecrestMask:
    mask: np.ndarray  # True=masked
    seen: np.ndarray  # True=unmasked


def build_secrest_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> SecrestMask:
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    # mask_zeros(tbl) on the W1cov>=cut parent sample.
    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        indices = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            indices[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[indices] = True  # keep Secrest behavior (-1 neighbors)

    # exclude discs
    if exclude_mask_fits:
        from astropy.coordinates import SkyCoord
        from astropy.table import Table
        import astropy.units as u

        tmask = Table.read(exclude_mask_fits)
        if "use" in tmask.colnames:
            tmask = tmask[tmask["use"] == True]  # noqa: E712
        if len(tmask):
            sc = SkyCoord(tmask["ra"], tmask["dec"], unit=u.deg, frame="icrs").galactic
            radius = np.deg2rad(np.asarray(tmask["radius"], dtype=float))
            for lon, lat, rad in zip(sc.l.deg, sc.b.deg, radius, strict=True):
                theta = np.deg2rad(90.0 - float(lat))
                phi = np.deg2rad(float(lon))
                vec = hp.ang2vec(theta, phi)
                disc = hp.query_disc(nside=int(nside), vec=vec, radius=float(rad), inclusive=True, nest=False)
                mask[disc] = True

    # galactic plane cut
    _lon_pix, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return SecrestMask(mask=mask, seen=~mask)


def fit_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    offset: np.ndarray | None,
    max_iter: int,
    beta_init: np.ndarray | None = None,
) -> np.ndarray:
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    off = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)

    if beta_init is None:
        mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
        beta0 = np.zeros(X.shape[1], dtype=float)
        beta0[0] = math.log(mu0)
    else:
        beta0 = np.asarray(beta_init, dtype=float).reshape(X.shape[1])

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = off + X @ beta
        eta = np.clip(eta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        return nll, np.asarray(grad, dtype=float)

    res = minimize(
        lambda b: fun_and_grad(b)[0],
        beta0,
        jac=lambda b: fun_and_grad(b)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    return np.asarray(res.x, dtype=float)


def summarize_b(b: np.ndarray) -> dict[str, Any]:
    b = np.asarray(b, dtype=float)
    D = np.linalg.norm(b, axis=1)
    b_mean = np.mean(b, axis=0)
    return {
        "n": int(b.shape[0]),
        "b_mean": [float(x) for x in b_mean],
        "D_of_b_mean": float(np.linalg.norm(b_mean)),
        "D_mean": float(np.mean(D)),
        "D_std": float(np.std(D)),
        "D_p16": float(np.percentile(D, 16)),
        "D_p50": float(np.percentile(D, 50)),
        "D_p84": float(np.percentile(D, 84)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Injection check for seasonal selection patterns using ecliptic-longitude templates. "
            "Inject a sin/cos(lambda) modulation into the log-intensity field and measure how much dipole amplitude "
            "is recovered if the fit does (or does not) include those templates."
        )
    )
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1-max", type=float, default=16.6)
    ap.add_argument("--w1-min", type=float, default=None)

    ap.add_argument("--n-mocks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-iter", type=int, default=250)

    ap.add_argument("--dipole-amp", type=float, default=0.0, help="Optional injected dipole amplitude in log-intensity.")
    ap.add_argument("--dipole-axis-lb", default="264.021,48.253", help="Injected dipole axis (l,b) in degrees.")

    ap.add_argument(
        "--lon-amps",
        default="0,0.01,0.02,0.03,0.04",
        help="Comma-separated list of sin/cos(lambda) modulation amplitudes (applied in the z-scored basis).",
    )
    ap.add_argument(
        "--lon-phase-deg",
        type=float,
        default=0.0,
        help="Phase in degrees for the injected combination: amp*(cos(phase)*cos_z + sin(phase)*sin_z).",
    )

    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits

    rng = np.random.default_rng(int(args.seed))

    outdir = Path(args.outdir or f"outputs/seasonal_selection_injection_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    nside = int(args.nside)
    npix = int(hp.nside2npix(nside))

    # Load catalog columns.
    with fits.open(args.catalog, memmap=True) as hdul:
        data = hdul[1].data
        w1 = np.asarray(data["w1"], dtype=float)
        w1cov = np.asarray(data["w1cov"], dtype=float)
        l = np.asarray(data["l"], dtype=float)
        b = np.asarray(data["b"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    base &= w1cov >= float(args.w1cov_min)

    ipix_mask_base = hp.ang2pix(
        nside,
        np.deg2rad(90.0 - b[base]),
        np.deg2rad(l[base]),
        nest=False,
    ).astype(np.int64)

    sel = base & (w1 <= float(args.w1_max))
    if args.w1_min is not None:
        sel &= w1 > float(args.w1_min)

    ipix_sel = hp.ang2pix(
        nside,
        np.deg2rad(90.0 - b[sel]),
        np.deg2rad(l[sel]),
        nest=False,
    ).astype(np.int64)

    counts = np.bincount(ipix_sel, minlength=npix).astype(float)

    secrest = build_secrest_mask(
        nside=nside,
        ipix_base=ipix_mask_base,
        exclude_mask_fits=str(args.exclude_mask_fits) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )
    seen = secrest.seen

    lon_pix, lat_pix = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    pix_unit = lb_to_unitvec(lon_pix, lat_pix)
    n_seen = pix_unit[seen]

    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    ecl = sc_pix.barycentricmeanecliptic
    elon = (np.asarray(ecl.lon.deg, dtype=float) % 360.0)
    elat = np.asarray(ecl.lat.deg, dtype=float)
    abs_elat = np.abs(elat)

    # Templates (z-scored on seen pixels).
    abs_elat_z = zscore(abs_elat, seen)[seen]
    sin_z = zscore(np.sin(np.deg2rad(elon)), seen)[seen]
    cos_z = zscore(np.cos(np.deg2rad(elon)), seen)[seen]

    # Base intensity model used for mocks: NO dipole, only abs_elat.
    y_data = counts[seen]
    X_mu = np.column_stack([np.ones_like(y_data), abs_elat_z])
    beta_mu = fit_poisson_glm(X_mu, y_data, offset=None, max_iter=int(args.max_iter))
    log_mu_hat = np.clip(X_mu @ beta_mu, -25.0, 25.0)

    # Recovery design matrices.
    X_fit_base = np.column_stack([np.ones_like(y_data), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2], abs_elat_z])
    X_fit_lon = np.column_stack(
        [
            np.ones_like(y_data),
            n_seen[:, 0],
            n_seen[:, 1],
            n_seen[:, 2],
            abs_elat_z,
            sin_z,
            cos_z,
        ]
    )

    # Optional injected dipole.
    inj_l, inj_b = (float(x) for x in str(args.dipole_axis_lb).split(","))
    inj_axis = lb_to_unitvec(np.array([inj_l]), np.array([inj_b]))[0]
    b_inj = float(args.dipole_amp) * inj_axis
    delta_dip = n_seen @ b_inj

    lon_amps = [float(x) for x in str(args.lon_amps).split(",") if x.strip() != ""]
    phase = math.radians(float(args.lon_phase_deg))
    mix = (math.cos(phase) * cos_z + math.sin(phase) * sin_z).astype(float)

    results: list[dict[str, Any]] = []

    for amp in lon_amps:
        amp = float(amp)
        b_base: list[list[float]] = []
        b_lon: list[list[float]] = []

        for i in range(int(args.n_mocks)):
            loglam = log_mu_hat + delta_dip + amp * mix
            lam = np.exp(np.clip(loglam, -25.0, 25.0))
            y = rng.poisson(lam).astype(float)

            beta_b = fit_poisson_glm(X_fit_base, y, offset=None, max_iter=int(args.max_iter))
            beta_l = fit_poisson_glm(X_fit_lon, y, offset=None, max_iter=int(args.max_iter))

            b_base.append([float(x) for x in beta_b[1:4]])
            b_lon.append([float(x) for x in beta_l[1:4]])

            if (i + 1) % 20 == 0 or (i + 1) == int(args.n_mocks):
                print(f"amp={amp:g}  {i+1}/{int(args.n_mocks)} mocks")

        b_base_arr = np.asarray(b_base, dtype=float)
        b_lon_arr = np.asarray(b_lon, dtype=float)
        results.append(
            {
                "lon_amp": amp,
                "summary": {
                    "baseline_fit": summarize_b(b_base_arr),
                    "with_lon_templates": summarize_b(b_lon_arr),
                },
                "samples": {
                    "b_baseline": [[float(x) for x in row] for row in b_base_arr.tolist()],
                    "b_with_lon_templates": [[float(x) for x in row] for row in b_lon_arr.tolist()],
                },
            }
        )

    payload: dict[str, Any] = {
        "meta": {
            "catalog": str(args.catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": int(nside),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_min": None if args.w1_min is None else float(args.w1_min),
            "w1_max": float(args.w1_max),
            "n_mocks": int(args.n_mocks),
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "dipole_amp": float(args.dipole_amp),
            "dipole_axis_lb_deg": [inj_l, inj_b],
            "lon_phase_deg": float(args.lon_phase_deg),
        },
        "results": results,
    }

    out_json = outdir / "seasonal_injection.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out_json}")

    # Plot summary vs amplitude.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amps = [r["lon_amp"] for r in results]

    p50_base = [r["summary"]["baseline_fit"]["D_p50"] for r in results]
    p16_base = [r["summary"]["baseline_fit"]["D_p16"] for r in results]
    p84_base = [r["summary"]["baseline_fit"]["D_p84"] for r in results]
    Dmeanvec_base = [r["summary"]["baseline_fit"]["D_of_b_mean"] for r in results]

    p50_lon = [r["summary"]["with_lon_templates"]["D_p50"] for r in results]
    p16_lon = [r["summary"]["with_lon_templates"]["D_p16"] for r in results]
    p84_lon = [r["summary"]["with_lon_templates"]["D_p84"] for r in results]
    Dmeanvec_lon = [r["summary"]["with_lon_templates"]["D_of_b_mean"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.4))

    ax.plot(amps, p50_base, "o-", label="median D (fit: dipole+abs_elat)")
    ax.fill_between(amps, p16_base, p84_base, alpha=0.15)
    ax.plot(amps, Dmeanvec_base, "o--", lw=1.2, label="|mean b| (fit: dipole+abs_elat)")

    ax.plot(amps, p50_lon, "o-", label="median D (fit: dipole+abs_elat+sin/cos(lambda))")
    ax.fill_between(amps, p16_lon, p84_lon, alpha=0.15)
    ax.plot(amps, Dmeanvec_lon, "o--", lw=1.2, label="|mean b| (fit: dipole+abs_elat+sin/cos(lambda))")

    ax.axhline(float(args.dipole_amp), color="k", ls=":", lw=1.2, label="injected dipole amp")

    ax.set_xlabel("Injected lon-template amplitude (z-scored basis)")
    ax.set_ylabel("Recovered dipole amplitude")
    ax.set_title(f"Seasonal selection injection (W1_max={float(args.w1_max):.2f}, nside={nside})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    out_png = outdir / "seasonal_injection.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Wrote: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
