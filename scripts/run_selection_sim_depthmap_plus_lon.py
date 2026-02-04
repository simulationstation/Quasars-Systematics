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
        return np.zeros_like(x)
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

    # mask_zeros on the W1cov>=cut parent sample.
    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        indices = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            indices[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[indices] = True  # match Secrest behavior (-1 neighbors)

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


def alpha_edge_from_counts(w1_vals: np.ndarray, *, w1_hi: float, delta: float) -> float:
    w1_vals = np.asarray(w1_vals, dtype=float)
    hi = float(w1_hi)
    lo = float(w1_hi - float(delta))
    n_hi = float(np.sum(w1_vals <= hi))
    n_lo = float(np.sum(w1_vals <= lo))
    n_hi = max(1.0, n_hi)
    n_lo = max(1.0, n_lo)
    return float((math.log(n_hi) - math.log(n_lo)) / float(delta))


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


def _load_lon_coeffs_from_scan(scan_json: Path) -> dict[float, tuple[float, float]]:
    obj = json.loads(scan_json.read_text())
    rows = obj.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{scan_json}: expected top-level 'rows' list")

    out: dict[float, tuple[float, float]] = {}
    for r in rows:
        w1_cut = float(r["w1_cut"])
        names = list(r["template_names"])
        beta = list(r["beta_hat"])
        tmpl_beta = beta[4:]
        if len(tmpl_beta) != len(names):
            raise ValueError(f"{scan_json}: template_names/beta_hat mismatch at w1_cut={w1_cut}")
        sin_val = None
        cos_val = None
        for name, val in zip(names, tmpl_beta, strict=True):
            if name in {"sin_elon_z", "sin_lambda_z"}:
                sin_val = float(val)
            if name in {"cos_elon_z", "cos_lambda_z"}:
                cos_val = float(val)
        if sin_val is None or cos_val is None:
            continue
        out[w1_cut] = (sin_val, cos_val)
    if not out:
        raise ValueError(f"{scan_json}: did not find sin/cos lon template coefficients")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Selection simulation using a real depth/coverage map PLUS an ecliptic-longitude pattern. "
            "Generates Poisson mock maps where selection is modulated by (i) a depth-map-derived offset scaled by alpha_edge, "
            "and (ii) optional sin/cos(ecliptic longitude) templates, then measures dipole bias when these selection terms are omitted."
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

    ap.add_argument("--w1-grid", default="15.5,16.6,0.05")
    ap.add_argument("--alpha-delta", type=float, default=0.05)

    ap.add_argument(
        "--depth-map-fits",
        default="REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits",
        help="HEALPix map used as the depth-linked selection driver (Galactic coords).",
    )
    ap.add_argument(
        "--depth-map-kind",
        choices=["delta_m_mag", "lognexp", "loginvvar"],
        default="delta_m_mag",
        help="Interpretation/transform applied to --depth-map-fits before use.",
    )
    ap.add_argument("--depth-sel-scale", type=float, default=0.0, help="Scale for the depth-map selection offset.")

    ap.add_argument(
        "--lon-coeffs-scan-json",
        default=None,
        help=(
            "Optional scan JSON (from reproduce_rvmp_fig5_catwise_poisson_glm.py) used to inject per-cut sin/cos lon coefficients. "
            "If provided, overrides --lon-sin/--lon-cos."
        ),
    )
    ap.add_argument("--lon-sin", type=float, default=0.0, help="Injected sin(lambda)_z coefficient (if scan-json not used).")
    ap.add_argument("--lon-cos", type=float, default=0.0, help="Injected cos(lambda)_z coefficient (if scan-json not used).")
    ap.add_argument("--lon-scale", type=float, default=1.0, help="Overall multiplier on injected lon coefficients.")

    ap.add_argument("--n-mocks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-iter", type=int, default=250)

    ap.add_argument("--dipole-amp", type=float, default=0.0, help="Optional injected dipole amplitude in log-intensity.")
    ap.add_argument("--dipole-axis-lb", default="264.021,48.253", help="Injected dipole axis (l,b) in degrees.")

    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits

    rng = np.random.default_rng(int(args.seed))

    outdir = Path(args.outdir or f"outputs/selection_sim_depthmap_plus_lon_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    w1_lo, w1_hi, w1_step = (float(x) for x in str(args.w1_grid).split(","))
    w1_cuts = np.round(np.arange(w1_lo, w1_hi + 0.5 * w1_step, w1_step), 10)

    nside = int(args.nside)
    npix = int(hp.nside2npix(nside))

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

    # Ecliptic templates.
    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    ecl = sc_pix.barycentricmeanecliptic
    elon = (np.asarray(ecl.lon.deg, dtype=float) % 360.0)
    elat = np.asarray(ecl.lat.deg, dtype=float)
    abs_elat = np.abs(elat)

    abs_elat_z = zscore(abs_elat, seen)[seen]
    sin_z = zscore(np.sin(np.deg2rad(elon)), seen)[seen]
    cos_z = zscore(np.cos(np.deg2rad(elon)), seen)[seen]

    # Load depth map and transform.
    depth_map = hp.read_map(str(args.depth_map_fits), verbose=False)
    if int(hp.get_nside(depth_map)) != int(nside):
        depth_map = hp.ud_grade(depth_map, nside_out=int(nside), order_in="RING", order_out="RING", power=0)
    depth_map = np.asarray(depth_map, dtype=float)
    unseen = ~np.isfinite(depth_map) | (depth_map == hp.UNSEEN)
    ok = seen & (~unseen)
    if not np.any(ok):
        raise SystemExit("Depth map has no finite values on seen pixels")
    fill = float(np.median(depth_map[ok]))
    depth_map[unseen] = fill

    if str(args.depth_map_kind) == "lognexp":
        depth_map = np.log(np.clip(depth_map, 1.0, np.inf))
    elif str(args.depth_map_kind) == "loginvvar":
        depth_map = np.log(np.clip(depth_map, 1e-12, np.inf))
    elif str(args.depth_map_kind) == "delta_m_mag":
        pass
    else:
        raise SystemExit("unsupported depth_map_kind")

    depth_centered = depth_map - float(np.median(depth_map[seen]))

    # Optional injected dipole.
    inj_l, inj_b = (float(x) for x in str(args.dipole_axis_lb).split(","))
    inj_axis = lb_to_unitvec(np.array([inj_l]), np.array([inj_b]))[0]
    b_inj = float(args.dipole_amp) * inj_axis
    delta_dip = n_seen @ b_inj

    # Precompute seen-footprint W1 values for alpha_edge.
    ipix_all_base = hp.ang2pix(
        nside,
        np.deg2rad(90.0 - b[base]),
        np.deg2rad(l[base]),
        nest=False,
    ).astype(np.int64)
    w1_seen = w1[base][seen[ipix_all_base]]

    # Optional: load injected lon coefficients per cut.
    lon_by_cut: dict[float, tuple[float, float]] | None = None
    if args.lon_coeffs_scan_json is not None:
        lon_by_cut = _load_lon_coeffs_from_scan(Path(str(args.lon_coeffs_scan_json)))

    # Recovery design matrices.
    X_base = np.column_stack([np.ones(np.sum(seen), dtype=float), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2], abs_elat_z])
    X_lon = np.column_stack(
        [
            np.ones(np.sum(seen), dtype=float),
            n_seen[:, 0],
            n_seen[:, 1],
            n_seen[:, 2],
            abs_elat_z,
            sin_z,
            cos_z,
        ]
    )

    results: list[dict[str, Any]] = []

    for w1_cut in w1_cuts:
        # Real-data counts (for baseline mu_hat fit).
        sel = base & (w1 <= float(w1_cut))
        ipix_sel = hp.ang2pix(
            nside,
            np.deg2rad(90.0 - b[sel]),
            np.deg2rad(l[sel]),
            nest=False,
        ).astype(np.int64)
        counts_real = np.bincount(ipix_sel, minlength=npix).astype(float)
        y_data = counts_real[seen]

        # Fit base intensity: intercept + abs_elat only (NO dipole), per cut.
        X_mu = np.column_stack([np.ones_like(y_data), abs_elat_z])
        beta_mu = fit_poisson_glm(X_mu, y_data, offset=None, max_iter=int(args.max_iter))
        log_mu_hat = np.clip(X_mu @ beta_mu, -25.0, 25.0)

        a_edge = alpha_edge_from_counts(w1_seen, w1_hi=float(w1_cut), delta=float(args.alpha_delta))

        depth_offset = float(args.depth_sel_scale) * float(a_edge) * depth_centered[seen]

        if lon_by_cut is not None:
            # Match by exact cut values (the scan JSON is produced on an identical grid).
            sin_coeff, cos_coeff = lon_by_cut[float(w1_cut)]
        else:
            sin_coeff, cos_coeff = float(args.lon_sin), float(args.lon_cos)
        sin_coeff *= float(args.lon_scale)
        cos_coeff *= float(args.lon_scale)
        lon_offset = sin_coeff * sin_z + cos_coeff * cos_z

        sel_offset = depth_offset + lon_offset

        b_base: list[list[float]] = []
        b_lon: list[list[float]] = []
        b_corr: list[list[float]] = []
        lon_fit_sin: list[float] = []
        lon_fit_cos: list[float] = []

        for i in range(int(args.n_mocks)):
            loglam = log_mu_hat + delta_dip + sel_offset
            lam = np.exp(np.clip(loglam, -25.0, 25.0))
            y = rng.poisson(lam).astype(float)

            beta_b = fit_poisson_glm(X_base, y, offset=None, max_iter=int(args.max_iter))
            beta_l = fit_poisson_glm(X_lon, y, offset=None, max_iter=int(args.max_iter))
            beta_c = fit_poisson_glm(X_base, y, offset=sel_offset, max_iter=int(args.max_iter))

            b_base.append([float(x) for x in beta_b[1:4]])
            b_lon.append([float(x) for x in beta_l[1:4]])
            b_corr.append([float(x) for x in beta_c[1:4]])
            lon_fit_sin.append(float(beta_l[5]))
            lon_fit_cos.append(float(beta_l[6]))

            if (i + 1) % 50 == 0 or (i + 1) == int(args.n_mocks):
                print(f"w1_cut={w1_cut:.2f}  {i+1}/{int(args.n_mocks)} mocks")

        b_base_arr = np.asarray(b_base, dtype=float)
        b_lon_arr = np.asarray(b_lon, dtype=float)
        b_corr_arr = np.asarray(b_corr, dtype=float)

        results.append(
            {
                "w1_cut": float(w1_cut),
                "alpha_edge": float(a_edge),
                "depth_sel_scale": float(args.depth_sel_scale),
                "lon_injected": {
                    "sin_lambda_z": float(sin_coeff),
                    "cos_lambda_z": float(cos_coeff),
                },
                "summary": {
                    "baseline_fit": summarize_b(b_base_arr),
                    "with_lon_templates": summarize_b(b_lon_arr),
                    "with_true_offset": summarize_b(b_corr_arr),
                },
                "lon_fit_summary": {
                    "sin_mean": float(np.mean(lon_fit_sin)),
                    "cos_mean": float(np.mean(lon_fit_cos)),
                    "sin_p16": float(np.percentile(lon_fit_sin, 16)),
                    "sin_p50": float(np.percentile(lon_fit_sin, 50)),
                    "sin_p84": float(np.percentile(lon_fit_sin, 84)),
                    "cos_p16": float(np.percentile(lon_fit_cos, 16)),
                    "cos_p50": float(np.percentile(lon_fit_cos, 50)),
                    "cos_p84": float(np.percentile(lon_fit_cos, 84)),
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
            "w1_grid": str(args.w1_grid),
            "alpha_delta": float(args.alpha_delta),
            "depth_map_fits": str(args.depth_map_fits),
            "depth_map_kind": str(args.depth_map_kind),
            "depth_map_fill": float(fill),
            "depth_sel_scale": float(args.depth_sel_scale),
            "lon_coeffs_scan_json": None if args.lon_coeffs_scan_json is None else str(args.lon_coeffs_scan_json),
            "lon_sin": float(args.lon_sin),
            "lon_cos": float(args.lon_cos),
            "lon_scale": float(args.lon_scale),
            "n_mocks": int(args.n_mocks),
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "dipole_amp": float(args.dipole_amp),
            "dipole_axis_lb_deg": [inj_l, inj_b],
        },
        "results": results,
    }

    out_json = outdir / "selection_sim_depthmap_plus_lon.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out_json}")

    # Plot D_of_b_mean vs W1 cut.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cuts = [r["w1_cut"] for r in results]
    D_base = [r["summary"]["baseline_fit"]["D_of_b_mean"] for r in results]
    D_lon = [r["summary"]["with_lon_templates"]["D_of_b_mean"] for r in results]
    D_corr = [r["summary"]["with_true_offset"]["D_of_b_mean"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.4))
    ax.plot(cuts, D_base, "o-", label="fit dipole+abs_elat (omits selection)")
    ax.plot(cuts, D_lon, "o-", label="fit dipole+abs_elat+sin/cos(lon) (omits selection)")
    ax.plot(cuts, D_corr, "o-", label="fit dipole+abs_elat with true selection offset")
    ax.axhline(float(args.dipole_amp), color="k", ls=":", lw=1.2, label="injected dipole amp")
    ax.set_xlabel("W1_max")
    ax.set_ylabel("Recovered dipole bias  |mean(b_vec)|")
    ax.set_title(
        f"Selection simulation (depth+lon)  depth_scale={float(args.depth_sel_scale):g}  lon_scale={float(args.lon_scale):g}"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    out_png = outdir / "selection_sim_depthmap_plus_lon.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Wrote: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
