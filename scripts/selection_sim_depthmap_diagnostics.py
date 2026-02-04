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

    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        indices = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            indices[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[indices] = True

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Diagnostics for the depth-map selection simulation at a single W1 cut: "
            "measure induced sinλ/cosλ coefficients and λ-wedge dipoles."
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
    ap.add_argument("--w1-cut", type=float, default=16.6)
    ap.add_argument("--alpha-delta", type=float, default=0.05)

    ap.add_argument(
        "--depth-map-fits",
        default="REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits",
    )
    ap.add_argument(
        "--depth-map-kind",
        choices=["delta_m_mag", "lognexp", "loginvvar"],
        default="delta_m_mag",
    )
    ap.add_argument("--sel-scale", type=float, default=2.7)

    ap.add_argument("--lambda-edges", default="0,90,180,270,360")

    ap.add_argument("--n-mocks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-iter", type=int, default=250)

    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits

    rng = np.random.default_rng(int(args.seed))

    outdir = Path(args.outdir or f"outputs/selection_sim_depthmap_diag_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    nside = int(args.nside)
    npix = int(hp.nside2npix(nside))

    edges = [float(x) for x in str(args.lambda_edges).split(",") if x.strip() != ""]
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 360.0:
        raise SystemExit("--lambda-edges must start at 0 and end at 360")

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

    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    ecl = sc_pix.barycentricmeanecliptic
    elon = (np.asarray(ecl.lon.deg, dtype=float) % 360.0)
    elat = np.asarray(ecl.lat.deg, dtype=float)
    abs_elat = np.abs(elat)

    abs_elat_z = zscore(abs_elat, seen)[seen]
    sin_z = zscore(np.sin(np.deg2rad(elon)), seen)[seen]
    cos_z = zscore(np.cos(np.deg2rad(elon)), seen)[seen]

    depth_map = hp.read_map(str(args.depth_map_fits), verbose=False)
    if int(hp.get_nside(depth_map)) != int(nside):
        depth_map = hp.ud_grade(depth_map, nside_out=int(nside), order_in="RING", order_out="RING", power=0)
    depth_map = np.asarray(depth_map, dtype=float)
    unseen = ~np.isfinite(depth_map) | (depth_map == hp.UNSEEN)
    ok = seen & (~unseen)
    fill = float(np.median(depth_map[ok]))
    depth_map[unseen] = fill

    if str(args.depth_map_kind) == "lognexp":
        depth_map = np.log(np.clip(depth_map, 1.0, np.inf))
    elif str(args.depth_map_kind) == "loginvvar":
        depth_map = np.log(np.clip(depth_map, 1e-12, np.inf))
    elif str(args.depth_map_kind) == "delta_m_mag":
        pass

    depth_centered = depth_map - float(np.median(depth_map[seen]))

    # Real-data counts at the cut (for baseline mu_hat fit).
    sel = base & (w1 <= float(args.w1_cut))
    ipix_sel = hp.ang2pix(
        nside,
        np.deg2rad(90.0 - b[sel]),
        np.deg2rad(l[sel]),
        nest=False,
    ).astype(np.int64)
    counts_real = np.bincount(ipix_sel, minlength=npix).astype(float)
    y_data = counts_real[seen]

    # Fit base intensity: intercept + abs_elat only (NO dipole).
    X_mu = np.column_stack([np.ones_like(y_data), abs_elat_z])
    beta_mu = fit_poisson_glm(X_mu, y_data, offset=None, max_iter=int(args.max_iter))
    log_mu_hat = np.clip(X_mu @ beta_mu, -25.0, 25.0)

    # alpha_edge for this cut
    ipix_all_base = hp.ang2pix(
        nside,
        np.deg2rad(90.0 - b[base]),
        np.deg2rad(l[base]),
        nest=False,
    ).astype(np.int64)
    w1_seen = w1[base][seen[ipix_all_base]]
    a_edge = alpha_edge_from_counts(w1_seen, w1_hi=float(args.w1_cut), delta=float(args.alpha_delta))

    sel_offset = float(args.sel_scale) * float(a_edge) * depth_centered[seen]

    # Fit designs.
    X_base = np.column_stack([np.ones_like(y_data), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2], abs_elat_z])
    X_lon = np.column_stack(
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

    # Bins: we store b_vec per bin.
    bins = [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    bin_seen = []
    for lo, hi in bins:
        in_bin = (elon >= lo) & (elon < hi)
        bin_seen.append(seen & in_bin)

    samples: list[dict[str, Any]] = []

    for i in range(int(args.n_mocks)):
        loglam = log_mu_hat + sel_offset
        lam = np.exp(np.clip(loglam, -25.0, 25.0))
        y_seen = rng.poisson(lam).astype(float)
        y_full = np.zeros(npix, dtype=float)
        y_full[seen] = y_seen

        beta_base = fit_poisson_glm(X_base, y_seen, offset=None, max_iter=int(args.max_iter))
        beta_lon = fit_poisson_glm(X_lon, y_seen, offset=None, max_iter=int(args.max_iter))
        beta_off = fit_poisson_glm(X_base, y_seen, offset=sel_offset, max_iter=int(args.max_iter))

        entry: dict[str, Any] = {
            "b_base": [float(x) for x in beta_base[1:4]],
            "b_lon": [float(x) for x in beta_lon[1:4]],
            "lon_templates": {
                "sin_lambda_z": float(beta_lon[5]),
                "cos_lambda_z": float(beta_lon[6]),
            },
            "b_with_true_offset": [float(x) for x in beta_off[1:4]],
        }

        # Wedge fits (baseline model only).
        wedge_b = []
        for seen_bin in bin_seen:
            if int(np.sum(seen_bin)) < 500:
                wedge_b.append(None)
                continue
            y_bin = y_full[seen_bin]
            n_bin = pix_unit[seen_bin]
            abs_elat_z_bin = zscore(abs_elat, seen_bin)[seen_bin]
            Xb = np.column_stack([np.ones_like(y_bin), n_bin[:, 0], n_bin[:, 1], n_bin[:, 2], abs_elat_z_bin])
            beta = fit_poisson_glm(Xb, y_bin, offset=None, max_iter=int(args.max_iter))
            wedge_b.append([float(x) for x in beta[1:4]])
        entry["wedge_b"] = wedge_b

        samples.append(entry)

        if (i + 1) % 50 == 0 or (i + 1) == int(args.n_mocks):
            print(f"{i+1}/{int(args.n_mocks)} mocks")

    # Aggregate lon-template coefficients.
    sin_coeff = np.array([s["lon_templates"]["sin_lambda_z"] for s in samples], dtype=float)
    cos_coeff = np.array([s["lon_templates"]["cos_lambda_z"] for s in samples], dtype=float)
    A = np.hypot(sin_coeff, cos_coeff)
    phi = np.degrees(np.arctan2(sin_coeff, cos_coeff))

    payload: dict[str, Any] = {
        "meta": {
            "w1_cut": float(args.w1_cut),
            "alpha_edge": float(a_edge),
            "alpha_delta": float(args.alpha_delta),
            "sel_scale": float(args.sel_scale),
            "depth_map_fits": str(args.depth_map_fits),
            "depth_map_kind": str(args.depth_map_kind),
            "n_mocks": int(args.n_mocks),
            "seed": int(args.seed),
            "lambda_edges_deg": edges,
        },
        "lon_template_summary": {
            "sin_mean": float(np.mean(sin_coeff)),
            "cos_mean": float(np.mean(cos_coeff)),
            "A_mean": float(np.mean(A)),
            "A_p16": float(np.percentile(A, 16)),
            "A_p50": float(np.percentile(A, 50)),
            "A_p84": float(np.percentile(A, 84)),
            "phi_mean_deg": float(np.mean(phi)),
        },
        "samples": samples,
    }

    out_json = outdir / "selection_sim_depthmap_diagnostics.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
