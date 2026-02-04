#!/usr/bin/env python3
"""Ecliptic-longitude wedge proxy with optional physical depth/coverage covariates.

This extends the existing ecliptic-longitude proxy idea by allowing an additional *physical*
scan/selection covariate (e.g. unWISE logNexp or invvar, or mean w1cov per pixel).

It still performs:
  1) per-ecliptic-longitude-bin dipole fits (to see wedge dependence)
  2) full-sky fits with and without sin/cos(lambda)

The goal is to check whether wedge-to-wedge dipole variability and/or full-sky amplitude
changes are absorbed when adding a physical scan/completeness predictor.
"""

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


def vec_to_lb(vec: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def axis_angle_deg(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Axis (sign-invariant) angle in degrees, in [0, 90]."""
    a = np.asarray(vec1, dtype=float).reshape(3)
    b = np.asarray(vec2, dtype=float).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if not np.isfinite(na) or not np.isfinite(nb) or na == 0.0 or nb == 0.0:
        return float("nan")
    dot = abs(float(np.dot(a, b)) / (na * nb))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(math.acos(dot)))


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
    mask: np.ndarray  # True = masked
    seen: np.ndarray  # True = unmasked


def build_secrest_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> SecrestMask:
    """Implements SkyMap.mask_zeros + fits2mask + galactic plane cut."""
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
        # Match Secrest behaviour (includes -1 neighbour indexing last pixel).
        mask[indices] = True

    # exclude discs
    if exclude_mask_fits:
        from astropy.table import Table
        from astropy.coordinates import SkyCoord
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
    lon_pix, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return SecrestMask(mask=mask, seen=~mask)


def fit_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    offset: np.ndarray | None,
    max_iter: int = 300,
    beta_init: np.ndarray | None = None,
    compute_cov: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Poisson GLM (log link) via L-BFGS; returns (beta, Fisher^{-1} approx) if invertible."""
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
    beta = np.asarray(res.x, dtype=float)

    if not bool(compute_cov):
        return beta, None
    try:
        eta = np.clip(off + X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        fisher = X.T @ (mu[:, None] * X)
        cov = np.linalg.inv(fisher)
    except Exception:  # noqa: BLE001
        cov = None
    return beta, cov


def _parse_edges(arg: str) -> list[float]:
    edges = [float(x) for x in arg.split(",") if x.strip() != ""]
    if len(edges) < 2:
        raise ValueError("--lambda-edges must have at least two comma-separated values")
    if sorted(edges) != edges:
        raise ValueError("--lambda-edges must be sorted increasing")
    if abs(edges[0]) > 1e-9 or abs(edges[-1] - 360.0) > 1e-9:
        raise ValueError("--lambda-edges must start at 0 and end at 360")
    return edges


def _load_map_or_none(path: str | None, *, nside: int, seen: np.ndarray, name: str) -> np.ndarray | None:
    if path is None:
        return None
    import healpy as hp

    m = hp.read_map(str(path), verbose=False)
    if int(hp.get_nside(m)) != int(nside):
        m = hp.ud_grade(m, nside_out=int(nside), order_in="RING", order_out="RING", power=0)
    m = np.asarray(m, dtype=float)

    unseen = ~np.isfinite(m) | (m == hp.UNSEEN)
    ok = np.asarray(seen, dtype=bool) & (~unseen)
    if not np.any(ok):
        raise SystemExit(f"Map {name} has no finite values on seen pixels: {path}")
    fill = float(np.median(m[ok]))
    m[unseen] = fill
    return m


def fit_region(
    *,
    counts: np.ndarray,
    seen_region: np.ndarray,
    pix_unit: np.ndarray,
    abs_elat_deg: np.ndarray,
    elon_rad: np.ndarray,
    add_lon_templates: bool,
    depth_map: np.ndarray | None,
    max_iter: int,
) -> dict[str, Any]:
    seen_region = np.asarray(seen_region, dtype=bool)
    y = np.asarray(counts[seen_region], dtype=float)
    n_seen = pix_unit[seen_region]

    templates: list[np.ndarray] = [zscore(abs_elat_deg, seen_region)[seen_region]]
    template_names = ["abs_elat_z"]

    if depth_map is not None:
        templates.append(zscore(depth_map, seen_region)[seen_region])
        template_names.append("depth_map_z")

    if bool(add_lon_templates):
        templates.append(zscore(np.sin(elon_rad), seen_region)[seen_region])
        template_names.append("sin_lambda_z")
        templates.append(zscore(np.cos(elon_rad), seen_region)[seen_region])
        template_names.append("cos_lambda_z")

    cols = [np.ones_like(y), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2]]
    cols.extend(templates)
    X = np.column_stack(cols)

    beta_hat, cov_beta = fit_poisson_glm(X, y, offset=None, max_iter=int(max_iter), compute_cov=True)
    bvec = np.asarray(beta_hat[1:4], dtype=float)
    D_hat = float(np.linalg.norm(bvec))
    l_hat, b_hat = vec_to_lb(bvec)

    out: dict[str, Any] = {
        "N_seen_pix": int(np.sum(seen_region)),
        "N_seen_src": int(np.sum(counts[seen_region])),
        "template_names": template_names,
        "beta_hat": [float(x) for x in beta_hat],
        "b_vec": [float(x) for x in bvec],
        "D_hat": D_hat,
        "l_hat_deg": float(l_hat),
        "b_hat_deg": float(b_hat),
    }
    if cov_beta is not None:
        cov_b = np.asarray(cov_beta[1:4, 1:4], dtype=float)
        out["cov_b"] = [[float(x) for x in row] for row in cov_b.tolist()]
    else:
        out["cov_b"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1-max", type=float, default=16.6)
    ap.add_argument("--w1-min", type=float, default=None)

    ap.add_argument(
        "--lambda-edges",
        default="0,90,180,270,360",
        help="Comma-separated ecliptic-longitude bin edges.",
    )
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--make-plot", action="store_true")

    ap.add_argument(
        "--depth",
        choices=["none", "lognexp", "invvar", "w1cov_mean"],
        default="none",
        help="Optional physical scan/coverage covariate added to each fit.",
    )
    ap.add_argument(
        "--lognexp-map",
        default="data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits",
        help="Required when --depth=lognexp.",
    )
    ap.add_argument(
        "--invvar-map",
        default="data/cache/unwise_invvar/neo7/invvar_healpix_nside64.fits",
        help="Required when --depth=invvar.",
    )

    args = ap.parse_args()

    import healpy as hp
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    outdir = Path(args.outdir or f"outputs/ecllon_proxy_with_depth_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    edges = _parse_edges(str(args.lambda_edges))

    # Load catalog columns (memmap to avoid giant copies).
    with fits.open(args.catalog, memmap=True) as hdul:
        data = hdul[1].data
        w1 = np.asarray(data["w1"], dtype=float)
        w1cov = np.asarray(data["w1cov"], dtype=float)
        l = np.asarray(data["l"], dtype=float)
        b = np.asarray(data["b"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    base &= w1cov >= float(args.w1cov_min)

    nside = int(args.nside)
    npix = int(hp.nside2npix(nside))

    # Footprint mask is defined on the parent sample (W1cov cut only).
    ipix_mask_base = hp.ang2pix(
        nside,
        np.deg2rad(90.0 - b[base]),
        np.deg2rad(l[base]),
        nest=False,
    ).astype(np.int64)

    # Analysis selection.
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

    secrest_mask = build_secrest_mask(
        nside=nside,
        ipix_base=ipix_mask_base,
        exclude_mask_fits=str(args.exclude_mask_fits) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )
    seen_all = secrest_mask.seen

    # Pixel centers in Galactic coordinates.
    lon_pix, lat_pix = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    pix_unit = lb_to_unitvec(lon_pix, lat_pix)

    # Pixel-center ecliptic coordinates.
    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    ecl = sc_pix.barycentricmeanecliptic
    elat = np.asarray(ecl.lat.deg, dtype=float)
    elon = np.asarray(ecl.lon.deg, dtype=float) % 360.0
    abs_elat = np.abs(elat)
    elon_rad = np.deg2rad(elon)

    # Optional depth map.
    depth_map = None
    if args.depth == "lognexp":
        depth_map = _load_map_or_none(str(args.lognexp_map), nside=nside, seen=seen_all, name="lognexp")
    elif args.depth == "invvar":
        depth_map = _load_map_or_none(str(args.invvar_map), nside=nside, seen=seen_all, name="invvar")
        assert depth_map is not None
        depth_map = np.log(np.clip(depth_map, 1e-12, np.inf))
    elif args.depth == "w1cov_mean":
        cnt_base = np.bincount(ipix_mask_base, minlength=npix).astype(float)
        sum_w1cov = np.bincount(ipix_mask_base, weights=w1cov[base].astype(float), minlength=npix).astype(float)
        w1cov_mean = np.divide(sum_w1cov, cnt_base, out=np.zeros_like(sum_w1cov), where=cnt_base != 0.0)
        depth_map = np.log(np.clip(w1cov_mean, 1.0, np.inf))
    else:
        depth_map = None

    # Reference axes.
    cmb_l, cmb_b = 264.021, 48.253
    cmb_vec = lb_to_unitvec(np.array([cmb_l]), np.array([cmb_b]))[0]

    full_base = fit_region(
        counts=counts,
        seen_region=seen_all,
        pix_unit=pix_unit,
        abs_elat_deg=abs_elat,
        elon_rad=elon_rad,
        add_lon_templates=False,
        depth_map=depth_map,
        max_iter=int(args.max_iter),
    )
    full_lon = fit_region(
        counts=counts,
        seen_region=seen_all,
        pix_unit=pix_unit,
        abs_elat_deg=abs_elat,
        elon_rad=elon_rad,
        add_lon_templates=True,
        depth_map=depth_map,
        max_iter=int(args.max_iter),
    )

    full_base["angle_to_cmb_deg"] = float(axis_angle_deg(np.asarray(full_base["b_vec"]), cmb_vec))
    full_lon["angle_to_cmb_deg"] = float(axis_angle_deg(np.asarray(full_lon["b_vec"]), cmb_vec))

    # Ecliptic-longitude bins.
    bins: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (elon >= float(lo)) & (elon < float(hi))
        seen_bin = seen_all & in_bin
        if int(np.sum(seen_bin)) < 500:
            bins.append(
                {
                    "lambda_lo_deg": float(lo),
                    "lambda_hi_deg": float(hi),
                    "skipped": True,
                    "reason": "too_few_seen_pixels",
                    "N_seen_pix": int(np.sum(seen_bin)),
                    "N_seen_src": int(np.sum(counts[seen_bin])),
                }
            )
            continue
        r = fit_region(
            counts=counts,
            seen_region=seen_bin,
            pix_unit=pix_unit,
            abs_elat_deg=abs_elat,
            elon_rad=elon_rad,
            add_lon_templates=False,
            depth_map=depth_map,
            max_iter=int(args.max_iter),
        )
        r["lambda_lo_deg"] = float(lo)
        r["lambda_hi_deg"] = float(hi)
        r["lambda_mid_deg"] = 0.5 * (float(lo) + float(hi))
        r["angle_to_cmb_deg"] = float(axis_angle_deg(np.asarray(r["b_vec"]), cmb_vec))
        r["angle_to_full_base_deg"] = float(axis_angle_deg(np.asarray(r["b_vec"]), np.asarray(full_base["b_vec"])))
        bins.append(r)

    payload: dict[str, Any] = {
        "meta": {
            "catalog": str(args.catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": int(nside),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_min": None if args.w1_min is None else float(args.w1_min),
            "w1_max": float(args.w1_max),
            "lambda_edges_deg": [float(x) for x in edges],
            "cmb_axis_lb_deg": [cmb_l, cmb_b],
            "depth": str(args.depth),
            "lognexp_map": str(args.lognexp_map),
            "invvar_map": str(args.invvar_map),
        },
        "fullsky": {"baseline": full_base, "with_lon_templates": full_lon},
        "bins": bins,
    }

    out_json = outdir / "ecllon_proxy_with_depth.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out_json}")

    if args.make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mid = []
        D = []
        for r in bins:
            if r.get("skipped"):
                continue
            mid.append(float(r["lambda_mid_deg"]))
            D.append(float(r["D_hat"]))

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8))
        ax.plot(mid, D, "o-", label="per-bin dipole D")
        ax.axhline(float(full_base["D_hat"]), color="k", ls=":", lw=1.2, label="full-sky baseline D")
        ax.set_xlabel("Ecliptic longitude bin center (deg)")
        ax.set_ylabel("Recovered dipole amplitude")
        ax.set_title(f"Ecliptic-longitude wedge proxy (depth={args.depth}, W1_max={float(args.w1_max):.2f})")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        out_png = outdir / "ecllon_proxy_with_depth.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"Wrote: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
