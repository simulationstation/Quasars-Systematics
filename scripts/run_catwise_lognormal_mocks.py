#!/usr/bin/env python3
"""
Clustered (lognormal) mocks for CatWISE dipole covariance.

Why this exists
---------------
Poisson-only resampling underestimates uncertainty for extragalactic tracers because it ignores
sample variance / LSS clustering. This script generates quick clustered mocks to estimate an
LSS+shot-noise covariance for the dipole fit.

Approach
--------
1) Build a HEALPix counts map (Secrest footprint mask: mask_zeros + exclusion discs + |b| cut).
2) Fit a Poisson GLM dipole model with optional templates/offsets to get mu_hat(p).
3) Estimate an approximate clustering power spectrum C_ell from fractional residuals y/mu_hat - 1,
   subtracting a white shot-noise floor and applying an f_sky correction (rough).
4) For each mock:
     - draw a Gaussian random field g with that C_ell (healpy.synfast),
     - convert to a lognormal multiplicative field exp(g - var/2),
     - generate Poisson counts with mean mu_hat * lognormal_field (and optional injected dipole),
     - refit the GLM and record the recovered dipole vector.

Notes
-----
This is intentionally lightweight and approximate (mask coupling is treated with a crude f_sky
correction). For publication-grade covariance, replace the C_ell estimation with a proper pseudo-C_ell
deconvolution (e.g. NaMaster) and/or use theory-driven C_ell.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
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
    """Axis angle in [0,90] degrees (sign-invariant)."""
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
    m = float(np.mean(x[valid])) if np.any(valid) else 0.0
    s = float(np.std(x[valid])) if np.any(valid) else 1.0
    if s == 0.0:
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

    # mask_zeros(tbl) on the W1cov>=80 full sample
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
    """
    Poisson GLM (log link) via L-BFGS.
    Returns (beta, cov_beta_approx) where cov is Fisher^{-1} if invertible.
    """
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


def poisson_glm_diagnostics(y: np.ndarray, mu: np.ndarray, *, n_params: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    dof = int(y.size) - int(n_params)
    dof_f = float(dof) if dof > 0 else float("nan")

    mu_pos = np.clip(mu, 1e-30, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0.0, y * np.log(y / mu_pos), 0.0)
        dev = 2.0 * np.sum(term - (y - mu_pos))
        pearson = np.sum((y - mu_pos) ** 2 / mu_pos)

    dev = float(dev)
    pearson = float(pearson)
    dev_over_dof = float(dev / dof_f) if np.isfinite(dof_f) else float("nan")
    pearson_over_dof = float(pearson / dof_f) if np.isfinite(dof_f) else float("nan")
    phi = pearson_over_dof
    phi_eff = float(max(1.0, phi)) if np.isfinite(phi) else float("nan")

    return {
        "dof": float(dof),
        "deviance": dev,
        "pearson_chi2": pearson,
        "deviance_over_dof": dev_over_dof,
        "pearson_over_dof": pearson_over_dof,
        "dispersion_phi": float(phi) if np.isfinite(phi) else float("nan"),
        "dispersion_phi_eff": phi_eff,
    }


def fit_mono_dipole_on_seen(m: np.ndarray, n_seen: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares fit m = a0 + a·n on seen pixels. Returns (a0, a_vec)."""
    X = np.column_stack([np.ones_like(m), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2]])
    beta, *_ = np.linalg.lstsq(X, m, rcond=None)
    return float(beta[0]), np.asarray(beta[1:4], dtype=float)


def estimate_cl_from_residuals(
    *,
    resid_seen: np.ndarray,
    mu_seen: np.ndarray,
    seen: np.ndarray,
    pix_unit: np.ndarray,
    lmax: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Estimate a crude clustering power spectrum from fractional residuals r=y/mu-1.

    Returns (cl_signal, meta). Uses:
      - monopole+dipole removal on seen pixels,
      - pseudo-C_ell from healpy.anafast on a full-sky map with masked pixels=0,
      - f_sky correction: C_ell ≈ C_ell_pseudo / f_sky,
      - white shot-noise subtraction estimated from mean(1/mu).
    """
    import healpy as hp

    npix = int(seen.size)
    n_seen = pix_unit[seen]

    r = np.asarray(resid_seen, dtype=float)
    a0, avec = fit_mono_dipole_on_seen(r, n_seen)
    r_clean = r - (a0 + n_seen @ avec)

    full = np.zeros(npix, dtype=float)
    full[seen] = r_clean

    cl_pseudo = hp.anafast(full, lmax=int(lmax))
    f_sky = float(np.mean(seen))
    if f_sky <= 0:
        raise ValueError("f_sky <= 0")
    cl_true = cl_pseudo / f_sky

    # Shot noise floor for the fractional residual map: Var(noise)~1/mu per pixel.
    var_noise = float(np.mean(1.0 / np.clip(mu_seen, 1.0, np.inf)))
    c_noise = float(4.0 * math.pi * var_noise / npix)
    cl_signal = cl_true - c_noise
    cl_signal = np.where(np.isfinite(cl_signal), cl_signal, 0.0)
    cl_signal = np.maximum(cl_signal, 0.0)
    cl_signal[:2] = 0.0

    meta = {
        "f_sky": f_sky,
        "noise_var_per_pix_est": var_noise,
        "noise_cl_white_est": c_noise,
        "mono_dipole_removed": {"a0": float(a0), "avec": [float(x) for x in avec]},
    }
    return np.asarray(cl_signal, dtype=float), meta


def gaussian_to_lognormal_factor(g: np.ndarray, *, seen: np.ndarray) -> tuple[np.ndarray, float]:
    g = np.asarray(g, dtype=float)
    g0 = g.copy()
    g0[seen] -= float(np.mean(g0[seen]))
    var = float(np.var(g0[seen]))
    fac = np.exp(g0 - 0.5 * var)
    return fac, var


def _auto_nproc(nproc: int) -> int:
    if int(nproc) > 0:
        return int(nproc)
    return int(os.cpu_count() or 1)


def _seed_for_mock(base_seed: int, i: int) -> int:
    # Keep within 32-bit range for legacy RNGs used by healpy.synfast.
    return int((int(base_seed) + 1_000_003 * int(i)) % 2_147_483_647)


_LN_STATE: dict[str, Any] | None = None


def _init_lognormal_worker(state: dict[str, Any]) -> None:
    global _LN_STATE
    _LN_STATE = state
    # Keep worker logs clean (spawn/forkserver do not inherit parent's warning filters).
    warnings.filterwarnings("ignore", message=r".*verbose.*deprecated.*", category=Warning)


def _run_one_mock(i: int) -> tuple[int, np.ndarray, float, float]:
    """
    Worker entrypoint for multiprocessing.
    Uses global `_LN_STATE` (populated in the parent before pool creation).
    """
    global _LN_STATE
    if _LN_STATE is None:
        raise RuntimeError("internal error: _LN_STATE is not initialized")

    import healpy as hp

    i = int(i)
    seed_i = _seed_for_mock(int(_LN_STATE["base_seed"]), i)
    np.random.seed(seed_i)  # noqa: NPY002

    cl_signal = _LN_STATE["cl_signal"]
    nside = int(_LN_STATE["nside"])
    lmax = int(_LN_STATE["lmax"])
    seen = _LN_STATE["seen"]
    mu_hat = _LN_STATE["mu_hat"]
    X = _LN_STATE["X"]
    offset = _LN_STATE["offset"]
    beta_hat = _LN_STATE["beta_hat"]
    max_iter = int(_LN_STATE["max_iter"])
    inject_D = float(_LN_STATE["inject_D"])
    b_inj = _LN_STATE["b_inj"]
    n_seen = _LN_STATE["n_seen"]
    mock_depth_alpha = float(_LN_STATE.get("mock_depth_alpha", 0.0))
    mock_depth_z_seen = _LN_STATE.get("mock_depth_z_seen")

    g = hp.synfast(cl_signal, nside=nside, lmax=lmax, new=True, verbose=False)
    fac, _ = gaussian_to_lognormal_factor(g, seen=seen)
    mu_i = mu_hat * fac[seen]
    if mock_depth_alpha != 0.0:
        if mock_depth_z_seen is None:
            raise RuntimeError("internal error: missing mock_depth_z_seen for mock_depth_alpha != 0")
        mu_i = mu_i * np.exp(mock_depth_alpha * mock_depth_z_seen)
    if inject_D != 0.0:
        mu_i = mu_i * np.exp(n_seen @ b_inj)
    rng_i = np.random.default_rng(seed_i)
    y_i = rng_i.poisson(mu_i)
    beta_i, _ = fit_poisson_glm(
        X,
        y_i,
        offset=offset,
        max_iter=max_iter,
        beta_init=beta_hat,
        compute_cov=False,
    )
    bvec = np.asarray(beta_i[1:4], dtype=float)
    D = float(np.linalg.norm(bvec))
    ang = float(axis_angle_deg(bvec, b_inj)) if inject_D != 0.0 else float("nan")
    return i, bvec, D, ang


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

    ap.add_argument("--eclip-template", choices=["none", "abs_elat", "abs_sin_elat"], default="abs_elat")
    ap.add_argument("--dust-template", choices=["none", "ebv_mean"], default="none")
    ap.add_argument(
        "--depth-mode",
        choices=["none", "w1cov_covariate", "w1cov_offset", "unwise_nexp_covariate", "unwise_nexp_offset", "depth_map_covariate", "depth_map_offset"],
        default="none",
    )
    ap.add_argument("--unwise-tiles-fits", default="data/external/unwise/tiles.fits")
    ap.add_argument("--nexp-tile-stats-json", default=None)
    ap.add_argument("--depth-map-fits", default=None)
    ap.add_argument("--depth-map-ordering", choices=["ring", "nest"], default="ring")
    ap.add_argument("--max-iter", type=int, default=250)

    ap.add_argument("--lmax", type=int, default=None, help="Max multipole for clustering C_ell (default: 3*nside-1).")
    ap.add_argument("--n-mocks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-proc", type=int, default=0, help="Number of processes (0=auto).")
    ap.add_argument(
        "--mp-start",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="Multiprocessing start method (spawn is most robust with healpy/OpenMP).",
    )
    ap.add_argument("--save-every", type=int, default=50, help="Flush partial outputs every N mocks.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing *.npy artifacts in outdir.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing *.npy artifacts in outdir.")

    ap.add_argument("--inject-dipole-amp", type=float, default=0.0, help="Optional injected dipole amplitude D≈|b|.")
    ap.add_argument("--inject-axis", default="cmb", help="'cmb' or 'l,b' in degrees for injection axis.")

    ap.add_argument(
        "--mock-depth-map-fits",
        default=None,
        help="Optional HEALPix depth map used only to inject a selection systematic in mocks.",
    )
    ap.add_argument("--mock-depth-map-ordering", choices=["ring", "nest"], default="ring")
    ap.add_argument(
        "--mock-depth-alpha",
        type=float,
        default=0.0,
        help="If non-zero, mocks are generated with mu -> mu * exp(alpha * depth_z) on seen pixels.",
    )

    ap.add_argument("--write-mock-betas", action="store_true", help="Store per-mock beta vectors in the output JSON.")
    args = ap.parse_args()

    import healpy as hp
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from scipy.spatial import cKDTree

    outdir = Path(args.outdir or f"outputs/catwise_lognormal_mocks_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Silence known healpy deprecations to keep run logs clean.
    warnings.filterwarnings("ignore", message=r".*verbose.*deprecated.*", category=Warning)

    # Load catalog columns.
    with fits.open(args.catalog, memmap=True) as hdul:
        data = hdul[1].data
        w1 = np.asarray(data["w1"], dtype=float)
        w1cov = np.asarray(data["w1cov"], dtype=float)
        l = np.asarray(data["l"], dtype=float)
        b = np.asarray(data["b"], dtype=float)
        ebv = None
        if args.dust_template != "none":
            if "ebv" not in data.names:
                raise SystemExit("catalog missing ebv")
            ebv = np.asarray(data["ebv"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    if ebv is not None:
        base &= np.isfinite(ebv)
    base &= w1cov >= float(args.w1cov_min)

    # Footprint mask is defined on the parent sample (W1cov cut only).
    theta_base = np.deg2rad(90.0 - b[base])
    phi_base = np.deg2rad(l[base])
    ipix_mask_base = hp.ang2pix(int(args.nside), theta_base, phi_base, nest=False).astype(np.int64)

    # Analysis selection.
    sel = base & (w1 <= float(args.w1_max))
    if args.w1_min is not None:
        sel &= w1 > float(args.w1_min)
    theta = np.deg2rad(90.0 - b[sel])
    phi = np.deg2rad(l[sel])
    ipix_sel = hp.ang2pix(int(args.nside), theta, phi, nest=False).astype(np.int64)

    npix = hp.nside2npix(int(args.nside))
    secrest_mask = build_secrest_mask(
        nside=int(args.nside), ipix_base=ipix_mask_base, exclude_mask_fits=str(args.exclude_mask_fits), b_cut_deg=float(args.b_cut)
    )
    seen = secrest_mask.seen

    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)
    pix_unit = lb_to_unitvec(lon_pix, lat_pix)
    n_seen = pix_unit[seen]

    counts = np.bincount(ipix_sel, minlength=npix).astype(float)
    y = counts[seen]

    # Pixel-center ecliptic latitude.
    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    elat_deg = sc_pix.barycentricmeanecliptic.lat.deg.astype(float)
    abs_elat = np.abs(elat_deg)
    abs_sin_elat = np.abs(np.sin(np.deg2rad(elat_deg)))

    # Base per-pixel means (catalog-proxy) for EBV and W1COV.
    cnt_base = np.bincount(ipix_mask_base, minlength=npix).astype(float)
    sum_w1cov = np.bincount(ipix_mask_base, weights=w1cov[base], minlength=npix).astype(float)
    w1cov_mean = np.divide(sum_w1cov, cnt_base, out=np.zeros_like(sum_w1cov), where=cnt_base != 0.0)
    if ebv is None:
        ebv_mean = np.zeros(npix, dtype=float)
    else:
        sum_ebv = np.bincount(ipix_mask_base, weights=ebv[base], minlength=npix).astype(float)
        ebv_mean = np.divide(sum_ebv, cnt_base, out=np.zeros_like(sum_ebv), where=cnt_base != 0.0)

    # Optional unWISE Nexp mapping (tile-level).
    nexp_pix = None
    if args.depth_mode.startswith("unwise_nexp_"):
        if args.nexp_tile_stats_json is None:
            raise SystemExit("--depth-mode unwise_nexp_* requires --nexp-tile-stats-json")
        tile_stats = json.loads(Path(str(args.nexp_tile_stats_json)).read_text())
        from astropy.table import Table as ATable

        tiles = ATable.read(args.unwise_tiles_fits, memmap=True)
        coadd_id = np.asarray(tiles["coadd_id"]).astype(str)
        ra = np.asarray(tiles["ra"], dtype=float)
        dec = np.asarray(tiles["dec"], dtype=float)
        g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
        tile_vec = lb_to_unitvec(g.l.deg, g.b.deg)
        tree = cKDTree(tile_vec)
        _, nn_idx = tree.query(pix_unit, k=1)
        pix_coadd = coadd_id[nn_idx]
        nexp_pix = np.array([float(tile_stats.get(str(cid), float("nan"))) for cid in pix_coadd], dtype=float)
        bad = seen & (~np.isfinite(nexp_pix) | (nexp_pix <= 0.0))
        ok = seen & np.isfinite(nexp_pix) & (nexp_pix > 0.0)
        fill = float(np.median(nexp_pix[ok])) if np.any(ok) else 1.0
        nexp_pix[bad] = fill

    # Optional generic depth map.
    depth_map = None
    if args.depth_mode.startswith("depth_map_"):
        if args.depth_map_fits is None:
            raise SystemExit("--depth-mode depth_map_* requires --depth-map-fits")
        depth_map = hp.read_map(str(args.depth_map_fits), verbose=False)
        nside_in = int(hp.get_nside(depth_map))
        order_in = "NEST" if str(args.depth_map_ordering).lower() == "nest" else "RING"
        if nside_in != int(args.nside):
            depth_map = hp.ud_grade(depth_map, nside_out=int(args.nside), order_in=order_in, order_out="RING", power=0)
        depth_map = np.asarray(depth_map, dtype=float)
        unseen = ~np.isfinite(depth_map) | (depth_map == hp.UNSEEN)
        ok = seen & (~unseen)
        fill = float(np.median(depth_map[ok])) if np.any(ok) else 0.0
        depth_map[unseen] = fill

    # Build design matrix.
    templates: list[np.ndarray] = []
    template_names: list[str] = []
    if args.eclip_template == "abs_elat":
        templates.append(zscore(abs_elat, seen)[seen])
        template_names.append("abs_elat_z")
    elif args.eclip_template == "abs_sin_elat":
        templates.append(zscore(abs_sin_elat, seen)[seen])
        template_names.append("abs_sin_elat_z")
    if args.dust_template == "ebv_mean":
        templates.append(zscore(ebv_mean, seen)[seen])
        template_names.append("ebv_mean_z")

    offset = None
    offset_name = None
    if args.depth_mode == "w1cov_covariate":
        templates.append(zscore(np.log(np.clip(w1cov_mean, 1.0, np.inf)), seen)[seen])
        template_names.append("log_w1cov_mean_z")
    elif args.depth_mode == "w1cov_offset":
        logw = np.log(np.clip(w1cov_mean, 1.0, np.inf))
        ref = float(np.median(logw[seen]))
        offset = (logw - ref)[seen]
        offset_name = "log_w1cov_mean_offset"
    elif args.depth_mode == "unwise_nexp_covariate":
        assert nexp_pix is not None
        logn = np.log(np.clip(nexp_pix, 1.0, np.inf))
        templates.append(zscore(logn, seen)[seen])
        template_names.append("log_unwise_nexp_z")
    elif args.depth_mode == "unwise_nexp_offset":
        assert nexp_pix is not None
        logn = np.log(np.clip(nexp_pix, 1.0, np.inf))
        ref = float(np.median(logn[seen]))
        offset = (logn - ref)[seen]
        offset_name = "log_unwise_nexp_offset"
    elif args.depth_mode == "depth_map_covariate":
        assert depth_map is not None
        templates.append(zscore(depth_map, seen)[seen])
        template_names.append("depth_map_z")
    elif args.depth_mode == "depth_map_offset":
        assert depth_map is not None
        ref = float(np.median(depth_map[seen]))
        offset = (depth_map - ref)[seen]
        offset_name = "depth_map_offset"

    cols = [np.ones_like(y), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2]]
    cols.extend(templates)
    X = np.column_stack(cols)

    beta_hat, _ = fit_poisson_glm(X, y, offset=offset, max_iter=int(args.max_iter))
    eta_hat = (np.zeros_like(y) if offset is None else offset) + X @ beta_hat
    eta_hat = np.clip(eta_hat, -25.0, 25.0)
    mu_hat = np.exp(eta_hat)
    diag = poisson_glm_diagnostics(y, mu_hat, n_params=int(X.shape[1]))

    # Residual-based clustering C_ell.
    lmax = int(args.lmax) if args.lmax is not None else int(3 * int(args.nside) - 1)
    resid = y / np.clip(mu_hat, 1.0, np.inf) - 1.0
    cl_signal, cl_meta = estimate_cl_from_residuals(resid_seen=resid, mu_seen=mu_hat, seen=seen, pix_unit=pix_unit, lmax=lmax)

    # Optional injected dipole vector.
    inject_D = float(args.inject_dipole_amp)
    if args.inject_axis.strip().lower() == "cmb":
        axis_l, axis_b = 264.021, 48.253
    else:
        parts = args.inject_axis.split(",")
        if len(parts) != 2:
            raise SystemExit("--inject-axis must be 'cmb' or 'l,b'")
        axis_l, axis_b = float(parts[0]), float(parts[1])
    n_axis = lb_to_unitvec(np.array([axis_l]), np.array([axis_b]))[0]
    b_inj = inject_D * n_axis

    # Optional injected depth systematic used only in mock generation.
    mock_depth_alpha = float(args.mock_depth_alpha)
    mock_depth_z_seen = None
    if mock_depth_alpha != 0.0:
        if args.mock_depth_map_fits is None:
            raise SystemExit("--mock-depth-alpha != 0 requires --mock-depth-map-fits")
        mock_depth_map = hp.read_map(str(args.mock_depth_map_fits), verbose=False)
        nside_in = int(hp.get_nside(mock_depth_map))
        order_in = "NEST" if str(args.mock_depth_map_ordering).lower() == "nest" else "RING"
        if nside_in != int(args.nside):
            mock_depth_map = hp.ud_grade(
                mock_depth_map,
                nside_out=int(args.nside),
                order_in=order_in,
                order_out="RING",
                power=0,
            )
        mock_depth_map = np.asarray(mock_depth_map, dtype=float)
        unseen = ~np.isfinite(mock_depth_map) | (mock_depth_map == hp.UNSEEN)
        ok = seen & (~unseen)
        fill = float(np.median(mock_depth_map[ok])) if np.any(ok) else 0.0
        mock_depth_map[unseen] = fill
        mock_depth_z_seen = zscore(mock_depth_map, seen)[seen].astype(float)

    n_mocks = int(args.n_mocks)
    if n_mocks < 1:
        raise SystemExit("--n-mocks must be >= 1")
    save_every = int(max(1, int(args.save_every)))

    # Persist partial results as .npy memmaps (resumability and low overhead).
    b_est_path = outdir / "b_est.npy"
    D_est_path = outdir / "D_est.npy"
    ang_path = outdir / "axis_angle_to_inj_deg.npy"
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive.")

    def open_or_create(path: Path, *, shape: tuple[int, ...]) -> np.memmap:
        if path.exists():
            if args.resume:
                arr = np.load(path, mmap_mode="r+")
                if tuple(arr.shape) != tuple(shape):
                    raise SystemExit(f"Existing {path} has shape {arr.shape} but expected {shape}.")
                if arr.dtype != np.dtype("f4"):
                    raise SystemExit(f"Existing {path} has dtype {arr.dtype} but expected float32.")
                return arr
            if not args.overwrite:
                raise SystemExit(f"Refusing to overwrite existing {path}; pass --overwrite or --resume.")
        arr = np.lib.format.open_memmap(path, mode="w+", dtype="f4", shape=shape)
        arr[:] = np.nan
        arr.flush()
        return arr

    b_est = open_or_create(b_est_path, shape=(n_mocks, 3))
    D_est = open_or_create(D_est_path, shape=(n_mocks,))
    ang_to_inj = open_or_create(ang_path, shape=(n_mocks,))

    nproc = _auto_nproc(int(args.n_proc))

    # Initialize worker state for both sequential and multiprocessing execution.
    global _LN_STATE
    _LN_STATE = {
        "base_seed": int(args.seed),
        "cl_signal": cl_signal,
        "nside": int(args.nside),
        "lmax": int(lmax),
        "seen": seen,
        "mu_hat": mu_hat,
        "X": X,
        "offset": offset,
        "beta_hat": beta_hat,
        "max_iter": int(args.max_iter),
        "inject_D": float(inject_D),
        "b_inj": b_inj,
        "n_seen": n_seen,
        "mock_depth_alpha": float(mock_depth_alpha),
        "mock_depth_z_seen": mock_depth_z_seen,
    }

    ok0 = np.isfinite(D_est)
    done = int(np.sum(ok0))
    todo = np.flatnonzero(~ok0)
    if todo.size and done:
        print(f"Resuming: {done}/{n_mocks} completed, {todo.size} remaining.")
    if nproc <= 1:
        for i in todo.tolist():
            j, bvec, D, ang = _run_one_mock(int(i))
            b_est[j] = bvec.astype(np.float32)
            D_est[j] = np.float32(D)
            ang_to_inj[j] = np.float32(ang)
            done += 1
            if (done % save_every) == 0 or done == n_mocks:
                b_est.flush()
                D_est.flush()
                ang_to_inj.flush()
                (outdir / "progress.txt").write_text(f"{done}/{n_mocks}\\n")
                print(f"[{done}/{n_mocks}] saved")
    else:
        import multiprocessing as mp

        ctx = mp.get_context(str(args.mp_start))
        if str(args.mp_start) == "fork":
            pool_kwargs = {"processes": int(nproc)}
        else:
            pool_kwargs = {"processes": int(nproc), "initializer": _init_lognormal_worker, "initargs": (_LN_STATE,)}
        with ctx.Pool(**pool_kwargs) as pool:
            for j, bvec, D, ang in pool.imap_unordered(_run_one_mock, todo.tolist(), chunksize=1):
                b_est[j] = bvec.astype(np.float32)
                D_est[j] = np.float32(D)
                ang_to_inj[j] = np.float32(ang)
                done += 1
                if (done % save_every) == 0 or done == n_mocks:
                    b_est.flush()
                    D_est.flush()
                    ang_to_inj.flush()
                    (outdir / "progress.txt").write_text(f"{done}/{n_mocks}\\n")
                    print(f"[{done}/{n_mocks}] saved")

    # Compute covariance on finite entries only.
    ok = np.isfinite(D_est)
    if int(np.sum(ok)) < 3:
        raise RuntimeError("Too few completed mocks to estimate covariance.")
    b_use = np.asarray(b_est[ok], dtype=float)
    D_use = np.asarray(D_est[ok], dtype=float)
    ang_use = np.asarray(ang_to_inj[ok], dtype=float)
    cov_b = np.cov(b_use.T, ddof=1)

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.nanpercentile(a, q))

    summary = {
        "n_completed": int(np.sum(ok)),
        "D_p16": pct(D_use, 16),
        "D_p50": pct(D_use, 50),
        "D_p84": pct(D_use, 84),
    }
    if inject_D != 0.0:
        summary["axis_angle_to_inj_p16_deg"] = pct(ang_use, 16)
        summary["axis_angle_to_inj_p50_deg"] = pct(ang_use, 50)
        summary["axis_angle_to_inj_p84_deg"] = pct(ang_use, 84)

    payload: dict[str, Any] = {
        "meta": {
            "catalog": str(args.catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": int(args.nside),
            "lmax": int(lmax),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_min": None if args.w1_min is None else float(args.w1_min),
            "w1_max": float(args.w1_max),
            "eclip_template": str(args.eclip_template),
            "dust_template": str(args.dust_template),
            "depth_mode": str(args.depth_mode),
            "template_names": template_names,
            "offset_name": offset_name,
            "n_mocks": int(args.n_mocks),
            "seed": int(args.seed),
            "n_proc": int(args.n_proc),
            "mp_start": str(args.mp_start),
            "save_every": int(save_every),
            "resume": bool(args.resume),
            "overwrite": bool(args.overwrite),
            "inject_dipole_amp": float(inject_D),
            "inject_axis_lb": [axis_l, axis_b],
            "mock_depth_alpha": float(mock_depth_alpha),
            "mock_depth_map_fits": None if args.mock_depth_map_fits is None else str(args.mock_depth_map_fits),
            "mock_depth_map_ordering": str(args.mock_depth_map_ordering),
        },
        "fit": {
            "beta_hat": [float(x) for x in beta_hat],
            "dipole_hat": {
                "D": float(np.linalg.norm(beta_hat[1:4])),
                "l_deg": vec_to_lb(beta_hat[1:4])[0],
                "b_deg": vec_to_lb(beta_hat[1:4])[1],
            },
            "diag": diag,
        },
        "cl_estimate": {
            "meta": cl_meta,
            "cl_signal": [float(x) for x in cl_signal.tolist()],
        },
        "mocks": {
            "cov_b": [[float(x) for x in row] for row in cov_b.tolist()],
            "summary": summary,
        },
    }
    if args.write_mock_betas:
        payload["mocks"]["b_est"] = [[float(x) for x in row] for row in b_use.tolist()]
        payload["mocks"]["D_est"] = [float(x) for x in D_use.tolist()]
        payload["mocks"]["axis_angle_to_inj_deg"] = [float(x) for x in ang_use.tolist()]
    payload["mocks"]["artifacts"] = {
        "b_est_npy": str(b_est_path),
        "D_est_npy": str(D_est_path),
        "axis_angle_to_inj_deg_npy": str(ang_path),
    }

    out_json = outdir / "lognormal_mocks_cov.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
