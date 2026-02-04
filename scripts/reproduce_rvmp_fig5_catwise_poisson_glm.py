#!/usr/bin/env python3
"""
RvMP Fig.5-style CatWISE dipole scan using a Poisson maximum-likelihood model.

This is the "likelihood" counterpart to `scripts/reproduce_rvmp_fig5_catwise.py`,
which reproduces the Secrest+22 weighted linear dipole estimator.

Model
-----
For each unmasked HEALPix pixel p:

  N_p ~ Poisson(mu_p)
  log mu_p = beta0 + b · n_p + Σ_k c_k T_{k,p} + offset_p

where:
  - n_p is the unit vector of the pixel center in Galactic coordinates,
  - b is a 3-vector; for small dipoles D ≈ |b|,
  - T_{k,p} are nuisance templates (ecliptic latitude trend, dust proxy, depth proxy),
  - offset_p is an optional fixed offset (e.g. log depth).

This matches the RvMP wording "marginalize over the ecliptic trend" more literally:
we include the trend as nuisance parameters in the likelihood and (approximately)
marginalize via a Gaussian approximation to the fitted parameter covariance.

Masking footprint
-----------------
Matches Secrest+22:
  - mask_zeros on the full W1cov>=80 map (zeros + neighbours),
  - exclude_master_revised.fits discs (use==True),
  - Galactic plane cut |b| < b_cut (pixel centers).

Optional injection
------------------
Supports an injected dipolar faint-limit modulation (pure selection effect):
  W1_eff = W1 - delta_m * cos(theta_axis)
and then apply W1_eff < W1_max (and optional W1_eff > W1_min).
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


def skyarea_deg2() -> float:
    return float(4.0 * math.pi * (180.0 / math.pi) ** 2)


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


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x[valid])) if np.any(valid) else 0.0
    s = float(np.std(x[valid])) if np.any(valid) else 1.0
    if s == 0.0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


def alpha_edge_from_cumcounts(w1_eff: np.ndarray, cut: float, *, delta: float) -> float:
    """Estimate alpha_edge = d ln N(<m) / dm at m=cut via finite difference."""
    w1_eff = np.asarray(w1_eff, dtype=float)
    n1 = int(np.sum(w1_eff <= float(cut)))
    n0 = int(np.sum(w1_eff <= float(cut) - float(delta)))
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    return float((math.log(n1) - math.log(n0)) / float(delta))


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

    # Fisher / covariance approximation: (X^T diag(mu) X)^{-1}
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
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape.")
    if y.ndim != 1:
        raise ValueError("y and mu must be 1D arrays.")

    dof = int(y.size) - int(n_params)
    dof_f = float(dof) if dof > 0 else float("nan")

    mu_pos = np.clip(mu, 1e-30, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        # deviance contribution: 2 * [y*log(y/mu) - (y-mu)], with 0*log(0)=0.
        term = np.where(y > 0.0, y * np.log(y / mu_pos), 0.0)
        dev = 2.0 * np.sum(term - (y - mu_pos))
        pearson = np.sum((y - mu_pos) ** 2 / mu_pos)

    dev = float(dev)
    pearson = float(pearson)
    dev_over_dof = float(dev / dof_f) if np.isfinite(dof_f) else float("nan")
    pearson_over_dof = float(pearson / dof_f) if np.isfinite(dof_f) else float("nan")

    # Quasi-Poisson dispersion estimate (conservative: never < 1).
    phi = pearson_over_dof
    phi_eff = float(max(1.0, phi)) if np.isfinite(phi) else float("nan")

    return {
        "n_obs": float(y.size),
        "n_params": float(n_params),
        "dof": float(dof),
        "deviance": dev,
        "pearson_chi2": pearson,
        "deviance_over_dof": dev_over_dof,
        "pearson_over_dof": pearson_over_dof,
        "dispersion_phi": float(phi) if np.isfinite(phi) else float("nan"),
        "dispersion_phi_eff": phi_eff,
    }


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square 2D array.")
    v = np.diag(cov)
    s = np.sqrt(np.clip(v, 0.0, np.inf))
    denom = s[:, None] * s[None, :]
    out = np.divide(cov, denom, out=np.zeros_like(cov), where=denom != 0.0)
    return out


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


def fit_template_dipole_vec(n_seen: np.ndarray, t_seen: np.ndarray) -> np.ndarray:
    """
    Fit template values to a monopole+dipole model:
        t = a0 + a · n
    Return the 3-vector a (dipole component).
    """
    n_seen = np.asarray(n_seen, dtype=float)
    t_seen = np.asarray(t_seen, dtype=float)
    X = np.column_stack([np.ones_like(t_seen), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2]])
    beta, *_ = np.linalg.lstsq(X, t_seen, rcond=None)
    return np.asarray(beta[1:4], dtype=float)


def jackknife_bvec_cov(
    *,
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray | None,
    beta_full: np.ndarray,
    jk_region: np.ndarray,
    jk_regions: np.ndarray,
    max_iter: int,
) -> dict[str, Any]:
    """
    Leave-one-region-out jackknife for the Poisson GLM dipole vector b=beta[1:4].
    Returns summary including jackknife covariance for b.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    reg = np.asarray(jk_region, dtype=np.int64)
    regions = np.asarray(jk_regions, dtype=np.int64)
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y shapes inconsistent.")
    if reg.shape != (y.size,):
        raise ValueError("jk_region must have shape (n_obs,).")

    b_list: list[np.ndarray] = []
    used: list[int] = []
    for r in regions.tolist():
        keep = reg != int(r)
        if int(np.sum(keep)) <= X.shape[1] + 5:
            continue
        beta_r, _ = fit_poisson_glm(
            X[keep],
            y[keep],
            offset=None if offset is None else offset[keep],
            max_iter=int(max_iter),
            beta_init=beta_full,
            compute_cov=False,
        )
        b_r = np.asarray(beta_r[1:4], dtype=float)
        if not np.all(np.isfinite(b_r)):
            continue
        b_list.append(b_r)
        used.append(int(r))

    if len(b_list) < 3:
        return {
            "n_regions_requested": int(regions.size),
            "n_regions_used": int(len(b_list)),
            "regions_used": used,
            "cov_b": None,
        }

    b = np.vstack(b_list)
    b_mean = np.mean(b, axis=0)
    n = float(b.shape[0])
    dif = b - b_mean[None, :]
    cov = (n - 1.0) / n * (dif.T @ dif)

    b_full = np.asarray(beta_full[1:4], dtype=float)
    ang = np.array([axis_angle_deg(v, b_full) for v in b], dtype=float)
    amp = np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore"):
        if float(np.linalg.norm(b_full)) > 0.0:
            u = b_full / float(np.linalg.norm(b_full))
            D_sigma = float(math.sqrt(max(0.0, float(u @ cov @ u))))
        else:
            D_sigma = float("nan")

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.nanpercentile(a, q))

    return {
        "n_regions_requested": int(regions.size),
        "n_regions_used": int(b.shape[0]),
        "regions_used": used,
        "b_mean": [float(x) for x in b_mean],
        "cov_b": [[float(x) for x in row] for row in cov.tolist()],
        "D_p16": pct(amp, 16),
        "D_p50": pct(amp, 50),
        "D_p84": pct(amp, 84),
        "D_sigma": D_sigma,
        "axis_angle_to_full_p16_deg": pct(ang, 16),
        "axis_angle_to_full_p50_deg": pct(ang, 50),
        "axis_angle_to_full_p84_deg": pct(ang, 84),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--mask-catalog",
        default=None,
        help=(
            "Optional separate catalog used ONLY to build the footprint mask and any per-pixel template means. "
            "Use this when the analysis catalog is a filtered/subsampled file (e.g. NVSS-removed)."
        ),
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1-min", type=float, default=None, help="Optional bright cut (keep only W1 > w1_min).")
    ap.add_argument("--w1-grid", default="15.5,16.6,0.05", help="start,stop,step (inclusive end).")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument(
        "--harmonic-lmax",
        type=int,
        default=1,
        help=(
            "If >1, include real spherical-harmonic nuisance templates Y_lm for 2<=l<=harmonic_lmax "
            "(z-scored on seen pixels). This is an amplitude-robustness diagnostic motivated by critiques "
            "that low-order multipoles beyond the dipole can be comparable in size and leak through the mask."
        ),
    )

    ap.add_argument(
        "--eclip-template",
        choices=["none", "abs_elat", "abs_sin_elat", "abs_elat_sincos_elon"],
        default="abs_elat",
    )
    ap.add_argument("--dust-template", choices=["none", "ebv_mean"], default="none")
    ap.add_argument(
        "--depth-mode",
        choices=[
            "none",
            "w1cov_covariate",
            "w1cov_offset",
            "unwise_nexp_covariate",
            "unwise_nexp_offset",
            "depth_map_covariate",
            "depth_map_offset",
            "delta_m_map_offset_alpha_edge",
            "delta_m_map_covariate_alpha_edge",
            "external_logreg_integrated_offset",
        ],
        default="none",
    )
    ap.add_argument(
        "--external-logreg-meta",
        default=None,
        help=(
            "Meta JSON produced by scripts/run_sdss_dr16q_completeness_model.py (or similar). "
            "Required for --depth-mode external_logreg_integrated_offset."
        ),
    )
    ap.add_argument(
        "--external-logreg-mrange",
        type=float,
        default=3.0,
        help=(
            "Magnitude range (mag) below W1_max used to approximate the cumulative completeness integral when "
            "--depth-mode external_logreg_integrated_offset."
        ),
    )
    ap.add_argument(
        "--external-logreg-dm",
        type=float,
        default=0.01,
        help=(
            "Magnitude step (mag) for the completeness integral when --depth-mode external_logreg_integrated_offset."
        ),
    )
    ap.add_argument(
        "--depth-map-fits",
        default=None,
        help=(
            "Optional HEALPix depth map FITS (assumed Galactic coords). Required for --depth-mode depth_map_*. "
            "If the map NSIDE differs from --nside, it is resampled with healpy.ud_grade."
        ),
    )
    ap.add_argument(
        "--depth-map-ordering",
        choices=["ring", "nest"],
        default="ring",
        help="HEALPix ordering of --depth-map-fits (used when resampling).",
    )
    ap.add_argument(
        "--depth-map-name",
        default=None,
        help="Optional label stored in output JSON (e.g. 'unwise_lognexp_nside64').",
    )
    ap.add_argument(
        "--extra-offset-map-fits",
        default=None,
        help=(
            "Optional additional HEALPix map used as a fixed log-intensity offset (Galactic coords). "
            "If provided, the map is median-centered on seen pixels and added to any depth-mode offset."
        ),
    )
    ap.add_argument(
        "--extra-offset-map-ordering",
        choices=["ring", "nest"],
        default="ring",
        help="HEALPix ordering of --extra-offset-map-fits (used when resampling).",
    )
    ap.add_argument(
        "--extra-offset-map-name",
        default=None,
        help="Optional label stored in output JSON for --extra-offset-map-fits.",
    )
    ap.add_argument(
        "--extra-template-map-fits",
        default=None,
        help=(
            "Optional additional HEALPix map used as a free nuisance template (Galactic coords). "
            "The template is z-scored on seen pixels and appended after the built-in templates."
        ),
    )
    ap.add_argument(
        "--extra-template-map-ordering",
        choices=["ring", "nest"],
        default="ring",
        help="HEALPix ordering of --extra-template-map-fits (used when resampling).",
    )
    ap.add_argument(
        "--extra-template-map-name",
        default=None,
        help="Optional label stored in output JSON for --extra-template-map-fits.",
    )
    ap.add_argument(
        "--unwise-tiles-fits",
        default="data/external/unwise/tiles.fits",
        help="unWISE tiles table (ra/dec/coadd_id). Used to build a HEALPix depth proxy via nearest-tile mapping.",
    )
    ap.add_argument(
        "--nexp-tile-stats-json",
        default=None,
        help=(
            "Optional JSON mapping {coadd_id: nexp_stat} (e.g. median W1 exposures from unWISE w1-n-m maps). "
            "Required for --depth-mode unwise_nexp_*."
        ),
    )

    ap.add_argument("--mc-draws", type=int, default=400, help="Approx marginalization draws from N(beta,cov).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--make-plot", action="store_true")
    ap.add_argument(
        "--w1-mode",
        choices=["cumulative", "differential"],
        default="cumulative",
        help="Selection mode: cumulative (W1_eff < cut) or differential bins between successive cuts.",
    )

    ap.add_argument("--jackknife-nside", type=int, default=0, help="If >0, compute a sky jackknife at this NSIDE.")
    ap.add_argument("--jackknife-stride", type=int, default=1, help="Compute jackknife every Nth cut (1=all).")
    ap.add_argument("--jackknife-max-iter", type=int, default=120, help="Max optimizer iterations per jackknife fit.")
    ap.add_argument(
        "--jackknife-max-regions",
        type=int,
        default=None,
        help="Optional cap on number of jackknife regions (subsampled deterministically).",
    )

    # Injection
    ap.add_argument("--inject-delta-m-mag", type=float, default=0.0)
    ap.add_argument("--inject-axis", default="cmb", help="'cmb' or 'l,b' in degrees for injection axis.")

    args = ap.parse_args()

    import healpy as hp
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    outdir = Path(args.outdir or f"outputs/catwise_rvmp_fig5_poisson_glm_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.catalog, memmap=True) as hdul:
        data = hdul[1].data
        w1 = np.asarray(data["w1"], dtype=float)
        w1cov = np.asarray(data["w1cov"], dtype=float)
        l = np.asarray(data["l"], dtype=float)
        b = np.asarray(data["b"], dtype=float)
        ebv = None
        if args.dust_template != "none":
            if "ebv" not in data.names:
                raise SystemExit(f"catalog missing ebv but --dust-template={args.dust_template}")
            ebv = np.asarray(data["ebv"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    if ebv is not None:
        base &= np.isfinite(ebv)
    base &= w1cov >= float(args.w1cov_min)

    theta = np.deg2rad(90.0 - b[base])
    phi = np.deg2rad(l[base])
    ipix_base = hp.ang2pix(int(args.nside), theta, phi, nest=False)
    ipix_mask_base = np.asarray(ipix_base, dtype=np.int64)

    # Optional injection effective faint-limit modulation.
    inject_delta_m = float(args.inject_delta_m_mag)
    if args.inject_axis.strip().lower() == "cmb":
        axis_l, axis_b = 264.021, 48.253
    else:
        parts = args.inject_axis.split(",")
        if len(parts) != 2:
            raise SystemExit("--inject-axis must be 'cmb' or 'l,b'")
        axis_l, axis_b = float(parts[0]), float(parts[1])

    if inject_delta_m != 0.0:
        n_src = lb_to_unitvec(l[base], b[base])
        n_axis = lb_to_unitvec(np.array([axis_l]), np.array([axis_b]))[0]
        cos_theta = n_src @ n_axis
        w1_eff = w1[base] - inject_delta_m * cos_theta
    else:
        w1_eff = w1[base].copy()

    # Apply bright-side cut on the same effective magnitude.
    if args.w1_min is not None:
        keep = w1_eff > float(args.w1_min)
        w1_eff = w1_eff[keep]
        ipix_eff = ipix_base[keep]
        ebv_eff = None if ebv is None else ebv[base][keep]
        w1cov_eff = w1cov[base][keep]
        l_eff = l[base][keep]
        b_eff = b[base][keep]
    else:
        ipix_eff = ipix_base
        ebv_eff = None if ebv is None else ebv[base]
        w1cov_eff = w1cov[base]
        l_eff = l[base]
        b_eff = b[base]

    # Sort by W1_eff for incremental cumulative updates.
    order = np.argsort(w1_eff)
    w1_eff_sorted = w1_eff[order]
    ipix_sorted = np.asarray(ipix_eff, dtype=np.int64)[order]

    # Parse grid.
    w1_start, w1_stop, w1_step = (float(x) for x in args.w1_grid.split(","))
    n_steps = int(round((w1_stop - w1_start) / w1_step)) + 1
    cuts = [w1_start + i * w1_step for i in range(n_steps)]

    # Optionally, build the footprint (mask_zeros) and template means from a separate catalog.
    if args.mask_catalog is not None:
        with fits.open(args.mask_catalog, memmap=True) as hdul:
            dm = hdul[1].data
            w1cov_m = np.asarray(dm["w1cov"], dtype=float)
            l_m = np.asarray(dm["l"], dtype=float)
            b_m = np.asarray(dm["b"], dtype=float)
            base_m = np.isfinite(w1cov_m) & np.isfinite(l_m) & np.isfinite(b_m)
            base_m &= w1cov_m >= float(args.w1cov_min)
            theta_m = np.deg2rad(90.0 - b_m[base_m])
            phi_m = np.deg2rad(l_m[base_m])
            ipix_mask_base = hp.ang2pix(int(args.nside), theta_m, phi_m, nest=False).astype(np.int64)

            if args.dust_template == "ebv_mean":
                if "ebv" not in dm.names:
                    raise SystemExit(f"mask-catalog missing ebv but --dust-template={args.dust_template}")
                ebv_m_base = np.asarray(dm["ebv"], dtype=float)[base_m]
            else:
                ebv_m_base = None

            w1cov_m_base = w1cov_m[base_m]
    else:
        ebv_m_base = None if ebv is None else ebv[base]
        w1cov_m_base = w1cov[base]

    npix = hp.nside2npix(int(args.nside))

    # Mask + pixel geometry.
    secrest_mask = build_secrest_mask(
        nside=int(args.nside),
        ipix_base=ipix_mask_base,
        exclude_mask_fits=args.exclude_mask_fits,
        b_cut_deg=float(args.b_cut),
    )
    mask = secrest_mask.mask
    seen = secrest_mask.seen

    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)  # galactic l,b
    pix_unit = lb_to_unitvec(lon_pix, lat_pix)

    # Optional jackknife region assignment (pixel-center, restricted to seen pixels).
    jk_nside = int(args.jackknife_nside)
    if jk_nside > 0:
        th_pix = np.deg2rad(90.0 - lat_pix)
        ph_pix = np.deg2rad(lon_pix % 360.0)
        jk_region_all = hp.ang2pix(jk_nside, th_pix, ph_pix, nest=False).astype(np.int64)
        jk_region_seen = jk_region_all[seen]
        jk_regions = np.unique(jk_region_seen)
        if args.jackknife_max_regions is not None and jk_regions.size > int(args.jackknife_max_regions):
            # Deterministic subsample: take evenly-spaced region IDs.
            idx = np.linspace(0, jk_regions.size - 1, int(args.jackknife_max_regions), dtype=int)
            jk_regions = jk_regions[idx]
    else:
        jk_region_seen = None
        jk_regions = None

    # Pixel-center ecliptic lat (templates).
    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    elat_deg = sc_pix.barycentricmeanecliptic.lat.deg.astype(float)
    elon_deg = sc_pix.barycentricmeanecliptic.lon.deg.astype(float)
    abs_elat = np.abs(elat_deg)
    abs_sin_elat = np.abs(np.sin(np.deg2rad(elat_deg)))
    sin_elon = np.sin(np.deg2rad(elon_deg))
    cos_elon = np.cos(np.deg2rad(elon_deg))

    # Optional: externally trained logistic-regression completeness model.
    ext_logreg = None
    if args.depth_mode == "external_logreg_integrated_offset":
        if args.external_logreg_meta is None:
            raise SystemExit("--depth-mode external_logreg_integrated_offset requires --external-logreg-meta")
        ext_meta_path = Path(str(args.external_logreg_meta))
        if not ext_meta_path.exists():
            raise SystemExit(f"Missing --external-logreg-meta: {ext_meta_path}")
        ext_meta = json.loads(ext_meta_path.read_text())

        if "logreg" not in ext_meta or "coef" not in ext_meta["logreg"] or "intercept" not in ext_meta["logreg"]:
            raise SystemExit(f"--external-logreg-meta missing logreg coef/intercept: {ext_meta_path}")
        feature_set = str(ext_meta.get("feature_set", ""))
        if feature_set not in ("depth_only", "depth_plus_ecliptic"):
            raise SystemExit(
                f"--external-logreg-meta feature_set={feature_set!r} not supported; expected 'depth_only' or "
                "'depth_plus_ecliptic'"
            )

        depth_map_path = Path(str(ext_meta["depth_map_fits"]))
        if not depth_map_path.exists():
            raise SystemExit(f"external completeness depth_map_fits missing: {depth_map_path}")

        ext_depth = hp.read_map(str(depth_map_path), verbose=False)
        nside_in = int(hp.get_nside(ext_depth))
        if nside_in != int(args.nside):
            ext_depth = hp.ud_grade(ext_depth, nside_out=int(args.nside), order_in="RING", order_out="RING", power=0)
        ext_depth = np.asarray(ext_depth, dtype=float)
        trans = str(ext_meta.get("depth_map_transform", "none"))
        if trans == "log":
            ext_depth = np.log(np.clip(ext_depth, 1e-12, np.inf))
        elif trans == "log10":
            ext_depth = np.log10(np.clip(ext_depth, 1e-12, np.inf))
        elif trans not in ("none", "identity", ""):
            raise SystemExit(f"Unsupported external completeness depth_map_transform={trans!r} in {ext_meta_path}")

        unseen = ~np.isfinite(ext_depth) | (ext_depth == hp.UNSEEN)
        ok = seen & (~unseen)
        if not np.any(ok):
            raise SystemExit("External completeness depth map has no finite values on seen pixels.")
        fill = float(np.median(ext_depth[ok]))
        ext_depth[unseen] = fill

        coef = np.asarray(ext_meta["logreg"]["coef"], dtype=float)
        intercept = float(ext_meta["logreg"]["intercept"])
        if feature_set == "depth_plus_ecliptic":
            if coef.size != 6:
                raise SystemExit(
                    f"Unsupported external completeness coef length {coef.size} (expected 6) in {ext_meta_path}"
                )
            b_w1, b_x, b_w1x, b_abs, b_sin, b_cos = [float(x) for x in coef]

            # Pixel-center ecliptic lon/lat consistent with scripts/run_sdss_dr16q_completeness_model.py
            # (J2000 obliquity formulae on ICRS RA/Dec).
            def ecl_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                ra = np.deg2rad(np.asarray(ra_deg, dtype=float) % 360.0)
                dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
                sin_dec = np.sin(dec)
                cos_dec = np.cos(dec)
                sin_ra = np.sin(ra)
                cos_ra = np.cos(ra)

                eps = math.radians(23.4392911)  # J2000 obliquity
                sin_eps = math.sin(eps)
                cos_eps = math.cos(eps)

                sin_beta = sin_dec * cos_eps - cos_dec * sin_eps * sin_ra
                beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))

                y = sin_ra * cos_eps + np.tan(dec) * sin_eps
                x = cos_ra
                lam = np.arctan2(y, x) % (2.0 * math.pi)

                return np.rad2deg(lam), np.rad2deg(beta)

            sc_icrs = sc_pix.icrs
            elon_lr, elat_lr = ecl_from_radec_deg(sc_icrs.ra.deg, sc_icrs.dec.deg)
            abs_elat_lr = np.abs(elat_lr).astype(float)
            sin_elon_lr = np.sin(np.deg2rad(elon_lr)).astype(float)
            cos_elon_lr = np.cos(np.deg2rad(elon_lr)).astype(float)
        else:
            if coef.size != 3:
                raise SystemExit(
                    f"Unsupported external completeness coef length {coef.size} (expected 3) in {ext_meta_path}"
                )
            b_w1, b_x, b_w1x = [float(x) for x in coef]
            b_abs = b_sin = b_cos = 0.0
            abs_elat_lr = sin_elon_lr = cos_elon_lr = 0.0

        denom = b_w1 + b_w1x * ext_depth
        denom_safe = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6 + 1e-6, denom).astype(float)
        num = intercept + b_x * ext_depth
        if feature_set == "depth_plus_ecliptic":
            num = num + b_abs * abs_elat_lr + b_sin * sin_elon_lr + b_cos * cos_elon_lr
        m50 = (-num / denom_safe).astype(float)

        ext_logreg = {
            "meta_path": str(ext_meta_path),
            "feature_set": feature_set,
            "depth_map_path": str(depth_map_path),
            "depth_map_transform": trans,
            "depth_map_nside_in": nside_in,
            "depth_map_fill_value": float(fill),
            "coef": [b_w1, b_x, b_w1x, b_abs, b_sin, b_cos],
            "intercept": float(intercept),
            "m50": m50,  # per pixel (nside used)
            "denom": denom_safe,  # per pixel (nside used)
            "mrange": float(args.external_logreg_mrange),
            "dm": float(args.external_logreg_dm),
        }

    # Optional: real spherical-harmonic nuisance templates (computed once; reused for all cuts).
    harmonic_templates_seen: list[np.ndarray] = []
    harmonic_template_names: list[str] = []
    if int(args.harmonic_lmax) > 1:
        try:
            # SciPy <1.17
            from scipy.special import sph_harm as _sph_harm  # type: ignore

            def sph_harm(l: int, m: int, th: np.ndarray, ph: np.ndarray) -> np.ndarray:
                # Legacy SciPy signature: sph_harm(m, l, phi, theta)
                return _sph_harm(m, l, ph, th)

        except Exception:
            # SciPy >=1.17
            from scipy.special import sph_harm_y as _sph_harm_y  # type: ignore

            def sph_harm(l: int, m: int, th: np.ndarray, ph: np.ndarray) -> np.ndarray:
                # New SciPy signature: sph_harm_y(n=l, m=m, theta, phi)
                return _sph_harm_y(l, m, th, ph)

        th = np.deg2rad(90.0 - lat_pix)  # colatitude
        ph = np.deg2rad(lon_pix % 360.0)  # longitude

        for ell in range(2, int(args.harmonic_lmax) + 1):
            y0 = sph_harm(ell, 0, th, ph).real.astype(float)
            harmonic_templates_seen.append(zscore(y0, seen)[seen])
            harmonic_template_names.append(f"Y{ell}_0_re_z")
            for m in range(1, ell + 1):
                y = sph_harm(ell, m, th, ph)
                harmonic_templates_seen.append(zscore((np.sqrt(2.0) * y.real).astype(float), seen)[seen])
                harmonic_template_names.append(f"Y{ell}_{m}_re_z")
                harmonic_templates_seen.append(zscore((np.sqrt(2.0) * y.imag).astype(float), seen)[seen])
                harmonic_template_names.append(f"Y{ell}_{m}_im_z")

    # Optional: independent unWISE depth-of-coverage proxy (Nexp), mapped per HEALPix pixel
    # via nearest unWISE tile center (as in experiments/quasar_dipole_hypothesis/vector_convergence_glm_cv.py).
    nexp_pix = None
    nexp_missing_frac = None
    if args.depth_mode.startswith("unwise_nexp_"):
        if args.nexp_tile_stats_json is None:
            raise SystemExit("--depth-mode unwise_nexp_* requires --nexp-tile-stats-json")

        tile_stats = json.load(open(args.nexp_tile_stats_json, "r"))
        from astropy.table import Table as ATable
        from scipy.spatial import cKDTree

        tiles = ATable.read(args.unwise_tiles_fits, memmap=True)
        coadd_id = np.asarray(tiles["coadd_id"]).astype(str)
        ra = np.asarray(tiles["ra"], dtype=float)
        dec = np.asarray(tiles["dec"], dtype=float)
        g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
        lg = np.deg2rad(g.l.deg % 360.0)
        bg = np.deg2rad(g.b.deg)
        cosb = np.cos(bg)
        tile_vec = np.column_stack([cosb * np.cos(lg), cosb * np.sin(lg), np.sin(bg)])
        # Only use tiles present in the stats JSON (otherwise nearest-neighbour mapping
        # can land on a missing tile and force large-area fill values).
        valid_tile = np.fromiter((str(cid) in tile_stats for cid in coadd_id), dtype=bool, count=coadd_id.size)
        if not np.any(valid_tile):
            raise SystemExit("nexp-tile-stats-json contains no keys matching tiles.fits coadd_id values.")
        tree = cKDTree(tile_vec[valid_tile])

        # Map each HEALPix pixel to nearest tile.
        _, nn_idx = tree.query(pix_unit, k=1)
        pix_coadd = coadd_id[valid_tile][np.asarray(nn_idx, dtype=int)]
        nexp_pix = np.array([float(tile_stats[str(cid)]) for cid in pix_coadd], dtype=float)

        bad = ~np.isfinite(nexp_pix) | (nexp_pix <= 0.0)
        nexp_missing_frac = float(np.mean(seen & bad))
        if np.any(bad):
            ok = seen & (~bad)
            if not np.any(ok):
                raise SystemExit("No valid Nexp values found on seen pixels; cannot use unwise_nexp depth-mode.")
            fill = float(np.median(nexp_pix[ok]))
            nexp_pix[bad] = fill

    # Optional: generic map-level depth proxy provided as a HEALPix map.
    depth_map = None
    depth_map_meta = None
    if str(args.depth_mode).startswith("depth_map_") or str(args.depth_mode).startswith("delta_m_map_"):
        if args.depth_map_fits is None:
            raise SystemExit("--depth-mode depth_map_* or delta_m_map_* requires --depth-map-fits")
        depth_map_path = Path(str(args.depth_map_fits))
        if not depth_map_path.exists():
            raise SystemExit(f"Missing --depth-map-fits: {depth_map_path}")

        depth_map = hp.read_map(str(depth_map_path), verbose=False)
        nside_in = int(hp.get_nside(depth_map))
        order_in = "NEST" if str(args.depth_map_ordering).lower() == "nest" else "RING"
        if nside_in != int(args.nside):
            depth_map = hp.ud_grade(depth_map, nside_out=int(args.nside), order_in=order_in, order_out="RING", power=0)
            order_in = "RING"

        depth_map = np.asarray(depth_map, dtype=float)
        unseen = ~np.isfinite(depth_map) | (depth_map == hp.UNSEEN)
        ok = seen & (~unseen)
        if not np.any(ok):
            raise SystemExit("Depth map has no finite values on seen pixels.")
        fill = float(np.median(depth_map[ok]))
        depth_map[unseen] = fill
        depth_map_meta = {
            "path": str(depth_map_path),
            "ordering_in": str(args.depth_map_ordering),
            "nside_in": nside_in,
            "nside_used": int(args.nside),
            "fill_value": float(fill),
            "missing_frac_seen": float(np.mean(seen & unseen)),
            "interpretation": (
                "delta_m_mag (effective faint-limit shift) if depth_mode=delta_m_map_offset_alpha_edge, "
                "else a generic depth/offset map"
            ),
            "name": None if args.depth_map_name is None else str(args.depth_map_name),
        }

    extra_offset_map = None
    extra_offset_map_meta = None
    if args.extra_offset_map_fits is not None:
        extra_path = Path(str(args.extra_offset_map_fits))
        if not extra_path.exists():
            raise SystemExit(f"Missing --extra-offset-map-fits: {extra_path}")
        extra_offset_map = hp.read_map(str(extra_path), verbose=False)
        nside_in = int(hp.get_nside(extra_offset_map))
        order_in = "NEST" if str(args.extra_offset_map_ordering).lower() == "nest" else "RING"
        if nside_in != int(args.nside):
            extra_offset_map = hp.ud_grade(
                extra_offset_map,
                nside_out=int(args.nside),
                order_in=order_in,
                order_out="RING",
                power=0,
            )
            order_in = "RING"
        extra_offset_map = np.asarray(extra_offset_map, dtype=float)
        unseen = ~np.isfinite(extra_offset_map) | (extra_offset_map == hp.UNSEEN)
        ok = seen & (~unseen)
        if not np.any(ok):
            raise SystemExit("Extra offset map has no finite values on seen pixels.")
        fill = float(np.median(extra_offset_map[ok]))
        extra_offset_map[unseen] = fill
        extra_offset_map_meta = {
            "path": str(extra_path),
            "ordering_in": str(args.extra_offset_map_ordering),
            "nside_in": nside_in,
            "nside_used": int(args.nside),
            "fill_value": float(fill),
            "missing_frac_seen": float(np.mean(seen & unseen)),
            "name": None if args.extra_offset_map_name is None else str(args.extra_offset_map_name),
        }

    extra_template_map = None
    extra_template_map_meta = None
    if args.extra_template_map_fits is not None:
        tmpl_path = Path(str(args.extra_template_map_fits))
        if not tmpl_path.exists():
            raise SystemExit(f"Missing --extra-template-map-fits: {tmpl_path}")
        extra_template_map = hp.read_map(str(tmpl_path), verbose=False)
        nside_in = int(hp.get_nside(extra_template_map))
        order_in = "NEST" if str(args.extra_template_map_ordering).lower() == "nest" else "RING"
        if nside_in != int(args.nside):
            extra_template_map = hp.ud_grade(
                extra_template_map,
                nside_out=int(args.nside),
                order_in=order_in,
                order_out="RING",
                power=0,
            )
            order_in = "RING"
        extra_template_map = np.asarray(extra_template_map, dtype=float)
        unseen = ~np.isfinite(extra_template_map) | (extra_template_map == hp.UNSEEN)
        ok = seen & (~unseen)
        if not np.any(ok):
            raise SystemExit("Extra template map has no finite values on seen pixels.")
        fill = float(np.median(extra_template_map[ok]))
        extra_template_map[unseen] = fill
        extra_template_map_meta = {
            "path": str(tmpl_path),
            "ordering_in": str(args.extra_template_map_ordering),
            "nside_in": nside_in,
            "nside_used": int(args.nside),
            "fill_value": float(fill),
            "missing_frac_seen": float(np.mean(seen & unseen)),
            "name": None if args.extra_template_map_name is None else str(args.extra_template_map_name),
        }

    # Base per-pixel means for EBV and W1COV.
    cnt_base = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), minlength=npix).astype(float)
    sum_w1cov = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), weights=np.asarray(w1cov_m_base, dtype=float), minlength=npix).astype(float)
    w1cov_mean = np.divide(sum_w1cov, cnt_base, out=np.zeros_like(sum_w1cov), where=cnt_base != 0.0)
    if ebv_m_base is None:
        ebv_mean = np.zeros(npix, dtype=float)
    else:
        sum_ebv = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), weights=np.asarray(ebv_m_base, dtype=float), minlength=npix).astype(float)
        ebv_mean = np.divide(sum_ebv, cnt_base, out=np.zeros_like(sum_ebv), where=cnt_base != 0.0)

    rng = np.random.default_rng(int(args.seed))
    counts = np.zeros(npix, dtype=np.int64)
    cursor = 0

    rows: list[dict[str, Any]] = []
    prev_counts = None
    prev_w1_cut = None
    for w1_cut in cuts:
        nxt = int(np.searchsorted(w1_eff_sorted, float(w1_cut), side="left"))
        if nxt > cursor:
            delta = ipix_sorted[cursor:nxt]
            counts += np.bincount(delta, minlength=npix).astype(np.int64)
            cursor = nxt

        w1_lo = None
        w1_hi = float(w1_cut)
        N_total = int(counts.sum())
        N_seen = int(np.sum(counts[seen]))
        N_cum_total = None
        N_cum_seen = None

        if args.w1_mode == "cumulative":
            y = counts[seen].astype(float)
        else:
            # Differential bin: counts between previous cut and this cut.
            if prev_counts is None:
                prev_counts = counts.copy()
                prev_w1_cut = float(w1_cut)
                continue
            if prev_w1_cut is None:
                raise RuntimeError("internal error: prev_w1_cut is None in differential mode")
            w1_lo = float(prev_w1_cut)
            w1_hi = float(w1_cut)
            counts_bin = counts - prev_counts
            y = counts_bin[seen].astype(float)
            N_total = int(counts_bin.sum())
            N_seen = int(np.sum(counts_bin[seen]))
            N_cum_total = int(counts.sum())
            N_cum_seen = int(np.sum(counts[seen]))
            prev_counts = counts.copy()
            prev_w1_cut = float(w1_cut)

        n_seen = pix_unit[seen]

        # Templates.
        templates: list[np.ndarray] = []
        template_names: list[str] = []
        if args.eclip_template == "abs_elat":
            templates.append(zscore(abs_elat, seen)[seen])
            template_names.append("abs_elat_z")
        elif args.eclip_template == "abs_sin_elat":
            templates.append(zscore(abs_sin_elat, seen)[seen])
            template_names.append("abs_sin_elat_z")
        elif args.eclip_template == "abs_elat_sincos_elon":
            templates.append(zscore(abs_elat, seen)[seen])
            template_names.append("abs_elat_z")
            templates.append(zscore(sin_elon, seen)[seen])
            template_names.append("sin_elon_z")
            templates.append(zscore(cos_elon, seen)[seen])
            template_names.append("cos_elon_z")

        if args.dust_template == "ebv_mean":
            templates.append(zscore(ebv_mean, seen)[seen])
            template_names.append("ebv_mean_z")

        if harmonic_templates_seen:
            templates.extend(harmonic_templates_seen)
            template_names.extend(harmonic_template_names)

        if extra_template_map is not None:
            templates.append(zscore(extra_template_map, seen)[seen])
            template_names.append("extra_template_map_z")

        offset = None
        offset_name = None
        if args.depth_mode == "w1cov_covariate":
            templates.append(zscore(np.log(np.clip(w1cov_mean, 1.0, np.inf)), seen)[seen])
            template_names.append("log_w1cov_mean_z")
        elif args.depth_mode == "w1cov_offset":
            # An offset has an implicit fixed coefficient of +1, so *do not z-score* it.
            # Instead normalize by a typical value so the offset is O(1) and centered.
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
        elif args.depth_mode == "delta_m_map_offset_alpha_edge":
            assert depth_map is not None
            # Interpret depth_map as δm(pix) [mag] from an externally calibrated completeness model.
            # The induced log-count modulation at W1_max is approximately:
            #   log N(pix) -> log N(pix) + alpha_edge(W1_max) * δm(pix),
            # where alpha_edge = d ln N(<m)/dm evaluated at the cut (global, not spatial).
            a_edge = alpha_edge_from_cumcounts(w1_eff, float(w1_hi), delta=float(w1_step))
            ref = float(np.median(depth_map[seen]))
            offset = (depth_map - ref)[seen] * float(a_edge)
            offset_name = "delta_m_offset_alpha_edge"
        elif args.depth_mode == "delta_m_map_covariate_alpha_edge":
            assert depth_map is not None
            a_edge = alpha_edge_from_cumcounts(w1_eff, float(w1_hi), delta=float(w1_step))
            ref = float(np.median(depth_map[seen]))
            t = (depth_map - ref) * float(a_edge)
            templates.append(zscore(t, seen)[seen])
            template_names.append("delta_m_alpha_edge_z")
        elif args.depth_mode == "external_logreg_integrated_offset":
            assert ext_logreg is not None

            # Approximate cumulative completeness in each pixel by integrating a logistic selection curve p(m|pix)
            # against an exponential number-count slope exp(alpha_edge * m).
            a_edge = float(alpha_edge_from_cumcounts(w1_eff, float(w1_hi), delta=float(w1_step)))
            dm = float(ext_logreg["dm"])
            mrange = float(ext_logreg["mrange"])

            if args.w1_mode == "cumulative":
                m_lo = float(w1_hi) - mrange
            else:
                if w1_lo is None:
                    raise RuntimeError("internal error: w1_lo is None in differential mode")
                m_lo = float(w1_lo)

            m_hi = float(w1_hi)
            if not (np.isfinite(m_lo) and np.isfinite(m_hi) and m_hi > m_lo):
                raise RuntimeError(f"invalid completeness integration bounds: [{m_lo}, {m_hi}]")

            n_steps = int(max(3.0, math.ceil((m_hi - m_lo) / dm) + 1.0))
            mgrid = np.linspace(m_lo, m_hi, n_steps, dtype=float)

            # Normalize weights to avoid huge exp ranges; normalization cancels in the ratio.
            w = np.exp(a_edge * (mgrid - m_hi)).astype(float)
            w_den = float(np.trapezoid(w, mgrid))
            if not np.isfinite(w_den) or w_den <= 0.0:
                raise RuntimeError("invalid completeness weight normalization")

            m50_seen = np.asarray(ext_logreg["m50"], dtype=float)[seen]
            denom_seen = np.asarray(ext_logreg["denom"], dtype=float)[seen]

            n_pix = int(m50_seen.size)
            comp = np.empty(n_pix, dtype=float)
            chunk = 6000
            mcol = mgrid[:, None]
            wcol = w[:, None]
            for start in range(0, n_pix, chunk):
                end = min(n_pix, start + chunk)
                m50_c = m50_seen[start:end][None, :]
                den_c = denom_seen[start:end][None, :]
                arg = den_c * (mcol - m50_c)
                arg = np.clip(arg, -50.0, 50.0)
                p = 1.0 / (1.0 + np.exp(-arg))
                num = np.trapezoid(wcol * p, mgrid, axis=0)
                comp[start:end] = num / w_den

            comp = np.clip(comp, 1e-6, 1.0)
            logc = np.log(comp)
            logc = logc - float(np.median(logc))
            offset = logc
            offset_name = "external_logreg_integrated_offset"

        if extra_offset_map is not None:
            ref = float(np.median(extra_offset_map[seen]))
            extra = (extra_offset_map - ref)[seen]
            offset = extra if offset is None else (offset + extra)
            if offset_name is None:
                offset_name = "extra_offset_map"
            else:
                offset_name = f"{offset_name}+extra_offset_map"

        cols = [np.ones_like(y), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2]]
        cols.extend(templates)
        X = np.column_stack(cols)

        beta, cov = fit_poisson_glm(X, y, offset=offset, max_iter=int(args.max_iter))
        bvec = np.asarray(beta[1:4], dtype=float)
        D_hat = float(np.linalg.norm(bvec))
        l_hat, b_hat = vec_to_lb(bvec)

        # Diagnostics + (quasi-)Poisson covariance scaling.
        eta_hat = (np.zeros_like(y) if offset is None else offset) + X @ beta
        eta_hat = np.clip(eta_hat, -25.0, 25.0)
        mu_hat = np.exp(eta_hat)
        diag = poisson_glm_diagnostics(y, mu_hat, n_params=int(X.shape[1]))
        phi_eff = float(diag["dispersion_phi_eff"])

        # Approx marginalization by sampling beta ~ N(beta_hat, cov_hat),
        # plus an overdispersion-scaled ("quasi-Poisson") version.
        draws = None
        draws_q = None
        if cov is not None and np.all(np.isfinite(cov)):
            draws = rng.multivariate_normal(beta, cov, size=int(args.mc_draws))
            if np.isfinite(phi_eff) and phi_eff != 1.0:
                draws_q = rng.multivariate_normal(beta, cov * phi_eff, size=int(args.mc_draws))
            else:
                draws_q = draws

        if draws is None:
            D16 = D50 = D84 = float("nan")
            l16 = l50 = l84 = float("nan")
            b16 = b50 = b84 = float("nan")
        else:
            b_draw = draws[:, 1:4]
            D_draw = np.linalg.norm(b_draw, axis=1)
            # Direction from the b-vector; undefined when b~0, but that's not our regime.
            lb = np.array([vec_to_lb(v) for v in b_draw])
            l_draw = lb[:, 0]
            b_draw_lat = lb[:, 1]

            def pct(a: np.ndarray, q: float) -> float:
                return float(np.nanpercentile(a, q))

            D16, D50, D84 = pct(D_draw, 16), pct(D_draw, 50), pct(D_draw, 84)
            l16, l50, l84 = pct(l_draw, 16), pct(l_draw, 50), pct(l_draw, 84)
            b16, b50, b84 = pct(b_draw_lat, 16), pct(b_draw_lat, 50), pct(b_draw_lat, 84)

        # Quasi-Poisson percentiles (cov scaled by phi).
        if draws_q is None:
            D16q = D50q = D84q = float("nan")
            l16q = l50q = l84q = float("nan")
            b16q = b50q = b84q = float("nan")
        else:
            bq = draws_q[:, 1:4]
            Dq = np.linalg.norm(bq, axis=1)
            lbq = np.array([vec_to_lb(v) for v in bq])
            lq = lbq[:, 0]
            bq_lat = lbq[:, 1]

            def pct(a: np.ndarray, q: float) -> float:
                return float(np.nanpercentile(a, q))

            D16q, D50q, D84q = pct(Dq, 16), pct(Dq, 50), pct(Dq, 84)
            l16q, l50q, l84q = pct(lq, 16), pct(lq, 50), pct(lq, 84)
            b16q, b50q, b84q = pct(bq_lat, 16), pct(bq_lat, 50), pct(bq_lat, 84)

        # Dipole components of the nuisance templates themselves (their l=1 vectors).
        template_dipoles: list[dict[str, Any]] = []
        for name, t in zip(template_names, templates, strict=True):
            tvec = fit_template_dipole_vec(n_seen, t)
            tl, tb = vec_to_lb(tvec)
            template_dipoles.append(
                {
                    "name": str(name),
                    "vec": [float(x) for x in tvec],
                    "l_deg": float(tl),
                    "b_deg": float(tb),
                    "amp": float(np.linalg.norm(tvec)),
                    "angle_to_fit_deg": float(axis_angle_deg(tvec, bvec)),
                }
            )
        if offset is not None and offset_name is not None:
            tvec = fit_template_dipole_vec(n_seen, offset)
            tl, tb = vec_to_lb(tvec)
            template_dipoles.append(
                {
                    "name": str(offset_name),
                    "vec": [float(x) for x in tvec],
                    "l_deg": float(tl),
                    "b_deg": float(tb),
                    "amp": float(np.linalg.norm(tvec)),
                    "angle_to_fit_deg": float(axis_angle_deg(tvec, bvec)),
                }
            )

        # Dipole–template degeneracy summary from Fisher covariance.
        corr_bt = None
        if cov is not None and np.all(np.isfinite(cov)) and X.shape[1] > 4:
            corr = cov_to_corr(cov)
            corr_bt = corr[1:4, 4:].tolist()

        # Optional sky jackknife.
        jk = None
        stride = int(max(1, int(args.jackknife_stride)))
        if jk_region_seen is not None and jk_regions is not None and (len(rows) % stride == 0):
            jk = jackknife_bvec_cov(
                X=X,
                y=y,
                offset=offset,
                beta_full=beta,
                jk_region=jk_region_seen,
                jk_regions=jk_regions,
                max_iter=int(args.jackknife_max_iter),
            )

        rows.append(
            {
                "w1_cut": w1_hi,
                "w1_lo": w1_lo,
                "w1_hi": w1_hi,
                "alpha_edge": float(alpha_edge_from_cumcounts(w1_eff, float(w1_hi), delta=float(w1_step))),
                "N_total": int(N_total),
                "N_seen": int(N_seen),
                "N_cum_total": N_cum_total,
                "N_cum_seen": N_cum_seen,
                "w1_mode": str(args.w1_mode),
                "dipole": {
                    "D_hat": D_hat,
                    "l_hat_deg": l_hat,
                    "b_hat_deg": b_hat,
                    "D_p16": D16,
                    "D_p50": D50,
                    "D_p84": D84,
                    "l_p16": l16,
                    "l_p50": l50,
                    "l_p84": l84,
                    "b_p16": b16,
                    "b_p50": b50,
                    "b_p84": b84,
                },
                "dipole_quasi": {
                    "dispersion_phi": float(diag["dispersion_phi"]),
                    "dispersion_phi_eff": float(diag["dispersion_phi_eff"]),
                    "D_p16": D16q,
                    "D_p50": D50q,
                    "D_p84": D84q,
                    "l_p16": l16q,
                    "l_p50": l50q,
                    "l_p84": l84q,
                    "b_p16": b16q,
                    "b_p50": b50q,
                    "b_p84": b84q,
                },
                "beta_hat": [float(x) for x in beta],
                "fit_diag": diag,
                "corr_b_templates": corr_bt,
                "corr_b_templates_names": template_names,
                "template_names": template_names,
                "offset_name": offset_name,
                "template_dipoles": template_dipoles,
                "jackknife": jk,
            }
        )

    payload = {
        "meta": {
            "catalog": str(args.catalog),
            "mask_catalog": None if args.mask_catalog is None else str(args.mask_catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": int(args.nside),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_min": None if args.w1_min is None else float(args.w1_min),
            "w1_grid": args.w1_grid,
            "eclip_template": args.eclip_template,
            "dust_template": args.dust_template,
            "depth_mode": args.depth_mode,
            "harmonic_lmax": int(args.harmonic_lmax),
            "unwise_tiles_fits": None if args.unwise_tiles_fits is None else str(args.unwise_tiles_fits),
            "nexp_tile_stats_json": None if args.nexp_tile_stats_json is None else str(args.nexp_tile_stats_json),
            "nexp_missing_frac_seen": nexp_missing_frac,
            "depth_map": depth_map_meta,
            "extra_offset_map": extra_offset_map_meta,
            "extra_template_map": extra_template_map_meta,
            "external_logreg_meta": None if ext_logreg is None else str(ext_logreg["meta_path"]),
            "external_logreg_mrange": None if ext_logreg is None else float(ext_logreg["mrange"]),
            "external_logreg_dm": None if ext_logreg is None else float(ext_logreg["dm"]),
            "mc_draws": int(args.mc_draws),
            "seed": int(args.seed),
            "w1_mode": str(args.w1_mode),
            "jackknife_nside": int(args.jackknife_nside),
            "jackknife_stride": int(args.jackknife_stride),
            "jackknife_max_iter": int(args.jackknife_max_iter),
            "jackknife_max_regions": None if args.jackknife_max_regions is None else int(args.jackknife_max_regions),
            "inject_delta_m_mag": float(args.inject_delta_m_mag),
            "inject_axis_lb": [axis_l, axis_b],
            "footprint_mask": "Secrest-style mask_zeros (data-derived) + exclusion discs + |b| cut (pixel-center).",
        },
        "rows": rows,
    }
    (outdir / "rvmp_fig5_poisson_glm.json").write_text(json.dumps(payload, indent=2))

    if args.make_plot:
        import matplotlib.pyplot as plt

        w1c = np.array([r["w1_cut"] for r in rows], dtype=float)
        D50 = np.array([r["dipole"]["D_p50"] for r in rows], dtype=float)
        D16 = np.array([r["dipole"]["D_p16"] for r in rows], dtype=float)
        D84 = np.array([r["dipole"]["D_p84"] for r in rows], dtype=float)
        l50 = np.array([r["dipole"]["l_p50"] for r in rows], dtype=float)
        l16 = np.array([r["dipole"]["l_p16"] for r in rows], dtype=float)
        l84 = np.array([r["dipole"]["l_p84"] for r in rows], dtype=float)
        b50 = np.array([r["dipole"]["b_p50"] for r in rows], dtype=float)
        b16 = np.array([r["dipole"]["b_p16"] for r in rows], dtype=float)
        b84 = np.array([r["dipole"]["b_p84"] for r in rows], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)
        ax = axes[0]
        ax.fill_between(w1c, D16, D84, alpha=0.25, color="C0", lw=0)
        ax.plot(w1c, D50, color="C0")
        ax.set_ylabel("D ≈ |b| (Poisson GLM)")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.fill_between(w1c, l16, l84, alpha=0.25, color="C1", lw=0)
        ax.plot(w1c, l50, color="C1")
        ax.axhline(264.021, color="k", ls="--", lw=1)
        ax.set_ylabel("l [deg]")
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.fill_between(w1c, b16, b84, alpha=0.25, color="C2", lw=0)
        ax.plot(w1c, b50, color="C2")
        ax.axhline(48.253, color="k", ls="--", lw=1)
        ax.set_ylabel("b [deg]")
        ax.set_xlabel("W1 cut (upper edge; effective if injected)")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(outdir / "rvmp_fig5_poisson_glm.png", dpi=200)
        plt.close(fig)

    D_last = rows[-1]["dipole"]["D_p50"]
    l_last = rows[-1]["dipole"]["l_p50"]
    b_last = rows[-1]["dipole"]["b_p50"]
    print(f"w1_cut_max={cuts[-1]:.3f}: D~{D_last:.4g}, (l,b)~({l_last:.1f},{b_last:+.1f})")
    print(f"Wrote: {outdir}/rvmp_fig5_poisson_glm.json")
    if args.make_plot:
        print(f"Wrote: {outdir}/rvmp_fig5_poisson_glm.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
