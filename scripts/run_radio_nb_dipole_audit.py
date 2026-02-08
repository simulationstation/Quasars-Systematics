#!/usr/bin/env python3
"""Negative-binomial radio dipole audit (paper-faithful to arXiv:2509.16732).

Implements the counts-in-cells negative binomial dipole estimator described in:
  Boehme et al. (2025), arXiv:2509.16732

Core details replicated:
  - HEALPix Nside=32 counts-in-cells (cells ~ 3.36 deg^2).
  - Survey footprints are defined geometrically (include zero-count cells).
  - Negative binomial p fixed from empirical mean/variance:
      p = 1 - mu/var, clipped into (0,1).
  - Dipole modulation via r_i = r0 * (1 + d cos(theta_i)).
  - Fit direction (RA,Dec) and amplitude d; r0 is nuisance (per-survey in joint fits).
  - Optional emcee MCMC (MAP-only also supported).

This script is deliberately self-contained and emits a Markdown report + JSON summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import healpy as hp
import numpy as np
from astropy.io import fits
from astropy_healpix import uniq_to_level_ipix
from scipy.optimize import minimize
from scipy.special import gammaln


R_EQ_TO_GAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ],
    dtype=float,
)


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def radec_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    c = np.cos(dec)
    return np.column_stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)])


def unitvec_to_radec_deg(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        return float("nan"), float("nan")
    u = v / n
    ra = float(np.degrees(np.arctan2(u[1], u[0])) % 360.0)
    dec = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
    return ra, dec


def ang_sep_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=float).reshape(3)
    b = np.asarray(v2, dtype=float).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def gal_lat_from_unitvec(eq_v: np.ndarray) -> np.ndarray:
    gv = np.asarray(eq_v, dtype=float) @ R_EQ_TO_GAL.T
    return np.degrees(np.arcsin(np.clip(gv[:, 2], -1.0, 1.0)))


def eq_vec_to_gal_vec(eq_v: np.ndarray) -> np.ndarray:
    return np.asarray(eq_v, dtype=float) @ R_EQ_TO_GAL.T


@dataclass(frozen=True)
class PixelGeom:
    nside: int
    nest: bool
    nvec_eq: np.ndarray  # (npix,3)
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    gal_b_deg: np.ndarray


def build_pixel_geom(nside: int, *, nest: bool = True) -> PixelGeom:
    npix = hp.nside2npix(int(nside))
    ipix = np.arange(npix, dtype=np.int64)
    vx, vy, vz = hp.pix2vec(int(nside), ipix, nest=bool(nest))
    nvec = np.column_stack([vx, vy, vz]).astype(float)
    ra = (np.degrees(np.arctan2(nvec[:, 1], nvec[:, 0])) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(np.clip(nvec[:, 2], -1.0, 1.0)))
    b = gal_lat_from_unitvec(nvec)
    return PixelGeom(
        nside=int(nside), nest=bool(nest), nvec_eq=nvec, ra_deg=ra, dec_deg=dec, gal_b_deg=b
    )


def infer_p_from_counts(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    mu = float(np.mean(y))
    var = float(np.var(y))
    if not np.isfinite(mu) or not np.isfinite(var) or var <= 0.0:
        p = 1e-6
    else:
        p = 1.0 - (mu / var)
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    r_mom = float((mu * mu) / (var - mu)) if var > mu else float(1e6)
    return {"mu": mu, "var": var, "p": p, "r_mom": r_mom}


@dataclass(frozen=True)
class SurveyCells:
    name: str
    freq_mhz: float
    cut_mjy: float
    n_src: int
    n_cell: int
    y: np.ndarray  # counts on valid cells (includes zeros)
    nvec_eq: np.ndarray  # (n_cell,3)
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    templates: np.ndarray  # (n_cell, n_t) z-scored template matrix (may include physics-motivated maps)
    template_names: tuple[str, ...]
    p: float
    mu: float
    var: float
    r_mom: float
    const_term: np.ndarray  # y*log(p) - logGamma(y+1)


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    s = float(np.std(x))
    if (not np.isfinite(s)) or s == 0.0:
        s = 1.0
    return (x - m) / s


def build_decra_templates(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    r = np.deg2rad(ra)
    cols = [
        _zscore_1d(dec),
        _zscore_1d(dec * dec),
        _zscore_1d(np.sin(r)),
        _zscore_1d(np.cos(r)),
    ]
    return np.column_stack(cols).astype(float)


def safe_zscore_on_valid(x: np.ndarray) -> np.ndarray:
    """Z-score a 1D vector, falling back to zeros if it is constant/invalid."""
    return _zscore_1d(np.asarray(x, dtype=float))


def _dip_hat_from_ra_sindec(ra_deg: float, sin_dec: float) -> np.ndarray:
    ra = math.radians(float(ra_deg))
    sd = float(sin_dec)
    if (not np.isfinite(sd)) or abs(sd) > 1.0:
        return np.array([float("nan"), float("nan"), float("nan")], dtype=float)
    cd = math.sqrt(max(0.0, 1.0 - sd * sd))
    return np.array([cd * math.cos(ra), cd * math.sin(ra), sd], dtype=float)


def nb_loglike_dipole(
    s: SurveyCells,
    *,
    d: float,
    dip_hat: np.ndarray,
    r0: float,
    eta_extra: np.ndarray | None = None,
    modulation: Literal["linear", "exp"] = "linear",
) -> float:
    if (not np.isfinite(d)) or (not np.isfinite(r0)) or d < 0.0 or r0 <= 0.0:
        return -np.inf
    p = float(s.p)
    log1mp = float(math.log1p(-p))
    cos_th = np.asarray(s.nvec_eq @ dip_hat, dtype=float)
    if eta_extra is None:
        eta_extra = 0.0
    if modulation == "linear":
        ri = r0 * (1.0 + float(d) * cos_th + eta_extra)
    elif modulation == "exp":
        eta = np.clip(float(d) * cos_th + eta_extra, -20.0, 20.0)
        ri = r0 * np.exp(eta)
    else:
        raise ValueError(f"unknown modulation={modulation!r}")
    if np.any(ri <= 0.0) or np.any(~np.isfinite(ri)):
        return -np.inf
    y = s.y.astype(float)
    ll = np.sum(gammaln(y + ri) - gammaln(ri) + ri * log1mp + s.const_term)
    return float(ll)


def joint_loglike(
    surveys: list[SurveyCells],
    *,
    d: float,
    dip_hat: np.ndarray,
    r0s: list[float],
    eta_extras: list[np.ndarray | None] | None = None,
    modulation: Literal["linear", "exp"] = "linear",
) -> float:
    if len(surveys) != len(r0s):
        raise ValueError("r0s length must match surveys")
    if eta_extras is None:
        eta_extras = [None] * len(surveys)
    if len(eta_extras) != len(surveys):
        raise ValueError("eta_extras length must match surveys")
    ll = 0.0
    for s, r0, eta_extra in zip(surveys, r0s, eta_extras):
        ll_i = nb_loglike_dipole(
            s,
            d=d,
            dip_hat=dip_hat,
            r0=float(r0),
            eta_extra=eta_extra,
            modulation=modulation,
        )
        if not np.isfinite(ll_i):
            return -np.inf
        ll += ll_i
    return float(ll)


def counts_from_fits(
    path: Path,
    *,
    ra_col: str,
    dec_col: str,
    flux_col: str,
    cut_mjy: float,
    geom: PixelGeom,
    valid_pix: np.ndarray,
    source_extra_mask_fn: Any | None,
    gal_b_cut_deg: float,
    chunk_rows: int,
    mean_cols: list[str] | None = None,
) -> tuple[np.ndarray, int, dict[str, np.ndarray]]:
    npix = geom.nvec_eq.shape[0]
    counts = np.zeros(npix, dtype=np.int64)
    n_sel = 0
    mean_cols = list(mean_cols or [])
    sum_maps = {c: np.zeros(npix, dtype=float) for c in mean_cols}
    cnt_maps = {c: np.zeros(npix, dtype=np.int64) for c in mean_cols}
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        n = len(data)
        for i0 in range(0, n, int(chunk_rows)):
            d = data[i0 : i0 + int(chunk_rows)]
            ra = np.asarray(d[ra_col], dtype=float)
            dec = np.asarray(d[dec_col], dtype=float)
            flux = np.asarray(d[flux_col], dtype=float)
            m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(flux) & (flux >= float(cut_mjy))
            if source_extra_mask_fn is not None:
                m &= np.asarray(source_extra_mask_fn(ra, dec), dtype=bool)
            if not np.any(m):
                continue
            ra = ra[m]
            dec = dec[m]
            v = radec_to_unitvec(ra, dec)
            glat = gal_lat_from_unitvec(v)
            m2 = np.abs(glat) >= float(gal_b_cut_deg)
            if not np.any(m2):
                continue
            v = v[m2]
            ip = hp.vec2pix(geom.nside, v[:, 0], v[:, 1], v[:, 2], nest=geom.nest).astype(np.int64, copy=False)
            ok = valid_pix[ip]
            if not np.any(ok):
                continue
            ip = ip[ok]
            n_sel += int(ip.size)
            counts += np.bincount(ip, minlength=npix).astype(np.int64, copy=False)
            if mean_cols:
                for c in mean_cols:
                    vals = np.asarray(d[c], dtype=float)[m]
                    vals = vals[m2]
                    vals = vals[ok]
                    good = np.isfinite(vals)
                    if not np.any(good):
                        continue
                    ipg = ip[good]
                    sum_maps[c] += np.bincount(ipg, weights=vals[good], minlength=npix).astype(float, copy=False)
                    cnt_maps[c] += np.bincount(ipg, minlength=npix).astype(np.int64, copy=False)

    means: dict[str, np.ndarray] = {}
    for c in mean_cols:
        num = sum_maps[c]
        den = cnt_maps[c].astype(float)
        mval = np.full(npix, np.nan, dtype=float)
        sel = den > 0
        mval[sel] = num[sel] / den[sel]
        fill = float(np.nanmedian(mval[sel])) if np.any(sel) else 0.0
        mval[~np.isfinite(mval)] = fill
        means[c] = mval
    return counts, int(n_sel), means


def counts_from_racs_csv(
    path: Path,
    *,
    cut_mjy: float,
    geom: PixelGeom,
    valid_pix: np.ndarray,
    dec_min: float,
    dec_max: float,
    gal_b_cut_deg: float,
) -> tuple[np.ndarray, int]:
    import pandas as pd

    npix = geom.nvec_eq.shape[0]
    counts = np.zeros(npix, dtype=np.int64)
    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=["ra", "dec", "total_flux_source", "gal_lat"],
        dtype={"ra": "f8", "dec": "f8", "total_flux_source": "f8", "gal_lat": "f8"},
    )
    ra = df["ra"].to_numpy()
    dec = df["dec"].to_numpy()
    flux = df["total_flux_source"].to_numpy()
    glat = df["gal_lat"].to_numpy()
    m = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(flux)
        & (flux >= float(cut_mjy))
        & (dec >= float(dec_min))
        & (dec <= float(dec_max))
        & (np.abs(glat) >= float(gal_b_cut_deg))
    )
    ra = ra[m]
    dec = dec[m]
    v = radec_to_unitvec(ra, dec)
    ip = hp.vec2pix(geom.nside, v[:, 0], v[:, 1], v[:, 2], nest=geom.nest).astype(np.int64, copy=False)
    ok = valid_pix[ip]
    ip = ip[ok]
    counts += np.bincount(ip, minlength=npix).astype(np.int64, copy=False)
    return counts, int(ip.size)


def lotss_valid_pix_from_moc(
    moc_path: Path,
    *,
    geom: PixelGeom,
    gal_b_cut_deg: float,
    mode: Literal["any", "full", "center"],
    lmax: int,
) -> np.ndarray:
    """Build a target-order valid-pixel mask from a MOC footprint.

    Modes:
      - any: include a pixel if any subpixel intersects the MOC
      - full: include a pixel if fully covered by the MOC
      - center: include a pixel if its center point is inside the MOC
    """
    npix_tar = geom.nvec_eq.shape[0]
    with fits.open(moc_path, memmap=True) as hdul:
        uniq = np.asarray(hdul[1].data["UNIQ"], dtype=np.int64)
    levels, ipix = uniq_to_level_ipix(uniq)
    levels = levels.astype(int)
    ipix = ipix.astype(np.int64)

    if mode == "center":
        level_to_set: dict[int, set[int]] = {}
        for L, p in zip(levels, ipix):
            level_to_set.setdefault(int(L), set()).add(int(p))
        covered = np.zeros(npix_tar, dtype=bool)
        vx, vy, vz = geom.nvec_eq[:, 0], geom.nvec_eq[:, 1], geom.nvec_eq[:, 2]
        for L, s in level_to_set.items():
            ip = hp.vec2pix(2**int(L), vx, vy, vz, nest=geom.nest).astype(np.int64, copy=False)
            covered |= np.isin(ip, list(s))
    else:
        # Expand to a common max level (nested indices are hierarchical/contiguous).
        if int(lmax) < int(round(math.log2(geom.nside))):
            raise ValueError("lmax must be >= log2(nside)")
        npix_max = 12 * (2**int(lmax)) ** 2
        cov_max = np.zeros(npix_max, dtype=bool)
        for L, p in zip(levels, ipix):
            L = int(L)
            p = int(p)
            if L == int(lmax):
                cov_max[p] = True
            elif L < int(lmax):
                f = 4 ** (int(lmax) - L)
                start = p * f
                cov_max[start : start + f] = True
            else:
                f = 4 ** (L - int(lmax))
                cov_max[p // f] = True
        block = 4 ** (int(lmax) - int(round(math.log2(geom.nside))))
        cov = cov_max.reshape(npix_tar, block)
        covered = np.any(cov, axis=1) if mode == "any" else np.all(cov, axis=1)

    covered &= np.abs(geom.gal_b_deg) >= float(gal_b_cut_deg)
    return covered


def moc_coverage_fraction(
    moc_path: Path,
    *,
    geom: PixelGeom,
    gal_b_cut_deg: float,
    lmax: int,
) -> np.ndarray:
    """Return the fractional MOC coverage of each target pixel (nested geometry assumed)."""
    if not geom.nest:
        raise ValueError("moc_coverage_fraction assumes nested target geometry (nest=True)")
    if int(lmax) < int(round(math.log2(geom.nside))):
        raise ValueError("lmax must be >= log2(nside)")

    npix_tar = geom.nvec_eq.shape[0]
    with fits.open(moc_path, memmap=True) as hdul:
        uniq = np.asarray(hdul[1].data["UNIQ"], dtype=np.int64)
    levels, ipix = uniq_to_level_ipix(uniq)
    levels = levels.astype(int)
    ipix = ipix.astype(np.int64)

    npix_max = 12 * (2**int(lmax)) ** 2
    cov_max = np.zeros(npix_max, dtype=bool)
    for L, p in zip(levels, ipix):
        L = int(L)
        p = int(p)
        if L == int(lmax):
            cov_max[p] = True
        elif L < int(lmax):
            f = 4 ** (int(lmax) - L)
            start = p * f
            cov_max[start : start + f] = True
        else:
            f = 4 ** (L - int(lmax))
            cov_max[p // f] = True

    block = 4 ** (int(lmax) - int(round(math.log2(geom.nside))))
    cov = cov_max.reshape(npix_tar, block)
    frac = np.mean(cov, axis=1).astype(float, copy=False)
    frac[np.abs(geom.gal_b_deg) < float(gal_b_cut_deg)] = 0.0
    return frac


def sample_haslam_k408(geom: PixelGeom, *, haslam_fits: Path) -> np.ndarray:
    """Sample the Haslam 408 MHz map at the target pixel centers (galactic, RING, nside=512)."""
    mp = hp.read_map(str(haslam_fits), nest=False, dtype=float, verbose=False)
    nside = hp.get_nside(mp)
    gv = eq_vec_to_gal_vec(geom.nvec_eq)
    ip = hp.vec2pix(int(nside), gv[:, 0], gv[:, 1], gv[:, 2], nest=False).astype(np.int64, copy=False)
    out = np.asarray(mp[ip], dtype=float)
    out[~np.isfinite(out)] = float(np.nanmedian(out[np.isfinite(out)])) if np.any(np.isfinite(out)) else 0.0
    return out


def build_survey_cells(
    name: str,
    freq_mhz: float,
    *,
    geom: PixelGeom,
    valid_pix: np.ndarray,
    counts: np.ndarray,
    cut_mjy: float,
    n_src: int,
    extra_templates: list[tuple[str, np.ndarray]] | None = None,
) -> SurveyCells:
    valid_idx = np.where(np.asarray(valid_pix, dtype=bool))[0]
    y = np.asarray(counts[valid_idx], dtype=np.int64)
    nvec = np.asarray(geom.nvec_eq[valid_idx], dtype=float)
    ra = (np.degrees(np.arctan2(nvec[:, 1], nvec[:, 0])) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(np.clip(nvec[:, 2], -1.0, 1.0)))
    cols = [build_decra_templates(ra, dec)]
    names: list[str] = ["dec_z", "dec2_z", "sin_ra_z", "cos_ra_z"]
    if extra_templates:
        for nm, raw in extra_templates:
            cols.append(safe_zscore_on_valid(raw))
            names.append(str(nm))
    templates = np.column_stack(cols).astype(float) if cols else np.zeros((y.size, 0), dtype=float)
    stats = infer_p_from_counts(y)
    p = float(stats["p"])
    const = y.astype(float) * math.log(p) - gammaln(y.astype(float) + 1.0)
    return SurveyCells(
        name=name,
        freq_mhz=float(freq_mhz),
        cut_mjy=float(cut_mjy),
        n_src=int(n_src),
        n_cell=int(y.size),
        y=y,
        nvec_eq=nvec,
        ra_deg=np.asarray(ra, dtype=float),
        dec_deg=np.asarray(dec, dtype=float),
        templates=np.asarray(templates, dtype=float),
        template_names=tuple(names),
        p=p,
        mu=float(stats["mu"]),
        var=float(stats["var"]),
        r_mom=float(stats["r_mom"]),
        const_term=np.asarray(const, dtype=float),
    )


def rebuild_survey_cells_with_counts(s: SurveyCells, y_new: np.ndarray) -> SurveyCells:
    y = np.asarray(y_new, dtype=np.int64)
    stats = infer_p_from_counts(y)
    p = float(stats["p"])
    const = y.astype(float) * math.log(p) - gammaln(y.astype(float) + 1.0)
    return SurveyCells(
        name=s.name,
        freq_mhz=s.freq_mhz,
        cut_mjy=s.cut_mjy,
        n_src=int(np.sum(y)),
        n_cell=int(y.size),
        y=y,
        nvec_eq=s.nvec_eq,
        ra_deg=s.ra_deg,
        dec_deg=s.dec_deg,
        templates=s.templates,
        template_names=s.template_names,
        p=p,
        mu=float(stats["mu"]),
        var=float(stats["var"]),
        r_mom=float(stats["r_mom"]),
        const_term=np.asarray(const, dtype=float),
    )


def summarize_values(vals: np.ndarray) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "p16": float(np.percentile(a, 16)),
        "p50": float(np.percentile(a, 50)),
        "p84": float(np.percentile(a, 84)),
    }


def info_criteria(*, loglike: float, k_params: int, n_obs: int) -> dict[str, float]:
    n = max(1, int(n_obs))
    k = max(1, int(k_params))
    ll = float(loglike)
    return {
        "k_params": float(k),
        "n_obs": float(n),
        "aic": float(2.0 * k - 2.0 * ll),
        "bic": float(k * math.log(float(n)) - 2.0 * ll),
    }


def map_fit_single(
    s: SurveyCells,
    *,
    d_max: float,
    ra0_deg: float,
    dec0_deg: float,
    log_r_min: float,
    log_r_max: float,
    nb_model: Literal["pure", "decra_linear", "decra_exp"] = "pure",
    tmpl_coef_bound: float = 1.5,
    tmpl_ridge: float = 0.0,
    n_starts: int = 1,
    start_seed: int = 0,
) -> dict[str, Any]:
    sin0 = float(math.sin(math.radians(float(dec0_deg))))
    x0 = [0.015, float(ra0_deg), sin0, float(math.log(max(1e-6, s.r_mom)))]
    bounds: list[tuple[float, float]] = [
        (0.0, float(d_max)),
        (0.0, 360.0),
        (-1.0, 1.0),
        (float(log_r_min), float(log_r_max)),
    ]
    n_t = 0
    if nb_model != "pure":
        n_t = int(s.templates.shape[1])
        x0.extend([0.0] * n_t)
        bounds.extend([(-float(tmpl_coef_bound), float(tmpl_coef_bound)) for _ in range(n_t)])
    x0_arr = np.asarray(x0, dtype=float)

    def nlp(x: np.ndarray) -> float:
        d, ra, sd, lr = map(float, x[:4])
        dip_hat = _dip_hat_from_ra_sindec(ra, sd)
        if not np.all(np.isfinite(dip_hat)):
            return 1e100
        eta_extra = None
        modulation: Literal["linear", "exp"] = "linear"
        if nb_model != "pure":
            coeff = np.asarray(x[4 : 4 + n_t], dtype=float)
            eta_extra = s.templates @ coeff
            modulation = "exp" if nb_model == "decra_exp" else "linear"
        ll = nb_loglike_dipole(s, d=d, dip_hat=dip_hat, r0=math.exp(lr), eta_extra=eta_extra, modulation=modulation)
        if not np.isfinite(ll):
            return 1e100
        nlp_val = -float(ll)
        if nb_model != "pure" and float(tmpl_ridge) > 0.0:
            nlp_val += 0.5 * float(tmpl_ridge) * float(np.dot(coeff, coeff))
        return nlp_val

    rng = np.random.default_rng(int(start_seed))
    starts = [x0_arr]
    for _ in range(max(0, int(n_starts) - 1)):
        x = x0_arr.copy()
        x[0] = float(rng.uniform(0.0, float(d_max)))
        x[1] = float(rng.uniform(0.0, 360.0))
        x[2] = float(rng.uniform(-1.0, 1.0))
        x[3] = float(np.clip(x[3] + rng.normal(0.0, 1.0), float(log_r_min), float(log_r_max)))
        if n_t:
            x[4 : 4 + n_t] = rng.normal(0.0, 0.05, size=n_t)
        starts.append(x)

    best_res = None
    for st in starts:
        res = minimize(
            nlp, st, method="L-BFGS-B", bounds=bounds, options={"maxiter": 700, "ftol": 1e-12}
        )
        if (best_res is None) or (float(res.fun) < float(best_res.fun)):
            best_res = res
    assert best_res is not None
    res = best_res
    xfit = np.asarray(res.x, dtype=float)
    d, ra, sd, lr = map(float, xfit[:4])
    dip_hat = _dip_hat_from_ra_sindec(ra, sd)
    out: dict[str, Any] = {
        "model": nb_model,
        "success": bool(res.success),
        "message": str(res.message),
        "n_starts": int(n_starts),
        "d": float(d),
        "ra_deg": float(ra),
        "dec_deg": float(math.degrees(math.asin(max(-1.0, min(1.0, sd))))),
        "r0": float(math.exp(lr)),
        "loglike": float(-res.fun),
        "dip_hat": [float(v) for v in dip_hat.tolist()],
    }
    if nb_model != "pure":
        coeff = np.asarray(xfit[4 : 4 + n_t], dtype=float)
        out["template_names"] = list(s.template_names[:n_t])
        out["template_coef"] = [float(v) for v in coeff.tolist()]
    k_params = 4 + (n_t if nb_model != "pure" else 0)
    out["ic"] = info_criteria(loglike=out["loglike"], k_params=int(k_params), n_obs=int(s.n_cell))
    return out


def map_fit_joint(
    surveys: list[SurveyCells],
    *,
    d_max: float,
    ra0_deg: float,
    dec0_deg: float,
    log_r_min: float,
    log_r_max: float,
    nb_model: Literal["pure", "decra_linear", "decra_exp"] = "pure",
    tmpl_coef_bound: float = 1.5,
    tmpl_ridge: float = 0.0,
    n_starts: int = 1,
    start_seed: int = 0,
) -> dict[str, Any]:
    sin0 = float(math.sin(math.radians(float(dec0_deg))))
    lr0s = [float(math.log(max(1e-6, s.r_mom))) for s in surveys]
    x0 = [0.015, float(ra0_deg), sin0] + lr0s
    bounds: list[tuple[float, float]] = [(0.0, float(d_max)), (0.0, 360.0), (-1.0, 1.0)] + [
        (float(log_r_min), float(log_r_max)) for _ in surveys
    ]
    n_s = len(surveys)
    n_t = int(surveys[0].templates.shape[1]) if surveys else 0
    if nb_model != "pure":
        for _ in surveys:
            x0.extend([0.0] * n_t)
            bounds.extend([(-float(tmpl_coef_bound), float(tmpl_coef_bound)) for _ in range(n_t)])
    x0_arr = np.asarray(x0, dtype=float)

    def nlp(x: np.ndarray) -> float:
        d = float(x[0])
        ra = float(x[1])
        sd = float(x[2])
        dip_hat = _dip_hat_from_ra_sindec(ra, sd)
        if not np.all(np.isfinite(dip_hat)):
            return 1e100
        r0s = [float(math.exp(v)) for v in x[3 : 3 + n_s]]
        eta_extras: list[np.ndarray | None] | None = None
        modulation: Literal["linear", "exp"] = "linear"
        if nb_model != "pure":
            eta_extras = []
            off = 3 + n_s
            coeff_blocks = []
            for i, s in enumerate(surveys):
                coeff = np.asarray(x[off + i * n_t : off + (i + 1) * n_t], dtype=float)
                eta_extras.append(s.templates @ coeff)
                coeff_blocks.append(coeff)
            modulation = "exp" if nb_model == "decra_exp" else "linear"
        ll = joint_loglike(surveys, d=d, dip_hat=dip_hat, r0s=r0s, eta_extras=eta_extras, modulation=modulation)
        if not np.isfinite(ll):
            return 1e100
        nlp_val = -float(ll)
        if nb_model != "pure" and float(tmpl_ridge) > 0.0:
            for coeff in coeff_blocks:
                nlp_val += 0.5 * float(tmpl_ridge) * float(np.dot(coeff, coeff))
        return nlp_val

    rng = np.random.default_rng(int(start_seed))
    starts = [x0_arr]
    for _ in range(max(0, int(n_starts) - 1)):
        x = x0_arr.copy()
        x[0] = float(rng.uniform(0.0, float(d_max)))
        x[1] = float(rng.uniform(0.0, 360.0))
        x[2] = float(rng.uniform(-1.0, 1.0))
        for j in range(n_s):
            idx = 3 + j
            x[idx] = float(
                np.clip(x[idx] + rng.normal(0.0, 1.0), float(log_r_min), float(log_r_max))
            )
        if nb_model != "pure":
            off = 3 + n_s
            x[off:] = rng.normal(0.0, 0.05, size=x.size - off)
        starts.append(x)

    best_res = None
    for st in starts:
        res = minimize(
            nlp, st, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1200, "ftol": 1e-12}
        )
        if (best_res is None) or (float(res.fun) < float(best_res.fun)):
            best_res = res
    assert best_res is not None
    res = best_res
    x = np.asarray(res.x, dtype=float)
    d, ra, sd = map(float, x[:3])
    dip_hat = _dip_hat_from_ra_sindec(ra, sd)
    r0s = [float(math.exp(v)) for v in x[3 : 3 + n_s]]
    out: dict[str, Any] = {
        "model": nb_model,
        "success": bool(res.success),
        "message": str(res.message),
        "n_starts": int(n_starts),
        "d": float(d),
        "ra_deg": float(ra),
        "dec_deg": float(math.degrees(math.asin(max(-1.0, min(1.0, sd))))),
        "r0s": {surveys[i].name: float(r0s[i]) for i in range(n_s)},
        "loglike": float(-res.fun),
        "dip_hat": [float(v) for v in dip_hat.tolist()],
    }
    if nb_model != "pure":
        names = list(surveys[0].template_names[:n_t])
        off = 3 + n_s
        out["template_names"] = names
        out["template_coef"] = {}
        for i, s in enumerate(surveys):
            coeff = np.asarray(x[off + i * n_t : off + (i + 1) * n_t], dtype=float)
            out["template_coef"][s.name] = [float(v) for v in coeff.tolist()]
    k_params = 3 + n_s + (n_s * n_t if nb_model != "pure" else 0)
    n_obs = sum(int(s.n_cell) for s in surveys)
    out["ic"] = info_criteria(loglike=out["loglike"], k_params=int(k_params), n_obs=int(n_obs))
    return out


def _unwrap_ra_deg(ra: np.ndarray, ra_ref: float) -> np.ndarray:
    ra = np.asarray(ra, dtype=float)
    return ((ra - float(ra_ref) + 180.0) % 360.0) - 180.0 + float(ra_ref)


def summarize_chain(chain: np.ndarray, *, ra_ref_deg: float, survey_names: list[str]) -> dict[str, Any]:
    chain = np.asarray(chain, dtype=float)
    d = chain[:, 0]
    ra = _unwrap_ra_deg(chain[:, 1], float(ra_ref_deg))
    sd = chain[:, 2]
    dec = np.degrees(np.arcsin(np.clip(sd, -1.0, 1.0)))
    out: dict[str, Any] = {
        "d": {k: float(v) for k, v in zip(["p16", "p50", "p84"], np.percentile(d, [16, 50, 84]))},
        "ra_deg": {k: float(v % 360.0) for k, v in zip(["p16", "p50", "p84"], np.percentile(ra, [16, 50, 84]))},
        "dec_deg": {k: float(v) for k, v in zip(["p16", "p50", "p84"], np.percentile(dec, [16, 50, 84]))},
    }
    lr = chain[:, 3:]
    for i, name in enumerate(survey_names):
        r0 = np.exp(lr[:, i])
        out[f"r0_{name}"] = {
            k: float(v) for k, v in zip(["p16", "p50", "p84"], np.percentile(r0, [16, 50, 84]))
        }
    return out


@dataclass(frozen=True)
class LogProbSingle:
    survey: SurveyCells
    d_max: float
    log_r_min: float
    log_r_max: float

    def __call__(self, x: np.ndarray) -> float:
        d, ra, sd, lr = map(float, x)
        if not (0.0 <= d <= float(self.d_max)):
            return -np.inf
        if not (0.0 <= ra <= 360.0):
            return -np.inf
        if not (-1.0 <= sd <= 1.0):
            return -np.inf
        if not (float(self.log_r_min) <= lr <= float(self.log_r_max)):
            return -np.inf
        dip_hat = _dip_hat_from_ra_sindec(ra, sd)
        if not np.all(np.isfinite(dip_hat)):
            return -np.inf
        return nb_loglike_dipole(self.survey, d=d, dip_hat=dip_hat, r0=math.exp(lr))


@dataclass(frozen=True)
class LogProbJoint:
    surveys: list[SurveyCells]
    d_max: float
    log_r_min: float
    log_r_max: float

    def __call__(self, x: np.ndarray) -> float:
        d = float(x[0])
        ra = float(x[1])
        sd = float(x[2])
        if not (0.0 <= d <= float(self.d_max)):
            return -np.inf
        if not (0.0 <= ra <= 360.0):
            return -np.inf
        if not (-1.0 <= sd <= 1.0):
            return -np.inf
        dip_hat = _dip_hat_from_ra_sindec(ra, sd)
        if not np.all(np.isfinite(dip_hat)):
            return -np.inf
        lr = np.asarray(x[3:], dtype=float)
        if np.any(lr < float(self.log_r_min)) or np.any(lr > float(self.log_r_max)):
            return -np.inf
        r0s = [float(math.exp(v)) for v in lr.tolist()]
        return joint_loglike(self.surveys, d=d, dip_hat=dip_hat, r0s=r0s)


def simulate_nb_counts(
    s: SurveyCells,
    *,
    d_inj: float,
    dip_hat: np.ndarray,
    r0: float,
    eta_extra: np.ndarray | None,
    modulation: Literal["linear", "exp"],
    rng: np.random.Generator,
) -> np.ndarray:
    cos_th = np.asarray(s.nvec_eq @ dip_hat, dtype=float)
    if eta_extra is None:
        eta_extra = 0.0
    if modulation == "linear":
        ri = float(r0) * (1.0 + float(d_inj) * cos_th + eta_extra)
    else:
        eta = np.clip(float(d_inj) * cos_th + eta_extra, -20.0, 20.0)
        ri = float(r0) * np.exp(eta)
    ri = np.asarray(ri, dtype=float)
    ri = np.clip(ri, 1e-6, np.inf)
    # Gamma-Poisson mixture for real-valued shape ri:
    #   lambda ~ Gamma(shape=ri, scale=p/(1-p)); y ~ Poisson(lambda)
    scale = float(s.p / (1.0 - s.p))
    lam = rng.gamma(shape=ri, scale=scale, size=ri.shape[0])
    y = rng.poisson(lam).astype(np.int64, copy=False)
    return y


def run_injection_recovery(
    surveys: list[SurveyCells],
    *,
    truth_r0s: dict[str, float],
    d_inj: float,
    inj_ra_deg: float,
    inj_dec_deg: float,
    coeff_inj: np.ndarray,
    nsim: int,
    seed: int,
    d_max: float,
    log_r_min: float,
    log_r_max: float,
    fit_n_starts: int,
    fit_start_seed: int,
    pure_ref_ra: float,
    pure_ref_dec: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    dip_hat = _dip_hat_from_ra_sindec(float(inj_ra_deg), math.sin(math.radians(float(inj_dec_deg))))

    d_pure = []
    d_tmpl = []
    ang_pure = []
    ang_tmpl = []
    ll_pure = []
    ll_tmpl = []

    for i in range(int(nsim)):
        sim_surveys = []
        for s in surveys:
            coeff_inj_full = np.zeros(int(s.templates.shape[1]), dtype=float)
            coeff_inj_arr = np.asarray(coeff_inj, dtype=float).reshape(-1)
            coeff_inj_full[: min(4, coeff_inj_full.size)] = coeff_inj_arr[: min(4, coeff_inj_full.size)]
            eta = s.templates @ coeff_inj_full
            y = simulate_nb_counts(
                s,
                d_inj=float(d_inj),
                dip_hat=dip_hat,
                r0=float(truth_r0s[s.name]),
                eta_extra=eta,
                modulation="linear",
                rng=rng,
            )
            sim_surveys.append(rebuild_survey_cells_with_counts(s, y))

        fit_p = map_fit_joint(
            sim_surveys,
            d_max=float(d_max),
            ra0_deg=float(pure_ref_ra),
            dec0_deg=float(pure_ref_dec),
            log_r_min=float(log_r_min),
            log_r_max=float(log_r_max),
            nb_model="pure",
            n_starts=int(fit_n_starts),
            start_seed=int(fit_start_seed + 2 * i + 1),
        )
        fit_t = map_fit_joint(
            sim_surveys,
            d_max=float(d_max),
            ra0_deg=float(pure_ref_ra),
            dec0_deg=float(pure_ref_dec),
            log_r_min=float(log_r_min),
            log_r_max=float(log_r_max),
            nb_model="decra_linear",
            tmpl_coef_bound=0.2,
            tmpl_ridge=20.0,
            n_starts=int(fit_n_starts),
            start_seed=int(fit_start_seed + 2 * i + 2),
        )

        d_pure.append(float(fit_p["d"]))
        d_tmpl.append(float(fit_t["d"]))
        ang_pure.append(float(ang_sep_deg(np.asarray(fit_p["dip_hat"], dtype=float), dip_hat)))
        ang_tmpl.append(float(ang_sep_deg(np.asarray(fit_t["dip_hat"], dtype=float), dip_hat)))
        ll_pure.append(float(fit_p["loglike"]))
        ll_tmpl.append(float(fit_t["loglike"]))

    d_p = np.asarray(d_pure, dtype=float)
    d_t = np.asarray(d_tmpl, dtype=float)
    a_p = np.asarray(ang_pure, dtype=float)
    a_t = np.asarray(ang_tmpl, dtype=float)
    llp = np.asarray(ll_pure, dtype=float)
    llt = np.asarray(ll_tmpl, dtype=float)
    return {
        "nsim": int(nsim),
        "seed": int(seed),
        "d_inj": float(d_inj),
        "inj_ra_deg": float(inj_ra_deg),
        "inj_dec_deg": float(inj_dec_deg),
        "coeff_inj_dec_dec2_sinra_cosra": [float(v) for v in np.asarray(coeff_inj, dtype=float).tolist()],
        "pure": {
            "d": summarize_values(d_p),
            "axis_err_deg": summarize_values(a_p),
            "loglike": summarize_values(llp),
        },
        "template_linear": {
            "d": summarize_values(d_t),
            "axis_err_deg": summarize_values(a_t),
            "loglike": summarize_values(llt),
        },
        "delta_template_minus_pure": {
            "d": summarize_values(d_t - d_p),
            "axis_err_deg": summarize_values(a_t - a_p),
            "loglike": summarize_values(llt - llp),
        },
    }


def run_emcee(
    log_prob: Any,
    *,
    x_map: np.ndarray,
    init_scales: np.ndarray,
    nwalkers: int,
    nburn: int,
    nstep: int,
    seed: int,
    nproc: int,
) -> dict[str, Any]:
    import emcee

    rng = np.random.default_rng(int(seed))
    x_map = np.asarray(x_map, dtype=float)
    init_scales = np.asarray(init_scales, dtype=float)
    ndim = int(x_map.size)

    p0 = x_map[None, :] + rng.normal(size=(int(nwalkers), ndim)) * init_scales[None, :]
    # Resample walkers that start outside bounds (log_prob = -inf).
    for _ in range(50):
        bad = ~np.isfinite(np.array([log_prob(x) for x in p0]))
        if not np.any(bad):
            break
        p0[bad] = x_map[None, :] + rng.normal(size=(int(np.sum(bad)), ndim)) * init_scales[None, :]

    pool = None
    if int(nproc) > 1:
        ctx = get_context("fork")
        pool = ctx.Pool(processes=int(nproc))
    try:
        sampler = emcee.EnsembleSampler(int(nwalkers), ndim, log_prob, pool=pool)
        sampler.run_mcmc(p0, int(nburn), progress=True)
        sampler.reset()
        sampler.run_mcmc(None, int(nstep), progress=True)
        chain = sampler.get_chain(flat=True)
        lnp = sampler.get_log_prob(flat=True)
        acc = sampler.acceptance_fraction
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {
        "nwalkers": int(nwalkers),
        "nburn": int(nburn),
        "nstep": int(nstep),
        "nsamp": int(chain.shape[0]),
        "acc_mean": float(np.mean(acc)),
        "acc_min": float(np.min(acc)),
        "acc_max": float(np.max(acc)),
        "chain": np.asarray(chain, dtype=float),
        "log_prob": np.asarray(lnp, dtype=float),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Negative-binomial radio dipole audit (arXiv:2509.16732)")
    lines.append("")
    lines.append("Paper-faithful reproduction of the NB counts-in-cells dipole estimator (Eq. 6-8).")
    lines.append("All fits include **zero-count** cells inside the survey footprint (Nside=32).")
    lines.append("")

    cfg = payload["config"]
    lines.append("## Config")
    lines.append(f"- created_utc: `{payload['meta']['created_utc']}`")
    lines.append(f"- nside: `{cfg['nside']}` (nest={cfg['nest']})")
    lines.append(f"- NB model: `{cfg['nb_model']}`")
    lines.append(f"- extra templates: `{cfg.get('extra_templates', [])}`")
    lines.append(f"- galactic cut: `|b| >= {cfg['gal_b_cut_deg']:.1f} deg`")
    lines.append(f"- LoTSS MOC mode: `{cfg['lotss_moc_mode']}` (lmax={cfg['lotss_moc_lmax']})")
    lines.append(
        f"- template controls: coef_bound={cfg['tmpl_coef_bound']}, ridge={cfg['tmpl_ridge']}"
    )
    lines.append(
        f"- do_mcmc: `{cfg['do_mcmc']}` / executed: `{cfg['mcmc_executed']}` "
        f"(walkers={cfg['nwalkers']}, burn={cfg['nburn']}, step={cfg['nstep']})"
    )
    lines.append(f"- multistart: n_starts={cfg['n_starts']}, start_seed={cfg['start_seed']}")
    lines.append("")

    lines.append("## Footprints / overdispersion (p from empirical mean/var)")
    for s in payload["surveys"]:
        lines.append(
            f"- **{s['name']}** cut={s['cut_mjy']} mJy: cells={s['n_cell']}, Nsrc={s['n_src']}, "
            f"mu={s['mu']:.2f}, var={s['var']:.2f}, p={s['p']:.6f}"
        )
    lines.append("")

    lines.append("## Fits")
    for name, fit in payload["fits"].items():
        mp = fit["map"]
        lines.append(f"- **{name}** MAP: d={mp['d']:.5f}, (RA,Dec)=({mp['ra_deg']:.1f},{mp['dec_deg']:.1f})")
        ic = mp.get("ic")
        if ic is not None:
            lines.append(
                f"  - loglike={mp['loglike']:.2f}, AIC={ic['aic']:.2f}, BIC={ic['bic']:.2f} (k={int(ic['k_params'])}, n={int(ic['n_obs'])})"
            )
        if fit.get("mcmc") is None:
            continue
        post = fit["mcmc"]["summary"]
        lines.append(
            f"  - d p50 [p16,p84] = {post['d']['p50']:.5f} [{post['d']['p16']:.5f},{post['d']['p84']:.5f}]"
        )
        lines.append(
            f"  - (RA,Dec) p50 = ({post['ra_deg']['p50']:.1f},{post['dec_deg']['p50']:.1f})"
        )
        lines.append(
            f"  - acc frac mean={fit['mcmc']['acc_mean']:.3f} (min={fit['mcmc']['acc_min']:.3f}, max={fit['mcmc']['acc_max']:.3f})"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append(
        "- LoTSS: the paper cites an 'inner masked region' from Hale+23 [27]. The public DR2 MOC is a best-effort proxy; "
        "small blanked facets are not encoded in the MOC."
    )
    lines.append(
        "- NB overdispersion inflates uncertainties vs Poisson, but it does not itself correct coherent spatial selection/calibration "
        "systematics that can mimic a dipole."
    )
    tri = payload["fits"].get("LoTSS-DR2 + RACS-low + NVSS", {}).get("map", {})
    if cfg["nb_model"] != "pure" and tri.get("template_names"):
        lines.append(f"- Template basis: `{tri['template_names']}`")
    if cfg["nb_model"] == "decra_exp":
        lines.append(
            "- This run uses an extended stress-test model with exp-modulated nuisance templates in r_i; "
            "this is broader than the paper's baseline Eq. 6-8 form."
        )
    if cfg["nb_model"] == "decra_linear":
        lines.append(
            "- This run uses a linear nuisance-template extension in r_i with bounded coefficients; "
            "this is a controlled stress test around the paper baseline."
        )
    inj = payload.get("injection_recovery")
    if inj is not None:
        lines.append(
            f"- Injection/recovery ran with nsim={inj['nsim']}, d_inj={inj['d_inj']:.5f}, "
            f"(RA,Dec)=({inj['inj_ra_deg']:.2f},{inj['inj_dec_deg']:.2f})."
        )
        lines.append(
            f"- Injection summary: pure axis err p50={inj['pure']['axis_err_deg']['p50']:.2f} deg vs "
            f"template-linear axis err p50={inj['template_linear']['axis_err_deg']['p50']:.2f} deg."
        )
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nvss-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/nvss/reference/NVSS.fit",
    )
    ap.add_argument(
        "--racs-csv-gz",
        default="data/external/radio_dipole/racs_low/racs_low_dr1_sources_galacticcut_v2021_08_v02_mincols.csv.gz",
    )
    ap.add_argument(
        "--lotss-fits",
        default="data/external/radio_dipole/lotss_dr2/LoTSS_DR2_v110_masked.srl.fits",
    )
    ap.add_argument("--lotss-moc", default="data/external/radio_dipole/lotss_dr2/dr2-moc.moc")
    ap.add_argument(
        "--haslam-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/reference/haslam408_dsds_Remazeilles2014_512.fits",
    )
    ap.add_argument(
        "--extra-templates",
        default="",
        help="Comma-separated extra templates: haslam_k408,nvss_e_s1_4,lotss_isl_rms,lotss_masked_fraction,lotss_moc_frac",
    )
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--gal-b-cut-deg", type=float, default=10.0)
    ap.add_argument("--lotss-moc-mode", choices=["any", "full", "center"], default="full")
    ap.add_argument("--lotss-moc-lmax", type=int, default=10)
    ap.add_argument("--lotss-dec-min", type=float, default=None)
    ap.add_argument("--lotss-dec-max", type=float, default=None)

    ap.add_argument("--nvss-cut-mjy", type=float, default=20.0)
    ap.add_argument("--racs-cut-mjy", type=float, default=20.0)
    ap.add_argument("--lotss-cut-mjy", type=float, default=5.0)
    ap.add_argument("--nb-model", choices=["pure", "decra_linear", "decra_exp"], default="pure")
    ap.add_argument("--tmpl-coef-bound", type=float, default=1.5)
    ap.add_argument("--tmpl-ridge", type=float, default=0.0)

    ap.add_argument("--do-mcmc", action="store_true", default=True)
    ap.add_argument("--no-mcmc", dest="do_mcmc", action="store_false")
    ap.add_argument("--nwalkers", type=int, default=48)
    ap.add_argument("--nburn", type=int, default=600)
    ap.add_argument("--nstep", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=20260208)
    ap.add_argument("--nproc", type=int, default=min(64, (os.cpu_count() or 1)))
    ap.add_argument("--n-starts", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=20260208)

    ap.add_argument("--chunk-rows", type=int, default=800_000)
    ap.add_argument("--d-max", type=float, default=0.2)
    ap.add_argument("--log-r-min", type=float, default=-20.0)
    ap.add_argument("--log-r-max", type=float, default=25.0)
    ap.add_argument("--injection-nsim", type=int, default=0)
    ap.add_argument("--inj-d", type=float, default=0.01735)
    ap.add_argument("--inj-ra", type=float, default=167.94)
    ap.add_argument("--inj-dec", type=float, default=-6.94)
    ap.add_argument("--inj-a-dec", type=float, default=0.03)
    ap.add_argument("--inj-a-dec2", type=float, default=-0.015)
    ap.add_argument("--inj-a-sinra", type=float, default=0.01)
    ap.add_argument("--inj-a-cosra", type=float, default=0.01)
    args = ap.parse_args()

    extra_templates_raw = [t.strip() for t in str(args.extra_templates).split(",") if t.strip()]
    extra_templates_set = set(extra_templates_raw)
    allowed_extra = {
        "haslam_k408",
        "nvss_e_s1_4",
        "lotss_isl_rms",
        "lotss_masked_fraction",
        "lotss_moc_frac",
    }
    unknown = sorted(extra_templates_set - allowed_extra)
    if unknown:
        raise SystemExit(f"Unknown --extra-templates entries: {unknown}. Allowed: {sorted(allowed_extra)}")

    outdir = Path(args.outdir or f"outputs/radio_nb_dipole_audit_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    geom = build_pixel_geom(int(args.nside), nest=True)

    # Paper's CMB dipole direction in equatorial coordinates.
    cmb_ra = 167.94
    cmb_dec = -6.94
    cmb_hat = _dip_hat_from_ra_sindec(cmb_ra, math.sin(math.radians(cmb_dec)))

    gal_ok = np.abs(geom.gal_b_deg) >= float(args.gal_b_cut_deg)
    nvss_valid = gal_ok & (geom.dec_deg >= -39.0)
    racs_valid = gal_ok & (geom.dec_deg >= -78.0) & (geom.dec_deg <= 28.0)
    lotss_valid = lotss_valid_pix_from_moc(
        Path(args.lotss_moc),
        geom=geom,
        gal_b_cut_deg=float(args.gal_b_cut_deg),
        mode=str(args.lotss_moc_mode),
        lmax=int(args.lotss_moc_lmax),
    )
    if args.lotss_dec_min is not None:
        lotss_valid &= geom.dec_deg >= float(args.lotss_dec_min)
    if args.lotss_dec_max is not None:
        lotss_valid &= geom.dec_deg <= float(args.lotss_dec_max)

    haslam_k408 = None
    if "haslam_k408" in extra_templates_set:
        haslam_k408 = sample_haslam_k408(geom, haslam_fits=Path(args.haslam_fits))

    lotss_moc_frac = None
    if "lotss_moc_frac" in extra_templates_set:
        lotss_moc_frac = moc_coverage_fraction(
            Path(args.lotss_moc),
            geom=geom,
            gal_b_cut_deg=float(args.gal_b_cut_deg),
            lmax=int(args.lotss_moc_lmax),
        )

    nvss_mean_cols = []
    if "nvss_e_s1_4" in extra_templates_set:
        nvss_mean_cols.append("e_S1_4")
    lotss_mean_cols = []
    if "lotss_isl_rms" in extra_templates_set:
        lotss_mean_cols.append("Isl_rms")
    if "lotss_masked_fraction" in extra_templates_set:
        lotss_mean_cols.append("Masked_Fraction")

    nvss_counts, nvss_n, nvss_means = counts_from_fits(
        Path(args.nvss_fits),
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="S1_4",
        cut_mjy=float(args.nvss_cut_mjy),
        geom=geom,
        valid_pix=nvss_valid,
        source_extra_mask_fn=lambda ra, dec: (dec >= -39.0),
        gal_b_cut_deg=float(args.gal_b_cut_deg),
        chunk_rows=int(args.chunk_rows),
        mean_cols=nvss_mean_cols,
    )
    racs_counts, racs_n = counts_from_racs_csv(
        Path(args.racs_csv_gz),
        cut_mjy=float(args.racs_cut_mjy),
        geom=geom,
        valid_pix=racs_valid,
        dec_min=-78.0,
        dec_max=28.0,
        gal_b_cut_deg=float(args.gal_b_cut_deg),
    )
    lotss_counts, lotss_n, lotss_means = counts_from_fits(
        Path(args.lotss_fits),
        ra_col="RA",
        dec_col="DEC",
        flux_col="Total_flux",
        cut_mjy=float(args.lotss_cut_mjy),
        geom=geom,
        valid_pix=lotss_valid,
        source_extra_mask_fn=(
            None
            if (args.lotss_dec_min is None and args.lotss_dec_max is None)
            else (
                lambda ra, dec: (
                    ((dec >= float(args.lotss_dec_min)) if args.lotss_dec_min is not None else np.ones_like(dec, dtype=bool))
                    & ((dec <= float(args.lotss_dec_max)) if args.lotss_dec_max is not None else np.ones_like(dec, dtype=bool))
                )
            )
        ),
        gal_b_cut_deg=float(args.gal_b_cut_deg),
        chunk_rows=int(args.chunk_rows),
        mean_cols=lotss_mean_cols,
    )

    def extra_templates_for(name: str, valid_pix: np.ndarray) -> list[tuple[str, np.ndarray]]:
        valid_idx = np.where(np.asarray(valid_pix, dtype=bool))[0]
        out: list[tuple[str, np.ndarray]] = []
        for t in extra_templates_raw:
            if t == "haslam_k408":
                raw = np.asarray(haslam_k408[valid_idx], dtype=float) if haslam_k408 is not None else np.zeros(valid_idx.size)
                out.append(("haslam_k408_z", raw))
            elif t == "nvss_e_s1_4":
                raw = (
                    np.asarray(nvss_means.get("e_S1_4", np.zeros_like(geom.ra_deg, dtype=float))[valid_idx], dtype=float)
                    if name == "NVSS"
                    else np.zeros(valid_idx.size, dtype=float)
                )
                out.append(("nvss_e_s1_4_z", raw))
            elif t == "lotss_isl_rms":
                raw = (
                    np.asarray(lotss_means.get("Isl_rms", np.zeros_like(geom.ra_deg, dtype=float))[valid_idx], dtype=float)
                    if name == "LoTSS-DR2"
                    else np.zeros(valid_idx.size, dtype=float)
                )
                out.append(("lotss_isl_rms_z", raw))
            elif t == "lotss_masked_fraction":
                raw = (
                    np.asarray(
                        lotss_means.get("Masked_Fraction", np.zeros_like(geom.ra_deg, dtype=float))[valid_idx],
                        dtype=float,
                    )
                    if name == "LoTSS-DR2"
                    else np.zeros(valid_idx.size, dtype=float)
                )
                out.append(("lotss_masked_fraction_z", raw))
            elif t == "lotss_moc_frac":
                raw = (
                    np.asarray(lotss_moc_frac[valid_idx], dtype=float) if (name == "LoTSS-DR2" and lotss_moc_frac is not None)
                    else np.zeros(valid_idx.size, dtype=float)
                )
                out.append(("lotss_moc_frac_z", raw))
            else:
                raise RuntimeError(f"Unhandled extra template: {t}")
        return out

    nvss = build_survey_cells(
        "NVSS",
        1400.0,
        geom=geom,
        valid_pix=nvss_valid,
        counts=nvss_counts,
        cut_mjy=float(args.nvss_cut_mjy),
        n_src=nvss_n,
        extra_templates=extra_templates_for("NVSS", nvss_valid),
    )
    racs = build_survey_cells(
        "RACS-low",
        888.0,
        geom=geom,
        valid_pix=racs_valid,
        counts=racs_counts,
        cut_mjy=float(args.racs_cut_mjy),
        n_src=racs_n,
        extra_templates=extra_templates_for("RACS-low", racs_valid),
    )
    lotss = build_survey_cells(
        "LoTSS-DR2",
        144.0,
        geom=geom,
        valid_pix=lotss_valid,
        counts=lotss_counts,
        cut_mjy=float(args.lotss_cut_mjy),
        n_src=lotss_n,
        extra_templates=extra_templates_for("LoTSS-DR2", lotss_valid),
    )

    fits_out: dict[str, Any] = {}
    map_nvss = map_fit_single(
        nvss,
        d_max=args.d_max,
        ra0_deg=cmb_ra,
        dec0_deg=cmb_dec,
        log_r_min=args.log_r_min,
        log_r_max=args.log_r_max,
        nb_model=str(args.nb_model),
        tmpl_coef_bound=float(args.tmpl_coef_bound),
        tmpl_ridge=float(args.tmpl_ridge),
        n_starts=int(args.n_starts),
        start_seed=int(args.start_seed) + 11,
    )
    map_racs = map_fit_single(
        racs,
        d_max=args.d_max,
        ra0_deg=cmb_ra,
        dec0_deg=cmb_dec,
        log_r_min=args.log_r_min,
        log_r_max=args.log_r_max,
        nb_model=str(args.nb_model),
        tmpl_coef_bound=float(args.tmpl_coef_bound),
        tmpl_ridge=float(args.tmpl_ridge),
        n_starts=int(args.n_starts),
        start_seed=int(args.start_seed) + 22,
    )
    map_lotss = map_fit_single(
        lotss,
        d_max=args.d_max,
        ra0_deg=cmb_ra,
        dec0_deg=cmb_dec,
        log_r_min=args.log_r_min,
        log_r_max=args.log_r_max,
        nb_model=str(args.nb_model),
        tmpl_coef_bound=float(args.tmpl_coef_bound),
        tmpl_ridge=float(args.tmpl_ridge),
        n_starts=int(args.n_starts),
        start_seed=int(args.start_seed) + 33,
    )
    fits_out["NVSS"] = {"map": map_nvss, "mcmc": None}
    fits_out["RACS-low"] = {"map": map_racs, "mcmc": None}
    fits_out["LoTSS-DR2"] = {"map": map_lotss, "mcmc": None}

    map_joint2 = map_fit_joint(
        [racs, nvss],
        d_max=args.d_max,
        ra0_deg=cmb_ra,
        dec0_deg=cmb_dec,
        log_r_min=args.log_r_min,
        log_r_max=args.log_r_max,
        nb_model=str(args.nb_model),
        tmpl_coef_bound=float(args.tmpl_coef_bound),
        tmpl_ridge=float(args.tmpl_ridge),
        n_starts=int(args.n_starts),
        start_seed=int(args.start_seed) + 44,
    )
    map_joint3 = map_fit_joint(
        [lotss, racs, nvss],
        d_max=args.d_max,
        ra0_deg=cmb_ra,
        dec0_deg=cmb_dec,
        log_r_min=args.log_r_min,
        log_r_max=args.log_r_max,
        nb_model=str(args.nb_model),
        tmpl_coef_bound=float(args.tmpl_coef_bound),
        tmpl_ridge=float(args.tmpl_ridge),
        n_starts=int(args.n_starts),
        start_seed=int(args.start_seed) + 55,
    )
    fits_out["RACS-low + NVSS"] = {"map": map_joint2, "mcmc": None}
    fits_out["LoTSS-DR2 + RACS-low + NVSS"] = {"map": map_joint3, "mcmc": None}

    mcmc_executed = bool(args.do_mcmc) and str(args.nb_model) == "pure"
    if mcmc_executed:
        # Singles
        for s, key, mp in [(nvss, "NVSS", map_nvss), (racs, "RACS-low", map_racs), (lotss, "LoTSS-DR2", map_lotss)]:
            lp = LogProbSingle(s, float(args.d_max), float(args.log_r_min), float(args.log_r_max))
            x_map = np.array([mp["d"], mp["ra_deg"], math.sin(math.radians(mp["dec_deg"])), math.log(mp["r0"])], dtype=float)
            init = np.array([5e-4, 1.5, 0.02, 0.06], dtype=float)
            m = run_emcee(
                lp,
                x_map=x_map,
                init_scales=init,
                nwalkers=int(args.nwalkers),
                nburn=int(args.nburn),
                nstep=int(args.nstep),
                seed=int(args.seed) + (hash(key) % 10_000),
                nproc=int(args.nproc),
            )
            m_out = {k: m[k] for k in ["nwalkers", "nburn", "nstep", "nsamp", "acc_mean", "acc_min", "acc_max"]}
            m_out["summary"] = summarize_chain(m["chain"], ra_ref_deg=float(mp["ra_deg"]), survey_names=[key])
            fits_out[key]["mcmc"] = m_out

        # Joint 2
        lp2 = LogProbJoint([racs, nvss], float(args.d_max), float(args.log_r_min), float(args.log_r_max))
        x2_map = np.array(
            [map_joint2["d"], map_joint2["ra_deg"], math.sin(math.radians(map_joint2["dec_deg"]))] +
            [math.log(map_joint2["r0s"][nm]) for nm in ["RACS-low", "NVSS"]],
            dtype=float,
        )
        init2 = np.array([5e-4, 1.2, 0.02, 0.06, 0.06], dtype=float)
        m2 = run_emcee(
            lp2,
            x_map=x2_map,
            init_scales=init2,
            nwalkers=int(args.nwalkers),
            nburn=int(args.nburn),
            nstep=int(args.nstep),
            seed=int(args.seed) + 2222,
            nproc=int(args.nproc),
        )
        m2_out = {k: m2[k] for k in ["nwalkers", "nburn", "nstep", "nsamp", "acc_mean", "acc_min", "acc_max"]}
        m2_out["summary"] = summarize_chain(m2["chain"], ra_ref_deg=float(map_joint2["ra_deg"]), survey_names=["RACS-low", "NVSS"])
        fits_out["RACS-low + NVSS"]["mcmc"] = m2_out

        # Joint 3
        lp3 = LogProbJoint([lotss, racs, nvss], float(args.d_max), float(args.log_r_min), float(args.log_r_max))
        x3_map = np.array(
            [map_joint3["d"], map_joint3["ra_deg"], math.sin(math.radians(map_joint3["dec_deg"]))] +
            [math.log(map_joint3["r0s"][nm]) for nm in ["LoTSS-DR2", "RACS-low", "NVSS"]],
            dtype=float,
        )
        init3 = np.array([5e-4, 1.2, 0.02, 0.06, 0.06, 0.06], dtype=float)
        m3 = run_emcee(
            lp3,
            x_map=x3_map,
            init_scales=init3,
            nwalkers=int(args.nwalkers),
            nburn=int(args.nburn),
            nstep=int(args.nstep),
            seed=int(args.seed) + 3333,
            nproc=int(args.nproc),
        )
        m3_out = {k: m3[k] for k in ["nwalkers", "nburn", "nstep", "nsamp", "acc_mean", "acc_min", "acc_max"]}
        m3_out["summary"] = summarize_chain(
            m3["chain"], ra_ref_deg=float(map_joint3["ra_deg"]), survey_names=["LoTSS-DR2", "RACS-low", "NVSS"]
        )
        fits_out["LoTSS-DR2 + RACS-low + NVSS"]["mcmc"] = m3_out

    injection_out: dict[str, Any] | None = None
    if int(args.injection_nsim) > 0:
        coeff_inj = np.array(
            [args.inj_a_dec, args.inj_a_dec2, args.inj_a_sinra, args.inj_a_cosra], dtype=float
        )
        injection_out = run_injection_recovery(
            [lotss, racs, nvss],
            truth_r0s=dict(map_joint3["r0s"]),
            d_inj=float(args.inj_d),
            inj_ra_deg=float(args.inj_ra),
            inj_dec_deg=float(args.inj_dec),
            coeff_inj=coeff_inj,
            nsim=int(args.injection_nsim),
            seed=int(args.seed) + 909,
            d_max=float(args.d_max),
            log_r_min=float(args.log_r_min),
            log_r_max=float(args.log_r_max),
            fit_n_starts=max(1, int(args.n_starts)),
            fit_start_seed=int(args.start_seed) + 4040,
            pure_ref_ra=float(map_joint3["ra_deg"]),
            pure_ref_dec=float(map_joint3["dec_deg"]),
        )

    payload: dict[str, Any] = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(args.seed),
            "paper": "arXiv:2509.16732",
            "cmb_ra_deg": float(cmb_ra),
            "cmb_dec_deg": float(cmb_dec),
            "cmb_hat": [float(x) for x in cmb_hat.tolist()],
        },
        "config": {
            "nside": int(args.nside),
            "nest": True,
            "nb_model": str(args.nb_model),
            "extra_templates": extra_templates_raw,
            "gal_b_cut_deg": float(args.gal_b_cut_deg),
            "lotss_moc_mode": str(args.lotss_moc_mode),
            "lotss_moc_lmax": int(args.lotss_moc_lmax),
            "lotss_dec_min": (None if args.lotss_dec_min is None else float(args.lotss_dec_min)),
            "lotss_dec_max": (None if args.lotss_dec_max is None else float(args.lotss_dec_max)),
            "do_mcmc": bool(args.do_mcmc),
            "mcmc_executed": bool(mcmc_executed),
            "nwalkers": int(args.nwalkers),
            "nburn": int(args.nburn),
            "nstep": int(args.nstep),
            "nproc": int(args.nproc),
            "tmpl_coef_bound": float(args.tmpl_coef_bound),
            "tmpl_ridge": float(args.tmpl_ridge),
            "n_starts": int(args.n_starts),
            "start_seed": int(args.start_seed),
            "injection_nsim": int(args.injection_nsim),
            "inj_d": float(args.inj_d),
            "inj_ra": float(args.inj_ra),
            "inj_dec": float(args.inj_dec),
            "inj_a_dec": float(args.inj_a_dec),
            "inj_a_dec2": float(args.inj_a_dec2),
            "inj_a_sinra": float(args.inj_a_sinra),
            "inj_a_cosra": float(args.inj_a_cosra),
        },
        "inputs": {
            "nvss_fits": str(args.nvss_fits),
            "racs_csv_gz": str(args.racs_csv_gz),
            "lotss_fits": str(args.lotss_fits),
            "lotss_moc": str(args.lotss_moc),
            "haslam_fits": str(args.haslam_fits),
        },
        "surveys": [
            {"name": nvss.name, "cut_mjy": nvss.cut_mjy, "n_cell": nvss.n_cell, "n_src": nvss.n_src, "mu": nvss.mu, "var": nvss.var, "p": nvss.p},
            {"name": racs.name, "cut_mjy": racs.cut_mjy, "n_cell": racs.n_cell, "n_src": racs.n_src, "mu": racs.mu, "var": racs.var, "p": racs.p},
            {"name": lotss.name, "cut_mjy": lotss.cut_mjy, "n_cell": lotss.n_cell, "n_src": lotss.n_src, "mu": lotss.mu, "var": lotss.var, "p": lotss.p},
        ],
        "fits": fits_out,
        "injection_recovery": injection_out,
    }

    jpath = outdir / "radio_nb_dipole_audit.json"
    rpath = outdir / "master_report.md"
    jpath.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(rpath, payload)

    print(json.dumps({"outdir": str(outdir), "json": str(jpath), "report": str(rpath)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
