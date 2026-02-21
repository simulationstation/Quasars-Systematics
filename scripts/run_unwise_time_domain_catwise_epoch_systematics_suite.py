#!/usr/bin/env python3
"""Run the CatWISE-parent unWISE epoch systematics suite.

This script performs the six requested follow-up tests on an existing
`run_unwise_time_domain_catwise_epoch_dipole.py` output:

1) Epoch re-fit with nuisance controls.
2) Per-epoch comparability audit table.
3) Common-support / matched-depth subset re-fit.
4) Template-attribution diagnostics from deviance partitions.
5) Injection-through-real-epoch-systematics Monte Carlo.
6) Direction-stability diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CATWISE_DEFAULT = (
    "data/external/zenodo_6784602/secrest_extracted/"
    "secrest+22_accepted/wise/reference/catwise_agns.fits"
)
EXCLUDE_DEFAULT = (
    "data/external/zenodo_6784602/secrest_extracted/"
    "secrest+22_accepted/wise/reference/exclude_master_revised.fits"
)
DEPTH_DEFAULT = "data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits"

# Secrest+22 / Planck CMB dipole axis used elsewhere in this repo.
CMB_L_DEG = 264.021
CMB_B_DEG = 48.253


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def read_healpix_map_fits(path: Path) -> np.ndarray:
    from astropy.io import fits

    with fits.open(str(path), memmap=True) as hdul:
        if len(hdul) == 0:
            raise RuntimeError(f"{path}: empty FITS")
        # Table HDU case.
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            names = getattr(data, "names", None)
            if not names:
                continue
            arr = np.asarray(data[names[0]], dtype=np.float64).ravel()
            if arr.size > 0:
                return arr
        # Image HDU case.
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            if isinstance(data, np.ndarray) and data.ndim >= 1 and np.issubdtype(data.dtype, np.number):
                arr = np.asarray(data, dtype=np.float64).ravel()
                if arr.size > 0:
                    return arr
    raise RuntimeError(f"{path}: could not parse HEALPix map payload")


def zscore(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    vals = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    ok = m & np.isfinite(vals)
    if np.count_nonzero(ok) == 0:
        return np.zeros_like(vals), float("nan"), float("nan")
    mu = float(np.mean(vals[ok]))
    sd = float(np.std(vals[ok]))
    if not np.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(vals), mu, sd
    out = np.zeros_like(vals)
    out[ok] = (vals[ok] - mu) / sd
    return out, mu, sd


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(ok) == 0:
        return np.full(q.shape, np.nan, dtype=np.float64)
    x = x[ok]
    w = w[ok]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    tot = float(cw[-1])
    if tot <= 0.0:
        return np.full(q.shape, np.nan, dtype=np.float64)
    targets = np.clip(q, 0.0, 1.0) * tot
    idx = np.searchsorted(cw, targets, side="left")
    idx = np.clip(idx, 0, x.size - 1)
    return x[idx]


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.clip(mu, 1e-12, np.inf)
    t = np.zeros_like(y)
    pos = y > 0
    t[pos] = y[pos] * np.log(y[pos] / mu[pos]) - (y[pos] - mu[pos])
    t[~pos] = mu[~pos]
    return float(2.0 * np.sum(t))


def fit_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    beta_init: np.ndarray | None = None,
    max_iter: int = 400,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float]:
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if beta_init is None:
        mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
        beta0 = np.zeros(X.shape[1], dtype=np.float64)
        beta0[0] = math.log(mu0)
    else:
        beta0 = np.asarray(beta_init, dtype=np.float64).copy()
        if beta0.shape != (X.shape[1],):
            raise ValueError(f"beta_init shape mismatch: {beta0.shape} vs {(X.shape[1],)}")

    def f_and_g(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        return nll, np.asarray(grad, dtype=np.float64)

    res = minimize(
        lambda b: f_and_g(b)[0],
        beta0,
        jac=lambda b: f_and_g(b)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    beta = np.asarray(res.x, dtype=np.float64)
    eta = np.clip(X @ beta, -25.0, 25.0)
    mu = np.exp(eta)
    dev = poisson_deviance(y, mu)

    cov = None
    try:
        fisher = X.T @ (mu[:, None] * X)
        cov = np.linalg.inv(fisher)
    except Exception:  # noqa: BLE001
        cov = None

    return beta, cov, mu, dev


def vector_to_lb_deg(v: np.ndarray) -> tuple[float, float]:
    vv = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(vv))
    if not np.isfinite(n) or n <= 0.0:
        return float("nan"), float("nan")
    x, y, z = vv / n
    l = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    b = math.degrees(math.asin(np.clip(z, -1.0, 1.0)))
    return float(l), float(b)


def lb_to_unitvec(l_deg: float, b_deg: float) -> np.ndarray:
    l = math.radians(float(l_deg))
    b = math.radians(float(b_deg))
    cb = math.cos(b)
    return np.array([cb * math.cos(l), cb * math.sin(l), math.sin(b)], dtype=np.float64)


def sep_to_cmb_deg(bvec: np.ndarray) -> float:
    u = np.asarray(bvec, dtype=np.float64)
    nu = float(np.linalg.norm(u))
    if not np.isfinite(nu) or nu <= 0.0:
        return float("nan")
    u = u / nu
    c = lb_to_unitvec(CMB_L_DEG, CMB_B_DEG)
    # Sign-invariant axis separation.
    dot = float(np.clip(abs(np.dot(u, c)), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def dipole_metrics(beta: np.ndarray, cov: np.ndarray | None, *, dip_idx: tuple[int, int, int]) -> dict[str, float]:
    b = np.asarray([beta[dip_idx[0]], beta[dip_idx[1]], beta[dip_idx[2]]], dtype=np.float64)
    D = float(np.linalg.norm(b))
    sigma = float("nan")
    if cov is not None and np.isfinite(D) and D > 0.0:
        cb = np.asarray(cov[np.ix_(dip_idx, dip_idx)], dtype=np.float64)
        try:
            varD = float(b.T @ cb @ b) / float(D * D)
            sigma = float(math.sqrt(max(0.0, varD)))
        except Exception:  # noqa: BLE001
            sigma = float("nan")
    l_deg, b_deg = vector_to_lb_deg(b)
    return {
        "D": D,
        "sigma_D": sigma,
        "bx": float(b[0]),
        "by": float(b[1]),
        "bz": float(b[2]),
        "l_deg": l_deg,
        "b_deg": b_deg,
        "sep_to_cmb_deg": sep_to_cmb_deg(b),
    }


def constant_amplitude_chi2(D: np.ndarray, sD: np.ndarray) -> dict[str, float]:
    from scipy.stats import chi2 as chi2_dist

    D = np.asarray(D, dtype=np.float64)
    sD = np.asarray(sD, dtype=np.float64)
    ok = np.isfinite(D) & np.isfinite(sD) & (sD > 0.0)
    if np.count_nonzero(ok) < 2:
        return {
            "n": int(np.count_nonzero(ok)),
            "D_weighted_mean": float("nan"),
            "chi2": float("nan"),
            "dof": 0,
            "p_value": float("nan"),
            "D_min": float("nan"),
            "D_max": float("nan"),
            "D_range": float("nan"),
        }
    d = D[ok]
    s = sD[ok]
    w = 1.0 / (s * s)
    d0 = float(np.sum(w * d) / np.sum(w))
    ch = float(np.sum((d - d0) ** 2 * w))
    dof = int(d.size - 1)
    p = float(chi2_dist.sf(ch, dof)) if dof > 0 else float("nan")
    return {
        "n": int(d.size),
        "D_weighted_mean": d0,
        "chi2": ch,
        "dof": dof,
        "p_value": p,
        "D_min": float(np.min(d)),
        "D_max": float(np.max(d)),
        "D_range": float(np.max(d) - np.min(d)),
    }


def build_seen_mask(
    *,
    nside: int,
    catwise_catalog: Path,
    exclude_mask_fits: Path,
    b_cut_deg: float,
    w1cov_min: float,
) -> np.ndarray:
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.table import Table
    import astropy.units as u

    npix = hp.nside2npix(int(nside))

    with fits.open(str(catwise_catalog), memmap=True) as hdul:
        d = hdul[1].data
        w1cov = np.asarray(d["w1cov"], dtype=float)
        l = np.asarray(d["l"], dtype=float)
        b = np.asarray(d["b"], dtype=float)

    sel = np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b) & (w1cov >= float(w1cov_min))
    ipix_base = hp.ang2pix(
        int(nside),
        np.deg2rad(90.0 - b[sel]),
        np.deg2rad(l[sel] % 360.0),
        nest=False,
    ).astype(np.int64)
    cnt = np.bincount(ipix_base, minlength=npix)
    mask = np.zeros(npix, dtype=bool)
    idx0 = np.where(cnt == 0)[0]
    if idx0.size:
        neigh = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            neigh[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[neigh] = True

    tmask = Table.read(str(exclude_mask_fits), memmap=True)
    if "use" in tmask.colnames:
        tmask = tmask[np.asarray(tmask["use"], dtype=bool)]
    if len(tmask):
        sc = SkyCoord(tmask["ra"], tmask["dec"], unit=u.deg, frame="icrs").galactic
        radius = np.deg2rad(np.asarray(tmask["radius"], dtype=float))
        for lon, lat, rad in zip(sc.l.deg, sc.b.deg, radius, strict=True):
            vec = hp.ang2vec(np.deg2rad(90.0 - float(lat)), np.deg2rad(float(lon)))
            disc = hp.query_disc(int(nside), vec=vec, radius=float(rad), inclusive=True, nest=False)
            mask[disc] = True

    _lon, lat = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat) < float(b_cut_deg)

    return ~mask


def build_templates(
    *,
    nside: int,
    seen: np.ndarray,
    depth_map: np.ndarray,
) -> dict[str, np.ndarray]:
    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    npix = hp.nside2npix(int(nside))
    if depth_map.shape[0] != npix:
        raise ValueError(f"Depth map npix mismatch: got {depth_map.shape[0]} expected {npix}")

    lon_gal, lat_gal = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    sc = SkyCoord(l=lon_gal * u.deg, b=lat_gal * u.deg, frame="galactic").barycentricmeanecliptic
    elat_deg = np.asarray(sc.lat.deg, dtype=np.float64)
    elon_deg = np.asarray(sc.lon.deg, dtype=np.float64)

    abs_elat = np.abs(elat_deg)
    sin_elon = np.sin(np.deg2rad(elon_deg))
    cos_elon = np.cos(np.deg2rad(elon_deg))

    abs_elat_z, _, _ = zscore(abs_elat, seen)
    sin_elon_z, _, _ = zscore(sin_elon, seen)
    cos_elon_z, _, _ = zscore(cos_elon, seen)
    depth_z, depth_mean, depth_std = zscore(depth_map, seen)

    return {
        "abs_elat_z": abs_elat_z,
        "sin_elon_z": sin_elon_z,
        "cos_elon_z": cos_elon_z,
        "depth_raw": depth_map.astype(np.float64),
        "depth_z": depth_z,
        "depth_mean_seen": np.array([depth_mean]),
        "depth_std_seen": np.array([depth_std]),
    }


@dataclass
class EpochFits:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    coef_corr: dict[str, Any]


def run_epoch_fits(
    *,
    counts: np.ndarray,
    parent_counts: np.ndarray,
    seen_mask: np.ndarray,
    nhat: np.ndarray,
    templates: dict[str, np.ndarray],
    epochs: list[int],
    date_iso_by_epoch: dict[int, str],
    fit_label: str,
) -> EpochFits:
    mask = np.asarray(seen_mask, dtype=bool)
    if np.count_nonzero(mask) == 0:
        raise RuntimeError(f"{fit_label}: empty mask")

    depth_raw = np.asarray(templates["depth_raw"], dtype=np.float64)
    abs_elat_z = np.asarray(templates["abs_elat_z"], dtype=np.float64)
    sin_elon_z = np.asarray(templates["sin_elon_z"], dtype=np.float64)
    cos_elon_z = np.asarray(templates["cos_elon_z"], dtype=np.float64)
    depth_z = np.asarray(templates["depth_z"], dtype=np.float64)

    ymask_parent = np.asarray(parent_counts[mask], dtype=np.float64)
    nh = np.asarray(nhat[mask], dtype=np.float64)
    tmat = np.column_stack(
        [
            abs_elat_z[mask],
            sin_elon_z[mask],
            cos_elon_z[mask],
            depth_z[mask],
        ]
    )
    X0 = np.ones((nh.shape[0], 1), dtype=np.float64)
    X_dip = np.column_stack([np.ones(nh.shape[0]), nh[:, 0], nh[:, 1], nh[:, 2]])
    X_nuis = np.column_stack([np.ones(nh.shape[0]), tmat])
    X_full = np.column_stack([np.ones(nh.shape[0]), nh[:, 0], nh[:, 1], nh[:, 2], tmat])

    rows: list[dict[str, Any]] = []
    b_init_dip = None
    b_init_nuis = None
    b_init_full = None
    b_init_null = None

    for e in epochs:
        y = np.asarray(counts[e, mask], dtype=np.float64)
        N = float(np.sum(y))
        if N <= 0.0:
            rows.append({"epoch": int(e), "date_utc": date_iso_by_epoch.get(int(e), ""), "N": 0})
            continue

        b0, c0, mu0, dev0 = fit_poisson_glm(X0, y, beta_init=b_init_null)
        b_init_null = b0
        bd, cd, mud, devd = fit_poisson_glm(X_dip, y, beta_init=b_init_dip)
        b_init_dip = bd
        bn, cn, mun, devn = fit_poisson_glm(X_nuis, y, beta_init=b_init_nuis)
        b_init_nuis = bn
        bf, cf, muf, devf = fit_poisson_glm(X_full, y, beta_init=b_init_full)
        b_init_full = bf

        md = dipole_metrics(bd, cd, dip_idx=(1, 2, 3))
        mf = dipole_metrics(bf, cf, dip_idx=(1, 2, 3))

        # Per-epoch comparability stats.
        n_pix_unmasked = int(np.count_nonzero(mask))
        n_pix_active = int(np.count_nonzero(y > 0))
        n_pix_zero = int(n_pix_unmasked - n_pix_active)
        f_pix_active = float(n_pix_active / n_pix_unmasked) if n_pix_unmasked > 0 else float("nan")

        parent_valid = ymask_parent > 0
        frac = np.zeros_like(y, dtype=np.float64)
        frac[parent_valid] = y[parent_valid] / ymask_parent[parent_valid]
        fq = weighted_quantile(frac[parent_valid], ymask_parent[parent_valid], [0.25, 0.5, 0.75])

        dq = weighted_quantile(depth_raw[mask], y, [0.25, 0.5, 0.75])
        nexp_q = np.exp(dq)

        r = {
            "fit_label": fit_label,
            "epoch": int(e),
            "date_utc": date_iso_by_epoch.get(int(e), ""),
            "N": int(N),
            "n_pix_unmasked": n_pix_unmasked,
            "n_pix_active": n_pix_active,
            "n_pix_zero": n_pix_zero,
            "f_pix_active": f_pix_active,
            "depth_log_nexp_q25": float(dq[0]),
            "depth_log_nexp_q50": float(dq[1]),
            "depth_log_nexp_q75": float(dq[2]),
            "depth_nexp_q25": float(nexp_q[0]),
            "depth_nexp_q50": float(nexp_q[1]),
            "depth_nexp_q75": float(nexp_q[2]),
            "parent_frac_q25": float(fq[0]),
            "parent_frac_q50": float(fq[1]),
            "parent_frac_q75": float(fq[2]),
            "D_dip_only": float(md["D"]),
            "sigma_D_dip_only": float(md["sigma_D"]),
            "l_dip_only_deg": float(md["l_deg"]),
            "b_dip_only_deg": float(md["b_deg"]),
            "sep_cmb_dip_only_deg": float(md["sep_to_cmb_deg"]),
            "D_full": float(mf["D"]),
            "sigma_D_full": float(mf["sigma_D"]),
            "l_full_deg": float(mf["l_deg"]),
            "b_full_deg": float(mf["b_deg"]),
            "sep_cmb_full_deg": float(mf["sep_to_cmb_deg"]),
            "dev_null": float(dev0),
            "dev_dip_only": float(devd),
            "dev_nuis_only": float(devn),
            "dev_full": float(devf),
            "frac_dev_explained_nuis_only": float((dev0 - devn) / dev0) if dev0 > 0 else float("nan"),
            "frac_dev_explained_dip_after_nuis": float((devn - devf) / devn) if devn > 0 else float("nan"),
            "frac_dev_explained_nuis_after_dip": float((devd - devf) / devd) if devd > 0 else float("nan"),
            "beta_dip_only": [float(x) for x in bd],
            "beta_nuis_only": [float(x) for x in bn],
            "beta_full": [float(x) for x in bf],
            "cov_full": None if cf is None else np.asarray(cf, dtype=np.float64).tolist(),
        }
        rows.append(r)

    # Constant-amplitude tests.
    D_d = np.array([rr.get("D_dip_only", np.nan) for rr in rows], dtype=float)
    s_d = np.array([rr.get("sigma_D_dip_only", np.nan) for rr in rows], dtype=float)
    D_f = np.array([rr.get("D_full", np.nan) for rr in rows], dtype=float)
    s_f = np.array([rr.get("sigma_D_full", np.nan) for rr in rows], dtype=float)

    summary = {
        "fit_label": fit_label,
        "n_epoch_rows": int(len(rows)),
        "constant_D_dip_only": constant_amplitude_chi2(D_d, s_d),
        "constant_D_full": constant_amplitude_chi2(D_f, s_f),
        "n_unmasked_pix": int(np.count_nonzero(mask)),
    }

    # Correlations between dipole and nuisance coefficients across epochs.
    B = []
    G = []
    cov_abs = []
    for rr in rows:
        bf = rr.get("beta_full")
        if not isinstance(bf, list) or len(bf) < 8:
            continue
        B.append([bf[1], bf[2], bf[3]])
        G.append([bf[4], bf[5], bf[6], bf[7]])
        cf = rr.get("cov_full")
        if cf is None:
            continue
        c = np.asarray(cf, dtype=np.float64)
        if c.shape != (8, 8):
            continue
        s = np.sqrt(np.clip(np.diag(c), 1e-30, np.inf))
        corr = c / (s[:, None] * s[None, :])
        sub = np.abs(corr[np.ix_([1, 2, 3], [4, 5, 6, 7])])
        cov_abs.append(float(np.nanmean(sub)))

    B = np.asarray(B, dtype=np.float64) if B else np.empty((0, 3))
    G = np.asarray(G, dtype=np.float64) if G else np.empty((0, 4))

    corr_table = []
    names_b = ["bx", "by", "bz"]
    names_g = ["abs_elat_z", "sin_elon_z", "cos_elon_z", "depth_z"]
    if B.shape[0] >= 3 and G.shape[0] == B.shape[0]:
        for i, bn in enumerate(names_b):
            for j, gn in enumerate(names_g):
                x = B[:, i]
                y = G[:, j]
                sx = float(np.std(x))
                sy = float(np.std(y))
                if sx <= 0.0 or sy <= 0.0:
                    corr = float("nan")
                else:
                    corr = float(np.corrcoef(x, y)[0, 1])
                corr_table.append({"dipole_component": bn, "template_component": gn, "corr": corr})

    coef_corr = {
        "fit_label": fit_label,
        "epoch_count_used": int(B.shape[0]),
        "dipole_template_corr_across_epochs": corr_table,
        "mean_abs_cov_corr_dipole_vs_templates_within_epoch": (
            float(np.mean(cov_abs)) if cov_abs else float("nan")
        ),
    }
    return EpochFits(rows=rows, summary=summary, coef_corr=coef_corr)


def direction_stability(rows: list[dict[str, Any]], key_prefix: str, epochs: list[int]) -> dict[str, float]:
    vecs = []
    for rr in rows:
        if int(rr.get("epoch", -1)) not in epochs:
            continue
        if key_prefix == "dip_only":
            b = np.array(
                [
                    rr.get("beta_dip_only", [np.nan] * 4)[1],
                    rr.get("beta_dip_only", [np.nan] * 4)[2],
                    rr.get("beta_dip_only", [np.nan] * 4)[3],
                ],
                dtype=np.float64,
            )
        else:
            b = np.array(
                [
                    rr.get("beta_full", [np.nan] * 8)[1],
                    rr.get("beta_full", [np.nan] * 8)[2],
                    rr.get("beta_full", [np.nan] * 8)[3],
                ],
                dtype=np.float64,
            )
        n = float(np.linalg.norm(b))
        if np.isfinite(n) and n > 0:
            vecs.append(b / n)
    if len(vecs) < 2:
        return {"n": int(len(vecs)), "adjacent_cos_mean": float("nan"), "adjacent_cos_median": float("nan")}
    arr = np.asarray(vecs, dtype=np.float64)
    cs = np.sum(arr[:-1] * arr[1:], axis=1)
    return {
        "n": int(arr.shape[0]),
        "adjacent_cos_mean": float(np.mean(cs)),
        "adjacent_cos_median": float(np.median(cs)),
    }


def run_injection_suite(
    *,
    counts: np.ndarray,
    nhat_mask: np.ndarray,
    templates_mask: np.ndarray,
    fit_rows: list[dict[str, Any]],
    epochs: list[int],
    nsim: int,
    seed: int,
    max_iter: int = 300,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))

    # Use nuisance-only coefficients per epoch as the empirical systematics model.
    alpha = []
    gamma = []
    obs_D = []
    obs_sD = []
    full_b = []
    full_w = []
    for rr in fit_rows:
        if int(rr["epoch"]) not in epochs:
            continue
        bn = np.asarray(rr["beta_nuis_only"], dtype=np.float64)
        alpha.append(float(bn[0]))
        gamma.append(np.asarray(bn[1:5], dtype=np.float64))
        obs_D.append(float(rr["D_dip_only"]))
        obs_sD.append(float(rr["sigma_D_dip_only"]))
        bf = np.asarray(rr["beta_full"], dtype=np.float64)
        full_b.append(np.asarray(bf[1:4], dtype=np.float64))
        sf = float(rr["sigma_D_full"])
        full_w.append(0.0 if (not np.isfinite(sf) or sf <= 0) else 1.0 / (sf * sf))

    alpha = np.asarray(alpha, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    obs_D = np.asarray(obs_D, dtype=np.float64)
    obs_sD = np.asarray(obs_sD, dtype=np.float64)
    full_b = np.asarray(full_b, dtype=np.float64)
    full_w = np.asarray(full_w, dtype=np.float64)

    if np.count_nonzero(full_w > 0) >= 1:
        b_true = np.sum(full_b * full_w[:, None], axis=0) / np.sum(full_w)
    else:
        b_true = np.nanmean(full_b, axis=0)

    X_dip = np.column_stack([np.ones(nhat_mask.shape[0]), nhat_mask[:, 0], nhat_mask[:, 1], nhat_mask[:, 2]])
    D_obs_stats = constant_amplitude_chi2(obs_D, obs_sD)
    obs_range = float(D_obs_stats["D_range"])
    obs_chi2 = float(D_obs_stats["chi2"])

    range_vals = np.zeros(int(nsim), dtype=np.float64)
    chi2_vals = np.zeros(int(nsim), dtype=np.float64)

    beta_init_epochs = [None for _ in range(len(epochs))]

    for i in range(int(nsim)):
        D_sim = np.full(len(epochs), np.nan, dtype=np.float64)
        s_sim = np.full(len(epochs), np.nan, dtype=np.float64)
        for j in range(len(epochs)):
            eta = alpha[j] + (nhat_mask @ b_true) + (templates_mask @ gamma[j])
            mu = np.exp(np.clip(eta, -25.0, 25.0))
            y = rng.poisson(mu)
            beta, cov, _mu, _dev = fit_poisson_glm(X_dip, y.astype(np.float64), beta_init=beta_init_epochs[j], max_iter=max_iter)
            beta_init_epochs[j] = beta
            met = dipole_metrics(beta, cov, dip_idx=(1, 2, 3))
            D_sim[j] = float(met["D"])
            s_sim[j] = float(met["sigma_D"])
        st = constant_amplitude_chi2(D_sim, s_sim)
        range_vals[i] = float(st["D_range"])
        chi2_vals[i] = float(st["chi2"])
        if (i + 1) % max(20, int(nsim) // 10) == 0:
            print(f"[injection] {i+1}/{nsim}", flush=True)

    p_range = float(np.mean(range_vals >= obs_range))
    p_chi2 = float(np.mean(chi2_vals >= obs_chi2))
    return {
        "nsim": int(nsim),
        "seed": int(seed),
        "b_true": [float(x) for x in b_true],
        "observed": {
            "D_range": obs_range,
            "chi2_const": obs_chi2,
            "stats": D_obs_stats,
        },
        "sim": {
            "range_mean": float(np.mean(range_vals)),
            "range_std": float(np.std(range_vals)),
            "chi2_mean": float(np.mean(chi2_vals)),
            "chi2_std": float(np.std(chi2_vals)),
            "p_range_ge_obs": p_range,
            "p_chi2_ge_obs": p_chi2,
        },
        "range_values": range_vals.tolist(),
        "chi2_values": chi2_vals.tolist(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fieldnames}
            w.writerow(out)


def make_figures(
    *,
    fig_dir: Path,
    base_rows: list[dict[str, Any]],
    common_rows: list[dict[str, Any]],
    common_depth_rows: list[dict[str, Any]],
    inj: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
        return np.array([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)

    def earr(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.array([int(r.get("epoch", -1)) for r in rows], dtype=np.int64)

    e = earr(base_rows)
    d0 = arr(base_rows, "D_dip_only")
    s0 = arr(base_rows, "sigma_D_dip_only")
    df = arr(base_rows, "D_full")
    sf = arr(base_rows, "sigma_D_full")

    ec = earr(common_depth_rows)
    dcd = arr(common_depth_rows, "D_full")
    scd = arr(common_depth_rows, "sigma_D_full")

    plt.figure(figsize=(8.4, 4.8))
    plt.errorbar(e, d0, yerr=s0, fmt="o-", ms=4, lw=1.2, label="Dipole-only (base mask)")
    plt.errorbar(e, df, yerr=sf, fmt="s-", ms=3.8, lw=1.1, label="Nuisance-controlled (base mask)")
    plt.errorbar(ec, dcd, yerr=scd, fmt="^-", ms=3.8, lw=1.1, label="Nuisance-controlled (common+depth)")
    plt.xlabel("Epoch")
    plt.ylabel("Dipole amplitude D")
    plt.title("Epoch amplitudes: baseline vs nuisance-controlled vs common-support")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "D_vs_epoch_model_comparison.png", dpi=180)
    plt.close()

    # Attribution fractions.
    frac_nuis = arr(base_rows, "frac_dev_explained_nuis_only")
    frac_dip_after_nuis = arr(base_rows, "frac_dev_explained_dip_after_nuis")
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(e, frac_nuis, "o-", lw=1.2, ms=4, label="(dev_null-dev_nuis)/dev_null")
    plt.plot(e, frac_dip_after_nuis, "s-", lw=1.1, ms=3.8, label="(dev_nuis-dev_full)/dev_nuis")
    plt.xlabel("Epoch")
    plt.ylabel("Fractional deviance reduction")
    plt.title("Template attribution by epoch")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "epoch_template_attribution.png", dpi=180)
    plt.close()

    # Injection histogram.
    rvals = np.asarray(inj["range_values"], dtype=np.float64)
    obs_range = float(inj["observed"]["D_range"])
    plt.figure(figsize=(7.2, 4.6))
    plt.hist(rvals, bins=40, alpha=0.8, color="#4477aa")
    plt.axvline(obs_range, color="#cc3311", lw=2, label=f"Observed range={obs_range:.4f}")
    plt.xlabel("Recovered dipole-only D range across epochs")
    plt.ylabel("Count")
    plt.title("Injection through measured epoch systematics")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "injection_range_hist.png", dpi=180)
    plt.close()

    # Comparability plot.
    n = arr(base_rows, "N")
    fpx = arr(base_rows, "f_pix_active")
    dmed = arr(base_rows, "depth_nexp_q50")
    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    ax1.plot(e, n, "o-", lw=1.2, ms=4, color="#004488", label="N selected")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("N selected", color="#004488")
    ax1.tick_params(axis="y", labelcolor="#004488")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(e, fpx, "s--", lw=1.1, ms=3.6, color="#bb5566", label="active-pixel fraction")
    ax2.plot(e, dmed, "^-.", lw=1.1, ms=3.6, color="#228833", label="median Nexp")
    ax2.set_ylabel("Coverage/depth metrics")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=8, loc="upper right")
    plt.title("Per-epoch comparability audit")
    fig.tight_layout()
    fig.savefig(fig_dir / "epoch_comparability_audit.png", dpi=180)
    plt.close(fig)

    # Direction separation.
    sep_d = arr(base_rows, "sep_cmb_dip_only_deg")
    sep_f = arr(base_rows, "sep_cmb_full_deg")
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(e, sep_d, "o-", lw=1.2, ms=4, label="Dipole-only")
    plt.plot(e, sep_f, "s-", lw=1.1, ms=3.8, label="Nuisance-controlled")
    plt.xlabel("Epoch")
    plt.ylabel("Sign-invariant sep to CMB axis [deg]")
    plt.title("Direction stability diagnostic")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "epoch_direction_stability.png", dpi=180)
    plt.close()


def write_master_report(
    *,
    path: Path,
    out_json: dict[str, Any],
) -> None:
    s_base = out_json["summaries"]["base"]
    s_common = out_json["summaries"]["common_support"]
    s_cdepth = out_json["summaries"]["common_support_depth"]
    inj = out_json["injection"]
    ddir = out_json["direction_stability"]
    mdef = out_json["common_support"]["matched_depth_definition"]
    fit_label = out_json["meta"]["fit_label"]

    text = f"""# CatWISE-parent epoch systematics suite ({fit_label})

Date: {utc_date()} (UTC)

This report executes the six-item robustness stack requested for the epoch-sliced CatWISE-parent unWISE test.

## 1) Epoch re-fit with nuisance controls

Templates in the full model:
- `abs(elat)_z`
- `sin(elon)_z`
- `cos(elon)_z`
- `depth_z` (from `{out_json["config"]["depth_map_fits"]}`)

Constant-`D` test (epochs 0–{out_json["config"]["epochs_max"]}, base mask):
- Dipole-only: `chi2={s_base["constant_D_dip_only"]["chi2"]:.3f}` for `dof={s_base["constant_D_dip_only"]["dof"]}` (`p={s_base["constant_D_dip_only"]["p_value"]:.3e}`)
- Nuisance-controlled: `chi2={s_base["constant_D_full"]["chi2"]:.3f}` for `dof={s_base["constant_D_full"]["dof"]}` (`p={s_base["constant_D_full"]["p_value"]:.3e}`)

## 2) Per-epoch comparability audit

Wrote `data/epoch_comparability_table_base.csv` with:
- `N`, active-pixel fraction, zero-pixel count
- weighted depth quantiles (`logNexp` and `Nexp`)
- parent-selection-fraction quantiles per pixel

## 3) Common-support / matched-depth subset

Common-support mask:
- seen+parent pixels: `{out_json["common_support"]["n_seen_parent_pix"]}`
- pixels with counts > 0 in every epoch 0–{out_json["config"]["epochs_max"]}: `{out_json["common_support"]["n_common_support_pix"]}`

Matched-depth refinement on common support:
- overlap depth interval: `[ {mdef["depth_q_lo"]:.6f}, {mdef["depth_q_hi"]:.6f} ]`
- retained pixels: `{out_json["common_support"]["n_common_support_depth_pix"]}`

Constant-`D` test (common+depth mask):
- Dipole-only: `chi2={s_cdepth["constant_D_dip_only"]["chi2"]:.3f}` for `dof={s_cdepth["constant_D_dip_only"]["dof"]}` (`p={s_cdepth["constant_D_dip_only"]["p_value"]:.3e}`)
- Nuisance-controlled: `chi2={s_cdepth["constant_D_full"]["chi2"]:.3f}` for `dof={s_cdepth["constant_D_full"]["dof"]}` (`p={s_cdepth["constant_D_full"]["p_value"]:.3e}`)

## 4) Template attribution on residual maps

Wrote per-epoch attribution metrics:
- `frac_dev_explained_nuis_only = (dev_null - dev_nuis)/dev_null`
- `frac_dev_explained_dip_after_nuis = (dev_nuis - dev_full)/dev_nuis`

Dipole/template coefficient coupling:
- mean abs covariance-correlation (dipole vs templates, within-epoch): `{out_json["coef_correlation"]["base"]["mean_abs_cov_corr_dipole_vs_templates_within_epoch"]:.4f}`

## 5) Injection through measured epoch systematics

Injected a constant dipole through epoch-specific nuisance fits and recovered with dipole-only GLM (`nsim={inj["nsim"]}`):
- observed dipole-only range: `{inj["observed"]["D_range"]:.6f}`
- simulated `P(range >= observed) = {inj["sim"]["p_range_ge_obs"]:.3e}`
- simulated `P(chi2_const >= observed) = {inj["sim"]["p_chi2_ge_obs"]:.3e}`

## 6) Direction stability add-on

Adjacent-epoch cosine similarity (base mask, epochs 0–{out_json["config"]["epochs_max"]}):
- Dipole-only: mean `{ddir["base_dip_only"]["adjacent_cos_mean"]:.4f}`, median `{ddir["base_dip_only"]["adjacent_cos_median"]:.4f}`
- Nuisance-controlled: mean `{ddir["base_full"]["adjacent_cos_mean"]:.4f}`, median `{ddir["base_full"]["adjacent_cos_median"]:.4f}`

## Outputs

Data:
- `data/epoch_fits_base.csv`
- `data/epoch_fits_common_support.csv`
- `data/epoch_fits_common_support_depth.csv`
- `data/epoch_comparability_table_base.csv`
- `data/coef_correlation_base.json`
- `data/injection_summary.json`
- `data/summary.json`

Figures:
- `figures/D_vs_epoch_model_comparison.png`
- `figures/epoch_comparability_audit.png`
- `figures/epoch_template_attribution.png`
- `figures/injection_range_hist.png`
- `figures/epoch_direction_stability.png`
"""
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-outdir",
        default="outputs/epoch_dipole_time_domain_catwise_parent_20260204_232537UTC",
        help="Output directory from run_unwise_time_domain_catwise_epoch_dipole.py",
    )
    ap.add_argument(
        "--report-dir",
        default=None,
        help="Report output dir (default REPORTS/unwise_time_domain_catwise_epoch_systematics_suite).",
    )
    ap.add_argument("--catwise-catalog", default=CATWISE_DEFAULT)
    ap.add_argument("--exclude-mask-fits", default=EXCLUDE_DEFAULT)
    ap.add_argument("--depth-map-fits", default=DEPTH_DEFAULT)
    ap.add_argument("--epochs-max", type=int, default=15)
    ap.add_argument("--injection-nsim", type=int, default=1000)
    ap.add_argument("--injection-seed", type=int, default=20260220)
    args = ap.parse_args()

    run_outdir = Path(args.run_outdir)
    report_dir = Path(args.report_dir or "REPORTS/unwise_time_domain_catwise_epoch_systematics_suite")
    data_dir = report_dir / "data"
    fig_dir = report_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run_outdir / "run_config.json").read_text())
    summary = json.loads((run_outdir / "summary.json").read_text())
    counts = np.load(run_outdir / "counts_by_epoch.npy")
    with np.load(run_outdir / "parent_index.npz", allow_pickle=False) as z:
        parent_pix64 = np.asarray(z["parent_pix64"], dtype=np.int64)

    import healpy as hp
    from astropy.time import Time

    nside = int(cfg["nside"])
    npix = hp.nside2npix(nside)
    if counts.shape[1] != npix:
        raise RuntimeError(f"counts npix mismatch: {counts.shape[1]} vs {npix}")

    parent_counts = np.bincount(parent_pix64, minlength=npix).astype(np.int64)
    seen = build_seen_mask(
        nside=nside,
        catwise_catalog=Path(args.catwise_catalog),
        exclude_mask_fits=Path(args.exclude_mask_fits),
        b_cut_deg=float(cfg["b_cut_deg"]),
        w1cov_min=float(cfg["w1cov_min"]),
    )

    # Direction basis.
    xpix, ypix, zpix = hp.pix2vec(nside, np.arange(npix, dtype=np.int64), nest=False)
    nhat = np.column_stack([xpix, ypix, zpix]).astype(np.float64)

    depth_map = read_healpix_map_fits(Path(args.depth_map_fits))
    templates = build_templates(nside=nside, seen=seen, depth_map=depth_map)

    epochs_max = int(args.epochs_max)
    epochs = [int(e) for e in range(min(counts.shape[0], epochs_max + 1))]

    date_iso_by_epoch: dict[int, str] = {}
    for r in summary.get("epochs", []):
        e = int(r["epoch"])
        mjd = float(r.get("mjd_mean", np.nan))
        if np.isfinite(mjd):
            date_iso_by_epoch[e] = Time(mjd, format="mjd").utc.isot
        else:
            date_iso_by_epoch[e] = ""

    print("Running base-mask epoch fits...", flush=True)
    base_fit = run_epoch_fits(
        counts=counts,
        parent_counts=parent_counts,
        seen_mask=seen,
        nhat=nhat,
        templates=templates,
        epochs=epochs,
        date_iso_by_epoch=date_iso_by_epoch,
        fit_label="base_mask",
    )

    # Common support and matched-depth support.
    seen_parent = seen & (parent_counts > 0)
    common_support = seen_parent & np.all(counts[epochs, :] > 0, axis=0)

    depth_raw = np.asarray(templates["depth_raw"], dtype=np.float64)
    q10_list = []
    q90_list = []
    for e in epochs:
        w = counts[e, common_support].astype(np.float64)
        d = depth_raw[common_support]
        q = weighted_quantile(d, w, [0.10, 0.90])
        q10_list.append(float(q[0]))
        q90_list.append(float(q[1]))
    qlo = float(np.nanmax(np.asarray(q10_list, dtype=np.float64)))
    qhi = float(np.nanmin(np.asarray(q90_list, dtype=np.float64)))
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        q = weighted_quantile(depth_raw[common_support], parent_counts[common_support].astype(np.float64), [0.05, 0.95])
        qlo, qhi = float(q[0]), float(q[1])
    common_support_depth = common_support & np.isfinite(depth_raw) & (depth_raw >= qlo) & (depth_raw <= qhi)

    print("Running common-support epoch fits...", flush=True)
    common_fit = run_epoch_fits(
        counts=counts,
        parent_counts=parent_counts,
        seen_mask=common_support,
        nhat=nhat,
        templates=templates,
        epochs=epochs,
        date_iso_by_epoch=date_iso_by_epoch,
        fit_label="common_support",
    )

    print("Running common-support + matched-depth epoch fits...", flush=True)
    common_depth_fit = run_epoch_fits(
        counts=counts,
        parent_counts=parent_counts,
        seen_mask=common_support_depth,
        nhat=nhat,
        templates=templates,
        epochs=epochs,
        date_iso_by_epoch=date_iso_by_epoch,
        fit_label="common_support_depth",
    )

    print("Running injection-through-systematics Monte Carlo...", flush=True)
    inj = run_injection_suite(
        counts=counts,
        nhat_mask=nhat[seen],
        templates_mask=np.column_stack(
            [
                templates["abs_elat_z"][seen],
                templates["sin_elon_z"][seen],
                templates["cos_elon_z"][seen],
                templates["depth_z"][seen],
            ]
        ),
        fit_rows=base_fit.rows,
        epochs=epochs,
        nsim=int(args.injection_nsim),
        seed=int(args.injection_seed),
    )

    print("Computing direction-stability summaries...", flush=True)
    direction = {
        "base_dip_only": direction_stability(base_fit.rows, "dip_only", epochs),
        "base_full": direction_stability(base_fit.rows, "full", epochs),
        "common_dip_only": direction_stability(common_fit.rows, "dip_only", epochs),
        "common_full": direction_stability(common_fit.rows, "full", epochs),
        "common_depth_dip_only": direction_stability(common_depth_fit.rows, "dip_only", epochs),
        "common_depth_full": direction_stability(common_depth_fit.rows, "full", epochs),
    }

    # Write data products.
    fit_fields = [
        "fit_label",
        "epoch",
        "date_utc",
        "N",
        "n_pix_unmasked",
        "n_pix_active",
        "n_pix_zero",
        "f_pix_active",
        "depth_log_nexp_q25",
        "depth_log_nexp_q50",
        "depth_log_nexp_q75",
        "depth_nexp_q25",
        "depth_nexp_q50",
        "depth_nexp_q75",
        "parent_frac_q25",
        "parent_frac_q50",
        "parent_frac_q75",
        "D_dip_only",
        "sigma_D_dip_only",
        "l_dip_only_deg",
        "b_dip_only_deg",
        "sep_cmb_dip_only_deg",
        "D_full",
        "sigma_D_full",
        "l_full_deg",
        "b_full_deg",
        "sep_cmb_full_deg",
        "dev_null",
        "dev_dip_only",
        "dev_nuis_only",
        "dev_full",
        "frac_dev_explained_nuis_only",
        "frac_dev_explained_dip_after_nuis",
        "frac_dev_explained_nuis_after_dip",
    ]
    write_csv(data_dir / "epoch_fits_base.csv", base_fit.rows, fit_fields)
    write_csv(data_dir / "epoch_fits_common_support.csv", common_fit.rows, fit_fields)
    write_csv(data_dir / "epoch_fits_common_support_depth.csv", common_depth_fit.rows, fit_fields)
    write_csv(data_dir / "epoch_comparability_table_base.csv", base_fit.rows, fit_fields[:17])

    (data_dir / "coef_correlation_base.json").write_text(json.dumps(base_fit.coef_corr, indent=2) + "\n")
    (data_dir / "coef_correlation_common_support.json").write_text(json.dumps(common_fit.coef_corr, indent=2) + "\n")
    (data_dir / "coef_correlation_common_support_depth.json").write_text(
        json.dumps(common_depth_fit.coef_corr, indent=2) + "\n"
    )
    (data_dir / "injection_summary.json").write_text(json.dumps(inj, indent=2) + "\n")

    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "date_utc": utc_date(),
            "fit_label": f"epoch_systematics_suite_{utc_tag()}",
            "run_outdir": str(run_outdir),
            "report_dir": str(report_dir),
        },
        "config": {
            "nside": nside,
            "epochs_max": epochs_max,
            "epochs_used": epochs,
            "depth_map_fits": str(args.depth_map_fits),
            "injection_nsim": int(args.injection_nsim),
            "injection_seed": int(args.injection_seed),
        },
        "common_support": {
            "n_seen_pix": int(np.count_nonzero(seen)),
            "n_seen_parent_pix": int(np.count_nonzero(seen_parent)),
            "n_common_support_pix": int(np.count_nonzero(common_support)),
            "n_common_support_depth_pix": int(np.count_nonzero(common_support_depth)),
            "matched_depth_definition": {
                "depth_q_lo": qlo,
                "depth_q_hi": qhi,
            },
        },
        "summaries": {
            "base": base_fit.summary,
            "common_support": common_fit.summary,
            "common_support_depth": common_depth_fit.summary,
        },
        "coef_correlation": {
            "base": base_fit.coef_corr,
            "common_support": common_fit.coef_corr,
            "common_support_depth": common_depth_fit.coef_corr,
        },
        "injection": inj,
        "direction_stability": direction,
    }
    (data_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    print("Rendering figures...", flush=True)
    make_figures(
        fig_dir=fig_dir,
        base_rows=base_fit.rows,
        common_rows=common_fit.rows,
        common_depth_rows=common_depth_fit.rows,
        inj=inj,
    )

    write_master_report(path=report_dir / "master_report.md", out_json=out)
    print(f"Wrote report -> {report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
