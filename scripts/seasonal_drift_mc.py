#!/usr/bin/env python3
"""
Correlated-cut Monte Carlo for CatWISE dipole direction "drift" vs faint cut.

Motivation
----------
Secrest noted that (i) W1 cuts are *nested* (W1<15.7 is a subset of W1<15.8),
so the cut-to-cut points are correlated, and (ii) apparent smooth direction
changes can therefore occur even under a null.

This script implements the robust version of that test:
  - build *differential* W1-bin HEALPix counts on the Secrest mask
  - fit a baseline mean intensity per bin with NO dipole (Poisson GLM on abs_elat)
  - simulate independent Poisson bin counts and cumulatively sum them to form
    correlated W1<cut maps
  - for each mock, fit a dipole per cut (Poisson GLM) and compute drift metrics

Optional "seasonal" injection
-----------------------------
To mimic scan/season-linked selection, inject ecliptic-longitude templates
sin(elon), cos(elon) into *per-bin* log-intensity. When --lon-from-scan is used,
we convert the per-cut coefficients (from a real-data cumulative fit) into a
per-bin set of coefficients so the *cumulative* injected coefficient matches
the scan values in a small-amplitude (linearized) sense.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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


def unitvec_from_lb(l_deg: float, b_deg: float) -> np.ndarray:
    l = math.radians(float(l_deg) % 360.0)
    b = math.radians(float(b_deg))
    return np.array([math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b)], dtype=float)


def vec_to_lb(vec: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def ang_sep_deg(u: np.ndarray, v: np.ndarray, *, axis: bool) -> float:
    u = np.asarray(u, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    c = float(np.dot(u, v))
    if axis:
        c = abs(c)
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(math.acos(c)))


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
    """Implements mask_zeros + exclusion discs + Galactic latitude cut."""
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
    beta_init: np.ndarray | None,
) -> np.ndarray:
    """Poisson GLM (log link) MLE via L-BFGS."""
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


def parse_grid(spec: str) -> np.ndarray:
    lo_s, hi_s, step_s = (s.strip() for s in str(spec).split(","))
    lo = float(lo_s)
    hi = float(hi_s)
    step = float(step_s)
    if step <= 0:
        raise ValueError("grid step must be > 0")
    cuts = np.round(np.arange(lo, hi + 0.5 * step, step), 10)
    if cuts.size < 2:
        raise ValueError("grid produced too few cuts")
    return cuts


def _lon_coeffs_by_cut(scan_json: Path) -> dict[float, tuple[float, float]]:
    obj = json.loads(scan_json.read_text())
    out: dict[float, tuple[float, float]] = {}
    for r in obj.get("rows", []):
        w1_cut = float(r["w1_cut"])
        names = list(r.get("template_names", []))
        beta = list(r.get("beta_hat", []))
        tmpl_beta = beta[4:]
        if len(tmpl_beta) != len(names):
            continue
        sin_val = None
        cos_val = None
        for name, val in zip(names, tmpl_beta, strict=True):
            if name in {"sin_elon_z", "sin_lambda_z"}:
                sin_val = float(val)
            elif name in {"cos_elon_z", "cos_lambda_z"}:
                cos_val = float(val)
        if sin_val is None or cos_val is None:
            continue
        out[w1_cut] = (sin_val, cos_val)
    if not out:
        raise ValueError(f"{scan_json}: no sin/cos(ecliptic lon) coefficients found")
    return out


def cum_to_bin_coeffs(
    *,
    cuts: np.ndarray,
    N_bin: np.ndarray,
    sin_cum: np.ndarray,
    cos_cum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert cumulative coefficients -> per-bin coefficients under a small-amplitude linearization:

      y_cum ≈ Σ_k N_k (1 + a_k t)  => a_cum(K) = (Σ_{k<=K} N_k a_k) / (Σ_{k<=K} N_k)

    Solve for a_k recursively so that the implied a_cum matches the target sequence.
    """
    cuts = np.asarray(cuts, dtype=float)
    N_bin = np.asarray(N_bin, dtype=float)
    sin_cum = np.asarray(sin_cum, dtype=float)
    cos_cum = np.asarray(cos_cum, dtype=float)
    if cuts.ndim != 1 or N_bin.shape != cuts.shape:
        raise ValueError("cuts and N_bin must have the same 1D shape")
    if sin_cum.shape != cuts.shape or cos_cum.shape != cuts.shape:
        raise ValueError("sin_cum/cos_cum shape mismatch")

    N_cum = np.cumsum(N_bin)
    if np.any(N_bin <= 0) or np.any(N_cum <= 0):
        raise ValueError("non-positive N_bin encountered; cannot convert coefficients")

    sin_bin = np.zeros_like(sin_cum)
    cos_bin = np.zeros_like(cos_cum)
    sin_bin[0] = sin_cum[0]
    cos_bin[0] = cos_cum[0]
    for k in range(1, cuts.size):
        sin_bin[k] = (N_cum[k] * sin_cum[k] - N_cum[k - 1] * sin_cum[k - 1]) / N_bin[k]
        cos_bin[k] = (N_cum[k] * cos_cum[k] - N_cum[k - 1] * cos_cum[k - 1]) / N_bin[k]
    return sin_bin, cos_bin


def drift_metrics(directions: np.ndarray, *, axis: bool) -> dict[str, float]:
    """
    directions: (n_cuts, 3) unit vectors.
    """
    u = np.asarray(directions, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3:
        raise ValueError("directions must have shape (n,3)")
    n = u.shape[0]
    if n < 2:
        return {"path_len_deg": float("nan"), "end_to_end_deg": float("nan"), "max_pair_deg": float("nan")}

    # step angles
    dots_step = np.sum(u[:-1] * u[1:], axis=1)
    if axis:
        dots_step = np.abs(dots_step)
    dots_step = np.clip(dots_step, -1.0, 1.0)
    steps = np.degrees(np.arccos(dots_step))
    path_len = float(np.sum(steps))

    # end-to-end
    dot_ee = float(np.dot(u[0], u[-1]))
    if axis:
        dot_ee = abs(dot_ee)
    dot_ee = float(np.clip(dot_ee, -1.0, 1.0))
    end_to_end = float(np.degrees(math.acos(dot_ee)))

    # max pair
    dots = u @ u.T
    if axis:
        dots = np.abs(dots)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    max_pair = float(np.max(ang[np.triu_indices(n, k=1)]))
    return {
        "path_len_deg": path_len,
        "end_to_end_deg": end_to_end,
        "max_pair_deg": max_pair,
        "median_step_deg": float(np.median(steps)),
        "max_step_deg": float(np.max(steps)),
    }


# Globals for forked workers (avoid pickling big arrays repeatedly).
G_LAM_BINS: np.ndarray | None = None  # (n_bins, n_seen)
G_X_BASE: np.ndarray | None = None  # (n_seen, p_base)
G_X_LON: np.ndarray | None = None  # (n_seen, p_lon) or None
G_MAX_ITER: int | None = None
G_INCLUDE_LON_FIT: bool | None = None


def _worker_init(lam_bins: np.ndarray, X_base: np.ndarray, X_lon: np.ndarray | None, max_iter: int, include_lon_fit: bool) -> None:
    global G_LAM_BINS, G_X_BASE, G_X_LON, G_MAX_ITER, G_INCLUDE_LON_FIT
    G_LAM_BINS = lam_bins
    G_X_BASE = X_base
    G_X_LON = X_lon
    G_MAX_ITER = int(max_iter)
    G_INCLUDE_LON_FIT = bool(include_lon_fit)


def _simulate_batch(args: tuple[int, int, bool]) -> dict[str, Any]:
    """
    Returns:
      - drift arrays for baseline fit
      - optional drift arrays for lon fit
      - per-cut sums for bvec and angles
    """
    n_sims, seed, axis = args
    lam_bins = np.asarray(G_LAM_BINS)
    X_base = np.asarray(G_X_BASE)
    X_lon = None if G_X_LON is None else np.asarray(G_X_LON)
    max_iter = int(G_MAX_ITER)
    include_lon_fit = bool(G_INCLUDE_LON_FIT)

    rng = np.random.default_rng(int(seed))
    n_bins, n_seen = lam_bins.shape

    drift_path = np.empty(int(n_sims), dtype=float)
    drift_end = np.empty(int(n_sims), dtype=float)
    drift_max = np.empty(int(n_sims), dtype=float)

    drift_path_lon = np.empty(int(n_sims), dtype=float) if include_lon_fit else None
    drift_end_lon = np.empty(int(n_sims), dtype=float) if include_lon_fit else None
    drift_max_lon = np.empty(int(n_sims), dtype=float) if include_lon_fit else None

    sum_b = np.zeros((n_bins, 3), dtype=float)
    sum_b_lon = np.zeros((n_bins, 3), dtype=float) if include_lon_fit else None

    for i in range(int(n_sims)):
        cum = np.zeros(n_seen, dtype=float)

        beta_prev = None
        beta_prev_lon = None
        dirs = np.empty((n_bins, 3), dtype=float)
        dirs_lon = np.empty((n_bins, 3), dtype=float) if include_lon_fit else None

        for k in range(n_bins):
            yk = rng.poisson(lam_bins[k]).astype(float)
            cum += yk

            beta = fit_poisson_glm(X_base, cum, offset=None, max_iter=max_iter, beta_init=beta_prev)
            beta_prev = beta
            bvec = np.asarray(beta[1:4], dtype=float)
            nb = float(np.linalg.norm(bvec))
            if nb <= 0 or not np.isfinite(nb):
                dirs[k] = np.array([np.nan, np.nan, np.nan])
            else:
                dirs[k] = bvec / nb
            sum_b[k] += bvec

            if include_lon_fit:
                assert X_lon is not None
                beta_l = fit_poisson_glm(X_lon, cum, offset=None, max_iter=max_iter, beta_init=beta_prev_lon)
                beta_prev_lon = beta_l
                bvec_l = np.asarray(beta_l[1:4], dtype=float)
                nl = float(np.linalg.norm(bvec_l))
                if nl <= 0 or not np.isfinite(nl):
                    dirs_lon[k] = np.array([np.nan, np.nan, np.nan])
                else:
                    dirs_lon[k] = bvec_l / nl
                sum_b_lon[k] += bvec_l

        m = drift_metrics(dirs, axis=bool(axis))
        drift_path[i] = float(m["path_len_deg"])
        drift_end[i] = float(m["end_to_end_deg"])
        drift_max[i] = float(m["max_pair_deg"])

        if include_lon_fit:
            ml = drift_metrics(dirs_lon, axis=bool(axis))
            drift_path_lon[i] = float(ml["path_len_deg"])
            drift_end_lon[i] = float(ml["end_to_end_deg"])
            drift_max_lon[i] = float(ml["max_pair_deg"])

    out: dict[str, Any] = {
        "n": int(n_sims),
        "drift_path": drift_path,
        "drift_end": drift_end,
        "drift_max": drift_max,
        "sum_b": sum_b,
    }
    if include_lon_fit:
        out.update(
            {
                "drift_path_lon": drift_path_lon,
                "drift_end_lon": drift_end_lon,
                "drift_max_lon": drift_max_lon,
                "sum_b_lon": sum_b_lon,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Correlated-cut drift Monte Carlo with optional seasonal injection.")
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
    ap.add_argument("--w1-grid", default="15.5,16.6,0.05", help="start,stop,step (inclusive).")

    ap.add_argument("--dipole-amp", type=float, default=0.0, help="Injected dipole amplitude in log-intensity.")
    ap.add_argument("--dipole-axis-lb", default="264.021,48.253", help="Injected dipole axis (l,b) in degrees.")

    ap.add_argument(
        "--lon-from-scan",
        default=None,
        help="Scan JSON (cumulative fit) providing per-cut sin/cos(elon) coefficients to inject.",
    )
    ap.add_argument("--lon-sin", type=float, default=0.0, help="Constant injected sin(elon)_z coefficient.")
    ap.add_argument("--lon-cos", type=float, default=0.0, help="Constant injected cos(elon)_z coefficient.")
    ap.add_argument("--lon-scale", type=float, default=1.0, help="Overall multiplier on injected lon coefficients.")

    ap.add_argument("--n-mocks", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument(
        "--axis-metric",
        action="store_true",
        help="Use axis angles (sign-invariant acos(|dot|)) for drift metrics instead of vector angles.",
    )
    ap.add_argument("--include-lon-fit", action="store_true", help="Also fit dipole+abs_elat+sin/cos(elon) per cut.")
    ap.add_argument("--n-proc", type=int, default=0, help="0 => use (CPU count - 1).")

    ap.add_argument("--real-scan-json", default=None, help="Optional real-data scan JSON to report observed drift metrics.")

    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits

    outdir = Path(args.outdir or f"outputs/seasonal_drift_mc_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    cuts = parse_grid(str(args.w1_grid))

    # Load catalog columns.
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
    ipix_base = hp.ang2pix(nside, np.deg2rad(90.0 - b[base]), np.deg2rad(l[base]), nest=False).astype(np.int64)

    secrest = build_secrest_mask(
        nside=nside,
        ipix_base=ipix_base,
        exclude_mask_fits=str(args.exclude_mask_fits) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )
    seen = secrest.seen
    n_seen = int(np.sum(seen))

    lon_pix, lat_pix = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    pix_unit = lb_to_unitvec(lon_pix, lat_pix)
    n_unit = pix_unit[seen]

    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic").barycentricmeanecliptic
    elon = (np.asarray(sc_pix.lon.deg, dtype=float) % 360.0)
    elat = np.asarray(sc_pix.lat.deg, dtype=float)
    abs_elat = np.abs(elat)
    abs_elat_z = zscore(abs_elat, seen)[seen]
    sin_z = zscore(np.sin(np.deg2rad(elon)), seen)[seen]
    cos_z = zscore(np.cos(np.deg2rad(elon)), seen)[seen]

    # Injected dipole in log-intensity.
    inj_l, inj_b = (float(x) for x in str(args.dipole_axis_lb).split(","))
    inj_axis = unitvec_from_lb(inj_l, inj_b)
    b_inj = float(args.dipole_amp) * inj_axis
    dip_term = n_unit @ b_inj
    dip_factor = np.exp(np.clip(dip_term, -25.0, 25.0))

    # Differential-bin counts and baseline mu-hat per bin (NO dipole).
    # Bin 0 is (w1 <= cuts[0]); bin k>0 is (cuts[k-1] < w1 <= cuts[k]).
    X_mu = np.column_stack([np.ones(n_seen, dtype=float), abs_elat_z])
    base_lam_bins = np.zeros((cuts.size, n_seen), dtype=float)
    N_bin = np.zeros(cuts.size, dtype=float)

    for k, cut in enumerate(cuts.tolist()):
        if k == 0:
            sel = base & (w1 <= float(cut))
        else:
            sel = base & (w1 > float(cuts[k - 1])) & (w1 <= float(cut))
        ipix = hp.ang2pix(nside, np.deg2rad(90.0 - b[sel]), np.deg2rad(l[sel]), nest=False).astype(np.int64)
        counts = np.bincount(ipix, minlength=npix).astype(float)[seen]
        beta_mu = fit_poisson_glm(X_mu, counts, offset=None, max_iter=int(args.max_iter), beta_init=None)
        log_mu = np.clip(X_mu @ beta_mu, -25.0, 25.0)
        lam = np.exp(log_mu)
        base_lam_bins[k] = lam
        N_bin[k] = float(np.sum(lam))

    # Optional seasonal injection coefficients per bin.
    sin_bin = np.zeros_like(cuts)
    cos_bin = np.zeros_like(cuts)
    if args.lon_from_scan is not None:
        by_cut = _lon_coeffs_by_cut(Path(str(args.lon_from_scan)))
        sin_cum = np.array([by_cut[float(c)][0] for c in cuts.tolist()], dtype=float)
        cos_cum = np.array([by_cut[float(c)][1] for c in cuts.tolist()], dtype=float)
        sin_bin, cos_bin = cum_to_bin_coeffs(cuts=cuts, N_bin=N_bin, sin_cum=sin_cum, cos_cum=cos_cum)
    else:
        sin_bin[:] = float(args.lon_sin)
        cos_bin[:] = float(args.lon_cos)

    sin_bin = float(args.lon_scale) * sin_bin
    cos_bin = float(args.lon_scale) * cos_bin
    lon_factor_bins = np.exp(
        np.clip(sin_bin[:, None] * sin_z[None, :] + cos_bin[:, None] * cos_z[None, :], -25.0, 25.0)
    )

    lam_bins = base_lam_bins * dip_factor[None, :] * lon_factor_bins

    # Recovery design matrices.
    X_base = np.column_stack([np.ones(n_seen, dtype=float), n_unit[:, 0], n_unit[:, 1], n_unit[:, 2], abs_elat_z])
    X_lon = (
        np.column_stack(
            [
                np.ones(n_seen, dtype=float),
                n_unit[:, 0],
                n_unit[:, 1],
                n_unit[:, 2],
                abs_elat_z,
                sin_z,
                cos_z,
            ]
        )
        if bool(args.include_lon_fit)
        else None
    )

    # Observed drift metrics (optional).
    observed: dict[str, Any] | None = None
    if args.real_scan_json is not None:
        obj = json.loads(Path(str(args.real_scan_json)).read_text())
        rows = sorted(obj["rows"], key=lambda r: float(r["w1_cut"]))
        dirs = np.vstack(
            [
                unitvec_from_lb(float(r["dipole"]["l_hat_deg"]), float(r["dipole"]["b_hat_deg"]))
                for r in rows
                if float(r["w1_cut"]) in set(cuts.tolist())
            ]
        )
        observed = drift_metrics(dirs, axis=bool(args.axis_metric))

    # Parallel simulation.
    n_mocks = int(args.n_mocks)
    if n_mocks <= 0:
        raise SystemExit("n_mocks must be > 0")

    n_proc = int(args.n_proc)
    if n_proc <= 0:
        n_proc = max(1, (os.cpu_count() or 2) - 1)

    # Split mocks into roughly equal batches.
    batch = int(math.ceil(n_mocks / n_proc))
    batches = [batch] * n_proc
    batches[-1] = n_mocks - batch * (n_proc - 1)
    batches = [b for b in batches if b > 0]

    # Deterministic per-batch seeds.
    ss = np.random.SeedSequence(int(args.seed))
    child = ss.spawn(len(batches))
    seeds = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in child]

    # Use fork to share read-only arrays.
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=int(n_proc),
        initializer=_worker_init,
        initargs=(lam_bins, X_base, X_lon, int(args.max_iter), bool(args.include_lon_fit)),
        maxtasksperchild=200,
    ) as pool:
        res = pool.map(_simulate_batch, [(int(n), int(seed), bool(args.axis_metric)) for n, seed in zip(batches, seeds, strict=True)])

    drift_path = np.concatenate([r["drift_path"] for r in res])
    drift_end = np.concatenate([r["drift_end"] for r in res])
    drift_max = np.concatenate([r["drift_max"] for r in res])

    sum_b = np.sum([r["sum_b"] for r in res], axis=0)
    sum_b_lon = None
    drift_path_lon = drift_end_lon = drift_max_lon = None
    if bool(args.include_lon_fit):
        drift_path_lon = np.concatenate([r["drift_path_lon"] for r in res])
        drift_end_lon = np.concatenate([r["drift_end_lon"] for r in res])
        drift_max_lon = np.concatenate([r["drift_max_lon"] for r in res])
        sum_b_lon = np.sum([r["sum_b_lon"] for r in res], axis=0)

    def p_ge(x: np.ndarray, thr: float) -> float:
        x = np.asarray(x, dtype=float)
        return float((np.sum(x >= float(thr)) + 1.0) / (x.size + 1.0))

    pvals = None
    if observed is not None:
        pvals = {
            "path_len_p": p_ge(drift_path, float(observed["path_len_deg"])),
            "end_to_end_p": p_ge(drift_end, float(observed["end_to_end_deg"])),
            "max_pair_p": p_ge(drift_max, float(observed["max_pair_deg"])),
        }

    # Per-cut bias summary: D_of_b_mean (norm of mean b-vector).
    mean_b = sum_b / float(n_mocks)
    D_of_b_mean = np.linalg.norm(mean_b, axis=1)
    per_cut = [
        {
            "w1_cut": float(cuts[k]),
            "mean_b": [float(x) for x in mean_b[k]],
            "D_of_b_mean": float(D_of_b_mean[k]),
            "mean_dir_lb_deg": list(vec_to_lb(mean_b[k])),
            "lon_injected": {"sin_elon_z": float(sin_bin[k]), "cos_elon_z": float(cos_bin[k])},
        }
        for k in range(cuts.size)
    ]

    per_cut_lon = None
    if bool(args.include_lon_fit) and sum_b_lon is not None:
        mean_b_l = sum_b_lon / float(n_mocks)
        D_of_b_mean_l = np.linalg.norm(mean_b_l, axis=1)
        per_cut_lon = [
            {
                "w1_cut": float(cuts[k]),
                "mean_b": [float(x) for x in mean_b_l[k]],
                "D_of_b_mean": float(D_of_b_mean_l[k]),
                "mean_dir_lb_deg": list(vec_to_lb(mean_b_l[k])),
            }
            for k in range(cuts.size)
        ]

    meta: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalog": str(args.catalog),
        "exclude_mask_fits": str(args.exclude_mask_fits),
        "nside": int(args.nside),
        "w1cov_min": float(args.w1cov_min),
        "b_cut": float(args.b_cut),
        "w1_grid": str(args.w1_grid),
        "n_seen_pix": int(n_seen),
        "dipole_amp": float(args.dipole_amp),
        "dipole_axis_lb_deg": [inj_l, inj_b],
        "lon_from_scan": None if args.lon_from_scan is None else str(args.lon_from_scan),
        "lon_sin": float(args.lon_sin),
        "lon_cos": float(args.lon_cos),
        "lon_scale": float(args.lon_scale),
        "n_mocks": int(n_mocks),
        "seed": int(args.seed),
        "max_iter": int(args.max_iter),
        "axis_metric": bool(args.axis_metric),
        "include_lon_fit": bool(args.include_lon_fit),
        "n_proc": int(n_proc),
        "real_scan_json": None if args.real_scan_json is None else str(args.real_scan_json),
    }

    summary: dict[str, Any] = {
        "meta": meta,
        "observed": observed,
        "pvals_vs_observed": pvals,
        "per_cut_baseline_fit": per_cut,
    }
    if per_cut_lon is not None:
        summary["per_cut_lon_fit"] = per_cut_lon

    out_json = outdir / "seasonal_drift_mc_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    np.savez_compressed(
        outdir / "seasonal_drift_mc_metrics.npz",
        drift_path=drift_path,
        drift_end=drift_end,
        drift_max=drift_max,
        drift_path_lon=np.array([]) if drift_path_lon is None else drift_path_lon,
        drift_end_lon=np.array([]) if drift_end_lon is None else drift_end_lon,
        drift_max_lon=np.array([]) if drift_max_lon is None else drift_max_lon,
    )

    # Simple plots.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def hist(ax, x: np.ndarray, title: str, obs: float | None) -> None:
        ax.hist(x, bins=50, alpha=0.8, color="#4C72B0")
        if obs is not None:
            ax.axvline(float(obs), color="k", lw=1.6, ls="--", label="observed")
            ax.legend(fontsize=8, loc="best")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.8), dpi=160, constrained_layout=True)
    hist(ax[0], drift_path, "Drift path length (deg)", None if observed is None else float(observed["path_len_deg"]))
    hist(ax[1], drift_end, "End-to-end drift (deg)", None if observed is None else float(observed["end_to_end_deg"]))
    hist(ax[2], drift_max, "Max pair drift (deg)", None if observed is None else float(observed["max_pair_deg"]))
    fig.suptitle("Correlated-cut drift Monte Carlo (baseline fit)")
    fig.savefig(outdir / "seasonal_drift_mc_hist.png")
    plt.close(fig)

    # Bias summary vs cut.
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), dpi=160)
    ax.plot(cuts, D_of_b_mean, "o-", label="baseline fit: dipole+abs_elat")
    if per_cut_lon is not None:
        ax.plot(cuts, [r["D_of_b_mean"] for r in per_cut_lon], "o-", label="lon fit: dipole+abs_elat+sin/cos(elon)")
    ax.axhline(float(args.dipole_amp), color="k", lw=1.2, ls=":", label="injected dipole amp")
    ax.set_xlabel("W1_max")
    ax.set_ylabel("|mean(b_vec)|")
    ax.set_title("Mean recovered dipole bias vs cut")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "seasonal_drift_mc_bias.png")
    plt.close(fig)

    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

