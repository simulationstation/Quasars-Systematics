#!/usr/bin/env python3
"""
Case-closed maximal-nuisance suite for the CatWISE quasar dipole.

Goal
----
Produce an end-to-end, reviewer-resistant bundle showing that the *apparent* percent-level CatWISE
number-count dipole (and its instability under nested faint cuts) is consistent with survey systematics.

This script is intentionally "do it all at once":
  1) Baseline (dipole + minimal ecliptic latitude) scan vs W1_max.
  2) Maximal reasonable nuisance GLM scan vs W1_max (ridge-regularized).
  3) Orthogonalized-nuisance variant (templates projected orthogonal to {1, nx, ny, nz}).
  4) Held-out sky validation (GroupKFold on ecliptic-longitude wedges).
  5) Cross-catalog robustness via external completeness offset maps (Gaia qsocand and SDSS DR16Q).
  6) Leave-one-template-out attribution at a representative cut.

Outputs
-------
Writes a self-contained report bundle under REPORTS/ with:
  - master_report.md
  - data/summary.json
  - data/*.csv (coefficients + LOTO)
  - figures/*.png

Notes
-----
This is a map-level Poisson GLM on masked HEALPix pixel counts:
  N_p ~ Poisson(mu_p)
  log mu_p = beta0 + b·n_p + c·T_p + offset_p

where b is the dipole vector (D ≈ |b| for small dipoles).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CMB_L_DEG = 264.021
CMB_B_DEG = 48.253
D_KIN_REF = 0.0046  # kinematic-scale reference used throughout this repo (see REPORTS/* injection checks)


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def git_head_hash(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)]).astype(float)


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
    mu = float(np.median(x[valid])) if np.any(valid) else 0.0
    sig = float(np.std(x[valid])) if np.any(valid) else 1.0
    if not np.isfinite(sig) or sig <= 0.0:
        sig = 1.0
    out = (x - mu) / sig
    out[~valid] = 0.0
    return out


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


def fit_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    offset: np.ndarray | None,
    max_iter: int,
    beta_init: np.ndarray | None,
    prior_prec_diag: np.ndarray | None,
    prior_mean: np.ndarray | None = None,
    out_info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Poisson GLM (log link) via damped Newton steps; optional diagonal Gaussian priors (ridge).

    Notes on solver choice
    ----------------------
    We intentionally avoid generic quasi-Newton stopping criteria that can declare convergence
    based on tiny relative changes in a very large log-likelihood value. In this application,
    that can yield materially different fitted dipoles depending on warm-start path, despite
    the objective being convex. A Newton solver with explicit gradient-based termination is
    deterministic and makes scan-vs-single-cut fits auditable.
    """

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    off = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)

    if beta_init is None:
        mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
        beta0 = np.zeros(X.shape[1], dtype=float)
        beta0[0] = math.log(mu0)
    else:
        beta0 = np.asarray(beta_init, dtype=float).reshape(X.shape[1])

    prec = None if prior_prec_diag is None else np.asarray(prior_prec_diag, dtype=float).reshape(X.shape[1])
    mu_prior = None if prior_mean is None else np.asarray(prior_mean, dtype=float).reshape(X.shape[1])

    def nll_grad_hess(beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        eta = np.clip(off + X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        hess = X.T @ (mu[:, None] * X)
        if prec is not None:
            dp = beta if mu_prior is None else (beta - mu_prior)
            nll = float(nll + 0.5 * np.sum(prec * dp * dp))
            grad = grad + prec * dp
            hess = hess + np.diag(prec)
        return nll, np.asarray(grad, dtype=float), np.asarray(hess, dtype=float)

    beta = beta0.copy()
    success = False
    message = "max_iter reached"
    status = 1

    # Gradient tolerances are on the raw objective scale (sums over pixels).
    # These values are conservative enough to avoid warm-start path dependence,
    # while still being feasible for bootstrap loops.
    gtol = 1e-3
    ftol = 1e-10

    nll_prev = float("inf")
    for it in range(int(max_iter)):
        nll, grad, hess = nll_grad_hess(beta)
        g_inf = float(np.max(np.abs(grad))) if grad.size else 0.0
        if g_inf <= gtol:
            success = True
            message = "converged: grad_inf <= gtol"
            status = 0
            nll_prev = nll
            break
        if np.isfinite(nll_prev) and abs(nll_prev - nll) <= ftol * max(1.0, abs(nll_prev)):
            success = True
            message = "converged: relative nll change <= ftol"
            status = 0
            nll_prev = nll
            break

        # Newton direction: solve H d = grad.
        try:
            step_dir = np.linalg.solve(hess, grad)
        except Exception:  # noqa: BLE001
            # Fallback to least squares if H is singular/ill-conditioned.
            step_dir, *_ = np.linalg.lstsq(hess, grad, rcond=None)
            step_dir = np.asarray(step_dir, dtype=float)

        # Backtracking line search to ensure decrease.
        step = 1.0
        for _ls in range(20):
            beta_try = beta - step * step_dir
            nll_try, _, _ = nll_grad_hess(beta_try)
            if np.isfinite(nll_try) and nll_try <= nll:
                beta = beta_try
                nll_prev = nll
                break
            step *= 0.5
        else:
            # Could not find a decreasing step.
            message = "failed: line search"
            status = 2
            break

    # Final diagnostics + covariance (inverse Hessian).
    cov = None
    try:
        nll, grad, hess = nll_grad_hess(beta)
        cov = np.linalg.inv(hess)
        if out_info is not None:
            out_info.update(
                {
                    "success": bool(success),
                    "nit": int(it + 1),
                    "status": int(status),
                    "fun": float(nll),
                    "message": str(message),
                    "grad_inf_norm": float(np.max(np.abs(grad))) if grad.size else float("nan"),
                    "grad_l2_norm": float(np.linalg.norm(grad)) if grad.size else float("nan"),
                }
            )
    except Exception:  # noqa: BLE001
        cov = None
        if out_info is not None:
            out_info.update(
                {
                    "success": bool(success),
                    "nit": int(it + 1),
                    "status": int(status),
                    "fun": float("nan"),
                    "message": str(message),
                    "grad_inf_norm": float("nan"),
                    "grad_l2_norm": float("nan"),
                }
            )

    return np.asarray(beta, dtype=float), cov


@dataclass(frozen=True)
class MaskBundle:
    mask: np.ndarray  # True masked
    seen: np.ndarray  # True unmasked
    ipix_mask_base: np.ndarray  # base sample pixels used for mask_zeros


def build_secrest_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> MaskBundle:
    """mask_zeros + exclusion discs + galactic plane cut (pixel centers)."""
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        neigh = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            neigh[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[neigh] = True  # includes -1 indexing as in Secrest utilities

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

    _, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return MaskBundle(mask=mask, seen=~mask, ipix_mask_base=np.asarray(ipix_base, dtype=np.int64))


def read_healpix_map(
    path: Path,
    *,
    nside_out: int,
    ordering_in: str = "RING",
    fill_from: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import healpy as hp

    m = hp.read_map(str(path), verbose=False)
    nside_in = int(hp.get_nside(m))
    order_in = "NEST" if ordering_in.upper().startswith("N") else "RING"
    if nside_in != int(nside_out):
        m = hp.ud_grade(m, nside_out=int(nside_out), order_in=order_in, order_out="RING", power=0)
        order_in = "RING"
    m = np.asarray(m, dtype=float)
    unseen = ~np.isfinite(m) | (m == hp.UNSEEN)
    if fill_from is None:
        ok = ~unseen
    else:
        ok = np.asarray(fill_from, dtype=bool) & (~unseen)
    if not np.any(ok):
        raise RuntimeError(f"Map has no finite values on requested support: {path}")
    fill = float(np.median(m[ok]))
    m[unseen] = fill
    meta = {
        "path": str(path),
        "nside_in": nside_in,
        "nside_used": int(nside_out),
        "ordering_in": ordering_in,
        "fill_value": fill,
        "missing_frac": float(np.mean(unseen)),
    }
    return m, meta


def alpha_edge_from_cumcounts(w1: np.ndarray, cut: float, *, delta: float) -> float:
    w1 = np.asarray(w1, dtype=float)
    n1 = int(np.sum(w1 <= float(cut)))
    n0 = int(np.sum(w1 <= float(cut) - float(delta)))
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    return float((math.log(n1) - math.log(n0)) / float(delta))


def proj_orthogonal_to(X_base: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Project columns of T to the orthogonal complement of span(X_base).

    Uses an unweighted least squares projection:
      T_perp = T - X_base (X_base^+ T)
    """
    X_base = np.asarray(X_base, dtype=float)
    T = np.asarray(T, dtype=float)
    coef, *_ = np.linalg.lstsq(X_base, T, rcond=None)
    return T - X_base @ coef


@dataclass(frozen=True)
class ScanRow:
    w1_cut: float
    N_seen: int
    bvec: tuple[float, float, float]
    D: float
    l_deg: float
    b_deg: float
    axis_sep_cmb_deg: float
    D_par_cmb: float
    D_perp_cmb: float
    diag: dict[str, float]
    opt: dict[str, Any]


def scan_glm(
    *,
    w1_sorted: np.ndarray,
    ipix_sorted: np.ndarray,
    seen: np.ndarray,
    n_seen: np.ndarray,
    X_nuis_seen: np.ndarray,
    nuisance_names: list[str],
    nuisance_prior_prec: np.ndarray | None,
    offset_seen: np.ndarray | None,
    cuts: list[float],
    max_iter: int,
    seed: int,
) -> tuple[list[ScanRow], dict[str, Any]]:
    """
    Run a cumulative scan and return rows + a small meta blob.

    Model columns are: [1, nx, ny, nz, nuis...]
    Prior/regularization is applied only to nuisance columns (not intercept/dipole).
    """
    rng = np.random.default_rng(int(seed))
    _ = rng  # reserved for future stochastic diagnostics

    npix = int(seen.size)
    counts_all = np.zeros(npix, dtype=np.int64)
    cursor = 0
    beta_prev = None

    rows: list[ScanRow] = []
    for w1_cut in cuts:
        # Inclusive faint-cut semantics: W1 <= W1_max.
        nxt = int(np.searchsorted(w1_sorted, float(w1_cut), side="right"))
        if nxt > cursor:
            delta = ipix_sorted[cursor:nxt]
            counts_all += np.bincount(delta, minlength=npix).astype(np.int64)
            cursor = nxt

        y = counts_all[seen].astype(float)
        N_seen = int(np.sum(counts_all[seen]))

        cols = [np.ones_like(y), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2]]
        if X_nuis_seen.size:
            cols.append(X_nuis_seen)
        X = np.column_stack(cols) if len(cols) > 1 else cols[0][:, None]

        prior = None
        if X.shape[1] > 4:
            prior = np.zeros(X.shape[1], dtype=float)
            if nuisance_prior_prec is None:
                raise RuntimeError("nuisance_prior_prec is required when nuisance columns are present.")
            if nuisance_prior_prec.shape != (X.shape[1] - 4,):
                raise RuntimeError(
                    f"nuisance_prior_prec shape {nuisance_prior_prec.shape} does not match nuisance columns "
                    f"{X.shape[1] - 4}."
                )
            prior[4:] = nuisance_prior_prec

        # Fit with convergence auditing. If the warm-started scan fit does not converge,
        # fall back to a cold-start refit with a larger iteration cap to avoid propagating
        # non-converged solutions across cuts.
        attempts: list[dict[str, Any]] = []
        results: list[tuple[np.ndarray, np.ndarray | None, dict[str, Any]]] = []
        beta_init_i = beta_prev
        max_iter_i = int(max_iter)
        for attempt in range(3):
            info: dict[str, Any] = {}
            beta_i, cov_i = fit_poisson_glm(
                X,
                y,
                offset=offset_seen,
                max_iter=int(max_iter_i),
                beta_init=beta_init_i,
                prior_prec_diag=prior,
                prior_mean=None,
                out_info=info,
            )
            info = dict(info)
            info.update(
                {
                    "attempt": int(attempt),
                    "init": "warm" if beta_init_i is not None else "cold",
                    "max_iter": int(max_iter_i),
                }
            )
            attempts.append(info)
            results.append((beta_i, cov_i, info))
            if bool(info.get("success", False)):
                break
            # Retry: cold start with a larger cap.
            beta_init_i = None
            max_iter_i = int(max_iter_i * 4)

        # Select the best attempt by minimum objective value (prefer converged fits).
        converged = [r for r in results if bool(r[2].get("success", False))]
        pool = converged if converged else results
        beta, _cov, best_info = min(pool, key=lambda r: float(r[2].get("fun", float("inf"))))
        beta_prev = beta if bool(best_info.get("success", False)) else None
        opt = {"selected": int(best_info.get("attempt", -1)), "attempts": attempts}
        bvec = np.asarray(beta[1:4], dtype=float)
        D = float(np.linalg.norm(bvec))
        l_deg, b_deg = vec_to_lb(bvec)
        u_cmb = lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0]
        axis_sep = float(axis_angle_deg(bvec, u_cmb))
        D_par = float(np.dot(bvec, u_cmb))
        D_perp = float(np.linalg.norm(bvec - D_par * u_cmb))

        eta = (np.zeros_like(y) if offset_seen is None else offset_seen) + X @ beta
        mu = np.exp(np.clip(eta, -25.0, 25.0))
        diag = poisson_glm_diagnostics(y, mu, n_params=int(X.shape[1]))

        rows.append(
            ScanRow(
                w1_cut=float(w1_cut),
                N_seen=N_seen,
                bvec=(float(bvec[0]), float(bvec[1]), float(bvec[2])),
                D=D,
                l_deg=float(l_deg),
                b_deg=float(b_deg),
                axis_sep_cmb_deg=axis_sep,
                D_par_cmb=D_par,
                D_perp_cmb=D_perp,
                diag=diag,
                opt=opt,
            )
        )

    meta = {
        "nuisance_names": nuisance_names,
        "nuisance_prior_prec_summary": None
        if nuisance_prior_prec is None
        else {
            "min": float(np.min(nuisance_prior_prec)),
            "p50": float(np.median(nuisance_prior_prec)),
            "max": float(np.max(nuisance_prior_prec)),
        },
        "offset_used": bool(offset_seen is not None),
        "max_iter": int(max_iter),
    }
    return rows, meta


def scan_glm_fixed_axis(
    *,
    w1_sorted: np.ndarray,
    ipix_sorted: np.ndarray,
    seen: np.ndarray,
    dipole_mode_seen: np.ndarray,
    X_nuis_seen: np.ndarray,
    nuisance_names: list[str],
    nuisance_prior_prec: np.ndarray | None,
    offset_seen: np.ndarray | None,
    cuts: list[float],
    max_iter: int,
    seed: int,
) -> tuple[list[ScanRow], dict[str, Any]]:
    """
    Cumulative scan with a *fixed-axis* dipole term:

      log mu_p = beta0 + D_cmb * (u_cmb · n_p) + c · T_p + offset_p

    The fitted coefficient beta[1] is the signed CMB-parallel dipole amplitude.
    """
    rng = np.random.default_rng(int(seed))
    _ = rng

    npix = int(seen.size)
    counts_all = np.zeros(npix, dtype=np.int64)
    cursor = 0
    beta_prev: np.ndarray | None = None

    u_cmb = lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0]

    rows: list[ScanRow] = []
    for w1_cut in cuts:
        # Inclusive faint-cut semantics: W1 <= W1_max.
        nxt = int(np.searchsorted(w1_sorted, float(w1_cut), side="right"))
        if nxt > cursor:
            delta = ipix_sorted[cursor:nxt]
            counts_all += np.bincount(delta, minlength=npix).astype(np.int64)
            cursor = nxt

        y = counts_all[seen].astype(float)
        N_seen = int(np.sum(counts_all[seen]))

        cols = [np.ones_like(y), dipole_mode_seen]
        if X_nuis_seen.size:
            cols.append(X_nuis_seen)
        X = np.column_stack(cols)

        prior = None
        if X.shape[1] > 2:
            prior = np.zeros(X.shape[1], dtype=float)
            if nuisance_prior_prec is None:
                raise RuntimeError("nuisance_prior_prec is required when nuisance columns are present.")
            if nuisance_prior_prec.shape != (X.shape[1] - 2,):
                raise RuntimeError(
                    f"nuisance_prior_prec shape {nuisance_prior_prec.shape} does not match nuisance columns "
                    f"{X.shape[1] - 2}."
                )
            prior[2:] = nuisance_prior_prec

        # As in scan_glm, audit convergence and avoid propagating non-converged warm starts.
        attempts: list[dict[str, Any]] = []
        results: list[tuple[np.ndarray, np.ndarray | None, dict[str, Any]]] = []
        beta_init_i = beta_prev
        max_iter_i = int(max_iter)
        for attempt in range(3):
            info: dict[str, Any] = {}
            beta_i, cov_i = fit_poisson_glm(
                X,
                y,
                offset=offset_seen,
                max_iter=int(max_iter_i),
                beta_init=beta_init_i,
                prior_prec_diag=prior,
                prior_mean=None,
                out_info=info,
            )
            info = dict(info)
            info.update(
                {
                    "attempt": int(attempt),
                    "init": "warm" if beta_init_i is not None else "cold",
                    "max_iter": int(max_iter_i),
                }
            )
            attempts.append(info)
            results.append((beta_i, cov_i, info))
            if bool(info.get("success", False)):
                break
            beta_init_i = None
            max_iter_i = int(max_iter_i * 4)

        converged = [r for r in results if bool(r[2].get("success", False))]
        pool = converged if converged else results
        beta, _cov, best_info = min(pool, key=lambda r: float(r[2].get("fun", float("inf"))))
        beta_prev = beta if bool(best_info.get("success", False)) else None
        opt = {"selected": int(best_info.get("attempt", -1)), "attempts": attempts}
        D_cmb = float(beta[1])
        bvec = u_cmb * D_cmb
        D = float(abs(D_cmb))

        eta = (np.zeros_like(y) if offset_seen is None else offset_seen) + X @ beta
        mu = np.exp(np.clip(eta, -25.0, 25.0))
        diag = poisson_glm_diagnostics(y, mu, n_params=int(X.shape[1]))

        rows.append(
            ScanRow(
                w1_cut=float(w1_cut),
                N_seen=N_seen,
                bvec=(float(bvec[0]), float(bvec[1]), float(bvec[2])),
                D=D,
                l_deg=float(CMB_L_DEG),
                b_deg=float(CMB_B_DEG),
                axis_sep_cmb_deg=0.0,
                D_par_cmb=D_cmb,
                D_perp_cmb=0.0,
                diag=diag,
                opt=opt,
            )
        )

    meta = {
        "nuisance_names": nuisance_names,
        "nuisance_prior_prec_summary": None
        if nuisance_prior_prec is None
        else {
            "min": float(np.min(nuisance_prior_prec)),
            "p50": float(np.median(nuisance_prior_prec)),
            "max": float(np.max(nuisance_prior_prec)),
        },
        "offset_used": bool(offset_seen is not None),
        "max_iter": int(max_iter),
        "dipole_axis": "CMB (fixed)",
    }
    return rows, meta


def rows_to_json(rows: list[ScanRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "w1_cut": float(r.w1_cut),
                "N_seen": int(r.N_seen),
                "dipole": {
                    "bvec": [float(x) for x in r.bvec],
                    "D": float(r.D),
                    "l_deg": float(r.l_deg),
                    "b_deg": float(r.b_deg),
                    "axis_sep_cmb_deg": float(r.axis_sep_cmb_deg),
                },
                "cmb_projection": {
                    "D_par": float(r.D_par_cmb),
                    "D_perp": float(r.D_perp_cmb),
                },
                "fit_diag": r.diag,
                "fit_opt": r.opt,
            }
        )
    return out


def plot_case_closed_figure(
    *,
    outpath: Path,
    cuts: np.ndarray,
    baseline: dict[str, np.ndarray],
    maximal: dict[str, np.ndarray],
    external: dict[str, np.ndarray],
    heldout: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), sharex=False)

    def panel(ax, title: str, D: np.ndarray, axis_sep: np.ndarray) -> None:
        ax2 = ax.twinx()
        ax.plot(cuts, D, color="C0", lw=2, label="D")
        ax.axhline(D_KIN_REF, color="0.4", ls="--", lw=1)
        ax2.plot(cuts, axis_sep, color="C1", lw=2, label="axis sep (CMB)")
        ax.set_title(title)
        ax.set_xlabel("W1_max")
        ax.set_ylabel("Dipole amplitude D")
        ax2.set_ylabel(r"Axis sep to CMB [deg]")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0.0)

    panel(axes[0, 0], "Raw (baseline) dipole-only recovery", baseline["D"], baseline["axis_sep"])
    panel(axes[0, 1], "Maximal nuisance GLM (ridge + low-ell)", maximal["D"], maximal["axis_sep"])
    panel(axes[1, 1], "Maximal + Gaia completeness template", external["D"], external["axis_sep"])

    # Held-out is a single representative cut, shown as a bar comparison.
    ax = axes[1, 0]
    labels = ["Baseline (free axis)", "Maximal (free axis)", "Held-out (CMB-fixed)"]
    Dvals = [heldout["D_baseline"], heldout["D_maximal_in"], heldout["D_heldout_test"]]
    seps = [heldout["axis_sep_baseline"], heldout["axis_sep_maximal_in"], heldout["axis_sep_heldout_test"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, Dvals, width=0.36, color="C0", label="D")
    ax.axhline(D_KIN_REF, color="0.4", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, seps, width=0.36, color="C1", label="axis sep (CMB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("D")
    ax2.set_ylabel("Axis sep [deg]")
    ax.set_title(f"Held-out validation (W1_max={heldout['w1_cut']:.2f})")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_cmb_fixed_amplitude_scan(
    *,
    outpath: Path,
    cuts: np.ndarray,
    Dpar_maximal: np.ndarray,
    Dpar_external: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.2), sharex=True)
    ax.axhline(0.0, color="0.5", lw=1)
    ax.axhline(D_KIN_REF, color="0.4", ls="--", lw=1, label=f"kinematic ref {D_KIN_REF:.4f}")
    ax.axhline(-D_KIN_REF, color="0.4", ls="--", lw=1)
    ax.plot(cuts, Dpar_maximal, color="C2", lw=2, label="CMB-fixed (maximal nuis)")
    ax.plot(cuts, Dpar_external, color="C3", lw=2, label="CMB-fixed + Gaia template")
    ax.set_xlabel("W1_max")
    ax.set_ylabel("Signed CMB-parallel dipole D_par")
    ax.set_title("CMB-fixed dipole amplitude vs W1_max")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_cmb_projection_par_perp_scan(
    *,
    outpath: Path,
    cuts: np.ndarray,
    series: dict[str, dict[str, np.ndarray]],
    cmb_fixed: dict[str, np.ndarray] | None = None,
) -> None:
    """
    Plot the CMB-parallel/perpendicular decomposition of the free-axis dipole vectors:
      D_par = b · u_cmb
      D_perp = |b - (b·u_cmb)u_cmb|
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True)
    ax0, ax1 = axes

    ax0.axhline(D_KIN_REF, color="0.4", ls="--", lw=1, label=f"kinematic ref {D_KIN_REF:.4f}")
    ax0.axhline(-D_KIN_REF, color="0.4", ls="--", lw=1)
    ax1.axhline(0.0, color="0.5", lw=1)

    for label, obj in series.items():
        ax0.plot(cuts, obj["D_par"], lw=2, label=label)
        ax1.plot(cuts, obj["D_perp"], lw=2, label=label)

    if cmb_fixed is not None:
        for label, dpar in cmb_fixed.items():
            ax0.plot(cuts, dpar, lw=2, ls=":", label=label)

    ax0.set_ylabel("CMB-parallel component D_par (signed)")
    ax1.set_ylabel("CMB-perp component D_perp")
    ax1.set_xlabel("W1_max")
    ax0.grid(alpha=0.3)
    ax1.grid(alpha=0.3)
    ax0.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
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
    ap.add_argument("--representative-cut", type=float, default=16.6)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--ridge-sigma",
        type=float,
        default=3.0,
        help="Gaussian prior sigma for nuisance coefficients (templates are standardized).",
    )
    ap.add_argument(
        "--harmonic-lmax",
        type=int,
        default=5,
        help="Include explicit real spherical-harmonic nuisance modes for 2<=ell<=harmonic_lmax (0/1 disables).",
    )
    ap.add_argument(
        "--harmonic-prior",
        choices=["ridge", "lognormal_cl"],
        default="lognormal_cl",
        help="Prior for harmonic coefficients: ridge (same as other templates) or lognormal_cl (C_ell-based).",
    )
    ap.add_argument(
        "--harmonic-prior-cl-json",
        default="REPORTS/Q_D_RES_2_2/data/lognormal_cov_w1max16p6_n500/lognormal_mocks_cov.json",
        help="C_ell JSON providing cl_estimate.cl_signal (used when --harmonic-prior=lognormal_cl).",
    )
    ap.add_argument(
        "--harmonic-prior-scale",
        type=float,
        default=1000.0,
        help="Multiply C_ell by this factor when forming harmonic priors (larger = weaker prior).",
    )
    ap.add_argument(
        "--harmonic-prior-min-cl",
        type=float,
        default=1e-12,
        help="Floor for C_ell when forming 1/C_ell priors (avoids singular priors).",
    )
    ap.add_argument(
        "--heldout-wedge-deg",
        type=float,
        default=15.0,
        help="Ecliptic-longitude wedge size used for GroupKFold group labels.",
    )
    ap.add_argument("--heldout-fold", type=int, default=0, help="Which fold index to report in the 2-fold split.")
    ap.add_argument("--out-report-dir", default="REPORTS/case_closed_maximal_nuisance_suite")
    ap.add_argument("--unwise-lognexp-map", default="data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits")
    ap.add_argument("--unwise-invvar-map", default="data/cache/unwise_invvar/neo7/invvar_healpix_nside64.fits")
    ap.add_argument(
        "--star-count-map",
        default="outputs/star_w1_zeropoint_map_allwise_mod8_snr20_msig0p1_full/star_w1_resid_count_nside64_all.fits",
    )
    ap.add_argument(
        "--gaia-logp-offset-map",
        default="REPORTS/external_completeness_gaia_qsocand_externalonly/data/logp_offset_map_nside64.fits",
    )
    ap.add_argument(
        "--sdss-delta-m-map",
        default="REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits",
    )
    ap.add_argument("--do-scan", action="store_true", help="Run the W1_max scan suite.")
    ap.add_argument("--do-heldout", action="store_true", help="Run held-out sky validation.")
    ap.add_argument("--do-loto", action="store_true", help="Run leave-one-template-out attribution (rep cut only).")
    ap.add_argument("--do-orthogonalized", action="store_true", help="Also run the orthogonalized-nuisance scan.")
    ap.add_argument("--do-gaia-replication", action="store_true", help="Run Gaia qsocand cross-catalog replication.")
    ap.add_argument("--gaia-qsocand-gz", default="data/external/gaia_dr3_extragal/qsocand.dat.gz")
    ap.add_argument("--gaia-pqso-min", type=float, default=0.8)
    ap.add_argument("--gaia-max-lines", type=int, default=None, help="Optional cap for smoke tests.")
    ap.add_argument("--bootstrap-nsim", type=int, default=400, help="Parametric bootstrap draws for D_par calibration.")
    ap.add_argument(
        "--bootstrap-kin-dpar",
        type=float,
        default=D_KIN_REF,
        help="Constrained-null CMB-parallel dipole amplitude used for bootstrap calibration.",
    )
    ap.add_argument(
        "--bootstrap-max-iter",
        type=int,
        default=250,
        help="Max optimizer iterations per bootstrap recovery fit (warm-started).",
    )
    ap.add_argument("--no-bootstrap", action="store_true", help="Disable the constrained-null parametric bootstrap.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    # "Do it all at once" default: if the caller didn't pick suites, run the full package.
    if not (args.do_scan or args.do_heldout or args.do_loto):
        args.do_scan = True
        args.do_heldout = True
        args.do_loto = True
        args.do_orthogonalized = True

    repo_root = Path(__file__).resolve().parents[1]
    report_dir = repo_root / str(args.out_report_dir)
    data_dir = report_dir / "data"
    fig_dir = report_dir / "figures"
    ensure_dir(data_dir)
    ensure_dir(fig_dir)

    # Record command line.
    cmdline = " ".join(shlex.quote(a) for a in os.sys.argv)
    head = git_head_hash(repo_root)

    import healpy as hp
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from sklearn.model_selection import GroupKFold

    # Load CatWISE.
    cat_path = repo_root / str(args.catalog)
    if not cat_path.exists():
        raise SystemExit(f"Missing catalog: {cat_path}")

    with fits.open(str(cat_path), memmap=True) as hdul:
        d = hdul[1].data
        w1 = np.asarray(d["w1"], dtype=float)
        w1cov = np.asarray(d["w1cov"], dtype=float)
        l = np.asarray(d["l"], dtype=float)
        b = np.asarray(d["b"], dtype=float)
        ebv = np.asarray(d["ebv"], dtype=float) if "ebv" in d.names else None

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    if ebv is not None:
        base &= np.isfinite(ebv)
    base &= w1cov >= float(args.w1cov_min)

    theta = np.deg2rad(90.0 - b[base])
    phi = np.deg2rad(l[base] % 360.0)
    ipix_base = hp.ang2pix(int(args.nside), theta, phi, nest=False).astype(np.int64)

    mask_bundle = build_secrest_mask(
        nside=int(args.nside),
        ipix_base=ipix_base,
        exclude_mask_fits=str(repo_root / str(args.exclude_mask_fits)) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )
    seen = mask_bundle.seen
    mask = mask_bundle.mask
    npix = hp.nside2npix(int(args.nside))

    # Pixel geometry and ecliptic templates (pixel centers).
    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)
    pix_unit = lb_to_unitvec(lon_pix, lat_pix)
    n_seen = pix_unit[seen]

    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    elat = sc_pix.barycentricmeanecliptic.lat.deg.astype(float)
    elon = sc_pix.barycentricmeanecliptic.lon.deg.astype(float)
    abs_elat = np.abs(elat)
    sin_elon = np.sin(np.deg2rad(elon))
    cos_elon = np.cos(np.deg2rad(elon))
    sin2_elon = np.sin(2.0 * np.deg2rad(elon))
    cos2_elon = np.cos(2.0 * np.deg2rad(elon))

    # Per-pixel EBV mean + W1cov mean from the base sample (used as map-level covariates).
    cnt_base = np.bincount(ipix_base, minlength=npix).astype(float)
    sum_w1cov = np.bincount(ipix_base, weights=w1cov[base].astype(float), minlength=npix).astype(float)
    w1cov_mean = np.divide(sum_w1cov, cnt_base, out=np.zeros_like(sum_w1cov), where=cnt_base != 0.0)
    if ebv is None:
        ebv_mean = np.zeros(npix, dtype=float)
    else:
        sum_ebv = np.bincount(ipix_base, weights=ebv[base].astype(float), minlength=npix).astype(float)
        ebv_mean = np.divide(sum_ebv, cnt_base, out=np.zeros_like(sum_ebv), where=cnt_base != 0.0)

    # Load additional maps (Galactic coords).
    lognexp_map, lognexp_meta = read_healpix_map(
        repo_root / str(args.unwise_lognexp_map),
        nside_out=int(args.nside),
        ordering_in="RING",
        fill_from=seen,
    )
    invvar_map, invvar_meta = read_healpix_map(
        repo_root / str(args.unwise_invvar_map),
        nside_out=int(args.nside),
        ordering_in="RING",
        fill_from=seen,
    )
    star_map, star_meta = read_healpix_map(
        repo_root / str(args.star_count_map),
        nside_out=int(args.nside),
        ordering_in="RING",
        fill_from=seen,
    )

    # External offset maps.
    gaia_logp_map, gaia_logp_meta = read_healpix_map(
        repo_root / str(args.gaia_logp_offset_map),
        nside_out=int(args.nside),
        ordering_in="RING",
        fill_from=seen,
    )
    sdss_dm_map, sdss_dm_meta = read_healpix_map(
        repo_root / str(args.sdss_delta_m_map),
        nside_out=int(args.nside),
        ordering_in="RING",
        fill_from=seen,
    )

    # Ridge prior precision for nuisance coefs (templates are standardized).
    if not np.isfinite(args.ridge_sigma) or float(args.ridge_sigma) <= 0.0:
        raise SystemExit("--ridge-sigma must be finite and > 0")
    ridge_prec = 1.0 / (float(args.ridge_sigma) ** 2)

    # Standardized nuisance templates (seen pixels only), with per-template priors/regularization.
    nuisance_cols: list[np.ndarray] = []
    nuisance_names: list[str] = []
    nuisance_prior_prec: list[float] = []

    def add_t(name: str, arr: np.ndarray, *, transform: str = "z", prior_prec: float | None = None) -> None:
        nonlocal nuisance_cols, nuisance_names, nuisance_prior_prec
        if transform == "z":
            col = zscore(arr, seen)[seen]
        elif transform == "log1p_z":
            col = zscore(np.log1p(np.clip(arr, 0.0, np.inf)), seen)[seen]
        elif transform == "log_z":
            col = zscore(np.log(np.clip(arr, 1e-12, np.inf)), seen)[seen]
        else:
            raise ValueError(f"unknown transform: {transform}")
        nuisance_cols.append(np.asarray(col, dtype=float))
        nuisance_names.append(str(name))
        nuisance_prior_prec.append(float(ridge_prec if prior_prec is None else prior_prec))

    # "Maximal reasonable" basis (not exhaustive).
    add_t("abs_elat_z", abs_elat, transform="z")
    add_t("sin_elon_z", sin_elon, transform="z")
    add_t("cos_elon_z", cos_elon, transform="z")
    add_t("sin2_elon_z", sin2_elon, transform="z")
    add_t("cos2_elon_z", cos2_elon, transform="z")
    add_t("ebv_mean_z", ebv_mean, transform="z")
    add_t("log_w1cov_mean_z", w1cov_mean, transform="log_z")
    add_t("lognexp_z", lognexp_map, transform="z")
    add_t("log_invvar_z", invvar_map, transform="log_z")
    add_t("log1p_starcount_z", star_map, transform="log1p_z")

    # Explicit low-ell harmonic nuisance modes (ℓ=2..lmax), with priors.
    harmonic_meta: dict[str, Any] | None = None
    if int(args.harmonic_lmax) > 1:
        try:
            # SciPy <1.17
            from scipy.special import sph_harm as _sph_harm  # type: ignore

            def sph_harm(l: int, m: int, th: np.ndarray, ph: np.ndarray) -> np.ndarray:
                return _sph_harm(m, l, ph, th)

        except Exception:
            # SciPy >=1.17
            from scipy.special import sph_harm_y as _sph_harm_y  # type: ignore

            def sph_harm(l: int, m: int, th: np.ndarray, ph: np.ndarray) -> np.ndarray:
                return _sph_harm_y(l, m, th, ph)

        prior_mode = str(args.harmonic_prior)
        cl_signal = None
        cl_path = None
        if prior_mode == "lognormal_cl":
            cl_path = repo_root / str(args.harmonic_prior_cl_json)
            if not cl_path.exists():
                raise SystemExit(f"Missing --harmonic-prior-cl-json: {cl_path}")
            cl_obj = json.loads(cl_path.read_text())
            try:
                cl_signal = np.asarray(cl_obj["cl_estimate"]["cl_signal"], dtype=float)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"{cl_path}: expected cl_estimate.cl_signal") from e

        scale = float(args.harmonic_prior_scale)
        cl_floor = float(args.harmonic_prior_min_cl)
        if not np.isfinite(scale) or scale <= 0.0:
            raise SystemExit("--harmonic-prior-scale must be finite and > 0")
        if not np.isfinite(cl_floor) or cl_floor <= 0.0:
            raise SystemExit("--harmonic-prior-min-cl must be finite and > 0")

        th = np.deg2rad(90.0 - lat_pix)  # colatitude
        ph = np.deg2rad(lon_pix % 360.0)  # longitude

        harm_names: list[str] = []
        harm_ells: list[int] = []
        harm_raw_std: list[float] = []
        harm_prior_prec: list[float] = []

        def _harm_col(raw_full: np.ndarray) -> tuple[np.ndarray, float] | None:
            raw_full = np.asarray(raw_full, dtype=float)
            raw0 = raw_full - float(np.mean(raw_full[seen]))
            std = float(np.std(raw0[seen]))
            if not np.isfinite(std) or std <= 0.0:
                return None
            return (raw0 / std)[seen], std

        for ell in range(2, int(args.harmonic_lmax) + 1):
            y0 = sph_harm(int(ell), 0, th, ph).real.astype(float)
            col = _harm_col(y0)
            if col is not None:
                name = f"Y{ell}_0_re_z"
                std_raw = float(col[1])
                prec = float(ridge_prec)
                if prior_mode == "lognormal_cl":
                    assert cl_signal is not None
                    if int(ell) >= int(cl_signal.size):
                        raise SystemExit(f"C_ell array too short for ell={ell} (size={cl_signal.size}).")
                    cl = float(cl_signal[int(ell)])
                    if not np.isfinite(cl) or cl <= 0.0:
                        cl = cl_floor
                    cl = max(cl_floor, cl)
                    var_beta = cl * scale * (std_raw**2)
                    prec = 1.0 / float(max(cl_floor, var_beta))
                nuisance_cols.append(np.asarray(col[0], dtype=float))
                nuisance_names.append(str(name))
                nuisance_prior_prec.append(float(prec))
                harm_names.append(name)
                harm_ells.append(int(ell))
                harm_raw_std.append(std_raw)
                harm_prior_prec.append(float(prec))

            for m in range(1, int(ell) + 1):
                y = sph_harm(int(ell), int(m), th, ph)
                yr = (np.sqrt(2.0) * y.real).astype(float)
                yi = (np.sqrt(2.0) * y.imag).astype(float)
                colr = _harm_col(yr)
                if colr is not None:
                    name = f"Y{ell}_{m}_re_z"
                    std_raw = float(colr[1])
                    prec = float(ridge_prec)
                    if prior_mode == "lognormal_cl":
                        assert cl_signal is not None
                        cl = float(cl_signal[int(ell)])
                        if not np.isfinite(cl) or cl <= 0.0:
                            cl = cl_floor
                        cl = max(cl_floor, cl)
                        var_beta = cl * scale * (std_raw**2)
                        prec = 1.0 / float(max(cl_floor, var_beta))
                    nuisance_cols.append(np.asarray(colr[0], dtype=float))
                    nuisance_names.append(str(name))
                    nuisance_prior_prec.append(float(prec))
                    harm_names.append(name)
                    harm_ells.append(int(ell))
                    harm_raw_std.append(std_raw)
                    harm_prior_prec.append(float(prec))

                coli = _harm_col(yi)
                if coli is not None:
                    name = f"Y{ell}_{m}_im_z"
                    std_raw = float(coli[1])
                    prec = float(ridge_prec)
                    if prior_mode == "lognormal_cl":
                        assert cl_signal is not None
                        cl = float(cl_signal[int(ell)])
                        if not np.isfinite(cl) or cl <= 0.0:
                            cl = cl_floor
                        cl = max(cl_floor, cl)
                        var_beta = cl * scale * (std_raw**2)
                        prec = 1.0 / float(max(cl_floor, var_beta))
                    nuisance_cols.append(np.asarray(coli[0], dtype=float))
                    nuisance_names.append(str(name))
                    nuisance_prior_prec.append(float(prec))
                    harm_names.append(name)
                    harm_ells.append(int(ell))
                    harm_raw_std.append(std_raw)
                    harm_prior_prec.append(float(prec))

        harmonic_meta = {
            "harmonic_lmax": int(args.harmonic_lmax),
            "prior": prior_mode,
            "prior_cl_json": None if cl_path is None else str(cl_path),
            "prior_scale": float(scale),
            "prior_min_cl": float(cl_floor),
            "n_harmonics": int(len(harm_names)),
            "prior_prec_summary": None
            if not harm_prior_prec
            else {"min": float(np.min(harm_prior_prec)), "p50": float(np.median(harm_prior_prec)), "max": float(np.max(harm_prior_prec))},
        }

    X_nuis_seen = np.column_stack(nuisance_cols) if nuisance_cols else np.zeros((n_seen.shape[0], 0), dtype=float)
    nuisance_prior_prec_arr = np.array(nuisance_prior_prec, dtype=float) if nuisance_prior_prec else None

    # Prepare cumulative selection ordering.
    order = np.argsort(w1[base])
    w1_sorted = w1[base][order]
    ipix_sorted = ipix_base[order]

    # Parse W1 grid.
    w1_start, w1_stop, w1_step = (float(x) for x in str(args.w1_grid).split(","))
    n_steps = int(round((w1_stop - w1_start) / w1_step)) + 1
    cuts = [w1_start + i * w1_step for i in range(n_steps)]
    # Ensure the representative cut is included in the scan grid so scan vs rep-cut
    # consistency can be audited deterministically.
    rep_cut = float(args.representative_cut)
    if not any(abs(float(c) - rep_cut) <= 1e-9 for c in cuts):
        cuts.append(rep_cut)
    cuts = sorted(float(c) for c in cuts)
    # Unique with tolerance to avoid float duplication.
    cuts_unique: list[float] = []
    for c in cuts:
        if not cuts_unique or abs(c - cuts_unique[-1]) > 1e-9:
            cuts_unique.append(float(c))
    cuts = cuts_unique

    # Baseline model: dipole + abs_elat (Secrest-style minimal ecliptic latitude trend).
    X_base_nuis = zscore(abs_elat, seen)[seen][:, None]
    base_names = ["abs_elat_z"]

    scan_payload: dict[str, Any] = {}

    if args.do_scan:
        baseline_rows, baseline_meta = scan_glm(
            w1_sorted=w1_sorted,
            ipix_sorted=ipix_sorted,
            seen=seen,
            n_seen=n_seen,
            X_nuis_seen=X_base_nuis,
            nuisance_names=base_names,
            nuisance_prior_prec=np.array([ridge_prec], dtype=float),
            offset_seen=None,
            cuts=cuts,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )
        maximal_rows, maximal_meta = scan_glm(
            w1_sorted=w1_sorted,
            ipix_sorted=ipix_sorted,
            seen=seen,
            n_seen=n_seen,
            X_nuis_seen=X_nuis_seen,
            nuisance_names=nuisance_names,
            nuisance_prior_prec=nuisance_prior_prec_arr,
            offset_seen=None,
            cuts=cuts,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )

        # Cross-catalog external completeness pattern (Gaia qsocand) as a *free* nuisance template.
        gaia_logp_z_seen = zscore(gaia_logp_map, seen)[seen]
        X_nuis_gaia_seen = np.column_stack([X_nuis_seen, gaia_logp_z_seen])
        prior_gaia = (
            np.concatenate([nuisance_prior_prec_arr, np.array([ridge_prec], dtype=float)])
            if nuisance_prior_prec_arr is not None
            else np.array([ridge_prec], dtype=float)
        )
        external_rows, external_meta = scan_glm(
            w1_sorted=w1_sorted,
            ipix_sorted=ipix_sorted,
            seen=seen,
            n_seen=n_seen,
            X_nuis_seen=X_nuis_gaia_seen,
            nuisance_names=nuisance_names + ["gaia_logp_offset_z"],
            nuisance_prior_prec=prior_gaia,
            offset_seen=None,
            cuts=cuts,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )

        # Fixed-axis (CMB) scan (physical mode), with the same nuisance basis + priors.
        u_cmb = lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0]
        dip_mode = (n_seen @ u_cmb).astype(float)
        cmb_rows, cmb_meta = scan_glm_fixed_axis(
            w1_sorted=w1_sorted,
            ipix_sorted=ipix_sorted,
            seen=seen,
            dipole_mode_seen=dip_mode,
            X_nuis_seen=X_nuis_seen,
            nuisance_names=nuisance_names,
            nuisance_prior_prec=nuisance_prior_prec_arr,
            offset_seen=None,
            cuts=cuts,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )
        cmb_ext_rows, cmb_ext_meta = scan_glm_fixed_axis(
            w1_sorted=w1_sorted,
            ipix_sorted=ipix_sorted,
            seen=seen,
            dipole_mode_seen=dip_mode,
            X_nuis_seen=X_nuis_gaia_seen,
            nuisance_names=nuisance_names + ["gaia_logp_offset_z"],
            nuisance_prior_prec=prior_gaia,
            offset_seen=None,
            cuts=cuts,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )

        scan_payload = {
            "baseline": {"meta": baseline_meta, "rows": rows_to_json(baseline_rows)},
            "maximal": {"meta": maximal_meta, "rows": rows_to_json(maximal_rows)},
            "external_gaia_template": {"meta": external_meta, "rows": rows_to_json(external_rows)},
            "cmb_fixed_maximal": {"meta": cmb_meta, "rows": rows_to_json(cmb_rows)},
            "cmb_fixed_external_gaia_template": {"meta": cmb_ext_meta, "rows": rows_to_json(cmb_ext_rows)},
        }

        if args.do_orthogonalized:
            # Orthogonalize nuisance templates to intercept+dipole on seen pixels.
            X_base = np.column_stack([np.ones(n_seen.shape[0]), n_seen])
            T_perp = proj_orthogonal_to(X_base, X_nuis_seen)
            # Re-standardize after projection.
            T_perp_z = np.column_stack([zscore(T_perp[:, j], np.ones(T_perp.shape[0], dtype=bool)) for j in range(T_perp.shape[1])])
            ortho_rows, ortho_meta = scan_glm(
                w1_sorted=w1_sorted,
                ipix_sorted=ipix_sorted,
                seen=seen,
                n_seen=n_seen,
                X_nuis_seen=T_perp_z,
                nuisance_names=[f"{n}_orth" for n in nuisance_names],
                nuisance_prior_prec=np.full(T_perp_z.shape[1], ridge_prec, dtype=float),
                offset_seen=None,
                cuts=cuts,
                max_iter=int(args.max_iter),
                seed=int(args.seed),
            )
            scan_payload["maximal_orth"] = {"meta": ortho_meta, "rows": rows_to_json(ortho_rows)}

        write_json(data_dir / "scan_suite.json", scan_payload)

    # Held-out validation at representative cut.
    heldout_payload: dict[str, Any] = {}
    heldout_summary_for_fig: dict[str, float] = {
        "w1_cut": float(args.representative_cut),
        "D_baseline": float("nan"),
        "D_maximal_in": float("nan"),
        "D_heldout_test": float("nan"),
        "axis_sep_baseline": float("nan"),
        "axis_sep_maximal_in": float("nan"),
        "axis_sep_heldout_test": float("nan"),
    }

    if args.do_heldout:
        rep_cut = float(args.representative_cut)
        # Build counts at rep_cut.
        # Inclusive faint-cut semantics: W1 <= W1_max.
        nxt = int(np.searchsorted(w1_sorted, rep_cut, side="right"))
        counts_all = np.bincount(ipix_sorted[:nxt], minlength=npix).astype(np.int64)
        y_all_seen = counts_all[seen].astype(float)

        # Group labels: ecliptic longitude wedges.
        wedge = float(args.heldout_wedge_deg)
        if not np.isfinite(wedge) or wedge <= 0.0:
            raise SystemExit("--heldout-wedge-deg must be finite and > 0")
        wedge_id = np.floor((elon[seen] % 360.0) / wedge).astype(int)

        # 2-fold split using GroupKFold.
        gkf = GroupKFold(n_splits=2)
        splits = list(gkf.split(X_nuis_seen, y_all_seen, groups=wedge_id))
        fold = int(args.heldout_fold) % 2
        train_idx, test_idx = splits[fold]

        # In-sample baseline fit (for context).
        Xb_all = np.column_stack([np.ones_like(y_all_seen), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2], X_base_nuis[:, 0]])
        prior_b = np.zeros(Xb_all.shape[1], dtype=float)
        prior_b[4:] = np.array([ridge_prec], dtype=float)
        beta_b, _ = fit_poisson_glm(
            Xb_all,
            y_all_seen,
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=prior_b,
        )
        bvec_b = beta_b[1:4]
        heldout_summary_for_fig["D_baseline"] = float(np.linalg.norm(bvec_b))
        heldout_summary_for_fig["axis_sep_baseline"] = float(
            axis_angle_deg(bvec_b, lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0])
        )

        # Baseline train->test deviance (for generalization context).
        beta_b_train, _ = fit_poisson_glm(
            Xb_all[train_idx],
            y_all_seen[train_idx],
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=beta_b,
            prior_prec_diag=prior_b,
        )
        eta_b_test = Xb_all[test_idx] @ beta_b_train
        mu_b_test = np.exp(np.clip(eta_b_test, -25.0, 25.0))
        diag_b_test = poisson_glm_diagnostics(y_all_seen[test_idx], mu_b_test, n_params=int(Xb_all.shape[1]))

        # In-sample maximal fit.
        Xm_all = np.column_stack([np.ones_like(y_all_seen), n_seen, X_nuis_seen])
        prior_m = np.zeros(Xm_all.shape[1], dtype=float)
        if nuisance_prior_prec_arr is None:
            raise RuntimeError("Internal error: nuisance_prior_prec_arr is None but maximal nuisance columns exist.")
        prior_m[4:] = nuisance_prior_prec_arr
        beta_m, _ = fit_poisson_glm(
            Xm_all,
            y_all_seen,
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=prior_m,
        )
        bvec_m = beta_m[1:4]
        heldout_summary_for_fig["D_maximal_in"] = float(np.linalg.norm(bvec_m))
        heldout_summary_for_fig["axis_sep_maximal_in"] = float(
            axis_angle_deg(bvec_m, lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0])
        )

        # Train on train pixels, evaluate on test.
        Xm_train = Xm_all[train_idx]
        y_train = y_all_seen[train_idx]
        beta_train, _ = fit_poisson_glm(
            Xm_train,
            y_train,
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=beta_m,
            prior_prec_diag=prior_m,
        )

        # Held-out deviance on test.
        Xm_test = Xm_all[test_idx]
        y_test = y_all_seen[test_idx]
        eta_test = Xm_test @ beta_train
        mu_test = np.exp(np.clip(eta_test, -25.0, 25.0))
        diag_test = poisson_glm_diagnostics(y_test, mu_test, n_params=int(Xm_all.shape[1]))

        # Held-out residual dipole: fix nuisance coefficients from training as offset, refit dipole+intercept.
        c_train = beta_train[4:]
        off_test = X_nuis_seen[test_idx] @ c_train
        Xdip_test = np.column_stack([np.ones_like(y_test), n_seen[test_idx]])
        beta_dip, _ = fit_poisson_glm(
            Xdip_test,
            y_test,
            offset=off_test,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=None,
        )
        bvec_test = beta_dip[1:4]
        heldout_summary_for_fig["D_heldout_test"] = float(np.linalg.norm(bvec_test))
        heldout_summary_for_fig["axis_sep_heldout_test"] = float(
            axis_angle_deg(bvec_test, lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0])
        )

        heldout_payload = {
            "representative_cut": rep_cut,
            "heldout": {
                "scheme": "GroupKFold n_splits=2 on ecliptic longitude wedges",
                "wedge_deg": wedge,
                "fold_index": fold,
                "n_train_pix": int(train_idx.size),
                "n_test_pix": int(test_idx.size),
            },
            "baseline_full": {
                "D": float(np.linalg.norm(bvec_b)),
                "lb_deg": list(vec_to_lb(bvec_b)),
                "axis_sep_cmb_deg": heldout_summary_for_fig["axis_sep_baseline"],
            },
            "baseline_train_model": {
                "beta": [float(x) for x in beta_b_train],
            },
            "maximal_full": {
                "D": float(np.linalg.norm(bvec_m)),
                "lb_deg": list(vec_to_lb(bvec_m)),
                "axis_sep_cmb_deg": heldout_summary_for_fig["axis_sep_maximal_in"],
            },
            "maximal_train_model": {
                "beta": [float(x) for x in beta_train],
            },
            "test_baseline_deviance": diag_b_test,
            "test_deviance": diag_test,
            "test_residual_dipole": {
                "D": float(np.linalg.norm(bvec_test)),
                "lb_deg": list(vec_to_lb(bvec_test)),
                "axis_sep_cmb_deg": heldout_summary_for_fig["axis_sep_heldout_test"],
            },
        }

        # CMB-fixed held-out measurement: train nuisance-only on train, then fit D_par on test
        # with that nuisance field as an offset (out-of-sample dipole estimate).
        u_cmb = lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0]
        dip_mode_all = (n_seen @ u_cmb).astype(float)
        if nuisance_prior_prec_arr is None:
            raise RuntimeError("Internal error: nuisance_prior_prec_arr is None but nuisance columns exist.")

        # Nuisance-only training fit: [1, T].
        Xn_train = np.column_stack([np.ones_like(y_all_seen[train_idx]), X_nuis_seen[train_idx]])
        prior_n = np.zeros(Xn_train.shape[1], dtype=float)
        prior_n[1:] = nuisance_prior_prec_arr
        beta_n_train, _ = fit_poisson_glm(
            Xn_train,
            y_all_seen[train_idx],
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=prior_n,
        )
        c_n_train = beta_n_train[1:]
        off_cmb_test = X_nuis_seen[test_idx] @ c_n_train

        # Test dipole fit (CMB fixed): [1, u_cmb·n] with nuisance offset.
        Xcmb_test = np.column_stack([np.ones_like(y_all_seen[test_idx]), dip_mode_all[test_idx]])
        beta_cmb_test, _ = fit_poisson_glm(
            Xcmb_test,
            y_all_seen[test_idx],
            offset=off_cmb_test,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=None,
        )
        Dpar_test = float(beta_cmb_test[1])

        heldout_payload["cmb_fixed_test"] = {
            "D_par": Dpar_test,
            "D_abs": float(abs(Dpar_test)),
            "axis_sep_cmb_deg": 0.0,
        }
        heldout_payload["cmb_fixed_train_nuisance_beta"] = [float(x) for x in beta_n_train]

        # For the headline held-out bar, treat this as the "physical" held-out residual amplitude.
        heldout_summary_for_fig["D_heldout_test"] = float(abs(Dpar_test))
        heldout_summary_for_fig["axis_sep_heldout_test"] = 0.0
        write_json(data_dir / "heldout_validation.json", heldout_payload)

    # LOTO attribution at representative cut (maximal model, in-sample).
    loto_payload: dict[str, Any] = {}
    if args.do_loto:
        rep_cut = float(args.representative_cut)
        # Inclusive faint-cut semantics: W1 <= W1_max.
        nxt = int(np.searchsorted(w1_sorted, rep_cut, side="right"))
        counts_all = np.bincount(ipix_sorted[:nxt], minlength=npix).astype(np.int64)
        y_seen = counts_all[seen].astype(float)

        Xfull = np.column_stack([np.ones_like(y_seen), n_seen, X_nuis_seen])
        prior_full = np.zeros(Xfull.shape[1], dtype=float)
        if nuisance_prior_prec_arr is None:
            raise RuntimeError("Internal error: nuisance_prior_prec_arr is None but nuisance columns exist.")
        prior_full[4:] = nuisance_prior_prec_arr
        beta_full, _ = fit_poisson_glm(
            Xfull,
            y_seen,
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=prior_full,
        )
        eta_full = Xfull @ beta_full
        mu_full = np.exp(np.clip(eta_full, -25.0, 25.0))
        diag_full = poisson_glm_diagnostics(y_seen, mu_full, n_params=int(Xfull.shape[1]))
        dev_full = float(diag_full["deviance"])
        b_full = beta_full[1:4]
        D_full = float(np.linalg.norm(b_full))

        rows: list[dict[str, Any]] = []
        for j, name in enumerate(nuisance_names):
            keep = [k for k in range(X_nuis_seen.shape[1]) if k != j]
            T = X_nuis_seen[:, keep]
            Xj = np.column_stack([np.ones_like(y_seen), n_seen, T])
            prior_j = np.zeros(Xj.shape[1], dtype=float)
            prior_j[4:] = nuisance_prior_prec_arr[np.asarray(keep, dtype=int)]
            beta_init = np.concatenate([beta_full[:4], beta_full[4 + np.asarray(keep, dtype=int)]])
            beta_j, _ = fit_poisson_glm(
                Xj,
                y_seen,
                offset=None,
                max_iter=int(args.max_iter),
                beta_init=beta_init,
                prior_prec_diag=prior_j,
            )
            eta_j = Xj @ beta_j
            mu_j = np.exp(np.clip(eta_j, -25.0, 25.0))
            diag_j = poisson_glm_diagnostics(y_seen, mu_j, n_params=int(Xj.shape[1]))
            dev_j = float(diag_j["deviance"])
            b_j = beta_j[1:4]
            rows.append(
                {
                    "dropped": name,
                    "delta_deviance": float(dev_j - dev_full),
                    "D_drop": float(np.linalg.norm(b_j)),
                    "axis_sep_cmb_deg": float(
                        axis_angle_deg(b_j, lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0])
                    ),
                }
            )

        rows_sorted = sorted(rows, key=lambda r: float(r["delta_deviance"]), reverse=True)
        loto_payload = {
            "representative_cut": rep_cut,
            "full": {
                "D": D_full,
                "lb_deg": list(vec_to_lb(b_full)),
                "deviance": dev_full,
            },
            "rows": rows_sorted,
        }
        write_json(data_dir / "loto_attribution.json", loto_payload)

        # Write a CSV for easy copy/paste.
        csv_path = data_dir / "loto_attribution.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dropped", "delta_deviance", "D_drop", "axis_sep_cmb_deg"])
            for r in rows_sorted:
                w.writerow([r["dropped"], f"{r['delta_deviance']:.6g}", f"{r['D_drop']:.6g}", f"{r['axis_sep_cmb_deg']:.6g}"])

    # Cross-catalog replication: Gaia DR3 QSO candidates (all-sky), measured on the same footprint.
    gaia_payload: dict[str, Any] = {}
    gaia_json_path = data_dir / "gaia_replication.json"
    if bool(args.do_gaia_replication):
        import gzip

        qsocand_path = repo_root / str(args.gaia_qsocand_gz)
        if not qsocand_path.exists():
            raise SystemExit(f"Missing Gaia qsocand file: {qsocand_path}")
        pqso_min = float(args.gaia_pqso_min)
        if not np.isfinite(pqso_min):
            raise SystemExit("--gaia-pqso-min must be finite")

        # CDS fixed-width parse (same byte ranges used in scripts/run_gaia_dr3_qsocand_completeness_model.py).
        pqso_slice = slice(186, 200)  # Bytes 187-200
        ra_slice = slice(948, 971)  # Bytes 949-971
        dec_slice = slice(972, 995)  # Bytes 973-995

        def _parse_float_field(line: str, s: slice) -> float | None:
            txt = line[s].strip()
            if not txt:
                return None
            try:
                return float(txt)
            except ValueError:
                return None

        def unitvec_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
            ra = np.deg2rad(np.asarray(ra_deg, dtype=float) % 360.0)
            dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
            cosd = np.cos(dec)
            return np.column_stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)]).astype(float)

        # ICRS -> Galactic rotation matrix.
        R_EQ_TO_GAL = np.array(
            [
                [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
                [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
            ],
            dtype=float,
        )

        def gal_ipix_from_unitvec_eq(v_eq: np.ndarray, *, nside: int) -> np.ndarray:
            v_gal = np.asarray(v_eq, dtype=float) @ R_EQ_TO_GAL.T
            lon = np.arctan2(v_gal[:, 1], v_gal[:, 0]) % (2.0 * math.pi)
            lat = np.arcsin(np.clip(v_gal[:, 2], -1.0, 1.0))
            theta = (0.5 * math.pi) - lat
            phi = lon
            return hp.ang2pix(int(nside), theta, phi, nest=False).astype(np.int64)

        counts_gaia = np.zeros(npix, dtype=np.int64)
        max_lines = args.gaia_max_lines
        n_read = 0
        buf_ra: list[float] = []
        buf_dec: list[float] = []
        chunk = 200000
        with gzip.open(qsocand_path, "rt", encoding="ascii", errors="ignore") as f:
            for line in f:
                n_read += 1
                if max_lines is not None and n_read > int(max_lines):
                    break
                pq = _parse_float_field(line, pqso_slice)
                if pq is None or pq < pqso_min:
                    continue
                ra = _parse_float_field(line, ra_slice)
                dec = _parse_float_field(line, dec_slice)
                if ra is None or dec is None:
                    continue
                buf_ra.append(float(ra))
                buf_dec.append(float(dec))
                if len(buf_ra) >= chunk:
                    v = unitvec_from_radec_deg(np.array(buf_ra), np.array(buf_dec))
                    ip = gal_ipix_from_unitvec_eq(v, nside=int(args.nside))
                    counts_gaia += np.bincount(ip, minlength=npix).astype(np.int64)
                    buf_ra.clear()
                    buf_dec.clear()
        if buf_ra:
            v = unitvec_from_radec_deg(np.array(buf_ra), np.array(buf_dec))
            ip = gal_ipix_from_unitvec_eq(v, nside=int(args.nside))
            counts_gaia += np.bincount(ip, minlength=npix).astype(np.int64)

        y_gaia_seen = counts_gaia[seen].astype(float)

        # Fit CMB-fixed dipole on Gaia using the same maximal nuisance basis projected ⟂ {1,u·n}.
        u_cmb = lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0]
        dip_mode_all = (n_seen @ u_cmb).astype(float)
        X_base_cmb = np.column_stack([np.ones_like(dip_mode_all), dip_mode_all])
        T_perp_cmb = proj_orthogonal_to(X_base_cmb, X_nuis_seen)
        T_perp_cmb_z = np.column_stack(
            [zscore(T_perp_cmb[:, j], np.ones(T_perp_cmb.shape[0], dtype=bool)) for j in range(T_perp_cmb.shape[1])]
        )
        X_gaia = np.column_stack([np.ones_like(y_gaia_seen), dip_mode_all, T_perp_cmb_z])
        prior_gaia = np.zeros(X_gaia.shape[1], dtype=float)
        prior_gaia[2:] = ridge_prec
        beta_gaia, _ = fit_poisson_glm(
            X_gaia,
            y_gaia_seen,
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=prior_gaia,
        )
        Dpar_gaia = float(beta_gaia[1])
        gaia_payload = {
            "pqso_min": pqso_min,
            "nside": int(args.nside),
            "n_total_used": int(np.sum(counts_gaia)),
            "n_seen_used": int(np.sum(counts_gaia[seen])),
            "cmb_fixed_fit": {
                "D_par": Dpar_gaia,
                "D_abs": float(abs(Dpar_gaia)),
                "axis_sep_cmb_deg": 0.0,
            },
            "notes": "Gaia qsocand dipole measured on the CatWISE Secrest-style seen footprint for comparability.",
        }
        write_json(gaia_json_path, gaia_payload)
    else:
        # Avoid stale artifacts if a previous run enabled Gaia.
        if gaia_json_path.exists():
            gaia_json_path.unlink()

    # Plots + summary.
    if not args.no_plots and args.do_scan:
        cuts_arr = np.array([r["w1_cut"] for r in scan_payload["baseline"]["rows"]], dtype=float)
        baseline_D = np.array([r["dipole"]["D"] for r in scan_payload["baseline"]["rows"]], dtype=float)
        baseline_sep = np.array([r["dipole"]["axis_sep_cmb_deg"] for r in scan_payload["baseline"]["rows"]], dtype=float)
        maximal_D = np.array([r["dipole"]["D"] for r in scan_payload["maximal"]["rows"]], dtype=float)
        maximal_sep = np.array([r["dipole"]["axis_sep_cmb_deg"] for r in scan_payload["maximal"]["rows"]], dtype=float)
        ext_D = np.array([r["dipole"]["D"] for r in scan_payload["external_gaia_template"]["rows"]], dtype=float)
        ext_sep = np.array(
            [r["dipole"]["axis_sep_cmb_deg"] for r in scan_payload["external_gaia_template"]["rows"]], dtype=float
        )

        plot_case_closed_figure(
            outpath=fig_dir / "case_closed_4panel.png",
            cuts=cuts_arr,
            baseline={"D": baseline_D, "axis_sep": baseline_sep},
            maximal={"D": maximal_D, "axis_sep": maximal_sep},
            external={"D": ext_D, "axis_sep": ext_sep},
            heldout=heldout_summary_for_fig,
        )

        cmb_Dpar = np.array([r["cmb_projection"]["D_par"] for r in scan_payload["cmb_fixed_maximal"]["rows"]], dtype=float)
        cmb_Dpar_ext = np.array(
            [r["cmb_projection"]["D_par"] for r in scan_payload["cmb_fixed_external_gaia_template"]["rows"]], dtype=float
        )
        plot_cmb_fixed_amplitude_scan(
            outpath=fig_dir / "cmb_fixed_amplitude_scan.png",
            cuts=cuts_arr,
            Dpar_maximal=cmb_Dpar,
            Dpar_external=cmb_Dpar_ext,
        )

        base_par = np.array([r["cmb_projection"]["D_par"] for r in scan_payload["baseline"]["rows"]], dtype=float)
        base_perp = np.array([r["cmb_projection"]["D_perp"] for r in scan_payload["baseline"]["rows"]], dtype=float)
        max_par = np.array([r["cmb_projection"]["D_par"] for r in scan_payload["maximal"]["rows"]], dtype=float)
        max_perp = np.array([r["cmb_projection"]["D_perp"] for r in scan_payload["maximal"]["rows"]], dtype=float)
        ext_par = np.array(
            [r["cmb_projection"]["D_par"] for r in scan_payload["external_gaia_template"]["rows"]], dtype=float
        )
        ext_perp = np.array(
            [r["cmb_projection"]["D_perp"] for r in scan_payload["external_gaia_template"]["rows"]], dtype=float
        )

        plot_cmb_projection_par_perp_scan(
            outpath=fig_dir / "cmb_projection_par_perp_scan.png",
            cuts=cuts_arr,
            series={
                "Baseline (free axis)": {"D_par": base_par, "D_perp": base_perp},
                "Maximal nuis (free axis)": {"D_par": max_par, "D_perp": max_perp},
                "Maximal + Gaia template": {"D_par": ext_par, "D_perp": ext_perp},
            },
            cmb_fixed={
                "CMB-fixed (maximal)": cmb_Dpar,
                "CMB-fixed + Gaia": cmb_Dpar_ext,
            },
        )

    # Compact coefficient table at representative cut (maximal model, in-sample).
    # Inclusive faint-cut semantics: W1 <= W1_max.
    nxt = int(np.searchsorted(w1_sorted, rep_cut, side="right"))
    counts_all = np.bincount(ipix_sorted[:nxt], minlength=npix).astype(np.int64)
    y_seen = counts_all[seen].astype(float)
    Xfull = np.column_stack([np.ones_like(y_seen), n_seen, X_nuis_seen])
    prior_full = np.zeros(Xfull.shape[1], dtype=float)
    if nuisance_prior_prec_arr is None:
        raise RuntimeError("Internal error: nuisance_prior_prec_arr is None but nuisance columns exist.")
    prior_full[4:] = nuisance_prior_prec_arr
    info_full: dict[str, Any] = {}
    beta_full, _ = fit_poisson_glm(
        Xfull,
        y_seen,
        offset=None,
        max_iter=int(args.max_iter),
        beta_init=None,
        prior_prec_diag=prior_full,
        out_info=info_full,
    )
    coeff_csv = data_dir / "coefficients_representative_cut.csv"
    with coeff_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "beta"])
        w.writerow(["intercept", f"{beta_full[0]:.10g}"])
        w.writerow(["b_x", f"{beta_full[1]:.10g}"])
        w.writerow(["b_y", f"{beta_full[2]:.10g}"])
        w.writerow(["b_z", f"{beta_full[3]:.10g}"])
        for name, val in zip(nuisance_names, beta_full[4:], strict=True):
            w.writerow([name, f"{float(val):.10g}"])

    # Representative-cut fits (free axis + CMB-fixed), plus constrained-null bootstrap calibration.
    rep_fits: dict[str, Any] = {}

    u_cmb = lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0]
    dip_mode_all = (n_seen @ u_cmb).astype(float)

    gaia_logp_z_seen = zscore(gaia_logp_map, seen)[seen]
    a_edge = float(alpha_edge_from_cumcounts(w1[base], rep_cut, delta=float(w1_step)))
    sdss_dm_seen = sdss_dm_map[seen].astype(float)
    sdss_dm_seen = sdss_dm_seen - float(np.median(sdss_dm_seen))
    sdss_t = sdss_dm_seen * a_edge
    sdss_sig = float(np.std(sdss_t)) if np.isfinite(np.std(sdss_t)) else 1.0
    if not np.isfinite(sdss_sig) or sdss_sig <= 0.0:
        sdss_sig = 1.0
    sdss_z_seen = (sdss_t - float(np.median(sdss_t))) / sdss_sig

    rep_fits["sdss_alpha_edge_at_cut"] = a_edge

    def _summarize_free_axis(beta: np.ndarray) -> dict[str, Any]:
        bvec = np.asarray(beta[1:4], dtype=float)
        D = float(np.linalg.norm(bvec))
        l_hat, b_hat = vec_to_lb(bvec)
        D_par = float(np.dot(bvec, u_cmb))
        D_perp = float(np.linalg.norm(bvec - D_par * u_cmb))
        return {
            "D": D,
            "lb_deg": [float(l_hat), float(b_hat)],
            "axis_sep_cmb_deg": float(axis_angle_deg(bvec, u_cmb)),
            "cmb_projection": {"D_par": D_par, "D_perp": D_perp},
        }

    rep_fits["free_axis_maximal"] = _summarize_free_axis(beta_full)
    rep_fits["free_axis_maximal_fit_opt"] = info_full

    def _fit_free_axis_with_extra(label: str, extra_cols: list[np.ndarray], extra_names: list[str]) -> None:
        if extra_cols:
            X = np.column_stack([np.ones_like(y_seen), n_seen, X_nuis_seen] + extra_cols)
            prior = np.zeros(X.shape[1], dtype=float)
            prior[4 : 4 + X_nuis_seen.shape[1]] = nuisance_prior_prec_arr
            prior[(4 + X_nuis_seen.shape[1]) :] = ridge_prec
            beta, _ = fit_poisson_glm(
                X,
                y_seen,
                offset=None,
                max_iter=int(args.max_iter),
                beta_init=None,
                prior_prec_diag=prior,
            )
            rep_fits[label] = _summarize_free_axis(beta) | {"extra_templates": extra_names}

    _fit_free_axis_with_extra("free_axis_plus_gaia_logp_template", [gaia_logp_z_seen], ["gaia_logp_offset_z"])
    _fit_free_axis_with_extra("free_axis_plus_sdss_delta_m_template", [sdss_z_seen], ["sdss_delta_m_alpha_edge_z"])
    _fit_free_axis_with_extra(
        "free_axis_plus_gaia_plus_sdss_templates",
        [gaia_logp_z_seen, sdss_z_seen],
        ["gaia_logp_offset_z", "sdss_delta_m_alpha_edge_z"],
    )

    def _fit_cmb_fixed(extra_cols: list[np.ndarray], extra_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = np.column_stack([np.ones_like(y_seen), dip_mode_all, X_nuis_seen] + extra_cols)
        prior = np.zeros(X.shape[1], dtype=float)
        prior[2 : 2 + X_nuis_seen.shape[1]] = nuisance_prior_prec_arr
        if extra_cols:
            prior[(2 + X_nuis_seen.shape[1]) :] = ridge_prec
        info: dict[str, Any] = {}
        beta, _ = fit_poisson_glm(
            X,
            y_seen,
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=prior,
            out_info=info,
        )
        rep_fits[f"fit_opt::{'+'.join(extra_names) if extra_names else 'cmb_fixed_maximal'}"] = info
        return beta, prior

    beta_cmb, prior_cmb = _fit_cmb_fixed([], [])
    Dpar_obs = float(beta_cmb[1])
    rep_fits["cmb_fixed_maximal"] = {
        "D_par": Dpar_obs,
        "D_abs": float(abs(Dpar_obs)),
        "axis_sep_cmb_deg": 0.0,
    }
    beta_cmb_gaia, prior_cmb_gaia = _fit_cmb_fixed([gaia_logp_z_seen], ["gaia_logp_offset_z"])
    rep_fits["cmb_fixed_plus_gaia_logp_template"] = {
        "D_par": float(beta_cmb_gaia[1]),
        "D_abs": float(abs(float(beta_cmb_gaia[1]))),
        "axis_sep_cmb_deg": 0.0,
    }

    # Constrained-null parametric bootstrap: simulate under (D_par = D_kin_ref) + fitted nuisance field,
    # then recover D_par with the same (CMB-fixed) model.
    bootstrap_payload: dict[str, Any] = {}
    if not bool(args.no_bootstrap):
        n_sim = int(args.bootstrap_nsim)
        if n_sim <= 0:
            raise SystemExit("--bootstrap-nsim must be > 0")
        D_true = float(args.bootstrap_kin_dpar)
        if not np.isfinite(D_true):
            raise SystemExit("--bootstrap-kin-dpar must be finite")

        # Fit nuisance field under constrained D_true by treating the dipole term as an offset.
        Xnull = np.column_stack([np.ones_like(y_seen), X_nuis_seen])
        prior_null = np.zeros(Xnull.shape[1], dtype=float)
        prior_null[1:] = nuisance_prior_prec_arr
        offset_null = dip_mode_all * D_true
        beta_null, _ = fit_poisson_glm(
            Xnull,
            y_seen,
            offset=offset_null,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=prior_null,
        )
        eta_null = offset_null + Xnull @ beta_null
        mu_null = np.exp(np.clip(eta_null, -25.0, 25.0))

        rng = np.random.default_rng(int(args.seed) + 991)
        dpar_sims = np.empty(n_sim, dtype=float)
        nit = np.empty(n_sim, dtype=int)
        success = np.zeros(n_sim, dtype=bool)
        grad_inf = np.empty(n_sim, dtype=float)
        progress_path = data_dir / "bootstrap_dpar_progress.json"
        t0 = time.time()
        Xfit = np.column_stack([np.ones_like(y_seen), dip_mode_all, X_nuis_seen])
        prior_fit = np.zeros(Xfit.shape[1], dtype=float)
        prior_fit[2:] = nuisance_prior_prec_arr
        beta_init = beta_cmb
        for i in range(n_sim):
            y_sim = rng.poisson(mu_null).astype(float)
            info: dict[str, Any] = {}
            beta_i, _ = fit_poisson_glm(
                Xfit,
                y_sim,
                offset=None,
                max_iter=int(args.bootstrap_max_iter),
                beta_init=beta_init,
                prior_prec_diag=prior_fit,
                out_info=info,
            )
            dpar_sims[i] = float(beta_i[1])
            success[i] = bool(info.get("success", True))
            nit[i] = int(info.get("nit", -1))
            grad_inf[i] = float(info.get("grad_inf_norm", float("nan")))
            beta_init = beta_i
            if (i + 1) % 25 == 0 or (i + 1) == n_sim:
                done = i + 1
                try:
                    write_json(
                        progress_path,
                        {
                            "done": int(done),
                            "nsim": int(n_sim),
                            "elapsed_s": float(time.time() - t0),
                            "success_frac": float(np.mean(success[:done])),
                            "grad_inf_p50": float(np.nanpercentile(grad_inf[:done], 50)),
                            "grad_inf_p90": float(np.nanpercentile(grad_inf[:done], 90)),
                            "nit_p50": float(np.percentile(nit[:done], 50)),
                            "nit_p90": float(np.percentile(nit[:done], 90)),
                            "nit_max": int(np.max(nit[:done])),
                            "dpar_sim_mean": float(np.mean(dpar_sims[:done])),
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass

        p_one_sided = float(np.mean(dpar_sims >= Dpar_obs))
        p_abs = float(np.mean(np.abs(dpar_sims) >= abs(Dpar_obs)))
        bootstrap_payload = {
            "nsim": int(n_sim),
            "D_true": D_true,
            "D_par_obs": Dpar_obs,
            "p_one_sided_ge_obs": p_one_sided,
            "p_abs_ge_obs": p_abs,
            "dpar_sim_summary": {
                "mean": float(np.mean(dpar_sims)),
                "p16": float(np.percentile(dpar_sims, 16)),
                "p50": float(np.percentile(dpar_sims, 50)),
                "p84": float(np.percentile(dpar_sims, 84)),
            },
            "fit_convergence": {
                "success_frac": float(np.mean(success)),
                "n_fail": int(np.sum(~success)),
                "nit_summary": {
                    "p50": float(np.percentile(nit, 50)),
                    "p90": float(np.percentile(nit, 90)),
                    "max": int(np.max(nit)),
                },
                "grad_inf_summary": {
                    "p50": float(np.nanpercentile(grad_inf, 50)),
                    "p90": float(np.nanpercentile(grad_inf, 90)),
                    "max": float(np.nanmax(grad_inf)),
                },
            },
        }
        write_json(data_dir / "bootstrap_dpar.json", bootstrap_payload)

        if not bool(args.no_plots):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.2))
            ax.hist(dpar_sims, bins=50, color="C0", alpha=0.75)
            ax.axvline(D_true, color="0.3", ls="--", lw=1.5, label=f"D_true={D_true:.4f}")
            ax.axvline(Dpar_obs, color="C3", lw=2, label=f"D_par,obs={Dpar_obs:.4f}")
            ax.set_xlabel("Recovered D_par (CMB-fixed)")
            ax.set_ylabel("Count")
            ax.set_title("Constrained-null parametric bootstrap for D_par")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=9)
            fig.tight_layout()
            fig.savefig(fig_dir / "bootstrap_dpar_hist.png", dpi=200)
            plt.close(fig)

    rep_fits["bootstrap_dpar"] = bootstrap_payload if bootstrap_payload else None

    # Consistency audit: scan at rep_cut must match the direct rep-cut fits (same model/data).
    # If this fails, treat it as a pipeline bug (typically non-converged scan fits or mismatched config),
    # not an astrophysical signal.
    if args.do_scan:
        def _find_scan_row(rows: list[dict[str, Any]], cut: float) -> dict[str, Any]:
            best = min(rows, key=lambda r: abs(float(r["w1_cut"]) - float(cut)))
            return best

        eps_D = 1e-4
        eps_axis = 0.5  # deg
        eps_par = 1e-4
        eps_perp = 1e-4

        scan_row_free = _find_scan_row(scan_payload["maximal"]["rows"], rep_cut)
        scan_row_cmb = _find_scan_row(scan_payload["cmb_fixed_maximal"]["rows"], rep_cut)

        rep_free = rep_fits["free_axis_maximal"]
        rep_cmb = rep_fits["cmb_fixed_maximal"]

        def _scan_opt_ok(row: dict[str, Any]) -> bool:
            opt = row.get("fit_opt", {})
            sel = int(opt.get("selected", -1))
            attempts = opt.get("attempts", [])
            if not isinstance(attempts, list) or sel < 0 or sel >= len(attempts):
                return False
            return bool(attempts[sel].get("success", False))

        diffs = {
            "free_axis_maximal": {
                "scan_cut": float(scan_row_free["w1_cut"]),
                "rep_cut": float(rep_cut),
                "scan": {
                    "D": float(scan_row_free["dipole"]["D"]),
                    "axis_sep_cmb_deg": float(scan_row_free["dipole"]["axis_sep_cmb_deg"]),
                    "D_par": float(scan_row_free["cmb_projection"]["D_par"]),
                    "D_perp": float(scan_row_free["cmb_projection"]["D_perp"]),
                },
                "rep": {
                    "D": float(rep_free["D"]),
                    "axis_sep_cmb_deg": float(rep_free["axis_sep_cmb_deg"]),
                    "D_par": float(rep_free["cmb_projection"]["D_par"]),
                    "D_perp": float(rep_free["cmb_projection"]["D_perp"]),
                },
            },
            "cmb_fixed_maximal": {
                "scan_cut": float(scan_row_cmb["w1_cut"]),
                "rep_cut": float(rep_cut),
                "scan": {"D_par": float(scan_row_cmb["cmb_projection"]["D_par"])},
                "rep": {"D_par": float(rep_cmb["D_par"])},
            },
            "tolerances": {
                "eps_D": float(eps_D),
                "eps_axis_deg": float(eps_axis),
                "eps_par": float(eps_par),
                "eps_perp": float(eps_perp),
            },
            "scan_opt_ok": {
                "free_axis_maximal": bool(_scan_opt_ok(scan_row_free)),
                "cmb_fixed_maximal": bool(_scan_opt_ok(scan_row_cmb)),
            },
        }

        def _bad() -> bool:
            if not diffs["scan_opt_ok"]["free_axis_maximal"]:
                return True
            if not diffs["scan_opt_ok"]["cmb_fixed_maximal"]:
                return True
            a = diffs["free_axis_maximal"]["scan"]
            b = diffs["free_axis_maximal"]["rep"]
            if abs(a["D"] - b["D"]) > eps_D:
                return True
            if abs(a["axis_sep_cmb_deg"] - b["axis_sep_cmb_deg"]) > eps_axis:
                return True
            if abs(a["D_par"] - b["D_par"]) > eps_par:
                return True
            if abs(a["D_perp"] - b["D_perp"]) > eps_perp:
                return True
            ac = diffs["cmb_fixed_maximal"]["scan"]["D_par"]
            bc = diffs["cmb_fixed_maximal"]["rep"]["D_par"]
            if abs(ac - bc) > eps_par:
                return True
            return False

        diffs["ok"] = not _bad()
        write_json(data_dir / "scan_rep_consistency.json", diffs)
        if not bool(diffs["ok"]):
            raise SystemExit(
                "Scan-vs-representative-cut mismatch detected (see data/scan_rep_consistency.json). "
                "Treat this as a pipeline inconsistency until resolved."
            )

    # Build master summary JSON.
    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "command": cmdline,
        "inputs": {
            "catalog": str(cat_path),
            "exclude_mask_fits": str(repo_root / str(args.exclude_mask_fits)) if args.exclude_mask_fits else None,
            "nside": int(args.nside),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_grid": str(args.w1_grid),
            "representative_cut": rep_cut,
            "ridge_sigma": float(args.ridge_sigma),
            "ridge_prec": float(ridge_prec),
            "harmonics": harmonic_meta,
            "bootstrap": {
                "nsim": None if bool(args.no_bootstrap) else int(args.bootstrap_nsim),
                "kin_dpar": float(args.bootstrap_kin_dpar),
                "max_iter": int(args.bootstrap_max_iter),
            },
            "maps": {
                "unwise_lognexp": lognexp_meta,
                "unwise_invvar": invvar_meta,
                "star_count": star_meta,
                "gaia_logp_offset": gaia_logp_meta,
                "sdss_delta_m": sdss_dm_meta,
            },
            "templates_maximal": nuisance_names,
        },
        "representative_cut_fits": rep_fits,
        "scan_suite": scan_payload if args.do_scan else None,
        "heldout_validation": heldout_payload if args.do_heldout else None,
        "loto_attribution": loto_payload if args.do_loto else None,
        "gaia_replication": gaia_payload if bool(args.do_gaia_replication) else None,
        "outputs": {
            "report_dir": str(report_dir),
            "case_closed_figure": str(fig_dir / "case_closed_4panel.png") if (not args.no_plots and args.do_scan) else None,
            "cmb_fixed_amplitude_scan": str(fig_dir / "cmb_fixed_amplitude_scan.png")
            if (not args.no_plots and args.do_scan)
            else None,
            "cmb_projection_par_perp_scan": str(fig_dir / "cmb_projection_par_perp_scan.png")
            if (not args.no_plots and args.do_scan)
            else None,
            "bootstrap_dpar_hist": str(fig_dir / "bootstrap_dpar_hist.png") if (not args.no_plots and (not bool(args.no_bootstrap))) else None,
            "scan_suite_json": str(data_dir / "scan_suite.json") if args.do_scan else None,
            "heldout_validation_json": str(data_dir / "heldout_validation.json") if args.do_heldout else None,
            "loto_attribution_json": str(data_dir / "loto_attribution.json") if args.do_loto else None,
            "coefficients_csv": str(data_dir / "coefficients_representative_cut.csv"),
            "bootstrap_dpar_json": str(data_dir / "bootstrap_dpar.json") if not bool(args.no_bootstrap) else None,
            "gaia_replication_json": str(gaia_json_path) if bool(args.do_gaia_replication) else None,
        },
    }
    write_json(data_dir / "summary.json", summary)

    # Write master_report.md (lightweight, points to artifacts).
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d (UTC)")
    lines: list[str] = []
    lines.append("# Case-closed maximal-nuisance dipole suite")
    lines.append("")
    lines.append(f"Date: {today}")
    lines.append("")
    lines.append("Goal: a single, auditable bundle testing whether the **percent-level CatWISE dipole** can be")
    lines.append("explained by **survey systematics** under a maximal reasonable nuisance basis, with held-out")
    lines.append("validation and external (cross-catalog) completeness templates.")
    lines.append("")
    lines.append("This suite is designed to be *hard to hand-wave away*: it includes a ridge-regularized maximal")
    lines.append("nuisance basis, an explicit CMB-fixed dipole fit (physical mode), a held-out sky check, and a")
    lines.append("leave-one-template-out ranking for attribution.")
    lines.append("")
    lines.append("## Headline (representative cut)")
    lines.append("")
    lines.append(f"- Representative cut: `W1_max = {rep_cut:.2f}`")
    lines.append(f"- Kinematic reference (repo convention): `D_kin ≈ {D_KIN_REF:.4f}` toward CMB `(l,b)=({CMB_L_DEG:.3f},{CMB_B_DEG:.3f})`")

    def _row_at(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
        if payload is None:
            return None
        blk = payload.get(key)
        if not isinstance(blk, dict) or "rows" not in blk:
            return None
        rows = blk["rows"]
        if not isinstance(rows, list) or not rows:
            return None
        for r in rows:
            if abs(float(r.get("w1_cut", float("nan"))) - float(rep_cut)) < 1e-9:
                return r
        return rows[-1]

    if args.do_scan:
        r_base = _row_at(scan_payload, "baseline")
        r_max = _row_at(scan_payload, "maximal")
        r_orth = _row_at(scan_payload, "maximal_orth")
        r_cmb = _row_at(scan_payload, "cmb_fixed_maximal")
        if r_base is not None:
            lines.append(
                f"- Baseline (free axis): `D={r_base['dipole']['D']:.5f}`, axis sep to CMB `={r_base['dipole']['axis_sep_cmb_deg']:.2f}°`"
            )
        if r_max is not None:
            lines.append(
                f"- Maximal nuis (free axis): `D={r_max['dipole']['D']:.5f}`, axis sep to CMB `={r_max['dipole']['axis_sep_cmb_deg']:.2f}°`"
            )
        if r_orth is not None:
            lines.append(
                f"- Maximal nuis (templates ⟂ {{1,nx,ny,nz}}): `D={r_orth['dipole']['D']:.5f}`, axis sep to CMB `={r_orth['dipole']['axis_sep_cmb_deg']:.2f}°`"
            )
        if r_cmb is not None:
            lines.append(f"- CMB-fixed (maximal nuis; priors): `D_par={r_cmb['cmb_projection']['D_par']:.5f}` (signed)")

    if args.do_heldout and isinstance(heldout_payload, dict) and "cmb_fixed_test" in heldout_payload:
        cmbt = heldout_payload["cmb_fixed_test"]
        lines.append(
            f"- Held-out CMB-fixed test (2-fold wedge split): `D_par,test={cmbt['D_par']:.5f}`, `|D|={cmbt['D_abs']:.5f}`"
        )

    if args.do_loto and isinstance(loto_payload, dict) and "rows" in loto_payload:
        top = loto_payload["rows"][:4]
        if top:
            lines.append("- LOTO attribution (top Δdeviance drivers):")
            for r in top:
                lines.append(f"  - `{r['dropped']}`: `Δdev={r['delta_deviance']:.2f}`")
    if rep_fits.get("bootstrap_dpar"):
        bp = rep_fits["bootstrap_dpar"]
        if bp:
            lines.append(
                f"- Bootstrap calibration (constrained D_true={bp['D_true']:.4f}): `p(D_par,sim ≥ D_par,obs)={bp['p_one_sided_ge_obs']:.3f}`"
            )
    if bool(args.do_gaia_replication) and gaia_payload:
        lines.append(
            f"- Gaia qsocand replication (PQSO≥{gaia_payload['pqso_min']:.2f}, same footprint): `|D_par|={gaia_payload['cmb_fixed_fit']['D_abs']:.5f}`"
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Summary JSON: `{data_dir / 'summary.json'}`")
    lines.append(f"- Scan suite JSON: `{data_dir / 'scan_suite.json'}`")
    if args.do_heldout:
        lines.append(f"- Held-out JSON: `{data_dir / 'heldout_validation.json'}`")
    if args.do_loto:
        lines.append(f"- LOTO JSON: `{data_dir / 'loto_attribution.json'}`")
        lines.append(f"- LOTO CSV: `{data_dir / 'loto_attribution.csv'}`")
    lines.append(f"- Coefficients CSV (rep cut): `{data_dir / 'coefficients_representative_cut.csv'}`")
    if not bool(args.no_bootstrap):
        lines.append(f"- Bootstrap JSON: `{data_dir / 'bootstrap_dpar.json'}`")
    if bool(args.do_gaia_replication):
        lines.append(f"- Gaia replication JSON: `{data_dir / 'gaia_replication.json'}`")
    if args.do_scan and not args.no_plots:
        lines.append(f"- Key figure: `REPORTS/case_closed_maximal_nuisance_suite/figures/case_closed_4panel.png`")
        lines.append(f"- CMB-fixed scan: `REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_fixed_amplitude_scan.png`")
        lines.append(f"- D_par/D_perp scan: `REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_projection_par_perp_scan.png`")
        if not bool(args.no_bootstrap):
            lines.append(f"- Bootstrap hist: `REPORTS/case_closed_maximal_nuisance_suite/figures/bootstrap_dpar_hist.png`")
        lines.append("")
        lines.append("![](REPORTS/case_closed_maximal_nuisance_suite/figures/case_closed_4panel.png)")
        lines.append("")
        lines.append("![](REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_fixed_amplitude_scan.png)")
        lines.append("")
        lines.append("![](REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_projection_par_perp_scan.png)")
        if not bool(args.no_bootstrap):
            lines.append("")
            lines.append("![](REPORTS/case_closed_maximal_nuisance_suite/figures/bootstrap_dpar_hist.png)")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("Run from repo root:")
    lines.append("")
    lines.append("```bash")
    # Prefer the venv interpreter if present (matches repo README).
    vpy = "./.venv/bin/python" if (repo_root / ".venv" / "bin" / "python").exists() else "python3"
    lines.append(f"{vpy} scripts/run_case_closed_maximal_nuisance_suite.py")
    lines.append("```")
    lines.append("")
    (report_dir / "master_report.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote report: {report_dir}")
    print(f"- {data_dir / 'summary.json'}")
    if args.do_scan and not args.no_plots:
        print(f"- {fig_dir / 'case_closed_4panel.png'}")
        print(f"- {fig_dir / 'cmb_fixed_amplitude_scan.png'}")
        print(f"- {fig_dir / 'cmb_projection_par_perp_scan.png'}")
        if not bool(args.no_bootstrap):
            print(f"- {fig_dir / 'bootstrap_dpar_hist.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
