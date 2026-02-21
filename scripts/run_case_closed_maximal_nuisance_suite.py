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
) -> tuple[np.ndarray, np.ndarray | None]:
    """Poisson GLM (log link) via L-BFGS; optional diagonal Gaussian priors (ridge)."""
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

    prec = None if prior_prec_diag is None else np.asarray(prior_prec_diag, dtype=float).reshape(X.shape[1])
    mu_prior = None if prior_mean is None else np.asarray(prior_mean, dtype=float).reshape(X.shape[1])

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(off + X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        if prec is not None:
            dp = beta if mu_prior is None else (beta - mu_prior)
            nll = float(nll + 0.5 * np.sum(prec * dp * dp))
            grad = grad + prec * dp
        return nll, np.asarray(grad, dtype=float)

    res = minimize(
        lambda b: fun_and_grad(b)[0],
        beta0,
        jac=lambda b: fun_and_grad(b)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    beta = np.asarray(res.x, dtype=float)

    cov = None
    try:
        eta = np.clip(off + X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        fisher = X.T @ (mu[:, None] * X)
        if prec is not None:
            fisher = fisher + np.diag(prec)
        cov = np.linalg.inv(fisher)
    except Exception:  # noqa: BLE001
        cov = None

    return beta, cov


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


def scan_glm(
    *,
    w1_sorted: np.ndarray,
    ipix_sorted: np.ndarray,
    seen: np.ndarray,
    n_seen: np.ndarray,
    X_nuis_seen: np.ndarray,
    nuisance_names: list[str],
    offset_seen: np.ndarray | None,
    cuts: list[float],
    ridge_prec: float,
    max_iter: int,
    seed: int,
) -> tuple[list[ScanRow], dict[str, Any]]:
    """
    Run a cumulative scan and return rows + a small meta blob.

    Model columns are: [1, nx, ny, nz, nuis...]
    Ridge prior is applied only to nuisance columns (not intercept/dipole).
    """
    rng = np.random.default_rng(int(seed))
    _ = rng  # reserved for future stochastic diagnostics

    npix = int(seen.size)
    counts_all = np.zeros(npix, dtype=np.int64)
    cursor = 0
    beta_prev = None

    rows: list[ScanRow] = []
    for w1_cut in cuts:
        nxt = int(np.searchsorted(w1_sorted, float(w1_cut), side="left"))
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
            prior[4:] = float(ridge_prec)

        beta, _cov = fit_poisson_glm(
            X,
            y,
            offset=offset_seen,
            max_iter=int(max_iter),
            beta_init=beta_prev,
            prior_prec_diag=prior,
            prior_mean=None,
        )
        beta_prev = beta
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
            )
        )

    meta = {
        "nuisance_names": nuisance_names,
        "ridge_prec_nuisance": float(ridge_prec),
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
    offset_seen: np.ndarray | None,
    cuts: list[float],
    ridge_prec: float,
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
        nxt = int(np.searchsorted(w1_sorted, float(w1_cut), side="left"))
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
            prior[2:] = float(ridge_prec)

        beta, _ = fit_poisson_glm(
            X,
            y,
            offset=offset_seen,
            max_iter=int(max_iter),
            beta_init=beta_prev,
            prior_prec_diag=prior,
            prior_mean=None,
        )
        beta_prev = beta
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
            )
        )

    meta = {
        "nuisance_names": nuisance_names,
        "ridge_prec_nuisance": float(ridge_prec),
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
    panel(axes[0, 1], "Maximal nuisance GLM (ridge)", maximal["D"], maximal["axis_sep"])
    panel(axes[1, 1], "Maximal + external completeness offset", external["D"], external["axis_sep"])

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
    ax.plot(cuts, Dpar_maximal, color="C2", lw=2, label="CMB-fixed (maximal nuis, orth templates)")
    ax.plot(cuts, Dpar_external, color="C3", lw=2, label="CMB-fixed + Gaia offset")
    ax.set_xlabel("W1_max")
    ax.set_ylabel("Signed CMB-parallel dipole D_par")
    ax.set_title("CMB-fixed dipole amplitude vs W1_max")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
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

    # Standardized nuisance templates (seen pixels only).
    nuisance_cols: list[np.ndarray] = []
    nuisance_names: list[str] = []

    def add_t(name: str, arr: np.ndarray, *, transform: str = "z") -> None:
        nonlocal nuisance_cols, nuisance_names
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

    X_nuis_seen = np.column_stack(nuisance_cols) if nuisance_cols else np.zeros((n_seen.shape[0], 0), dtype=float)

    # Ridge prior precision for nuisance coefs (templates are standardized).
    if not np.isfinite(args.ridge_sigma) or float(args.ridge_sigma) <= 0.0:
        raise SystemExit("--ridge-sigma must be finite and > 0")
    ridge_prec = 1.0 / (float(args.ridge_sigma) ** 2)

    # Prepare cumulative selection ordering.
    order = np.argsort(w1[base])
    w1_sorted = w1[base][order]
    ipix_sorted = ipix_base[order]

    # Parse W1 grid.
    w1_start, w1_stop, w1_step = (float(x) for x in str(args.w1_grid).split(","))
    n_steps = int(round((w1_stop - w1_start) / w1_step)) + 1
    cuts = [w1_start + i * w1_step for i in range(n_steps)]

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
            offset_seen=None,
            cuts=cuts,
            ridge_prec=ridge_prec,
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
            offset_seen=None,
            cuts=cuts,
            ridge_prec=ridge_prec,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )

        # Cross-catalog external offset variant: Gaia logp offset added as fixed offset.
        gaia_offset_seen = gaia_logp_map[seen]
        gaia_offset_seen = gaia_offset_seen - float(np.median(gaia_offset_seen))
        external_rows, external_meta = scan_glm(
            w1_sorted=w1_sorted,
            ipix_sorted=ipix_sorted,
            seen=seen,
            n_seen=n_seen,
            X_nuis_seen=X_nuis_seen,
            nuisance_names=nuisance_names + ["gaia_logp_offset"],
            offset_seen=gaia_offset_seen,
            cuts=cuts,
            ridge_prec=ridge_prec,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )

        # Fixed-axis (CMB) scan with nuisance templates projected ⟂ {1, u_cmb·n}.
        u_cmb = lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0]
        dip_mode = (n_seen @ u_cmb).astype(float)
        X_base_cmb = np.column_stack([np.ones_like(dip_mode), dip_mode])
        T_perp_cmb = proj_orthogonal_to(X_base_cmb, X_nuis_seen)
        T_perp_cmb_z = np.column_stack(
            [zscore(T_perp_cmb[:, j], np.ones(T_perp_cmb.shape[0], dtype=bool)) for j in range(T_perp_cmb.shape[1])]
        )
        cmb_rows, cmb_meta = scan_glm_fixed_axis(
            w1_sorted=w1_sorted,
            ipix_sorted=ipix_sorted,
            seen=seen,
            dipole_mode_seen=dip_mode,
            X_nuis_seen=T_perp_cmb_z,
            nuisance_names=[f"{n}_perp_cmb" for n in nuisance_names],
            offset_seen=None,
            cuts=cuts,
            ridge_prec=ridge_prec,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )
        cmb_ext_rows, cmb_ext_meta = scan_glm_fixed_axis(
            w1_sorted=w1_sorted,
            ipix_sorted=ipix_sorted,
            seen=seen,
            dipole_mode_seen=dip_mode,
            X_nuis_seen=T_perp_cmb_z,
            nuisance_names=[f"{n}_perp_cmb" for n in nuisance_names] + ["gaia_logp_offset"],
            offset_seen=gaia_offset_seen,
            cuts=cuts,
            ridge_prec=ridge_prec,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )

        scan_payload = {
            "baseline": {"meta": baseline_meta, "rows": rows_to_json(baseline_rows)},
            "maximal": {"meta": maximal_meta, "rows": rows_to_json(maximal_rows)},
            "external_gaia": {"meta": external_meta, "rows": rows_to_json(external_rows)},
            "cmb_fixed_maximal": {"meta": cmb_meta, "rows": rows_to_json(cmb_rows)},
            "cmb_fixed_external_gaia": {"meta": cmb_ext_meta, "rows": rows_to_json(cmb_ext_rows)},
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
                offset_seen=None,
                cuts=cuts,
                ridge_prec=ridge_prec,
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
        nxt = int(np.searchsorted(w1_sorted, rep_cut, side="left"))
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
        prior_b[4:] = ridge_prec
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
        prior_m[4:] = ridge_prec
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

        # CMB-fixed held-out measurement: train nuisance-only on train with templates ⟂ {1, u_cmb·n},
        # then fit D_par on test with that nuisance field as an offset.
        u_cmb = lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0]
        dip_mode_all = (n_seen @ u_cmb).astype(float)
        X_base_cmb = np.column_stack([np.ones_like(dip_mode_all), dip_mode_all])
        T_perp_cmb = proj_orthogonal_to(X_base_cmb, X_nuis_seen)
        T_perp_cmb_z = np.column_stack(
            [zscore(T_perp_cmb[:, j], np.ones(T_perp_cmb.shape[0], dtype=bool)) for j in range(T_perp_cmb.shape[1])]
        )

        # Nuisance-only training fit: [1, T_perp].
        Xn_train = np.column_stack([np.ones_like(y_all_seen[train_idx]), T_perp_cmb_z[train_idx]])
        prior_n = np.zeros(Xn_train.shape[1], dtype=float)
        prior_n[1:] = ridge_prec
        beta_n_train, _ = fit_poisson_glm(
            Xn_train,
            y_all_seen[train_idx],
            offset=None,
            max_iter=int(args.max_iter),
            beta_init=None,
            prior_prec_diag=prior_n,
        )
        c_n_train = beta_n_train[1:]
        off_cmb_test = T_perp_cmb_z[test_idx] @ c_n_train

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
        nxt = int(np.searchsorted(w1_sorted, rep_cut, side="left"))
        counts_all = np.bincount(ipix_sorted[:nxt], minlength=npix).astype(np.int64)
        y_seen = counts_all[seen].astype(float)

        Xfull = np.column_stack([np.ones_like(y_seen), n_seen, X_nuis_seen])
        prior_full = np.zeros(Xfull.shape[1], dtype=float)
        prior_full[4:] = ridge_prec
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
            prior_j[4:] = ridge_prec
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
        ext_D = np.array([r["dipole"]["D"] for r in scan_payload["external_gaia"]["rows"]], dtype=float)
        ext_sep = np.array([r["dipole"]["axis_sep_cmb_deg"] for r in scan_payload["external_gaia"]["rows"]], dtype=float)

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
            [r["cmb_projection"]["D_par"] for r in scan_payload["cmb_fixed_external_gaia"]["rows"]], dtype=float
        )
        plot_cmb_fixed_amplitude_scan(
            outpath=fig_dir / "cmb_fixed_amplitude_scan.png",
            cuts=cuts_arr,
            Dpar_maximal=cmb_Dpar,
            Dpar_external=cmb_Dpar_ext,
        )

    # Compact coefficient table at representative cut (maximal model, in-sample).
    rep_cut = float(args.representative_cut)
    nxt = int(np.searchsorted(w1_sorted, rep_cut, side="left"))
    counts_all = np.bincount(ipix_sorted[:nxt], minlength=npix).astype(np.int64)
    y_seen = counts_all[seen].astype(float)
    Xfull = np.column_stack([np.ones_like(y_seen), n_seen, X_nuis_seen])
    prior_full = np.zeros(Xfull.shape[1], dtype=float)
    prior_full[4:] = ridge_prec
    beta_full, _ = fit_poisson_glm(
        Xfull,
        y_seen,
        offset=None,
        max_iter=int(args.max_iter),
        beta_init=None,
        prior_prec_diag=prior_full,
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

    # Representative-cut external-offset fits (cross-catalog robustness).
    rep_fits: dict[str, Any] = {}

    def _fit_with_offset(label: str, off: np.ndarray | None) -> None:
        beta, _ = fit_poisson_glm(
            Xfull,
            y_seen,
            offset=off,
            max_iter=int(args.max_iter),
            beta_init=beta_full,
            prior_prec_diag=prior_full,
        )
        bvec = beta[1:4]
        D = float(np.linalg.norm(bvec))
        l_hat, b_hat = vec_to_lb(bvec)
        rep_fits[label] = {
            "D": D,
            "lb_deg": [float(l_hat), float(b_hat)],
            "axis_sep_cmb_deg": float(
                axis_angle_deg(bvec, lb_to_unitvec(np.array([CMB_L_DEG]), np.array([CMB_B_DEG]))[0])
            ),
        }

    gaia_off = gaia_logp_map[seen].astype(float)
    gaia_off = gaia_off - float(np.median(gaia_off))

    a_edge = float(alpha_edge_from_cumcounts(w1[base], rep_cut, delta=float(w1_step)))
    sdss_dm_seen = sdss_dm_map[seen].astype(float)
    sdss_dm_seen = sdss_dm_seen - float(np.median(sdss_dm_seen))
    sdss_off = sdss_dm_seen * a_edge

    _fit_with_offset("maximal_no_external", None)
    _fit_with_offset("maximal_plus_gaia_logp_offset", gaia_off)
    _fit_with_offset("maximal_plus_sdss_delta_m_alpha_edge_offset", sdss_off)
    _fit_with_offset("maximal_plus_gaia_plus_sdss_offsets", gaia_off + sdss_off)
    rep_fits["sdss_alpha_edge_at_cut"] = a_edge

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
            "scan_suite_json": str(data_dir / "scan_suite.json") if args.do_scan else None,
            "heldout_validation_json": str(data_dir / "heldout_validation.json") if args.do_heldout else None,
            "loto_attribution_json": str(data_dir / "loto_attribution.json") if args.do_loto else None,
            "coefficients_csv": str(data_dir / "coefficients_representative_cut.csv"),
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
    lines.append("validation and external (cross-catalog) completeness offsets.")
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
            lines.append(
                f"- CMB-fixed (templates ⟂ {{1,u·n}}): `D_par={r_cmb['cmb_projection']['D_par']:.5f}` (signed)"
            )

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
    if bool(args.do_gaia_replication):
        lines.append(f"- Gaia replication JSON: `{data_dir / 'gaia_replication.json'}`")
    if args.do_scan and not args.no_plots:
        lines.append(f"- Key figure: `REPORTS/case_closed_maximal_nuisance_suite/figures/case_closed_4panel.png`")
        lines.append(f"- CMB-fixed scan: `REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_fixed_amplitude_scan.png`")
        lines.append("")
        lines.append("![](REPORTS/case_closed_maximal_nuisance_suite/figures/case_closed_4panel.png)")
        lines.append("")
        lines.append("![](REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_fixed_amplitude_scan.png)")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
