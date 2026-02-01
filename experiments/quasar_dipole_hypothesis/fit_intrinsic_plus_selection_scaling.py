#!/usr/bin/env python3
"""
Mixture-style dipole model across multiple faint cuts (W1_max):

  d_obs(W1_max)  ≈  d_intrinsic  +  alpha_edge(W1_max) * delta_m_vec

where:
  - d_obs is the *vector* number-count dipole (3-vector) from the catalog selection,
    computed via the usual vector-sum estimator:
        d_obs = 3 * mean(n_hat)
  - alpha_edge is the faint-edge slope at the cut:
        alpha_edge = d ln N / d m_max  ≈  n_edge(m_max) / (N * Δm)
    (units: 1/mag)
  - delta_m_vec is an effective *magnitude-cut dipole* (units: mag) that parameterizes
    selection/missingness that couples to the hard faint cut.
  - d_intrinsic is an "intrinsic" number-count dipole that does *not* scale with alpha_edge.

This is a concrete way to separate:
  - a selection-driven contribution (scales with alpha_edge)
  - from an intrinsic contribution (does not).

If delta_m_vec is stable and d_intrinsic is small, that supports a selection mechanism
as a plausible explanation pathway for the observed dipole.

Outputs:
  - scaling_fit.json
  - scaling_fit.png

Notes:
  - This is still an approximation (first-order response in a magnitude-limited sample).
  - It does not use per-object redshifts; it's purely a selection/cut-sensitivity test.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def vec_to_lb(vec: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size != 3:
        raise ValueError("expected 3-vector")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def estimate_alpha_edge(w1: np.ndarray, w1_max: float, dm: float) -> float:
    """
    Estimate alpha_edge = d ln N / d m_max using a single bin of width dm at the faint edge.
    """

    w1 = np.asarray(w1, dtype=float)
    sel_all = w1 <= float(w1_max)
    N = int(sel_all.sum())
    if N <= 0:
        return float("nan")
    lo = float(w1_max) - float(dm)
    n_edge = int(((w1 > lo) & (w1 <= float(w1_max))).sum())
    # n_edge / (dm) approximates n(m_max) (counts per mag) within the selected set.
    return float(n_edge) / (float(dm) * float(N))


@dataclass
class CutRow:
    w1_max: float
    N: int
    alpha_edge: float
    dvec: np.ndarray  # shape (3,)

    @property
    def D(self) -> float:
        return float(np.linalg.norm(self.dvec))

    @property
    def sigma_comp(self) -> float:
        # For isotropy, Var(d_component) ~= 3/N  (since d = 3 * mean(n), Var(mean(n_x)) = 1/(3N)).
        return math.sqrt(3.0 / float(self.N)) if self.N > 0 else float("nan")


def weighted_least_squares(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve argmin_beta sum_i w_i (y_i - X_i beta)^2.
    Returns (beta, cov_beta) with cov_beta based on an assumed known variance scale (1/w).
    """

    w = np.asarray(w, dtype=float)
    sw = np.sqrt(np.clip(w, 0.0, np.inf))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    XtWX = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(XtWX)
    return beta, cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)

    ap.add_argument("--w1max-grid", default="16.0,16.6,0.05", help="Grid spec 'start,stop,step'.")
    ap.add_argument("--alpha-dm", type=float, default=0.05, help="Bin width at faint edge for alpha estimate.")
    ap.add_argument("--min-N", type=int, default=200_000, help="Skip cuts with fewer than this many objects.")

    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/quasar_intrinsic_plus_selection_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Heavy import late.
    from astropy.table import Table

    tab = Table.read(args.catalog, memmap=True)
    l = np.asarray(tab["l"], dtype=float)
    b = np.asarray(tab["b"], dtype=float)
    w1 = np.asarray(tab["w1"], dtype=float)
    w1cov = np.asarray(tab["w1cov"], dtype=float)

    base = (
        np.isfinite(l)
        & np.isfinite(b)
        & np.isfinite(w1)
        & np.isfinite(w1cov)
        & (np.abs(b) > float(args.b_cut))
        & (w1cov >= float(args.w1cov_min))
    )

    l = l[base]
    b = b[base]
    w1 = w1[base]

    unit = lb_to_unitvec(l, b)

    # Parse grid.
    start_s, stop_s, step_s = [s.strip() for s in args.w1max_grid.split(",")]
    start, stop, step = float(start_s), float(stop_s), float(step_s)
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    w1max_values = start + step * np.arange(n)

    rows: List[CutRow] = []
    for w1_max in w1max_values:
        sel = w1 <= float(w1_max)
        N = int(sel.sum())
        if N < int(args.min_N):
            continue
        dvec = 3.0 * unit[sel].mean(axis=0)
        alpha = estimate_alpha_edge(w1[sel], float(w1_max), float(args.alpha_dm))
        rows.append(CutRow(w1_max=float(w1_max), N=N, alpha_edge=float(alpha), dvec=np.asarray(dvec, dtype=float)))

    if len(rows) < 3:
        raise SystemExit(f"Not enough cuts retained (got {len(rows)}). Try lowering --min-N or widening grid.")

    # Build vector regression d_j = d_intr + alpha_j * dm_vec.
    # Stack as 3*K observations.
    K = len(rows)
    y = np.zeros(3 * K, dtype=float)
    X = np.zeros((3 * K, 6), dtype=float)
    w = np.zeros(3 * K, dtype=float)

    for j, r in enumerate(rows):
        # Each component shares the same variance estimate.
        sig = r.sigma_comp
        wt = 1.0 / (sig * sig) if np.isfinite(sig) and sig > 0 else 1.0
        for k in range(3):
            idx = 3 * j + k
            y[idx] = float(r.dvec[k])
            # d_intr component
            X[idx, k] = 1.0
            # dm_vec component (scaled by alpha_edge)
            X[idx, 3 + k] = float(r.alpha_edge)
            w[idx] = wt

    beta, cov = weighted_least_squares(y, X, w)
    d_intr = beta[:3]
    dm_vec = beta[3:]

    # Parameter uncertainties from the WLS normal matrix (scale-free in this approximation).
    sigma_beta = np.sqrt(np.diag(cov))
    d_intr_sigma = sigma_beta[:3]
    dm_vec_sigma = sigma_beta[3:]

    # Derived: amplitudes and directions.
    d_intr_amp = float(np.linalg.norm(d_intr))
    d_intr_l, d_intr_b = vec_to_lb(d_intr)
    dm_amp = float(np.linalg.norm(dm_vec))
    dm_l, dm_b = vec_to_lb(dm_vec)

    # Residuals.
    pred = X @ beta
    resid = y - pred
    chi2 = float(np.sum(w * resid * resid))
    dof = int(max(0, y.size - beta.size))

    # Save per-cut summaries.
    cut_summaries = []
    for r in rows:
        cut_summaries.append(
            {
                "w1_max": r.w1_max,
                "N": r.N,
                "alpha_edge": r.alpha_edge,
                "D": r.D,
                "dvec": [float(x) for x in r.dvec],
                "sigma_comp": r.sigma_comp,
            }
        )

    out = {
        "inputs": {
            "catalog": args.catalog,
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1max_grid": args.w1max_grid,
            "alpha_dm": float(args.alpha_dm),
            "min_N": int(args.min_N),
            "K_cuts": int(K),
        },
        "fit": {
            "d_intr": {
                "vec": [float(x) for x in d_intr],
                "sigma_vec": [float(x) for x in d_intr_sigma],
                "amp": d_intr_amp,
                "l_deg": d_intr_l,
                "b_deg": d_intr_b,
            },
            "delta_m_vec": {
                "vec_mag_units": [float(x) for x in dm_vec],
                "sigma_vec_mag_units": [float(x) for x in dm_vec_sigma],
                "amp_mag": dm_amp,
                "l_deg": dm_l,
                "b_deg": dm_b,
            },
            "chi2": chi2,
            "dof": dof,
        },
        "cuts": cut_summaries,
        "notes": (
            "This is a first-order selection-scaling model: d_obs = d_intr + alpha_edge*delta_m_vec. "
            "delta_m_vec is interpretable as an effective magnitude-cut dipole that would generate the "
            "observed count dipole through the faint-end slope. Identifiability comes from varying W1_max "
            "(changing alpha_edge)."
        ),
    }

    with open(outdir / "scaling_fit.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    if args.make_plots:
        import matplotlib.pyplot as plt

        w1m = np.array([r.w1_max for r in rows])
        alpha = np.array([r.alpha_edge for r in rows])
        D = np.array([r.D for r in rows])

        # Predicted D along fitted vectors isn't directly comparable (vector vs norm),
        # but alpha scaling should show up in component projections.
        # We'll show projection of d_obs onto dm_vec direction.
        if dm_amp > 0:
            dm_hat = dm_vec / dm_amp
            proj = np.array([float(np.dot(r.dvec, dm_hat)) for r in rows])
            pred_proj = float(np.dot(d_intr, dm_hat)) + alpha * dm_amp
        else:
            proj = np.zeros_like(alpha)
            pred_proj = np.zeros_like(alpha)

        fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=200)

        axs[0].plot(w1m, D, "o-", lw=2)
        axs[0].set_xlabel("W1_max")
        axs[0].set_ylabel("D (|d_obs|)")
        axs[0].set_title("Measured dipole amplitude vs W1_max")

        axs[1].plot(w1m, alpha, "o-", lw=2)
        axs[1].set_xlabel("W1_max")
        axs[1].set_ylabel(r"$\alpha_{\rm edge}$ (1/mag)")
        axs[1].set_title("Faint-edge slope vs W1_max")

        axs[2].plot(alpha, proj, "o", label="data (proj onto dm_hat)")
        axs[2].plot(alpha, pred_proj, "-", lw=2, label="fit: proj = const + alpha*|dm|")
        axs[2].set_xlabel(r"$\alpha_{\rm edge}$ (1/mag)")
        axs[2].set_ylabel("projection of d_obs")
        axs[2].set_title("Scaling separation (intrinsic + selection)")
        axs[2].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(outdir / "scaling_fit.png")
        plt.close(fig)

    print(json.dumps(out["fit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
