#!/usr/bin/env python3
"""
Fit a 2-component dipole model across multiple W1_max cuts, with a *fixed* selection axis.

Model:
  d_obs(W1_max)  ≈  d_intrinsic  +  alpha_edge(W1_max) * delta_m_amp * n_axis

where:
  - d_obs is the 3-vector number-count dipole from vector-sum estimator
  - alpha_edge is d ln N / d m_max (1/mag) estimated from the faint edge
  - n_axis is a *fixed* unit vector (default: Secrest dipole axis)
  - delta_m_amp is the magnitude-cut dipole amplitude (mag)
  - d_intrinsic is the remaining "intrinsic" dipole vector (dimensionless)

This version is much better conditioned than the fully-free vector decomposition, because the
selection component direction is fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def unitvec_from_lb(l_deg: float, b_deg: float) -> np.ndarray:
    return lb_to_unitvec(np.array([l_deg], dtype=float), np.array([b_deg], dtype=float))[0]


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
    w1 = np.asarray(w1, dtype=float)
    sel_all = w1 <= float(w1_max)
    N = int(sel_all.sum())
    if N <= 0:
        return float("nan")
    lo = float(w1_max) - float(dm)
    n_edge = int(((w1 > lo) & (w1 <= float(w1_max))).sum())
    return float(n_edge) / (float(dm) * float(N))


def load_axis_from_secrest_json(path: str) -> Tuple[float, float]:
    d = json.load(open(path, "r"))
    return float(d["dipole"]["l_deg"]), float(d["dipole"]["b_deg"])


@dataclass
class CutRow:
    w1_max: float
    N: int
    alpha_edge: float
    dvec: np.ndarray

    @property
    def D(self) -> float:
        return float(np.linalg.norm(self.dvec))

    @property
    def sigma_comp(self) -> float:
        return math.sqrt(3.0 / float(self.N)) if self.N > 0 else float("nan")


def wls(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w, dtype=float)
    sw = np.sqrt(np.clip(w, 0.0, np.inf))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    return beta, cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)

    ap.add_argument("--w1max-grid", default="15.6,16.6,0.05")
    ap.add_argument("--alpha-dm", type=float, default=0.05)
    ap.add_argument("--min-N", type=int, default=200_000)

    ap.add_argument("--axis-from", choices=["secrest", "custom", "cmb"], default="secrest")
    ap.add_argument(
        "--secrest-json",
        default="REPORTS/Q_D_RES/secrest_reproduction_dipole.json",
    )
    ap.add_argument("--axis-l", type=float, default=None)
    ap.add_argument("--axis-b", type=float, default=None)
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/quasar_intrinsic_plus_selection_fixedaxis_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.axis_from == "custom":
        if args.axis_l is None or args.axis_b is None:
            raise SystemExit("--axis-l/--axis-b required for --axis-from=custom")
        axis_l, axis_b = float(args.axis_l), float(args.axis_b)
    elif args.axis_from == "cmb":
        axis_l, axis_b = 264.021, 48.253
    else:
        axis_l, axis_b = load_axis_from_secrest_json(args.secrest_json)
    axis_unit = unitvec_from_lb(axis_l, axis_b)

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

    if len(rows) < 4:
        raise SystemExit(f"Not enough cuts retained (got {len(rows)}). Try lowering --min-N or widening grid.")

    K = len(rows)
    # Unknowns: d_intr (3) + dm_amp (1)  => 4 params
    y = np.zeros(3 * K, dtype=float)
    X = np.zeros((3 * K, 4), dtype=float)
    w = np.zeros(3 * K, dtype=float)

    for j, r in enumerate(rows):
        sig = r.sigma_comp
        wt = 1.0 / (sig * sig) if np.isfinite(sig) and sig > 0 else 1.0
        for k in range(3):
            idx = 3 * j + k
            y[idx] = float(r.dvec[k])
            X[idx, k] = 1.0  # d_intr[k]
            X[idx, 3] = float(r.alpha_edge) * float(axis_unit[k])  # dm_amp * alpha * axis_hat[k]
            w[idx] = wt

    beta, cov = wls(y, X, w)
    d_intr = beta[:3]
    dm_amp = float(beta[3])

    sigma = np.sqrt(np.diag(cov))
    d_intr_sigma = sigma[:3]
    dm_amp_sigma = float(sigma[3])

    d_intr_amp = float(np.linalg.norm(d_intr))
    d_intr_l, d_intr_b = vec_to_lb(d_intr)

    # Residual fit quality.
    resid = y - X @ beta
    chi2 = float(np.sum(w * resid * resid))
    dof = int(max(0, y.size - beta.size))

    out = {
        "inputs": {
            "catalog": args.catalog,
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1max_grid": args.w1max_grid,
            "alpha_dm": float(args.alpha_dm),
            "min_N": int(args.min_N),
            "axis_from": args.axis_from,
            "axis_l_deg": float(axis_l),
            "axis_b_deg": float(axis_b),
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
            "delta_m_amp_mag": {"value": float(dm_amp), "sigma": float(dm_amp_sigma)},
            "chi2": chi2,
            "dof": dof,
        },
        "cuts": [
            {
                "w1_max": r.w1_max,
                "N": r.N,
                "alpha_edge": r.alpha_edge,
                "D": r.D,
                "dvec": [float(x) for x in r.dvec],
                "sigma_comp": r.sigma_comp,
            }
            for r in rows
        ],
        "notes": (
            "Fixed-axis scaling model: d_obs = d_intr + alpha_edge*dm_amp*axis_hat. "
            "Identifiability comes from changing alpha_edge by varying W1_max."
        ),
    }

    with open(outdir / "fixed_axis_scaling_fit.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    if args.make_plots:
        import matplotlib.pyplot as plt

        w1m = np.array([r.w1_max for r in rows])
        alpha = np.array([r.alpha_edge for r in rows])
        D = np.array([r.D for r in rows])

        # project observed dipole onto axis_hat
        proj = np.array([float(np.dot(r.dvec, axis_unit)) for r in rows])
        pred_proj = float(np.dot(d_intr, axis_unit)) + alpha * dm_amp

        fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=200)
        axs[0].plot(w1m, D, "o-", lw=2)
        axs[0].set_xlabel("W1_max")
        axs[0].set_ylabel("D (|d_obs|)")
        axs[0].set_title("Measured dipole amplitude vs W1_max")

        axs[1].plot(w1m, alpha, "o-", lw=2)
        axs[1].set_xlabel("W1_max")
        axs[1].set_ylabel(r"$\alpha_{\rm edge}$ (1/mag)")
        axs[1].set_title("Faint-edge slope vs W1_max")

        axs[2].plot(alpha, proj, "o", label="data: proj(d_obs, axis)")
        axs[2].plot(alpha, pred_proj, "-", lw=2, label="fit: proj = const + alpha*dm_amp")
        axs[2].set_xlabel(r"$\alpha_{\rm edge}$ (1/mag)")
        axs[2].set_ylabel("projection onto axis")
        axs[2].set_title("Fixed-axis scaling separation")
        axs[2].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(outdir / "fixed_axis_scaling_fit.png")
        plt.close(fig)

    print(json.dumps(out["fit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
