#!/usr/bin/env python3
"""
Fit a magnitude-dipole correction amplitude delta_m *with* nuisance templates.

This is the next "sensible" step after the toy correction:
  - We keep the Secrest/CatWISE hard faint cut (W1 <= W1_MAX).
  - We apply a direction-dependent magnitude correction relative to a chosen axis:
        W1_corr = W1 - sign * delta_m * cos(theta_axis)
    which is equivalent to selecting with a position-dependent threshold:
        W1 <= W1_MAX + sign * delta_m * cos(theta_axis)
  - For each delta_m we:
      (a) compute the raw vector-sum dipole of the selected sample, and
      (b) fit a HEALPix-binned Poisson WLS model for counts that includes:
            N_pix = A + B·n_hat + c1*T_ebv + c2*T_|elat| + c3*T_w1cov
          and report the residual dipole D = |B|/A after marginalizing templates.

We then report the delta_m that minimizes the residual dipole after templates.

Notes:
  - This is still a mechanism check, not a publication-grade systematics model.
  - It is designed to answer: "does the required delta_m survive once dust/ecliptic/coverage are marginalized?"
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import healpy as hp
import numpy as np
from astropy.table import Table

from secrest_utils import apply_baseline_cuts


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


def solve_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    sw = np.sqrt(np.clip(w, 0.0, np.inf))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    chi2 = float(np.sum(w * resid * resid))
    dof = int(max(0, X.shape[0] - X.shape[1]))
    sigma2 = chi2 / dof if dof > 0 else float("nan")
    XtWX = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(XtWX) * sigma2
    return beta, cov, chi2, dof


@dataclass
class FitRow:
    delta_m: float
    N: int
    raw_amp: float
    raw_l: float
    raw_b: float
    raw_proj_axis: float
    fit_amp: float
    fit_amp_sigma: float
    fit_l: float
    fit_b: float
    fit_proj_axis: float
    chi2: float
    dof: int


def dipole_from_unit(unit: np.ndarray, sel: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    sel = np.asarray(sel, dtype=bool)
    n = int(sel.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), np.array([np.nan, np.nan, np.nan])
    sum_vec = unit[sel].sum(axis=0)
    dip_vec = 3.0 * sum_vec / float(n)
    amp = float(np.linalg.norm(dip_vec))
    l, b = vec_to_lb(dip_vec)
    return amp, l, b, dip_vec


def counts_fit_dipole(
    Np: np.ndarray,
    pix_valid: np.ndarray,
    pix_unit: np.ndarray,
    z_ebv: np.ndarray,
    z_abs_elat: np.ndarray,
    z_w1cov: np.ndarray,
) -> Tuple[float, float, float, float, float, float, int]:
    """
    Fit count-level model with WLS weights w=1/N and return:
      (amp, amp_sigma, l_deg, b_deg, proj_axis_placeholder, chi2, dof)
    proj_axis is handled outside with a provided axis vector (depends on axis).
    """
    y = Np[pix_valid].astype(float)
    w = 1.0 / np.clip(y, 1.0, np.inf)
    X = np.column_stack(
        [
            np.ones_like(y),
            pix_unit[pix_valid, 0],
            pix_unit[pix_valid, 1],
            pix_unit[pix_valid, 2],
            z_ebv[pix_valid],
            z_abs_elat[pix_valid],
            z_w1cov[pix_valid],
        ]
    )
    beta, cov, chi2, dof = solve_wls(X, y, w)
    A = float(beta[0])
    B = np.asarray(beta[1:4], dtype=float)
    dvec = B / A if A != 0.0 else np.array([np.nan, np.nan, np.nan])
    amp = float(np.linalg.norm(dvec))
    l, b = vec_to_lb(dvec)

    # Delta-method sigma for amp = |B|/A.
    amp_sigma = float("nan")
    if A != 0.0 and np.all(np.isfinite(cov[:4, :4])):
        covA = float(cov[0, 0])
        covB = np.asarray(cov[1:4, 1:4], dtype=float)
        covAB = np.asarray(cov[0, 1:4], dtype=float)
        normB = float(np.linalg.norm(B))
        if normB > 0:
            u = B / normB
            dD_dA = -normB / (A * A)
            dD_dB = u / A
            var = (dD_dA**2) * covA + float(dD_dB @ covB @ dD_dB) + 2.0 * float(dD_dA * (covAB @ dD_dB))
            amp_sigma = float(math.sqrt(max(0.0, var)))

    return amp, amp_sigma, l, b, chi2, dof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Secrest/CatWISE FITS (expects l,b,w1,w1cov,ebv,elat).")
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-min", type=float, default=None)
    ap.add_argument("--w1-max", type=float, default=16.4)

    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--delta-m-max", type=float, default=0.03)
    ap.add_argument("--grid-n", type=int, default=61)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--make-plots", action="store_true")

    ap.add_argument("--axis-from", choices=["custom", "secrest", "cmb", "sn_best"], default="secrest")
    ap.add_argument("--axis-l", type=float, default=None)
    ap.add_argument("--axis-b", type=float, default=None)
    ap.add_argument(
        "--secrest-json",
        default="Q_D_RES/secrest_reproduction_dipole.json",
        help="Axis source when --axis-from=secrest.",
    )
    ap.add_argument(
        "--sn-scan-json",
        default=(
            "outputs/horizon_anisotropy_fullscan_null100_dipoleT_field_axispar_nside4_surveyz_20260131_225012UTC/"
            "scan_summary.json"
        ),
        help="Axis source when --axis-from=sn_best.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir or "outputs/quasar_magshift_with_templates")
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve axis.
    if args.axis_from == "custom":
        if args.axis_l is None or args.axis_b is None:
            raise SystemExit("--axis-l/--axis-b required for --axis-from=custom")
        axis_l, axis_b = float(args.axis_l), float(args.axis_b)
    elif args.axis_from == "secrest":
        axis = json.load(open(args.secrest_json, "r"))
        axis_l = float(axis["dipole"]["l_deg"])
        axis_b = float(axis["dipole"]["b_deg"])
    elif args.axis_from == "cmb":
        axis_l, axis_b = 264.021, 48.253
    else:
        scan = json.load(open(args.sn_scan_json, "r"))
        best = scan.get("best_axis") or {}
        axis_l = float(best["axis_l_deg"])
        axis_b = float(best["axis_b_deg"])
    axis_vec = unitvec_from_lb(axis_l, axis_b)

    tbl = Table.read(args.catalog)
    l_all = np.asarray(tbl["l"], dtype=float)
    b_all = np.asarray(tbl["b"], dtype=float)
    w1_all = np.asarray(tbl["w1"], dtype=float)
    ebv_all = np.asarray(tbl["ebv"], dtype=float)
    elat_all = np.asarray(tbl["elat"], dtype=float)
    w1cov_all = np.asarray(tbl["w1cov"], dtype=float)

    # Baseline "pre" cuts (exclude W1 max so selection changes with delta_m).
    pre_mask, cuts = apply_baseline_cuts(
        tbl,
        b_cut=float(args.b_cut),
        w1cov_min=float(args.w1cov_min),
        w1_max=None,
        w1_min=args.w1_min,
        existing_mask=None,
    )

    l = l_all[pre_mask]
    b = b_all[pre_mask]
    w1 = w1_all[pre_mask]
    ebv = ebv_all[pre_mask]
    elat = elat_all[pre_mask]
    w1cov = w1cov_all[pre_mask]

    unit = lb_to_unitvec(l, b)
    cosang = np.clip(unit @ axis_vec, -1.0, 1.0)

    nside = int(args.nside)
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - b), np.deg2rad(l % 360.0), nest=False)

    # Precompute pixel center unit vectors for regression.
    th_all, ph_all = hp.pix2ang(nside, np.arange(npix), nest=False)
    bcent = (np.pi / 2.0) - th_all
    lcent = ph_all
    pix_unit = np.column_stack([np.cos(bcent) * np.cos(lcent), np.cos(bcent) * np.sin(lcent), np.sin(bcent)])

    # Compute template means per pixel from the *pre-mask* pool (fixed across delta_m).
    pre_counts = np.bincount(pix, minlength=npix).astype(float)
    pre_valid = pre_counts > 0

    def mean_by_pix(values: np.ndarray) -> np.ndarray:
        s = np.bincount(pix, weights=values, minlength=npix).astype(float)
        out = np.zeros(npix, dtype=float)
        out[pre_valid] = s[pre_valid] / pre_counts[pre_valid]
        return out

    t_ebv = mean_by_pix(ebv)
    t_abs_elat = np.abs(mean_by_pix(elat))
    t_w1cov = mean_by_pix(w1cov)

    def zscore_template(t: np.ndarray) -> np.ndarray:
        tv = t[pre_valid]
        m = float(tv.mean())
        s = float(tv.std(ddof=0))
        if s == 0.0:
            return np.zeros_like(t)
        z = (t - m) / s
        z[~pre_valid] = 0.0
        return z

    z_ebv = zscore_template(t_ebv)
    z_abs_elat = zscore_template(t_abs_elat)
    z_w1cov = zscore_template(t_w1cov)

    def run_grid(delta_grid: np.ndarray, sign: float) -> Dict[str, Any]:
        rows: list[Dict[str, Any]] = []
        for dm in delta_grid:
            # W1_corr <= W1_max  <=>  W1 <= W1_max + sign*dm*cos(theta)
            thresh = float(args.w1_max) + (sign * float(dm)) * cosang
            sel = w1 <= thresh
            N = int(sel.sum())

            raw_amp, raw_l, raw_b, raw_dvec = dipole_from_unit(unit, sel)
            raw_proj = float(raw_dvec @ axis_vec) if np.all(np.isfinite(raw_dvec)) else float("nan")

            Np = np.bincount(pix[sel], minlength=npix).astype(float)
            pix_valid = Np > 0
            fit_amp, fit_sigma, fit_l, fit_b, chi2, dof = counts_fit_dipole(
                Np, pix_valid, pix_unit, z_ebv, z_abs_elat, z_w1cov
            )
            # projection of fitted dipole vector onto axis
            # fitted dvec is unit * amp; reconstruct from (l,b,amp)
            if np.isfinite(fit_amp):
                # dvec direction from fit_l, fit_b
                ddir = unitvec_from_lb(fit_l, fit_b)
                fit_proj = float((ddir * fit_amp) @ axis_vec)
            else:
                fit_proj = float("nan")

            rows.append(
                {
                    "delta_m": float(dm),
                    "N": N,
                    "raw_amp": raw_amp,
                    "raw_l_deg": raw_l,
                    "raw_b_deg": raw_b,
                    "raw_proj_axis": raw_proj,
                    "fit_amp": fit_amp,
                    "fit_amp_sigma": fit_sigma,
                    "fit_l_deg": fit_l,
                    "fit_b_deg": fit_b,
                    "fit_proj_axis": fit_proj,
                    "chi2": chi2,
                    "dof": dof,
                }
            )

        # Choose best by minimizing fit_amp
        best = min(rows, key=lambda r: r["fit_amp"] if np.isfinite(r["fit_amp"]) else float("inf"))
        best_by_chi2 = min(rows, key=lambda r: r["chi2"] if np.isfinite(r["chi2"]) else float("inf"))
        return {"sign": float(sign), "rows": rows, "best_by_fit_amp": best, "best_by_chi2": best_by_chi2}

    # Coarse grid and optional refine around best fit_amp.
    grid = np.linspace(0.0, float(args.delta_m_max), int(args.grid_n))
    scan_plus = run_grid(grid, sign=+1.0)
    scan_minus = run_grid(grid, sign=-1.0)

    scan_plus_ref: Dict[str, Any] | None = None
    scan_minus_ref: Dict[str, Any] | None = None
    if bool(args.refine):
        best0 = min(scan_plus["best_by_fit_amp"], scan_minus["best_by_fit_amp"], key=lambda r: r["fit_amp"])
        dm0 = float(best0["delta_m"])
        lo = max(0.0, dm0 - 0.01)
        hi = min(float(args.delta_m_max), dm0 + 0.01)
        grid2 = np.linspace(lo, hi, 101)
        scan_plus_ref = run_grid(grid2, sign=+1.0)
        scan_minus_ref = run_grid(grid2, sign=-1.0)

    out = {
        "inputs": {
            "catalog": args.catalog,
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_min": args.w1_min,
            "w1_max": float(args.w1_max),
            "nside": nside,
            "delta_m_max": float(args.delta_m_max),
            "grid_n": int(args.grid_n),
            "refine": bool(args.refine),
            "axis_from": args.axis_from,
            "axis_l_deg": axis_l,
            "axis_b_deg": axis_b,
            "templates": ["ebv", "|elat|", "w1cov"],
        },
        "cuts": cuts,
        "scan": {"sign_plus": scan_plus, "sign_minus": scan_minus},
        "scan_refined": {"sign_plus": scan_plus_ref, "sign_minus": scan_minus_ref},
    }

    with open(outdir / "magshift_with_templates.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    if bool(args.make_plots):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def to_arrays(rows: list[Dict[str, Any]]):
            dm = np.array([r["delta_m"] for r in rows], dtype=float)
            raw = np.array([r["raw_amp"] for r in rows], dtype=float)
            fit = np.array([r["fit_amp"] for r in rows], dtype=float)
            fit_sig = np.array([r["fit_amp_sigma"] for r in rows], dtype=float)
            return dm, raw, fit, fit_sig

        plt.figure(figsize=(10, 6))
        for label, scan in [("+", scan_plus), ("-", scan_minus)]:
            dm, raw, fit, fit_sig = to_arrays(scan["rows"])
            plt.plot(dm, raw, label=f"raw dipole (sign {label})", alpha=0.6)
            plt.plot(dm, fit, label=f"fit dipole+templates (sign {label})")
            if np.all(np.isfinite(fit_sig)):
                plt.fill_between(dm, fit - fit_sig, fit + fit_sig, alpha=0.15)
        plt.xlabel("delta_m [mag]")
        plt.ylabel("dipole amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "magshift_with_templates.png", dpi=200)
        plt.close()

    best = min(scan_plus["best_by_fit_amp"], scan_minus["best_by_fit_amp"], key=lambda r: r["fit_amp"])
    best_sign = "+" if best is scan_plus["best_by_fit_amp"] else "-"
    print(
        f"Best after templates: sign={best_sign} delta_m={best['delta_m']:.5f} fit_amp={best['fit_amp']:.5f} "
        f"(fit l,b)=({best['fit_l_deg']:.2f},{best['fit_b_deg']:.2f}) raw_amp={best['raw_amp']:.5f}"
    )
    print(f"Wrote: {outdir / 'magshift_with_templates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
