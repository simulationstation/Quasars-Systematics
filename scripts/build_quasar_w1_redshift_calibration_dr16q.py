#!/usr/bin/env python3
"""Build quasar W1->redshift calibration priors from SDSS DR16Q."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dr16q-fits",
        default="data/external/sdss_dr16q/DR16Q_v4.fits",
    )
    ap.add_argument("--z-col", default="Z")
    ap.add_argument("--w1-col", default="W1_MAG")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument(
        "--out-calibration",
        default="configs/shared_redshift_calibration_from_dr16q.json",
    )
    ap.add_argument("--w1-ref", type=float, default=16.4)
    ap.add_argument("--w1-cut-min", type=float, default=15.5)
    ap.add_argument("--w1-cut-max", type=float, default=16.6)
    ap.add_argument("--w1-cut-step", type=float, default=0.05)
    ap.add_argument("--z-min", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=7.0)
    ap.add_argument("--min-count", type=int, default=500)
    return ap.parse_args()


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    w = 1.0 / np.asarray(sigma, dtype=float)
    xw = x * w[:, None]
    yw = y * w
    beta, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = y - x @ beta
    chi2 = float(np.sum((resid / sigma) ** 2))
    dof = int(y.size - x.shape[1])
    fisher = xw.T @ xw
    cov = np.linalg.inv(fisher)
    if dof > 0:
        cov = cov * max(1.0, chi2 / dof)
    return beta, cov, chi2, dof


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / (
        args.run_tag if args.run_tag else f"shared_redshift_quasar_calibration_{utc_tag()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    path = Path(args.dr16q_fits).resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing DR16Q fits: {path}")

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        cols = {c.name.upper(): c.name for c in hdul[1].columns}
    z_key = cols.get(str(args.z_col).upper())
    w1_key = cols.get(str(args.w1_col).upper())
    if z_key is None:
        raise KeyError(f"column not found: {args.z_col}")
    if w1_key is None:
        raise KeyError(f"column not found: {args.w1_col}")

    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        z_all = np.asarray(d[z_key], dtype=float)
        w1_all = np.asarray(d[w1_key], dtype=float)

    mask = np.isfinite(z_all) & np.isfinite(w1_all)
    mask &= (z_all >= float(args.z_min)) & (z_all <= float(args.z_max))
    z = z_all[mask]
    w1 = w1_all[mask]
    if z.size == 0:
        raise RuntimeError("no valid DR16Q rows after filtering")

    cuts = np.arange(
        float(args.w1_cut_min),
        float(args.w1_cut_max) + 0.5 * float(args.w1_cut_step),
        float(args.w1_cut_step),
    )

    curve_rows: list[dict[str, Any]] = []
    for cut in cuts:
        m = w1 <= float(cut)
        n = int(np.sum(m))
        if n < int(args.min_count):
            continue
        z_sub = z[m]
        z16, z50, z84 = [float(x) for x in np.percentile(z_sub, [16.0, 50.0, 84.0])]
        robust_sigma = 0.5 * max(0.0, z84 - z16)
        sigma_med = 1.253 * robust_sigma / math.sqrt(max(1, n))
        sigma_med = max(sigma_med, 1e-4)
        curve_rows.append(
            {
                "w1_cut": float(cut),
                "n": n,
                "z_p16": z16,
                "z_p50": z50,
                "z_p84": z84,
                "sigma_median_est": float(sigma_med),
            }
        )

    if len(curve_rows) < 3:
        raise RuntimeError("insufficient W1 cuts with enough rows for calibration fit")

    x = np.asarray([r["w1_cut"] - float(args.w1_ref) for r in curve_rows], dtype=float)
    y = np.asarray([r["z_p50"] for r in curve_rows], dtype=float)
    sigma = np.asarray([r["sigma_median_est"] for r in curve_rows], dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    beta, cov, chi2, dof = _weighted_lstsq(design, y, sigma)
    z_ref_mu = float(beta[0])
    slope_mu = float(beta[1])
    z_ref_sigma = float(math.sqrt(max(0.0, cov[0, 0])))
    slope_sigma = float(math.sqrt(max(0.0, cov[1, 1])))

    z_ref_lo = max(0.01, z_ref_mu - 5.0 * max(z_ref_sigma, 1e-3))
    z_ref_hi = z_ref_mu + 5.0 * max(z_ref_sigma, 1e-3)
    if z_ref_hi <= z_ref_lo:
        z_ref_hi = z_ref_lo + 0.1
    slope_lo = max(0.0, slope_mu - 5.0 * max(slope_sigma, 1e-3))
    slope_hi = slope_mu + 5.0 * max(slope_sigma, 1e-3)
    if slope_hi <= slope_lo:
        slope_hi = slope_lo + 0.1

    calibration = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "catalog": str(path),
            "z_col": str(z_key),
            "w1_col": str(w1_key),
            "selection": {
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
                "w1_cut_min": float(args.w1_cut_min),
                "w1_cut_max": float(args.w1_cut_max),
                "w1_cut_step": float(args.w1_cut_step),
                "min_count": int(args.min_count),
            },
            "n_rows_used": int(z.size),
        },
        "quasar": {
            "w1_ref": float(args.w1_ref),
            "z_ref": {
                "mu": z_ref_mu,
                "sigma": z_ref_sigma,
                "bounds": [z_ref_lo, z_ref_hi],
            },
            "slope_per_mag": {
                "mu": slope_mu,
                "sigma": slope_sigma,
                "bounds": [slope_lo, slope_hi],
            },
            "fit_quality": {
                "chi2": float(chi2),
                "dof": int(dof),
                "chi2_red": float(chi2 / dof) if dof > 0 else float("nan"),
            },
        },
        "curve": curve_rows,
        "notes": [
            "Curve is cumulative: z statistics for W1 <= cut.",
            "Uncertainty per point is an estimated median error from robust spread and sample size.",
            "Use this as a calibration prior, not as a final population model.",
        ],
    }

    out_cal = Path(args.out_calibration)
    out_cal.parent.mkdir(parents=True, exist_ok=True)
    out_cal.write_text(json.dumps(calibration, indent=2) + "\n")
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    with (out_dir / "quasar_w1_redshift_curve.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["w1_cut", "n", "z_p16", "z_p50", "z_p84", "sigma_median_est"])
        for r in curve_rows:
            w.writerow([r["w1_cut"], r["n"], r["z_p16"], r["z_p50"], r["z_p84"], r["sigma_median_est"]])

    # Plot calibration curve and linear fit.
    w1_grid = np.linspace(float(args.w1_cut_min), float(args.w1_cut_max), 200)
    y_fit = z_ref_mu + slope_mu * (w1_grid - float(args.w1_ref))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.errorbar(
        [r["w1_cut"] for r in curve_rows],
        [r["z_p50"] for r in curve_rows],
        yerr=[r["sigma_median_est"] for r in curve_rows],
        fmt="o",
        alpha=0.85,
        label="DR16Q cumulative medians",
    )
    ax.plot(w1_grid, y_fit, lw=2.0, label="Weighted linear fit")
    ax.set_xlabel("W1 cumulative cut")
    ax.set_ylabel("Effective redshift proxy")
    ax.set_title("Quasar redshift calibration from DR16Q")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "quasar_w1_redshift_calibration.png", dpi=180)
    plt.close(fig)

    md = []
    md.append("# Quasar W1 Redshift Calibration (DR16Q)")
    md.append("")
    md.append(f"- Source: `{path}`")
    md.append(f"- Used rows: `{z.size}`")
    md.append(f"- `z_ref(mu,sigma)` at W1_ref={args.w1_ref}: `{z_ref_mu:.6f} ± {z_ref_sigma:.6f}`")
    md.append(f"- `slope_per_mag(mu,sigma)`: `{slope_mu:.6f} ± {slope_sigma:.6f}`")
    md.append(f"- Fit chi2/dof: `{(chi2 / dof) if dof > 0 else float('nan'):.6f}`")
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append(f"- Calibration JSON: `{out_cal.resolve()}`")
    md.append(f"- Curve CSV: `{(out_dir / 'quasar_w1_redshift_curve.csv').resolve()}`")
    md.append(f"- Figure: `{(out_dir / 'quasar_w1_redshift_calibration.png').resolve()}`")
    (out_dir / "master_report.md").write_text("\n".join(md) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "out_calibration": str(out_cal.resolve()),
                "z_ref_mu": z_ref_mu,
                "z_ref_sigma": z_ref_sigma,
                "slope_mu": slope_mu,
                "slope_sigma": slope_sigma,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

