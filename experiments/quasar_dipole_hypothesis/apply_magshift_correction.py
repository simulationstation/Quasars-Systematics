#!/usr/bin/env python3
"""
Apply an anisotropic magnitude correction to a CatWISE/Secrest-style catalog and re-measure the dipole.

This is intended as a *mechanism check*:
  If the observed number-count dipole is (partly) caused by a direction-dependent magnitude bias
  interacting with a hard faint-end cut (W1 <= W1_MAX), then applying the opposite correction
  should reduce the measured dipole amplitude.

Important:
  - This is NOT a full selection/systematics analysis.
  - It is a useful diagnostic once you have the real Secrest CatWISE catalog on disk.

Inputs:
  - FITS catalog with at least columns: l, b, w1, w1cov (Secrest+22 format)
  - An axis (l_axis, b_axis) and a correction amplitude delta_m_amp (mag)

Operation:
  - Apply baseline cuts *except* the faint W1 max cut.
  - Compute baseline dipole using the original W1 max cut.
  - Compute corrected dipole by replacing w1 -> w1 - sign * delta_m_amp * cos(theta_axis)
    and then reapplying the same W1 max cut.

We report both signs (+/-) and the sign that yields the smaller amplitude (purely descriptive).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from astropy.table import Table

from secrest_utils import apply_baseline_cuts, compute_dipole


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.radians(np.asarray(l_deg, dtype=float))
    b = np.radians(np.asarray(b_deg, dtype=float))
    cos_b = np.cos(b)
    return np.stack([cos_b * np.cos(l), cos_b * np.sin(l), np.sin(b)], axis=-1)


def unitvec_from_lb_scalar(l_deg: float, b_deg: float) -> np.ndarray:
    return lb_to_unitvec(np.array([l_deg], dtype=float), np.array([b_deg], dtype=float))[0]


def axis_angle_deg(l1: float, b1: float, l2: float, b2: float) -> float:
    u = unitvec_from_lb_scalar(l1, b1)
    v = unitvec_from_lb_scalar(l2, b2)
    dot = abs(float(np.dot(u, v)))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Path to Secrest/CatWISE FITS catalog (must have l,b,w1,w1cov).")
    ap.add_argument("--outdir", default=None, help="Output directory (default: outputs/quasar_magshift_correction).")
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-max", type=float, default=16.4)
    ap.add_argument("--w1-min", type=float, default=None)

    ap.add_argument("--axis-from", choices=["custom", "secrest", "cmb", "sn_best"], default="custom")
    ap.add_argument("--axis-l", type=float, default=None, help="Axis longitude (deg) if --axis-from=custom")
    ap.add_argument("--axis-b", type=float, default=None, help="Axis latitude (deg) if --axis-from=custom")
    ap.add_argument(
        "--secrest-json",
        default="REPORTS/Q_D_RES/secrest_reproduction_dipole.json",
        help="If --axis-from=secrest, read axis from this JSON (dipole.json-style artifact).",
    )
    ap.add_argument(
        "--sn-scan-json",
        default=(
            "outputs/horizon_anisotropy_fullscan_null100_dipoleT_field_axispar_nside4_surveyz_20260131_225012UTC/"
            "scan_summary.json"
        ),
        help="If --axis-from=sn_best, read axis from this scan_summary.json.",
    )

    ap.add_argument("--delta-m-amp", type=float, required=True, help="Dipole correction amplitude in magnitudes.")
    args = ap.parse_args()

    outdir = Path(args.outdir or "outputs/quasar_magshift_correction")
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve axis choice.
    if args.axis_from == "custom":
        if args.axis_l is None or args.axis_b is None:
            raise SystemExit("--axis-l and --axis-b are required when --axis-from=custom")
        axis_l = float(args.axis_l)
        axis_b = float(args.axis_b)
    elif args.axis_from == "secrest":
        secrest = json.load(open(args.secrest_json, "r"))
        axis_l = float(secrest["dipole"]["l_deg"])
        axis_b = float(secrest["dipole"]["b_deg"])
    elif args.axis_from == "cmb":
        axis_l = 264.021
        axis_b = 48.253
    else:
        scan = json.load(open(args.sn_scan_json, "r"))
        best = scan.get("best_axis") or {}
        axis_l = float(best["axis_l_deg"])
        axis_b = float(best["axis_b_deg"])

    tbl = Table.read(args.catalog)
    l = np.asarray(tbl["l"], dtype=float)
    b = np.asarray(tbl["b"], dtype=float)
    w1 = np.asarray(tbl["w1"], dtype=float)

    # Apply baseline cuts but *exclude* the faint-end W1 max cut so we can re-apply it after correction.
    mask_pre, cuts = apply_baseline_cuts(
        tbl,
        b_cut=float(args.b_cut),
        w1cov_min=float(args.w1cov_min),
        w1_max=None,
        w1_min=args.w1_min,
        existing_mask=None,
    )

    # Baseline sample with original W1 max cut.
    base_mask = mask_pre & (w1 <= float(args.w1_max))
    l0 = l[base_mask]
    b0 = b[base_mask]
    D0, l0_d, b0_d, _ = compute_dipole(l0, b0)

    # Correction is evaluated on the same pre-mask pool; selection changes with corrected W1.
    axis_vec = unitvec_from_lb_scalar(axis_l, axis_b)
    n_vec = lb_to_unitvec(l[mask_pre], b[mask_pre])
    cosang = np.clip(n_vec @ axis_vec, -1.0, 1.0)

    results: Dict[str, Any] = {
        "inputs": {
            "catalog": args.catalog,
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_min": args.w1_min,
            "w1_max": float(args.w1_max),
            "axis_from": args.axis_from,
            "axis_l_deg": axis_l,
            "axis_b_deg": axis_b,
            "delta_m_amp": float(args.delta_m_amp),
        },
        "baseline": {
            "N": int(base_mask.sum()),
            "dipole": {"amplitude": float(D0), "l_deg": float(l0_d), "b_deg": float(b0_d)},
        },
        "corrected": {},
        "notes": [
            "This is a descriptive mechanism check: selection is applied to corrected magnitudes.",
            "A professional test must also marginalize dust/depth/ecliptic templates and validate cuts.",
        ],
    }

    best_sign = None
    best_D = None
    for sign in [+1.0, -1.0]:
        # w1_corr = w1 - sign * delta_m_amp * cos(theta)
        w1_corr = w1[mask_pre] - sign * float(args.delta_m_amp) * cosang
        sel = w1_corr <= float(args.w1_max)

        l_sel = l[mask_pre][sel]
        b_sel = b[mask_pre][sel]
        D, l_d, b_d, _ = compute_dipole(l_sel, b_sel)

        key = "sign_plus" if sign > 0 else "sign_minus"
        results["corrected"][key] = {
            "sign": sign,
            "N": int(sel.sum()),
            "dipole": {"amplitude": float(D), "l_deg": float(l_d), "b_deg": float(b_d)},
        }
        if best_D is None or float(D) < float(best_D):
            best_D = float(D)
            best_sign = sign

    results["corrected"]["best_by_amplitude"] = {"sign": best_sign, "amplitude": best_D}

    # Quick direction comparison: axis vs baseline dipole.
    results["baseline"]["axis_angle_deg"] = axis_angle_deg(axis_l, axis_b, float(l0_d), float(b0_d))

    with open(outdir / "magshift_correction_result.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Wrote: {outdir / 'magshift_correction_result.json'}")
    print(f"Baseline: D={D0:.5f} (l,b)=({l0_d:.2f},{b0_d:.2f})  N={base_mask.sum():,}")
    for k in ["sign_plus", "sign_minus"]:
        r = results["corrected"][k]
        dd = r["dipole"]
        print(f"Corrected {k}: D={dd['amplitude']:.5f} (l,b)=({dd['l_deg']:.2f},{dd['b_deg']:.2f})  N={r['N']:,}")
    print(f"Best sign by amplitude: {best_sign}  D={best_D:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
