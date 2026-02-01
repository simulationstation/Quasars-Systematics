#!/usr/bin/env python3
"""
Fit a direction-dependent magnitude modulation amplitude (delta_m) that minimizes the measured dipole.

Step 3:
  Treat the observed dipole as potentially arising from a tiny direction-dependent magnitude bias
  interacting with a hard faint cut (W1 <= W1_MAX). Fit delta_m (mag) by scanning amplitudes and
  measuring the resulting catalog dipole after applying the correction and re-applying the faint cut.

Mechanism:
  For axis direction n_axis and object direction n_i:
    cos(theta_i) = n_i dot n_axis
  Apply correction (two possible signs):
    W1_corr = W1 - sign * delta_m * cos(theta)
  Then select by W1_corr <= W1_MAX and compute the dipole of the selected sample.

Outputs:
  - JSON with baseline dipole and a scan table over delta_m for sign=+1 and sign=-1.
  - Optional PNG with amplitude and |axis projection| vs delta_m.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Secrest/CatWISE FITS (expects l,b,w1,w1cov).")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-max", type=float, default=16.4)
    ap.add_argument("--w1-min", type=float, default=None)

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

    ap.add_argument("--delta-m-max", type=float, default=0.05)
    ap.add_argument("--grid-n", type=int, default=101, help="Number of points in [0, delta-m-max].")
    ap.add_argument("--refine", action="store_true", help="Refine around the best coarse solution.")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir or "outputs/quasar_magshift_fit")
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

    # Apply baseline cuts but exclude W1 max so selection can change with delta_m.
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

    unit = lb_to_unitvec(l, b)
    cosang = np.clip(unit @ axis_vec, -1.0, 1.0)

    # Baseline dipole with raw W1 max selection.
    base_sel = w1 <= float(args.w1_max)
    sum_base = unit[base_sel].sum(axis=0)
    N_base = int(base_sel.sum())
    dip_base = 3.0 * sum_base / float(N_base)
    D_base = float(np.linalg.norm(dip_base))
    l_base, b_base = vec_to_lb(dip_base)
    proj_base = float(dip_base @ axis_vec)

    def eval_scan(delta_grid: np.ndarray, sign: float) -> Dict[str, Any]:
        amps: list[float] = []
        projs: list[float] = []
        Ns: list[int] = []
        dirs: list[Tuple[float, float]] = []
        for dm in delta_grid:
            w1corr = w1 - sign * float(dm) * cosang
            sel = w1corr <= float(args.w1_max)
            N = int(sel.sum())
            if N == 0:
                amps.append(float("nan"))
                projs.append(float("nan"))
                Ns.append(0)
                dirs.append((float("nan"), float("nan")))
                continue
            sumv = unit[sel].sum(axis=0)
            dip = 3.0 * sumv / float(N)
            amps.append(float(np.linalg.norm(dip)))
            projs.append(float(dip @ axis_vec))
            dirs.append(vec_to_lb(dip))
            Ns.append(N)

        i_best = int(np.nanargmin(amps))
        best_l, best_b = dirs[i_best]
        return {
            "sign": float(sign),
            "delta_m_grid": delta_grid.tolist(),
            "amplitude": amps,
            "proj_axis": projs,
            "N": Ns,
            "dipole_lb": [[float(x), float(y)] for (x, y) in dirs],
            "best_by_amplitude": {
                "delta_m": float(delta_grid[i_best]),
                "amplitude": float(amps[i_best]),
                "proj_axis": float(projs[i_best]),
                "N": int(Ns[i_best]),
                "l_deg": float(best_l),
                "b_deg": float(best_b),
            },
        }

    # Coarse scan
    grid = np.linspace(0.0, float(args.delta_m_max), int(args.grid_n))
    scan_plus = eval_scan(grid, sign=+1.0)
    scan_minus = eval_scan(grid, sign=-1.0)

    # Optional refinement around the global best from the coarse scan.
    scan_plus_ref: Dict[str, Any] | None = None
    scan_minus_ref: Dict[str, Any] | None = None
    if bool(args.refine):
        best_coarse = min(
            scan_plus["best_by_amplitude"], scan_minus["best_by_amplitude"], key=lambda r: r["amplitude"]
        )
        dm0 = float(best_coarse["delta_m"])
        lo = max(0.0, dm0 - 0.01)
        hi = min(float(args.delta_m_max), dm0 + 0.01)
        grid2 = np.linspace(lo, hi, 101)
        scan_plus_ref = eval_scan(grid2, sign=+1.0)
        scan_minus_ref = eval_scan(grid2, sign=-1.0)

    result = {
        "inputs": {
            "catalog": args.catalog,
            "axis_from": args.axis_from,
            "axis_l_deg": axis_l,
            "axis_b_deg": axis_b,
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_min": args.w1_min,
            "w1_max": float(args.w1_max),
            "delta_m_max": float(args.delta_m_max),
            "grid_n": int(args.grid_n),
            "refine": bool(args.refine),
        },
        "baseline": {
            "N": N_base,
            "amplitude": D_base,
            "l_deg": l_base,
            "b_deg": b_base,
            "proj_axis": proj_base,
        },
        "scan": {"sign_plus": scan_plus, "sign_minus": scan_minus},
        "scan_refined": {"sign_plus": scan_plus_ref, "sign_minus": scan_minus_ref},
    }

    with open(outdir / "magshift_fit.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    if bool(args.make_plots):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot_one(scan: Dict[str, Any], label: str):
            dm = np.asarray(scan["delta_m_grid"], dtype=float)
            amp = np.asarray(scan["amplitude"], dtype=float)
            proj = np.asarray(scan["proj_axis"], dtype=float)
            plt.plot(dm, amp, label=f"amp sign {label}")
            plt.plot(dm, np.abs(proj), linestyle="--", label=f"|proj| sign {label}")

        plt.figure(figsize=(10, 5))
        plot_one(scan_plus, "+")
        plot_one(scan_minus, "-")
        plt.axhline(D_base, color="k", alpha=0.3, label="baseline amp")
        plt.xlabel("delta_m amplitude [mag]")
        plt.ylabel("dipole amplitude / |axis projection|")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "magshift_fit.png", dpi=200)
        plt.close()

    best = min(scan_plus["best_by_amplitude"], scan_minus["best_by_amplitude"], key=lambda r: r["amplitude"])
    print(
        f"Baseline: D={D_base:.5f} axis_proj={proj_base:.5f} (l,b)=({l_base:.2f},{b_base:.2f}) N={N_base:,}"
    )
    # Determine sign label for printing
    best_sign_label = "+" if best is scan_plus["best_by_amplitude"] else "-"
    print(
        f"Best by amplitude: sign={best_sign_label} delta_m={best['delta_m']:.5f} -> "
        f"D={best['amplitude']:.5f} axis_proj={best['proj_axis']:.5f} "
        f"(l,b)=({best['l_deg']:.2f},{best['b_deg']:.2f}) N={best['N']:,}"
    )
    print(f"Wrote: {outdir / 'magshift_fit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
