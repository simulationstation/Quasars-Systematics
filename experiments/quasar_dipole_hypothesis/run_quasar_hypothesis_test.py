#!/usr/bin/env python3
"""
Quasar dipole vs "Spinning Foam" / horizon-anisotropy axis: quick hypothesis checks.

This script does NOT claim physical causation. It computes:
  - angular separations between:
      * Secrest/CatWISE dipole direction
      * CMB dipole direction
      * our latest SN-based horizon-anisotropy best axis (hemisphere scan)
      * our latest SN-based dipole-fit direction (fit to the axis z-score field)
  - the chance alignment p-value for an *axis* (sign-invariant) under isotropy
  - an optional Mollweide sky plot (Galactic coordinates)

Defaults are wired to paths already present in this repo:
  - REPORTS/Q_D_RES/secrest_reproduction_dipole.json
  - outputs/horizon_anisotropy_fullscan_null100_dipoleT_field_axispar_nside4_surveyz_20260131_225012UTC/scan_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple


CMB_L_DEG = 264.021
CMB_B_DEG = 48.253


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def unitvec_from_lb(l_deg: float, b_deg: float) -> Tuple[float, float, float]:
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    return (
        math.cos(b) * math.cos(l),
        math.cos(b) * math.sin(l),
        math.sin(b),
    )


def angle_deg(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def axis_angle_deg(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    # Axis has no sign: treat n and -n as equivalent.
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    dot = abs(dot)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def axis_alignment_p_value(theta_deg: float) -> float:
    """
    For an axis (sign-invariant), the statistic is theta = arccos(|n·m|) in [0, 90] degrees.

    Under isotropy, |cos(theta)| is uniform in [0,1], so:
      P(theta <= theta0) = 1 - cos(theta0)
    """

    th = math.radians(theta_deg)
    return 1.0 - math.cos(th)


def mollweide_lon_lat_from_lb(l_deg: float, b_deg: float) -> Tuple[float, float]:
    """
    Convert Galactic (l,b) to Mollweide long/lat in radians with a common astro convention:
      - l increases to the left, with l=0 at the center.

    This is purely for visualization (relative geometry).
    """

    # Map l in [0,360) to [-180,180)
    lon_deg = ((l_deg + 180.0) % 360.0) - 180.0
    # Flip so longitudes increase to the left.
    lon_deg = -lon_deg
    return (math.radians(lon_deg), math.radians(b_deg))


@dataclass(frozen=True)
class Axis:
    name: str
    l_deg: float
    b_deg: float

    def vec(self) -> Tuple[float, float, float]:
        return unitvec_from_lb(self.l_deg, self.b_deg)


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--secrest-json",
        default="REPORTS/Q_D_RES/secrest_reproduction_dipole.json",
        help="Path to Secrest/CatWISE dipole reproduction JSON.",
    )
    ap.add_argument(
        "--anisotropy-scan-json",
        default=(
            "outputs/horizon_anisotropy_fullscan_null100_dipoleT_field_axispar_nside4_surveyz_20260131_225012UTC/"
            "scan_summary.json"
        ),
        help="Path to our SN anisotropy scan_summary.json.",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: outputs/quasar_dipole_hypothesis_<timestamp>UTC).",
    )
    ap.add_argument("--no-plot", action="store_true", help="Skip sky plot generation.")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join("outputs", f"quasar_dipole_hypothesis_{utc_tag()}")
    os.makedirs(outdir, exist_ok=True)

    secrest = load_json(args.secrest_json)
    scan = load_json(args.anisotropy_scan_json)

    secrest_axis = Axis(
        "CatWISE/Secrest dipole (reproduced)",
        float(secrest["dipole"]["l_deg"]),
        float(secrest["dipole"]["b_deg"]),
    )
    cmb_axis = Axis("CMB dipole", CMB_L_DEG, CMB_B_DEG)

    best = scan.get("best_axis") or {}
    sn_best = Axis(
        "SN horizon-anisotropy best hemisphere axis",
        float(best.get("axis_l_deg", float("nan"))),
        float(best.get("axis_b_deg", float("nan"))),
    )

    dip = scan.get("dipole_fit") or {}
    # This is the dipole-fit direction of the z-score field (a *vector* direction).
    sn_dip_fit = Axis(
        "SN z-field dipole-fit direction",
        float(dip.get("D_l_deg", float("nan"))),
        float(dip.get("D_b_deg", float("nan"))),
    )

    axes = [secrest_axis, cmb_axis, sn_best, sn_dip_fit]

    def pairwise(a: Axis, b: Axis) -> Dict[str, float]:
        v1 = a.vec()
        v2 = b.vec()
        ang = angle_deg(v1, v2)
        ang_axis = axis_angle_deg(v1, v2)
        return {
            "angle_deg": ang,
            "axis_angle_deg": ang_axis,
            "p_axis_alignment": axis_alignment_p_value(ang_axis),
        }

    results = {
        "inputs": {
            "secrest_json": args.secrest_json,
            "anisotropy_scan_json": args.anisotropy_scan_json,
        },
        "axes": {a.name: {"l_deg": a.l_deg, "b_deg": a.b_deg} for a in axes},
        "comparisons": {
            "secrest_vs_cmb": pairwise(secrest_axis, cmb_axis),
            "secrest_vs_sn_best": pairwise(secrest_axis, sn_best),
            "secrest_vs_sn_dipfit": pairwise(secrest_axis, sn_dip_fit),
            "sn_best_vs_cmb": pairwise(sn_best, cmb_axis),
        },
        "notes": (
            "p_axis_alignment is the chance that a random *axis* (sign-invariant) is at least as aligned "
            "with Secrest as the measured axis_angle_deg (smaller is more aligned). This is NOT a physical "
            "causation claim; it is a geometric coincidence metric."
        ),
    }

    with open(os.path.join(outdir, "axis_alignment_summary.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(json.dumps(results, indent=2, sort_keys=True))

    if not args.no_plot:
        # Local import so the script still runs without matplotlib for text-only use.
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 4.5), dpi=200)
        ax = fig.add_subplot(111, projection="mollweide")
        ax.grid(True, alpha=0.3)

        colors = {
            secrest_axis.name: "#e74c3c",  # red
            cmb_axis.name: "#3498db",  # blue
            sn_best.name: "#2ecc71",  # green
            sn_dip_fit.name: "#f1c40f",  # yellow
        }

        for a in axes:
            lon, lat = mollweide_lon_lat_from_lb(a.l_deg, a.b_deg)
            ax.scatter([lon], [lat], s=80, color=colors.get(a.name, "k"), label=a.name, zorder=5)

        ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.25), fontsize=7, frameon=False)
        ax.set_title("Dipole/Axis Directions in Galactic Coordinates (Mollweide)", fontsize=10)

        out_png = os.path.join(outdir, "axis_alignment_mollweide.png")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
