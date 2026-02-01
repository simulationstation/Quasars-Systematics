#!/usr/bin/env python3
"""Compute how big an (anisotropic) magnitude shift must be to explain the CatWISE/Secrest dipole.

This is a forward *plausibility* check, not a full systematics-controlled analysis.

For a magnitude-limited catalog with faint-end cut m_max, a small shift in the effective
selection threshold produces (to first order):

  d ln N / d m_max = n(m_max) / N
  delta N / N ~= (d ln N / d m_max) * delta m

If we assume a dipolar modulation in the effective magnitude cut

  delta m(n_hat) = delta_m_amp * cos(theta)

then the induced number-count dipole amplitude is approximately

  D_counts ~= alpha_edge * delta_m_amp

where alpha_edge = d ln N / d m at the faint edge.

We estimate alpha_edge from the quasar-dipole-fun Stage-5 quick-mode W1 binning
(external/quasar-dipole-fun/results/slicing_quick/w1_bins.json), which provides N per
magnitude bin for a 200k-source subset.

Outputs:
  - required_magshift_summary.json
  - w1_density_log.png

Caveats:
  - CatWISE dipole measurements are extremely systematics-sensitive (masking, dust, depth, ecliptic).
  - This calculation only answers order-of-magnitude feasibility for a magnitude-shift mechanism.
  - A proper test needs an explicit selection/systematics model and (ideally) redshift information.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def parse_bin_label(label: str) -> Tuple[float, float]:
    """Parse labels like "[15.922, 16.130)" or "[16.284, 16.400]" -> (lo, hi)."""

    m = re.match(r"^[\[(]\s*([0-9.]+)\s*,\s*([0-9.]+)\s*[\])]\s*$", label.strip())
    if not m:
        raise ValueError(f"Unrecognized bin label: {label!r}")
    return float(m.group(1)), float(m.group(2))


@dataclass(frozen=True)
class MagBin:
    label: str
    lo: float
    hi: float
    n: int

    @property
    def width(self) -> float:
        return float(self.hi - self.lo)

    @property
    def density_per_mag(self) -> float:
        w = self.width
        return float(self.n) / w if w > 0 else float("nan")


def unitvec_from_lb(l_deg: float, b_deg: float) -> Tuple[float, float, float]:
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    return (
        math.cos(b) * math.cos(l),
        math.cos(b) * math.sin(l),
        math.sin(b),
    )


def axis_angle_deg(l1: float, b1: float, l2: float, b2: float) -> float:
    """Axis angle: sign-invariant (n ~ -n), range [0, 90] degrees."""

    u = unitvec_from_lb(l1, b1)
    v = unitvec_from_lb(l2, b2)
    dot = abs(u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def mag_to_frac_distance(delta_m: float) -> float:
    """delta d_L / d_L corresponding to a distance-modulus shift delta_m."""

    # mu = 5 log10(d_L) + const => delta mu = (5/ln10) delta ln d_L
    return (math.log(10.0) / 5.0) * delta_m


def estimate_alpha_edge(bins_sorted: List[MagBin], n_total: int, k_edge: int) -> Tuple[float, float, List[str]]:
    """Return (n_edge_per_mag, alpha_edge, labels_used)."""

    k = max(1, min(int(k_edge), len(bins_sorted)))
    edge = bins_sorted[-k:]
    n_edge_per_mag = sum(b.density_per_mag for b in edge) / float(len(edge))
    alpha = n_edge_per_mag / float(n_total)
    return n_edge_per_mag, alpha, [b.label for b in edge]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--secrest-json",
        default="Q_D_RES/secrest_reproduction_dipole.json",
        help="Secrest/CatWISE dipole reproduction JSON (for D_obs and direction).",
    )
    ap.add_argument(
        "--w1-bins-json",
        default=None,
        help=(
            "Optional: precomputed W1 binning JSON (contains N per magnitude bin). "
            "If omitted, this script cannot run (it is a back-of-envelope check)."
        ),
    )
    ap.add_argument(
        "--sn-scan-json",
        default=(
            "outputs/horizon_anisotropy_fullscan_null100_dipoleT_field_axispar_nside4_surveyz_20260131_225012UTC/"
            "scan_summary.json"
        ),
        help="Optional: our SN anisotropy scan_summary.json (for axis projection bookkeeping).",
    )
    ap.add_argument(
        "--edge-bins",
        type=int,
        default=2,
        help="How many faint-end magnitude bins to average to estimate n(m_max).",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: outputs/quasar_dipole_magshift_<timestamp>UTC).",
    )
    args = ap.parse_args()

    outdir = args.outdir or os.path.join("outputs", f"quasar_dipole_magshift_{utc_tag()}")
    os.makedirs(outdir, exist_ok=True)

    secrest = json.load(open(args.secrest_json, "r"))
    D_obs = float(secrest["dipole"]["amplitude"])
    l_qso = float(secrest["dipole"]["l_deg"])
    b_qso = float(secrest["dipole"]["b_deg"])

    if not args.w1_bins_json:
        raise RuntimeError("--w1-bins-json is required (this repo does not ship the quick-binning artifact).")

    w1 = json.load(open(args.w1_bins_json, "r"))
    baseline = w1.get("baseline") or {}
    n_total = int(baseline.get("N", 0))
    if n_total <= 0:
        raise RuntimeError("w1_bins_json missing baseline.N")

    bins: List[MagBin] = []
    for row in w1.get("bins", []):
        lo, hi = parse_bin_label(str(row["label"]))
        bins.append(MagBin(label=str(row["label"]), lo=lo, hi=hi, n=int(row["N"])))
    if not bins:
        raise RuntimeError("w1_bins_json missing bins[]")

    bins_sorted = sorted(bins, key=lambda b: b.hi)

    n_edge_per_mag, alpha_edge, edge_labels = estimate_alpha_edge(bins_sorted, n_total, int(args.edge_bins))

    # D_counts ~= alpha_edge * delta_m_amp => delta_m_amp ~= D_obs / alpha_edge
    delta_m_amp = D_obs / alpha_edge
    frac_dL_amp = mag_to_frac_distance(delta_m_amp)

    sens: Dict[str, Any] = {}
    for k in range(1, min(4, len(bins_sorted)) + 1):
        n_edge_k, alpha_k, labels_k = estimate_alpha_edge(bins_sorted, n_total, k)
        dm_k = D_obs / alpha_k
        sens[f"edge_bins_{k}"] = {
            "labels": labels_k,
            "n_edge_per_mag": n_edge_k,
            "alpha_edge_dlnN_dm": alpha_k,
            "delta_m_amp_mag": dm_k,
            "frac_dL_amp": mag_to_frac_distance(dm_k),
        }

    # Optional: compare to our SN anisotropy best-axis direction.
    sn_axis = None
    sn_vs_secrest = None
    sn_projection_scaling = None
    if args.sn_scan_json and os.path.exists(args.sn_scan_json):
        scan = json.load(open(args.sn_scan_json, "r"))
        best = scan.get("best_axis") or {}
        if "axis_l_deg" in best and "axis_b_deg" in best:
            l_sn = float(best["axis_l_deg"])
            b_sn = float(best["axis_b_deg"])
            sn_axis = {"l_deg": l_sn, "b_deg": b_sn}
            theta = axis_angle_deg(l_sn, b_sn, l_qso, b_qso)
            c = math.cos(math.radians(theta))
            sn_vs_secrest = {"axis_angle_deg": theta, "cos_axis_angle": c}
            if c > 1e-6:
                scale = 1.0 / c
                dm_proj = delta_m_amp * scale
                sn_projection_scaling = {
                    "scale_factor": scale,
                    "delta_m_amp_mag": dm_proj,
                    "delta_m_peak_to_trough_mag": 2.0 * dm_proj,
                    "frac_dL_amp": mag_to_frac_distance(dm_proj),
                }

    summary: Dict[str, Any] = {
        "inputs": {
            "secrest_json": args.secrest_json,
            "w1_bins_json": args.w1_bins_json,
            "sn_scan_json": args.sn_scan_json if (args.sn_scan_json and os.path.exists(args.sn_scan_json)) else None,
            "edge_bins": int(args.edge_bins),
        },
        "secrest": {"D_obs": D_obs, "l_deg": l_qso, "b_deg": b_qso},
        "counts_response": {
            "N_total_sample": n_total,
            "edge_bins_used": edge_labels,
            "n_edge_per_mag": n_edge_per_mag,
            "alpha_edge_dlnN_dm": alpha_edge,
        },
        "required_magshift": {
            "delta_m_amp_mag": delta_m_amp,
            "delta_m_peak_to_trough_mag": 2.0 * delta_m_amp,
            "frac_dL_amp": frac_dL_amp,
            "frac_dL_peak_to_trough": 2.0 * frac_dL_amp,
        },
        "alpha_sensitivity": sens,
        "sn_axis": sn_axis,
        "sn_vs_secrest": sn_vs_secrest,
        "sn_projection_scaling": sn_projection_scaling,
        "notes": [
            "This is a first-order magnitude-cut response calculation: deltaN/N ~= (d ln N / d m_max) * delta m.",
            "alpha_edge is estimated from a quick-mode 200k-source subset (quasar-dipole-fun Stage-5).",
            "A full test would model dust/depth/ecliptic templates and selection explicitly; this is only an order-of-magnitude check.",
        ],
    }

    with open(os.path.join(outdir, "required_magshift_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Plot the approximate differential counts density vs magnitude (subset).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centers = [(b.lo + b.hi) / 2.0 for b in bins_sorted]
        dens = [b.density_per_mag for b in bins_sorted]
        plt.figure(figsize=(6.0, 3.8))
        plt.plot(centers, dens, "o-", ms=4)
        plt.yscale("log")
        plt.xlabel("W1 magnitude")
        plt.ylabel("dN/dm (per mag) [quick subset]")
        plt.title("CatWISE quick-mode: magnitude density estimate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "w1_density_log.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    print(f"[Secrest] D_obs={D_obs:.5f} at (l,b)=({l_qso:.2f},{b_qso:.2f}) deg")
    print(f"[Counts] alpha_edge=dlnN/dm ~= {alpha_edge:.3f} per mag  (edge_bins={int(args.edge_bins)}; N={n_total})")
    print(f"[Req] delta_m_amp ~= {delta_m_amp:.4f} mag  (peak-to-trough {2.0*delta_m_amp:.4f} mag)")
    print(f"[Req] frac dL amp ~= {frac_dL_amp*100.0:.3f}%  (peak-to-trough {2.0*frac_dL_amp*100.0:.3f}%)")
    if sn_axis and sn_vs_secrest:
        print(f"[SN axis] best-axis (l,b)=({sn_axis['l_deg']:.2f},{sn_axis['b_deg']:.2f}) deg")
        print(f"[SN axis] axis_angle(SN,Secrest)={sn_vs_secrest['axis_angle_deg']:.2f} deg  cos={sn_vs_secrest['cos_axis_angle']:.3f}")
        if sn_projection_scaling:
            print(f"[SN axis] projection scale to match amplitude ~{sn_projection_scaling['scale_factor']:.2f}x")

    print(f"Wrote: {os.path.join(outdir, 'required_magshift_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
