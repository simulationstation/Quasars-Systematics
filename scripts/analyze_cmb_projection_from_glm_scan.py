#!/usr/bin/env python3
"""Analyze CMB-projected dipole components from RVMP-fig5-style Poisson GLM scan JSONs.

This is a post-processing diagnostic to help interpret direction drift:

Given per-cut dipole estimates (D_hat, l_hat_deg, b_hat_deg) and a reference axis (CMB dipole),
compute:
  - signed angle to the CMB direction
  - axis (sign-invariant) angle to the CMB axis
  - parallel component D_parallel = D * cos(theta_signed)
  - perpendicular component D_perp = D * sin(theta_signed)

Outputs:
  - cmb_projection_summary.json
  - cmb_projection_summary.csv
  - cmb_projection_plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def unitvec_lb(l_deg: float, b_deg: float) -> np.ndarray:
    l = np.deg2rad(float(l_deg))
    b = np.deg2rad(float(b_deg))
    v = np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("Invalid (l,b) for unit vector.")
    return v / n


def clamp(x: float, a: float, b: float) -> float:
    return float(max(a, min(b, x)))


@dataclass(frozen=True)
class RowOut:
    w1_hi: float
    w1_lo: float | None
    w1_mode: str
    D_hat: float
    l_hat_deg: float
    b_hat_deg: float
    cos_to_cmb: float
    theta_signed_deg: float
    theta_axis_deg: float
    D_parallel: float
    D_perp: float


def load_scan(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "meta" not in obj or "rows" not in obj:
        raise ValueError(f"{path}: expected keys {{meta, rows}}")
    if not isinstance(obj["rows"], list) or not obj["rows"]:
        raise ValueError(f"{path}: rows is empty")
    return obj


def analyze_one(path: Path) -> dict[str, Any]:
    obj = load_scan(path)
    meta = obj["meta"]
    rows = obj["rows"]

    if "inject_axis_lb" not in meta:
        raise ValueError(f"{path}: meta.inject_axis_lb missing")
    l_cmb, b_cmb = meta["inject_axis_lb"]
    u_cmb = unitvec_lb(float(l_cmb), float(b_cmb))

    out_rows: list[RowOut] = []
    for r in rows:
        dip = r.get("dipole", {})
        D_hat = float(dip["D_hat"])
        l_hat = float(dip["l_hat_deg"])
        b_hat = float(dip["b_hat_deg"])

        u = unitvec_lb(l_hat, b_hat)
        cos = float(np.dot(u, u_cmb))
        cos_c = clamp(cos, -1.0, 1.0)
        theta_signed = float(np.rad2deg(np.arccos(cos_c)))
        theta_axis = float(np.rad2deg(np.arccos(abs(cos_c))))
        D_parallel = float(D_hat * cos)
        D_perp = float(D_hat * math.sqrt(max(0.0, 1.0 - cos * cos)))

        w1_hi = float(r.get("w1_hi", r.get("w1_cut")))
        w1_lo = r.get("w1_lo", None)
        w1_lo_f = float(w1_lo) if w1_lo is not None else None
        w1_mode = str(r.get("w1_mode", meta.get("w1_mode", "")))

        out_rows.append(
            RowOut(
                w1_hi=w1_hi,
                w1_lo=w1_lo_f,
                w1_mode=w1_mode,
                D_hat=D_hat,
                l_hat_deg=l_hat,
                b_hat_deg=b_hat,
                cos_to_cmb=cos,
                theta_signed_deg=theta_signed,
                theta_axis_deg=theta_axis,
                D_parallel=D_parallel,
                D_perp=D_perp,
            )
        )

    out_rows = sorted(out_rows, key=lambda rr: (rr.w1_hi, -1e9 if rr.w1_lo is None else rr.w1_lo))
    return {
        "path": str(path),
        "meta": {
            "inject_axis_lb": [float(l_cmb), float(b_cmb)],
            "w1_grid": meta.get("w1_grid"),
            "w1_mode": meta.get("w1_mode"),
            "depth_mode": meta.get("depth_mode"),
            "eclip_template": meta.get("eclip_template"),
            "dust_template": meta.get("dust_template"),
            "nside": meta.get("nside"),
        },
        "rows": [rr.__dict__ for rr in out_rows],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows for CSV.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="One or more scan JSON files.")
    ap.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels (same length as --inputs). Defaults to basenames.",
    )
    ap.add_argument("--outdir", type=str, default="", help="Output directory (default: outputs/cmb_projection_<tag>)")
    args = ap.parse_args()

    inputs = [Path(p).expanduser().resolve() for p in args.inputs]
    labels = list(args.labels) if args.labels else [p.stem for p in inputs]
    if len(labels) != len(inputs):
        raise ValueError("--labels must be omitted or match length of --inputs.")

    outdir = Path(args.outdir) if str(args.outdir).strip() else Path(f"outputs/cmb_projection_{utc_tag()}")
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    bundles = []
    for p, lab in zip(inputs, labels, strict=True):
        b = analyze_one(p)
        b["label"] = str(lab)
        bundles.append(b)

    # Flatten rows for CSV.
    flat_rows: list[dict[str, Any]] = []
    for b in bundles:
        meta = b["meta"]
        for r in b["rows"]:
            flat_rows.append(
                {
                    "label": b["label"],
                    "w1_hi": r["w1_hi"],
                    "w1_lo": r["w1_lo"],
                    "w1_mode": r["w1_mode"],
                    "D_hat": r["D_hat"],
                    "theta_signed_deg": r["theta_signed_deg"],
                    "theta_axis_deg": r["theta_axis_deg"],
                    "D_parallel": r["D_parallel"],
                    "D_perp": r["D_perp"],
                    "depth_mode": meta.get("depth_mode"),
                    "eclip_template": meta.get("eclip_template"),
                }
            )

    (outdir / "cmb_projection_summary.json").write_text(json.dumps({"bundles": bundles}, indent=2) + "\n", encoding="utf-8")
    write_csv(outdir / "cmb_projection_summary.csv", flat_rows)

    # Plot.
    plt.figure(figsize=(8.0, 7.5))

    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)

    for b in bundles:
        lab = b["label"]
        rows = b["rows"]
        w1 = np.array([float(r["w1_hi"]) for r in rows], dtype=float)
        D = np.array([float(r["D_hat"]) for r in rows], dtype=float)
        Dpar = np.array([float(r["D_parallel"]) for r in rows], dtype=float)
        Dperp = np.array([float(r["D_perp"]) for r in rows], dtype=float)
        ax1.plot(w1, D, marker="o", linewidth=1.5, label=lab)
        ax2.plot(w1, Dpar, marker="o", linewidth=1.5, label=lab)
        ax3.plot(w1, Dperp, marker="o", linewidth=1.5, label=lab)

    ax1.set_ylabel("D (total)")
    ax2.set_ylabel("D‖CMB")
    ax3.set_ylabel("D⊥CMB")
    ax3.set_xlabel("W1_max (w1_hi)")

    ax1.grid(alpha=0.25)
    ax2.grid(alpha=0.25)
    ax3.grid(alpha=0.25)
    ax1.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig_path = outdir / "cmb_projection_plot.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print(f"Wrote {outdir/'cmb_projection_summary.json'}")
    print(f"Wrote {outdir/'cmb_projection_summary.csv'}")
    print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

