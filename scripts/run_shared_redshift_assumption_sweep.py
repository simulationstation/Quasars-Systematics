#!/usr/bin/env python3
"""Robustness sweep for cross-probe shared redshift multiplier claims.

This script varies assumption-level z_eff mappings and tests whether a shared
redshift-shape model is robustly supported across quasar + radio probes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_shared_redshift_assumed_config import (  # noqa: E402
    QuasarAssumption,
    RadioSurveyAssumption,
    build_quasar_points,
    build_radio_points,
)
from scripts.run_shared_redshift_multiplier_test import (  # noqa: E402
    ProbeData,
    _fit_independent_gamma,
    _fit_shared_gamma,
    _fit_strict_constant,
    _profile_shared_gamma,
    flatten_probes,
)


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _to_probe_data(probes_cfg: list[dict[str, Any]]) -> list[ProbeData]:
    out: list[ProbeData] = []
    for p in probes_cfg:
        name = str(p["name"])
        pts = p["points"]
        z = np.asarray([float(r["z"]) for r in pts], dtype=float)
        y = np.asarray([float(r["value"]) for r in pts], dtype=float)
        s = np.asarray([float(r["sigma"]) for r in pts], dtype=float)
        labels = [str(r["label"]) for r in pts]
        out.append(ProbeData(name=name, z=z, y=y, s=s, labels=labels))
    return out


def _quantiles(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    return {
        "p16": float(np.percentile(arr, 16.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p84": float(np.percentile(arr, 84.0)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quasar-json",
        default="outputs/glm_baseline_abs_elat/rvmp_fig5_poisson_glm.json",
    )
    ap.add_argument(
        "--radio-json",
        default="outputs/radio_combined_same_logic_audit_20260208_060931UTC/radio_combined_same_logic_audit.json",
    )
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--gamma-min", type=float, default=-5.0)
    ap.add_argument("--gamma-max", type=float, default=5.0)
    ap.add_argument("--gamma-grid-size", type=int, default=401)
    ap.add_argument("--q-z-ref-grid", default="1.0,1.2,1.4")
    ap.add_argument("--q-z-slope-grid", default="0.4,0.7,1.0,1.3")
    ap.add_argument("--radio-eta-grid", default="0.1,0.25,0.4")
    ap.add_argument("--radio-z-scale-grid", default="0.8,1.0,1.2")
    ap.add_argument("--q-w1-ref", type=float, default=16.4)
    ap.add_argument("--q-sigma-floor-frac", type=float, default=0.15)
    ap.add_argument("--radio-sigma-floor-frac", type=float, default=0.15)
    ap.add_argument("--nvss-z-ref-base", type=float, default=0.9)
    ap.add_argument("--racs-z-ref-base", type=float, default=0.8)
    ap.add_argument("--lotss-z-ref-base", type=float, default=1.2)
    ap.add_argument("--nvss-cut-ref-mjy", type=float, default=20.0)
    ap.add_argument("--racs-cut-ref-mjy", type=float, default=20.0)
    ap.add_argument("--lotss-cut-ref-mjy", type=float, default=5.0)
    return ap.parse_args()


def _parse_grid(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty grid: {s!r}")
    return vals


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / (
        args.run_tag if args.run_tag else f"shared_redshift_assumption_sweep_{utc_tag()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    q_z_ref_grid = _parse_grid(args.q_z_ref_grid)
    q_z_slope_grid = _parse_grid(args.q_z_slope_grid)
    radio_eta_grid = _parse_grid(args.radio_eta_grid)
    radio_z_scale_grid = _parse_grid(args.radio_z_scale_grid)

    rows: list[dict[str, Any]] = []
    quasar_json = Path(args.quasar_json).resolve()
    radio_json = Path(args.radio_json).resolve()

    combos = list(
        itertools.product(q_z_ref_grid, q_z_slope_grid, radio_eta_grid, radio_z_scale_grid)
    )

    for i, (q_z_ref, q_z_slope, radio_eta, radio_z_scale) in enumerate(combos):
        q_assume = QuasarAssumption(
            w1_ref=float(args.q_w1_ref),
            z_ref=float(q_z_ref),
            z_slope_per_mag=float(q_z_slope),
            sigma_floor_frac=float(args.q_sigma_floor_frac),
        )
        radio_assume = {
            "NVSS": RadioSurveyAssumption(
                z_ref=float(args.nvss_z_ref_base * radio_z_scale),
                cut_ref_mjy=float(args.nvss_cut_ref_mjy),
                eta=float(radio_eta),
                sigma_floor_frac=float(args.radio_sigma_floor_frac),
            ),
            "RACS-low": RadioSurveyAssumption(
                z_ref=float(args.racs_z_ref_base * radio_z_scale),
                cut_ref_mjy=float(args.racs_cut_ref_mjy),
                eta=float(radio_eta),
                sigma_floor_frac=float(args.radio_sigma_floor_frac),
            ),
            "LoTSS-DR2": RadioSurveyAssumption(
                z_ref=float(args.lotss_z_ref_base * radio_z_scale),
                cut_ref_mjy=float(args.lotss_cut_ref_mjy),
                eta=float(radio_eta),
                sigma_floor_frac=float(args.radio_sigma_floor_frac),
            ),
        }

        try:
            q_points, _ = build_quasar_points(quasar_json, q_assume)
            r_probes, _ = build_radio_points(radio_json, radio_assume)
            probes_cfg = [{"name": "quasar_catwise", "points": q_points}] + r_probes
            probes = _to_probe_data(probes_cfg)
            flat = flatten_probes(probes)
            fit_shared = _fit_shared_gamma(flat, gamma_min=args.gamma_min, gamma_max=args.gamma_max)
            fit_indep = _fit_independent_gamma(flat, gamma_min=args.gamma_min, gamma_max=args.gamma_max)
            fit_strict = _fit_strict_constant(flat, gamma_min=args.gamma_min, gamma_max=args.gamma_max)
            profile = _profile_shared_gamma(
                flat,
                gamma_min=args.gamma_min,
                gamma_max=args.gamma_max,
                n_grid=int(args.gamma_grid_size),
            )
            row = {
                "idx": i,
                "ok": True,
                "q_z_ref": float(q_z_ref),
                "q_z_slope_per_mag": float(q_z_slope),
                "radio_eta": float(radio_eta),
                "radio_z_scale": float(radio_z_scale),
                "gamma_shared": float(fit_shared["gamma"]),
                "delta_bic_independent_minus_shared": float(
                    fit_indep["bic"] - fit_shared["bic"]
                ),
                "delta_bic_strict_minus_shared": float(fit_strict["bic"] - fit_shared["bic"]),
                "shared_shape_strong_bic": bool((fit_indep["bic"] - fit_shared["bic"]) > 6.0),
                "strict_constant_supported_bic": bool(
                    (fit_strict["bic"] - fit_shared["bic"]) > 6.0
                ),
                "gamma_well_constrained": bool(profile["is_well_constrained_on_grid"]),
                "gamma_1sigma_interval": profile["interval_1sigma_delta_chi2_1"],
                "chi2_shared": float(fit_shared["chi2"]),
                "chi2_indep": float(fit_indep["chi2"]),
                "chi2_strict": float(fit_strict["chi2"]),
                "n_points": int(flat.y.size),
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "idx": i,
                "ok": False,
                "q_z_ref": float(q_z_ref),
                "q_z_slope_per_mag": float(q_z_slope),
                "radio_eta": float(radio_eta),
                "radio_z_scale": float(radio_z_scale),
                "error": str(exc),
            }
        rows.append(row)

    ok_rows = [r for r in rows if r.get("ok")]
    if not ok_rows:
        raise RuntimeError("no successful sweep rows")

    d_bic_shared = np.asarray(
        [float(r["delta_bic_independent_minus_shared"]) for r in ok_rows], dtype=float
    )
    d_bic_strict = np.asarray(
        [float(r["delta_bic_strict_minus_shared"]) for r in ok_rows], dtype=float
    )
    gammas = np.asarray([float(r["gamma_shared"]) for r in ok_rows], dtype=float)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"quasar_json": str(quasar_json), "radio_json": str(radio_json)},
        "grid": {
            "q_z_ref_grid": q_z_ref_grid,
            "q_z_slope_grid": q_z_slope_grid,
            "radio_eta_grid": radio_eta_grid,
            "radio_z_scale_grid": radio_z_scale_grid,
        },
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "n_fail": len(rows) - len(ok_rows),
        "n_shared_shape_strong_bic": int(
            sum(bool(r["shared_shape_strong_bic"]) for r in ok_rows)
        ),
        "n_strict_constant_supported_bic": int(
            sum(bool(r["strict_constant_supported_bic"]) for r in ok_rows)
        ),
        "n_gamma_well_constrained": int(sum(bool(r["gamma_well_constrained"]) for r in ok_rows)),
        "frac_shared_shape_strong_bic": float(
            sum(bool(r["shared_shape_strong_bic"]) for r in ok_rows) / len(ok_rows)
        ),
        "frac_strict_constant_supported_bic": float(
            sum(bool(r["strict_constant_supported_bic"]) for r in ok_rows) / len(ok_rows)
        ),
        "frac_gamma_well_constrained": float(
            sum(bool(r["gamma_well_constrained"]) for r in ok_rows) / len(ok_rows)
        ),
        "delta_bic_independent_minus_shared_stats": _quantiles(d_bic_shared),
        "delta_bic_strict_minus_shared_stats": _quantiles(d_bic_strict),
        "gamma_shared_stats": _quantiles(gammas),
    }

    best_by_shared = max(ok_rows, key=lambda r: float(r["delta_bic_independent_minus_shared"]))
    best_by_strict = max(ok_rows, key=lambda r: float(r["delta_bic_strict_minus_shared"]))
    summary["best_shared_shape_case"] = best_by_shared
    summary["best_strict_constant_case"] = best_by_strict

    (out_dir / "sweep_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Visual summary
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax0, ax1 = axes
    ax0.hist(d_bic_shared, bins=24, alpha=0.85, color="#1f77b4")
    ax0.axvline(6.0, color="k", ls="--", lw=1)
    ax0.set_xlabel("delta BIC (independent - shared)")
    ax0.set_ylabel("count")
    ax0.set_title("Shared-shape support")

    ax1.hist(d_bic_strict, bins=24, alpha=0.85, color="#ff7f0e")
    ax1.axvline(6.0, color="k", ls="--", lw=1)
    ax1.set_xlabel("delta BIC (strict - shared)")
    ax1.set_ylabel("count")
    ax1.set_title("Strict-constant support")
    fig.tight_layout()
    fig.savefig(out_dir / "bic_histograms.png", dpi=180)
    plt.close(fig)

    md: list[str] = []
    md.append("# Shared Redshift Assumption Sweep")
    md.append("")
    md.append(f"- Total combos: `{summary['n_total']}`")
    md.append(f"- Successful combos: `{summary['n_ok']}`")
    md.append(f"- Shared-shape strong support fraction (delta BIC > 6): `{summary['frac_shared_shape_strong_bic']:.3f}`")
    md.append(f"- Strict shared-constant support fraction (delta BIC > 6): `{summary['frac_strict_constant_supported_bic']:.3f}`")
    md.append(f"- Gamma well-constrained fraction: `{summary['frac_gamma_well_constrained']:.3f}`")
    md.append("")
    md.append("## Distribution summaries")
    md.append("")
    md.append(
        f"- delta BIC (independent-shape minus shared-shape), p16/p50/p84: "
        f"`{summary['delta_bic_independent_minus_shared_stats']['p16']:.3f} / "
        f"{summary['delta_bic_independent_minus_shared_stats']['p50']:.3f} / "
        f"{summary['delta_bic_independent_minus_shared_stats']['p84']:.3f}`"
    )
    md.append(
        f"- delta BIC (strict-constant minus shared-shape), p16/p50/p84: "
        f"`{summary['delta_bic_strict_minus_shared_stats']['p16']:.3f} / "
        f"{summary['delta_bic_strict_minus_shared_stats']['p50']:.3f} / "
        f"{summary['delta_bic_strict_minus_shared_stats']['p84']:.3f}`"
    )
    md.append(
        f"- shared gamma, p16/p50/p84: "
        f"`{summary['gamma_shared_stats']['p16']:.4f} / "
        f"{summary['gamma_shared_stats']['p50']:.4f} / "
        f"{summary['gamma_shared_stats']['p84']:.4f}`"
    )
    md.append("")
    md.append("## Best cases")
    md.append("")
    md.append(
        "- Best shared-shape support case: "
        f"`delta_bic_independent_minus_shared={best_by_shared['delta_bic_independent_minus_shared']:.3f}`, "
        f"`gamma_shared={best_by_shared['gamma_shared']:.4f}`, "
        f"`q_z_ref={best_by_shared['q_z_ref']:.3f}`, "
        f"`q_z_slope={best_by_shared['q_z_slope_per_mag']:.3f}`, "
        f"`radio_eta={best_by_shared['radio_eta']:.3f}`, "
        f"`radio_z_scale={best_by_shared['radio_z_scale']:.3f}`."
    )
    md.append(
        "- Best strict-constant support case: "
        f"`delta_bic_strict_minus_shared={best_by_strict['delta_bic_strict_minus_shared']:.3f}`, "
        f"`gamma_shared={best_by_strict['gamma_shared']:.4f}`, "
        f"`q_z_ref={best_by_strict['q_z_ref']:.3f}`, "
        f"`q_z_slope={best_by_strict['q_z_slope_per_mag']:.3f}`, "
        f"`radio_eta={best_by_strict['radio_eta']:.3f}`, "
        f"`radio_z_scale={best_by_strict['radio_z_scale']:.3f}`."
    )
    (out_dir / "master_report.md").write_text("\n".join(md) + "\n")

    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "n_ok": len(ok_rows)}, indent=2))


if __name__ == "__main__":
    main()

