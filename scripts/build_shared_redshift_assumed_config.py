#!/usr/bin/env python3
"""Build a shared-redshift-multiplier config from in-repo quasar+radio outputs.

This is an assumption-driven bridge: it maps observed anomaly series to effective
redshift points using explicit, user-editable parameterizations.
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

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


@dataclass(frozen=True)
class QuasarAssumption:
    w1_ref: float
    z_ref: float
    z_slope_per_mag: float
    sigma_floor_frac: float


@dataclass(frozen=True)
class RadioSurveyAssumption:
    z_ref: float
    cut_ref_mjy: float
    eta: float
    sigma_floor_frac: float


def _sigma_from_p16_p84(p16: float, p84: float, d: float, sigma_floor_frac: float) -> float:
    sig_stat = 0.5 * max(0.0, float(p84) - float(p16))
    sig_floor = abs(float(d)) * float(sigma_floor_frac)
    sig = float(math.sqrt(sig_stat * sig_stat + sig_floor * sig_floor))
    return max(sig, 1e-8)


def _sigma_from_bvec_cov(b_vec: list[float], cov_b: list[list[float]], d: float, sigma_floor_frac: float) -> float:
    b = np.asarray(b_vec, dtype=float).reshape(3)
    cov = np.asarray(cov_b, dtype=float)
    if cov.shape != (3, 3):
        raise ValueError(f"expected 3x3 cov_b, got {cov.shape}")
    n = float(np.linalg.norm(b))
    if not np.isfinite(n) or n <= 0.0:
        sig_stat = float(np.sqrt(max(0.0, float(np.trace(cov))))) / math.sqrt(3.0)
    else:
        g = b / n
        sig_stat = float(math.sqrt(max(0.0, float(g @ cov @ g))))
    sig_floor = abs(float(d)) * float(sigma_floor_frac)
    sig = float(math.sqrt(sig_stat * sig_stat + sig_floor * sig_floor))
    return max(sig, 1e-8)


def build_quasar_points(quasar_json: Path, a: QuasarAssumption) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    obj = json.loads(quasar_json.read_text())
    rows = obj.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{quasar_json}: expected non-empty `rows` list")

    points: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for r in rows:
        if str(r.get("w1_mode", "")).strip().lower() != "cumulative":
            continue
        w1 = float(r["w1_cut"])
        dip = r.get("dipole", {})
        d = float(dip["D_hat"])
        p16 = float(dip["D_p16"])
        p84 = float(dip["D_p84"])
        sigma = _sigma_from_p16_p84(p16, p84, d, a.sigma_floor_frac)
        z_eff = float(a.z_ref + a.z_slope_per_mag * (w1 - a.w1_ref))
        z_eff = max(0.01, z_eff)
        label = f"W1<= {w1:.2f}"
        points.append(
            {
                "label": label,
                "z": z_eff,
                "value": d,
                "sigma": sigma,
            }
        )
        debug_rows.append(
            {
                "label": label,
                "w1_cut": w1,
                "z_eff_assumed": z_eff,
                "D_hat": d,
                "D_p16": p16,
                "D_p84": p84,
                "sigma_used": sigma,
            }
        )
    points.sort(key=lambda x: float(x["z"]))
    debug_rows.sort(key=lambda x: float(x["z_eff_assumed"]))
    if len(points) < 2:
        raise ValueError("quasar points after filtering are <2")
    return points, debug_rows


def _radio_assumption_for(survey: str, assumptions: dict[str, RadioSurveyAssumption]) -> RadioSurveyAssumption:
    if survey not in assumptions:
        raise KeyError(f"missing radio assumptions for survey={survey!r}")
    return assumptions[survey]


def build_radio_points(
    radio_json: Path, assumptions: dict[str, RadioSurveyAssumption]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    obj = json.loads(radio_json.read_text())
    scans = obj.get("per_survey_flux_scans")
    if not isinstance(scans, dict) or not scans:
        raise ValueError(f"{radio_json}: expected non-empty `per_survey_flux_scans` dict")

    probes: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for survey, rows in scans.items():
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        a = _radio_assumption_for(survey, assumptions)
        pts: list[dict[str, Any]] = []
        for r in rows:
            cut = float(r["cut_mjy"])
            d = float(r["D"])
            sigma = _sigma_from_bvec_cov(
                b_vec=r["b_vec"],
                cov_b=r["cov_b"],
                d=d,
                sigma_floor_frac=a.sigma_floor_frac,
            )
            z_eff = float(a.z_ref * ((a.cut_ref_mjy / cut) ** a.eta))
            z_eff = max(0.01, z_eff)
            label = f"{survey} S>={cut:.1f} mJy"
            pts.append({"label": label, "z": z_eff, "value": d, "sigma": sigma})
            debug_rows.append(
                {
                    "probe": f"radio_{survey}",
                    "label": label,
                    "survey": survey,
                    "cut_mjy": cut,
                    "z_eff_assumed": z_eff,
                    "D_hat": d,
                    "sigma_used": sigma,
                    "assumed_z_ref": a.z_ref,
                    "assumed_cut_ref_mjy": a.cut_ref_mjy,
                    "assumed_eta": a.eta,
                }
            )
        pts.sort(key=lambda x: float(x["z"]))
        if len(pts) >= 2:
            probes.append({"name": f"radio_{survey}", "points": pts})
    if not probes:
        raise ValueError("no radio probes built from per_survey_flux_scans")
    return probes, debug_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quasar-json",
        default="outputs/glm_baseline_abs_elat/rvmp_fig5_poisson_glm.json",
        help="Quasar GLM scan JSON with `rows` and dipole summaries.",
    )
    ap.add_argument(
        "--radio-json",
        default="outputs/radio_combined_same_logic_audit_20260208_060931UTC/radio_combined_same_logic_audit.json",
        help="Radio combined audit JSON with `per_survey_flux_scans`.",
    )
    ap.add_argument("--out-root", default="outputs", help="Output root.")
    ap.add_argument("--run-tag", default=None, help="Optional fixed run tag.")
    ap.add_argument(
        "--out-config",
        default="configs/shared_redshift_multiplier_from_repo_assumed.json",
        help="Where to write the generated config JSON.",
    )
    ap.add_argument("--q-w1-ref", type=float, default=16.4, help="Quasar W1 reference cut.")
    ap.add_argument("--q-z-ref", type=float, default=1.2, help="Assumed z_eff at q-w1-ref.")
    ap.add_argument(
        "--q-z-slope-per-mag",
        type=float,
        default=0.85,
        help="Assumed dz_eff/dW1 for quasar cuts.",
    )
    ap.add_argument(
        "--q-sigma-floor-frac",
        type=float,
        default=0.15,
        help="Relative sigma floor added in quadrature for quasar points.",
    )
    ap.add_argument("--nvss-z-ref", type=float, default=0.9)
    ap.add_argument("--nvss-cut-ref-mjy", type=float, default=20.0)
    ap.add_argument("--nvss-eta", type=float, default=0.25)
    ap.add_argument("--racs-z-ref", type=float, default=0.8)
    ap.add_argument("--racs-cut-ref-mjy", type=float, default=20.0)
    ap.add_argument("--racs-eta", type=float, default=0.25)
    ap.add_argument("--lotss-z-ref", type=float, default=1.2)
    ap.add_argument("--lotss-cut-ref-mjy", type=float, default=5.0)
    ap.add_argument("--lotss-eta", type=float, default=0.25)
    ap.add_argument(
        "--radio-sigma-floor-frac",
        type=float,
        default=0.15,
        help="Relative sigma floor added in quadrature for radio points.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / (
        args.run_tag if args.run_tag else f"shared_redshift_input_build_{utc_tag()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    q_assume = QuasarAssumption(
        w1_ref=float(args.q_w1_ref),
        z_ref=float(args.q_z_ref),
        z_slope_per_mag=float(args.q_z_slope_per_mag),
        sigma_floor_frac=float(args.q_sigma_floor_frac),
    )
    radio_assume = {
        "NVSS": RadioSurveyAssumption(
            z_ref=float(args.nvss_z_ref),
            cut_ref_mjy=float(args.nvss_cut_ref_mjy),
            eta=float(args.nvss_eta),
            sigma_floor_frac=float(args.radio_sigma_floor_frac),
        ),
        "RACS-low": RadioSurveyAssumption(
            z_ref=float(args.racs_z_ref),
            cut_ref_mjy=float(args.racs_cut_ref_mjy),
            eta=float(args.racs_eta),
            sigma_floor_frac=float(args.radio_sigma_floor_frac),
        ),
        "LoTSS-DR2": RadioSurveyAssumption(
            z_ref=float(args.lotss_z_ref),
            cut_ref_mjy=float(args.lotss_cut_ref_mjy),
            eta=float(args.lotss_eta),
            sigma_floor_frac=float(args.radio_sigma_floor_frac),
        ),
    }

    quasar_json = Path(args.quasar_json).resolve()
    radio_json = Path(args.radio_json).resolve()
    q_points, q_debug = build_quasar_points(quasar_json, q_assume)
    radio_probes, r_debug = build_radio_points(radio_json, radio_assume)

    config = {
        "notes": (
            "Assumption-driven z_eff mapping built from repo outputs. "
            "Interpret only as a sensitivity test; not direct redshift measurements."
        ),
        "probes": [{"name": "quasar_catwise", "points": q_points}] + radio_probes,
    }
    out_config = Path(args.out_config)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(json.dumps(config, indent=2) + "\n")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"quasar_json": str(quasar_json), "radio_json": str(radio_json)},
        "assumptions": {
            "quasar": {
                "w1_ref": q_assume.w1_ref,
                "z_ref": q_assume.z_ref,
                "z_slope_per_mag": q_assume.z_slope_per_mag,
                "sigma_floor_frac": q_assume.sigma_floor_frac,
            },
            "radio": {
                k: {
                    "z_ref": v.z_ref,
                    "cut_ref_mjy": v.cut_ref_mjy,
                    "eta": v.eta,
                    "sigma_floor_frac": v.sigma_floor_frac,
                }
                for k, v in radio_assume.items()
            },
        },
        "n_points": {
            "quasar_catwise": len(q_points),
            **{p["name"]: len(p["points"]) for p in radio_probes},
        },
        "out_config": str(out_config.resolve()),
    }
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    csv_path = out_dir / "assumed_zeff_points.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "probe",
                "label",
                "z_eff_assumed",
                "value_D",
                "sigma_used",
                "extra_1",
                "extra_2",
                "extra_3",
            ]
        )
        for r in q_debug:
            w.writerow(
                [
                    "quasar_catwise",
                    r["label"],
                    r["z_eff_assumed"],
                    r["D_hat"],
                    r["sigma_used"],
                    r["w1_cut"],
                    r["D_p16"],
                    r["D_p84"],
                ]
            )
        for r in r_debug:
            w.writerow(
                [
                    r["probe"],
                    r["label"],
                    r["z_eff_assumed"],
                    r["D_hat"],
                    r["sigma_used"],
                    r["cut_mjy"],
                    r["assumed_z_ref"],
                    r["assumed_eta"],
                ]
            )

    md = []
    md.append("# Shared Redshift Input Build")
    md.append("")
    md.append(f"- Config: `{out_config.resolve()}`")
    md.append(f"- Points CSV: `{csv_path.resolve()}`")
    md.append(f"- Quasar points: `{len(q_points)}`")
    for p in radio_probes:
        md.append(f"- {p['name']} points: `{len(p['points'])}`")
    md.append("")
    md.append("## Assumptions")
    md.append("")
    md.append(
        f"- Quasar mapping: `z_eff = {q_assume.z_ref:.4f} + {q_assume.z_slope_per_mag:.4f}*(W1_cut - {q_assume.w1_ref:.4f})`"
    )
    md.append(
        f"- Quasar sigma floor: `sigma_floor_frac={q_assume.sigma_floor_frac:.4f}` (added in quadrature)."
    )
    for name, a in radio_assume.items():
        md.append(
            f"- {name} mapping: `z_eff = {a.z_ref:.4f}*(S_ref/S)^eta`, with `S_ref={a.cut_ref_mjy:.4f} mJy`, `eta={a.eta:.4f}`, `sigma_floor_frac={a.sigma_floor_frac:.4f}`."
        )
    (out_dir / "master_report.md").write_text("\n".join(md) + "\n")

    print(
        json.dumps(
            {"status": "ok", "out_dir": str(out_dir), "out_config": str(out_config.resolve())},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

