#!/usr/bin/env python3
"""Build a unified fit scaffold for object-level shared-redshift modeling.

This stage does not infer a physical constant. It packages quasar/radio series
into a consistent table with explicit mapping priors, so downstream fitting can
replace fixed z_eff assumptions with a constrained mapping model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _sigma_from_p16_p84(p16: float, p84: float, value: float, floor_frac: float) -> tuple[float, float, float]:
    sigma_stat = 0.5 * max(0.0, float(p84) - float(p16))
    sigma_floor = abs(float(value)) * float(floor_frac)
    sigma_total = float(math.sqrt(sigma_stat * sigma_stat + sigma_floor * sigma_floor))
    return sigma_total, sigma_stat, sigma_floor


def _sigma_from_cov_b(
    b_vec: list[float], cov_b: list[list[float]], value: float, floor_frac: float
) -> tuple[float, float, float]:
    b = np.asarray(b_vec, dtype=float).reshape(3)
    cov = np.asarray(cov_b, dtype=float)
    if cov.shape != (3, 3):
        raise ValueError(f"cov_b must be 3x3; got {cov.shape}")
    n = float(np.linalg.norm(b))
    if not np.isfinite(n) or n <= 0.0:
        sigma_stat = float(np.sqrt(max(0.0, float(np.trace(cov))))) / math.sqrt(3.0)
    else:
        g = b / n
        sigma_stat = float(math.sqrt(max(0.0, float(g @ cov @ g))))
    sigma_floor = abs(float(value)) * float(floor_frac)
    sigma_total = float(math.sqrt(sigma_stat * sigma_stat + sigma_floor * sigma_floor))
    return sigma_total, sigma_stat, sigma_floor


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
    ap.add_argument(
        "--out-fit-input",
        default="configs/shared_redshift_object_level_fit_input.json",
        help="Path to write fit-input JSON consumed by the object-level fitter.",
    )
    ap.add_argument(
        "--schema-template",
        default="configs/shared_redshift_object_level_schema_template.json",
        help="Schema metadata template to copy into run outputs (if present).",
    )
    ap.add_argument(
        "--calibration-json",
        default=None,
        help="Optional calibration JSON to override prior means/sigmas/bounds.",
    )

    ap.add_argument("--q-w1-ref", type=float, default=16.4)
    ap.add_argument("--q-sigma-floor-frac", type=float, default=0.15)
    ap.add_argument("--radio-sigma-floor-frac", type=float, default=0.15)

    ap.add_argument("--q-z-ref-prior-mu", type=float, default=1.2)
    ap.add_argument("--q-z-ref-prior-sigma", type=float, default=0.35)
    ap.add_argument("--q-z-ref-min", type=float, default=0.1)
    ap.add_argument("--q-z-ref-max", type=float, default=3.5)
    ap.add_argument("--q-slope-prior-mu", type=float, default=0.85)
    ap.add_argument("--q-slope-prior-sigma", type=float, default=0.40)
    ap.add_argument("--q-slope-min", type=float, default=0.0)
    ap.add_argument("--q-slope-max", type=float, default=3.0)

    ap.add_argument("--nvss-z-ref-prior-mu", type=float, default=0.9)
    ap.add_argument("--nvss-z-ref-prior-sigma", type=float, default=0.30)
    ap.add_argument("--nvss-cut-ref-mjy", type=float, default=20.0)
    ap.add_argument("--nvss-eta-prior-mu", type=float, default=0.25)
    ap.add_argument("--nvss-eta-prior-sigma", type=float, default=0.15)

    ap.add_argument("--racs-z-ref-prior-mu", type=float, default=0.8)
    ap.add_argument("--racs-z-ref-prior-sigma", type=float, default=0.30)
    ap.add_argument("--racs-cut-ref-mjy", type=float, default=20.0)
    ap.add_argument("--racs-eta-prior-mu", type=float, default=0.25)
    ap.add_argument("--racs-eta-prior-sigma", type=float, default=0.15)

    ap.add_argument("--lotss-z-ref-prior-mu", type=float, default=1.2)
    ap.add_argument("--lotss-z-ref-prior-sigma", type=float, default=0.35)
    ap.add_argument("--lotss-cut-ref-mjy", type=float, default=5.0)
    ap.add_argument("--lotss-eta-prior-mu", type=float, default=0.25)
    ap.add_argument("--lotss-eta-prior-sigma", type=float, default=0.15)

    ap.add_argument("--radio-z-ref-min", type=float, default=0.05)
    ap.add_argument("--radio-z-ref-max", type=float, default=4.0)
    ap.add_argument("--radio-eta-min", type=float, default=0.0)
    ap.add_argument("--radio-eta-max", type=float, default=1.5)

    ap.add_argument("--gamma-prior-mu", type=float, default=0.0)
    ap.add_argument("--gamma-prior-sigma", type=float, default=2.5)
    ap.add_argument("--gamma-min", type=float, default=-5.0)
    ap.add_argument("--gamma-max", type=float, default=5.0)
    return ap.parse_args()


def _build_quasar_points(
    quasar_json: Path,
    q_sigma_floor_frac: float,
) -> list[dict[str, Any]]:
    obj = json.loads(quasar_json.read_text())
    rows = obj.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{quasar_json}: expected non-empty `rows`")

    out: list[dict[str, Any]] = []
    for r in rows:
        mode = str(r.get("w1_mode", "")).strip().lower()
        if mode != "cumulative":
            continue
        w1 = float(r["w1_cut"])
        dip = r.get("dipole", {})
        d = float(dip["D_hat"])
        d16 = float(dip["D_p16"])
        d84 = float(dip["D_p84"])
        sigma, sigma_stat, sigma_floor = _sigma_from_p16_p84(
            p16=d16,
            p84=d84,
            value=d,
            floor_frac=q_sigma_floor_frac,
        )
        out.append(
            {
                "probe_name": "quasar_catwise",
                "family": "quasar",
                "survey": "CatWISE",
                "selection_kind": "w1_max",
                "selection_value": w1,
                "label": f"W1<= {w1:.2f}",
                "value": d,
                "sigma": sigma,
                "sigma_stat": sigma_stat,
                "sigma_floor": sigma_floor,
            }
        )
    out.sort(key=lambda x: float(x["selection_value"]))
    if len(out) < 2:
        raise ValueError("quasar scaffold has <2 points")
    return out


def _build_radio_points(
    radio_json: Path,
    radio_sigma_floor_frac: float,
) -> list[dict[str, Any]]:
    obj = json.loads(radio_json.read_text())
    scans = obj.get("per_survey_flux_scans")
    if not isinstance(scans, dict) or not scans:
        raise ValueError(f"{radio_json}: expected non-empty `per_survey_flux_scans`")

    out: list[dict[str, Any]] = []
    for survey, rows in scans.items():
        if not isinstance(rows, list):
            continue
        for r in rows:
            cut = float(r["cut_mjy"])
            d = float(r["D"])
            sigma, sigma_stat, sigma_floor = _sigma_from_cov_b(
                b_vec=r["b_vec"],
                cov_b=r["cov_b"],
                value=d,
                floor_frac=radio_sigma_floor_frac,
            )
            out.append(
                {
                    "probe_name": f"radio_{survey}",
                    "family": "radio",
                    "survey": survey,
                    "selection_kind": "flux_min_mjy",
                    "selection_value": cut,
                    "label": f"{survey} S>={cut:.1f} mJy",
                    "value": d,
                    "sigma": sigma,
                    "sigma_stat": sigma_stat,
                    "sigma_floor": sigma_floor,
                }
            )
    out.sort(key=lambda x: (str(x["survey"]), float(x["selection_value"])))
    if len(out) < 2:
        raise ValueError("radio scaffold has <2 points")
    return out


def _apply_prior_override(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in ["mu", "sigma", "bounds"]:
        if key in src:
            dst[key] = src[key]


def _apply_calibration_overrides(
    fit_input: dict[str, Any],
    calibration_path: Path,
) -> dict[str, Any]:
    cal = json.loads(calibration_path.read_text())
    applied: dict[str, Any] = {"calibration_path": str(calibration_path.resolve())}

    q = cal.get("quasar")
    if isinstance(q, dict):
        if "w1_ref" in q:
            fit_input["mapping_model"]["quasar"]["w1_ref"] = float(q["w1_ref"])
            applied["quasar_w1_ref"] = float(q["w1_ref"])
        if isinstance(q.get("z_ref"), dict):
            _apply_prior_override(fit_input["priors"]["quasar"]["z_ref"], q["z_ref"])
            applied["quasar_z_ref"] = fit_input["priors"]["quasar"]["z_ref"]
        if isinstance(q.get("slope_per_mag"), dict):
            _apply_prior_override(
                fit_input["priors"]["quasar"]["slope_per_mag"], q["slope_per_mag"]
            )
            applied["quasar_slope_per_mag"] = fit_input["priors"]["quasar"]["slope_per_mag"]

    radio = cal.get("radio")
    if isinstance(radio, dict):
        for survey in ["NVSS", "RACS-low", "LoTSS-DR2"]:
            if survey not in radio or not isinstance(radio[survey], dict):
                continue
            rs = radio[survey]
            if isinstance(rs.get("z_ref"), dict):
                _apply_prior_override(fit_input["priors"]["radio"][survey]["z_ref"], rs["z_ref"])
            if isinstance(rs.get("eta"), dict):
                _apply_prior_override(fit_input["priors"]["radio"][survey]["eta"], rs["eta"])
            applied[f"radio_{survey}"] = fit_input["priors"]["radio"][survey]

    g = cal.get("gamma")
    if isinstance(g, dict):
        _apply_prior_override(fit_input["priors"]["gamma"], g)
        applied["gamma"] = fit_input["priors"]["gamma"]
    return applied


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / (
        args.run_tag if args.run_tag else f"shared_redshift_object_scaffold_{utc_tag()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    quasar_json = Path(args.quasar_json).resolve()
    radio_json = Path(args.radio_json).resolve()

    q_points = _build_quasar_points(
        quasar_json=quasar_json,
        q_sigma_floor_frac=float(args.q_sigma_floor_frac),
    )
    r_points = _build_radio_points(
        radio_json=radio_json,
        radio_sigma_floor_frac=float(args.radio_sigma_floor_frac),
    )
    points = q_points + r_points
    for i, p in enumerate(points):
        p["point_id"] = int(i)

    fit_input = {
        "notes": (
            "Scaffold for object-level shared-redshift mapping fit. "
            "Mapping priors are explicit and should be updated once direct n(z) calibration is available."
        ),
        "inputs": {
            "quasar_json": str(quasar_json),
            "radio_json": str(radio_json),
        },
        "points": points,
        "mapping_model": {
            "quasar": {
                "w1_ref": float(args.q_w1_ref),
            },
            "radio": {
                "surveys": {
                    "NVSS": {"cut_ref_mjy": float(args.nvss_cut_ref_mjy)},
                    "RACS-low": {"cut_ref_mjy": float(args.racs_cut_ref_mjy)},
                    "LoTSS-DR2": {"cut_ref_mjy": float(args.lotss_cut_ref_mjy)},
                }
            },
        },
        "priors": {
            "gamma": {
                "mu": float(args.gamma_prior_mu),
                "sigma": float(args.gamma_prior_sigma),
                "bounds": [float(args.gamma_min), float(args.gamma_max)],
            },
            "quasar": {
                "z_ref": {
                    "mu": float(args.q_z_ref_prior_mu),
                    "sigma": float(args.q_z_ref_prior_sigma),
                    "bounds": [float(args.q_z_ref_min), float(args.q_z_ref_max)],
                },
                "slope_per_mag": {
                    "mu": float(args.q_slope_prior_mu),
                    "sigma": float(args.q_slope_prior_sigma),
                    "bounds": [float(args.q_slope_min), float(args.q_slope_max)],
                },
            },
            "radio": {
                "NVSS": {
                    "z_ref": {
                        "mu": float(args.nvss_z_ref_prior_mu),
                        "sigma": float(args.nvss_z_ref_prior_sigma),
                        "bounds": [float(args.radio_z_ref_min), float(args.radio_z_ref_max)],
                    },
                    "eta": {
                        "mu": float(args.nvss_eta_prior_mu),
                        "sigma": float(args.nvss_eta_prior_sigma),
                        "bounds": [float(args.radio_eta_min), float(args.radio_eta_max)],
                    },
                },
                "RACS-low": {
                    "z_ref": {
                        "mu": float(args.racs_z_ref_prior_mu),
                        "sigma": float(args.racs_z_ref_prior_sigma),
                        "bounds": [float(args.radio_z_ref_min), float(args.radio_z_ref_max)],
                    },
                    "eta": {
                        "mu": float(args.racs_eta_prior_mu),
                        "sigma": float(args.racs_eta_prior_sigma),
                        "bounds": [float(args.radio_eta_min), float(args.radio_eta_max)],
                    },
                },
                "LoTSS-DR2": {
                    "z_ref": {
                        "mu": float(args.lotss_z_ref_prior_mu),
                        "sigma": float(args.lotss_z_ref_prior_sigma),
                        "bounds": [float(args.radio_z_ref_min), float(args.radio_z_ref_max)],
                    },
                    "eta": {
                        "mu": float(args.lotss_eta_prior_mu),
                        "sigma": float(args.lotss_eta_prior_sigma),
                        "bounds": [float(args.radio_eta_min), float(args.radio_eta_max)],
                    },
                },
            },
        },
    }

    calibration_applied: dict[str, Any] | None = None
    if args.calibration_json is not None:
        cal_path = Path(args.calibration_json).resolve()
        if not cal_path.exists():
            raise FileNotFoundError(f"missing calibration json: {cal_path}")
        calibration_applied = _apply_calibration_overrides(fit_input, cal_path)
        fit_input["calibration_applied"] = calibration_applied

    out_fit_input = Path(args.out_fit_input)
    out_fit_input.parent.mkdir(parents=True, exist_ok=True)
    out_fit_input.write_text(json.dumps(fit_input, indent=2) + "\n")

    (out_dir / "fit_input.json").write_text(json.dumps(fit_input, indent=2) + "\n")
    (out_dir / "points.json").write_text(json.dumps(points, indent=2) + "\n")

    with (out_dir / "points.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "point_id",
                "probe_name",
                "family",
                "survey",
                "selection_kind",
                "selection_value",
                "label",
                "value",
                "sigma",
                "sigma_stat",
                "sigma_floor",
            ]
        )
        for p in points:
            w.writerow(
                [
                    p["point_id"],
                    p["probe_name"],
                    p["family"],
                    p["survey"],
                    p["selection_kind"],
                    p["selection_value"],
                    p["label"],
                    p["value"],
                    p["sigma"],
                    p["sigma_stat"],
                    p["sigma_floor"],
                ]
            )

    schema_path = Path(args.schema_template)
    if schema_path.exists():
        (out_dir / "schema_template.json").write_text(schema_path.read_text())

    if calibration_applied is not None:
        (out_dir / "calibration_applied.json").write_text(
            json.dumps(calibration_applied, indent=2) + "\n"
        )

    counts: dict[str, int] = {}
    for p in points:
        k = str(p["probe_name"])
        counts[k] = counts.get(k, 0) + 1

    md = []
    md.append("# Shared Redshift Object Scaffold")
    md.append("")
    md.append(f"- Created UTC: `{datetime.now(timezone.utc).isoformat()}`")
    md.append(f"- Quasar input: `{quasar_json}`")
    md.append(f"- Radio input: `{radio_json}`")
    md.append(f"- Total points: `{len(points)}`")
    if calibration_applied is not None:
        md.append(f"- Calibration overrides: `{calibration_applied.get('calibration_path')}`")
    for k, v in sorted(counts.items()):
        md.append(f"- {k}: `{v}` points")
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append(f"- Fit input: `{(out_dir / 'fit_input.json').resolve()}`")
    md.append(f"- Points CSV: `{(out_dir / 'points.csv').resolve()}`")
    md.append(f"- Mirrored fit input config: `{out_fit_input.resolve()}`")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "- This scaffold is a bridge layer: it exposes priors and mapping assumptions explicitly, "
        "so the next fit step can measure identifiability and prior-vs-data tension."
    )
    (out_dir / "master_report.md").write_text("\n".join(md) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "out_fit_input": str(out_fit_input.resolve()),
                "n_points": len(points),
                "calibration_applied": calibration_applied is not None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
