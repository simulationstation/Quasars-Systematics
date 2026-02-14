#!/usr/bin/env python3
"""
Reviewer-seed reproduction: write a timestamped report.md + summary.json using
vendored small artifacts under artifacts/ (no large external downloads).

This intentionally does NOT rerun heavy pipelines; it is a headline verifier.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


def utc_timestamp_compact() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y%m%d_%H%M%SUTC")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def unit_vec_from_lb_deg(l_deg: float, b_deg: float) -> tuple[float, float, float]:
    lr = math.radians(l_deg)
    br = math.radians(b_deg)
    return (
        math.cos(br) * math.cos(lr),
        math.cos(br) * math.sin(lr),
        math.sin(br),
    )


def angle_deg_sign_invariant_lb(
    l1_deg: float, b1_deg: float, l2_deg: float, b2_deg: float
) -> float:
    v1 = unit_vec_from_lb_deg(l1_deg, b1_deg)
    v2 = unit_vec_from_lb_deg(l2_deg, b2_deg)
    dot = abs(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


@dataclasses.dataclass(frozen=True)
class Headlines:
    # GLM cumulative scan
    D_hat_median: float
    D_hat_min: float
    D_hat_max: float
    axis_sep_cmb_deg_w1_15p5: float
    axis_sep_cmb_deg_w1_16p6: float
    # lognormal mocks
    logn_D_hat: float
    logn_sigma_D: float
    logn_snr: float
    # correlated-cut drift MC
    drift_end_to_end_p: float
    drift_max_pair_p: float
    # unWISE epoch slicing
    epoch_D_glm_min: float
    epoch_D_glm_max: float
    epoch_D_vecsum_min: float
    epoch_D_vecsum_max: float
    epoch_N_min: int
    epoch_N_max: int


def compute_headlines(inputs_dir: Path) -> tuple[Headlines, dict[str, Any]]:
    # Planck CMB dipole direction in Galactic coordinates.
    cmb_l_deg = 264.021
    cmb_b_deg = 48.253

    scan = read_json(inputs_dir / "rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json")
    scan_rows = scan["rows"]
    d_hats = [r["dipole"]["D_hat"] for r in scan_rows]
    d_hats_sorted = sorted(d_hats)
    d_med = d_hats_sorted[len(d_hats_sorted) // 2]

    def row_at_w1(w1: float) -> dict[str, Any]:
        for r in scan_rows:
            if abs(float(r["w1_cut"]) - w1) < 1e-9:
                return r
        raise KeyError(f"Missing scan row for w1_cut={w1}")

    r_15p5 = row_at_w1(15.5)
    r_16p6 = row_at_w1(16.6)
    sep_15p5 = angle_deg_sign_invariant_lb(
        r_15p5["dipole"]["l_hat_deg"],
        r_15p5["dipole"]["b_hat_deg"],
        cmb_l_deg,
        cmb_b_deg,
    )
    sep_16p6 = angle_deg_sign_invariant_lb(
        r_16p6["dipole"]["l_hat_deg"],
        r_16p6["dipole"]["b_hat_deg"],
        cmb_l_deg,
        cmb_b_deg,
    )

    logn = read_json(inputs_dir / "lognormal_mocks_cov_w1max16p6_n500.json")
    logn_D_hat = float(logn["fit"]["dipole_hat"]["D"])
    D_p16 = float(logn["mocks"]["summary"]["D_p16"])
    D_p84 = float(logn["mocks"]["summary"]["D_p84"])
    logn_sigma = 0.5 * (D_p84 - D_p16)
    logn_snr = logn_D_hat / logn_sigma if logn_sigma > 0 else float("nan")

    drift = read_json(inputs_dir / "drift_mc_null_summary.json")
    drift_p_end = float(drift["pvals_vs_observed"]["end_to_end_p"])
    drift_p_maxpair = float(drift["pvals_vs_observed"]["max_pair_p"])

    epoch = read_json(inputs_dir / "epoch_amplitude.json")
    epochs = [e for e in epoch["epochs"] if e.get("epoch") is not None and e["epoch"] <= 15]
    D_glm_vals = [float(e["D_glm"]) for e in epochs]
    D_vec_vals = [float(e["D_vecsum"]) for e in epochs]
    N_vals = [int(e["N"]) for e in epochs]

    headlines = Headlines(
        D_hat_median=float(d_med),
        D_hat_min=float(min(d_hats)),
        D_hat_max=float(max(d_hats)),
        axis_sep_cmb_deg_w1_15p5=float(sep_15p5),
        axis_sep_cmb_deg_w1_16p6=float(sep_16p6),
        logn_D_hat=logn_D_hat,
        logn_sigma_D=float(logn_sigma),
        logn_snr=float(logn_snr),
        drift_end_to_end_p=drift_p_end,
        drift_max_pair_p=drift_p_maxpair,
        epoch_D_glm_min=float(min(D_glm_vals)),
        epoch_D_glm_max=float(max(D_glm_vals)),
        epoch_D_vecsum_min=float(min(D_vec_vals)),
        epoch_D_vecsum_max=float(max(D_vec_vals)),
        epoch_N_min=int(min(N_vals)),
        epoch_N_max=int(max(N_vals)),
    )

    raw = {
        "cmb_axis_galactic_lb_deg": [cmb_l_deg, cmb_b_deg],
        "scan_n_cuts": len(scan_rows),
        "scan_w1_cuts": [r["w1_cut"] for r in scan_rows],
        "scan_source_meta": scan.get("meta", {}),
        "lognormal_source_meta": logn.get("meta", {}),
        "drift_source_meta": drift.get("meta", {}),
        "epoch_source_meta": epoch.get("meta", {}),
    }
    return headlines, raw


def fmt(x: float, ndp: int = 3) -> str:
    if not math.isfinite(x):
        return str(x)
    return f"{x:.{ndp}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory. Default: outputs/reviewer_seed_<timestamp>/",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inputs_dir = repo_root / "artifacts" / "quasar_reviewer_seed"
    if not inputs_dir.is_dir():
        raise SystemExit(f"Missing inputs dir: {inputs_dir}")

    out_dir = Path(args.out) if args.out else (repo_root / "outputs" / f"reviewer_seed_{utc_timestamp_compact()}")
    ensure_dir(out_dir)

    headlines, raw = compute_headlines(inputs_dir)

    summary = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs_dir": str(inputs_dir),
        "paper_tex": "mnras_letter/main.tex",
        "dois": {
            "quasar_dipole_reproducibility_archive": "10.5281/zenodo.18643926",
            "supplementary_assets_archive": "10.5281/zenodo.18489200"
        },
        "headlines": dataclasses.asdict(headlines),
        "raw_meta": raw,
    }
    write_json(out_dir / "summary.json", summary)

    report = f"""# Reviewer Seed (Headline Verification)

This report is generated from vendored artifacts under `artifacts/` and is intended as a quick check of the headline values quoted in:

- `mnras_letter/main.tex`

## Headline Numbers

### CatWISE cumulative W1max scan (Poisson GLM)

- Dipole amplitude (D_hat) across cuts: median {fmt(headlines.D_hat_median, 5)}; range [{fmt(headlines.D_hat_min, 5)}, {fmt(headlines.D_hat_max, 5)}]
- Sign-invariant axis separation to CMB:
  - W1max = 15.5: {fmt(headlines.axis_sep_cmb_deg_w1_15p5, 2)} deg
  - W1max = 16.6: {fmt(headlines.axis_sep_cmb_deg_w1_16p6, 2)} deg

### Clustered lognormal mocks (W1max = 16.6; n=500)

- D_hat ≈ {fmt(headlines.logn_D_hat, 5)}
- sigma_D ≈ {fmt(headlines.logn_sigma_D, 5)} (from (p84-p16)/2)
- S/N ≈ {fmt(headlines.logn_snr, 2)}

### Correlated-cut drift Monte Carlo (Poisson-only null)

- p(end-to-end drift span) = {fmt(headlines.drift_end_to_end_p, 5)}
- p(max-pair drift span) = {fmt(headlines.drift_max_pair_p, 5)}

### unWISE time-domain (epoch slicing; epochs 0-15)

- D_glm range: [{fmt(headlines.epoch_D_glm_min, 3)}, {fmt(headlines.epoch_D_glm_max, 3)}]
- D_vecsum range: [{fmt(headlines.epoch_D_vecsum_min, 3)}, {fmt(headlines.epoch_D_vecsum_max, 3)}]
- N per epoch range: [{headlines.epoch_N_min}, {headlines.epoch_N_max}]

## DOIs

- Quasar-dipole reproducibility archive: 10.5281/zenodo.18643926
- Supplementary assets: 10.5281/zenodo.18489200
"""
    (out_dir / "report.md").write_text(report)

    print(f"[ok] wrote {out_dir}/report.md and {out_dir}/summary.json")


if __name__ == "__main__":
    main()
