#!/usr/bin/env python3
"""Identifiability and power simulations for shared-redshift multiplier models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shared_redshift_multiplier_test import (
    FlatData,
    _design_independent_gamma,
    _design_shared_gamma,
    _design_strict_constant,
    _fit_independent_gamma,
    _fit_shared_gamma,
    _fit_strict_constant,
    _profile_shared_gamma,
    _safe_feature,
    _weighted_lstsq,
    flatten_probes,
    load_config,
)


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="configs/shared_redshift_multiplier_from_repo_assumed.json",
    )
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--gamma-min", type=float, default=-5.0)
    ap.add_argument("--gamma-max", type=float, default=5.0)
    ap.add_argument("--gamma-grid-size", type=int, default=161)
    ap.add_argument("--gamma-truth-grid", default="-0.0625,1.0,3.67")
    ap.add_argument("--sigma-scale-grid", default="1.0,0.5,0.25,0.1")
    ap.add_argument("--n-ident", type=int, default=150)
    ap.add_argument("--n-select", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _parse_grid(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty grid: {s!r}")
    return vals


def _quantiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        "p16": float(np.percentile(x, 16.0)),
        "p50": float(np.percentile(x, 50.0)),
        "p84": float(np.percentile(x, 84.0)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _make_flat(flat_template: FlatData, y: np.ndarray, sigma_scale: float) -> FlatData:
    return FlatData(
        probe_names=list(flat_template.probe_names),
        probe_idx=np.asarray(flat_template.probe_idx, dtype=int),
        z=np.asarray(flat_template.z, dtype=float),
        y=np.asarray(y, dtype=float),
        s=np.asarray(flat_template.s, dtype=float) * float(sigma_scale),
        labels=list(flat_template.labels),
    )


def _shared_fit_from_profile_grid(
    flat: FlatData,
    gamma_min: float,
    gamma_max: float,
    gamma_grid_size: int,
) -> dict[str, Any]:
    profile = _profile_shared_gamma(
        flat,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        n_grid=gamma_grid_size,
    )
    gamma_hat = float(profile["gamma_best_grid"])
    x = _design_shared_gamma(flat, gamma_hat)
    beta, chi2 = _weighted_lstsq(x, flat.y, flat.s)
    n = int(flat.y.size)
    p = int(len(flat.probe_names))
    k = int(2 * p + 1)
    loglike = float(-0.5 * (chi2 + np.sum(np.log(2.0 * np.pi * flat.s * flat.s))))
    aic = float(2 * k - 2 * loglike)
    bic = float(np.log(n) * k - 2 * loglike)
    return {
        "gamma": gamma_hat,
        "chi2": float(chi2),
        "aic": aic,
        "bic": bic,
        "profile": profile,
    }


def _fit_shared_at_fixed_gamma(flat: FlatData, gamma: float) -> dict[str, Any]:
    x = _design_shared_gamma(flat, gamma)
    beta, chi2 = _weighted_lstsq(x, flat.y, flat.s)
    y_hat = x @ beta
    p = len(flat.probe_names)
    return {
        "gamma": float(gamma),
        "intercepts": {flat.probe_names[i]: float(beta[i]) for i in range(p)},
        "slopes": {flat.probe_names[i]: float(beta[p + i]) for i in range(p)},
        "chi2": float(chi2),
        "y_hat": y_hat.tolist(),
    }


def _fit_strict_at_fixed_gamma(flat: FlatData, gamma: float) -> dict[str, Any]:
    x = _design_strict_constant(flat, gamma)
    beta, chi2 = _weighted_lstsq(x, flat.y, flat.s)
    p = len(flat.probe_names)
    y_hat = x @ beta
    return {
        "gamma": float(gamma),
        "intercepts": {flat.probe_names[i]: float(beta[i]) for i in range(p)},
        "slope_shared": float(beta[p]),
        "chi2": float(chi2),
        "y_hat": y_hat.tolist(),
    }


def _mu_from_independent(flat: FlatData, fit_indep: dict[str, Any]) -> np.ndarray:
    mu = np.zeros_like(flat.y, dtype=float)
    for i, name in enumerate(flat.probe_names):
        mask = flat.probe_idx == i
        a = float(fit_indep["intercepts"][name])
        b = float(fit_indep["slopes"][name])
        g = float(fit_indep["gammas"][name])
        mu[mask] = a + b * _safe_feature(flat.z[mask], g)
    return mu


def _run_gamma_identifiability(
    flat_template: FlatData,
    gamma_true: float,
    sigma_scale: float,
    n_rep: int,
    gamma_min: float,
    gamma_max: float,
    gamma_grid_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    fit_fixed = _fit_shared_at_fixed_gamma(flat_template, gamma_true)
    mu = np.asarray(fit_fixed["y_hat"], dtype=float)

    gamma_hat = []
    interval_width = []
    include_true = 0
    well_constrained = 0
    dbic_indep_minus_shared = []
    dbic_strict_minus_shared = []

    for _ in range(n_rep):
        y = mu + rng.normal(0.0, flat_template.s * sigma_scale)
        flat = _make_flat(flat_template, y=y, sigma_scale=sigma_scale)

        fit_shared = _shared_fit_from_profile_grid(
            flat,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_grid_size=gamma_grid_size,
        )
        fit_indep = _fit_independent_gamma(flat, gamma_min=gamma_min, gamma_max=gamma_max)
        fit_strict = _fit_strict_constant(flat, gamma_min=gamma_min, gamma_max=gamma_max)
        profile = fit_shared["profile"]

        gamma_hat.append(float(fit_shared["gamma"]))
        dbic_indep_minus_shared.append(float(fit_indep["bic"] - fit_shared["bic"]))
        dbic_strict_minus_shared.append(float(fit_strict["bic"] - fit_shared["bic"]))

        interval = profile.get("interval_1sigma_delta_chi2_1")
        if interval is None:
            interval_width.append(float("nan"))
        else:
            lo, hi = float(interval[0]), float(interval[1])
            interval_width.append(float(hi - lo))
            if lo <= gamma_true <= hi:
                include_true += 1
        if bool(profile.get("is_well_constrained_on_grid")):
            well_constrained += 1

    gamma_hat_arr = np.asarray(gamma_hat, dtype=float)
    width_arr = np.asarray(interval_width, dtype=float)
    abs_err = np.abs(gamma_hat_arr - gamma_true)
    return {
        "gamma_true": float(gamma_true),
        "sigma_scale": float(sigma_scale),
        "n_rep": int(n_rep),
        "gamma_hat": _quantiles(gamma_hat_arr),
        "abs_error": _quantiles(abs_err),
        "interval_width_1sigma": _quantiles(width_arr[np.isfinite(width_arr)]),
        "frac_interval_contains_true": float(include_true / n_rep),
        "frac_well_constrained": float(well_constrained / n_rep),
        "delta_bic_independent_minus_shared": _quantiles(
            np.asarray(dbic_indep_minus_shared, dtype=float)
        ),
        "delta_bic_strict_minus_shared": _quantiles(
            np.asarray(dbic_strict_minus_shared, dtype=float)
        ),
    }


def _run_model_selection(
    flat_template: FlatData,
    truth_name: str,
    mu_truth: np.ndarray,
    sigma_scale: float,
    n_rep: int,
    gamma_min: float,
    gamma_max: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    counts = {
        "shared_gamma_per_probe_slope": 0,
        "independent_gamma_per_probe_slope": 0,
        "strict_shared_constant": 0,
    }
    dbic_indep_minus_shared = []
    dbic_strict_minus_shared = []

    for _ in range(n_rep):
        y = mu_truth + rng.normal(0.0, flat_template.s * sigma_scale)
        flat = _make_flat(flat_template, y=y, sigma_scale=sigma_scale)

        fit_shared = _shared_fit_from_profile_grid(
            flat,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_grid_size=101,
        )
        fit_indep = _fit_independent_gamma(flat, gamma_min=gamma_min, gamma_max=gamma_max)
        fit_strict = _fit_strict_constant(flat, gamma_min=gamma_min, gamma_max=gamma_max)

        bic_map = {
            "shared_gamma_per_probe_slope": float(fit_shared["bic"]),
            "independent_gamma_per_probe_slope": float(fit_indep["bic"]),
            "strict_shared_constant": float(fit_strict["bic"]),
        }
        best = min(bic_map, key=bic_map.get)
        counts[best] += 1
        dbic_indep_minus_shared.append(float(fit_indep["bic"] - fit_shared["bic"]))
        dbic_strict_minus_shared.append(float(fit_strict["bic"] - fit_shared["bic"]))

    return {
        "truth_name": truth_name,
        "sigma_scale": float(sigma_scale),
        "n_rep": int(n_rep),
        "best_model_fractions": {
            name: float(counts[name] / n_rep) for name in counts
        },
        "delta_bic_independent_minus_shared": _quantiles(
            np.asarray(dbic_indep_minus_shared, dtype=float)
        ),
        "delta_bic_strict_minus_shared": _quantiles(
            np.asarray(dbic_strict_minus_shared, dtype=float)
        ),
    }


def _write_markdown(out_path: Path, summary: dict[str, Any]) -> None:
    lines = []
    lines.append("# Shared-Redshift Identifiability Simulation")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Config: `{summary['config']}`")
    lines.append(f"- Points: `{summary['n_points']}` across probes `{summary['probe_names']}`")
    lines.append("")
    lines.append("## Gamma identifiability")
    lines.append("")
    for row in summary["gamma_identifiability"]:
        lines.append(
            f"- gamma_true={row['gamma_true']:.4f}, sigma_scale={row['sigma_scale']:.3f}: "
            f"frac_well_constrained={row['frac_well_constrained']:.3f}, "
            f"frac_interval_contains_true={row['frac_interval_contains_true']:.3f}, "
            f"median_abs_error={row['abs_error']['p50']:.4f}, "
            f"median_interval_width={row['interval_width_1sigma']['p50']:.4f}"
        )
    lines.append("")
    lines.append("## Model selection")
    lines.append("")
    for row in summary["model_selection"]:
        bmf = row["best_model_fractions"]
        lines.append(
            f"- truth={row['truth_name']}, sigma_scale={row['sigma_scale']:.3f}: "
            f"best(shared/indep/strict)="
            f"{bmf['shared_gamma_per_probe_slope']:.3f}/"
            f"{bmf['independent_gamma_per_probe_slope']:.3f}/"
            f"{bmf['strict_shared_constant']:.3f}, "
            f"median_delta_bic_indep_minus_shared={row['delta_bic_independent_minus_shared']['p50']:.3f}, "
            f"median_delta_bic_strict_minus_shared={row['delta_bic_strict_minus_shared']['p50']:.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def _make_plots(out_dir: Path, summary: dict[str, Any]) -> None:
    gi = summary["gamma_identifiability"]
    gt = sorted(set(float(r["gamma_true"]) for r in gi))
    ss = sorted(set(float(r["sigma_scale"]) for r in gi))

    heat = np.full((len(gt), len(ss)), np.nan, dtype=float)
    for r in gi:
        i = gt.index(float(r["gamma_true"]))
        j = ss.index(float(r["sigma_scale"]))
        heat[i, j] = float(r["frac_well_constrained"])

    fig0, ax0 = plt.subplots(figsize=(7.5, 3.8))
    im = ax0.imshow(heat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax0.set_xticks(np.arange(len(ss)))
    ax0.set_xticklabels([f"{x:.2f}" for x in ss])
    ax0.set_yticks(np.arange(len(gt)))
    ax0.set_yticklabels([f"{x:.3f}" for x in gt])
    ax0.set_xlabel("Sigma scale")
    ax0.set_ylabel("Gamma truth")
    ax0.set_title("Fraction gamma well-constrained")
    fig0.colorbar(im, ax=ax0, label="Fraction")
    fig0.tight_layout()
    fig0.savefig(out_dir / "gamma_identifiability_heatmap.png", dpi=180)
    plt.close(fig0)

    ms = summary["model_selection"]
    labels = [f"{r['truth_name']}\n(s={r['sigma_scale']:.2f})" for r in ms]
    shared = [r["best_model_fractions"]["shared_gamma_per_probe_slope"] for r in ms]
    indep = [r["best_model_fractions"]["independent_gamma_per_probe_slope"] for r in ms]
    strict = [r["best_model_fractions"]["strict_shared_constant"] for r in ms]
    x = np.arange(len(labels))

    fig1, ax1 = plt.subplots(figsize=(10.5, 4.2))
    ax1.bar(x, shared, label="best shared")
    ax1.bar(x, indep, bottom=shared, label="best independent")
    ax1.bar(x, strict, bottom=np.asarray(shared) + np.asarray(indep), label="best strict")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Fraction")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Model-selection outcomes by truth scenario")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "model_selection_fractions.png", dpi=180)
    plt.close(fig1)


def main() -> None:
    args = parse_args()
    if args.gamma_min >= args.gamma_max:
        raise ValueError("require gamma_min < gamma_max")
    if args.n_ident <= 0 or args.n_select <= 0:
        raise ValueError("n-ident and n-select must be positive")

    probes = load_config(Path(args.config))
    flat_template = flatten_probes(probes)
    gamma_truth_grid = _parse_grid(args.gamma_truth_grid)
    sigma_scale_grid = _parse_grid(args.sigma_scale_grid)

    out_dir = Path(args.out_root) / (
        args.run_tag if args.run_tag else f"shared_redshift_identifiability_{utc_tag()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_shared_obs = _fit_shared_gamma(
        flat_template, gamma_min=args.gamma_min, gamma_max=args.gamma_max
    )
    fit_indep_obs = _fit_independent_gamma(
        flat_template, gamma_min=args.gamma_min, gamma_max=args.gamma_max
    )
    fit_strict_obs = _fit_strict_constant(
        flat_template, gamma_min=args.gamma_min, gamma_max=args.gamma_max
    )
    fit_shared_3p67 = _fit_shared_at_fixed_gamma(flat_template, gamma=3.67)
    fit_strict_3p67 = _fit_strict_at_fixed_gamma(flat_template, gamma=3.67)

    truth_mu_map = {
        "shared_obs_gamma": np.asarray(
            _fit_shared_at_fixed_gamma(
                flat_template, gamma=float(fit_shared_obs["gamma"])
            )["y_hat"],
            dtype=float,
        ),
        "shared_gamma_3p67": np.asarray(fit_shared_3p67["y_hat"], dtype=float),
        "independent_obs": _mu_from_independent(flat_template, fit_indep_obs),
        "strict_gamma_3p67": np.asarray(fit_strict_3p67["y_hat"], dtype=float),
        "strict_obs_gamma": np.asarray(
            _fit_strict_at_fixed_gamma(
                flat_template, gamma=float(fit_strict_obs["gamma"])
            )["y_hat"],
            dtype=float,
        ),
    }

    rng_master = np.random.default_rng(args.seed)

    gamma_ident_rows = []
    for gamma_true in gamma_truth_grid:
        for sigma_scale in sigma_scale_grid:
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            row = _run_gamma_identifiability(
                flat_template=flat_template,
                gamma_true=gamma_true,
                sigma_scale=sigma_scale,
                n_rep=int(args.n_ident),
                gamma_min=args.gamma_min,
                gamma_max=args.gamma_max,
                gamma_grid_size=int(args.gamma_grid_size),
                rng=rng,
            )
            row["seed"] = seed
            gamma_ident_rows.append(row)

    model_sel_rows = []
    for truth_name in [
        "shared_obs_gamma",
        "shared_gamma_3p67",
        "independent_obs",
        "strict_gamma_3p67",
    ]:
        for sigma_scale in [1.0, 0.5, 0.25]:
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            row = _run_model_selection(
                flat_template=flat_template,
                truth_name=truth_name,
                mu_truth=np.asarray(truth_mu_map[truth_name], dtype=float),
                sigma_scale=sigma_scale,
                n_rep=int(args.n_select),
                gamma_min=args.gamma_min,
                gamma_max=args.gamma_max,
                rng=rng,
            )
            row["seed"] = seed
            model_sel_rows.append(row)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "n_points": int(flat_template.y.size),
        "n_probes": int(len(flat_template.probe_names)),
        "probe_names": list(flat_template.probe_names),
        "settings": {
            "gamma_min": float(args.gamma_min),
            "gamma_max": float(args.gamma_max),
            "gamma_grid_size": int(args.gamma_grid_size),
            "gamma_truth_grid": gamma_truth_grid,
            "sigma_scale_grid": sigma_scale_grid,
            "n_ident": int(args.n_ident),
            "n_select": int(args.n_select),
            "seed": int(args.seed),
        },
        "observed_fits": {
            "shared": fit_shared_obs,
            "independent": fit_indep_obs,
            "strict": fit_strict_obs,
        },
        "gamma_identifiability": gamma_ident_rows,
        "model_selection": model_sel_rows,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(out_dir / "master_report.md", summary)
    _make_plots(out_dir, summary)

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "n_gamma_identifiability_rows": len(gamma_ident_rows),
                "n_model_selection_rows": len(model_sel_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
