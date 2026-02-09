#!/usr/bin/env python3
"""Cross-probe test for a shared redshift multiplier.

This script fits nested models to user-supplied measurements
`(probe, z_eff, value, sigma)`:

1) Shared-shape model (recommended baseline):
     value = a_probe + b_probe * [(1 + z)^gamma - 1]
2) Independent-shape model:
     value = a_probe + b_probe * [(1 + z)^gamma_probe - 1]
3) Strict shared-constant model:
     value = a_probe + b_shared * [(1 + z)^gamma - 1]

Model comparison (AIC/BIC) answers whether a common redshift scaling is supported
across probes and whether a single shared amplitude is tenable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


@dataclass(frozen=True)
class ProbeData:
    name: str
    z: np.ndarray
    y: np.ndarray
    s: np.ndarray
    labels: list[str]


@dataclass(frozen=True)
class FlatData:
    probe_names: list[str]
    probe_idx: np.ndarray
    z: np.ndarray
    y: np.ndarray
    s: np.ndarray
    labels: list[str]


def _safe_feature(z: np.ndarray, gamma: np.ndarray | float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    # Center at z=0 so intercept is directly interpretable and gamma is less degenerate.
    return np.exp(np.asarray(gamma, dtype=float) * np.log1p(z)) - 1.0


def _validate_point(p: dict[str, Any], probe_name: str, i: int) -> None:
    if "z" not in p or "value" not in p or "sigma" not in p:
        raise ValueError(f"{probe_name}: point {i} must contain z, value, sigma")
    z = float(p["z"])
    s = float(p["sigma"])
    if not np.isfinite(z) or z <= -1.0:
        raise ValueError(f"{probe_name}: point {i} has invalid z={z}")
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"{probe_name}: point {i} has invalid sigma={s}")


def load_config(config_path: Path) -> list[ProbeData]:
    cfg = json.loads(config_path.read_text())
    probes = cfg.get("probes")
    if not isinstance(probes, list) or len(probes) < 2:
        raise ValueError("config must contain at least two probes in `probes`")

    out: list[ProbeData] = []
    for probe in probes:
        name = str(probe.get("name", "")).strip()
        points = probe.get("points")
        if not name:
            raise ValueError("each probe must include a non-empty `name`")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"{name}: `points` must contain at least two entries")
        z_list: list[float] = []
        y_list: list[float] = []
        s_list: list[float] = []
        labels: list[str] = []
        for i, p in enumerate(points):
            if not isinstance(p, dict):
                raise ValueError(f"{name}: point {i} must be an object")
            _validate_point(p, name, i)
            z_list.append(float(p["z"]))
            y_list.append(float(p["value"]))
            s_list.append(float(p["sigma"]))
            labels.append(str(p.get("label", f"{name}_{i}")))
        out.append(
            ProbeData(
                name=name,
                z=np.asarray(z_list, dtype=float),
                y=np.asarray(y_list, dtype=float),
                s=np.asarray(s_list, dtype=float),
                labels=labels,
            )
        )
    return out


def flatten_probes(probes: list[ProbeData]) -> FlatData:
    names = [p.name for p in probes]
    probe_idx: list[np.ndarray] = []
    z: list[np.ndarray] = []
    y: list[np.ndarray] = []
    s: list[np.ndarray] = []
    labels: list[str] = []
    for i, p in enumerate(probes):
        n = len(p.z)
        probe_idx.append(np.full(n, i, dtype=int))
        z.append(p.z)
        y.append(p.y)
        s.append(p.s)
        labels.extend(p.labels)
    return FlatData(
        probe_names=names,
        probe_idx=np.concatenate(probe_idx),
        z=np.concatenate(z),
        y=np.concatenate(y),
        s=np.concatenate(s),
        labels=labels,
    )


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    w = 1.0 / np.asarray(s, dtype=float)
    Xw = X * w[:, None]
    yw = y * w
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    chi2 = float(np.sum((resid / s) ** 2))
    return beta, chi2


def _design_shared_gamma(flat: FlatData, gamma: float) -> np.ndarray:
    n = flat.y.size
    p = len(flat.probe_names)
    f = _safe_feature(flat.z, gamma)
    X = np.zeros((n, 2 * p), dtype=float)
    for i in range(p):
        m = flat.probe_idx == i
        X[m, i] = 1.0
        X[m, p + i] = f[m]
    return X


def _design_independent_gamma(flat: FlatData, gammas: np.ndarray) -> np.ndarray:
    n = flat.y.size
    p = len(flat.probe_names)
    X = np.zeros((n, 2 * p), dtype=float)
    for i in range(p):
        m = flat.probe_idx == i
        fi = _safe_feature(flat.z[m], gammas[i])
        X[m, i] = 1.0
        X[m, p + i] = fi
    return X


def _design_strict_constant(flat: FlatData, gamma: float) -> np.ndarray:
    n = flat.y.size
    p = len(flat.probe_names)
    f = _safe_feature(flat.z, gamma)
    X = np.zeros((n, p + 1), dtype=float)
    for i in range(p):
        X[flat.probe_idx == i, i] = 1.0
    X[:, p] = f
    return X


def _loglike(chi2: float, s: np.ndarray) -> float:
    s = np.asarray(s, dtype=float)
    return float(-0.5 * (chi2 + np.sum(np.log(2.0 * np.pi * s * s))))


def _fit_shared_gamma(flat: FlatData, gamma_min: float, gamma_max: float) -> dict[str, Any]:
    def objective(x: np.ndarray) -> float:
        g = float(x[0])
        X = _design_shared_gamma(flat, g)
        _, c2 = _weighted_lstsq(X, flat.y, flat.s)
        return c2

    res = minimize(
        objective,
        x0=np.array([0.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(gamma_min, gamma_max)],
    )
    g = float(res.x[0])
    X = _design_shared_gamma(flat, g)
    beta, c2 = _weighted_lstsq(X, flat.y, flat.s)
    p = len(flat.probe_names)
    a = beta[:p]
    b = beta[p:]
    k = 2 * p + 1
    n = flat.y.size
    dof = n - k
    ll = _loglike(c2, flat.s)
    return {
        "name": "shared_gamma_per_probe_slope",
        "converged": bool(res.success),
        "message": str(res.message),
        "gamma": g,
        "intercepts": {flat.probe_names[i]: float(a[i]) for i in range(p)},
        "slopes": {flat.probe_names[i]: float(b[i]) for i in range(p)},
        "chi2": float(c2),
        "dof": int(dof),
        "chi2_red": float(c2 / dof) if dof > 0 else float("nan"),
        "p_value": float(chi2_dist.sf(c2, dof)) if dof > 0 else float("nan"),
        "loglike": ll,
        "aic": float(2 * k - 2 * ll),
        "bic": float(math.log(n) * k - 2 * ll),
        "n_params": int(k),
    }


def _profile_shared_gamma(
    flat: FlatData, gamma_min: float, gamma_max: float, n_grid: int
) -> dict[str, Any]:
    if n_grid < 25:
        raise ValueError("n_grid must be >= 25 for a useful profile")
    grid = np.linspace(gamma_min, gamma_max, int(n_grid))
    chi2_vals: list[float] = []
    for g in grid:
        X = _design_shared_gamma(flat, float(g))
        _, c2 = _weighted_lstsq(X, flat.y, flat.s)
        chi2_vals.append(float(c2))
    chi2_arr = np.asarray(chi2_vals, dtype=float)
    i_min = int(np.argmin(chi2_arr))
    g_best = float(grid[i_min])
    c2_min = float(chi2_arr[i_min])

    def _interval(delta: float) -> tuple[float, float] | None:
        mask = chi2_arr <= (c2_min + float(delta))
        if not np.any(mask):
            return None
        idx = np.where(mask)[0]
        return float(grid[idx[0]]), float(grid[idx[-1]])

    int_1sigma = _interval(1.0)
    int_2sigma = _interval(4.0)
    constrained = True
    for interval in [int_1sigma, int_2sigma]:
        if interval is None:
            constrained = False
            continue
        lo, hi = interval
        if abs(lo - gamma_min) < 1e-12 or abs(hi - gamma_max) < 1e-12:
            constrained = False
    return {
        "gamma_grid_min": float(gamma_min),
        "gamma_grid_max": float(gamma_max),
        "n_grid": int(n_grid),
        "gamma_best_grid": g_best,
        "chi2_min_grid": c2_min,
        "interval_1sigma_delta_chi2_1": int_1sigma,
        "interval_2sigma_delta_chi2_4": int_2sigma,
        "is_well_constrained_on_grid": bool(constrained),
        "grid": [{"gamma": float(g), "chi2": float(c)} for g, c in zip(grid, chi2_arr)],
    }


def _fit_independent_gamma(flat: FlatData, gamma_min: float, gamma_max: float) -> dict[str, Any]:
    p = len(flat.probe_names)

    def objective(x: np.ndarray) -> float:
        gammas = np.asarray(x, dtype=float)
        X = _design_independent_gamma(flat, gammas)
        _, c2 = _weighted_lstsq(X, flat.y, flat.s)
        return c2

    res = minimize(
        objective,
        x0=np.zeros(p, dtype=float),
        method="L-BFGS-B",
        bounds=[(gamma_min, gamma_max)] * p,
    )
    gammas = np.asarray(res.x, dtype=float)
    X = _design_independent_gamma(flat, gammas)
    beta, c2 = _weighted_lstsq(X, flat.y, flat.s)
    a = beta[:p]
    b = beta[p:]
    k = 3 * p
    n = flat.y.size
    dof = n - k
    ll = _loglike(c2, flat.s)
    return {
        "name": "independent_gamma_per_probe_slope",
        "converged": bool(res.success),
        "message": str(res.message),
        "gammas": {flat.probe_names[i]: float(gammas[i]) for i in range(p)},
        "intercepts": {flat.probe_names[i]: float(a[i]) for i in range(p)},
        "slopes": {flat.probe_names[i]: float(b[i]) for i in range(p)},
        "chi2": float(c2),
        "dof": int(dof),
        "chi2_red": float(c2 / dof) if dof > 0 else float("nan"),
        "p_value": float(chi2_dist.sf(c2, dof)) if dof > 0 else float("nan"),
        "loglike": ll,
        "aic": float(2 * k - 2 * ll),
        "bic": float(math.log(n) * k - 2 * ll),
        "n_params": int(k),
    }


def _fit_strict_constant(flat: FlatData, gamma_min: float, gamma_max: float) -> dict[str, Any]:
    def objective(x: np.ndarray) -> float:
        g = float(x[0])
        X = _design_strict_constant(flat, g)
        _, c2 = _weighted_lstsq(X, flat.y, flat.s)
        return c2

    res = minimize(
        objective,
        x0=np.array([0.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(gamma_min, gamma_max)],
    )
    g = float(res.x[0])
    X = _design_strict_constant(flat, g)
    beta, c2 = _weighted_lstsq(X, flat.y, flat.s)
    p = len(flat.probe_names)
    a = beta[:p]
    b_shared = float(beta[p])
    k = p + 2
    n = flat.y.size
    dof = n - k
    ll = _loglike(c2, flat.s)
    return {
        "name": "strict_shared_constant",
        "converged": bool(res.success),
        "message": str(res.message),
        "gamma": g,
        "intercepts": {flat.probe_names[i]: float(a[i]) for i in range(p)},
        "slope_shared": b_shared,
        "chi2": float(c2),
        "dof": int(dof),
        "chi2_red": float(c2 / dof) if dof > 0 else float("nan"),
        "p_value": float(chi2_dist.sf(c2, dof)) if dof > 0 else float("nan"),
        "loglike": ll,
        "aic": float(2 * k - 2 * ll),
        "bic": float(math.log(n) * k - 2 * ll),
        "n_params": int(k),
    }


def _bootstrap_shared_gamma(
    probes: list[ProbeData],
    gamma_min: float,
    gamma_max: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if n_bootstrap <= 0:
        return {"n_bootstrap": 0}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        boot_probes: list[ProbeData] = []
        for p in probes:
            n = p.z.size
            idx = rng.integers(0, n, size=n)
            boot_probes.append(
                ProbeData(
                    name=p.name,
                    z=p.z[idx],
                    y=p.y[idx],
                    s=p.s[idx],
                    labels=[p.labels[j] for j in idx],
                )
            )
        flat = flatten_probes(boot_probes)
        fit = _fit_shared_gamma(flat, gamma_min=gamma_min, gamma_max=gamma_max)
        if fit["converged"] and np.isfinite(fit["gamma"]):
            values.append(float(fit["gamma"]))
    if not values:
        return {"n_bootstrap": n_bootstrap, "n_valid": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n_bootstrap": int(n_bootstrap),
        "n_valid": int(arr.size),
        "gamma_mean": float(np.mean(arr)),
        "gamma_std": float(np.std(arr)),
        "gamma_p16": float(np.percentile(arr, 16.0)),
        "gamma_p50": float(np.percentile(arr, 50.0)),
        "gamma_p84": float(np.percentile(arr, 84.0)),
    }


def _format_value(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.5f}"


def _write_markdown_report(
    out_path: Path,
    config_path: Path,
    fit_shared: dict[str, Any],
    fit_indep: dict[str, Any],
    fit_strict: dict[str, Any],
    comparisons: dict[str, Any],
    bootstrap: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Shared Redshift Multiplier Test")
    lines.append("")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Created UTC: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("## Model comparison")
    lines.append("")
    lines.append("| model | chi2 | dof | chi2/dof | gamma | AIC | BIC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| shared_gamma_per_probe_slope | "
        f"{_format_value(fit_shared['chi2'])} | {fit_shared['dof']} | {_format_value(fit_shared['chi2_red'])} | "
        f"{_format_value(fit_shared['gamma'])} | {_format_value(fit_shared['aic'])} | {_format_value(fit_shared['bic'])} |"
    )
    lines.append(
        "| independent_gamma_per_probe_slope | "
        f"{_format_value(fit_indep['chi2'])} | {fit_indep['dof']} | {_format_value(fit_indep['chi2_red'])} | "
        "n/a | "
        f"{_format_value(fit_indep['aic'])} | {_format_value(fit_indep['bic'])} |"
    )
    lines.append(
        "| strict_shared_constant | "
        f"{_format_value(fit_strict['chi2'])} | {fit_strict['dof']} | {_format_value(fit_strict['chi2_red'])} | "
        f"{_format_value(fit_strict['gamma'])} | {_format_value(fit_strict['aic'])} | {_format_value(fit_strict['bic'])} |"
    )
    lines.append("")
    lines.append("## Key deltas (positive favors denominator model)")
    lines.append("")
    lines.append(
        f"- `delta_bic_independent_minus_shared = {comparisons['delta_bic_independent_minus_shared']:.5f}` "
        "(>0 favors shared gamma)"
    )
    lines.append(
        f"- `delta_bic_strict_minus_shared = {comparisons['delta_bic_strict_minus_shared']:.5f}` "
        "(>0 favors per-probe slope over strict shared constant)"
    )
    lines.append("")
    lines.append("## Shared-gamma profile")
    lines.append("")
    lines.append(
        f"- grid best: `gamma={profile['gamma_best_grid']:.5f}`, `chi2_min={profile['chi2_min_grid']:.5f}`"
    )
    int1 = profile.get("interval_1sigma_delta_chi2_1")
    int2 = profile.get("interval_2sigma_delta_chi2_4")
    lines.append(f"- 1-sigma (delta chi2=1) interval: `{int1}`")
    lines.append(f"- 2-sigma (delta chi2=4) interval: `{int2}`")
    lines.append(f"- well-constrained on grid: `{bool(profile['is_well_constrained_on_grid'])}`")
    lines.append("")
    if bootstrap.get("n_bootstrap", 0) > 0 and bootstrap.get("n_valid", 0) > 0:
        lines.append("## Bootstrap")
        lines.append("")
        lines.append(
            "- shared-gamma bootstrap: "
            f"`n_valid={bootstrap['n_valid']}/{bootstrap['n_bootstrap']}`, "
            f"`gamma p16/p50/p84 = {bootstrap['gamma_p16']:.5f} / {bootstrap['gamma_p50']:.5f} / {bootstrap['gamma_p84']:.5f}`"
        )
        lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If `delta_bic_independent_minus_shared > +6`, shared redshift shape is strongly supported.")
    lines.append("- If `delta_bic_independent_minus_shared < -6`, probes prefer different redshift shapes.")
    lines.append("- If `delta_bic_strict_minus_shared < -6`, a single shared amplitude is likely too rigid.")
    lines.append("- Features are centered as `[(1+z)^gamma - 1]` to reduce intercept/gamma degeneracy.")
    out_path.write_text("\n".join(lines) + "\n")


def _make_plot(
    out_png: Path,
    probes: list[ProbeData],
    fit_shared: dict[str, Any],
    fit_indep: dict[str, Any],
    fit_strict: dict[str, Any],
) -> None:
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 6))
    z_all = np.concatenate([p.z for p in probes])
    z_min = max(0.0, float(np.min(z_all)) - 0.05)
    z_max = float(np.max(z_all)) + 0.05
    z_grid = np.linspace(z_min, z_max, 200)

    for i, p in enumerate(probes):
        c = colors(i % 10)
        ax.errorbar(p.z, p.y, yerr=p.s, fmt="o", color=c, alpha=0.85, label=f"{p.name} data")

        a_shared = float(fit_shared["intercepts"][p.name])
        b_shared = float(fit_shared["slopes"][p.name])
        g_shared = float(fit_shared["gamma"])
        y_shared = a_shared + b_shared * _safe_feature(z_grid, g_shared)
        ax.plot(z_grid, y_shared, color=c, lw=2.0, alpha=0.9, label=f"{p.name} shared-gamma")

        a_ind = float(fit_indep["intercepts"][p.name])
        b_ind = float(fit_indep["slopes"][p.name])
        g_ind = float(fit_indep["gammas"][p.name])
        y_ind = a_ind + b_ind * _safe_feature(z_grid, g_ind)
        ax.plot(z_grid, y_ind, color=c, lw=1.5, ls="--", alpha=0.8, label=f"{p.name} independent-gamma")

        a_str = float(fit_strict["intercepts"][p.name])
        b_str = float(fit_strict["slope_shared"])
        g_str = float(fit_strict["gamma"])
        y_str = a_str + b_str * _safe_feature(z_grid, g_str)
        ax.plot(z_grid, y_str, color=c, lw=1.2, ls=":", alpha=0.8, label=f"{p.name} strict-shared")

    ax.set_xlabel("Effective redshift z")
    ax.set_ylabel("Anomaly metric")
    ax.set_title("Shared redshift multiplier test")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON config with probes/points.")
    ap.add_argument("--out-root", default="outputs", help="Output root directory.")
    ap.add_argument("--run-tag", default=None, help="Optional fixed run tag.")
    ap.add_argument("--gamma-min", type=float, default=-5.0, help="Lower bound on gamma.")
    ap.add_argument("--gamma-max", type=float, default=5.0, help="Upper bound on gamma.")
    ap.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap draws for shared gamma.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap.")
    ap.add_argument(
        "--gamma-grid-size",
        type=int,
        default=401,
        help="Grid size for shared-gamma profile likelihood scan.",
    )
    ap.add_argument("--no-plot", action="store_true", help="Disable summary plot.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    probes = load_config(config_path)
    flat = flatten_probes(probes)

    if args.gamma_min >= args.gamma_max:
        raise ValueError("require gamma_min < gamma_max")

    out_dir = Path(args.out_root) / (
        args.run_tag if args.run_tag else f"shared_redshift_multiplier_test_{utc_tag()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_shared = _fit_shared_gamma(flat, gamma_min=args.gamma_min, gamma_max=args.gamma_max)
    profile = _profile_shared_gamma(
        flat,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        n_grid=int(args.gamma_grid_size),
    )
    fit_indep = _fit_independent_gamma(flat, gamma_min=args.gamma_min, gamma_max=args.gamma_max)
    fit_strict = _fit_strict_constant(flat, gamma_min=args.gamma_min, gamma_max=args.gamma_max)

    comparisons = {
        "delta_aic_independent_minus_shared": float(fit_indep["aic"] - fit_shared["aic"]),
        "delta_bic_independent_minus_shared": float(fit_indep["bic"] - fit_shared["bic"]),
        "delta_aic_strict_minus_shared": float(fit_strict["aic"] - fit_shared["aic"]),
        "delta_bic_strict_minus_shared": float(fit_strict["bic"] - fit_shared["bic"]),
    }
    bootstrap = _bootstrap_shared_gamma(
        probes,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "n_points": int(flat.y.size),
        "n_probes": int(len(flat.probe_names)),
        "probe_names": list(flat.probe_names),
        "fit_shared": fit_shared,
        "fit_independent": fit_indep,
        "fit_strict": fit_strict,
        "comparisons": comparisons,
        "bootstrap": bootstrap,
        "profile_shared_gamma": profile,
    }
    (out_dir / "shared_redshift_multiplier_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    _write_markdown_report(
        out_path=out_dir / "master_report.md",
        config_path=config_path,
        fit_shared=fit_shared,
        fit_indep=fit_indep,
        fit_strict=fit_strict,
        comparisons=comparisons,
        bootstrap=bootstrap,
        profile=profile,
    )

    if not args.no_plot:
        _make_plot(
            out_png=out_dir / "shared_redshift_multiplier_fit.png",
            probes=probes,
            fit_shared=fit_shared,
            fit_indep=fit_indep,
            fit_strict=fit_strict,
        )

    print(json.dumps({"status": "ok", "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
