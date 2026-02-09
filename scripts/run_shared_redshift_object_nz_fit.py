#!/usr/bin/env python3
"""Hierarchical effective-n(z) mapping fit for shared-redshift dipole modeling."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fit-input",
        default="configs/shared_redshift_object_level_fit_input.json",
    )
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--gamma-grid-size", type=int, default=121)
    ap.add_argument(
        "--shared-radio-eta",
        action="store_true",
        help="Tie all radio eta parameters to one shared eta.",
    )
    return ap.parse_args()


def _safe_feature(z: np.ndarray, gamma: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.exp(float(gamma) * np.log1p(z)) - 1.0


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    w = 1.0 / np.asarray(s, dtype=float)
    xw = x * w[:, None]
    yw = y * w
    beta, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = y - x @ beta
    chi2 = float(np.sum((resid / s) ** 2))
    return beta, chi2


def _interval_from_profile(grid: np.ndarray, score: np.ndarray, delta: float = 1.0) -> tuple[float, float] | None:
    m = np.isfinite(score)
    if not np.any(m):
        return None
    s = score[m]
    g = grid[m]
    smin = float(np.min(s))
    mask = s <= (smin + float(delta))
    if not np.any(mask):
        return None
    idx = np.where(mask)[0]
    return float(g[idx[0]]), float(g[idx[-1]])


def _local_sigma_fd(
    model: "MappingFitModel",
    x_map: np.ndarray,
    index: int,
    prior_sigma: float,
    bounds: tuple[float, float],
) -> float:
    """Finite-difference local sigma from objective curvature at the MAP."""
    x0 = np.asarray(x_map, dtype=float)
    f0 = float(model.objective(x0))
    step = max(1e-6, 0.1 * max(1e-6, float(prior_sigma)))
    left_room = float(x0[index] - bounds[0])
    right_room = float(bounds[1] - x0[index])
    step = min(step, 0.45 * left_room, 0.45 * right_room)
    if not np.isfinite(step) or step <= 1e-8:
        return float("nan")
    xp = x0.copy()
    xm = x0.copy()
    xp[index] += step
    xm[index] -= step
    fp = float(model.objective(xp))
    fm = float(model.objective(xm))
    sec = (fp - 2.0 * f0 + fm) / (step * step)
    if not np.isfinite(sec) or sec <= 0.0:
        return float("nan")
    return float(math.sqrt(1.0 / sec))


class MappingFitModel:
    def __init__(self, cfg: dict[str, Any], shared_radio_eta: bool = False) -> None:
        self.cfg = cfg
        self.points = list(cfg["points"])
        self.mapping = dict(cfg["mapping_model"])
        self.priors = dict(cfg["priors"])
        self.shared_radio_eta = bool(shared_radio_eta)

        self.probe_names = sorted({str(p["probe_name"]) for p in self.points})
        self.probe_to_idx = {name: i for i, name in enumerate(self.probe_names)}

        self.y = np.asarray([float(p["value"]) for p in self.points], dtype=float)
        self.s = np.asarray([max(1e-9, float(p["sigma"])) for p in self.points], dtype=float)
        self.selection = np.asarray([float(p["selection_value"]) for p in self.points], dtype=float)
        self.probe_idx = np.asarray([self.probe_to_idx[str(p["probe_name"])] for p in self.points], dtype=int)
        self.family = np.asarray([str(p["family"]) for p in self.points], dtype=object)
        self.survey = np.asarray([str(p["survey"]) for p in self.points], dtype=object)

        # Nonlinear parameter registry.
        self.p_names: list[str] = []
        self.p_mu: list[float] = []
        self.p_sigma: list[float] = []
        self.p_bounds: list[tuple[float, float]] = []

        def add_param(name: str, mu: float, sigma: float, bounds: list[float] | tuple[float, float]) -> None:
            b0, b1 = float(bounds[0]), float(bounds[1])
            self.p_names.append(name)
            self.p_mu.append(float(mu))
            self.p_sigma.append(max(1e-9, float(sigma)))
            self.p_bounds.append((b0, b1))

        gamma_prior = self.priors["gamma"]
        add_param(
            "gamma",
            mu=float(gamma_prior["mu"]),
            sigma=float(gamma_prior["sigma"]),
            bounds=gamma_prior["bounds"],
        )

        q_prior = self.priors["quasar"]
        add_param(
            "q_z_ref",
            mu=float(q_prior["z_ref"]["mu"]),
            sigma=float(q_prior["z_ref"]["sigma"]),
            bounds=q_prior["z_ref"]["bounds"],
        )
        add_param(
            "q_slope",
            mu=float(q_prior["slope_per_mag"]["mu"]),
            sigma=float(q_prior["slope_per_mag"]["sigma"]),
            bounds=q_prior["slope_per_mag"]["bounds"],
        )

        self.radio_surveys = sorted(self.priors["radio"].keys())
        for survey in self.radio_surveys:
            rp = self.priors["radio"][survey]
            add_param(
                f"r_{survey}_z_ref",
                mu=float(rp["z_ref"]["mu"]),
                sigma=float(rp["z_ref"]["sigma"]),
                bounds=rp["z_ref"]["bounds"],
            )

        if self.shared_radio_eta:
            # Use pooled prior around the mean of survey priors.
            eta_mus = [float(self.priors["radio"][s]["eta"]["mu"]) for s in self.radio_surveys]
            eta_sigmas = [float(self.priors["radio"][s]["eta"]["sigma"]) for s in self.radio_surveys]
            eta_bounds = [self.priors["radio"][s]["eta"]["bounds"] for s in self.radio_surveys]
            lo = max(float(b[0]) for b in eta_bounds)
            hi = min(float(b[1]) for b in eta_bounds)
            add_param(
                "r_eta_shared",
                mu=float(np.mean(eta_mus)),
                sigma=float(np.mean(eta_sigmas)),
                bounds=[lo, hi],
            )
        else:
            for survey in self.radio_surveys:
                rp = self.priors["radio"][survey]
                add_param(
                    f"r_{survey}_eta",
                    mu=float(rp["eta"]["mu"]),
                    sigma=float(rp["eta"]["sigma"]),
                    bounds=rp["eta"]["bounds"],
                )

        self.p_mu_arr = np.asarray(self.p_mu, dtype=float)
        self.p_sigma_arr = np.asarray(self.p_sigma, dtype=float)
        self.p_bounds_arr = np.asarray(self.p_bounds, dtype=float)

    def _as_dict(self, x: np.ndarray) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.p_names, np.asarray(x, dtype=float))}

    def z_eff(self, params: dict[str, float]) -> np.ndarray:
        z = np.zeros_like(self.y, dtype=float)
        w1_ref = float(self.mapping["quasar"]["w1_ref"])
        radio_surv_map = dict(self.mapping["radio"]["surveys"])
        for i in range(z.size):
            fam = str(self.family[i])
            if fam == "quasar":
                z_i = float(params["q_z_ref"] + params["q_slope"] * (self.selection[i] - w1_ref))
            elif fam == "radio":
                s = str(self.survey[i])
                z_ref = float(params[f"r_{s}_z_ref"])
                if self.shared_radio_eta:
                    eta = float(params["r_eta_shared"])
                else:
                    eta = float(params[f"r_{s}_eta"])
                cut_ref = float(radio_surv_map[s]["cut_ref_mjy"])
                z_i = float(z_ref * ((cut_ref / self.selection[i]) ** eta))
            else:
                raise ValueError(f"unknown family: {fam}")
            z[i] = max(1e-4, z_i)
        return z

    def design(self, gamma: float, z_eff: np.ndarray) -> np.ndarray:
        n = self.y.size
        p = len(self.probe_names)
        f = _safe_feature(z_eff, gamma)
        x = np.zeros((n, 2 * p), dtype=float)
        for i in range(p):
            m = self.probe_idx == i
            x[m, i] = 1.0
            x[m, p + i] = f[m]
        return x

    def objective(self, x_nonlin: np.ndarray) -> float:
        x_nonlin = np.asarray(x_nonlin, dtype=float)
        # Hard bounds as guardrail.
        if np.any(x_nonlin < self.p_bounds_arr[:, 0]) or np.any(x_nonlin > self.p_bounds_arr[:, 1]):
            return 1e30
        params = self._as_dict(x_nonlin)
        z_eff = self.z_eff(params)
        x = self.design(gamma=params["gamma"], z_eff=z_eff)
        _, chi2_data = _weighted_lstsq(x, self.y, self.s)
        chi2_prior = float(np.sum(((x_nonlin - self.p_mu_arr) / self.p_sigma_arr) ** 2))
        return float(0.5 * (chi2_data + chi2_prior))

    def solve_linear(self, x_nonlin: np.ndarray) -> dict[str, Any]:
        params = self._as_dict(np.asarray(x_nonlin, dtype=float))
        z_eff = self.z_eff(params)
        x = self.design(gamma=params["gamma"], z_eff=z_eff)
        beta, chi2_data = _weighted_lstsq(x, self.y, self.s)
        chi2_prior = float(np.sum(((np.asarray(x_nonlin, dtype=float) - self.p_mu_arr) / self.p_sigma_arr) ** 2))
        p = len(self.probe_names)
        return {
            "params": params,
            "z_eff": z_eff,
            "beta": beta,
            "intercepts": {self.probe_names[i]: float(beta[i]) for i in range(p)},
            "slopes": {self.probe_names[i]: float(beta[p + i]) for i in range(p)},
            "chi2_data": float(chi2_data),
            "chi2_prior": float(chi2_prior),
            "objective": float(0.5 * (chi2_data + chi2_prior)),
        }


def _write_report(out_path: Path, summary: dict[str, Any]) -> None:
    s = summary
    lines = []
    lines.append("# Shared Redshift Object-level Mapping Fit")
    lines.append("")
    lines.append(f"- Created UTC: `{s['created_utc']}`")
    lines.append(f"- Fit input: `{s['fit_input']}`")
    lines.append(f"- Points: `{s['n_points']}` across probes `{s['probe_names']}`")
    lines.append(f"- Shared radio eta: `{s['shared_radio_eta']}`")
    lines.append("")
    lines.append("## MAP fit")
    lines.append("")
    lines.append(f"- Objective (0.5*chi2_total): `{s['map']['objective']:.6f}`")
    lines.append(f"- chi2_data: `{s['map']['chi2_data']:.6f}`")
    lines.append(f"- chi2_prior: `{s['map']['chi2_prior']:.6f}`")
    lines.append(f"- gamma_map: `{s['map']['params']['gamma']:.6f}`")
    lines.append(
        f"- gamma profile 1-sigma interval (delta chi2_eff=1): `{s['gamma_profile']['interval_1sigma']}`"
    )
    lines.append(f"- gamma profile constrained on grid: `{s['gamma_profile']['is_well_constrained_on_grid']}`")
    lines.append(f"- gamma 1-sigma width: `{s['gamma_profile']['width_1sigma']:.6f}`")
    lines.append(
        f"- gamma information ratio (width/(2*prior_sigma)): `{s['gamma_profile']['information_ratio_vs_prior']:.6f}`"
    )
    lines.append(f"- gamma decisively constrained: `{s['gamma_profile']['is_decisive']}`")
    lines.append("")
    lines.append("## Nonlinear identifiability")
    lines.append("")
    for r in s["nonlinear_param_diagnostics"]:
        lines.append(
            f"- {r['name']}: map={r['map']:.5f}, prior_sigma={r['prior_sigma']:.5f}, "
            f"post_sigma_fd={r['post_sigma_fd']:.5f}, shrinkage_fd={r['shrinkage_fd']:.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def _plot_gamma_profile(out_png: Path, profile: dict[str, Any]) -> None:
    g = np.asarray(profile["gamma_grid"], dtype=float)
    dchi2 = np.asarray(profile["delta_chi2_eff_grid"], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(g, dchi2, lw=2.0)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.axhline(4.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("gamma")
    ax.set_ylabel("delta chi2_eff")
    ax.set_title("Gamma profile (mapping params profiled)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fit_input_path = Path(args.fit_input).resolve()
    cfg = json.loads(fit_input_path.read_text())

    out_dir = Path(args.out_root) / (
        args.run_tag if args.run_tag else f"shared_redshift_object_nz_fit_{utc_tag()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MappingFitModel(cfg=cfg, shared_radio_eta=bool(args.shared_radio_eta))

    x0 = model.p_mu_arr.copy()
    res = minimize(
        model.objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=[tuple(b) for b in model.p_bounds],
    )
    x_map = np.asarray(res.x, dtype=float)
    map_sol = model.solve_linear(x_map)

    # Approximate posterior covariance of nonlinear params from inverse Hessian.
    try:
        cov = np.asarray(res.hess_inv.todense(), dtype=float)  # type: ignore[attr-defined]
    except Exception:
        cov = np.full((len(model.p_names), len(model.p_names)), np.nan, dtype=float)
    post_std_laplace = np.sqrt(np.maximum(0.0, np.diag(cov)))

    post_std_fd = np.asarray(
        [
            _local_sigma_fd(
                model=model,
                x_map=x_map,
                index=i,
                prior_sigma=float(model.p_sigma_arr[i]),
                bounds=tuple(model.p_bounds[i]),
            )
            for i in range(len(model.p_names))
        ],
        dtype=float,
    )

    # Profile gamma by optimizing all other nonlinear params at fixed gamma.
    gamma_bounds = model.p_bounds[0]
    gamma_grid = np.linspace(float(gamma_bounds[0]), float(gamma_bounds[1]), int(args.gamma_grid_size))

    x_curr = x_map.copy()
    prof_obj = []
    for g in gamma_grid:
        def obj_rest(x_rest: np.ndarray) -> float:
            x_full = np.concatenate(([float(g)], np.asarray(x_rest, dtype=float)))
            return model.objective(x_full)

        b_rest = model.p_bounds[1:]
        res_rest = minimize(
            obj_rest,
            x0=x_curr[1:],
            method="L-BFGS-B",
            bounds=[tuple(b) for b in b_rest],
        )
        x_curr = np.concatenate(([float(g)], np.asarray(res_rest.x, dtype=float)))
        prof_obj.append(float(model.objective(x_curr)))

    prof_obj_arr = np.asarray(prof_obj, dtype=float)
    obj_min = float(np.min(prof_obj_arr))
    delta_chi2_eff = 2.0 * (prof_obj_arr - obj_min)
    int1 = _interval_from_profile(gamma_grid, delta_chi2_eff, delta=1.0)
    int2 = _interval_from_profile(gamma_grid, delta_chi2_eff, delta=4.0)
    constrained = bool(int1 is not None and int2 is not None)
    if int1 is not None:
        constrained = constrained and (int1[0] > gamma_grid[0]) and (int1[1] < gamma_grid[-1])
    if int2 is not None:
        constrained = constrained and (int2[0] > gamma_grid[0]) and (int2[1] < gamma_grid[-1])
    if int1 is None:
        width_1sigma = float("nan")
    else:
        width_1sigma = float(int1[1] - int1[0])
    gamma_prior_sigma = float(model.p_sigma_arr[0])
    info_ratio = (
        float(width_1sigma / (2.0 * gamma_prior_sigma))
        if np.isfinite(width_1sigma) and gamma_prior_sigma > 0
        else float("nan")
    )
    is_decisive = bool(constrained and np.isfinite(width_1sigma) and (width_1sigma < 1.0))

    # Point-level table at MAP.
    point_rows = []
    for i, p in enumerate(model.points):
        row = dict(p)
        row["z_eff_map"] = float(map_sol["z_eff"][i])
        row["model_value_map"] = float(
            map_sol["intercepts"][str(p["probe_name"])]
            + map_sol["slopes"][str(p["probe_name"])]
            * _safe_feature(np.asarray([row["z_eff_map"]], dtype=float), map_sol["params"]["gamma"])[0]
        )
        row["residual_map"] = float(row["value"] - row["model_value_map"])
        point_rows.append(row)

    nl_diag = []
    for i, name in enumerate(model.p_names):
        ps = float(post_std_laplace[i]) if np.isfinite(post_std_laplace[i]) else float("nan")
        pr = float(model.p_sigma_arr[i])
        nl_diag.append(
            {
                "name": name,
                "map": float(x_map[i]),
                "prior_mu": float(model.p_mu_arr[i]),
                "prior_sigma": pr,
                "post_sigma_laplace": ps,
                "post_sigma_fd": float(post_std_fd[i]) if np.isfinite(post_std_fd[i]) else float("nan"),
                "shrinkage_laplace": float(ps / pr) if np.isfinite(ps) and pr > 0 else float("nan"),
                "shrinkage_fd": float(post_std_fd[i] / pr)
                if np.isfinite(post_std_fd[i]) and pr > 0
                else float("nan"),
            }
        )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit_input": str(fit_input_path),
        "shared_radio_eta": bool(args.shared_radio_eta),
        "n_points": int(len(model.points)),
        "n_probes": int(len(model.probe_names)),
        "probe_names": list(model.probe_names),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(res.success),
            "message": str(res.message),
            "n_iter": int(getattr(res, "nit", -1)),
        },
        "map": {
            "params": map_sol["params"],
            "intercepts": map_sol["intercepts"],
            "slopes": map_sol["slopes"],
            "objective": float(map_sol["objective"]),
            "chi2_data": float(map_sol["chi2_data"]),
            "chi2_prior": float(map_sol["chi2_prior"]),
        },
        "nonlinear_param_diagnostics": nl_diag,
        "gamma_profile": {
            "gamma_grid": gamma_grid.tolist(),
            "objective_grid": prof_obj_arr.tolist(),
            "delta_chi2_eff_grid": delta_chi2_eff.tolist(),
            "interval_1sigma": int1,
            "interval_2sigma": int2,
            "width_1sigma": width_1sigma,
            "information_ratio_vs_prior": info_ratio,
            "is_well_constrained_on_grid": bool(constrained),
            "is_decisive": bool(is_decisive),
            "gamma_best_profile": float(gamma_grid[int(np.argmin(prof_obj_arr))]),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "point_table_map.json").write_text(json.dumps(point_rows, indent=2) + "\n")

    # Write CSV for easier downstream use.
    csv_header = [
        "point_id",
        "probe_name",
        "family",
        "survey",
        "selection_kind",
        "selection_value",
        "label",
        "value",
        "sigma",
        "z_eff_map",
        "model_value_map",
        "residual_map",
    ]
    with (out_dir / "point_table_map.csv").open("w", newline="") as f:
        import csv

        w = csv.writer(f)
        w.writerow(csv_header)
        for r in point_rows:
            w.writerow([r.get(k) for k in csv_header])

    _write_report(out_dir / "master_report.md", summary)
    _plot_gamma_profile(out_dir / "gamma_profile.png", summary["gamma_profile"])

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "gamma_map": float(map_sol["params"]["gamma"]),
                "gamma_interval_1sigma": int1,
                "gamma_well_constrained": bool(constrained),
                "gamma_is_decisive": bool(is_decisive),
                "gamma_information_ratio_vs_prior": info_ratio,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
