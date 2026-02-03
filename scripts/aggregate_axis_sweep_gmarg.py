#!/usr/bin/env python3
"""Aggregate fixed-axis g-scan outputs into a single summary JSON.

This expects a sweep root directory containing:
  - axes.json (optional but recommended): list of axes that were run
  - many subdirs each with fixed_axis_gscan_full.json

The summary focuses on the g-marginalized evidence gain from allowing g to vary:
  logBF(mu(g) vs mu(g=0)) = logZ_mu(g) - lpd_mu(g=0),
which includes a proper prior-volume (complexity) penalty for g.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Sweep root directory.")
    ap.add_argument("--out", type=str, default="axis_sweep_summary.json", help="Output filename (relative to root).")
    ap.add_argument("--edge-mass-thresh", type=float, default=0.01, help="Posterior edge-mass threshold for boundary flags.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    axes_path = root / "axes.json"
    axes_meta = load_json(axes_path) if axes_path.exists() else None
    edge_thresh = float(args.edge_mass_thresh)

    runs = []
    for path_s in sorted(glob(str(root / "*" / "fixed_axis_gscan_full.json"))):
        path = Path(path_s)
        obj = load_json(path)

        axis = obj["run"]["axis"]
        g_marg = obj.get("g_marginalization", {})
        if not g_marg:
            raise RuntimeError(f"Missing g_marginalization in {path}")

        runs.append(
            {
                "path": str(path),
                "axis": axis,
                "n_draws": int(obj["run"]["n_draws"]),
                "n_events": int(obj["run"]["n_events"]),
                "g_grid": obj["g_grid"],
                "logBF_mu_over_mu0": float(g_marg["logBF_mu_over_mu0"]),
                "logBF_mu_over_gr": float(g_marg["logBF_mu_over_gr"]),
                "posterior_mean_g": float(g_marg["posterior_mean"]),
                "posterior_std_g": float(g_marg["posterior_std"]),
                "edge_mass_low": float(g_marg["edge_mass_low"]),
                "edge_mass_high": float(g_marg["edge_mass_high"]),
                "best_g_on_grid": float(g_marg["best_g_on_grid"]),
                "best_is_edge": bool(g_marg["best_is_edge"]),
                "g_prior": g_marg.get("prior", {}),
                "needs_g_extension": bool(
                    bool(g_marg["best_is_edge"]) or float(g_marg["edge_mass_low"]) > edge_thresh or float(g_marg["edge_mass_high"]) > edge_thresh
                ),
            }
        )

    # Identify random axes vs named axes if axes.json exists.
    random_names: set[str] = set()
    if axes_meta is not None and isinstance(axes_meta, dict) and "axes" in axes_meta:
        for a in axes_meta["axes"]:
            if str(a.get("kind", "")) == "random":
                random_names.add(str(a.get("name")))

    def _is_random(name: str) -> bool:
        return name in random_names if random_names else name.startswith("rand")

    random_runs = [r for r in runs if _is_random(str(r["axis"]["name"]))]
    special_runs = [r for r in runs if not _is_random(str(r["axis"]["name"]))]

    random_vals = np.array([r["logBF_mu_over_mu0"] for r in random_runs], dtype=float) if random_runs else np.array([], dtype=float)

    percentiles = {}
    if random_vals.size > 0 and np.all(np.isfinite(random_vals)):
        for r in special_runs:
            v = float(r["logBF_mu_over_mu0"])
            pct = float(np.mean(random_vals <= v))
            percentiles[str(r["axis"]["name"])] = {"logBF_mu_over_mu0": v, "percentile_among_random": pct}

    runs_sorted = sorted(runs, key=lambda r: float(r["logBF_mu_over_mu0"]), reverse=True)
    top10 = runs_sorted[:10]

    out = {
        "root": str(root),
        "n_axes_found": int(len(runs)),
        "n_random": int(len(random_runs)),
        "n_special": int(len(special_runs)),
        "edge_mass_thresh": float(edge_thresh),
        "top10_by_logBF_mu_over_mu0": top10,
        "special_percentiles": percentiles,
        "needs_g_extension": [r for r in runs_sorted if bool(r["needs_g_extension"])],
        "runs": runs_sorted,
    }

    out_path = root / str(args.out)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

