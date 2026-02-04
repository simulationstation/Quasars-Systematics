#!/usr/bin/env python3
"""
Diagnostics: can the fitted delta_m be tied to concrete survey/systematics mechanisms?

This script is meant to complement the "fit delta_m to cancel the dipole" mechanism check.
It produces several *independent* diagnostics using the Secrest/CatWISE catalog:

  1) "Count-match" magnitude shift along a chosen axis:
       Split the sky into hemispheres about the axis (cos>0 vs cos<0).
       Using the *same* base cuts, estimate how much you'd have to shift the faint cut
       (W1_max) in one hemisphere so that its cumulative counts match the other hemisphere.
       This yields an empirical delta_m that does NOT use the dipole vector sum directly.

  2) Sensitivity sweeps:
       - Dipole amplitude vs W1 faint cut (W1_max).
       - Dipole amplitude vs W1 coverage threshold (w1cov_min).
     If the effect is selection/systematics-driven, these curves often change materially
     with the cut location/quality cuts.

  3) Basic "mechanism correlates":
       Compare w1cov distributions between the two hemispheres about the axis.
       Large differences support an Eddington/photometric-scatter selection mechanism.

Outputs:
  - summary JSON
  - several plots (PNG)

Notes:
  - This does not claim the anomaly is "explained". It quantifies plausibility and
    points at which survey quantities the required delta_m is degenerate with.
  - The axis is configurable; default is the reproduced Secrest/CatWISE dipole axis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def unitvec_from_lb(l_deg: float, b_deg: float) -> np.ndarray:
    return lb_to_unitvec(np.array([l_deg], dtype=float), np.array([b_deg], dtype=float))[0]


def vec_to_lb(vec: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size != 3:
        raise ValueError("expected 3-vector")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def dipole_vector(unit: np.ndarray, sel: np.ndarray) -> Tuple[float, float, float]:
    sel = np.asarray(sel, dtype=bool)
    n = int(sel.sum())
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    dvec = 3.0 * unit[sel].sum(axis=0) / float(n)
    amp = float(np.linalg.norm(dvec))
    l, b = vec_to_lb(dvec)
    return amp, l, b


def load_axis_from_secrest_json(path: str) -> Tuple[float, float]:
    with open(path, "r") as f:
        d = json.load(f)
    l = float(d["dipole"]["l_deg"])
    b = float(d["dipole"]["b_deg"])
    return l, b


def count_match_delta_m(w1_fore: np.ndarray, w1_aft: np.ndarray, w1_max: float) -> Dict[str, float]:
    """
    Estimate delta_m by matching hemisphere counts at a given w1_max.

    For example, delta_fore_to_match_aft is the shift such that:
      N_fore(w1 <= w1_max + delta) = N_aft(w1 <= w1_max)
    """

    w1_fore = np.asarray(w1_fore, dtype=float)
    w1_aft = np.asarray(w1_aft, dtype=float)
    w1_fore = w1_fore[np.isfinite(w1_fore)]
    w1_aft = w1_aft[np.isfinite(w1_aft)]

    fore_sorted = np.sort(w1_fore)
    aft_sorted = np.sort(w1_aft)

    n_fore0 = int(np.searchsorted(fore_sorted, w1_max, side="right"))
    n_aft0 = int(np.searchsorted(aft_sorted, w1_max, side="right"))

    def q_at_count(arr_sorted: np.ndarray, k: int) -> float:
        if k <= 0:
            return float(arr_sorted[0])
        if k >= len(arr_sorted):
            return float(arr_sorted[-1])
        return float(arr_sorted[k - 1])

    # Shift fore cut so that fore count equals aft count at w1_max.
    m_fore_eq = q_at_count(fore_sorted, n_aft0)
    delta_fore_to_match_aft = m_fore_eq - float(w1_max)

    # Shift aft cut so that aft count equals fore count at w1_max.
    m_aft_eq = q_at_count(aft_sorted, n_fore0)
    delta_aft_to_match_fore = m_aft_eq - float(w1_max)

    return {
        "w1_max": float(w1_max),
        "N_fore_at_w1_max": float(n_fore0),
        "N_aft_at_w1_max": float(n_aft0),
        "delta_fore_to_match_aft_mag": float(delta_fore_to_match_aft),
        "delta_aft_to_match_fore_mag": float(delta_aft_to_match_fore),
        "delta_sym_avg_mag": float(0.5 * (delta_fore_to_match_aft - delta_aft_to_match_fore)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        required=True,
        help="Secrest/CatWISE FITS (expects l,b,w1,w1cov,ebv,elon,elat).",
    )
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-max", type=float, default=16.4, help="Default faint cut for baseline.")
    ap.add_argument(
        "--w1-max-for-cdf",
        type=float,
        default=17.0,
        help="Upper limit when constructing magnitude distributions for the count-match diagnostic.",
    )

    ap.add_argument("--axis-from", choices=["secrest_json", "custom"], default="secrest_json")
    ap.add_argument(
        "--secrest-json",
        default="REPORTS/Q_D_RES/secrest_reproduction_dipole.json",
        help="Dipole JSON containing axis (used when --axis-from=secrest_json).",
    )
    ap.add_argument("--axis-l", type=float, default=None)
    ap.add_argument("--axis-b", type=float, default=None)

    ap.add_argument(
        "--w1max-grid",
        default="16.0,16.8,0.05",
        help="Grid spec 'start,stop,step' for dipole vs W1_max sweep.",
    )
    ap.add_argument(
        "--w1cov-grid",
        default="50,200,10",
        help="Grid spec 'start,stop,step' for dipole vs w1cov_min sweep (baseline w1_max is used).",
    )

    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/quasar_magshift_mechanisms_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.axis_from == "custom":
        if args.axis_l is None or args.axis_b is None:
            raise SystemExit("--axis-l/--axis-b required for --axis-from=custom")
        axis_l, axis_b = float(args.axis_l), float(args.axis_b)
    else:
        axis_l, axis_b = load_axis_from_secrest_json(args.secrest_json)

    # Import heavy deps late so --help is fast.
    from astropy.table import Table

    tab = Table.read(args.catalog, memmap=True)

    l = np.asarray(tab["l"], dtype=float)
    b = np.asarray(tab["b"], dtype=float)
    w1 = np.asarray(tab["w1"], dtype=float)
    w1cov = np.asarray(tab["w1cov"], dtype=float)
    ebv = np.asarray(tab["ebv"], dtype=float)
    elon = np.asarray(tab["elon"], dtype=float)
    elat = np.asarray(tab["elat"], dtype=float)

    base = (
        np.isfinite(l)
        & np.isfinite(b)
        & np.isfinite(w1)
        & np.isfinite(w1cov)
        & (np.abs(b) > float(args.b_cut))
        & (w1cov >= float(args.w1cov_min))
    )

    unit = lb_to_unitvec(l[base], b[base])
    axis_unit = unitvec_from_lb(axis_l, axis_b)
    cos_axis = unit @ axis_unit

    w1_base = w1[base]
    w1cov_base = w1cov[base]
    ebv_base = ebv[base]
    elon_base = elon[base]
    elat_base = elat[base]

    # 1) Baseline dipole (vector sum).
    sel_baseline = w1_base <= float(args.w1_max)
    D0, l0, b0 = dipole_vector(unit, sel_baseline)

    # 2) Count-match delta_m diagnostic using distributions up to w1_max_for_cdf.
    cdf_mask = w1_base <= float(args.w1_max_for_cdf)
    fore = cdf_mask & (cos_axis > 0)
    aft = cdf_mask & (cos_axis < 0)

    delta_match = count_match_delta_m(w1_base[fore], w1_base[aft], float(args.w1_max))

    hemi_stats = {
        "N_fore_total": int(fore.sum()),
        "N_aft_total": int(aft.sum()),
        "w1cov_fore_mean": float(np.mean(w1cov_base[fore])) if fore.any() else float("nan"),
        "w1cov_aft_mean": float(np.mean(w1cov_base[aft])) if aft.any() else float("nan"),
        "w1cov_fore_median": float(np.median(w1cov_base[fore])) if fore.any() else float("nan"),
        "w1cov_aft_median": float(np.median(w1cov_base[aft])) if aft.any() else float("nan"),
        "ebv_fore_mean": float(np.mean(ebv_base[fore])) if fore.any() else float("nan"),
        "ebv_aft_mean": float(np.mean(ebv_base[aft])) if aft.any() else float("nan"),
        "abs_elat_fore_mean": float(np.mean(np.abs(elat_base[fore]))) if fore.any() else float("nan"),
        "abs_elat_aft_mean": float(np.mean(np.abs(elat_base[aft]))) if aft.any() else float("nan"),
    }

    # Is there an actual W1 zero-point shift? (If delta_m were a true photometric bias,
    # we might see a mean W1 difference at fixed magnitude ranges.)
    mean_w1_bins = [(15.0, 15.5), (15.8, 16.0), (16.2, 16.35), (16.35, 16.4), (16.4, 16.55)]
    w1_mean_by_bin: List[Dict[str, float]] = []
    hemi_all_fore = (cos_axis > 0)
    hemi_all_aft = (cos_axis < 0)
    for lo, hi in mean_w1_bins:
        sel_bin = (w1_base >= float(lo)) & (w1_base < float(hi))
        sel_f = sel_bin & hemi_all_fore
        sel_a = sel_bin & hemi_all_aft
        w1_mean_by_bin.append(
            {
                "w1_lo": float(lo),
                "w1_hi": float(hi),
                "N_fore": int(sel_f.sum()),
                "N_aft": int(sel_a.sum()),
                "mean_w1_fore": float(np.mean(w1_base[sel_f])) if sel_f.any() else float("nan"),
                "mean_w1_aft": float(np.mean(w1_base[sel_a])) if sel_a.any() else float("nan"),
                "mean_diff_fore_minus_aft": (
                    float(np.mean(w1_base[sel_f]) - np.mean(w1_base[sel_a])) if (sel_f.any() and sel_a.any()) else float("nan")
                ),
            }
        )

    # 3) Sensitivity sweeps.
    def parse_grid(spec: str) -> np.ndarray:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"bad grid spec {spec!r}, expected 'start,stop,step'")
        start, stop, step = map(float, parts)
        # Inclusive stop (within float rounding).
        n = int(math.floor((stop - start) / step + 1e-9)) + 1
        return start + step * np.arange(n)

    w1max_grid = parse_grid(args.w1max_grid)
    w1cov_grid = parse_grid(args.w1cov_grid)

    sweep_w1max: List[Dict[str, float]] = []
    for w1m in w1max_grid:
        sel = w1_base <= float(w1m)
        D, ll, bb = dipole_vector(unit, sel)
        sweep_w1max.append({"w1_max": float(w1m), "N": int(sel.sum()), "D": float(D), "l_deg": float(ll), "b_deg": float(bb)})

    sweep_w1cov: List[Dict[str, float]] = []
    for covm in w1cov_grid:
        sel_cov = (w1_base <= float(args.w1_max)) & (w1cov_base >= float(covm))
        D, ll, bb = dipole_vector(unit, sel_cov)
        sweep_w1cov.append(
            {"w1cov_min": float(covm), "N": int(sel_cov.sum()), "D": float(D), "l_deg": float(ll), "b_deg": float(bb)}
        )

    results = {
        "inputs": {
            "catalog": str(args.catalog),
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_max": float(args.w1_max),
            "w1_max_for_cdf": float(args.w1_max_for_cdf),
            "axis_from": args.axis_from,
            "axis_l_deg": float(axis_l),
            "axis_b_deg": float(axis_b),
            "secrest_json": str(args.secrest_json),
            "w1max_grid": args.w1max_grid,
            "w1cov_grid": args.w1cov_grid,
        },
        "baseline_dipole": {"N": int(sel_baseline.sum()), "D": float(D0), "l_deg": float(l0), "b_deg": float(b0)},
        "count_match": delta_match,
        "hemisphere_stats": hemi_stats,
        "w1_mean_by_bin": w1_mean_by_bin,
        "sweeps": {"dipole_vs_w1max": sweep_w1max, "dipole_vs_w1covmin": sweep_w1cov},
        "notes": (
            "count_match.delta_sym_avg_mag is a simple empirical estimate of a hemisphere-to-hemisphere "
            "faint-cut shift that would equalize cumulative counts at w1_max. It is NOT a maximum-likelihood "
            "calibration solution. Sweeps show how sensitive the catalog dipole is to selection thresholds."
        ),
    }

    with open(outdir / "mechanism_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    if args.make_plots:
        import matplotlib.pyplot as plt

        # Plot 1: Fore vs Aft CDF around faint end.
        bins = np.linspace(min(15.5, float(args.w1_max) - 0.8), float(args.w1_max_for_cdf), 120)
        h_fore, _ = np.histogram(w1_base[fore], bins=bins)
        h_aft, _ = np.histogram(w1_base[aft], bins=bins)
        c_fore = np.cumsum(h_fore) / max(1, int(np.sum(h_fore)))
        c_aft = np.cumsum(h_aft) / max(1, int(np.sum(h_aft)))
        x = 0.5 * (bins[:-1] + bins[1:])

        plt.figure(figsize=(8.5, 4.5), dpi=200)
        plt.plot(x, c_fore, label="Fore (cos>0)", lw=2)
        plt.plot(x, c_aft, label="Aft (cos<0)", lw=2)
        plt.axvline(float(args.w1_max), color="k", ls="--", alpha=0.7, label=f"W1_max={args.w1_max:.2f}")
        dm = float(delta_match["delta_sym_avg_mag"])
        plt.axvline(float(args.w1_max) + dm, color="k", ls=":", alpha=0.7, label=f"W1_max+delta_sym ({dm:+.4f} mag)")
        plt.xlabel("W1")
        plt.ylabel("CDF")
        plt.title("Hemispheric W1 CDFs about the Secrest axis (count-match diagnostic)")
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / "hemisphere_w1_cdf.png")
        plt.close()

        # Plot 1b: Fore/Aft differential signal vs W1 (where does the asymmetry live?).
        dens_fore = h_fore / max(1, int(np.sum(h_fore)))
        dens_aft = h_aft / max(1, int(np.sum(h_aft)))
        ratio = dens_fore / np.clip(dens_aft, 1e-12, np.inf)

        plt.figure(figsize=(8.5, 4.5), dpi=200)
        plt.plot(x, ratio, lw=2)
        plt.axhline(1.0, color="k", ls="--", alpha=0.7)
        plt.axvline(float(args.w1_max), color="k", ls=":", alpha=0.7, label=f"W1_max={args.w1_max:.2f}")
        plt.ylim(0.8, 1.2)
        plt.xlabel("W1")
        plt.ylabel("Fore/Aft (normalized bin density ratio)")
        plt.title("Where does the hemispheric asymmetry appear in W1? (about Secrest axis)")
        plt.tight_layout()
        plt.savefig(outdir / "hemisphere_w1_ratio.png")
        plt.close()

        # Plot 2: Dipole amplitude vs W1_max.
        plt.figure(figsize=(7.0, 4.0), dpi=200)
        plt.plot([r["w1_max"] for r in sweep_w1max], [r["D"] for r in sweep_w1max], marker="o", lw=2)
        plt.axvline(float(args.w1_max), color="k", ls="--", alpha=0.7)
        plt.xlabel("W1_max (faint cut)")
        plt.ylabel("Dipole amplitude D")
        plt.title("Dipole amplitude sensitivity to faint cut")
        plt.tight_layout()
        plt.savefig(outdir / "dipole_vs_w1max.png")
        plt.close()

        # Plot 3: Dipole amplitude vs w1cov_min.
        plt.figure(figsize=(7.0, 4.0), dpi=200)
        plt.plot([r["w1cov_min"] for r in sweep_w1cov], [r["D"] for r in sweep_w1cov], marker="o", lw=2)
        plt.axvline(float(args.w1cov_min), color="k", ls="--", alpha=0.7)
        plt.xlabel("w1cov_min")
        plt.ylabel("Dipole amplitude D")
        plt.title("Dipole amplitude sensitivity to WISE coverage threshold")
        plt.tight_layout()
        plt.savefig(outdir / "dipole_vs_w1covmin.png")
        plt.close()

        # Plot 4: w1cov distributions by hemisphere.
        plt.figure(figsize=(7.0, 4.0), dpi=200)
        cov_bins = np.linspace(np.percentile(w1cov_base[cdf_mask], 1), np.percentile(w1cov_base[cdf_mask], 99), 60)
        plt.hist(w1cov_base[fore], bins=cov_bins, histtype="step", lw=2, density=True, label="Fore")
        plt.hist(w1cov_base[aft], bins=cov_bins, histtype="step", lw=2, density=True, label="Aft")
        plt.xlabel("w1cov")
        plt.ylabel("density")
        plt.title("w1cov distribution by hemisphere about axis")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "w1cov_hemisphere_hist.png")
        plt.close()

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
