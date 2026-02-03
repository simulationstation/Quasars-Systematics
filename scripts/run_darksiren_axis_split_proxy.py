#!/usr/bin/env python3
"""
Fast proxy test linking a *chosen sky axis* (e.g., CMB dipole or a quasar-derived axis)
to the GWTC-3 dark-siren score reported in EntropyPaper (run bundle: 2-1-c-m).

This script does **not** refit mu(A). It asks a simpler diagnostic question:

  "Is the per-event HE-vs-GR log predictive density difference (ΔLPD) concentrated
   preferentially in one hemisphere about a chosen axis?"

Method:
  - Load per-event score table (event_scores_*.json) from a known production run.
  - For each event, read the public GWTC-3 "multi-order" sky map FITS (UNIQ + PROBDENSITY).
  - Compute P(head) = ∫_{axis·n>0} p(n) dΩ using pixel-center sign as a proxy.
  - Split the total score with these probabilities:
      ΔLPD_head = Σ ΔLPD_i * P_i(head)
      ΔLPD_tail = Σ ΔLPD_i * (1 - P_i(head))
      Δ(ΔLPD)   = ΔLPD_head - ΔLPD_tail = Σ ΔLPD_i * (2P_i(head) - 1)
  - Estimate a null distribution with a permutation test (shuffle ΔLPD across events),
    holding P(head) fixed.

Outputs:
  - JSON summary (per-event P(head) + totals + permutation p-value)
  - Optional PNG diagnostic plot (scatter + null histogram)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


@dataclass(frozen=True)
class AxisSpec:
    name: str
    frame: str
    lon_deg: float
    lat_deg: float


def axis_spec_from_args(args: argparse.Namespace) -> AxisSpec:
    preset = (args.axis or "").strip().lower()
    if preset:
        if preset == "cmb":
            return AxisSpec(name="cmb", frame="galactic", lon_deg=264.021, lat_deg=48.253)
        if preset == "secrest":
            # Secrest+22 CatWISE dipole direction often quoted around (l,b)~(238.2, 28.8).
            # Use the repo's baseline reproduction axis from Q_D_RES/dipole_master_tests.md.
            return AxisSpec(name="secrest", frame="galactic", lon_deg=236.01, lat_deg=28.77)
        if preset in {"ecliptic_north", "ecl_north"}:
            return AxisSpec(name="ecliptic_north", frame="barycentricmeanecliptic", lon_deg=0.0, lat_deg=90.0)
        raise ValueError(f"Unknown --axis preset: {args.axis!r}")

    if args.axis_lon_deg is None or args.axis_lat_deg is None:
        raise ValueError("Provide either --axis <preset> or both --axis-lon-deg and --axis-lat-deg")
    frame = (args.axis_frame or "galactic").strip().lower()
    if frame not in {"galactic", "icrs", "barycentricmeanecliptic"}:
        raise ValueError("--axis-frame must be one of: galactic, icrs, barycentricmeanecliptic")
    return AxisSpec(
        name=args.axis_name or "custom",
        frame=frame,
        lon_deg=float(args.axis_lon_deg),
        lat_deg=float(args.axis_lat_deg),
    )


def axis_unitvec_icrs(axis: AxisSpec) -> np.ndarray:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    lon = float(axis.lon_deg)
    lat = float(axis.lat_deg)
    frame = axis.frame.lower()

    if frame == "galactic":
        sc = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame="galactic").icrs
    elif frame == "barycentricmeanecliptic":
        sc = SkyCoord(lon * u.deg, lat * u.deg, frame="barycentricmeanecliptic").icrs
    elif frame == "icrs":
        sc = SkyCoord(ra=lon * u.deg, dec=lat * u.deg, frame="icrs")
    else:  # pragma: no cover
        raise ValueError(f"unsupported axis frame: {axis.frame!r}")

    cart = sc.cartesian
    v = np.array([cart.x.value, cart.y.value, cart.z.value], dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("axis unit vector is invalid")
    return v / n


def uniq_to_order_ipix(uniq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode IVOA HEALPix UNIQ values into (order, ipix_nest)."""
    u = np.asarray(uniq, dtype=np.int64)
    if u.ndim != 1:
        u = u.reshape(-1)
    if np.any(u <= 0):
        raise ValueError("UNIQ contains non-positive entries")
    # For HEALPix UNIQ: uniq = 4 * 4^order + ipix (NESTED), with ipix in [0, 12*4^order).
    log2u = np.floor(np.log2(u)).astype(np.int64)
    order = ((log2u - 2) // 2).astype(np.int64)
    base = (np.int64(1) << (2 * order + 2)).astype(np.int64)
    ipix = u - base
    if np.any(ipix < 0):
        raise ValueError("UNIQ decode produced negative ipix (unexpected)")
    return order, ipix


def resolve_skymap_path(path_str: str, *, skymap_dir: Path) -> Path:
    p = Path(path_str)
    # 1) absolute path as-is
    if p.is_absolute() and p.exists():
        return p
    # 2) direct join (in case caller preserves directory structure)
    cand = skymap_dir / p
    if cand.exists():
        return cand
    # 3) by basename (default expected layout in this repo)
    cand = skymap_dir / p.name
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Cannot resolve skymap path {path_str!r} under skymap_dir={str(skymap_dir)!r}")


def compute_p_head_for_skymap(*, skymap_fits: Path, axis_icrs: np.ndarray) -> tuple[float, float]:
    """Return (p_head, total_prob_mass)."""
    from astropy.io import fits
    import healpy as hp

    with fits.open(skymap_fits) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"{skymap_fits} does not look like a GWTC skymap FITS (needs table HDU)")
        tab = hdul[1].data
        uniq = np.asarray(tab["UNIQ"], dtype=np.int64)
        probdensity = np.asarray(tab["PROBDENSITY"], dtype=float)

    order, ipix = uniq_to_order_ipix(uniq)

    total = 0.0
    head = 0.0
    for ord_val in np.unique(order):
        mask = order == ord_val
        nside = int(1 << int(ord_val))
        x, y, z = hp.pix2vec(nside, ipix[mask], nest=True)
        area = float(hp.nside2pixarea(nside))
        prob = probdensity[mask] * area
        dots = axis_icrs[0] * x + axis_icrs[1] * y + axis_icrs[2] * z
        head += float(np.sum(prob[dots > 0.0]))
        total += float(np.sum(prob))

    if total <= 0.0 or not np.isfinite(total):
        raise ValueError(f"total probability mass for {skymap_fits} is invalid: {total}")
    p_head = head / total
    p_head = float(np.clip(p_head, 0.0, 1.0))
    return p_head, total


def permutation_p_value(weights: np.ndarray, values: np.ndarray, *, n_perm: int, seed: int) -> dict[str, Any]:
    w = np.asarray(weights, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    if w.shape != v.shape:
        raise ValueError("weights and values must have same shape")
    rng = np.random.default_rng(int(seed))

    obs = float(np.dot(v, w))
    perm = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        perm[i] = float(np.dot(rng.permutation(v), w))

    abs_obs = abs(obs)
    p_two_sided = float((np.sum(np.abs(perm) >= abs_obs) + 1.0) / (len(perm) + 1.0))
    sd = float(np.std(perm)) if len(perm) else float("nan")
    z = float(obs / sd) if sd > 0 else float("nan")
    return {
        "obs": obs,
        "null_mean": float(np.mean(perm)) if len(perm) else float("nan"),
        "null_sd": sd,
        "zscore": z,
        "p_two_sided": p_two_sided,
        "n_perm": int(n_perm),
        "seed": int(seed),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--event-scores-json",
        type=str,
        default="data/dark_sirens/2-1-c-m/production_36events/event_scores_M0_start101.json",
        help="Per-event ΔLPD table (list of dicts).",
    )
    ap.add_argument(
        "--skymap-dir",
        type=str,
        default="data/external/zenodo_5546663/skymaps",
        help="Directory containing the GWTC-3 skymap FITS files (basenames).",
    )
    ap.add_argument("--axis", type=str, default="cmb", help="Axis preset: cmb, secrest, ecliptic_north.")
    ap.add_argument("--axis-name", type=str, default=None, help="Name for a custom axis.")
    ap.add_argument("--axis-frame", type=str, default="galactic", help="Frame for custom axis.")
    ap.add_argument("--axis-lon-deg", type=float, default=None, help="Custom axis longitude (l/ra/lon).")
    ap.add_argument("--axis-lat-deg", type=float, default=None, help="Custom axis latitude (b/dec/lat).")
    ap.add_argument("--n-perm", type=int, default=2000, help="Permutation count for null p-value.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for permutations.")
    ap.add_argument("--make-plot", action="store_true", help="Write a diagnostic PNG.")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: outputs/darksiren_axis_proxy_*)")
    args = ap.parse_args()

    event_scores_path = Path(args.event_scores_json)
    skymap_dir = Path(args.skymap_dir)
    if not event_scores_path.exists():
        raise FileNotFoundError(f"--event-scores-json not found: {event_scores_path}")
    if not skymap_dir.exists():
        raise FileNotFoundError(f"--skymap-dir not found: {skymap_dir}")

    outdir = Path(args.outdir) if args.outdir else Path("outputs") / f"darksiren_axis_proxy_{utc_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)

    axis = axis_spec_from_args(args)
    axis_icrs = axis_unitvec_icrs(axis)

    with event_scores_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError("event_scores JSON must be a non-empty list")

    events_out: list[dict[str, Any]] = []
    p_head_list: list[float] = []
    delta_list: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("event_scores entries must be dicts")
        event = str(row.get("event", ""))
        skymap_path_str = str(row.get("skymap_path", ""))
        if not event or not skymap_path_str:
            raise ValueError("event_scores entry missing event/skymap_path")
        skymap_path = resolve_skymap_path(skymap_path_str, skymap_dir=skymap_dir)
        p_head, prob_total = compute_p_head_for_skymap(skymap_fits=skymap_path, axis_icrs=axis_icrs)

        delta = float(row.get("delta_lpd", float("nan")))
        if not np.isfinite(delta):
            raise ValueError(f"delta_lpd non-finite for {event}: {delta}")

        out_row = dict(row)
        out_row.update(
            {
                "skymap_resolved": str(skymap_path),
                "p_head": p_head,
                "p_tail": 1.0 - p_head,
                "prob_total": float(prob_total),
            }
        )
        events_out.append(out_row)
        p_head_list.append(p_head)
        delta_list.append(delta)

    p_head_arr = np.asarray(p_head_list, dtype=float)
    delta_arr = np.asarray(delta_list, dtype=float)
    weights = 2.0 * p_head_arr - 1.0

    d_total = float(np.sum(delta_arr))
    d_head = float(np.sum(delta_arr * p_head_arr))
    d_tail = float(np.sum(delta_arr * (1.0 - p_head_arr)))
    d_diff = float(d_head - d_tail)

    perm = permutation_p_value(weights, delta_arr, n_perm=int(args.n_perm), seed=int(args.seed))

    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "event_scores_json": str(event_scores_path),
            "skymap_dir": str(skymap_dir),
            "axis": {"name": axis.name, "frame": axis.frame, "lon_deg": axis.lon_deg, "lat_deg": axis.lat_deg},
            "n_events": int(len(events_out)),
        },
        "totals": {
            "delta_lpd_total": d_total,
            "delta_lpd_head": d_head,
            "delta_lpd_tail": d_tail,
            "delta_lpd_head_minus_tail": d_diff,
        },
        "permutation_test": perm,
        "events": events_out,
    }

    out_json = outdir / "axis_split_proxy.json"
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    if bool(args.make_plot):
        import matplotlib.pyplot as plt

        # Figure: (scatter) ΔLPD vs P(head) + (hist) null distribution of head-tail difference.
        fig = plt.figure(figsize=(9.5, 4.0))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.scatter(p_head_arr, delta_arr, s=18, alpha=0.85)
        ax0.axvline(0.5, color="k", lw=1, alpha=0.4)
        ax0.set_xlabel("P(event in head hemisphere)")
        ax0.set_ylabel("Per-event ΔLPD (HE − GR)")
        ax0.set_title(f"Axis: {axis.name} ({axis.frame})")

        # Annotate top-|ΔLPD| events (keep it uncluttered).
        k = min(6, len(delta_arr))
        idx = np.argsort(-np.abs(delta_arr))[:k]
        for i in idx:
            ax0.annotate(str(events_out[i]["event"]), (p_head_arr[i], delta_arr[i]), fontsize=7, alpha=0.85)

        ax1 = fig.add_subplot(gs[0, 1])
        rng = np.random.default_rng(int(args.seed))
        perm_vals = np.array([float(np.dot(rng.permutation(delta_arr), weights)) for _ in range(int(args.n_perm))])
        ax1.hist(perm_vals, bins=30, alpha=0.85, color="#4C72B0")
        ax1.axvline(d_diff, color="crimson", lw=2, label="observed")
        ax1.set_xlabel("ΔLPD(head) − ΔLPD(tail)")
        ax1.set_ylabel("count")
        ax1.set_title(f"Permutation null (p={perm['p_two_sided']:.3f})")
        ax1.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        fig.savefig(outdir / "axis_split_proxy.png", dpi=180)
        plt.close(fig)

    summary_txt = (
        f"axis={axis.name} frame={axis.frame} lon={axis.lon_deg:.3f} lat={axis.lat_deg:.3f}\n"
        f"n_events={len(events_out)}\n"
        f"ΔLPD_total={d_total:.6f}\n"
        f"ΔLPD_head={d_head:.6f}\n"
        f"ΔLPD_tail={d_tail:.6f}\n"
        f"ΔLPD_head_minus_tail={d_diff:.6f}\n"
        f"perm_p_two_sided={perm['p_two_sided']:.6f} (n_perm={perm['n_perm']})\n"
    )
    (outdir / "axis_split_proxy.txt").write_text(summary_txt, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
