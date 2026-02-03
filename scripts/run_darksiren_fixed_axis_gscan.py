#!/usr/bin/env python3
"""
Fixed-axis anisotropy scan for the EntropyPaper dark-siren score.

We test a 1-parameter directional modulation of the HE (mu-model) GW distance prediction:

  dL_gw(z, n) -> dL_gw(z) * exp(g * cos(theta)),

where theta is the angle to a chosen fixed sky axis.

This is a *targeted diagnostic*:
  - Uses the cached PE histogram + galaxy lists from an existing production run
    (so we do not rerun the expensive sky/cat extraction).
  - Computes a cat-only score (in-catalog host term). It does not (yet) recompute
    missing-host mixture terms.

Outputs:
  - JSON summary with ΔLPD(g) curve + quadratic fit near the peak
  - PNG plot of ΔLPD(g) vs g
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def logmeanexp_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return float("-inf")
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


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


def parse_g_grid(spec: str) -> np.ndarray:
    s = str(spec).strip()
    if not s:
        raise ValueError("--g-grid is empty")

    # Allow "gmin,gmax,step"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 3:
        gmin, gmax, step = (float(x) for x in parts)
        if step <= 0:
            raise ValueError("g step must be >0")
        n = int(np.floor((gmax - gmin) / step + 0.5)) + 1
        grid = gmin + step * np.arange(n, dtype=float)
        grid = grid[(grid >= gmin - 1e-12) & (grid <= gmax + 1e-12)]
        return np.asarray(grid, dtype=float)

    # Otherwise interpret as a list of values "g1,g2,g3"
    grid = np.array([float(x) for x in parts], dtype=float)
    if grid.size < 1:
        raise ValueError("Need at least one g value")
    return grid


def downsample_posterior(post, *, draw_idx: list[int]):
    from entropy_horizon_recon.sirens import MuForwardPosterior

    idx = np.asarray(draw_idx, dtype=int)

    def _sel(a):
        a = np.asarray(a)
        return a[idx]

    return MuForwardPosterior(
        x_grid=np.asarray(post.x_grid, dtype=float),
        logmu_x_samples=_sel(post.logmu_x_samples),
        z_grid=np.asarray(post.z_grid, dtype=float),
        H_samples=_sel(post.H_samples),
        H0=_sel(post.H0),
        omega_m0=_sel(post.omega_m0),
        omega_k0=_sel(post.omega_k0),
        sigma8_0=_sel(post.sigma8_0) if getattr(post, "sigma8_0", None) is not None else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=str, default="cmb", help="Axis preset: cmb|secrest|ecliptic_north")
    ap.add_argument("--axis-name", type=str, default=None)
    ap.add_argument("--axis-frame", type=str, default="galactic")
    ap.add_argument("--axis-lon-deg", type=float, default=None)
    ap.add_argument("--axis-lat-deg", type=float, default=None)
    ap.add_argument("--g-grid", type=str, default="-0.02,0.02,0.01", help="Either gmin,gmax,step or g1,g2,...")

    ap.add_argument(
        "--event-scores-json",
        type=str,
        default="data/dark_sirens/2-1-c-m/production_36events/event_scores_M0_start101.json",
    )
    ap.add_argument(
        "--summary-json",
        type=str,
        default="data/dark_sirens/2-1-c-m/production_36events/summary_M0_start101.json",
    )
    ap.add_argument(
        "--posterior-run-dir",
        type=str,
        default="data/entropy_posteriors/M0_start101",
        help="Directory containing samples/mu_forward_posterior.npz",
    )
    ap.add_argument(
        "--cache-outdir",
        type=str,
        default="outputs/dark_siren_gap_pe_scaleup36max_20260201_155611UTC",
        help="Existing dark-siren production output dir that contains cache/ (event_*.npz) and cache_terms/ (cat_*.npz).",
    )
    ap.add_argument("--run-label", type=str, default="M0_start101", help="Run label used in cache_terms filenames.")
    ap.add_argument("--galaxy-chunk-size", type=int, default=50_000)
    ap.add_argument("--max-draws", type=int, default=0, help="If >0, use only the first N posterior draws (fast mode).")
    ap.add_argument("--max-events", type=int, default=0, help="0 means all events.")
    ap.add_argument("--sanity-check-event", type=str, default="", help="If set, compare g=0 cat arrays to cache_terms and exit.")

    ap.add_argument("--outdir", type=str, default="", help="Output directory (default: outputs/darksiren_fixed_axis_gscan_<axis>_<tag>)")
    ap.add_argument("--make-plot", action="store_true", help="Write a PNG plot to outdir.")
    args = ap.parse_args()

    axis = axis_spec_from_args(args)
    axis_icrs = axis_unitvec_icrs(axis)
    g_grid = parse_g_grid(args.g_grid)
    g_grid = np.unique(np.asarray(g_grid, dtype=float))

    base_out = Path(args.cache_outdir).expanduser().resolve()
    cache_dir = base_out / "cache"
    cache_terms_dir = base_out / "cache_terms"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing cache dir: {cache_dir}")
    if not cache_terms_dir.exists():
        raise FileNotFoundError(f"Missing cache_terms dir: {cache_terms_dir}")

    rows = json.loads(Path(args.event_scores_json).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("event-scores-json must be a non-empty list")
    # Sort by n_gal so early progress prints quickly (commutative sum).
    rows = sorted(rows, key=lambda r: int(r.get("n_gal", 0)))
    events = [str(r["event"]) for r in rows]
    if int(args.max_events) > 0:
        events = events[: int(args.max_events)]

    summ = json.loads(Path(args.summary_json).read_text())
    draw_idx = [int(i) for i in summ["draw_idx"]]

    from entropy_horizon_recon.dark_sirens_pe import PePixelDistanceHistogram, compute_dark_siren_logL_draws_from_pe_hist
    from entropy_horizon_recon.gw_distance_priors import GWDistancePrior
    from entropy_horizon_recon.sirens import MuForwardPosterior, load_mu_forward_posterior

    post_full = load_mu_forward_posterior(Path(args.posterior_run_dir))
    post = downsample_posterior(post_full, draw_idx=draw_idx)
    if int(args.max_draws) > 0:
        keep = int(args.max_draws)
        if keep <= 0:
            raise ValueError("--max-draws must be positive when provided.")
        if keep < int(post.H_samples.shape[0]):
            post = MuForwardPosterior(
                x_grid=post.x_grid,
                logmu_x_samples=post.logmu_x_samples[:keep],
                z_grid=post.z_grid,
                H_samples=post.H_samples[:keep],
                H0=post.H0[:keep],
                omega_m0=post.omega_m0[:keep],
                omega_k0=post.omega_k0[:keep],
                sigma8_0=post.sigma8_0[:keep] if post.sigma8_0 is not None else None,
            )
    n_draws = int(post.H_samples.shape[0])

    # Default GW distance prior matches the production manifest (auto -> powerlaw k=2).
    gw_prior = GWDistancePrior(mode="dL_powerlaw", powerlaw_k=2.0)

    # Precompute cos(theta) for pixel centers at the PE nside (assumed common across events).
    import healpy as hp

    # Try to infer pe_nside from one cached event.
    with np.load(cache_dir / f"event_{events[0]}.npz", allow_pickle=True) as d0:
        meta0 = json.loads(str(d0["meta"].tolist()))
        pe_nside = int(meta0.get("pe_nside", 64))
    npix = int(hp.nside2npix(pe_nside))
    x, y, z = hp.pix2vec(pe_nside, np.arange(npix, dtype=np.int64), nest=True)
    cos_theta_pix = axis_icrs[0] * x + axis_icrs[1] * y + axis_icrs[2] * z
    cos_theta_pix = np.asarray(cos_theta_pix, dtype=np.float32)

    # Sanity check mode: compare g=0 arrays for one event against cache_terms.
    if str(args.sanity_check_event).strip():
        ev = str(args.sanity_check_event).strip()
        ev_path = cache_dir / f"event_{ev}.npz"
        if not ev_path.exists():
            raise FileNotFoundError(f"Missing event cache: {ev_path}")
        cat_path = cache_terms_dir / f"cat_{ev}__{str(args.run_label)}.npz"
        if not cat_path.exists():
            raise FileNotFoundError(f"Missing cat cache_terms: {cat_path}")
        with np.load(ev_path, allow_pickle=True) as d:
            z_gal = np.asarray(d["z"], dtype=float)
            w_gal = np.asarray(d["w"], dtype=float)
            ipix_gal = np.asarray(d["ipix"], dtype=np.int64)
            pe = PePixelDistanceHistogram(
                nside=pe_nside,
                nest=True,
                p_credible=float(meta0.get("p_credible", 0.9)),
                pix_sel=np.asarray(d["pe_pix_sel"], dtype=np.int64),
                prob_pix=np.asarray(d["pe_prob_pix"], dtype=float),
                dL_edges=np.asarray(d["pe_dL_edges"], dtype=float),
                pdf_bins=np.asarray(d["pe_pdf_bins"], dtype=float),
            )
        cos_gal = cos_theta_pix[ipix_gal]
        logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist(
            event=ev,
            pe=pe,
            post=post,
            z_gal=z_gal,
            w_gal=w_gal,
            ipix_gal=ipix_gal,
            convention=str(summ.get("convention", "A")),
            gw_distance_prior=gw_prior,
            distance_mode="full",
            gal_chunk_size=int(args.galaxy_chunk_size),
            g_aniso=0.0,
            cos_theta_gal=cos_gal,
            compute_gr=True,
        )
        with np.load(cat_path, allow_pickle=True) as d:
            ref_mu = np.asarray(d["logL_cat_mu"], dtype=float)[:n_draws]
            ref_gr = np.asarray(d["logL_cat_gr"], dtype=float)[:n_draws]
        print(
            json.dumps(
                {
                    "event": ev,
                    "n_draws": int(n_draws),
                    "max_abs_diff_mu": float(np.max(np.abs(logL_mu - ref_mu))),
                    "max_abs_diff_gr": float(np.max(np.abs(logL_gr - ref_gr))),
                    "median_abs_diff_mu": float(np.median(np.abs(logL_mu - ref_mu))),
                    "median_abs_diff_gr": float(np.median(np.abs(logL_gr - ref_gr))),
                },
                indent=2,
            )
        )
        return 0

    outdir = Path(args.outdir) if str(args.outdir).strip() else Path(f"outputs/darksiren_fixed_axis_gscan_{axis.name}_{utc_tag()}")
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Accumulate per-draw totals.
    mu_totals = {float(g): np.zeros((n_draws,), dtype=float) for g in g_grid.tolist()}
    gr_total = np.zeros((n_draws,), dtype=float)

    per_event: list[dict[str, Any]] = []
    for i, ev in enumerate(events):
        ev_path = cache_dir / f"event_{ev}.npz"
        if not ev_path.exists():
            raise FileNotFoundError(f"Missing event cache: {ev_path}")

        with np.load(ev_path, allow_pickle=True) as d:
            meta = json.loads(str(d["meta"].tolist()))
            z_gal = np.asarray(d["z"], dtype=float)
            w_gal = np.asarray(d["w"], dtype=float)
            ipix_gal = np.asarray(d["ipix"], dtype=np.int64)

            pe = PePixelDistanceHistogram(
                nside=int(meta.get("pe_nside", pe_nside)),
                nest=True,
                p_credible=float(meta.get("p_credible", 0.9)),
                pix_sel=np.asarray(d["pe_pix_sel"], dtype=np.int64),
                prob_pix=np.asarray(d["pe_prob_pix"], dtype=float),
                dL_edges=np.asarray(d["pe_dL_edges"], dtype=float),
                pdf_bins=np.asarray(d["pe_pdf_bins"], dtype=float),
            )

        cos_gal = cos_theta_pix[ipix_gal]

        # Compute GR once (at g=0) and reuse for all g.
        logL_gr_ev = None
        for j, g in enumerate(g_grid.tolist()):
            g = float(g)
            logL_mu_ev, logL_gr_ev_j = compute_dark_siren_logL_draws_from_pe_hist(
                event=str(ev),
                pe=pe,
                post=post,
                z_gal=z_gal,
                w_gal=w_gal,
                ipix_gal=ipix_gal,
                convention=str(summ.get("convention", "A")),
                gw_distance_prior=gw_prior,
                distance_mode="full",
                gal_chunk_size=int(args.galaxy_chunk_size),
                g_aniso=g,
                cos_theta_gal=cos_gal,
                compute_gr=(logL_gr_ev is None),
            )
            mu_totals[g] += logL_mu_ev
            if logL_gr_ev is None:
                logL_gr_ev = np.asarray(logL_gr_ev_j, dtype=float)
                gr_total += logL_gr_ev

        per_event.append(
            {
                "event": str(ev),
                "n_gal": int(meta.get("n_gal", int(z_gal.size))),
                "sky_area_deg2": float(meta.get("sky_area_deg2", float("nan"))),
            }
        )
        if (i + 1) % 1 == 0:
            print(f"[gscan] {i+1}/{len(events)} {ev} n_gal={int(z_gal.size)}", flush=True)

    lpd_gr = logmeanexp_1d(gr_total)
    curve = []
    for g in g_grid.tolist():
        g = float(g)
        lpd_mu = logmeanexp_1d(mu_totals[g])
        curve.append({"g": g, "lpd_mu": lpd_mu, "lpd_gr": lpd_gr, "delta_lpd": lpd_mu - lpd_gr})

    # Quadratic fit: delta_lpd(g) ≈ a g^2 + b g + c (expect a<0 near peak).
    xs = np.array([r["g"] for r in curve], dtype=float)
    ys = np.array([r["delta_lpd"] for r in curve], dtype=float)
    fit = {}
    if xs.size >= 3 and np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
        a, b, c = np.polyfit(xs, ys, deg=2)
        g_hat = float(-b / (2.0 * a)) if a != 0 else float("nan")
        # Approximate 1-sigma from curvature (treating delta_lpd as log-evidence in g with flat prior).
        # For a log-likelihood ~ -(g-g0)^2/(2σ^2), the quadratic coefficient is a = -1/(2σ^2).
        sigma = float(np.sqrt(-1.0 / (2.0 * a))) if a < 0 else float("nan")
        fit = {"a": float(a), "b": float(b), "c": float(c), "g_hat": g_hat, "sigma_g": sigma}

    out = {
        "run": {
            "timestamp_utc": utc_tag(),
            "axis": {"name": axis.name, "frame": axis.frame, "lon_deg": axis.lon_deg, "lat_deg": axis.lat_deg},
            "axis_icrs_unitvec": [float(x) for x in axis_icrs.tolist()],
            "pe_nside": int(pe_nside),
            "n_events": int(len(events)),
            "n_draws": int(n_draws),
            "note": "cat-only fixed-axis anisotropy scan (missing-host + full mixture not recomputed)",
        },
        "g_grid": [float(g) for g in g_grid.tolist()],
        "curve": curve,
        "quad_fit": fit,
        "events": per_event,
    }
    (outdir / "fixed_axis_gscan.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {outdir/'fixed_axis_gscan.json'}")

    if args.make_plot:
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(xs, ys, marker="o")
        plt.axvline(0.0, color="k", alpha=0.2, linewidth=1)
        plt.axhline(float(ys[xs == 0][0]) if np.any(xs == 0) else 0.0, color="k", alpha=0.15, linewidth=1)
        plt.xlabel("g (anisotropy strength)")
        plt.ylabel("ΔLPD(g) = LPD_mu(g) − LPD_gr")
        plt.title(f"Fixed-axis g scan ({axis.name}), cat-only")
        if fit:
            plt.text(
                0.02,
                0.98,
                f"g_hat≈{fit['g_hat']:.4g}, σ_g≈{fit['sigma_g']:.4g}",
                transform=plt.gca().transAxes,
                ha="left",
                va="top",
            )
        plt.tight_layout()
        fig_path = outdir / "fixed_axis_gscan.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()
        print(f"Wrote {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
