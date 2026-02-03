#!/usr/bin/env python3
"""
Fixed-axis anisotropy scan for the EntropyPaper dark-siren score (FULL likelihood pieces).

This extends the earlier cat-only diagnostic by including:
  - Missing-host mixture term (out-of-catalog integral)
  - Global f_miss marginalization (shared nuisance) with the same beta prior metadata as the
    production summary JSON (when available)
  - Selection-normalization alpha(model) from O3 injections, applied at the correct combined-draw level

We test a 1-parameter directional modulation of the HE (mu-model) GW distance prediction:

  dL_gw(z, n) -> dL_gw(z, n) * exp(g * cos(theta)),

where theta is the angle to a chosen fixed sky axis.

Outputs:
  - fixed_axis_gscan_full.json: ΔLPD(g) curve + selection/mixture metadata
  - fixed_axis_gscan_full.png:  ΔLPD(g) vs g
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln


# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Globals for multiprocessing (forked workers inherit these).
_GSCAN_CACHE_DIR: Path | None = None
_GSCAN_POST = None
_GSCAN_PRE = None
_GSCAN_GW_PRIOR = None
_GSCAN_CONVENTION: str | None = None
_GSCAN_G_GRID: list[float] | None = None
_GSCAN_COS_THETA_PIX: np.ndarray | None = None
_GSCAN_PE_NSIDE: int | None = None
_GSCAN_P_CREDIBLE: float | None = None
_GSCAN_GAL_CHUNK_SIZE: int | None = None
_GSCAN_MISSING_PIXEL_CHUNK_SIZE: int | None = None
_GSCAN_DISABLE_MISSING: bool | None = None


def _gscan_full_compute_event(ev: str) -> dict[str, Any]:
    """Compute per-event logL vectors (cat + missing) for all g in the global grid."""
    if (
        _GSCAN_CACHE_DIR is None
        or _GSCAN_POST is None
        or _GSCAN_PRE is None
        or _GSCAN_GW_PRIOR is None
        or _GSCAN_CONVENTION is None
        or _GSCAN_G_GRID is None
        or _GSCAN_COS_THETA_PIX is None
        or _GSCAN_PE_NSIDE is None
        or _GSCAN_P_CREDIBLE is None
        or _GSCAN_GAL_CHUNK_SIZE is None
        or _GSCAN_MISSING_PIXEL_CHUNK_SIZE is None
        or _GSCAN_DISABLE_MISSING is None
    ):
        raise RuntimeError("Worker globals not initialized.")

    import healpy as hp
    from scipy.special import logsumexp

    from entropy_horizon_recon.dark_sirens_incompleteness import compute_missing_host_logL_draws_from_histogram
    from entropy_horizon_recon.dark_sirens_pe import PePixelDistanceHistogram
    from entropy_horizon_recon.sirens import predict_dL_em, predict_r_gw_em

    ev = str(ev)
    ev_path = _GSCAN_CACHE_DIR / f"event_{ev}.npz"
    if not ev_path.exists():
        raise FileNotFoundError(f"Missing event cache: {ev_path}")

    with np.load(ev_path, allow_pickle=True) as d:
        meta = json.loads(str(d["meta"].tolist()))
        z_gal = np.asarray(d["z"], dtype=float)
        w_gal = np.asarray(d["w"], dtype=float)
        ipix_gal = np.asarray(d["ipix"], dtype=np.int64)
        pe = PePixelDistanceHistogram(
            nside=int(meta.get("pe_nside", int(_GSCAN_PE_NSIDE))),
            nest=True,
            p_credible=float(meta.get("p_credible", float(_GSCAN_P_CREDIBLE))),
            pix_sel=np.asarray(d["pe_pix_sel"], dtype=np.int64),
            prob_pix=np.asarray(d["pe_prob_pix"], dtype=float),
            dL_edges=np.asarray(d["pe_dL_edges"], dtype=float),
            pdf_bins=np.asarray(d["pe_pdf_bins"], dtype=float),
        )

    cos_theta_pix = np.asarray(_GSCAN_COS_THETA_PIX, dtype=float)
    cos_gal = cos_theta_pix[ipix_gal]
    cos_pix_sel = cos_theta_pix[np.asarray(pe.pix_sel, dtype=np.int64)]

    convention = str(_GSCAN_CONVENTION)
    g_grid = [float(g) for g in _GSCAN_G_GRID]

    # ------------------------------
    # Cat term: compute μ for all g with a single set of distance predictions per chunk.
    # This avoids recomputing dL_em(z) and R(z) for each g, which is the dominant cost.
    # ------------------------------
    z = np.asarray(z_gal, dtype=float)
    w = np.asarray(w_gal, dtype=float)
    ipix = np.asarray(ipix_gal, dtype=np.int64)
    cos = np.asarray(cos_gal, dtype=float)
    if z.ndim != 1 or w.ndim != 1 or ipix.ndim != 1 or cos.ndim != 1 or not (z.shape == w.shape == ipix.shape == cos.shape):
        raise ValueError("Invalid galaxy arrays in event cache (shape mismatch).")
    if z.size == 0:
        raise ValueError("No galaxies provided.")

    if not bool(pe.nest):
        raise ValueError("PE pixel histogram must use NESTED ordering.")

    npix = int(hp.nside2npix(int(pe.nside)))
    pix_to_row = np.full((npix,), -1, dtype=np.int32)
    pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
    row = pix_to_row[ipix]
    good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0) & np.isfinite(cos)
    if not np.any(good):
        raise ValueError("All galaxies map outside the PE credible region (or have invalid z/w).")
    z = z[good]
    w = w[good]
    row = row[good].astype(np.int64, copy=False)
    cos = cos[good]

    prob = np.asarray(pe.prob_pix, dtype=float)[row]

    prior = _GSCAN_GW_PRIOR
    post = _GSCAN_POST

    edges = np.asarray(pe.dL_edges, dtype=float)
    nb = int(edges.size - 1)
    pdf_flat = np.asarray(pe.pdf_bins, dtype=float).reshape(-1)
    logpdf_flat = np.log(np.clip(pdf_flat, 1e-300, np.inf))
    logpdf_floor = float(np.log(1e-300))

    prior_mode = str(getattr(prior, "mode", ""))
    use_fast_logbins = prior_mode == "dL_powerlaw"
    log_edges = np.log(edges)
    dlog = float(log_edges[1] - log_edges[0])
    if not (np.isfinite(dlog) and dlog > 0.0):
        use_fast_logbins = False
    if not np.allclose(np.diff(log_edges), dlog, rtol=1e-8, atol=0.0):
        use_fast_logbins = False
    log_edges0 = float(log_edges[0])
    inv_dlog = 1.0 / dlog if use_fast_logbins else float("nan")
    k_prior = float(getattr(prior, "powerlaw_k", 0.0)) if use_fast_logbins else float("nan")

    chunk = int(_GSCAN_GAL_CHUNK_SIZE)
    if chunk <= 0:
        raise ValueError("gal_chunk_size must be positive.")

    n_draws = int(post.H_samples.shape[0])
    g_grid_u = [float(g) for g in sorted(set(g_grid))]
    cat_mu_by_g: dict[float, np.ndarray] = {g: np.full((n_draws,), -np.inf, dtype=float) for g in g_grid_u}
    cat_gr = np.full((n_draws,), -np.inf, dtype=float)

    for a in range(0, z.size, chunk):
        b = min(z.size, a + chunk)
        z_c = np.asarray(z[a:b], dtype=float)
        w_c = np.asarray(w[a:b], dtype=float)
        row_c = np.asarray(row[a:b], dtype=np.int64)
        prob_c = np.asarray(prob[a:b], dtype=float)
        cos_c = np.asarray(cos[a:b], dtype=float)

        z_u, inv = np.unique(z_c, return_inverse=True)
        dL_em_u = predict_dL_em(post, z_eval=z_u)
        _, R_u = predict_r_gw_em(post, z_eval=z_u, convention=convention, allow_extrapolation=False)
        if not use_fast_logbins:
            dL_gw_u = dL_em_u * np.asarray(R_u, dtype=float)
            dL_em = dL_em_u[:, inv]
            dL_gw = dL_gw_u[:, inv]

        logw = np.log(np.clip(w_c, 1e-30, np.inf))[None, :]
        logprob = np.log(np.clip(prob_c, 1e-300, np.inf))[None, :]
        row_nb = row_c.reshape((1, -1)) * nb  # (1, n_chunk) broadcasted

        def _chunk_logL(dL: np.ndarray) -> np.ndarray:
            dL = np.asarray(dL, dtype=float)
            bin_idx = np.searchsorted(edges, dL, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL) & (dL > 0.0)

            lin = row_nb + np.clip(bin_idx, 0, nb - 1)
            pdf = pdf_flat[lin]
            pdf = np.where(valid, pdf, 0.0)

            logpdf = np.log(np.clip(pdf, 1e-300, np.inf))
            logprior = prior.log_pi_dL(np.clip(dL, 1e-6, np.inf))
            logterm = logw + logprob + logpdf - logprior
            logterm = np.where(np.isfinite(logprior), logterm, -np.inf)
            return logsumexp(logterm, axis=1)

        def _chunk_logL_logdL(logdL: np.ndarray, *, work_int: np.ndarray) -> np.ndarray:
            logdL = np.asarray(logdL, dtype=float)
            if not use_fast_logbins:
                raise RuntimeError("Fast log-binning requested but unavailable.")
            work_int[:] = np.floor((logdL - log_edges0) * inv_dlog).astype(np.int64)
            valid = (work_int >= 0) & (work_int < nb) & np.isfinite(logdL)
            lin = row_nb + np.clip(work_int, 0, nb - 1)
            logpdf = logpdf_flat[lin]
            logpdf = np.where(valid, logpdf, logpdf_floor)
            logprior = k_prior * logdL
            logterm = logw + logprob + logpdf - logprior
            return logsumexp(logterm, axis=1)

        if use_fast_logbins:
            work_int = np.empty((n_draws, z_c.size), dtype=np.int64)
            logdL_em_u = np.log(np.clip(dL_em_u, 1e-12, np.inf))
            logdL_em = logdL_em_u[:, inv]
            cat_gr = np.logaddexp(cat_gr, _chunk_logL_logdL(logdL_em, work_int=work_int))

            logR_u = np.log(np.clip(R_u, 1e-12, np.inf))
            logdL_gw_u = logdL_em_u + logR_u
            logdL_gw = logdL_gw_u[:, inv]
            logdL_tmp = np.empty_like(logdL_gw)
            gcos = cos_c.reshape((1, -1))

            for g in g_grid_u:
                if np.isclose(g, 0.0):
                    cat_mu_by_g[g] = np.logaddexp(cat_mu_by_g[g], _chunk_logL_logdL(logdL_gw, work_int=work_int))
                else:
                    np.add(logdL_gw, float(g) * gcos, out=logdL_tmp)
                    cat_mu_by_g[g] = np.logaddexp(cat_mu_by_g[g], _chunk_logL_logdL(logdL_tmp, work_int=work_int))
        else:
            # GR (independent of g).
            cat_gr = np.logaddexp(cat_gr, _chunk_logL(dL_em))

            # μ for each g (reuse predicted dL_em/R).
            for g in g_grid_u:
                if np.isclose(g, 0.0):
                    dL_mu = dL_gw
                else:
                    scale = np.exp(float(g) * cos_c).reshape((1, -1))
                    dL_mu = dL_gw * scale
                cat_mu_by_g[g] = np.logaddexp(cat_mu_by_g[g], _chunk_logL(dL_mu))

    miss_mu_by_g: dict[float, np.ndarray] = {}
    if bool(_GSCAN_DISABLE_MISSING):
        miss_gr = np.full((n_draws,), float("-inf"), dtype=float)
        for g in g_grid_u:
            miss_mu_by_g[float(g)] = np.full((n_draws,), float("-inf"), dtype=float)
    else:
        miss_mu0, miss_gr = compute_missing_host_logL_draws_from_histogram(
            prob_pix=np.asarray(pe.prob_pix, dtype=float),
            pdf_bins=np.asarray(pe.pdf_bins, dtype=float),
            dL_edges=np.asarray(pe.dL_edges, dtype=float),
            pre=_GSCAN_PRE,
            gw_distance_prior=_GSCAN_GW_PRIOR,
            distance_mode="full",
            pixel_chunk_size=int(_GSCAN_MISSING_PIXEL_CHUNK_SIZE),
            g_aniso=0.0,
            cos_theta_pix=cos_pix_sel,
            compute_gr=True,
        )
        miss_mu_by_g[0.0] = np.asarray(miss_mu0, dtype=float)
        miss_gr = np.asarray(miss_gr, dtype=float)
        for g in g_grid_u:
            if np.isclose(g, 0.0):
                continue
            miss_mu, _ = compute_missing_host_logL_draws_from_histogram(
                prob_pix=np.asarray(pe.prob_pix, dtype=float),
                pdf_bins=np.asarray(pe.pdf_bins, dtype=float),
                dL_edges=np.asarray(pe.dL_edges, dtype=float),
                pre=_GSCAN_PRE,
                gw_distance_prior=_GSCAN_GW_PRIOR,
                distance_mode="full",
                pixel_chunk_size=int(_GSCAN_MISSING_PIXEL_CHUNK_SIZE),
                g_aniso=float(g),
                cos_theta_pix=cos_pix_sel,
                compute_gr=False,
            )
            miss_mu_by_g[float(g)] = np.asarray(miss_mu, dtype=float)

    return {
        "event": ev,
        "meta": {
            "event": ev,
            "n_gal": int(meta.get("n_gal", int(z_gal.size))),
            "sky_area_deg2": float(meta.get("sky_area_deg2", float("nan"))),
            "n_pix_sel": int(np.asarray(pe.pix_sel).size),
        },
        "cat_gr": cat_gr,
        "miss_gr": miss_gr,
        "cat_mu_by_g": {float(g): np.asarray(cat_mu_by_g[float(g)], dtype=float) for g in g_grid_u},
        "miss_mu_by_g": miss_mu_by_g,
    }


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _format_duration_s(s: float) -> str:
    s = float(s)
    if not np.isfinite(s) or s < 0.0:
        return "?"
    m, sec = divmod(int(s + 0.5), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _write_progress_json(outdir: Path, obj: dict[str, Any]) -> None:
    tmp = outdir / "progress.json.tmp"
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(outdir / "progress.json")


def logmeanexp_axis(x: np.ndarray, *, axis: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("logmeanexp_axis: empty array")
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)


def logmeanexp_1d(x: np.ndarray) -> float:
    return float(logmeanexp_axis(np.asarray(x, dtype=float).reshape(1, -1), axis=1)[0])


def logsumexp_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return float("-inf")
    m = float(np.max(x))
    return float(m + np.log(np.sum(np.exp(x - m))))


def trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("trapz_weights expects 1D array with >=2 entries")
    if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("trapz_weights expects finite strictly increasing x")
    dx = np.diff(x)
    w = np.empty_like(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def log_trapz_integral(x: np.ndarray, logf: np.ndarray) -> float:
    """Compute log ∫ exp(logf(x)) dx via trapezoidal weights on the given x grid."""
    x = np.asarray(x, dtype=float)
    logf = np.asarray(logf, dtype=float)
    if x.ndim != 1 or logf.ndim != 1 or x.shape != logf.shape:
        raise ValueError("log_trapz_integral expects matching 1D arrays.")
    w = trapz_weights(x)
    lw = np.log(np.clip(w, 1e-300, np.inf))
    # Stable logsumexp
    m = float(np.max(logf + lw))
    if not np.isfinite(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(logf + lw - m))))


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

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 3:
        gmin, gmax, step = (float(x) for x in parts)
        if step <= 0:
            raise ValueError("g step must be >0")
        n = int(np.floor((gmax - gmin) / step + 0.5)) + 1
        grid = gmin + step * np.arange(n, dtype=float)
        grid = grid[(grid >= gmin - 1e-12) & (grid <= gmax + 1e-12)]
        grid = np.asarray(grid, dtype=float)
        # Stabilize float representations (e.g., avoid ~1e-16 keys for 0.0).
        grid = np.round(grid, 12)
        grid[np.isclose(grid, 0.0, atol=1e-12, rtol=0.0)] = 0.0
        return grid

    grid = np.array([float(x) for x in parts], dtype=float)
    if grid.size < 1:
        raise ValueError("Need at least one g value")
    grid = np.round(grid, 12)
    grid[np.isclose(grid, 0.0, atol=1e-12, rtol=0.0)] = 0.0
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


def g_prior_logpdf(
    g: np.ndarray,
    *,
    prior_type: str,
    mu: float,
    sigma: float,
    uniform_min: float,
    uniform_max: float,
) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    t = str(prior_type).strip().lower()
    if t == "normal":
        sigma = float(sigma)
        if not (np.isfinite(sigma) and sigma > 0.0):
            raise ValueError("--g-prior-sigma must be finite and >0.")
        mu = float(mu)
        return -0.5 * ((g - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))
    if t == "uniform":
        a = float(uniform_min)
        b = float(uniform_max)
        if not (np.isfinite(a) and np.isfinite(b) and b > a):
            raise ValueError("--g-prior-uniform-min/max must be finite and max>min.")
        out = np.full_like(g, -np.inf, dtype=float)
        m = (g >= a) & (g <= b)
        out[m] = -np.log(b - a)
        return out
    raise ValueError("--g-prior-type must be one of: normal, uniform")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=str, default="cmb", help="Axis preset: cmb|secrest|ecliptic_north")
    ap.add_argument("--axis-name", type=str, default=None)
    ap.add_argument("--axis-frame", type=str, default="galactic")
    ap.add_argument("--axis-lon-deg", type=float, default=None)
    ap.add_argument("--axis-lat-deg", type=float, default=None)
    ap.add_argument("--g-grid", type=str, default="-0.05,0.05,0.01", help="Either gmin,gmax,step or g1,g2,...")
    ap.add_argument("--g-prior-type", type=str, default="normal", choices=["normal", "uniform"])
    ap.add_argument("--g-prior-mu", type=float, default=0.0, help="Mean for Normal g prior.")
    ap.add_argument("--g-prior-sigma", type=float, default=0.2, help="Stddev for Normal g prior.")
    ap.add_argument("--g-prior-uniform-min", type=float, default=-1.0, help="Min for Uniform g prior.")
    ap.add_argument("--g-prior-uniform-max", type=float, default=1.0, help="Max for Uniform g prior.")

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
        help="Existing dark-siren production output dir that contains cache/ and cache_terms/cache_missing.",
    )
    ap.add_argument("--run-label", type=str, default="M0_start101", help="Run label used in cache_* filenames.")

    ap.add_argument("--galaxy-chunk-size", type=int, default=50_000)
    ap.add_argument("--missing-pixel-chunk-size", type=int, default=5_000)
    ap.add_argument("--max-draws", type=int, default=0, help="If >0, use only the first N posterior draws (fast mode).")
    ap.add_argument("--max-events", type=int, default=0, help="0 means all events.")
    ap.add_argument(
        "--nproc",
        type=int,
        default=0,
        help="Worker processes for per-event logL computations (0=auto, 1=serial).",
    )
    ap.add_argument(
        "--progress-seconds",
        type=float,
        default=30.0,
        help="Emit a heartbeat progress line every N seconds during long phases.",
    )

    ap.add_argument(
        "--injections-hdf",
        type=str,
        default="",
        help="O3 injections HDF5 file for selection-alpha(model).",
    )
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0)
    ap.add_argument("--disable-selection", action="store_true", help="Skip selection alpha correction.")
    ap.add_argument(
        "--selection-progress-every",
        type=int,
        default=25,
        help="Emit selection-alpha progress every N posterior draws (0 disables).",
    )

    ap.add_argument("--disable-missing", action="store_true", help="Skip missing-host term (cat-only).")
    ap.add_argument(
        "--f-miss-mode",
        type=str,
        default="summary",
        choices=["summary", "fixed", "marginalize"],
        help="How to handle f_miss (missing-host mixture fraction).",
    )
    ap.add_argument("--f-miss", type=float, default=None, help="Fixed f_miss when --f-miss-mode=fixed.")

    ap.add_argument(
        "--sanity-check-event",
        type=str,
        default="",
        help="If set, compare g=0 cat+missing arrays to cache_terms/cache_missing and exit.",
    )

    ap.add_argument("--outdir", type=str, default="", help="Output directory (default: outputs/darksiren_fixed_axis_gscan_full_<axis>_<tag>)")
    ap.add_argument("--make-plot", action="store_true", help="Write a PNG plot to outdir.")
    args = ap.parse_args()

    axis = axis_spec_from_args(args)
    axis_icrs = axis_unitvec_icrs(axis)
    g_grid = np.unique(parse_g_grid(args.g_grid).astype(float))

    base_out = Path(args.cache_outdir).expanduser().resolve()
    cache_dir = base_out / "cache"
    cache_terms_dir = base_out / "cache_terms"
    cache_missing_dir = base_out / "cache_missing"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing cache dir: {cache_dir}")
    if not cache_terms_dir.exists():
        raise FileNotFoundError(f"Missing cache_terms dir: {cache_terms_dir}")
    if not cache_missing_dir.exists():
        raise FileNotFoundError(f"Missing cache_missing dir: {cache_missing_dir}")

    rows = json.loads(Path(args.event_scores_json).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("event-scores-json must be a non-empty list")
    rows = sorted(rows, key=lambda r: int(r.get("n_gal", 0)))
    events = [str(r["event"]) for r in rows]
    if int(args.max_events) > 0:
        events = events[: int(args.max_events)]

    summ = json.loads(Path(args.summary_json).read_text())
    convention = str(summ.get("convention", "A"))
    draw_idx = [int(i) for i in summ["draw_idx"]]

    from entropy_horizon_recon.dark_sirens_incompleteness import compute_missing_host_logL_draws_from_histogram, precompute_missing_host_prior
    from entropy_horizon_recon.dark_sirens_pe import PePixelDistanceHistogram, compute_dark_siren_logL_draws_from_pe_hist
    from entropy_horizon_recon.dark_sirens_selection import (
        compute_selection_alpha_from_injections_g_grid,
        compute_selection_alpha_from_injections,
        load_o3_injections,
        resolve_o3_sensitivity_injection_file,
    )
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

    # Defaults match production: auto -> dL^2 powerlaw.
    gw_prior = GWDistancePrior(mode="dL_powerlaw", powerlaw_k=2.0)

    # Missing-host prior precompute (draw-wise background).
    mix = dict(summ.get("mixture", {}))
    miss_pre_meta = dict(mix.get("missing_pre", {}))
    missing_z_max = float(miss_pre_meta.get("z_max", 0.3))
    host_prior_z_mode = str(miss_pre_meta.get("host_prior_z_mode", "comoving_uniform"))
    host_prior_z_k = float(miss_pre_meta.get("host_prior_z_k", 0.0))
    pre = precompute_missing_host_prior(
        post,
        convention=convention,
        z_max=missing_z_max,
        host_prior_z_mode=host_prior_z_mode,
        host_prior_z_k=host_prior_z_k,
    )

    # Infer pe_nside from one cached event and precompute cos(theta) for pixel centers.
    import healpy as hp

    with np.load(cache_dir / f"event_{events[0]}.npz", allow_pickle=True) as d0:
        meta0 = json.loads(str(d0["meta"].tolist()))
        pe_nside = int(meta0.get("pe_nside", 64))
        p_credible = float(meta0.get("p_credible", 0.9))

    npix = int(hp.nside2npix(pe_nside))
    x, y, z = hp.pix2vec(pe_nside, np.arange(npix, dtype=np.int64), nest=True)
    cos_theta_pix = axis_icrs[0] * x + axis_icrs[1] * y + axis_icrs[2] * z
    cos_theta_pix = np.asarray(cos_theta_pix, dtype=np.float32)

    # Sanity check: compare g=0 arrays for one event against cache_terms/cache_missing.
    if str(args.sanity_check_event).strip():
        ev = str(args.sanity_check_event).strip()
        ev_path = cache_dir / f"event_{ev}.npz"
        if not ev_path.exists():
            raise FileNotFoundError(f"Missing event cache: {ev_path}")
        cat_path = cache_terms_dir / f"cat_{ev}__{str(args.run_label)}.npz"
        miss_path = cache_missing_dir / f"missing_{ev}__{str(args.run_label)}.npz"
        if not cat_path.exists():
            raise FileNotFoundError(f"Missing cat cache_terms: {cat_path}")
        if not miss_path.exists():
            raise FileNotFoundError(f"Missing missing cache: {miss_path}")

        with np.load(ev_path, allow_pickle=True) as d:
            meta = json.loads(str(d["meta"].tolist()))
            z_gal = np.asarray(d["z"], dtype=float)
            w_gal = np.asarray(d["w"], dtype=float)
            ipix_gal = np.asarray(d["ipix"], dtype=np.int64)
            pe = PePixelDistanceHistogram(
                nside=int(meta.get("pe_nside", pe_nside)),
                nest=True,
                p_credible=float(meta.get("p_credible", p_credible)),
                pix_sel=np.asarray(d["pe_pix_sel"], dtype=np.int64),
                prob_pix=np.asarray(d["pe_prob_pix"], dtype=float),
                dL_edges=np.asarray(d["pe_dL_edges"], dtype=float),
                pdf_bins=np.asarray(d["pe_pdf_bins"], dtype=float),
            )

        cos_gal = cos_theta_pix[ipix_gal]
        cat_mu, cat_gr = compute_dark_siren_logL_draws_from_pe_hist(
            event=ev,
            pe=pe,
            post=post,
            z_gal=z_gal,
            w_gal=w_gal,
            ipix_gal=ipix_gal,
            convention=convention,
            gw_distance_prior=gw_prior,
            distance_mode="full",
            gal_chunk_size=int(args.galaxy_chunk_size),
            g_aniso=0.0,
            cos_theta_gal=cos_gal,
            compute_gr=True,
        )
        cos_pix_sel = cos_theta_pix[np.asarray(pe.pix_sel, dtype=np.int64)]
        miss_mu, miss_gr = compute_missing_host_logL_draws_from_histogram(
            prob_pix=np.asarray(pe.prob_pix, dtype=float),
            pdf_bins=np.asarray(pe.pdf_bins, dtype=float),
            dL_edges=np.asarray(pe.dL_edges, dtype=float),
            pre=pre,
            gw_distance_prior=gw_prior,
            distance_mode="full",
            pixel_chunk_size=int(args.missing_pixel_chunk_size),
            g_aniso=0.0,
            cos_theta_pix=cos_pix_sel,
            compute_gr=True,
        )

        with np.load(cat_path, allow_pickle=True) as d:
            ref_cat_mu = np.asarray(d["logL_cat_mu"], dtype=float)[:n_draws]
            ref_cat_gr = np.asarray(d["logL_cat_gr"], dtype=float)[:n_draws]
        with np.load(miss_path, allow_pickle=True) as d:
            ref_miss_mu = np.asarray(d["logL_missing_mu"], dtype=float)[:n_draws]
            ref_miss_gr = np.asarray(d["logL_missing_gr"], dtype=float)[:n_draws]

        out = {
            "event": ev,
            "n_draws": int(n_draws),
            "max_abs_diff_cat_mu": float(np.max(np.abs(cat_mu - ref_cat_mu))),
            "max_abs_diff_cat_gr": float(np.max(np.abs(cat_gr - ref_cat_gr))),
            "max_abs_diff_missing_mu": float(np.max(np.abs(miss_mu - ref_miss_mu))),
            "max_abs_diff_missing_gr": float(np.max(np.abs(miss_gr - ref_miss_gr))),
        }
        print(json.dumps(out, indent=2))
        return 0

    outdir = Path(args.outdir) if str(args.outdir).strip() else Path(f"outputs/darksiren_fixed_axis_gscan_full_{axis.name}_{utc_tag()}")
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    progress_seconds = float(args.progress_seconds)
    if not np.isfinite(progress_seconds) or progress_seconds <= 0.0:
        progress_seconds = 30.0

    t0 = time.monotonic()
    print(
        "[gscan_full] CONFIG "
        f"axis={axis.name} n_events={len(events)} n_draws={int(post.H_samples.shape[0])} "
        f"g_grid=[{', '.join(f'{float(g):.3g}' for g in g_grid.tolist())}] nproc_req={int(args.nproc)}",
        flush=True,
    )
    _write_progress_json(
        outdir,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "init",
            "axis": axis.name,
            "n_events": int(len(events)),
            "n_draws": int(post.H_samples.shape[0]),
            "g_grid": [float(g) for g in g_grid.tolist()],
        },
    )

    # Compute per-event logL vectors (cat + missing) using multiprocessing.
    g_keys = [float(g) for g in g_grid.tolist()]
    n_ev = int(len(events))

    # Initialize globals for worker processes (fork will share these).
    global _GSCAN_CACHE_DIR, _GSCAN_POST, _GSCAN_PRE, _GSCAN_GW_PRIOR, _GSCAN_CONVENTION, _GSCAN_G_GRID
    global _GSCAN_COS_THETA_PIX, _GSCAN_PE_NSIDE, _GSCAN_P_CREDIBLE, _GSCAN_GAL_CHUNK_SIZE, _GSCAN_MISSING_PIXEL_CHUNK_SIZE
    global _GSCAN_DISABLE_MISSING

    _GSCAN_CACHE_DIR = cache_dir
    _GSCAN_POST = post
    _GSCAN_PRE = pre
    _GSCAN_GW_PRIOR = gw_prior
    _GSCAN_CONVENTION = convention
    _GSCAN_G_GRID = g_keys
    _GSCAN_COS_THETA_PIX = cos_theta_pix
    _GSCAN_PE_NSIDE = int(pe_nside)
    _GSCAN_P_CREDIBLE = float(p_credible)
    _GSCAN_GAL_CHUNK_SIZE = int(args.galaxy_chunk_size)
    _GSCAN_MISSING_PIXEL_CHUNK_SIZE = int(args.missing_pixel_chunk_size)
    _GSCAN_DISABLE_MISSING = bool(args.disable_missing)

    # Pre-allocate in event order for deterministic reduction.
    ev_to_idx = {str(ev): i for i, ev in enumerate(events)}
    events_meta: list[dict[str, Any] | None] = [None] * n_ev
    cat_gr_list: list[np.ndarray | None] = [None] * n_ev
    miss_gr_list: list[np.ndarray | None] = [None] * n_ev
    cat_mu_by_g: dict[float, list[np.ndarray | None]] = {g: [None] * n_ev for g in g_keys}
    miss_mu_by_g: dict[float, list[np.ndarray | None]] = {g: [None] * n_ev for g in g_keys}

    nproc_req = int(args.nproc)
    if nproc_req < 0:
        raise ValueError("--nproc must be >= 0")
    if nproc_req == 0:
        try:
            n_aff = len(os.sched_getaffinity(0))
        except Exception:
            n_aff = os.cpu_count() or 1
        nproc_eff = max(1, min(int(n_aff), n_ev))
    else:
        nproc_eff = max(1, min(int(nproc_req), n_ev))

    if nproc_eff == 1:
        for done_i, ev in enumerate(events, start=1):
            res = _gscan_full_compute_event(str(ev))
            i_ev = int(ev_to_idx[str(ev)])
            events_meta[i_ev] = dict(res["meta"])
            cat_gr_list[i_ev] = np.asarray(res["cat_gr"], dtype=float)
            miss_gr_list[i_ev] = np.asarray(res["miss_gr"], dtype=float)
            for g in g_keys:
                cat_mu_by_g[g][i_ev] = np.asarray(res["cat_mu_by_g"][g], dtype=float)
                miss_mu_by_g[g][i_ev] = np.asarray(res["miss_mu_by_g"][g], dtype=float)
            m = res["meta"]
            elapsed_s = time.monotonic() - t0
            rate = float(done_i) / elapsed_s if elapsed_s > 0.0 else 0.0
            eta_s = (float(n_ev - done_i) / rate) if rate > 0.0 else float("inf")
            print(
                f"[gscan_full] {done_i}/{n_ev} {m['event']} n_gal={int(m['n_gal'])} n_pix_sel={int(m['n_pix_sel'])} "
                f"elapsed={_format_duration_s(elapsed_s)} eta={_format_duration_s(eta_s)}",
                flush=True,
            )
            _write_progress_json(
                outdir,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "stage": "per_event_logL",
                    "axis": axis.name,
                    "n_events": int(n_ev),
                    "done_events": int(done_i),
                    "elapsed_s": float(elapsed_s),
                    "eta_s": float(eta_s) if np.isfinite(eta_s) else None,
                    "last_event": str(m.get("event", ev)),
                    "last_event_n_gal": int(m.get("n_gal", 0)),
                    "last_event_n_pix_sel": int(m.get("n_pix_sel", 0)),
                },
            )
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=int(nproc_eff), mp_context=ctx) as ex:
            futures = {ex.submit(_gscan_full_compute_event, str(ev)): str(ev) for ev in events}
            pending = set(futures.keys())
            last_heartbeat = time.monotonic()
            done_i = 0
            print(
                f"[gscan_full] starting per-event logL with nproc_eff={int(nproc_eff)} (events={int(n_ev)})",
                flush=True,
            )
            while pending:
                done_set, pending = wait(pending, timeout=float(progress_seconds), return_when=FIRST_COMPLETED)
                for fut in done_set:
                    done_i += 1
                    ev = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        raise RuntimeError(f"Per-event computation failed for event={ev}") from e
                    i_ev = int(ev_to_idx[ev])
                    events_meta[i_ev] = dict(res["meta"])
                    cat_gr_list[i_ev] = np.asarray(res["cat_gr"], dtype=float)
                    miss_gr_list[i_ev] = np.asarray(res["miss_gr"], dtype=float)
                    for g in g_keys:
                        cat_mu_by_g[g][i_ev] = np.asarray(res["cat_mu_by_g"][g], dtype=float)
                        miss_mu_by_g[g][i_ev] = np.asarray(res["miss_mu_by_g"][g], dtype=float)
                    m = res["meta"]
                    elapsed_s = time.monotonic() - t0
                    rate = float(done_i) / elapsed_s if elapsed_s > 0.0 else 0.0
                    eta_s = (float(n_ev - done_i) / rate) if rate > 0.0 else float("inf")
                    print(
                        f"[gscan_full] {done_i}/{n_ev} {m['event']} n_gal={int(m['n_gal'])} n_pix_sel={int(m['n_pix_sel'])} "
                        f"elapsed={_format_duration_s(elapsed_s)} eta={_format_duration_s(eta_s)}",
                        flush=True,
                    )
                    _write_progress_json(
                        outdir,
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "stage": "per_event_logL",
                            "axis": axis.name,
                            "n_events": int(n_ev),
                            "done_events": int(done_i),
                            "elapsed_s": float(elapsed_s),
                            "eta_s": float(eta_s) if np.isfinite(eta_s) else None,
                            "last_event": str(m.get("event", ev)),
                            "last_event_n_gal": int(m.get("n_gal", 0)),
                            "last_event_n_pix_sel": int(m.get("n_pix_sel", 0)),
                        },
                    )

                now = time.monotonic()
                if now - last_heartbeat >= float(progress_seconds):
                    elapsed_s = now - t0
                    rate = float(done_i) / elapsed_s if (elapsed_s > 0.0 and done_i > 0) else 0.0
                    eta_s = (float(n_ev - done_i) / rate) if rate > 0.0 else float("inf")
                    print(
                        f"[gscan_full] HEARTBEAT {done_i}/{n_ev} ({100.0*done_i/max(1,n_ev):.1f}%) "
                        f"elapsed={_format_duration_s(elapsed_s)} eta={_format_duration_s(eta_s)}",
                        flush=True,
                    )
                    _write_progress_json(
                        outdir,
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "stage": "per_event_logL",
                            "axis": axis.name,
                            "n_events": int(n_ev),
                            "done_events": int(done_i),
                            "elapsed_s": float(elapsed_s),
                            "eta_s": float(eta_s) if np.isfinite(eta_s) else None,
                        },
                    )
                    last_heartbeat = now

    # Finalize (strip Nones).
    if any(m is None for m in events_meta):
        raise RuntimeError("Internal error: missing per-event metadata entries.")
    if any(v is None for v in cat_gr_list) or any(v is None for v in miss_gr_list):
        raise RuntimeError("Internal error: missing per-event GR vectors.")
    for g in g_keys:
        if any(v is None for v in cat_mu_by_g[g]) or any(v is None for v in miss_mu_by_g[g]):
            raise RuntimeError(f"Internal error: missing per-event μ vectors for g={g}.")

    events_meta = [m for m in events_meta if m is not None]
    cat_gr_list = [v for v in cat_gr_list if v is not None]
    miss_gr_list = [v for v in miss_gr_list if v is not None]
    cat_mu_by_g = {g: [v for v in cat_mu_by_g[g] if v is not None] for g in g_keys}
    miss_mu_by_g = {g: [v for v in miss_mu_by_g[g] if v is not None] for g in g_keys}

    elapsed_s = time.monotonic() - t0
    print(
        f"[gscan_full] computed per-event logL vectors for {n_ev} events; computing selection alpha... "
        f"(elapsed={_format_duration_s(elapsed_s)})",
        flush=True,
    )
    _write_progress_json(
        outdir,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "selection_alpha",
            "axis": axis.name,
            "n_events": int(n_ev),
            "done_events": int(n_ev),
            "elapsed_s": float(elapsed_s),
        },
    )

    # Selection alpha(model), global.
    log_alpha_mu_by_g: dict[float, np.ndarray] = {}
    log_alpha_gr: np.ndarray | None = None
    sel_meta = dict(summ.get("selection_alpha", {}))
    injections_hdf_used: str | None = None
    if bool(args.disable_selection):
        log_alpha_gr = None
        for g in g_grid.tolist():
            log_alpha_mu_by_g[float(g)] = np.zeros((n_draws,), dtype=float)
    else:
        inj_arg = str(getattr(args, "injections_hdf", "")).strip()
        if inj_arg:
            inj_path = Path(inj_arg).expanduser().resolve()
        else:
            inj_path = resolve_o3_sensitivity_injection_file(
                events=list(events),
                base_dir="data/cache/gw/zenodo",
                record_id=7890437,
                population="mixture",
                auto_download=True,
            )
        injections_hdf_used = str(inj_path)
        injections = load_o3_injections(str(inj_path), ifar_threshold_yr=float(args.selection_ifar_thresh_yr))
        sel_grid = compute_selection_alpha_from_injections_g_grid(
            post=post,
            injections=injections,
            convention=convention,
            z_max=float(sel_meta.get("z_max", missing_z_max)),
            g_grid=g_grid,
            snr_threshold=sel_meta.get("snr_threshold", None),
            det_model=str(sel_meta.get("det_model", "snr_binned")),
            snr_binned_nbins=int(sel_meta.get("snr_binned_nbins", 200)),
            weight_mode=str(sel_meta.get("weight_mode", "inv_sampling_pdf")),
            mu_det_distance="gw",
            axis_icrs=axis_icrs,
            progress_every_draws=int(args.selection_progress_every),
        )
        log_alpha_gr = np.log(np.clip(np.asarray(sel_grid["alpha_gr"], dtype=float), 1e-300, 1.0))
        mu_by_g = sel_grid["alpha_mu_by_g"]
        assert isinstance(mu_by_g, dict)
        for g in g_grid.tolist():
            gg = float(g)
            if gg not in mu_by_g:
                raise RuntimeError(f"Selection alpha missing g={gg} in alpha_mu_by_g")
            log_alpha_mu_by_g[gg] = np.log(np.clip(np.asarray(mu_by_g[gg], dtype=float), 1e-300, 1.0))

    elapsed_s = time.monotonic() - t0
    print(f"[gscan_full] selection alpha complete; building ΔLPD(g) curve... (elapsed={_format_duration_s(elapsed_s)})", flush=True)
    _write_progress_json(
        outdir,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "build_curve",
            "axis": axis.name,
            "n_events": int(n_ev),
            "done_events": int(n_ev),
            "elapsed_s": float(elapsed_s),
        },
    )

    # f_miss handling.
    f_miss_ref = float(mix.get("f_miss_ref", mix.get("f_miss", 0.0)))
    f_mode = str(args.f_miss_mode)
    do_marg = False
    f_meta = dict(mix.get("f_miss_meta", {}))
    if f_mode == "summary":
        do_marg = str(f_meta.get("mode", "fixed")) == "marginalize"
    elif f_mode == "marginalize":
        do_marg = True
    elif f_mode == "fixed":
        do_marg = False
    else:  # pragma: no cover
        raise ValueError("Unknown --f-miss-mode.")

    if not do_marg:
        if f_mode == "fixed" and args.f_miss is not None:
            f_miss_ref = float(args.f_miss)
        if not (np.isfinite(f_miss_ref) and 0.0 <= f_miss_ref <= 1.0):
            raise ValueError("f_miss must be finite and in [0,1].")

    # Build curve over g.
    curve: list[dict[str, Any]] = []
    for g in g_grid.tolist():
        g = float(g)

        cat_mu_list = cat_mu_by_g[g]
        miss_mu_list = miss_mu_by_g[g]
        if int(len(cat_mu_list)) != n_ev or int(len(miss_mu_list)) != n_ev:
            raise RuntimeError("Internal error: missing per-event mu vectors.")

        # Fixed f_ref totals at this g.
        if bool(args.disable_missing):
            logL_mu_ref = np.sum(np.stack(cat_mu_list, axis=0), axis=0)
            logL_gr_ref = np.sum(np.stack(cat_gr_list, axis=0), axis=0)
        else:
            log_a = np.log1p(-f_miss_ref) if f_miss_ref < 1.0 else -np.inf
            log_b = np.log(f_miss_ref) if f_miss_ref > 0.0 else -np.inf
            logL_mu_ref = np.zeros((n_draws,), dtype=float)
            logL_gr_ref = np.zeros((n_draws,), dtype=float)
            for i_ev in range(n_ev):
                logL_mu_ref += np.logaddexp(log_a + cat_mu_list[i_ev], log_b + miss_mu_list[i_ev])
                logL_gr_ref += np.logaddexp(log_a + cat_gr_list[i_ev], log_b + miss_gr_list[i_ev])

        lpd_mu_data_ref = float(logmeanexp_1d(logL_mu_ref))
        lpd_gr_data_ref = float(logmeanexp_1d(logL_gr_ref))

        # Apply selection correction at combined-draw level.
        if log_alpha_gr is not None:
            logL_mu_ref_sel = logL_mu_ref - float(n_ev) * log_alpha_mu_by_g[g]
            logL_gr_ref_sel = logL_gr_ref - float(n_ev) * log_alpha_gr
        else:
            logL_mu_ref_sel = logL_mu_ref
            logL_gr_ref_sel = logL_gr_ref

        lpd_mu_ref = float(logmeanexp_1d(logL_mu_ref_sel))
        lpd_gr_ref = float(logmeanexp_1d(logL_gr_ref_sel))

        # Optional global marginalization over f_miss.
        if do_marg and not bool(args.disable_missing):
            prior = dict(f_meta.get("prior", {}))
            grid = dict(f_meta.get("grid", {}))
            n_f = int(grid.get("n", 401))
            eps = float(grid.get("eps", 1e-6))
            f_grid = np.linspace(eps, 1.0 - eps, n_f)
            w_f = trapz_weights(f_grid)
            logw_f = np.log(np.clip(w_f, 1e-300, np.inf))
            logf = np.log(f_grid)
            log1mf = np.log1p(-f_grid)

            prior_type = str(prior.get("type", "uniform"))
            if prior_type == "uniform":
                log_prior_f = np.zeros_like(f_grid, dtype=float)
            elif prior_type == "beta":
                a = float(prior["alpha"])
                b = float(prior["beta"])
                log_prior_f = (a - 1.0) * logf + (b - 1.0) * log1mf - float(betaln(a, b))
            else:
                raise ValueError("Unknown f_miss prior type for marginalization.")

            logL_mu_fd = np.zeros((n_f, n_draws), dtype=float)
            logL_gr_fd = np.zeros((n_f, n_draws), dtype=float)
            for i_ev in range(n_ev):
                ev_mu = np.logaddexp(log1mf[:, None] + cat_mu_list[i_ev][None, :], logf[:, None] + miss_mu_list[i_ev][None, :])
                ev_gr = np.logaddexp(log1mf[:, None] + cat_gr_list[i_ev][None, :], logf[:, None] + miss_gr_list[i_ev][None, :])
                logL_mu_fd += ev_mu
                logL_gr_fd += ev_gr

            if log_alpha_gr is not None:
                logL_mu_fd = logL_mu_fd - float(n_ev) * log_alpha_mu_by_g[g].reshape((1, -1))
                logL_gr_fd = logL_gr_fd - float(n_ev) * log_alpha_gr.reshape((1, -1))

            lpd_mu_f = logmeanexp_axis(logL_mu_fd, axis=1)
            lpd_gr_f = logmeanexp_axis(logL_gr_fd, axis=1)
            log_int_mu = log_prior_f + lpd_mu_f + logw_f
            log_int_gr = log_prior_f + lpd_gr_f + logw_f
            lpd_mu = float(logsumexp_1d(log_int_mu))
            lpd_gr = float(logsumexp_1d(log_int_gr))

            # Data-only marginalized totals (undo selection correction).
            if log_alpha_gr is not None:
                logL_mu_fd_data = logL_mu_fd + float(n_ev) * log_alpha_mu_by_g[g].reshape((1, -1))
                logL_gr_fd_data = logL_gr_fd + float(n_ev) * log_alpha_gr.reshape((1, -1))
                lpd_mu_f_data = logmeanexp_axis(logL_mu_fd_data, axis=1)
                lpd_gr_f_data = logmeanexp_axis(logL_gr_fd_data, axis=1)
                lpd_mu_data = float(logsumexp_1d(log_prior_f + lpd_mu_f_data + logw_f))
                lpd_gr_data = float(logsumexp_1d(log_prior_f + lpd_gr_f_data + logw_f))
            else:
                lpd_mu_data = float(lpd_mu)
                lpd_gr_data = float(lpd_gr)

            def _posterior_mean_f(log_int: np.ndarray) -> float:
                m0 = float(np.max(log_int))
                w0 = np.exp(log_int - m0)
                denom = float(np.sum(w0))
                return float(np.sum(w0 * f_grid) / denom) if denom > 0 else float("nan")

            f_post_mu = _posterior_mean_f(log_int_mu)
            f_post_gr = _posterior_mean_f(log_int_gr)
        else:
            # No marginalization: use fixed f_ref totals.
            lpd_mu = float(lpd_mu_ref)
            lpd_gr = float(lpd_gr_ref)
            lpd_mu_data = float(lpd_mu_data_ref)
            lpd_gr_data = float(lpd_gr_data_ref)
            f_post_mu = None
            f_post_gr = None

        curve.append(
            {
                "g": float(g),
                "lpd_mu_total": float(lpd_mu),
                "lpd_gr_total": float(lpd_gr),
                "delta_lpd_total": float(lpd_mu - lpd_gr),
                "lpd_mu_data": float(lpd_mu_data),
                "lpd_gr_data": float(lpd_gr_data),
                "delta_lpd_data": float(lpd_mu_data - lpd_gr_data),
                "delta_lpd_sel": float((lpd_mu - lpd_gr) - (lpd_mu_data - lpd_gr_data)),
                "lpd_mu_total_at_f_ref": float(lpd_mu_ref),
                "lpd_gr_total_at_f_ref": float(lpd_gr_ref),
                "delta_lpd_total_at_f_ref": float(lpd_mu_ref - lpd_gr_ref),
                "f_miss_posterior_mean_mu": float(f_post_mu) if f_post_mu is not None else None,
                "f_miss_posterior_mean_gr": float(f_post_gr) if f_post_gr is not None else None,
            }
        )

    # Quadratic fit for ΔLPD_total vs g (use whatever grid user provides).
    xs = np.array([r["g"] for r in curve], dtype=float)
    ys = np.array([r["delta_lpd_total"] for r in curve], dtype=float)
    fit = {}
    if xs.size >= 3 and np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
        a, b, c = np.polyfit(xs, ys, deg=2)
        g_hat = float(-b / (2.0 * a)) if a != 0 else float("nan")
        sigma = float(np.sqrt(-1.0 / (2.0 * a))) if a < 0 else float("nan")
        fit = {"a": float(a), "b": float(b), "c": float(c), "g_hat": g_hat, "sigma_g": sigma}

    # g-marginalization with an explicit prior (complexity penalty).
    by_g = {round(float(r["g"]), 12): r for r in curve}
    if 0.0 not in by_g:
        raise RuntimeError("g_grid must include 0.0 for g-marginalization baselines.")
    logp_g = g_prior_logpdf(
        xs,
        prior_type=str(args.g_prior_type),
        mu=float(args.g_prior_mu),
        sigma=float(args.g_prior_sigma),
        uniform_min=float(args.g_prior_uniform_min),
        uniform_max=float(args.g_prior_uniform_max),
    )

    lpd_mu = np.array([float(r["lpd_mu_total"]) for r in curve], dtype=float)
    lpd_gr = np.array([float(r["lpd_gr_total"]) for r in curve], dtype=float)
    if not np.all(np.isfinite(lpd_gr)):
        raise RuntimeError("Non-finite lpd_gr_total encountered.")
    lpd_gr0 = float(by_g[0.0]["lpd_gr_total"])
    lpd_mu0 = float(by_g[0.0]["lpd_mu_total"])

    logZ_mu = log_trapz_integral(xs, logp_g + lpd_mu)
    # GR has no g parameter; compare to its (constant) evidence.
    logBF_mu_over_gr = float(logZ_mu - lpd_gr0)
    logBF_mu_over_mu0 = float(logZ_mu - lpd_mu0)

    # Posterior over g given μ model.
    log_post_unnorm = logp_g + lpd_mu
    # Convert the trapezoid integration into discrete masses per grid point.
    w = trapz_weights(xs)
    logw = np.log(np.clip(w, 1e-300, np.inf))
    log_mass = log_post_unnorm + logw - logZ_mu
    mass = np.exp(np.clip(log_mass, -800.0, 800.0))
    mass = mass / float(np.sum(mass)) if float(np.sum(mass)) > 0 else mass
    g_mean = float(np.sum(mass * xs))
    g_var = float(np.sum(mass * (xs - g_mean) ** 2))
    g_std = float(np.sqrt(max(g_var, 0.0)))

    # Edge mass (use trapezoid endpoint masses as a boundary diagnostic).
    edge_mass_low = float(mass[0]) if mass.size > 0 else float("nan")
    edge_mass_high = float(mass[-1]) if mass.size > 0 else float("nan")
    best_idx = int(np.nanargmax(log_post_unnorm))
    best_is_edge = bool(best_idx in (0, int(xs.size - 1)))

    g_marg = {
        "prior": {
            "type": str(args.g_prior_type),
            "mu": float(args.g_prior_mu),
            "sigma": float(args.g_prior_sigma),
            "uniform_min": float(args.g_prior_uniform_min),
            "uniform_max": float(args.g_prior_uniform_max),
        },
        "logZ_mu": float(logZ_mu),
        "logBF_mu_over_gr": float(logBF_mu_over_gr),
        "logBF_mu_over_mu0": float(logBF_mu_over_mu0),
        "posterior_mean": float(g_mean),
        "posterior_std": float(g_std),
        "edge_mass_low": float(edge_mass_low),
        "edge_mass_high": float(edge_mass_high),
        "best_g_on_grid": float(xs[best_idx]),
        "best_is_edge": bool(best_is_edge),
    }

    out = {
        "run": {
            "timestamp_utc": utc_tag(),
            "axis": {"name": axis.name, "frame": axis.frame, "lon_deg": axis.lon_deg, "lat_deg": axis.lat_deg},
            "axis_icrs_unitvec": [float(x) for x in axis_icrs.tolist()],
            "pe_nside": int(pe_nside),
            "p_credible": float(p_credible),
            "n_events": int(n_ev),
            "n_draws": int(n_draws),
            "nproc_requested": int(nproc_req),
            "nproc_effective_events": int(nproc_eff),
            "cache_outdir": str(base_out),
            "run_label": str(args.run_label),
        },
        "g_grid": [float(g) for g in g_grid.tolist()],
        "curve": curve,
        "quad_fit": fit,
        "g_marginalization": g_marg,
        "selection": {
            "disabled": bool(args.disable_selection),
            "injections_hdf": injections_hdf_used if injections_hdf_used is not None else str(args.injections_hdf),
            "ifar_threshold_yr": float(args.selection_ifar_thresh_yr),
            "meta_from_summary": sel_meta,
        },
        "mixture": {
            "disabled_missing": bool(args.disable_missing),
            "f_miss_mode": str(args.f_miss_mode),
            "f_miss_ref_used": float(f_miss_ref),
            "f_miss_meta_from_summary": f_meta,
            "missing_pre_from_summary": miss_pre_meta,
        },
        "events": events_meta,
    }
    (outdir / "fixed_axis_gscan_full.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {outdir/'fixed_axis_gscan_full.json'}")

    if args.make_plot:
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(xs, ys, marker="o")
        plt.axvline(0.0, color="k", alpha=0.2, linewidth=1)
        plt.xlabel("g (anisotropy strength)")
        plt.ylabel("ΔLPD_total(g) = LPD_mu(g) − LPD_gr")
        plt.title(f"Fixed-axis g scan ({axis.name}), FULL")
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
        fig_path = outdir / "fixed_axis_gscan_full.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()
        print(f"Wrote {fig_path}")

    elapsed_s = time.monotonic() - t0
    _write_progress_json(
        outdir,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "done",
            "axis": axis.name,
            "n_events": int(n_ev),
            "done_events": int(n_ev),
            "elapsed_s": float(elapsed_s),
        },
    )
    print(f"[gscan_full] DONE axis={axis.name} elapsed={_format_duration_s(elapsed_s)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
