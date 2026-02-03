#!/usr/bin/env python3
"""
Fast "is there a dipole?" proxy for the EntropyPaper dark-siren preference.

We use the per-event score table (ΔLPD_i = HE − GR) from a production run bundle (2-1-c-m)
and each event's public GWTC-3 sky posterior p(n) from the multi-order HEALPix FITS.

Proxy construction:
  - For each event i, compute the sky posterior mean unit vector:
      m_i = ∫ n p_i(n) dΩ / ∫ p_i(n) dΩ   (in ICRS Cartesian)
  - Fit a simple dipole regression:
      ΔLPD_i ≈ a + d · m_i
    where d is a 3-vector. The best-fit axis is d/|d|.

Significance:
  - Permutation test: shuffle ΔLPD_i across events, refit, and compare |d| to observed.

This does NOT refit mu(A) and does not establish a physical anisotropic model.
It is a quick diagnostic for whether the *per-event preference* is direction-dependent.
"""

from __future__ import annotations

import argparse
import json
import math
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


def vec_icrs_to_lb(vec: np.ndarray) -> tuple[float, float]:
    """Return (l_deg, b_deg) from an ICRS Cartesian vector."""
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    sc = SkyCoord(x=v[0] * u.one, y=v[1] * u.one, z=v[2] * u.one, representation_type="cartesian", frame="icrs")
    gal = sc.galactic
    return float(gal.l.deg % 360.0), float(gal.b.deg)


def uniq_to_order_ipix(uniq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode IVOA HEALPix UNIQ values into (order, ipix_nest)."""
    u = np.asarray(uniq, dtype=np.int64)
    if u.ndim != 1:
        u = u.reshape(-1)
    if np.any(u <= 0):
        raise ValueError("UNIQ contains non-positive entries")
    log2u = np.floor(np.log2(u)).astype(np.int64)
    order = ((log2u - 2) // 2).astype(np.int64)
    base = (np.int64(1) << (2 * order + 2)).astype(np.int64)
    ipix = u - base
    if np.any(ipix < 0):
        raise ValueError("UNIQ decode produced negative ipix (unexpected)")
    return order, ipix


def resolve_skymap_path(path_str: str, *, skymap_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    cand = skymap_dir / p
    if cand.exists():
        return cand
    cand = skymap_dir / p.name
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Cannot resolve skymap path {path_str!r} under skymap_dir={str(skymap_dir)!r}")


def compute_mean_vec_for_skymap(*, skymap_fits: Path) -> tuple[np.ndarray, float]:
    """Return (m_vec, total_prob_mass) for the sky posterior in ICRS Cartesian."""
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
    s = np.zeros(3, dtype=float)
    for ord_val in np.unique(order):
        mask = order == ord_val
        nside = int(1 << int(ord_val))
        x, y, z = hp.pix2vec(nside, ipix[mask], nest=True)
        area = float(hp.nside2pixarea(nside))
        prob = probdensity[mask] * area
        total += float(np.sum(prob))
        s[0] += float(np.sum(prob * x))
        s[1] += float(np.sum(prob * y))
        s[2] += float(np.sum(prob * z))

    if total <= 0.0 or not np.isfinite(total):
        raise ValueError(f"total probability mass for {skymap_fits} is invalid: {total}")
    m = s / total
    return m, total


def fit_dipole_regression(m: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit y ≈ a + d·m by least squares; return fit pieces."""
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if m.shape != (len(y), 3):
        raise ValueError("m must be shape (N,3)")

    X = np.column_stack([np.ones(len(y), dtype=float), m])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(beta[0])
    d = np.asarray(beta[1:4], dtype=float)
    amp = float(np.linalg.norm(d))
    if amp > 0 and np.isfinite(amp):
        axis = d / amp
    else:
        axis = np.array([float("nan"), float("nan"), float("nan")], dtype=float)
    yhat = X @ beta
    resid = y - yhat
    return {
        "a": a,
        "d_vec_icrs": d.tolist(),
        "d_amp": amp,
        "axis_icrs": axis.tolist(),
        "rss": float(np.sum(resid**2)),
        "tss": float(np.sum((y - float(np.mean(y))) ** 2)),
    }


def permutation_amp_p_value(m: np.ndarray, y: np.ndarray, *, n_perm: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    obs = fit_dipole_regression(m, y)["d_amp"]
    perm = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        perm[i] = fit_dipole_regression(m, rng.permutation(y))["d_amp"]
    p = float((np.sum(perm >= obs) + 1.0) / (len(perm) + 1.0))
    return {
        "obs_d_amp": float(obs),
        "null_mean": float(np.mean(perm)) if len(perm) else float("nan"),
        "null_sd": float(np.std(perm)) if len(perm) else float("nan"),
        "p_one_sided": p,
        "n_perm": int(n_perm),
        "seed": int(seed),
    }


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--event-scores-json",
        type=str,
        default="data/dark_sirens/2-1-c-m/production_36events/event_scores_M0_start101.json",
    )
    ap.add_argument("--skymap-dir", type=str, default="data/external/zenodo_5546663/skymaps")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--make-plot", action="store_true")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    event_scores_path = Path(args.event_scores_json)
    skymap_dir = Path(args.skymap_dir)
    if not event_scores_path.exists():
        raise FileNotFoundError(f"--event-scores-json not found: {event_scores_path}")
    if not skymap_dir.exists():
        raise FileNotFoundError(f"--skymap-dir not found: {skymap_dir}")

    outdir = Path(args.outdir) if args.outdir else Path("outputs") / f"darksiren_score_dipole_{utc_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(event_scores_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("event_scores JSON must be a non-empty list")

    m_list: list[np.ndarray] = []
    y_list: list[float] = []
    n_gal: list[float] = []
    sky_area: list[float] = []

    events_out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("event_scores entries must be dicts")
        event = str(row.get("event", ""))
        skymap_path_str = str(row.get("skymap_path", ""))
        if not event or not skymap_path_str:
            raise ValueError("event_scores entry missing event/skymap_path")
        delta = float(row.get("delta_lpd", float("nan")))
        if not np.isfinite(delta):
            raise ValueError(f"delta_lpd non-finite for {event}: {delta}")

        smap = resolve_skymap_path(skymap_path_str, skymap_dir=skymap_dir)
        m, prob_total = compute_mean_vec_for_skymap(skymap_fits=smap)

        out_row = dict(row)
        out_row.update(
            {
                "skymap_resolved": str(smap),
                "mean_vec_icrs": m.tolist(),
                "prob_total": float(prob_total),
            }
        )
        events_out.append(out_row)
        m_list.append(m)
        y_list.append(delta)
        n_gal.append(float(row.get("n_gal", float("nan"))))
        sky_area.append(float(row.get("sky_area_deg2", float("nan"))))

    m_arr = np.asarray(m_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=float)

    fit = fit_dipole_regression(m_arr, y_arr)
    perm = permutation_amp_p_value(m_arr, y_arr, n_perm=int(args.n_perm), seed=int(args.seed))

    # Compare axis to a few reference directions (in Galactic).
    axis_icrs = np.asarray(fit["axis_icrs"], dtype=float)
    l_hat, b_hat = vec_icrs_to_lb(axis_icrs)
    ref_axes = {
        "cmb": AxisSpec(name="cmb", frame="galactic", lon_deg=264.021, lat_deg=48.253),
        "secrest": AxisSpec(name="secrest", frame="galactic", lon_deg=236.01, lat_deg=28.77),
    }
    angles: dict[str, float] = {}
    for k, ax in ref_axes.items():
        ref = axis_unitvec_icrs(ax)
        dot = float(np.dot(ref, axis_icrs))
        dot = float(np.clip(abs(dot), -1.0, 1.0))
        angles[k] = float(np.degrees(math.acos(dot)))

    # Extra simple diagnostics: correlation with sky-area / n_gal.
    n_gal_arr = np.asarray(n_gal, dtype=float)
    sky_area_arr = np.asarray(sky_area, dtype=float)
    diag = {
        "pearson_r_delta_vs_log10_n_gal": pearson_r(y_arr, np.log10(np.clip(n_gal_arr, 1.0, np.inf))),
        "pearson_r_delta_vs_log10_sky_area_deg2": pearson_r(y_arr, np.log10(np.clip(sky_area_arr, 1.0, np.inf))),
    }

    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "event_scores_json": str(event_scores_path),
            "skymap_dir": str(skymap_dir),
            "n_events": int(len(events_out)),
        },
        "fit": fit,
        "fit_axis_galactic": {"l_deg": l_hat, "b_deg": b_hat, "angle_to_ref_axes_deg": angles},
        "permutation_test": perm,
        "simple_diagnostics": diag,
        "events": events_out,
    }

    (outdir / "score_dipole_proxy.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    if bool(args.make_plot):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10.5, 4.0))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])

        ax0 = fig.add_subplot(gs[0, 0])
        proj = m_arr @ axis_icrs
        ax0.scatter(proj, y_arr, s=20, alpha=0.85)
        ax0.set_xlabel("Event mean sky projection onto fitted axis (m·axis)")
        ax0.set_ylabel("Per-event ΔLPD (HE − GR)")
        ax0.set_title(f"Fit axis (Gal): (l,b)=({l_hat:.1f}°, {b_hat:.1f}°)")

        ax1 = fig.add_subplot(gs[0, 1])
        rng = np.random.default_rng(int(args.seed))
        perm_amps = np.array([fit_dipole_regression(m_arr, rng.permutation(y_arr))["d_amp"] for _ in range(int(args.n_perm))])
        ax1.hist(perm_amps, bins=30, alpha=0.85, color="#4C72B0")
        ax1.axvline(fit["d_amp"], color="crimson", lw=2, label="observed")
        ax1.set_xlabel("|d| (dipole amplitude in ΔLPD units)")
        ax1.set_ylabel("count")
        ax1.set_title(f"Permutation p={perm['p_one_sided']:.3f}")
        ax1.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        fig.savefig(outdir / "score_dipole_proxy.png", dpi=180)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

