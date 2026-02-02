#!/usr/bin/env python3
"""
Reproduce an RvMP Fig.5-style CatWISE dipole-vs-faint-cut scan using the
Secrest+22 "accepted" CatWISE AGN sample and their published masking/correction
logic.

Purpose (verification task)
---------------------------
Nathan Secrest's referee-style critique (email) includes two key points:
  (1) a naive "linear"/vector-sum estimator can be biased (especially on a masked sky),
  (2) the CatWISE dipole must account for the strong ecliptic-latitude trend (WISE scan pattern),
      and Secrest+22 report stability vs W1 cut after that correction.

This script implements a self-contained subset of Secrest's released code
(`hpmap_utilities.SkyMap.fit_dipole`, `mask_zeros`, `fits2mask`, `xyfit/binxy`)
without importing dustmaps, so it runs in this repo's environment.

Method (matches Secrest pipeline)
--------------------------------
For each faint cut W1_max (using *all* sources with W1 < W1_max, not bins):
  1) Build an NSIDE=64 HEALPix counts map from the catalog (after W1cov>=80).
  2) Apply a fixed footprint mask:
       - "zero-coverage" pixels (from the full W1cov>=80 sample) plus neighbours,
       - Secrest exclusion discs (`exclude_master_revised.fits`, use==True),
       - Galactic plane mask |b| < b_cut (pixel-center latitude).
  3) Fit a linear trend of density vs |ecliptic latitude| using Secrest's binned
     regression (`xyfit` with bin size n=200), then force intercept=0.
  4) Convert the fitted slope from density units to counts-per-pixel units and
     apply the Secrest correction weight:
         w_p = 1 - p0 * (|elat|_p / count_p)
     (with safe handling for count_p=0).
  5) Fit monopole + dipole on the weighted *count* map using the same linear solve
     as `SkyMap.fit_dipole`, and estimate uncertainty via Monte Carlo:
       - Poisson-resample per-pixel counts (shot noise),
       - Gaussian-resample the ecliptic slope p0.

Optional injection mode
-----------------------
To test whether "stability vs faint cut" rules out selection gradients, an
optional synthetic selection modulation can be applied:
  select sources via  W1_eff = W1 - delta_m * cos(theta_axis)
  then apply W1_eff < W1_max.

This models an effective dipolar modulation of the faint limit by ±delta_m.
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


def skyarea_deg2() -> float:
    return float(4.0 * math.pi * (180.0 / math.pi) ** 2)


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def vec_to_lb(vec: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def binxy(xvar: np.ndarray, yvar: np.ndarray, binsize: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Secrest hpmap_utilities.binxy (bins of fixed size after sorting by x)."""
    xvar = np.asarray(xvar, dtype=float)
    yvar = np.asarray(yvar, dtype=float)
    msk = np.isfinite(xvar) & np.isfinite(yvar)
    x, y = xvar[msk], yvar[msk]
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    bins = np.arange(0, x.size, int(binsize))
    if bins.size < 2:
        raise ValueError("Not enough points for binxy with requested binsize.")
    xc = np.empty(bins.size - 1, dtype=float)
    ymean = np.empty(bins.size - 1, dtype=float)
    yse = np.empty(bins.size - 1, dtype=float)
    for i in range(bins.size - 1):
        sl = slice(bins[i], bins[i + 1])
        xc[i] = float(np.mean(x[sl]))
        yi = y[sl]
        ymean[i] = float(np.mean(yi))
        yse[i] = float(np.std(yi, ddof=1) / math.sqrt(binsize))
    # Avoid zero standard errors which break weighted polyfit.
    yse = np.where(yse <= 0.0, np.nanmin(yse[yse > 0.0]) if np.any(yse > 0.0) else 1.0, yse)
    return xc, ymean, yse


@dataclass(frozen=True)
class SecrestMask:
    mask: np.ndarray  # True = masked
    seen: np.ndarray  # True = unmasked
    xyz_seen: np.ndarray  # shape (n_seen, 3)
    A: np.ndarray  # 4x4 coefficient matrix


def build_secrest_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> SecrestMask:
    """Implements SkyMap.mask_zeros + fits2mask + galactic plane cut, then precomputes A."""
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    # --- mask_zeros(tbl) on the W1cov>=80 full sample ---
    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        indices = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            indices[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        # Match Secrest behaviour (including -1 neighbours which index the last pixel).
        mask[indices] = True

    # --- fits2mask(exclude_master_revised.fits) ---
    if exclude_mask_fits:
        from astropy.table import Table
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        tmask = Table.read(exclude_mask_fits)
        if "use" in tmask.colnames:
            tmask = tmask[tmask["use"] == True]  # noqa: E712
        if len(tmask):
            sc = SkyCoord(tmask["ra"], tmask["dec"], unit=u.deg, frame="icrs").galactic
            radius = np.deg2rad(np.asarray(tmask["radius"], dtype=float))
            for lon, lat, rad in zip(sc.l.deg, sc.b.deg, radius, strict=True):
                theta = np.deg2rad(90.0 - float(lat))
                phi = np.deg2rad(float(lon))
                vec = hp.ang2vec(theta, phi)
                disc = hp.query_disc(nside=int(nside), vec=vec, radius=float(rad), inclusive=True, nest=False)
                mask[disc] = True

    # --- Galactic plane cut on pixel centers ---
    lon_pix, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    seen = ~mask
    xyz = np.vstack(hp.pix2vec(int(nside), np.arange(npix), nest=False)).T  # (npix, 3)
    xyz_seen = xyz[seen]

    # set_coefficient_matrix() from hpmap_utilities.SkyMap
    x, y, z = xyz_seen.T
    A = np.zeros((4, 4), dtype=float)
    A[0, 0] = xyz_seen.shape[0]
    A[1, 0] = float(np.sum(x))
    A[2, 0] = float(np.sum(y))
    A[3, 0] = float(np.sum(z))
    A[1, 1] = float(np.sum(x * x))
    A[2, 1] = float(np.sum(x * y))
    A[3, 1] = float(np.sum(x * z))
    A[2, 2] = float(np.sum(y * y))
    A[3, 2] = float(np.sum(y * z))
    A[3, 3] = float(np.sum(z * z))
    A[0, 1] = A[1, 0]
    A[0, 2] = A[2, 0]
    A[0, 3] = A[3, 0]
    A[1, 2] = A[2, 1]
    A[1, 3] = A[3, 1]
    A[2, 3] = A[3, 2]

    return SecrestMask(mask=mask, seen=seen, xyz_seen=xyz_seen, A=A)


def fit_dipole_linear_solve(A: np.ndarray, xyz_seen: np.ndarray, m_seen: np.ndarray, w_seen: np.ndarray | None) -> tuple[float, float, np.ndarray]:
    """Secrest SkyMap.fit_dipole core (returns mono, D, vec)."""
    m_fit = np.asarray(m_seen, dtype=float)
    if w_seen is not None:
        m_fit = m_fit * np.asarray(w_seen, dtype=float)
    x, y, z = xyz_seen.T
    b = np.zeros(4, dtype=float)
    b[0] = float(np.sum(m_fit))
    b[1] = float(np.sum(m_fit * x))
    b[2] = float(np.sum(m_fit * y))
    b[3] = float(np.sum(m_fit * z))
    sol = np.linalg.solve(A, b)
    mono = float(sol[0])
    vec = np.asarray(sol[1:4], dtype=float)
    norm = float(np.linalg.norm(vec))
    D = float(norm / mono) if mono != 0.0 else float("nan")
    return mono, D, vec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--mask-catalog",
        default=None,
        help=(
            "Optional separate catalog used ONLY to build the Secrest footprint mask (mask_zeros + exclude + |b| cut). "
            "Use this when the analysis catalog is a filtered/subsampled file (e.g. NVSS-removed), so mask_zeros "
            "is not spuriously triggered by missing sources."
        ),
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1-grid", default="15.5,16.6,0.05", help="start,stop,step (inclusive end).")
    ap.add_argument("--xyfit-binsize", type=int, default=200, help="Secrest xyfit bin size.")
    ap.add_argument("--nsim", type=int, default=400, help="Monte Carlo draws per W1 cut.")
    ap.add_argument("--seed", type=int, default=12345)

    # Injection: dipolar modulation of faint limit by ±delta_m.
    ap.add_argument("--inject-delta-m-mag", type=float, default=0.0)
    ap.add_argument("--inject-axis", default="cmb", help="'cmb' or 'l,b' in degrees for injection axis.")

    ap.add_argument("--make-plot", action="store_true")
    args = ap.parse_args()

    import healpy as hp
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from scipy.stats import sem

    outdir = Path(args.outdir or f"outputs/catwise_rvmp_fig5_repro_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load columns (memmap to avoid giant copies).
    with fits.open(args.catalog, memmap=True) as hdul:
        data = hdul[1].data
        w1 = np.asarray(data["w1"], dtype=float)
        w1cov = np.asarray(data["w1cov"], dtype=float)
        l = np.asarray(data["l"], dtype=float)
        b = np.asarray(data["b"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    base &= w1cov >= float(args.w1cov_min)

    # Source-to-pixel mapping (galactic lon/lat already provided).
    theta = np.deg2rad(90.0 - b[base])
    phi = np.deg2rad(l[base])
    ipix_analysis = hp.ang2pix(int(args.nside), theta, phi, nest=False)
    ipix_mask_base = ipix_analysis

    # If requested, build the mask_zeros footprint from a separate (typically full) catalog.
    if args.mask_catalog is not None:
        with fits.open(args.mask_catalog, memmap=True) as hdul:
            dmask = hdul[1].data
            w1cov_m = np.asarray(dmask["w1cov"], dtype=float)
            l_m = np.asarray(dmask["l"], dtype=float)
            b_m = np.asarray(dmask["b"], dtype=float)
        base_m = np.isfinite(w1cov_m) & np.isfinite(l_m) & np.isfinite(b_m)
        base_m &= w1cov_m >= float(args.w1cov_min)
        theta_m = np.deg2rad(90.0 - b_m[base_m])
        phi_m = np.deg2rad(l_m[base_m])
        ipix_mask_base = hp.ang2pix(int(args.nside), theta_m, phi_m, nest=False)

    # Optional injection effective faint-limit modulation.
    inject_delta_m = float(args.inject_delta_m_mag)
    if args.inject_axis.strip().lower() == "cmb":
        axis_l, axis_b = 264.021, 48.253
    else:
        parts = args.inject_axis.split(",")
        if len(parts) != 2:
            raise SystemExit("--inject-axis must be 'cmb' or 'l,b'")
        axis_l, axis_b = float(parts[0]), float(parts[1])

    if inject_delta_m != 0.0:
        n_src = lb_to_unitvec(l[base], b[base])
        n_axis = lb_to_unitvec(np.array([axis_l]), np.array([axis_b]))[0]
        cos_theta = n_src @ n_axis
        w1_eff = w1[base] - inject_delta_m * cos_theta
    else:
        w1_eff = w1[base].copy()

    # Sort by the (possibly injected) effective magnitude so all cuts can be updated incrementally.
    order = np.argsort(w1_eff)
    w1_eff_sorted = w1_eff[order]
    ipix_sorted = np.asarray(ipix_analysis, dtype=np.int64)[order]

    # Parse grid.
    w1_start, w1_stop, w1_step = (float(x) for x in args.w1_grid.split(","))
    n_steps = int(round((w1_stop - w1_start) / w1_step)) + 1
    cuts = [w1_start + i * w1_step for i in range(n_steps)]

    # Build fixed Secrest footprint mask and A-matrix.
    secrest_mask = build_secrest_mask(
        nside=int(args.nside),
        ipix_base=ipix_mask_base,
        exclude_mask_fits=args.exclude_mask_fits,
        b_cut_deg=float(args.b_cut),
    )
    seen = secrest_mask.seen

    # Pixel-center ecliptic latitude (for |elat| trend).
    npix = hp.nside2npix(int(args.nside))
    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)  # galactic l,b
    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    elat_abs = np.abs(sc_pix.barycentricmeanecliptic.lat.deg).astype(float)

    scl = npix / skyarea_deg2()  # pix/deg^2

    rng = np.random.default_rng(int(args.seed))
    counts = np.zeros(npix, dtype=np.int64)
    cursor = 0

    rows: list[dict[str, Any]] = []

    for w1_cut in cuts:
        # Update cumulative counts for this cut.
        nxt = int(np.searchsorted(w1_eff_sorted, float(w1_cut), side="left"))
        if nxt > cursor:
            delta = ipix_sorted[cursor:nxt]
            counts += np.bincount(delta, minlength=npix).astype(np.int64)
            cursor = nxt

        count_seen = counts[seen].astype(float)
        density_seen = count_seen * scl
        abselat_seen = elat_abs[seen]

        # Secrest xyfit: density vs |elat|, deg=1, binsize=200. Then force intercept=0.
        x, y, s = binxy(abselat_seen, density_seen, binsize=int(args.xyfit_binsize))
        p, pcov = np.polyfit(x, y, deg=1, w=1.0 / s, cov=True)
        p = np.asarray(p, dtype=float)
        p[-1] = 0.0
        p0 = float(p[0] / scl)  # deg^-2 -> pix^-1 (Secrest convention)
        p0_sigma = float(math.sqrt(float(pcov[0, 0])) / scl) if np.isfinite(pcov[0, 0]) and pcov[0, 0] >= 0 else float("nan")

        # Build correction weight w = 1 - p0 * (|elat|/count). Safe for count=0.
        abselat_over_count = np.divide(abselat_seen, count_seen, out=np.zeros_like(abselat_seen), where=count_seen != 0.0)
        w_seen = 1.0 - p0 * abselat_over_count

        # Point estimate.
        mono_hat, D_hat, vec_hat = fit_dipole_linear_solve(secrest_mask.A, secrest_mask.xyz_seen, count_seen, w_seen)
        l_hat, b_hat = vec_to_lb(vec_hat)

        # Monte Carlo uncertainty: Poisson count noise + Gaussian p0.
        Ds = np.empty(int(args.nsim), dtype=float)
        ls = np.empty(int(args.nsim), dtype=float)
        bs = np.empty(int(args.nsim), dtype=float)
        for i in range(int(args.nsim)):
            # Shot noise on per-pixel counts.
            m_seen = rng.poisson(count_seen)
            # Slope uncertainty (small) – keep deterministic even if p0_sigma is nan.
            p0_i = p0 if not np.isfinite(p0_sigma) else float(p0 + rng.normal(0.0, p0_sigma))
            w_i = 1.0 - p0_i * abselat_over_count
            mono_i, D_i, vec_i = fit_dipole_linear_solve(secrest_mask.A, secrest_mask.xyz_seen, m_seen, w_i)
            l_i, b_i = vec_to_lb(vec_i)
            Ds[i], ls[i], bs[i] = D_i, l_i, b_i

        def pct(a: np.ndarray, q: float) -> float:
            return float(np.nanpercentile(a, q))

        rows.append(
            {
                "w1_cut": float(w1_cut),
                "N_total": int(counts.sum()),
                "N_seen": int(np.sum(counts[seen])),
                "p0_count_per_pix_per_deg": p0,
                "p0_sigma": p0_sigma,
                "dipole": {
                    "D_hat": D_hat,
                    "l_hat_deg": l_hat,
                    "b_hat_deg": b_hat,
                    "D_p16": pct(Ds, 16),
                    "D_p50": pct(Ds, 50),
                    "D_p84": pct(Ds, 84),
                    "l_p16": pct(ls, 16),
                    "l_p50": pct(ls, 50),
                    "l_p84": pct(ls, 84),
                    "b_p16": pct(bs, 16),
                    "b_p50": pct(bs, 50),
                    "b_p84": pct(bs, 84),
                },
            }
        )

    payload = {
        "meta": {
            "catalog": str(args.catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": int(args.nside),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_grid": args.w1_grid,
            "xyfit_binsize": int(args.xyfit_binsize),
            "nsim": int(args.nsim),
            "seed": int(args.seed),
            "inject_delta_m_mag": inject_delta_m,
            "inject_axis_lb": [axis_l, axis_b],
        },
        "rows": rows,
    }
    (outdir / "rvmp_fig5_repro.json").write_text(json.dumps(payload, indent=2))

    if args.make_plot:
        import matplotlib.pyplot as plt

        w1c = np.array([r["w1_cut"] for r in rows], dtype=float)
        D50 = np.array([r["dipole"]["D_p50"] for r in rows], dtype=float)
        D16 = np.array([r["dipole"]["D_p16"] for r in rows], dtype=float)
        D84 = np.array([r["dipole"]["D_p84"] for r in rows], dtype=float)
        l50 = np.array([r["dipole"]["l_p50"] for r in rows], dtype=float)
        l16 = np.array([r["dipole"]["l_p16"] for r in rows], dtype=float)
        l84 = np.array([r["dipole"]["l_p84"] for r in rows], dtype=float)
        b50 = np.array([r["dipole"]["b_p50"] for r in rows], dtype=float)
        b16 = np.array([r["dipole"]["b_p16"] for r in rows], dtype=float)
        b84 = np.array([r["dipole"]["b_p84"] for r in rows], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)
        ax = axes[0]
        ax.fill_between(w1c, D16, D84, alpha=0.25, color="C0", lw=0)
        ax.plot(w1c, D50, color="C0")
        ax.set_ylabel("Dipole amplitude D")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.fill_between(w1c, l16, l84, alpha=0.25, color="C1", lw=0)
        ax.plot(w1c, l50, color="C1")
        ax.axhline(264.021, color="k", ls="--", lw=1)
        ax.set_ylabel("l [deg]")
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.fill_between(w1c, b16, b84, alpha=0.25, color="C2", lw=0)
        ax.plot(w1c, b50, color="C2")
        ax.axhline(48.253, color="k", ls="--", lw=1)
        ax.set_ylabel("b [deg]")
        ax.set_xlabel("W1_max (effective, if injected)")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(outdir / "rvmp_fig5_repro.png", dpi=200)
        plt.close(fig)

    # Minimal console summary for the user.
    D_last = rows[-1]["dipole"]["D_p50"]
    l_last = rows[-1]["dipole"]["l_p50"]
    b_last = rows[-1]["dipole"]["b_p50"]
    print(f"w1_cut_max={cuts[-1]:.3f}: D~{D_last:.4g}, (l,b)~({l_last:.1f},{b_last:+.1f})")
    print(f"Wrote: {outdir}/rvmp_fig5_repro.json")
    if args.make_plot:
        print(f"Wrote: {outdir}/rvmp_fig5_repro.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
