#!/usr/bin/env python3
"""
Secrest-style systematics audit for the CatWISE quasar dipole.

This script mirrors the key "referee-expected" checks in Secrest+21/22:
  - build an NSIDE map with the Secrest footprint mask (mask_zeros + exclude discs + |b| cut)
  - fit and remove the ecliptic-latitude density trend (linear; force intercept=0)
  - subtract the best-fit dipole from the corrected density map
  - quantify residual trends versus common systematics via binned chi2/dof

It is intended to be run both on:
  - the default CatWISE accepted sample, and
  - a filtered/subsampled variant (e.g. NVSS-removed/homogenized),
using --mask-catalog to ensure the footprint mask is defined independently
of the analysis catalog.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def skyarea_deg2() -> float:
    return float(4.0 * math.pi * (180.0 / math.pi) ** 2)


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
    """Secrest hpmap_utilities.binxy: fixed-size bins after sorting by x."""
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
    # Avoid zero standard errors which break weighted fits.
    if np.any(yse <= 0.0):
        floor = float(np.nanmin(yse[yse > 0.0])) if np.any(yse > 0.0) else 1.0
        yse = np.where(yse <= 0.0, floor, yse)
    return xc, ymean, yse


def get_chi2(y: np.ndarray, yerr: np.ndarray, y_fit: np.ndarray, *, k: int) -> tuple[float, int]:
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    chi2 = float(np.sum((y - y_fit) ** 2 / (yerr**2)))
    dof = int(y.size) - int(k)
    return chi2, dof


@dataclass(frozen=True)
class FitSummary:
    chi2: float
    dof: int
    chi2_over_dof: float


def fit_constant_via_bins(xvar: np.ndarray, yvar: np.ndarray, *, binsize: int) -> tuple[FitSummary, dict]:
    """Secrest xyfit(..., deg=0) analogue: constant model vs binned means."""
    xc, yb, yse = binxy(xvar, yvar, binsize)
    mu = float(np.mean(yvar))
    fx = np.full_like(yb, mu)
    chi2, dof = get_chi2(yb, yse, fx, k=1)
    rchi2 = float(chi2 / dof) if dof > 0 else float("nan")
    return FitSummary(chi2=chi2, dof=dof, chi2_over_dof=rchi2), {
        "mu": mu,
        "n_bins": int(yb.size),
        "x_bin_mean": xc.tolist(),
        "y_bin_mean": yb.tolist(),
        "y_bin_se": yse.tolist(),
    }


def fit_linear_trend_density_vs_abselat(abselat_seen: np.ndarray, density_seen: np.ndarray, *, binsize: int) -> tuple[float, float]:
    """Fit density ~ p0 * abselat + p1, then force p1=0; return (p0, sigma_p0)."""
    xc, yb, yse = binxy(abselat_seen, density_seen, binsize)
    p, pcov = np.polyfit(xc, yb, deg=1, w=1.0 / yse, cov=True)
    perr = np.sqrt(np.diag(pcov))
    p0 = float(p[0])
    p0_sigma = float(perr[0])
    return p0, p0_sigma


def build_secrest_mask(
    *,
    nside: int,
    ipix_mask_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> np.ndarray:
    """mask_zeros + exclusion discs + galactic plane cut (pixel centers). Returns mask bool array."""
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    # mask_zeros(tbl) using the base selection (typically W1cov>=80, no W1 cut)
    cnt_base = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        indices = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            indices[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        # Match Secrest behaviour (includes -1 neighbour indexing last pixel).
        mask[indices] = True

    # exclusion discs (exclude_master_revised.fits)
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

    # galactic plane cut (pixel centers)
    lon_pix, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return mask


def fit_dipole_map_linear(mask_seen: np.ndarray, xyz_seen: np.ndarray, m_seen: np.ndarray, w_seen: np.ndarray | None) -> tuple[float, float, np.ndarray]:
    """Secrest SkyMap.fit_dipole core (mono, D, vec) on seen pixels."""
    xyz_seen = np.asarray(xyz_seen, dtype=float)
    m_fit = np.asarray(m_seen, dtype=float)
    if w_seen is not None:
        m_fit = m_fit * np.asarray(w_seen, dtype=float)

    x, y, z = xyz_seen.T
    # Coefficient matrix A depends only on geometry.
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

    b = np.zeros(4, dtype=float)
    b[0] = float(np.sum(m_fit))
    b[1] = float(np.sum(m_fit * x))
    b[2] = float(np.sum(m_fit * y))
    b[3] = float(np.sum(m_fit * z))

    sol = np.linalg.solve(A, b)
    mono = float(sol[0])
    vec = np.asarray(sol[1:4], dtype=float)
    D = float(np.linalg.norm(vec) / mono) if mono != 0.0 else float("nan")
    return mono, D, vec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", required=True, help="Analysis catalog FITS (CatWISE accepted or filtered).")
    ap.add_argument(
        "--mask-catalog",
        default=None,
        help="Catalog used ONLY to build footprint mask and per-pixel mean templates (recommended for filtered catalogs).",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1-cut", type=float, default=16.4)
    ap.add_argument("--xyfit-binsize", type=int, default=200)
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    outdir = Path(args.outdir or f"outputs/secrest_systematics_audit_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    # --- Load mask-catalog (for footprint + mean templates) ---
    mask_catalog = str(args.mask_catalog) if args.mask_catalog else str(args.catalog)
    with fits.open(mask_catalog, memmap=True) as hdul:
        dm = hdul[1].data
        w1cov_m = np.asarray(dm["w1cov"], dtype=float)
        l_m = np.asarray(dm["l"], dtype=float)
        b_m = np.asarray(dm["b"], dtype=float)
        base_m = np.isfinite(w1cov_m) & np.isfinite(l_m) & np.isfinite(b_m)
        base_m &= w1cov_m >= float(args.w1cov_min)
        th_m = np.deg2rad(90.0 - b_m[base_m])
        ph_m = np.deg2rad(l_m[base_m])
        ipix_mask_base = hp.ang2pix(int(args.nside), th_m, ph_m, nest=False).astype(np.int64)

        ebv_m = np.asarray(dm["ebv"], dtype=float) if "ebv" in dm.names else None
        Tb_m = np.asarray(dm["Tb"], dtype=float) if "Tb" in dm.names else None

        ebv_base = ebv_m[base_m] if ebv_m is not None else None
        Tb_base = Tb_m[base_m] if Tb_m is not None else None

    npix = hp.nside2npix(int(args.nside))
    scl = float(npix / skyarea_deg2())  # pix / deg^2 (Secrest SkyMap.scl)

    mask = build_secrest_mask(
        nside=int(args.nside),
        ipix_mask_base=ipix_mask_base,
        exclude_mask_fits=str(args.exclude_mask_fits) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )
    seen = ~mask

    # Pixel-center coordinate maps.
    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)  # galactic l,b
    sc_gal = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    elon_deg = sc_gal.barycentricmeanecliptic.lon.deg.astype(float)
    elat_deg = sc_gal.barycentricmeanecliptic.lat.deg.astype(float)
    dec_deg = sc_gal.icrs.dec.deg.astype(float)
    sgb_deg = sc_gal.supergalactic.sgb.deg.astype(float)
    absb = np.abs(lat_pix.astype(float))
    abssgb = np.abs(sgb_deg)
    abselat = np.abs(elat_deg)

    # Pixel xyz for dipole templates.
    xyz = np.vstack(hp.pix2vec(int(args.nside), np.arange(npix), nest=False)).T
    xyz_seen = xyz[seen]

    # Per-pixel mean EBV/Tb maps (from mask-catalog base selection).
    cnt_base = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), minlength=npix).astype(float)
    if ebv_base is not None:
        sum_ebv = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), weights=np.asarray(ebv_base, dtype=float), minlength=npix).astype(float)
        ebv_mean = np.divide(sum_ebv, cnt_base, out=np.zeros_like(sum_ebv), where=cnt_base != 0.0)
    else:
        ebv_mean = np.zeros(npix, dtype=float)
    if Tb_base is not None:
        sum_Tb = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), weights=np.asarray(Tb_base, dtype=float), minlength=npix).astype(float)
        Tb_mean = np.divide(sum_Tb, cnt_base, out=np.zeros_like(sum_Tb), where=cnt_base != 0.0)
    else:
        Tb_mean = np.zeros(npix, dtype=float)

    # --- Load analysis catalog and build counts map at W1 cut ---
    with fits.open(str(args.catalog), memmap=True) as hdul:
        d = hdul[1].data
        w1 = np.asarray(d["w1"], dtype=float)
        w1cov = np.asarray(d["w1cov"], dtype=float)
        l = np.asarray(d["l"], dtype=float)
        b = np.asarray(d["b"], dtype=float)
        ok = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
        ok &= w1cov >= float(args.w1cov_min)
        ok &= w1 < float(args.w1_cut)

        th = np.deg2rad(90.0 - b[ok])
        ph = np.deg2rad(l[ok])
        ipix = hp.ang2pix(int(args.nside), th, ph, nest=False).astype(np.int64)

    counts = np.bincount(ipix, minlength=npix).astype(float)
    density = counts * scl  # counts * (pix/deg^2) => deg^-2

    # --- Ecliptic-latitude density trend fit (force intercept=0) ---
    p0_density, p0_density_sigma = fit_linear_trend_density_vs_abselat(
        abselat[seen],
        density[seen],
        binsize=int(args.xyfit_binsize),
    )
    density_corr = density - p0_density * abselat

    # --- Dipole fit on corrected density (for residual systematics diagnostics) ---
    mono_den, d_den, vec_den = fit_dipole_map_linear(seen, xyz_seen, density_corr[seen], w_seen=None)
    vecu_den = vec_den / float(np.linalg.norm(vec_den))
    dipole_map_den = mono_den * (1.0 + float(d_den) * (xyz @ vecu_den))
    density_residual = density_corr - dipole_map_den

    # --- Final Secrest-style dipole on counts with ecliptic weight ---
    p0_count = p0_density / scl  # (deg^-2 / deg) * (deg^2/pix) => pix^-1 / deg
    p0_count_sigma = p0_density_sigma / scl
    with np.errstate(divide="ignore", invalid="ignore"):
        abselat_over_count = np.divide(abselat, counts, out=np.zeros_like(abselat), where=counts != 0.0)
    w = np.ones(npix, dtype=float)
    w[seen] = 1.0 - p0_count * abselat_over_count[seen]
    mono_cnt, d_cnt, vec_cnt = fit_dipole_map_linear(seen, xyz_seen, counts[seen], w_seen=w[seen])
    l_cnt, b_cnt = vec_to_lb(vec_cnt)

    # --- Systematics residual tests (Secrest 1_mk_mask.py style) ---
    systematics = {
        "ebv_mean": ebv_mean,
        "Tb_mean": Tb_mean,
        "dec_deg": dec_deg,
        "elat_deg": elat_deg,
        "b_deg": lat_pix.astype(float),
        "absb_deg": absb,
        "sgb_deg": sgb_deg,
        "abssgb_deg": abssgb,
        "elon_deg": elon_deg,
        "abselat_deg": abselat,
    }
    sys_results: dict[str, dict] = {}
    diag_blobs: dict[str, dict] = {}
    for key, xmap in systematics.items():
        fit, diag = fit_constant_via_bins(xmap[seen], density_residual[seen], binsize=int(args.xyfit_binsize))
        sys_results[key] = {
            "chi2": fit.chi2,
            "dof": fit.dof,
            "chi2_over_dof": fit.chi2_over_dof,
        }
        diag_blobs[key] = diag

    out = {
        "meta": {
            "catalog": str(args.catalog),
            "mask_catalog": str(mask_catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": int(args.nside),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_cut": float(args.w1_cut),
            "xyfit_binsize": int(args.xyfit_binsize),
        },
        "mask": {
            "npix": int(npix),
            "n_seen": int(np.sum(seen)),
            "f_sky": float(np.mean(seen)),
        },
        "ecliptic_trend_fit": {
            "p0_density_deg2_per_deg": float(p0_density),
            "p0_density_sigma": float(p0_density_sigma),
            "p0_count_per_pix_per_deg": float(p0_count),
            "p0_count_sigma": float(p0_count_sigma),
        },
        "dipole_on_density_corrected": {
            "mono": float(mono_den),
            "D_hat": float(d_den),
            "l_hat_deg": float(vec_to_lb(vec_den)[0]),
            "b_hat_deg": float(vec_to_lb(vec_den)[1]),
        },
        "dipole_on_counts_weighted": {
            "mono": float(mono_cnt),
            "D_hat": float(d_cnt),
            "l_hat_deg": float(l_cnt),
            "b_hat_deg": float(b_cnt),
        },
        "systematics_chi2": sys_results,
        "diagnostics": diag_blobs,
    }

    (outdir / "secrest_systematics_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    if args.make_plots:
        # Multi-panel diagnostic plot (subset).
        keys_plot = ["ebv_mean", "Tb_mean", "dec_deg", "elat_deg", "b_deg", "absb_deg", "sgb_deg", "abssgb_deg"]
        fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.5), sharey=True)
        axes = axes.ravel()
        for ax, key in zip(axes, keys_plot, strict=True):
            diag = diag_blobs[key]
            x = np.array(diag["x_bin_mean"], dtype=float)
            y = np.array(diag["y_bin_mean"], dtype=float)
            ye = np.array(diag["y_bin_se"], dtype=float)
            mu = float(diag["mu"])
            ax.errorbar(x, y, yerr=ye, ls="none", marker=".", markersize=4)
            ax.axhline(mu, color="k", lw=1, ls="--")
            ax.set_title(f"{key}\\n$\\chi^2/\\nu$={sys_results[key]['chi2_over_dof']:.2f}", fontsize=10)
            ax.grid(alpha=0.25)
        fig.suptitle(f"Secrest-style residual systematics audit (W1<{float(args.w1_cut):.2f})", fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / "figures" / "residual_systematics_grid.png", dpi=180)
        plt.close(fig)

    print(f"Wrote {outdir/'secrest_systematics_audit.json'}")
    if args.make_plots:
        print(f"Wrote {outdir/'figures'/'residual_systematics_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

