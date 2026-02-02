#!/usr/bin/env python3
"""
Build a CatWISE "no-NVSS" sample following Secrest+22 `wise/code/2_rm_nvss.py`.

Why
---
Our Secrest-style reproduction shows that the CatWISE dipole direction drifts
with the faint cut. RvMP (2025) summarizes the CatWISE dipole as stable near the
CMB direction after marginalizing the ecliptic trend. A plausible difference is
the optional NVSS cross-match removal/homogenization step in Secrest+22.

This script constructs the Secrest "WISE sample with NVSS sources removed" and
the additional homogenization-by-random-removal in WISE-only regions.

Outputs
-------
Writes a compact FITS catalog containing (source_id, l, b, w1, w1cov) for the
post-removal sample, under `--outdir` (default: outputs/...).

Notes
-----
This is an internal audit tool:
  - Uses the same Secrest footprint mask logic (mask_zeros + exclude discs + |b| cut).
  - Builds an NVSS sample with Secrest-like masking (mask_zeros + artifacts discs
    + Tb cut) and flux cut S>=Scut.
  - Removes CatWISE sources that match those NVSS IDs (via the published match file).
  - Homogenizes by removing additional random CatWISE sources in WISE-only pixels,
    drawing the removal counts from the distribution of NVSS-match removals in
    WISE∩NVSS pixels.
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


@dataclass(frozen=True)
class Mask:
    mask: np.ndarray  # True=masked
    seen: np.ndarray  # True=unmasked


def build_mask_zeros(*, nside: int, ipix: np.ndarray) -> np.ndarray:
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    cnt = np.bincount(np.asarray(ipix, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt == 0)[0]
    mask = np.zeros(npix, dtype=bool)
    if idx0.size:
        indices = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            indices[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        # Match Secrest behaviour (includes -1 neighbour indexing last pixel).
        mask[indices] = True
    return mask


def apply_disc_mask(mask: np.ndarray, *, nside: int, fits_path: str, use_key: str | None) -> None:
    import healpy as hp
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    t = Table.read(fits_path)
    if use_key and use_key in t.colnames:
        t = t[t[use_key] == True]  # noqa: E712
    if not len(t):
        return
    sc = SkyCoord(t["ra"], t["dec"], unit=u.deg, frame="icrs").galactic
    radius = np.deg2rad(np.asarray(t["radius"], dtype=float))
    for lon, lat, rad in zip(sc.l.deg, sc.b.deg, radius, strict=True):
        theta = np.deg2rad(90.0 - float(lat))
        phi = np.deg2rad(float(lon))
        vec = hp.ang2vec(theta, phi)
        disc = hp.query_disc(nside=int(nside), vec=vec, radius=float(rad), inclusive=True, nest=False)
        mask[disc] = True


def build_catwise_mask(*, nside: int, ipix_base: np.ndarray, exclude_mask_fits: str, b_cut_deg: float) -> Mask:
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = build_mask_zeros(nside=int(nside), ipix=ipix_base)
    apply_disc_mask(mask, nside=int(nside), fits_path=exclude_mask_fits, use_key="use")
    _, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)
    return Mask(mask=mask, seen=~mask)


def compute_nvss_masks(
    *,
    nside: int,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    tb_cut_K: float,
    artifacts_fits: str,
    chunk: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (ipix64, mask_total) for the full NVSS catalog (no flux cut).

    mask_total includes:
      - mask_zeros + neighbours
      - nvss artifacts discs
      - Tb cut (pixel-mean Tb > tb_cut_K), matching the intent of SkyMap.mask_Tcut
    """
    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    npix = hp.nside2npix(int(nside))
    ipix64 = np.empty(ra_deg.size, dtype=np.int64)
    cnt = np.zeros(npix, dtype=np.int64)
    tb_sum = np.zeros(npix, dtype=float)

    # Haslam 408 MHz map used by Secrest (bundled in their repo).
    # We use the same file path relative to the extracted hpmap_utilities.py.
    haslam = hp.read_map(
        "data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/reference/haslam408_dsds_Remazeilles2014_512.fits",
        verbose=False,
    )

    for start in range(0, ra_deg.size, int(chunk)):
        sl = slice(start, min(ra_deg.size, start + int(chunk)))
        sc = SkyCoord(ra_deg[sl] * u.deg, dec_deg[sl] * u.deg, frame="icrs").galactic
        l = sc.l.deg.astype(float)
        b = sc.b.deg.astype(float)
        theta = np.deg2rad(90.0 - b)
        phi = np.deg2rad(l)
        ip = hp.ang2pix(int(nside), theta, phi, nest=False).astype(np.int64)
        ipix64[sl] = ip
        cnt += np.bincount(ip, minlength=npix).astype(np.int64)

        # Tb per source (Haslam nside=512)
        ip512 = hp.ang2pix(512, theta, phi, nest=False)
        tb = haslam[ip512].astype(float)
        tb_sum += np.bincount(ip, weights=tb, minlength=npix).astype(float)

    mask = build_mask_zeros(nside=int(nside), ipix=ipix64)
    apply_disc_mask(mask, nside=int(nside), fits_path=artifacts_fits, use_key=None)

    tb_mean = np.divide(tb_sum, cnt, out=np.zeros_like(tb_sum), where=cnt != 0)
    mask |= tb_mean > float(tb_cut_K)

    return ipix64, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nside", type=int, default=64)

    ap.add_argument(
        "--catwise",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cut-max", type=float, default=16.5, help="Secrest W1 cut used to define the WISE sample.")

    ap.add_argument(
        "--nvss",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/nvss/reference/NVSS.fit",
    )
    ap.add_argument(
        "--nvss-artifacts",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/nvss/reference/nvss_artifacts.fits",
    )
    ap.add_argument(
        "--nvss-catwise-match",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/nvss/reference/NVSS_CatWISE2020_40arcsec_best_symmetric.fits",
    )
    ap.add_argument("--nvss-scut-mjy", type=float, default=10.0)
    ap.add_argument("--nvss-tbcut-K", type=float, default=50.0)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-homogenize", action="store_true", help="Only remove NVSS matches; skip extra random removals.")
    args = ap.parse_args()

    import healpy as hp
    from astropy.io import fits
    from astropy.table import Table

    outdir = Path(args.outdir or f"outputs/catwise_no_nvss_homog_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load CatWISE accepted sample ---
    with fits.open(args.catwise, memmap=True) as hdul:
        d = hdul[1].data
        source_id = np.asarray(d["source_id"])
        w1 = np.asarray(d["w1"], dtype=float)
        w1cov = np.asarray(d["w1cov"], dtype=float)
        l = np.asarray(d["l"], dtype=float)
        b = np.asarray(d["b"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    base &= w1cov >= float(args.w1cov_min)

    theta = np.deg2rad(90.0 - b[base])
    phi = np.deg2rad(l[base])
    ipix_base = hp.ang2pix(int(args.nside), theta, phi, nest=False).astype(np.int64)

    cat_mask = build_catwise_mask(
        nside=int(args.nside),
        ipix_base=ipix_base,
        exclude_mask_fits=args.exclude_mask_fits,
        b_cut_deg=float(args.b_cut),
    )

    # Define the WISE/CatWISE sample for NVSS removal (Secrest uses W1<w1cut_max).
    wsel = base.copy()
    wsel[base] &= w1[base] < float(args.w1cut_max)
    # Apply pixel mask.
    ipix_all = hp.ang2pix(int(args.nside), np.deg2rad(90.0 - b), np.deg2rad(l), nest=False).astype(np.int64)
    wsel &= cat_mask.seen[ipix_all]

    wise_source_id = source_id[wsel]
    wise_ipix = ipix_all[wsel].astype(np.int64)
    wise_w1 = w1[wsel]
    wise_w1cov = w1cov[wsel]
    wise_l = l[wsel]
    wise_b = b[wsel]

    # --- Load NVSS, build its mask, apply flux cut ---
    with fits.open(args.nvss, memmap=True) as hdul:
        nd = hdul[1].data
        nvss_id = np.asarray(nd["NVSS"])
        ra = np.asarray(nd["RAJ2000"], dtype=float)
        dec = np.asarray(nd["DEJ2000"], dtype=float)
        S = np.asarray(nd["S1_4"], dtype=float)  # mJy

    ipix_nvss_all, nvss_mask = compute_nvss_masks(
        nside=int(args.nside),
        ra_deg=ra,
        dec_deg=dec,
        tb_cut_K=float(args.nvss_tbcut_K),
        artifacts_fits=args.nvss_artifacts,
    )
    nvss_seen = ~nvss_mask

    nvss_keep = np.isfinite(S) & (S >= float(args.nvss_scut_mjy)) & nvss_seen[ipix_nvss_all]
    nvss_id_keep = np.asarray(nvss_id[nvss_keep])
    nvss_ipix_keep = np.asarray(ipix_nvss_all[nvss_keep], dtype=np.int64)

    # --- Cross-match: which CatWISE source_ids match NVSS IDs in the kept NVSS sample? ---
    with fits.open(args.nvss_catwise_match, memmap=True) as hdul:
        md = hdul[1].data
        match_nvss = np.asarray(md["nvss"])
        match_source_id = np.asarray(md["source_id"])

    nvss_id_keep_sorted = np.sort(nvss_id_keep)
    idx = np.searchsorted(nvss_id_keep_sorted, match_nvss)
    # Avoid OOB indexing for idx==len(array); numpy evaluates both sides of `&`.
    idx_clip = np.clip(idx, 0, max(0, nvss_id_keep_sorted.size - 1))
    in_keep = (idx < nvss_id_keep_sorted.size) & (nvss_id_keep_sorted[idx_clip] == match_nvss)
    rm_source_ids = np.unique(match_source_id[in_keep])

    rm_source_ids_sorted = np.sort(rm_source_ids)
    idx2 = np.searchsorted(rm_source_ids_sorted, wise_source_id)
    has_nvss_match = (idx2 < rm_source_ids_sorted.size) & (rm_source_ids_sorted[idx2] == wise_source_id)

    rng = np.random.default_rng(int(args.seed))
    remove = has_nvss_match.copy()

    # --- Homogenize by random removal in WISE-only regions (Secrest 2_rm_nvss.py) ---
    if not args.no_homogenize:
        ipix_wise = np.unique(wise_ipix)
        ipix_nvss = np.unique(nvss_ipix_keep)
        ipix_wise_nvss = np.intersect1d(ipix_wise, ipix_nvss)
        ipix_wise_only = np.setdiff1d(ipix_wise, ipix_nvss)

        match_counts_per_pix = np.bincount(wise_ipix[has_nvss_match], minlength=hp.nside2npix(int(args.nside)))
        nvss_counts_dist = match_counts_per_pix[ipix_wise_nvss].astype(int)
        if nvss_counts_dist.size == 0:
            print("Warning: no overlap pixels between WISE and NVSS; skipping homogenization.")
        else:
            n_rm = rng.choice(nvss_counts_dist, size=ipix_wise_only.size, replace=True)

            # Group WISE sources by pixel for efficient per-pixel random removal.
            order = np.argsort(wise_ipix)
            ip_sorted = wise_ipix[order]
            uniq_pix, starts = np.unique(ip_sorted, return_index=True)
            ends = np.r_[starts[1:], ip_sorted.size]

            for pix, k in zip(ipix_wise_only, n_rm, strict=True):
                if k <= 0:
                    continue
                pos = int(np.searchsorted(uniq_pix, pix))
                if pos >= uniq_pix.size or uniq_pix[pos] != pix:
                    continue
                sl = order[starts[pos] : ends[pos]]
                if sl.size == 0:
                    continue
                if k >= sl.size:
                    remove[sl] = True
                else:
                    choice = rng.choice(sl, size=int(k), replace=False)
                    remove[choice] = True

    keep = ~remove

    out_cat = Table()
    out_cat["source_id"] = wise_source_id[keep]
    out_cat["l"] = wise_l[keep]
    out_cat["b"] = wise_b[keep]
    out_cat["w1"] = wise_w1[keep]
    out_cat["w1cov"] = wise_w1cov[keep]
    out_path = outdir / "catwise_no_nvss_homog.fits"
    out_cat.write(out_path, overwrite=True)

    summary = {
        "meta": {
            "nside": int(args.nside),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1cut_max": float(args.w1cut_max),
            "nvss_scut_mjy": float(args.nvss_scut_mjy),
            "nvss_tbcut_K": float(args.nvss_tbcut_K),
            "homogenize": not bool(args.no_homogenize),
            "seed": int(args.seed),
        },
        "counts": {
            "wise_in": int(wise_source_id.size),
            "nvss_keep": int(nvss_id_keep.size),
            "wise_removed_nvss_match": int(np.sum(has_nvss_match)),
            "wise_removed_total": int(np.sum(remove)),
            "wise_out": int(np.sum(keep)),
        },
        "paths": {"out_catalog": str(out_path)},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote: {out_path}")
    print(json.dumps(summary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
