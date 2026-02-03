#!/usr/bin/env python3
"""
Build a small, independent unWISE depth proxy as a HEALPix map (Galactic coords).

Inputs (both already small and optionally version-controlled):
  - unWISE tiles table: data/external/unwise/tiles.fits
  - per-tile Nexp statistic: data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json

Method:
  - Convert each unWISE tile center (ICRS ra/dec) to Galactic unit vectors.
  - For each HEALPix pixel center (Galactic), find the nearest tile center (KD-tree).
  - Assign the per-tile Nexp statistic to that pixel.

This yields an imaging-derived, catalog-independent depth proxy at map level.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-fits", default="data/external/unwise/tiles.fits")
    ap.add_argument("--tile-stats-json", default="data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument(
        "--out-fits",
        default=None,
        help="Output HEALPix FITS map path (defaults under data/cache/unwise_nexp/neo7/).",
    )
    ap.add_argument(
        "--out-meta-json",
        default=None,
        help="Optional sidecar metadata JSON path (defaults next to --out-fits).",
    )
    ap.add_argument(
        "--value-mode",
        choices=["nexp", "log_nexp"],
        default="nexp",
        help="Map values: Nexp or log(Nexp).",
    )
    ap.add_argument("--nest", action="store_true", help="Write NESTED HEALPix ordering (default: RING).")
    args = ap.parse_args()

    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table
    from scipy.spatial import cKDTree

    nside = int(args.nside)
    if nside < 1:
        raise SystemExit("--nside must be >= 1")
    npix = hp.nside2npix(nside)

    tiles_fits = Path(str(args.tiles_fits))
    stats_json = Path(str(args.tile_stats_json))
    if not tiles_fits.exists():
        raise SystemExit(f"Missing tiles FITS: {tiles_fits}")
    if not stats_json.exists():
        raise SystemExit(f"Missing tile stats JSON: {stats_json}")

    out_fits = Path(
        args.out_fits
        or f"data/cache/unwise_nexp/neo7/{'lognexp' if args.value_mode == 'log_nexp' else 'nexp'}_healpix_nside{nside}.fits"
    )
    out_fits.parent.mkdir(parents=True, exist_ok=True)
    out_meta = Path(args.out_meta_json or (out_fits.with_suffix(".meta.json")))

    tile_stats = json.loads(stats_json.read_text())
    tiles = Table.read(tiles_fits, memmap=True)
    if "coadd_id" not in tiles.colnames or "ra" not in tiles.colnames or "dec" not in tiles.colnames:
        raise SystemExit("tiles.fits missing one of required columns: coadd_id, ra, dec")

    coadd_id = np.asarray(tiles["coadd_id"]).astype(str)
    ra = np.asarray(tiles["ra"], dtype=float)
    dec = np.asarray(tiles["dec"], dtype=float)
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic

    tile_vec = lb_to_unitvec(sc.l.deg, sc.b.deg)
    # Only build the tree on tiles that actually exist in the stats JSON.
    valid_tile = np.fromiter((str(cid) in tile_stats for cid in coadd_id), dtype=bool, count=coadd_id.size)
    if not np.any(valid_tile):
        raise SystemExit("tile-stats-json contains no keys matching tiles.fits coadd_id values.")
    tree = cKDTree(tile_vec[valid_tile])

    # Pixel centers in Galactic l,b.
    lon_pix, lat_pix = hp.pix2ang(nside, np.arange(npix), lonlat=True, nest=bool(args.nest))
    pix_vec = lb_to_unitvec(lon_pix, lat_pix)
    _, nn_idx = tree.query(pix_vec, k=1)
    pix_coadd = coadd_id[valid_tile][np.asarray(nn_idx, dtype=int)]
    nexp = np.array([float(tile_stats[str(cid)]) for cid in pix_coadd], dtype=float)

    bad = ~np.isfinite(nexp) | (nexp <= 0.0)
    missing_frac = float(np.mean(bad))
    ok = ~bad
    if not np.any(ok):
        raise SystemExit("No valid Nexp values found in tile stats JSON.")

    fill = float(np.median(nexp[ok]))
    nexp[bad] = fill

    if args.value_mode == "log_nexp":
        nexp = np.log(np.clip(nexp, 1.0, np.inf))

    # Write map.
    hp.write_map(str(out_fits), nexp.astype(np.float32), overwrite=True, dtype=np.float32, nest=bool(args.nest))

    meta = {
        "tiles_fits": str(tiles_fits),
        "tile_stats_json": str(stats_json),
        "nside": nside,
        "ordering": "NEST" if args.nest else "RING",
        "coord": "galactic",
        "value_mode": str(args.value_mode),
        "fill_value": float(fill) if args.value_mode == "nexp" else float(np.log(max(1.0, fill))),
        "missing_frac": missing_frac,
        "npix": int(npix),
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"Wrote: {out_fits} ({out_fits.stat().st_size/1e6:.3f} MB)")
    print(f"Wrote: {out_meta}")
    print(f"missing_frac={missing_frac:.4f} fill={fill:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
