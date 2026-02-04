#!/usr/bin/env python3
"""
Build a small, imaging-derived unWISE W1 invvar depth proxy as a HEALPix map (Galactic coords).

Inputs:
  - unWISE tiles table: data/external/unwise/tiles.fits
  - per-tile invvar statistic: data/cache/unwise_invvar/neo7/w1_invvar_m_tile_stats_median.json

Method:
  - Convert each unWISE tile center (ICRS ra/dec) to Galactic unit vectors.
  - For each HEALPix pixel center (Galactic), find the nearest tile center (KD-tree).
  - Assign the per-tile invvar statistic to that pixel.

This yields a map-level proxy that (unlike raw Nexp) partially captures background-dependent noise.
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--tile-stats-json", default="data/cache/unwise_invvar/neo7/w1_invvar_m_tile_stats_median.json")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument(
        "--out-fits",
        default=None,
        help="Output HEALPix FITS map path (defaults under data/cache/unwise_invvar/neo7/).",
    )
    ap.add_argument(
        "--out-meta-json",
        default=None,
        help="Optional sidecar metadata JSON path (defaults next to --out-fits).",
    )
    ap.add_argument(
        "--value-mode",
        choices=["invvar", "log_invvar"],
        default="log_invvar",
        help="Map values: invvar or log(invvar).",
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
        or f"data/cache/unwise_invvar/neo7/{'loginvvar' if args.value_mode == 'log_invvar' else 'invvar'}_healpix_nside{nside}.fits"
    )
    out_fits.parent.mkdir(parents=True, exist_ok=True)
    out_meta = Path(args.out_meta_json or (out_fits.with_suffix(".meta.json")))

    tile_stats = json.loads(stats_json.read_text())
    tiles = Table.read(tiles_fits, memmap=True)
    if not {"coadd_id", "ra", "dec"}.issubset(set(tiles.colnames)):
        raise SystemExit("tiles.fits missing required columns: coadd_id, ra, dec")

    coadd_id = np.asarray(tiles["coadd_id"]).astype(str)
    ra = np.asarray(tiles["ra"], dtype=float)
    dec = np.asarray(tiles["dec"], dtype=float)
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic

    tile_vec = lb_to_unitvec(sc.l.deg, sc.b.deg)
    valid_tile = np.fromiter((str(cid) in tile_stats for cid in coadd_id), dtype=bool, count=coadd_id.size)
    if not np.any(valid_tile):
        raise SystemExit("tile-stats-json contains no keys matching tiles.fits coadd_id values.")
    tree = cKDTree(tile_vec[valid_tile])

    lon_pix, lat_pix = hp.pix2ang(nside, np.arange(npix), lonlat=True, nest=bool(args.nest))
    pix_vec = lb_to_unitvec(lon_pix, lat_pix)
    _, nn_idx = tree.query(pix_vec, k=1)
    pix_coadd = coadd_id[valid_tile][np.asarray(nn_idx, dtype=int)]
    invvar = np.array([float(tile_stats[str(cid)]) for cid in pix_coadd], dtype=float)

    bad = ~np.isfinite(invvar) | (invvar <= 0.0)
    missing_frac = float(np.mean(bad))
    ok = ~bad
    if not np.any(ok):
        raise SystemExit("No valid invvar values found in tile stats JSON.")

    fill = float(np.median(invvar[ok]))
    invvar[bad] = fill

    if args.value_mode == "log_invvar":
        invvar = np.log(np.clip(invvar, 1e-12, np.inf))

    hp.write_map(str(out_fits), invvar.astype(np.float32), overwrite=True, dtype=np.float32, nest=bool(args.nest))

    meta = {
        "tiles_fits": str(tiles_fits),
        "tile_stats_json": str(stats_json),
        "nside": nside,
        "ordering": "NEST" if args.nest else "RING",
        "coord": "galactic",
        "value_mode": str(args.value_mode),
        "fill_value": float(fill) if args.value_mode == "invvar" else float(np.log(max(1e-12, fill))),
        "missing_frac": missing_frac,
        "npix": int(npix),
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"Wrote: {out_fits} ({out_fits.stat().st_size/1e6:.3f} MB)")
    print(f"Wrote: {out_meta}")
    print(f"missing_frac={missing_frac:.4f} fill={fill:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

