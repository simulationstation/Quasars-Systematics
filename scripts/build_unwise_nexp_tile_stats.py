#!/usr/bin/env python3
"""
Build an independent WISE/unWISE depth-of-coverage template from unWISE W1 `n` maps.

This script downloads (if needed) per-tile files:
  https://unwise.me/data/<version>/unwise-coadds/fulldepth/<grp>/<coadd_id>/unwise-<coadd_id>-w1-n-m.fits.gz

and computes a robust per-tile summary statistic (default: median of positive pixels),
writing a JSON mapping:
  { "<coadd_id>": <nexp_stat_float>, ... }

Why this exists:
  - In the quasar dipole "vector convergence" GLM+CV workflow, using catalog-derived `w1cov`
    as a depth proxy is vulnerable to criticism (sampled only where sources exist).
  - unWISE `w1-n-m` maps are *imaging-derived* and independent of the quasar selection.
  - We summarize them per coadd tile and use log(Nexp) as a Poisson-GLM offset.

This is intentionally resumable:
  - If the output JSON exists, already-computed tiles are skipped.
  - Downloads are cached on disk.
  - JSON is checkpointed periodically.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _atomic_write_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _tile_url(version: str, coadd_id: str) -> str:
    grp = coadd_id[:3]
    return (
        f"https://unwise.me/data/{version}/unwise-coadds/fulldepth/"
        f"{grp}/{coadd_id}/unwise-{coadd_id}-w1-n-m.fits.gz"
    )


def _tile_cache_path(cache_root: Path, version: str, coadd_id: str) -> Path:
    grp = coadd_id[:3]
    return cache_root / version / "w1-n-m" / grp / coadd_id / f"unwise-{coadd_id}-w1-n-m.fits.gz"


def _download(url: str, out: Path, retries: int = 4, backoff_s: float = 1.5) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    for i in range(int(retries)):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            out.write_bytes(data)
            return
        except (urllib.error.URLError, TimeoutError) as e:
            if i == retries - 1:
                raise RuntimeError(f"download failed after {retries} tries: {url}") from e
            time.sleep(backoff_s * (2.0**i))


def _nexp_stat_from_fits_gz(path: Path, stat: str) -> float:
    # Heavy deps local to worker.
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    v = np.asarray(data, dtype=float)
    v = v[np.isfinite(v)]
    v = v[v > 0]
    if v.size == 0:
        return float("nan")
    if stat == "median":
        return float(np.median(v))
    if stat == "mean":
        return float(np.mean(v))
    if stat == "p50":
        return float(np.percentile(v, 50))
    raise ValueError(f"unknown stat: {stat}")


def _process_one_tile(args: Tuple[str, str, str, str]) -> Tuple[str, float]:
    """
    ProcessPool-friendly wrapper:
      (coadd_id, version, cache_root_str, stat) -> (coadd_id, stat_value)
    """

    coadd_id, version, cache_root_str, stat = args
    cache_root = Path(cache_root_str)
    url = _tile_url(version, coadd_id)
    path = _tile_cache_path(cache_root, version, coadd_id)
    _download(url, path)
    val = _nexp_stat_from_fits_gz(path, stat)
    return coadd_id, val


def _coadd_ids_needed_from_catalog(
    *,
    catalog: str,
    tiles_fits: str,
    nvss_crossmatch: str | None,
    exclude_mask_fits: str | None,
    b_cut: float,
    w1cov_min: float,
    w1_max: float,
    nside: int,
) -> List[str]:
    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table
    from scipy.spatial import cKDTree

    tab = Table.read(catalog, memmap=True)
    source_id = np.asarray(tab["source_id"]).astype(str)
    l = np.asarray(tab["l"], dtype=float)
    b = np.asarray(tab["b"], dtype=float)
    w1 = np.asarray(tab["w1"], dtype=float)
    w1cov = np.asarray(tab["w1cov"], dtype=float)

    base = np.isfinite(l) & np.isfinite(b) & np.isfinite(w1) & np.isfinite(w1cov)
    base &= (np.abs(b) > float(b_cut)) & (w1cov >= float(w1cov_min)) & (w1 <= float(w1_max))

    # Optional NVSS removal.
    if nvss_crossmatch:
        nv = Table.read(nvss_crossmatch, memmap=True)
        nv_ids = np.asarray(nv["source_id"]).astype(str)
        base &= ~np.isin(source_id, nv_ids)

    l = l[base]
    b = b[base]
    theta = np.deg2rad(90.0 - b)
    phi = np.deg2rad(l % 360.0)
    pix = hp.ang2pix(int(nside), theta, phi, nest=True)

    # Optional Secrest exclude regions (pixel-level).
    if exclude_mask_fits:
        ex_path = Path(exclude_mask_fits)
        if ex_path.exists():
            ex = Table.read(ex_path, memmap=True)
            use = np.asarray(ex["use"], dtype=bool) if "use" in ex.colnames else np.ones(len(ex), dtype=bool)
            ex = ex[use]
            if len(ex) > 0:
                g = SkyCoord(
                    ra=np.asarray(ex["ra"], dtype=float) * u.deg,
                    dec=np.asarray(ex["dec"], dtype=float) * u.deg,
                    frame="icrs",
                ).galactic
                l0 = g.l.deg
                b0 = g.b.deg
                rdeg = np.asarray(ex["radius"], dtype=float)
                exclude = set()
                for ll, bb, rr in zip(l0, b0, rdeg, strict=True):
                    vec = hp.ang2vec(np.deg2rad(90.0 - bb), np.deg2rad(ll))
                    disc = hp.query_disc(int(nside), vec, np.deg2rad(rr), nest=True)
                    exclude.update(int(x) for x in disc)
                if exclude:
                    ex_pix = np.fromiter(exclude, dtype=np.int64)
                    pix = pix[~np.isin(pix, ex_pix)]

    uniq_pix = np.unique(pix)

    # Build KD-tree of tile centers in Galactic unit vectors.
    tiles = Table.read(tiles_fits, memmap=True)
    coadd = np.asarray(tiles["coadd_id"]).astype(str)
    ra = np.asarray(tiles["ra"], dtype=float)
    dec = np.asarray(tiles["dec"], dtype=float)
    g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    lg = np.deg2rad(g.l.deg % 360.0)
    bg = np.deg2rad(g.b.deg)
    cosb = np.cos(bg)
    tile_vec = np.column_stack([cosb * np.cos(lg), cosb * np.sin(lg), np.sin(bg)])
    tree = cKDTree(tile_vec)

    th, ph = hp.pix2ang(int(nside), uniq_pix, nest=True)
    pix_vec = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    _, nn = tree.query(pix_vec, k=1)
    needed = sorted(set(str(x) for x in coadd[nn]))
    return needed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="neo7", help="unWISE release tag (e.g. neo7, neo6, allwise).")
    ap.add_argument("--tiles-fits", default="data/external/unwise/tiles.fits")
    ap.add_argument("--cache-root", default="data/cache/unwise_nexp")
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path. Default: data/cache/unwise_nexp/<version>/w1_n_m_tile_stats_<stat>.json",
    )
    ap.add_argument("--stat", choices=["median", "mean", "p50"], default="median")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument(
        "--on-error",
        choices=["fail", "skip"],
        default="skip",
        help="How to handle per-tile failures (network/404/corrupt FITS). 'skip' records 0.0 for that tile and continues.",
    )

    # Optional: derive the needed coadd_id list from a catalog selection.
    ap.add_argument("--catalog", default=None, help="If set, compute needed tiles from this FITS catalog.")
    ap.add_argument("--nvss-crossmatch", default=None)
    ap.add_argument("--exclude-mask-fits", default=None)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-max", type=float, default=16.4)
    ap.add_argument("--nside", type=int, default=64)

    # Or: provide an explicit list of coadd IDs.
    ap.add_argument("--coadd-ids", default=None, help="Text file with one coadd_id per line.")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    out_json = Path(
        args.out_json
        or (cache_root / args.version / f"w1_n_m_tile_stats_{args.stat}.json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Determine the tile list.
    if args.coadd_ids:
        coadd_ids = [ln.strip() for ln in Path(args.coadd_ids).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    elif args.catalog:
        coadd_ids = _coadd_ids_needed_from_catalog(
            catalog=args.catalog,
            tiles_fits=args.tiles_fits,
            nvss_crossmatch=args.nvss_crossmatch,
            exclude_mask_fits=args.exclude_mask_fits,
            b_cut=float(args.b_cut),
            w1cov_min=float(args.w1cov_min),
            w1_max=float(args.w1_max),
            nside=int(args.nside),
        )
    else:
        raise SystemExit("Provide either --catalog or --coadd-ids")

    # Load existing results if present (resume).
    stats: Dict[str, float] = {}
    if out_json.exists():
        stats = json.loads(out_json.read_text())

    todo = [cid for cid in coadd_ids if cid not in stats]
    print(f"unWISE Nexp tile stats: have={len(stats)} todo={len(todo)} total={len(coadd_ids)} out={out_json}")

    done = 0
    errors: Dict[str, str] = {}
    if todo:
        # Use processes so FITS decompression/stat computation can use multiple CPU cores
        # (threading is typically GIL-limited for this workload).
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {
                ex.submit(_process_one_tile, (cid, args.version, str(cache_root), args.stat)): cid
                for cid in todo
            }
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    k, v = fut.result()
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    errors[cid] = msg
                    if args.on_error == "fail":
                        raise RuntimeError(f"failed tile {cid}: {msg}") from e
                    # Record a sentinel so we don't hammer the server retrying the same tile forever.
                    # Downstream code treats <=0 as invalid and drops affected pixels.
                    stats[cid] = 0.0
                    done += 1
                    if done % int(args.checkpoint_every) == 0:
                        _atomic_write_json(out_json, stats)
                        print(f"checkpoint: {done}/{len(todo)} done (errors={len(errors)})")
                    continue
                stats[k] = float(v)
                done += 1
                if done % int(args.checkpoint_every) == 0:
                    _atomic_write_json(out_json, stats)
                    print(f"checkpoint: {done}/{len(todo)} done (errors={len(errors)})")

    _atomic_write_json(out_json, stats)
    if errors:
        err_path = out_json.with_suffix(out_json.suffix + ".errors.json")
        _atomic_write_json(err_path, errors)
        print(f"errors: {len(errors)} (wrote {err_path})")
    print(f"wrote: {out_json} (n={len(stats)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
