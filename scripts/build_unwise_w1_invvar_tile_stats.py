#!/usr/bin/env python3
"""
Build an imaging-derived WISE/unWISE depth proxy from unWISE W1 inverse-variance (invvar) maps.

Why this exists
--------------
For "amplitude closure" we want a map-level proxy that is closer to the *actual noise* (including
background) than raw exposure counts. unWISE provides per-tile inverse-variance maps:

  https://unwise.me/data/<version>/unwise-coadds/fulldepth/<grp>/<coadd_id>/unwise-<coadd_id>-w1-invvar-m.fits.gz

This script downloads (as needed) those files and computes a robust per-tile summary statistic
(default: median of positive finite pixels), writing a JSON mapping:

  { "<coadd_id>": <invvar_stat_float>, ... }

Design goals
------------
- Resumable: if the output JSON exists, already-computed tiles are skipped.
- Checkpointed: writes the JSON periodically so interruption doesn't lose progress.
- Optional caching: you can keep downloads, but by default we delete per-tile files after computing
  the statistic (to avoid hundreds of GB of cached FITS on disk).
"""

from __future__ import annotations

import argparse
import json
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
        f"{grp}/{coadd_id}/unwise-{coadd_id}-w1-invvar-m.fits.gz"
    )


def _tile_cache_path(cache_root: Path, version: str, coadd_id: str) -> Path:
    grp = coadd_id[:3]
    return cache_root / version / "w1-invvar-m" / grp / coadd_id / f"unwise-{coadd_id}-w1-invvar-m.fits.gz"


def _download(url: str, out: Path, retries: int = 4, backoff_s: float = 1.5) -> None:
    import shutil

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return

    for i in range(int(retries)):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                with out.open("wb") as f:
                    shutil.copyfileobj(r, f)
            if out.stat().st_size <= 0:
                raise RuntimeError("download produced empty file")
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            if i == retries - 1:
                raise RuntimeError(f"download failed after {retries} tries: {url}") from e
            time.sleep(backoff_s * (2.0**i))


def _invvar_stat_from_fits_gz(path: Path, stat: str) -> float:
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


def _process_one_tile(args: Tuple[str, str, str, str, bool]) -> Tuple[str, float]:
    """
    ProcessPool-friendly wrapper:
      (coadd_id, version, cache_root_str, stat, keep_downloads) -> (coadd_id, stat_value)
    """

    coadd_id, version, cache_root_str, stat, keep_downloads = args
    cache_root = Path(cache_root_str)
    url = _tile_url(version, coadd_id)
    path = _tile_cache_path(cache_root, version, coadd_id)
    _download(url, path)
    val = _invvar_stat_from_fits_gz(path, stat)
    if not keep_downloads:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Best-effort cleanup; leave it if the filesystem complains.
            pass
    return coadd_id, val


def _coadd_ids_from_tiles_fits(tiles_fits: str) -> List[str]:
    from astropy.table import Table

    tiles = Table.read(tiles_fits, memmap=True)
    if "coadd_id" not in tiles.colnames:
        raise SystemExit("tiles.fits missing required column: coadd_id")
    return [str(x) for x in np.asarray(tiles["coadd_id"]).astype(str)]


def _filter_coadd_ids_by_tile_center_gal_b(
    *,
    tiles_fits: str,
    coadd_ids: List[str],
    b_cut_deg: float,
) -> List[str]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    tiles = Table.read(tiles_fits, memmap=True)
    if not {"coadd_id", "ra", "dec"}.issubset(set(tiles.colnames)):
        raise SystemExit("tiles.fits missing required columns: coadd_id, ra, dec")

    coadd = np.asarray(tiles["coadd_id"]).astype(str)
    ra = np.asarray(tiles["ra"], dtype=float)
    dec = np.asarray(tiles["dec"], dtype=float)
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    b = np.asarray(sc.b.deg, dtype=float)

    keep = np.abs(b) >= float(b_cut_deg)
    keep_set = set(str(x) for x in coadd[keep])
    return [cid for cid in coadd_ids if cid in keep_set]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="neo7", help="unWISE release tag (e.g. neo7).")
    ap.add_argument("--tiles-fits", default="data/external/unwise/tiles.fits")
    ap.add_argument("--cache-root", default="data/cache/unwise_invvar", help="Cache root for downloads and outputs.")
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path. Default: <cache-root>/<version>/w1_invvar_m_tile_stats_<stat>.json",
    )
    ap.add_argument("--stat", choices=["median", "mean", "p50"], default="median")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--keep-downloads", action="store_true", help="Keep downloaded FITS in the cache.")
    ap.add_argument(
        "--tile-center-gal-b-cut",
        type=float,
        default=None,
        help="Optional cut on |b| (deg) using unWISE tile centers, to reduce downloads.",
    )
    ap.add_argument(
        "--coadd-ids",
        default=None,
        help="Optional text file with one coadd_id per line. If omitted, uses all coadd_id entries in --tiles-fits.",
    )
    ap.add_argument(
        "--on-error",
        choices=["fail", "skip"],
        default="skip",
        help="Per-tile error handling. 'skip' records 0.0 for that tile and continues.",
    )
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    out_json = Path(
        args.out_json or (cache_root / args.version / f"w1_invvar_m_tile_stats_{args.stat}.json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Determine tile list.
    if args.coadd_ids:
        coadd_ids = [
            ln.strip()
            for ln in Path(args.coadd_ids).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    else:
        coadd_ids = _coadd_ids_from_tiles_fits(str(args.tiles_fits))

    if args.tile_center_gal_b_cut is not None:
        coadd_ids = _filter_coadd_ids_by_tile_center_gal_b(
            tiles_fits=str(args.tiles_fits),
            coadd_ids=coadd_ids,
            b_cut_deg=float(args.tile_center_gal_b_cut),
        )

    # Load existing results if present (resume).
    stats: Dict[str, float] = {}
    if out_json.exists():
        stats = json.loads(out_json.read_text())

    todo = [cid for cid in coadd_ids if cid not in stats]
    print(
        "unWISE W1 invvar tile stats: "
        f"have={len(stats)} todo={len(todo)} total={len(coadd_ids)} out={out_json}"
    )

    done = 0
    errors: Dict[str, str] = {}
    if todo:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {
                ex.submit(
                    _process_one_tile,
                    (cid, args.version, str(cache_root), args.stat, bool(args.keep_downloads)),
                ): cid
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
                    # Sentinel to avoid hammering the same tile forever.
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

