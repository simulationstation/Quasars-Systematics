#!/usr/bin/env python3
"""
Download a high-SNR AllWISE *star* sample from IRSA TAP (resumable, chunked by RA).

This is used to build an *independent* W1 zero-point / δm map from calibrator-like stars,
by predicting W1 from 2MASS colors and mapping residuals on the sky.

Why TAP + RA chunking?
---------------------
Full-sky filtered queries can be slow. Adding a simple RA window constraint makes
queries return quickly and reliably. We also apply a modulo downsampling on `cntr`
to keep per-chunk sizes manageable.

Outputs
-------
Writes per-chunk CSV files under:
  data/cache/allwise_star_sample/<tag>/chunks/

and a manifest JSON:
  data/cache/allwise_star_sample/<tag>/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


IRSA_TAP_SYNC = "https://irsa.ipac.caltech.edu/TAP/sync"


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


@dataclass(frozen=True)
class Chunk:
    ra_lo: float
    ra_hi: float
    idx: int

    def label(self) -> str:
        return f"ra_{self.ra_lo:06.1f}_{self.ra_hi:06.1f}".replace(".", "p")


def build_query(
    *,
    ra_lo: float,
    ra_hi: float,
    mod_n: int,
    mod_k: int,
    w1_lo: float,
    w1_hi: float,
    w1snr_min: float,
    msig_max: float,
    glat_min: float,
) -> str:
    # Keep the selected column list compact; add more only if needed.
    cols = [
        "cntr",
        "ra",
        "dec",
        "glat",
        "glon",
        "elon",
        "elat",
        "w1mjdmean",
        "w1mjdmin",
        "w1mjdmax",
        "w1mpro",
        "w1sigmpro",
        "w1snr",
        "w1cov",
        "j_m_2mass",
        "h_m_2mass",
        "k_m_2mass",
        "j_msig_2mass",
        "h_msig_2mass",
        "k_msig_2mass",
    ]
    # Filters target bright, point-like, artifact-clean sources with good 2MASS.
    where = [
        f"ra >= {float(ra_lo)} AND ra < {float(ra_hi)}",
        "ext_flg = 0",
        "cc_flags = '0000'",
        "SUBSTR(ph_qual, 1, 1) = 'A'",
        f"w1mpro BETWEEN {float(w1_lo)} AND {float(w1_hi)}",
        f"w1snr > {float(w1snr_min)}",
        "j_m_2mass IS NOT NULL AND h_m_2mass IS NOT NULL AND k_m_2mass IS NOT NULL",
        f"j_msig_2mass < {float(msig_max)} AND h_msig_2mass < {float(msig_max)} AND k_msig_2mass < {float(msig_max)}",
        f"ABS(glat) > {float(glat_min)}",
    ]
    if mod_n > 1:
        where.append(f"MOD(cntr, {int(mod_n)}) = {int(mod_k)}")

    q = f"SELECT {', '.join(cols)} FROM allwise_p3as_psd WHERE " + " AND ".join(where)
    return q


def tap_csv(query: str, *, timeout_s: int) -> str:
    params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    url = IRSA_TAP_SYNC + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=float(timeout_s)) as r:
        return r.read().decode("utf-8", "replace")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def count_rows(csv_text: str) -> int:
    # Count data rows in a CSV string with a header row.
    n = 0
    reader = csv.reader(csv_text.splitlines())
    for i, _row in enumerate(reader):
        if i == 0:
            continue
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/cache/allwise_star_sample")
    ap.add_argument("--tag", default=None, help="Output subfolder tag (default: timestamp).")
    ap.add_argument("--ra-step-deg", type=float, default=10.0)
    ap.add_argument("--mod-n", type=int, default=8, help="Downsample: keep rows where MOD(cntr, mod_n)=mod_k.")
    ap.add_argument("--mod-k", type=int, default=0)
    ap.add_argument("--w1-range", default="8,12", help="W1 magnitude range 'lo,hi' for high-SNR stars.")
    ap.add_argument("--w1snr-min", type=float, default=50.0, help="Minimum W1 SNR (quality/precision control).")
    ap.add_argument("--msig-max", type=float, default=0.05, help="Maximum 2MASS J/H/K magnitude error (mag).")
    ap.add_argument("--glat-min", type=float, default=30.0, help="Galactic latitude cut (deg).")
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--sleep-s", type=float, default=1.0, help="Politeness sleep between chunks.")
    args = ap.parse_args()

    out_root = Path(str(args.out_root))
    tag = str(args.tag or utc_tag())
    outdir = out_root / tag
    chunks_dir = outdir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    w1_lo_s, w1_hi_s = (s.strip() for s in str(args.w1_range).split(","))
    w1_lo = float(w1_lo_s)
    w1_hi = float(w1_hi_s)

    ra_step = float(args.ra_step_deg)
    if ra_step <= 0 or ra_step > 180:
        raise SystemExit("--ra-step-deg must be in (0, 180].")

    # Build chunks covering [0,360).
    edges = [0.0]
    while edges[-1] < 360.0 - 1e-9:
        edges.append(min(360.0, edges[-1] + ra_step))
    chunks = [Chunk(ra_lo=edges[i], ra_hi=edges[i + 1], idx=i) for i in range(len(edges) - 1)]

    manifest_path = outdir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    rows_total = int(manifest.get("rows_total", 0))
    done_chunks = set(manifest.get("done_chunks", []))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tap_sync": IRSA_TAP_SYNC,
        "table": "allwise_p3as_psd",
        "ra_step_deg": ra_step,
        "mod_n": int(args.mod_n),
        "mod_k": int(args.mod_k),
        "w1_range": [w1_lo, w1_hi],
        "w1snr_min": float(args.w1snr_min),
        "msig_max": float(args.msig_max),
        "glat_min": float(args.glat_min),
    }
    manifest.setdefault("meta", meta)
    manifest.setdefault("chunks", {})

    for ch in chunks:
        lab = ch.label()
        if lab in done_chunks and (chunks_dir / f"{lab}.csv").exists():
            continue

        q = build_query(
            ra_lo=ch.ra_lo,
            ra_hi=ch.ra_hi,
            mod_n=int(args.mod_n),
            mod_k=int(args.mod_k),
            w1_lo=w1_lo,
            w1_hi=w1_hi,
            w1snr_min=float(args.w1snr_min),
            msig_max=float(args.msig_max),
            glat_min=float(args.glat_min),
        )

        last_err = None
        txt = None
        for attempt in range(int(args.retry)):
            try:
                txt = tap_csv(q, timeout_s=int(args.timeout_s))
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)
                time.sleep(1.5 * (2.0**attempt))
        if txt is None:
            raise RuntimeError(f"Failed chunk {lab}: {last_err}")

        nrows = count_rows(txt)
        out_csv = chunks_dir / f"{lab}.csv"
        write_atomic(out_csv, txt)

        rows_total += nrows
        done_chunks.add(lab)
        manifest["chunks"][lab] = {"ra_lo": ch.ra_lo, "ra_hi": ch.ra_hi, "rows": nrows, "path": str(out_csv)}
        manifest["rows_total"] = rows_total
        manifest["done_chunks"] = sorted(done_chunks)
        write_atomic(manifest_path, json.dumps(manifest, indent=2))

        print(f"{lab}: rows={nrows} total={rows_total} wrote={out_csv}")
        time.sleep(float(args.sleep_s))

    print(f"DONE: chunks={len(done_chunks)}/{len(chunks)} rows_total={rows_total} out={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
