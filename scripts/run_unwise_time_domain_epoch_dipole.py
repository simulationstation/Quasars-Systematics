#!/usr/bin/env python3
"""Epoch-resolved dipole *amplitude* test using the unWISE time-domain catalog (IRSA580).

Goal
----
Build an epoch-sliced, quasar-like selection from time-resolved unWISE/NEOWISE photometry
and measure the dipole amplitude D per epoch.

This script reads the unWISE time-domain parquet catalog directly from the public IRSA S3 bucket
(anonymous access) and constructs HEALPix count maps (Galactic coords) per EPOCH.

Selection (default)
-------------------
- Quality: primary==1, flags_unwise==0, flags_info==0, flux>0, dflux>0
- SNR cuts: (flux/dflux) >= snr_min in both W1 and W2
- Quasar-like cut: W1 <= w1_max (Vega) and (W1-W2) >= w1w2_min (Vega)

Implementation notes
--------------------
- The time-domain table stores band-specific rows; within each (EPOCH, band) group inside a file,
  the band-stripped unwise_detid key is sorted and unique. We exploit that to pair W1/W2 via
  searchsorted without per-file sorting.
- A fixed Secrest-style footprint mask is applied (mask_zeros + exclusion discs + |b| cut),
  so the sky region is identical across epochs.
- Dipole amplitudes are computed by a Poisson GLM fit on the masked HEALPix maps:
    log mu_p = beta0 + b · n_p
  For small amplitudes, D \approx |b|.

Outputs
-------
Writes under --outdir:
- shards/shard_XXX.npz : resumable per-shard partials
- counts_by_epoch.npy  : [n_epoch, npix] int64 counts
- summary.json         : per-epoch N, mean MJD, D, sigma_D
- D_vs_epoch.png       : amplitude vs epoch time (if matplotlib available)

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


S3_PREFIX = "nasa-irsa-wise/unwise/neo7/catalogs/time_domain/healpix_k5"
ROW_COUNTS_URL = (
    "https://irsa.ipac.caltech.edu/data/download/parquet/unwise/neo7/time_domain/healpix_k5/"
    "row-counts-per-file.csv"
)

# WISE Vega flux zero points (Jy). (Matches W1~16.6 -> ~71 uJy sanity.)
F0_W1_JY = 309.540
F0_W2_JY = 171.787

# ICRS (J2000) -> Galactic rotation matrix (IAU standard; used by astropy).
ICRS_TO_GAL = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=np.float64,
)


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _ensure_env_sane() -> None:
    # Avoid slow IMDS calls in some environments.
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")


def download_row_counts_csv(path: Path) -> None:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npz")
    with urllib.request.urlopen(ROW_COUNTS_URL, timeout=60) as r:
        data = r.read()
    tmp.write_bytes(data)
    tmp.replace(path)


def load_row_counts(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = str(row["path"]).strip()
            n = int(row["nrows"])
            rows.append((p, n))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def binpack_shards(items: list[tuple[str, int]], n_shards: int) -> list[list[str]]:
    """Greedy bin-packing by nrows for crude load balance."""
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    loads = np.zeros(n_shards, dtype=np.int64)

    for path, nrows in sorted(items, key=lambda x: x[1], reverse=True):
        j = int(np.argmin(loads))
        shards[j].append(path)
        loads[j] += int(nrows)

    return shards


@dataclass(frozen=True)
class Mask:
    seen: np.ndarray  # True=unmasked


def build_secrest_mask(
    *,
    nside: int,
    catwise_catalog: Path,
    exclude_mask_fits: Path,
    b_cut_deg: float,
    w1cov_min: float,
) -> Mask:
    """Secrest-style footprint mask: mask_zeros + exclusion discs + |b| cut."""

    import healpy as hp
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    npix = hp.nside2npix(int(nside))

    # Parent sample for mask_zeros is the full W1cov>=min map (no |b| cut yet).
    with fits.open(str(catwise_catalog), memmap=True) as hdul:
        d = hdul[1].data
        w1cov = np.asarray(d["w1cov"], dtype=float)
        l = np.asarray(d["l"], dtype=float)
        b = np.asarray(d["b"], dtype=float)

    sel = np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b) & (w1cov >= float(w1cov_min))
    theta = np.deg2rad(90.0 - b[sel])
    phi = np.deg2rad(l[sel] % 360.0)
    ipix_base = hp.ang2pix(int(nside), theta, phi, nest=False)

    cnt = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    mask = np.zeros(npix, dtype=bool)
    idx0 = np.where(cnt == 0)[0]
    if idx0.size:
        neigh = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            neigh[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[neigh] = True  # includes -1 index per Secrest behavior

    # Exclusion discs
    tmask = Table.read(str(exclude_mask_fits), memmap=True)
    if "use" in tmask.colnames:
        tmask = tmask[np.asarray(tmask["use"], dtype=bool)]
    if len(tmask):
        sc = SkyCoord(tmask["ra"], tmask["dec"], unit=u.deg, frame="icrs").galactic
        radius = np.deg2rad(np.asarray(tmask["radius"], dtype=float))
        for lon, lat, rad in zip(sc.l.deg, sc.b.deg, radius, strict=True):
            th = np.deg2rad(90.0 - float(lat))
            ph = np.deg2rad(float(lon))
            vec = hp.ang2vec(th, ph)
            disc = hp.query_disc(int(nside), vec=vec, radius=float(rad), inclusive=True, nest=False)
            mask[disc] = True

    # Galactic plane cut (pixel centers).
    _lon_pix, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return Mask(seen=~mask)


def detid_to_key_s20(detid_binary) -> np.ndarray:
    """Return band-stripped key (S20) from unwise_detid (bytes)."""
    import pyarrow as pa

    if isinstance(detid_binary, pa.ChunkedArray):
        detid_binary = detid_binary.combine_chunks()

    det_fixed = detid_binary.cast(pa.binary(22))
    buf = det_fixed.buffers()[1]
    det = np.frombuffer(buf, dtype="S22", count=len(det_fixed))

    m = det.view("S1").reshape(-1, 22)
    key_buf = np.empty((det.shape[0], 20), dtype="S1")
    key_buf[:, 0:8] = m[:, 0:8]
    key_buf[:, 8:] = m[:, 10:]
    return key_buf.view("S20").ravel()


def icrs_to_gal_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)

    m = ICRS_TO_GAL
    xg = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z
    yg = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z
    zg = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z
    return xg, yg, zg


def read_parquet_table(path: str, *, fs, columns: list[str], max_tries: int = 4):
    import pyarrow.parquet as pq

    last_err: Exception | None = None
    for t in range(1, int(max_tries) + 1):
        try:
            return pq.read_table(path, filesystem=fs, columns=columns, use_threads=False)
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep_s = min(60.0, 2.0 ** (t - 1))
            print(f"[read retry {t}/{max_tries}] {path}: {type(e).__name__}: {e} (sleep {sleep_s:.1f}s)", flush=True)
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


@dataclass
class ShardState:
    next_i: int
    counts: np.ndarray  # [n_epoch, npix]
    n_sel: np.ndarray  # [n_epoch]
    mjd_sum: np.ndarray  # [n_epoch]
    mjd_cnt: np.ndarray  # [n_epoch]


def save_state(path: Path, state: ShardState) -> None:
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(
        tmp,
        next_i=np.int64(state.next_i),
        counts=state.counts,
        n_sel=state.n_sel,
        mjd_sum=state.mjd_sum,
        mjd_cnt=state.mjd_cnt,
    )
    tmp.replace(path)


def load_state(path: Path, *, n_epoch: int, npix: int) -> ShardState:
    with np.load(path, allow_pickle=False) as z:
        next_i = int(z["next_i"])
        counts = np.asarray(z["counts"], dtype=np.int64)
        n_sel = np.asarray(z["n_sel"], dtype=np.int64)
        mjd_sum = np.asarray(z["mjd_sum"], dtype=np.float64)
        mjd_cnt = np.asarray(z["mjd_cnt"], dtype=np.int64)
    if counts.shape != (n_epoch, npix):
        raise RuntimeError(f"Bad counts shape in {path}: {counts.shape}")
    return ShardState(next_i=next_i, counts=counts, n_sel=n_sel, mjd_sum=mjd_sum, mjd_cnt=mjd_cnt)


def process_one_file(
    *,
    s3_rel_path: str,
    fs,
    nside: int,
    npix: int,
    seen: np.ndarray,
    n_epoch: int,
    flux_w1_min_ujy: float,
    ratio_max: float,
    snr_min: float,
    b_cut_sin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (counts[n_epoch,npix], n_sel[n_epoch], mjd_sum[n_epoch], mjd_cnt[n_epoch])."""

    import healpy as hp

    s3_path = f"{S3_PREFIX}/{s3_rel_path}"
    cols = [
        "unwise_detid",
        "band",
        "EPOCH",
        "ra",
        "dec",
        "flux",
        "dflux",
        "nm",
        "primary",
        "flags_unwise",
        "flags_info",
        "MJDMEAN",
    ]
    t = read_parquet_table(s3_path, fs=fs, columns=cols)

    # Extract arrays.
    band = t["band"].to_numpy(zero_copy_only=False)
    epoch = t["EPOCH"].to_numpy(zero_copy_only=False)
    ra = t["ra"].to_numpy(zero_copy_only=False)
    dec = t["dec"].to_numpy(zero_copy_only=False)
    flux = t["flux"].to_numpy(zero_copy_only=False)
    dflux = t["dflux"].to_numpy(zero_copy_only=False)
    primary = t["primary"].to_numpy(zero_copy_only=False)
    flags_unwise = t["flags_unwise"].to_numpy(zero_copy_only=False)
    flags_info = t["flags_info"].to_numpy(zero_copy_only=False)
    mjd = t["MJDMEAN"].to_numpy(zero_copy_only=False)

    # Base quality cuts (shared).
    good = (primary == 1) & (flags_unwise == 0) & (flags_info == 0)
    good &= np.isfinite(ra) & np.isfinite(dec) & np.isfinite(flux) & np.isfinite(dflux) & np.isfinite(mjd)
    good &= (flux > 0.0) & (dflux > 0.0)

    if not np.any(good):
        zc = np.zeros((n_epoch, npix), dtype=np.int64)
        zn = np.zeros(n_epoch, dtype=np.int64)
        zm = np.zeros(n_epoch, dtype=np.float64)
        zmc = np.zeros(n_epoch, dtype=np.int64)
        return zc, zn, zm, zmc

    key = detid_to_key_s20(t["unwise_detid"])

    counts = np.zeros((n_epoch, npix), dtype=np.int64)
    n_sel = np.zeros(n_epoch, dtype=np.int64)
    mjd_sum = np.zeros(n_epoch, dtype=np.float64)
    mjd_cnt = np.zeros(n_epoch, dtype=np.int64)

    # Process per-epoch to exploit sorted, unique keys within each (EPOCH, band).
    for e in np.unique(epoch[good]):
        e = int(e)
        if e < 0 or e >= n_epoch:
            continue

        sel_e = good & (epoch == e)
        sel1 = sel_e & (band == 1)
        sel2 = sel_e & (band == 2)
        if not np.any(sel1) or not np.any(sel2):
            continue

        # Band-specific cuts BEFORE matching.
        # W1: magnitude (flux) cut + SNR.
        flux1 = flux[sel1]
        dflux1 = dflux[sel1]
        snr1 = flux1 / dflux1
        keep1 = (flux1 >= float(flux_w1_min_ujy)) & (snr1 >= float(snr_min))
        if not np.any(keep1):
            continue

        # W2: just SNR (keep flux>0 from base).
        flux2 = flux[sel2]
        dflux2 = dflux[sel2]
        snr2 = flux2 / dflux2
        keep2 = snr2 >= float(snr_min)
        if not np.any(keep2):
            continue

        k1 = key[sel1][keep1]
        k2 = key[sel2][keep2]

        # Within an (EPOCH, band) group inside a file, keys are sorted + unique.
        # Pair W1->W2 via searchsorted.
        pos = np.searchsorted(k2, k1)
        m = pos < k2.size
        if np.any(m):
            m[m] &= (k2[pos[m]] == k1[m])
        if not np.any(m):
            continue

        flux1m = flux1[keep1][m]
        flux2m = flux2[keep2][pos[m]]

        # Color cut: (W1-W2) >= w1w2_min (Vega) <=> flux1/flux2 <= ratio_max.
        mcol = flux1m <= float(ratio_max) * flux2m
        if not np.any(mcol):
            continue

        # Positions from W1 rows.
        ra_sel = ra[sel1][keep1][m][mcol]
        dec_sel = dec[sel1][keep1][m][mcol]
        mjd_sel = mjd[sel1][keep1][m][mcol]

        xg, yg, zg = icrs_to_gal_unitvec(ra_sel, dec_sel)
        mlat = np.abs(zg) >= float(b_cut_sin)
        if not np.any(mlat):
            continue

        xg = xg[mlat]
        yg = yg[mlat]
        zg = zg[mlat]
        mjd_sel = mjd_sel[mlat]

        ipix = hp.vec2pix(int(nside), xg, yg, zg, nest=False)
        mseen = seen[ipix]
        if not np.any(mseen):
            continue

        ipix = ipix[mseen]
        mjd_sel = mjd_sel[mseen]

        bc = np.bincount(np.asarray(ipix, dtype=np.int64), minlength=npix).astype(np.int64)
        counts[e] += bc
        n_sel[e] += int(ipix.size)
        mjd_sum[e] += float(np.sum(mjd_sel))
        mjd_cnt[e] += int(mjd_sel.size)

    return counts, n_sel, mjd_sum, mjd_cnt


def run_shard(
    shard_idx: int,
    shard_paths: list[str],
    *,
    outdir: Path,
    nside: int,
    npix: int,
    seen: np.ndarray,
    n_epoch: int,
    flux_w1_min_ujy: float,
    ratio_max: float,
    snr_min: float,
    b_cut_sin: float,
    checkpoint_every: int,
) -> dict:
    from pyarrow import fs as pafs

    fs = pafs.S3FileSystem(anonymous=True, region="us-west-2")

    shard_dir = outdir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    state_path = shard_dir / f"shard_{shard_idx:03d}.npz"

    if state_path.exists():
        st = load_state(state_path, n_epoch=n_epoch, npix=npix)
    else:
        st = ShardState(
            next_i=0,
            counts=np.zeros((n_epoch, npix), dtype=np.int64),
            n_sel=np.zeros(n_epoch, dtype=np.int64),
            mjd_sum=np.zeros(n_epoch, dtype=np.float64),
            mjd_cnt=np.zeros(n_epoch, dtype=np.int64),
        )

    t0 = time.time()
    n_files = len(shard_paths)

    for i in range(st.next_i, n_files):
        rel = shard_paths[i]
        c, n, ms, mc = process_one_file(
            s3_rel_path=rel,
            fs=fs,
            nside=nside,
            npix=npix,
            seen=seen,
            n_epoch=n_epoch,
            flux_w1_min_ujy=flux_w1_min_ujy,
            ratio_max=ratio_max,
            snr_min=snr_min,
            b_cut_sin=b_cut_sin,
        )
        st.counts += c
        st.n_sel += n
        st.mjd_sum += ms
        st.mjd_cnt += mc
        st.next_i = i + 1

        if checkpoint_every > 0 and (st.next_i % checkpoint_every == 0 or st.next_i == n_files):
            save_state(state_path, st)

        if st.next_i % max(1, checkpoint_every) == 0 or st.next_i == n_files:
            dt = time.time() - t0
            rate = st.next_i / dt if dt > 0 else float("nan")
            print(
                f"[shard {shard_idx:03d}] {st.next_i}/{n_files} files ({rate:.3f} file/s) N_sel_total={int(st.n_sel.sum())}",
                flush=True,
            )

    dt = time.time() - t0
    return {
        "shard": int(shard_idx),
        "n_files": int(n_files),
        "seconds": float(dt),
        "n_sel_total": int(st.n_sel.sum()),
    }


def fit_poisson_glm_dipole(y: np.ndarray, nhat: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Poisson GLM: log mu = beta0 + b·n, returns (beta[4], cov[4,4])."""

    from scipy.optimize import minimize

    y = np.asarray(y, dtype=np.float64)
    nhat = np.asarray(nhat, dtype=np.float64)

    X = np.column_stack([np.ones_like(y), nhat[:, 0], nhat[:, 1], nhat[:, 2]])

    mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
    beta0 = np.array([math.log(mu0), 0.0, 0.0, 0.0], dtype=np.float64)

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        eta = np.clip(eta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        return nll, np.asarray(grad, dtype=np.float64)

    res = minimize(
        lambda b: fun_and_grad(b)[0],
        beta0,
        jac=lambda b: fun_and_grad(b)[1],
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-12},
    )
    beta = np.asarray(res.x, dtype=np.float64)

    # Fisher approximation
    cov = None
    try:
        eta = np.clip(X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        fisher = X.T @ (mu[:, None] * X)
        cov = np.linalg.inv(fisher)
    except Exception:  # noqa: BLE001
        cov = None

    return beta, cov


def main(argv: Iterable[str] | None = None) -> int:
    _ensure_env_sane()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=None, help="Output directory (default: outputs/epoch_dipole_<UTC>).")

    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)

    ap.add_argument("--w1-max", type=float, default=16.4)
    ap.add_argument("--w1w2-min", type=float, default=0.8)
    ap.add_argument("--snr-min", type=float, default=5.0)

    ap.add_argument("--n-epoch", type=int, default=17, help="Number of EPOCH bins (default 17: 0..16).")

    ap.add_argument("--n-shards", type=int, default=64)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=5, help="Checkpoint per shard every N files.")

    ap.add_argument("--max-files", type=int, default=None, help="Debug: process only the first N files total.")

    ap.add_argument(
        "--catwise-catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument(
        "--row-counts-csv",
        default="data/cache/unwise_time_domain/row-counts-per-file.csv",
        help="Local cached copy of row-counts-per-file.csv.",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.outdir or f"outputs/epoch_dipole_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    row_csv = Path(args.row_counts_csv)
    if not row_csv.exists():
        print(f"Downloading row-counts csv -> {row_csv}", flush=True)
        download_row_counts_csv(row_csv)

    items = load_row_counts(row_csv)
    if args.max_files is not None:
        items = items[: int(args.max_files)]

    print(f"Files: {len(items)}", flush=True)

    nside = int(args.nside)

    # Fixed footprint mask.
    print("Building footprint mask...", flush=True)
    mask = build_secrest_mask(
        nside=nside,
        catwise_catalog=Path(args.catwise_catalog),
        exclude_mask_fits=Path(args.exclude_mask_fits),
        b_cut_deg=float(args.b_cut),
        w1cov_min=float(args.w1cov_min),
    )

    import healpy as hp

    npix = hp.nside2npix(nside)
    n_epoch = int(args.n_epoch)

    # Selection thresholds.
    flux_w1_min_ujy = float(F0_W1_JY * 1e6 * 10.0 ** (-0.4 * float(args.w1_max)))
    ratio_max = float((F0_W1_JY / F0_W2_JY) * 10.0 ** (-0.4 * float(args.w1w2_min)))
    snr_min = float(args.snr_min)
    b_cut_sin = float(math.sin(math.radians(float(args.b_cut))))

    cfg = {
        "s3_prefix": S3_PREFIX,
        "row_counts_url": ROW_COUNTS_URL,
        "nside": nside,
        "b_cut_deg": float(args.b_cut),
        "w1cov_min": float(args.w1cov_min),
        "w1_max_vega": float(args.w1_max),
        "w1w2_min_vega": float(args.w1w2_min),
        "snr_min": snr_min,
        "n_epoch": n_epoch,
        "flux_w1_min_ujy": flux_w1_min_ujy,
        "ratio_max_flux1_over_flux2": ratio_max,
        "n_shards": int(args.n_shards),
        "workers": int(args.workers),
        "checkpoint_every": int(args.checkpoint_every),
        "max_files": None if args.max_files is None else int(args.max_files),
        "timestamp_utc": utc_tag(),
    }
    (outdir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    shards = binpack_shards(items, int(args.n_shards))

    # Multiprocessing over shards.
    import multiprocessing as mp

    seen = mask.seen

    print(f"Starting pool: workers={int(args.workers)} shards={len(shards)}", flush=True)

    t0 = time.time()
    results: list[dict] = []

    with mp.get_context("spawn").Pool(processes=int(args.workers)) as pool:
        jobs = []
        for si, spaths in enumerate(shards):
            jobs.append(
                pool.apply_async(
                    run_shard,
                    (si, spaths),
                    dict(
                        outdir=outdir,
                        nside=nside,
                        npix=npix,
                        seen=seen,
                        n_epoch=n_epoch,
                        flux_w1_min_ujy=flux_w1_min_ujy,
                        ratio_max=ratio_max,
                        snr_min=snr_min,
                        b_cut_sin=b_cut_sin,
                        checkpoint_every=int(args.checkpoint_every),
                    ),
                )
            )

        for j in jobs:
            try:
                r = j.get()
                results.append(r)
                done = len(results)
                dt = time.time() - t0
                print(f"[main] done {done}/{len(jobs)} shards (elapsed {dt/3600:.2f} hr)", flush=True)
            except KeyboardInterrupt:
                raise

    # Merge shards.
    counts = np.zeros((n_epoch, npix), dtype=np.int64)
    n_sel = np.zeros(n_epoch, dtype=np.int64)
    mjd_sum = np.zeros(n_epoch, dtype=np.float64)
    mjd_cnt = np.zeros(n_epoch, dtype=np.int64)

    shard_dir = outdir / "shards"
    for si in range(len(shards)):
        p = shard_dir / f"shard_{si:03d}.npz"
        if not p.exists():
            raise RuntimeError(f"Missing shard output: {p}")
        st = load_state(p, n_epoch=n_epoch, npix=npix)
        counts += st.counts
        n_sel += st.n_sel
        mjd_sum += st.mjd_sum
        mjd_cnt += st.mjd_cnt

    np.save(outdir / "counts_by_epoch.npy", counts)

    # Fit dipole per epoch.
    ip = np.arange(npix, dtype=np.int64)
    xpix, ypix, zpix = hp.pix2vec(nside, ip, nest=False)
    nhat = np.column_stack([xpix, ypix, zpix])[seen]

    summary_rows = []
    for e in range(n_epoch):
        y = counts[e][seen].astype(np.float64)
        N = float(np.sum(y))
        if N <= 0:
            summary_rows.append(
                {
                    "epoch": int(e),
                    "N": 0,
                    "mjd_mean": float("nan"),
                    "D": float("nan"),
                    "sigma_D": float("nan"),
                    "beta": [float("nan")] * 4,
                }
            )
            continue

        beta, cov = fit_poisson_glm_dipole(y, nhat)
        bvec = beta[1:4]
        D = float(np.linalg.norm(bvec))
        sigma_D = float("nan")
        if cov is not None and np.isfinite(D) and D > 0:
            cb = np.asarray(cov[1:4, 1:4], dtype=np.float64)
            varD = float(bvec.T @ cb @ bvec) / float(D * D)
            sigma_D = float(math.sqrt(max(0.0, varD)))

        mjd_mean = float(mjd_sum[e] / mjd_cnt[e]) if mjd_cnt[e] > 0 else float("nan")

        summary_rows.append(
            {
                "epoch": int(e),
                "N": int(n_sel[e]),
                "mjd_mean": mjd_mean,
                "D": D,
                "sigma_D": sigma_D,
                "beta": [float(x) for x in beta],
            }
        )
        print(f"Epoch {e:02d}: N={int(n_sel[e])} D={D:.5f} +/- {sigma_D:.5f} mjd_mean={mjd_mean:.1f}", flush=True)

    summary = {
        "config": cfg,
        "shards": results,
        "epochs": summary_rows,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Plot (best-effort).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mjd = np.array([r["mjd_mean"] for r in summary_rows], dtype=float)
        D = np.array([r["D"] for r in summary_rows], dtype=float)
        sD = np.array([r["sigma_D"] for r in summary_rows], dtype=float)

        ok = np.isfinite(mjd) & np.isfinite(D)
        plt.figure(figsize=(7.5, 4.5))
        if np.any(ok):
            plt.errorbar(mjd[ok], D[ok], yerr=sD[ok], fmt="o-", lw=1.2, ms=4)
        plt.xlabel("Mean MJD (per epoch)")
        plt.ylabel("Dipole amplitude D (Poisson GLM)")
        plt.title("Epoch-resolved dipole amplitude")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(outdir / "D_vs_epoch.png", dpi=180)
        plt.close()
    except Exception as e:  # noqa: BLE001
        print(f"Plot failed: {type(e).__name__}: {e}", flush=True)

    dt = time.time() - t0
    print(f"Done. Elapsed {dt/3600:.2f} hr. Out: {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
