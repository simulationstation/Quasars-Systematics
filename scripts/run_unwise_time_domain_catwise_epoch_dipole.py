#!/usr/bin/env python3
"""Epoch-resolved dipole *amplitude* test using CatWISE AGNs as the parent sample.

Goal
----
Build an epoch-sliced catalog that matches the published CatWISE/Secrest "accepted" quasar
selection as closely as possible, then measure the dipole *amplitude* D per epoch using the
unWISE time-domain catalog (IRSA580).

Key idea
--------
Instead of selecting "quasar-like" objects from the time-domain table directly (which can
admit large populations with time-domain color noise), we:

1) Start from the Secrest+22 accepted CatWISE AGN catalog (Zenodo 6784602), applying the same
   baseline cuts used in our other reproductions (W1, W1-W2, W1cov, footprint mask).
2) For each epoch, require that the object has a valid matched (W1,W2) time-domain measurement
   passing simple quality filters; optionally re-apply W1 and/or color cuts using the epoch
   photometry.
3) Count those *CatWISE-parent* objects on the sky and fit the dipole amplitude D per epoch.

This is the most direct "time stability" kill-shot: a real cosmological/kinematic dipole should
be stable across epochs, whereas scan/coverage/background selection can imprint time-varying
dipolar completeness in the same parent population.

Outputs
-------
Writes under --outdir:
- shards/shard_XXX.npz : resumable per-shard partials
- counts_by_epoch.npy  : [n_epoch, npix] int64 counts (HEALPix, Galactic, RING)
- summary.json         : per-epoch N, mean MJD, D, sigma_D
- D_vs_epoch.png       : amplitude vs epoch time (best-effort)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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

F0_W1_JY = 309.540
F0_W2_JY = 171.787

_K5_RE = re.compile(r"healpix_k5=(\d+)")

_PARENT: "ParentIndex | None" = None


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _ensure_env_sane() -> None:
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")


def download_row_counts_csv(path: Path) -> None:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.csv")
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
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    loads = np.zeros(n_shards, dtype=np.int64)
    for path, nrows in sorted(items, key=lambda x: x[1], reverse=True):
        j = int(np.argmin(loads))
        shards[j].append(path)
        loads[j] += int(nrows)
    return shards


def parse_k5_from_path(rel_path: str) -> int:
    m = _K5_RE.search(rel_path)
    if not m:
        raise ValueError(f"Could not parse healpix_k5 from path: {rel_path}")
    return int(m.group(1))


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
    import healpy as hp
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    npix = hp.nside2npix(int(nside))

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
        mask[neigh] = True  # includes -1 indexing (matches Secrest util behavior)

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


def radec_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def match_to_parent(
    det_vec: np.ndarray,
    parent_vec: np.ndarray,
    *,
    cos_radius: float,
) -> np.ndarray:
    """Return matched parent indices for each detection, or -1 if none within radius."""

    det_vec = np.asarray(det_vec, dtype=np.float64)
    parent_vec = np.asarray(parent_vec, dtype=np.float64)
    if det_vec.size == 0 or parent_vec.size == 0:
        return np.full(det_vec.shape[0], -1, dtype=np.int32)

    # Chunk in detections to avoid large temporary allocations.
    n_det = det_vec.shape[0]
    out = np.full(n_det, -1, dtype=np.int32)
    chunk = 8192
    for i0 in range(0, n_det, chunk):
        i1 = min(n_det, i0 + chunk)
        dots = det_vec[i0:i1] @ parent_vec.T
        j = np.argmax(dots, axis=1)
        mx = dots[np.arange(i1 - i0), j]
        ok = mx >= float(cos_radius)
        if np.any(ok):
            out_chunk = out[i0:i1]
            out_chunk[ok] = j[ok].astype(np.int32, copy=False)
    return out


@dataclass
class ParentIndex:
    # Parent catalog sorted by healpix_k5 (nside=32, NEST, ICRS).
    parent_vec: np.ndarray  # [N,3] float64
    parent_pix64: np.ndarray  # [N] int64 (Galactic, RING)
    k5_cum: np.ndarray  # [12289] int64, cumulative counts per k5

    def slice_for_k5(self, k5: int) -> tuple[np.ndarray, np.ndarray]:
        k5 = int(k5)
        if k5 < 0 or k5 + 1 >= self.k5_cum.size:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)
        i0 = int(self.k5_cum[k5])
        i1 = int(self.k5_cum[k5 + 1])
        if i1 <= i0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)
        return self.parent_vec[i0:i1], self.parent_pix64[i0:i1]


def _init_worker(parent_npz: str) -> None:
    global _PARENT
    with np.load(parent_npz, allow_pickle=False) as z:
        parent_vec = np.asarray(z["parent_vec"], dtype=np.float64)
        parent_pix64 = np.asarray(z["parent_pix64"], dtype=np.int64)
        k5_cum = np.asarray(z["k5_cum"], dtype=np.int64)
    _PARENT = ParentIndex(parent_vec=parent_vec, parent_pix64=parent_pix64, k5_cum=k5_cum)


def build_parent_index(
    *,
    catwise_catalog: Path,
    exclude_mask_fits: Path,
    nside_counts: int,
    b_cut_deg: float,
    w1cov_min: float,
    w1_max: float,
    w1w2_min: float,
) -> tuple[ParentIndex, dict]:
    """Return (ParentIndex, meta) for the filtered CatWISE parent sample."""

    import healpy as hp
    from astropy.io import fits

    mask = build_secrest_mask(
        nside=nside_counts,
        catwise_catalog=catwise_catalog,
        exclude_mask_fits=exclude_mask_fits,
        b_cut_deg=b_cut_deg,
        w1cov_min=w1cov_min,
    )
    seen = mask.seen

    with fits.open(str(catwise_catalog), memmap=True) as hdul:
        d = hdul[1].data
        ra = np.asarray(d["ra"], dtype=np.float64)
        dec = np.asarray(d["dec"], dtype=np.float64)
        l = np.asarray(d["l"], dtype=np.float64)
        b = np.asarray(d["b"], dtype=np.float64)
        w1 = np.asarray(d["w1"], dtype=np.float64)
        w12 = np.asarray(d["w12"], dtype=np.float64)
        w1cov = np.asarray(d["w1cov"], dtype=np.float64)

    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(l) & np.isfinite(b)
    good &= np.isfinite(w1) & np.isfinite(w12) & np.isfinite(w1cov)
    good &= w1cov >= float(w1cov_min)
    good &= w1 <= float(w1_max)
    good &= w12 >= float(w1w2_min)

    # Apply the exact same pixel-level mask used for all dipole fits.
    ipix64 = hp.ang2pix(
        int(nside_counts),
        np.deg2rad(90.0 - b),
        np.deg2rad(l % 360.0),
        nest=False,
    ).astype(np.int64)
    good &= seen[ipix64]

    ra = ra[good]
    dec = dec[good]
    ipix64 = ipix64[good]

    # healpix_k5 in ICRS is nside=32, NEST.
    k5 = hp.ang2pix(
        32,
        np.deg2rad(90.0 - dec),
        np.deg2rad(ra % 360.0),
        nest=True,
    ).astype(np.int32)

    vec = radec_to_unitvec(ra, dec)

    order = np.argsort(k5, kind="mergesort")
    k5s = k5[order]
    vecs = vec[order]
    pixs = ipix64[order]

    counts = np.bincount(k5s.astype(np.int64), minlength=12288).astype(np.int64)
    k5_cum = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(counts)])

    meta = {
        "n_parent": int(vecs.shape[0]),
        "w1cov_min": float(w1cov_min),
        "w1_max": float(w1_max),
        "w1w2_min": float(w1w2_min),
        "b_cut_deg": float(b_cut_deg),
        "nside_counts": int(nside_counts),
        "parent_k5_nonzero": int(np.sum(counts > 0)),
    }
    return ParentIndex(parent_vec=vecs, parent_pix64=pixs, k5_cum=k5_cum), meta


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
    nside_counts: int,
    npix: int,
    n_epoch: int,
    flux_w1_min_ujy: float,
    ratio_max: float,
    snr_w1_min: float,
    snr_w2_min: float,
    apply_color_cut: bool,
    match_radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (counts[n_epoch,npix], n_sel[n_epoch], mjd_sum[n_epoch], mjd_cnt[n_epoch])."""

    k5 = parse_k5_from_path(s3_rel_path)
    parent = _PARENT
    if parent is None:
        raise RuntimeError("Worker parent index not initialized")

    parent_vec, parent_pix64 = parent.slice_for_k5(k5)
    if parent_vec.shape[0] == 0:
        zc = np.zeros((n_epoch, npix), dtype=np.int64)
        zn = np.zeros(n_epoch, dtype=np.int64)
        zm = np.zeros(n_epoch, dtype=np.float64)
        zmc = np.zeros(n_epoch, dtype=np.int64)
        return zc, zn, zm, zmc

    s3_path = f"{S3_PREFIX}/{s3_rel_path}"
    cols = [
        "unwise_detid",
        "band",
        "EPOCH",
        "ra",
        "dec",
        "flux",
        "dflux",
        "primary",
        "flags_unwise",
        "flags_info",
        "MJDMEAN",
    ]
    t = read_parquet_table(s3_path, fs=fs, columns=cols)

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

    cos_radius = float(math.cos(math.radians(float(match_radius_arcsec) / 3600.0)))

    for e in np.unique(epoch[good]):
        e = int(e)
        if e < 0 or e >= n_epoch:
            continue

        sel_e = good & (epoch == e)
        sel1 = sel_e & (band == 1)
        sel2 = sel_e & (band == 2)
        if not np.any(sel1) or not np.any(sel2):
            continue

        flux1 = flux[sel1]
        dflux1 = dflux[sel1]
        snr1 = flux1 / dflux1
        keep1 = (flux1 >= float(flux_w1_min_ujy)) & (snr1 >= float(snr_w1_min))
        if not np.any(keep1):
            continue

        flux2 = flux[sel2]
        dflux2 = dflux[sel2]
        snr2 = flux2 / dflux2
        keep2 = snr2 >= float(snr_w2_min)
        if not np.any(keep2):
            continue

        k1 = key[sel1][keep1]
        k2 = key[sel2][keep2]
        pos = np.searchsorted(k2, k1)
        m = pos < k2.size
        if np.any(m):
            m[m] &= (k2[pos[m]] == k1[m])
        if not np.any(m):
            continue

        flux1m = flux1[keep1][m]
        flux2m = flux2[keep2][pos[m]]

        if apply_color_cut:
            mcol = flux1m <= float(ratio_max) * flux2m
            if not np.any(mcol):
                continue
        else:
            mcol = slice(None)

        ra_sel = ra[sel1][keep1][m][mcol]
        dec_sel = dec[sel1][keep1][m][mcol]
        mjd_sel = mjd[sel1][keep1][m][mcol]

        det_vec = radec_to_unitvec(ra_sel, dec_sel)
        match_idx = match_to_parent(det_vec, parent_vec, cos_radius=cos_radius)
        okm = match_idx >= 0
        if not np.any(okm):
            continue

        # Deduplicate by parent object index (not by pixel).
        pobj = np.unique(match_idx[okm].astype(np.int64, copy=False))
        if pobj.size == 0:
            continue

        pix = parent_pix64[pobj]
        bc = np.bincount(pix, minlength=npix).astype(np.int64)
        counts[e] += bc
        n_sel[e] += int(pobj.size)
        mjd_sum[e] += float(np.sum(mjd_sel[okm]))
        mjd_cnt[e] += int(np.sum(okm))

    return counts, n_sel, mjd_sum, mjd_cnt


def fit_poisson_glm_dipole(y: np.ndarray, nhat: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
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

    cov = None
    try:
        eta = np.clip(X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        fisher = X.T @ (mu[:, None] * X)
        cov = np.linalg.inv(fisher)
    except Exception:  # noqa: BLE001
        cov = None

    return beta, cov


def run_shard(
    shard_idx: int,
    shard_paths: list[str],
    *,
    outdir: Path,
    nside_counts: int,
    npix: int,
    n_epoch: int,
    flux_w1_min_ujy: float,
    ratio_max: float,
    snr_w1_min: float,
    snr_w2_min: float,
    apply_color_cut: bool,
    match_radius_arcsec: float,
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
            nside_counts=nside_counts,
            npix=npix,
            n_epoch=n_epoch,
            flux_w1_min_ujy=flux_w1_min_ujy,
            ratio_max=ratio_max,
            snr_w1_min=snr_w1_min,
            snr_w2_min=snr_w2_min,
            apply_color_cut=apply_color_cut,
            match_radius_arcsec=match_radius_arcsec,
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


def main(argv: Iterable[str] | None = None) -> int:
    _ensure_env_sane()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=None, help="Output directory (default: outputs/epoch_dipole_catwise_parent_<UTC>).")

    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)

    ap.add_argument("--w1-max", type=float, default=16.4)
    ap.add_argument("--w1w2-min", type=float, default=0.8)
    ap.add_argument("--snr-w1-min", type=float, default=5.0)
    ap.add_argument("--snr-w2-min", type=float, default=0.0)
    ap.add_argument("--apply-color-cut", action="store_true", help="Re-apply W1-W2>=min using epoch photometry.")

    ap.add_argument("--match-radius-arcsec", type=float, default=2.0)

    ap.add_argument("--n-epoch", type=int, default=17, help="Number of EPOCH bins (default 17: 0..16).")
    ap.add_argument("--n-shards", type=int, default=64)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=5)
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

    outdir = Path(args.outdir or f"outputs/epoch_dipole_catwise_parent_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    row_csv = Path(args.row_counts_csv)
    if not row_csv.exists():
        print(f"Downloading row-counts csv -> {row_csv}", flush=True)
        download_row_counts_csv(row_csv)

    items = load_row_counts(row_csv)
    if args.max_files is not None:
        items = items[: int(args.max_files)]

    print("Building CatWISE parent index...", flush=True)
    parent, parent_meta = build_parent_index(
        catwise_catalog=Path(args.catwise_catalog),
        exclude_mask_fits=Path(args.exclude_mask_fits),
        nside_counts=int(args.nside),
        b_cut_deg=float(args.b_cut),
        w1cov_min=float(args.w1cov_min),
        w1_max=float(args.w1_max),
        w1w2_min=float(args.w1w2_min),
    )

    # Prefilter to only healpix_k5 partitions that contain at least one parent object.
    nonzero_k5 = np.where(np.diff(parent.k5_cum) > 0)[0].astype(int)
    nonzero_k5_set = set(int(x) for x in nonzero_k5.tolist())
    items_f = []
    for p, n in items:
        k5 = parse_k5_from_path(p)
        if k5 in nonzero_k5_set:
            items_f.append((p, n))

    print(f"Files total={len(items)} after_k5_filter={len(items_f)}", flush=True)

    import healpy as hp

    nside = int(args.nside)
    npix = hp.nside2npix(nside)
    n_epoch = int(args.n_epoch)

    flux_w1_min_ujy = float(F0_W1_JY * 1e6 * 10.0 ** (-0.4 * float(args.w1_max)))
    ratio_max = float((F0_W1_JY / F0_W2_JY) * 10.0 ** (-0.4 * float(args.w1w2_min)))

    cfg = {
        "s3_prefix": S3_PREFIX,
        "row_counts_url": ROW_COUNTS_URL,
        "nside": nside,
        "b_cut_deg": float(args.b_cut),
        "w1cov_min": float(args.w1cov_min),
        "w1_max_vega": float(args.w1_max),
        "w1w2_min_vega": float(args.w1w2_min),
        "snr_w1_min": float(args.snr_w1_min),
        "snr_w2_min": float(args.snr_w2_min),
        "apply_color_cut": bool(args.apply_color_cut),
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "n_epoch": n_epoch,
        "flux_w1_min_ujy": flux_w1_min_ujy,
        "ratio_max_flux1_over_flux2": ratio_max,
        "n_shards": int(args.n_shards),
        "workers": int(args.workers),
        "checkpoint_every": int(args.checkpoint_every),
        "max_files": None if args.max_files is None else int(args.max_files),
        "parent_meta": parent_meta,
        "timestamp_utc": utc_tag(),
    }
    (outdir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    parent_npz = outdir / "parent_index.npz"
    np.savez_compressed(parent_npz, parent_vec=parent.parent_vec, parent_pix64=parent.parent_pix64, k5_cum=parent.k5_cum)

    shards = binpack_shards(items_f, int(args.n_shards))

    print(f"Starting pool: workers={int(args.workers)} shards={len(shards)}", flush=True)

    import multiprocessing as mp

    t0 = time.time()
    results: list[dict] = []

    with mp.get_context("spawn").Pool(
        processes=int(args.workers),
        initializer=_init_worker,
        initargs=(str(parent_npz),),
    ) as pool:
        jobs = []
        for si, spaths in enumerate(shards):
            jobs.append(
                pool.apply_async(
                    run_shard,
                    (si, spaths),
                    dict(
                        outdir=outdir,
                        nside_counts=nside,
                        npix=npix,
                        n_epoch=n_epoch,
                        flux_w1_min_ujy=flux_w1_min_ujy,
                        ratio_max=ratio_max,
                        snr_w1_min=float(args.snr_w1_min),
                        snr_w2_min=float(args.snr_w2_min),
                        apply_color_cut=bool(args.apply_color_cut),
                        match_radius_arcsec=float(args.match_radius_arcsec),
                        checkpoint_every=int(args.checkpoint_every),
                    ),
                )
            )

        for j in jobs:
            r = j.get()
            results.append(r)
            done = len(results)
            dt = time.time() - t0
            print(f"[main] done {done}/{len(jobs)} shards (elapsed {dt/3600:.2f} hr)", flush=True)

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

    # Fit dipole per epoch on the same seen mask as the parent definition.
    mask = build_secrest_mask(
        nside=nside,
        catwise_catalog=Path(args.catwise_catalog),
        exclude_mask_fits=Path(args.exclude_mask_fits),
        b_cut_deg=float(args.b_cut),
        w1cov_min=float(args.w1cov_min),
    )
    seen = mask.seen

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
            {"epoch": int(e), "N": int(n_sel[e]), "mjd_mean": mjd_mean, "D": D, "sigma_D": sigma_D, "beta": [float(x) for x in beta]}
        )
        print(f"Epoch {e:02d}: N={int(n_sel[e])} D={D:.5f} +/- {sigma_D:.5f} mjd_mean={mjd_mean:.1f}", flush=True)

    summary = {"config": cfg, "shards": results, "epochs": summary_rows}
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
        plt.title("CatWISE-parent epoch-resolved dipole amplitude")
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
