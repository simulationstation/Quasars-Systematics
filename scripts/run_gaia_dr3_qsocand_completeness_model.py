#!/usr/bin/env python3
"""
Externally validated, map-level completeness model using Gaia DR3 QSO candidates (all-sky).

Goal
----
Build a *spatial* completeness/selection model for the Secrest+22 "accepted" CatWISE AGN sample using an
all-sky external QSO-like truth set:

  Gaia DR3 QSO candidates (CDS I/356, file qsocand.dat; column PQSO)

We treat Gaia qsocand as a "truth-like" external tracer and define labels:
  y = 1 if a Gaia QSO candidate has a CatWISE-accepted match within a small radius
  y = 0 otherwise

We then fit a logistic model for P(y=1 | x) using *map-level / geometry* predictors available everywhere:
  - unWISE logNexp (depth proxy; map-level, Galactic coords)
  - ecliptic geometry proxies: |β|, sinλ, cosλ (from RA/Dec; global)

This yields an all-sky, externally trained map that can be used as a *log-completeness offset* template in
Poisson GLM dipole analyses.

Important limitation
--------------------
Gaia qsocand does not provide W1. This is therefore not a magnitude-dependent completeness model; it is an
all-sky, externally trained *spatial* completeness proxy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def unitvec_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float) % 360.0)
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cosd = np.cos(dec)
    return np.column_stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)]).astype(float)


# ICRS/J2000 -> Galactic rotation matrix (same numerical values used widely, e.g. Hipparcos/IAU)
_R_EQ_TO_GAL = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=float,
)


def gal_ipix_from_unitvec_eq(v_eq: np.ndarray, *, nside: int) -> np.ndarray:
    """Fast Galactic ipix from ICRS unit vectors using the fixed rotation matrix."""
    import healpy as hp

    v_eq = np.asarray(v_eq, dtype=float)
    v_gal = v_eq @ _R_EQ_TO_GAL.T
    lon = np.arctan2(v_gal[:, 1], v_gal[:, 0]) % (2.0 * math.pi)
    lat = np.arcsin(np.clip(v_gal[:, 2], -1.0, 1.0))
    theta = (0.5 * math.pi) - lat
    phi = lon
    return hp.ang2pix(int(nside), theta, phi, nest=False)


def ecl_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast J2000 ecliptic lon/lat from ICRS RA/Dec (same formulas as SDSS model script)."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float) % 360.0)
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    sin_ra = np.sin(ra)
    cos_ra = np.cos(ra)

    eps = math.radians(23.4392911)  # J2000 obliquity
    sin_eps = math.sin(eps)
    cos_eps = math.cos(eps)

    sin_beta = sin_dec * cos_eps - cos_dec * sin_eps * sin_ra
    beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))

    y = sin_ra * cos_eps + np.tan(dec) * sin_eps
    x = cos_ra
    lam = np.arctan2(y, x) % (2.0 * math.pi)

    return np.rad2deg(lam), np.rad2deg(beta)


def zscore(x: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xm = x[mask]
    mu = float(np.median(xm))
    sig = float(np.std(xm))
    if not np.isfinite(sig) or sig <= 0.0:
        sig = 1.0
    return (x - mu) / sig


def safe_float_col(arr: object) -> np.ndarray:
    return np.asarray(arr, dtype=float)


def build_secrest_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """mask_zeros + exclusion discs + galactic plane cut (pixel centers). Returns (mask, seen)."""
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        neigh = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            neigh[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[neigh] = True  # includes -1 indexing as in Secrest utilities

    if exclude_mask_fits:
        from astropy.coordinates import SkyCoord
        from astropy.table import Table
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

    _, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return mask, ~mask


@dataclass(frozen=True)
class GaiaParseConfig:
    # CDS ReadMe: Byte-by-byte Description of file: qsocand.dat
    pqso_slice: slice = slice(186, 200)  # Bytes 187-200
    ra_slice: slice = slice(948, 971)  # Bytes 949-971
    dec_slice: slice = slice(972, 995)  # Bytes 973-995


def _parse_float_field(line: str, s: slice) -> float | None:
    txt = line[s].strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qsocand-gz", default="data/external/gaia_dr3_extragal/qsocand.dat.gz")
    ap.add_argument(
        "--catwise-catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument(
        "--unwise-lognexp-map",
        default="data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits",
        help="HEALPix map in Galactic coords.",
    )
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--pqso-min", type=float, default=0.8)
    ap.add_argument("--match-radius-arcsec", type=float, default=2.0)
    ap.add_argument(
        "--feature-set",
        choices=["external_only", "external_plus_catwise_maps"],
        default="external_only",
        help=(
            "Which predictors to use. 'external_only' uses (logNexp, |β|, sinλ, cosλ). "
            "'external_plus_catwise_maps' additionally uses (log W1cov_mean, EBV_mean) binned from CatWISE."
        ),
    )
    ap.add_argument("--chunk-lines", type=int, default=250000, help="Chunk size in PQSO-filtered rows.")
    ap.add_argument("--progress-every", type=int, default=500000)
    ap.add_argument("--n-records", type=int, default=6649162, help="Expected record count for percent progress.")
    ap.add_argument("--max-lines", type=int, default=None, help="Optional cap for smoke tests.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--export-report-dir", default=None, help="If set, copy final artifacts into this report folder.")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    from astropy.io import fits
    import healpy as hp
    from scipy.spatial import cKDTree
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    outdir = Path(args.outdir or f"outputs/external_completeness_gaia_qsocand_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    qsocand_path = Path(str(args.qsocand_gz))
    if not qsocand_path.exists():
        raise SystemExit(f"Missing {qsocand_path} (download from CDS I/356, qsocand.dat.gz).")

    # -------------------- Load CatWISE + build Secrest footprint --------------------
    with fits.open(str(args.catwise_catalog), memmap=True) as hdul:
        c = hdul[1].data
        cat_ra = safe_float_col(c["ra"])
        cat_dec = safe_float_col(c["dec"])
        cat_l = safe_float_col(c["l"])
        cat_b = safe_float_col(c["b"])
        cat_w1cov = safe_float_col(c["w1cov"])
        cat_ebv = safe_float_col(c["ebv"]) if "ebv" in c.names else None

    cat_base = (
        np.isfinite(cat_ra)
        & np.isfinite(cat_dec)
        & np.isfinite(cat_l)
        & np.isfinite(cat_b)
        & np.isfinite(cat_w1cov)
        & (cat_w1cov >= float(args.w1cov_min))
    )
    if cat_ebv is not None:
        cat_base &= np.isfinite(cat_ebv)

    theta_base = np.deg2rad(90.0 - cat_b[cat_base])
    phi_base = np.deg2rad(cat_l[cat_base])
    ipix_cat_base = hp.ang2pix(int(args.nside), theta_base, phi_base, nest=False)
    _, seen = build_secrest_mask(
        nside=int(args.nside),
        ipix_base=ipix_cat_base,
        exclude_mask_fits=str(args.exclude_mask_fits) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )

    # Map-level predictors derived from the CatWISE "accepted" table (coverage + EBV proxy).
    # These are not number-count fluctuations; they are per-source ancillary values that can be binned to pixels.
    npix = hp.nside2npix(int(args.nside))
    cnt_base = np.bincount(ipix_cat_base.astype(np.int64), minlength=npix).astype(float)
    sum_w1cov = np.bincount(ipix_cat_base.astype(np.int64), weights=cat_w1cov[cat_base].astype(float), minlength=npix).astype(float)
    w1cov_mean = np.divide(sum_w1cov, cnt_base, out=np.zeros_like(sum_w1cov), where=cnt_base != 0.0)
    if cat_ebv is None:
        ebv_mean = np.zeros(npix, dtype=float)
    else:
        sum_ebv = np.bincount(ipix_cat_base.astype(np.int64), weights=cat_ebv[cat_base].astype(float), minlength=npix).astype(float)
        ebv_mean = np.divide(sum_ebv, cnt_base, out=np.zeros_like(sum_ebv), where=cnt_base != 0.0)

    # Matching tree: restrict to seen footprint + w1cov cut.
    ipix_cat_all = hp.ang2pix(int(args.nside), np.deg2rad(90.0 - cat_b), np.deg2rad(cat_l), nest=False)
    cat_keep = cat_base & seen[ipix_cat_all]
    cat_ra_k = cat_ra[cat_keep]
    cat_dec_k = cat_dec[cat_keep]
    tree = cKDTree(unitvec_from_radec_deg(cat_ra_k, cat_dec_k))

    theta_max = np.deg2rad(float(args.match_radius_arcsec) / 3600.0)
    dmax = float(2.0 * np.sin(theta_max / 2.0))

    # External map-level depth proxy
    lognexp = hp.read_map(str(args.unwise_lognexp_map), verbose=False)
    if int(hp.get_nside(lognexp)) != int(args.nside):
        lognexp = hp.ud_grade(lognexp, nside_out=int(args.nside), order_in="RING", order_out="RING", power=0)
    lognexp = np.asarray(lognexp, dtype=float)

    # -------------------- Parse Gaia + build training arrays (chunked) --------------------
    cfg = GaiaParseConfig()

    feats: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    g_chunks: list[np.ndarray] = []

    n_lines = 0
    kept_ra: list[float] = []
    kept_dec: list[float] = []

    def process_chunk(ra_chunk: np.ndarray, dec_chunk: np.ndarray) -> int:
        if ra_chunk.size == 0:
            return 0

        v_eq = unitvec_from_radec_deg(ra_chunk, dec_chunk)
        ipix = gal_ipix_from_unitvec_eq(v_eq, nside=int(args.nside))
        keep = seen[ipix]
        if not np.any(keep):
            return int(ra_chunk.size)

        v_eq = v_eq[keep]
        ipix = ipix[keep]
        ra_chunk = ra_chunk[keep]
        dec_chunk = dec_chunk[keep]

        x_depth = lognexp[ipix]
        x_cov = np.log(np.clip(w1cov_mean[ipix], 1.0, np.inf))
        x_ebv = ebv_mean[ipix]
        ok_depth = np.isfinite(x_depth) & (x_depth != hp.UNSEEN)
        if not np.any(ok_depth):
            return int(ra_chunk.size)

        v_eq = v_eq[ok_depth]
        ipix = ipix[ok_depth]
        ra_chunk = ra_chunk[ok_depth]
        dec_chunk = dec_chunk[ok_depth]
        x_depth = x_depth[ok_depth]
        x_cov = x_cov[ok_depth]
        x_ebv = x_ebv[ok_depth]

        elon, elat = ecl_from_radec_deg(ra_chunk, dec_chunk)
        abs_elat = np.abs(elat)
        sin_elon = np.sin(np.deg2rad(elon))
        cos_elon = np.cos(np.deg2rad(elon))
        g = np.floor((elon % 360.0) / 15.0).astype(int)

        dist, _ = tree.query(v_eq, k=1, workers=-1)
        matched = np.asarray(dist, dtype=float) <= dmax

        if args.feature_set == "external_only":
            feats.append(np.column_stack([x_depth, abs_elat, sin_elon, cos_elon]).astype(float))
        else:
            feats.append(np.column_stack([x_depth, x_cov, x_ebv, abs_elat, sin_elon, cos_elon]).astype(float))
        y_chunks.append(matched.astype(int))
        g_chunks.append(g.astype(int))
        return int(ra_chunk.size)

    with gzip.open(str(qsocand_path), mode="rt", encoding="ascii", errors="ignore") as f:
        for line in f:
            n_lines += 1
            if args.max_lines is not None and n_lines > int(args.max_lines):
                break
            if len(line) < 995:
                continue

            pqso = _parse_float_field(line, cfg.pqso_slice)
            if pqso is None or not np.isfinite(pqso) or pqso < float(args.pqso_min):
                continue

            ra = _parse_float_field(line, cfg.ra_slice)
            dec = _parse_float_field(line, cfg.dec_slice)
            if ra is None or dec is None:
                continue
            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue

            kept_ra.append(float(ra))
            kept_dec.append(float(dec))

            if len(kept_ra) >= int(args.chunk_lines):
                process_chunk(np.asarray(kept_ra, dtype=float), np.asarray(kept_dec, dtype=float))
                kept_ra.clear()
                kept_dec.clear()

            if (n_lines % int(args.progress_every)) == 0:
                n_used = int(sum(len(x) for x in y_chunks))
                frac = min(1.0, float(n_lines) / float(args.n_records or 1))
                print(
                    f"[gaia-model] lines={n_lines:,} used={n_used:,} progress={100.0*frac:5.1f}%",
                    flush=True,
                )

    process_chunk(np.asarray(kept_ra, dtype=float), np.asarray(kept_dec, dtype=float))

    nfeat = 4 if args.feature_set == "external_only" else 6
    X = np.concatenate(feats, axis=0) if feats else np.empty((0, nfeat), dtype=float)
    y_arr = np.concatenate(y_chunks, axis=0) if y_chunks else np.empty((0,), dtype=int)
    g_arr = np.concatenate(g_chunks, axis=0) if g_chunks else np.empty((0,), dtype=int)

    # Standardize features on the training mask (all rows).
    Xs = X.copy()
    mu = np.median(Xs, axis=0)
    sig = np.std(Xs, axis=0)
    sig = np.where(sig <= 0.0, 1.0, sig)
    Xs = (Xs - mu) / sig

    # -------------------- Sky-holdout validation --------------------
    gkf = GroupKFold(n_splits=5)
    p_cv = np.zeros(len(y_arr), dtype=float)
    base_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)

    for tr, te in gkf.split(Xs, y_arr, groups=g_arr):
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        lr.fit(Xs[tr], y_arr[tr])
        p_cv[te] = lr.predict_proba(Xs[te])[:, 1]

    auc = float(roc_auc_score(y_arr, p_cv))
    brier = float(brier_score_loss(y_arr, p_cv))

    # Fit final model
    base_lr.fit(Xs, y_arr)
    coef = base_lr.coef_.reshape(-1).astype(float)
    intercept = float(base_lr.intercept_.reshape(-1)[0])

    # -------------------- Map products --------------------
    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)
    # Pixel-center ecliptic coords: convert pixel centers (Galactic lon/lat) -> ICRS -> ecliptic
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic").icrs
    elon_p, elat_p = ecl_from_radec_deg(sc_pix.ra.deg, sc_pix.dec.deg)
    abs_elat_p = np.abs(elat_p)
    sin_elon_p = np.sin(np.deg2rad(elon_p))
    cos_elon_p = np.cos(np.deg2rad(elon_p))

    ok_pix = seen & np.isfinite(lognexp) & (lognexp != hp.UNSEEN)
    if args.feature_set == "external_only":
        Xp = np.column_stack([lognexp, abs_elat_p, sin_elon_p, cos_elon_p])
    else:
        Xp = np.column_stack(
            [
                lognexp,
                np.log(np.clip(w1cov_mean, 1.0, np.inf)),
                ebv_mean,
                abs_elat_p,
                sin_elon_p,
                cos_elon_p,
            ]
        )
    Xp = (Xp - mu) / sig
    lin = intercept + Xp @ coef
    p_map = sigmoid(lin)

    p_map_out = np.full(npix, hp.UNSEEN, dtype=float)
    p_map_out[ok_pix] = p_map[ok_pix]

    # Log-completeness offset for GLM: log(p) centered on median across seen pixels.
    p_clip = np.clip(p_map_out, 1e-6, 1.0)
    logp = np.log(p_clip)
    med = float(np.median(logp[ok_pix]))
    logp_off = np.full(npix, hp.UNSEEN, dtype=float)
    logp_off[ok_pix] = logp[ok_pix] - med

    hp.write_map(str(outdir / "p_map_nside64.fits"), p_map_out, overwrite=True, dtype=float)
    hp.write_map(str(outdir / "logp_offset_map_nside64.fits"), logp_off, overwrite=True, dtype=float)

    meta = {
        "gaia_qsocand": {"path": str(qsocand_path), "pqso_min": float(args.pqso_min)},
        "catwise": {
            "catalog": str(args.catwise_catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "w1cov_min": float(args.w1cov_min),
            "b_cut_deg": float(args.b_cut),
            "match_radius_arcsec": float(args.match_radius_arcsec),
        },
        "nside": int(args.nside),
        "n_used": int(len(y_arr)),
        "positive_rate": float(np.mean(y_arr)),
        "feature_set": str(args.feature_set),
        "features": (
            ["lognexp", "abs_elat_deg", "sin_elon", "cos_elon"]
            if args.feature_set == "external_only"
            else ["lognexp", "log_w1cov_mean", "ebv_mean", "abs_elat_deg", "sin_elon", "cos_elon"]
        ),
        "standardize": {"median": [float(x) for x in mu], "std": [float(x) for x in sig]},
        "logreg": {"intercept": intercept, "coef": [float(x) for x in coef]},
        "cv": {
            "scheme": "GroupKFold on ecliptic longitude wedges (15 deg bins)",
            "n_splits": 5,
            "roc_auc": auc,
            "brier": brier,
        },
        "maps": {"p_map_fits": "p_map_nside64.fits", "logp_offset_map_fits": "logp_offset_map_nside64.fits"},
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    if args.make_plots:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5.0, 4.0))
        nb = 20
        bins = np.linspace(0.0, 1.0, nb + 1)
        which = np.digitize(p_cv, bins) - 1
        p_mean = np.full(nb, np.nan)
        y_mean = np.full(nb, np.nan)
        for bidx in range(nb):
            m = which == bidx
            if np.any(m):
                p_mean[bidx] = float(np.mean(p_cv[m]))
                y_mean[bidx] = float(np.mean(y_arr[m]))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        plt.plot(p_mean, y_mean, "o-", label="Gaia qsocand holdout")
        plt.xlabel("Predicted P(in CatWISE)")
        plt.ylabel("Empirical fraction")
        plt.title(f"Calibration (AUC={auc:.3f}, Brier={brier:.3f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "calibration_curve.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10.0, 4.0))
        hp.mollview(logp_off, title="Gaia-trained log P(offset) map", unit="log prob", fig=plt.gcf().number)
        plt.savefig(outdir / "logp_offset_mollweide.png", dpi=200)
        plt.close()

    if args.export_report_dir:
        report_dir = Path(str(args.export_report_dir))
        (report_dir / "data").mkdir(parents=True, exist_ok=True)
        (report_dir / "figures").mkdir(parents=True, exist_ok=True)

        (report_dir / "data" / "meta.json").write_text((outdir / "meta.json").read_text())
        for fn in ["p_map_nside64.fits", "logp_offset_map_nside64.fits"]:
            (report_dir / "data" / fn).write_bytes((outdir / fn).read_bytes())
        for fn in ["calibration_curve.png", "logp_offset_mollweide.png"]:
            p = outdir / fn
            if p.exists():
                (report_dir / "figures" / fn).write_bytes(p.read_bytes())

    print(json.dumps({**meta["cv"], "n_used": int(len(y_arr)), "positive_rate": float(np.mean(y_arr))}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
