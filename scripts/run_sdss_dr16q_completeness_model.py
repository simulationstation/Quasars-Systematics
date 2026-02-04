#!/usr/bin/env python3
"""
Externally validated, map-level completeness model using SDSS DR16Q as a truth set.

Goal
----
Build a model for the probability that an *externally selected* quasar (SDSS DR16Q)
appears in the Secrest+22 "accepted" CatWISE AGN candidate catalog, as a function of:
  - source brightness (WISE W1),
  - map-level imaging depth proxy (unWISE logNexp HEALPix map),
  - scan-geometry proxies in ecliptic coordinates (|β|, sinλ, cosλ).

This provides an *externally validated* (i.e., not trained on CatWISE count fluctuations)
completeness/selection model that can be converted into an effective limiting-magnitude
shift δm(pix) and used as a map-level template/offset in dipole analyses.

Notes
-----
- SDSS DR16Q is optical spectroscopic, so membership is independent of WISE scan depth.
- W1 magnitudes used here are taken from DR16Q if available; column names vary, so we
  detect W1/W2 columns heuristically.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def ecl_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast J2000 ecliptic lon/lat from ICRS RA/Dec.

    Uses the standard obliquity ε (J2000) and the conventional formulae:
      sin β = sin δ cos ε − cos δ sin ε sin α
      tan λ = (sin α cos ε + tan δ sin ε) / cos α
    """

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


def find_first_col(names: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def safe_float_col(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    return out


def unitvec_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float) % 360.0)
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cosd = np.cos(dec)
    return np.column_stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)]).astype(float)


@dataclass(frozen=True)
class MatchResult:
    matched: np.ndarray  # bool per truth object
    sep_arcsec: np.ndarray  # angular separation to nearest CatWISE source


def match_truth_to_catwise(
    *,
    truth_ra: np.ndarray,
    truth_dec: np.ndarray,
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    max_sep_arcsec: float,
) -> MatchResult:
    """Nearest-neighbour match using a 3D unit-vector KD-tree (fast for millions of points)."""
    from scipy.spatial import cKDTree

    truth_v = unitvec_from_radec_deg(truth_ra, truth_dec)
    cat_v = unitvec_from_radec_deg(cat_ra, cat_dec)

    tree = cKDTree(cat_v)

    theta_max = np.deg2rad(float(max_sep_arcsec) / 3600.0)
    dmax = float(2.0 * np.sin(theta_max / 2.0))

    dist, _ = tree.query(truth_v, k=1, workers=-1)
    dist = np.asarray(dist, dtype=float)
    matched = dist <= dmax

    theta = 2.0 * np.arcsin(np.clip(dist / 2.0, 0.0, 1.0))
    sep_arcsec = np.rad2deg(theta) * 3600.0
    return MatchResult(matched=np.asarray(matched, dtype=bool), sep_arcsec=np.asarray(sep_arcsec, dtype=float))


def build_secrest_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Implements the Secrest-style HEALPix footprint mask (mask_zeros + exclude discs + |b| cut).

    Returns (mask, seen) where mask=True means masked and seen=True means unmasked.
    """

    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    # mask_zeros on the "base" W1cov>=80 parent map
    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        neigh = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            neigh[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        # Match Secrest behaviour (includes -1 neighbour indexing last pixel).
        mask[neigh] = True

    # exclude discs
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
                disc = hp.query_disc(
                    nside=int(nside),
                    vec=vec,
                    radius=float(rad),
                    inclusive=True,
                    nest=False,
                )
                mask[disc] = True

    # galactic plane cut on pixel centers
    _, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return mask, ~mask


def compute_alpha_edge(
    w1: np.ndarray,
    cut: float,
    *,
    delta: float,
) -> float:
    """Estimate alpha_edge = d ln N(<m) / dm at m=cut via a finite difference."""

    w1 = np.asarray(w1, dtype=float)
    n1 = int(np.sum(w1 <= float(cut)))
    n0 = int(np.sum(w1 <= float(cut) - float(delta)))
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    return float((math.log(n1) - math.log(n0)) / float(delta))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dr16q-fits",
        default="data/external/sdss_dr16q/DR16Q_v4.fits",
        help="SDSS DR16Q FITS (from SDSS SAS).",
    )
    ap.add_argument(
        "--catwise-catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument(
        "--depth-map-fits",
        default="data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits",
        help="HEALPix depth proxy map in Galactic coords (RING).",
    )
    ap.add_argument(
        "--depth-map-name",
        default="unwise_lognexp_nside64",
        help="Short label for the depth map (stored in metadata and filenames).",
    )
    ap.add_argument(
        "--depth-map-transform",
        choices=["none", "log", "log10"],
        default="none",
        help=(
            "Optional transform applied to the depth map before use as a feature. "
            "Use 'none' for logNexp maps; use 'log10' or 'log' for positive linear-depth maps."
        ),
    )
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--match-radius-arcsec", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--max-truth",
        type=int,
        default=None,
        help="Optional cap on truth-set size (random subsample after quality cuts).",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument(
        "--feature-set",
        choices=["depth_only", "depth_plus_ecliptic"],
        default="depth_plus_ecliptic",
        help=(
            "Which predictors to use for the external completeness model. "
            "'depth_only' uses (W1, logNexp, W1×logNexp). "
            "'depth_plus_ecliptic' additionally uses (|β|, sinλ, cosλ) as scan-geometry proxies."
        ),
    )
    ap.add_argument(
        "--alpha-edge-cut",
        type=float,
        default=16.6,
        help="Reference W1 cut used for alpha_edge scaling diagnostics.",
    )
    ap.add_argument(
        "--alpha-edge-delta",
        type=float,
        default=0.05,
        help="Finite-difference step (mag) for alpha_edge estimate.",
    )
    args = ap.parse_args()

    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import healpy as hp
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss

    outdir = Path(args.outdir or f"outputs/external_completeness_dr16q_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------- Load CatWISE (for mask + matching target) --------------------
    with fits.open(args.catwise_catalog, memmap=True) as hdul:
        c = hdul[1].data
        cat_ra = safe_float_col(c["ra"])
        cat_dec = safe_float_col(c["dec"])
        cat_l = safe_float_col(c["l"])
        cat_b = safe_float_col(c["b"])
        cat_w1 = safe_float_col(c["w1"])
        cat_w1cov = safe_float_col(c["w1cov"])

    cat_base = (
        np.isfinite(cat_ra)
        & np.isfinite(cat_dec)
        & np.isfinite(cat_l)
        & np.isfinite(cat_b)
        & np.isfinite(cat_w1cov)
        & (cat_w1cov >= float(args.w1cov_min))
    )
    theta = np.deg2rad(90.0 - cat_b[cat_base])
    phi = np.deg2rad(cat_l[cat_base])
    ipix_cat_base = hp.ang2pix(int(args.nside), theta, phi, nest=False)
    mask, seen = build_secrest_mask(
        nside=int(args.nside),
        ipix_base=ipix_cat_base,
        exclude_mask_fits=str(args.exclude_mask_fits) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )

    # For matching: restrict to the seen footprint + w1cov cut (keeps memory smaller)
    ipix_cat_all = hp.ang2pix(
        int(args.nside),
        np.deg2rad(90.0 - cat_b),
        np.deg2rad(cat_l),
        nest=False,
    )
    cat_keep = cat_base & seen[ipix_cat_all]
    cat_ra_k = cat_ra[cat_keep]
    cat_dec_k = cat_dec[cat_keep]
    cat_w1_k = cat_w1[cat_keep]

    # alpha_edge diagnostic from CatWISE itself (global, not spatial)
    alpha_edge_ref = compute_alpha_edge(cat_w1_k, float(args.alpha_edge_cut), delta=float(args.alpha_edge_delta))

    # -------------------- Load DR16Q truth set --------------------
    dr16q_path = Path(str(args.dr16q_fits))
    if not dr16q_path.exists():
        raise SystemExit(f"Missing DR16Q FITS: {dr16q_path}")

    with fits.open(str(dr16q_path), memmap=True) as hdul:
        t = hdul[1].data
        names = list(t.names)

        ra_col = find_first_col(names, ["ra", "RA", "RAJ2000"])
        dec_col = find_first_col(names, ["dec", "DEC", "DEJ2000"])
        if ra_col is None or dec_col is None:
            raise SystemExit(f"Could not find RA/DEC columns in DR16Q. Available: {names[:50]}")

        # Try common WISE column names (DR16Q contains multiple sets; prefer unWISE/AllWISE magnitudes if present).
        w1_col = find_first_col(
            names,
            [
                "W1MAG",
                "W1_MAG",
                "W1",
                "W1mag",
                "W1_MAG_VEGA",
                "W1MAG_VEGA",
                "W1MAG_vega",
            ],
        )
        w2_col = find_first_col(names, ["W2MAG", "W2_MAG", "W2", "W2mag"])

        if w1_col is None:
            raise SystemExit(
                "Could not find a W1 magnitude column in DR16Q. "
                f"Available columns (first 80): {names[:80]}"
            )

        truth_ra = safe_float_col(t[ra_col])
        truth_dec = safe_float_col(t[dec_col])
        truth_w1 = safe_float_col(t[w1_col])
        truth_w2 = None if w2_col is None else safe_float_col(t[w2_col])

    ok = np.isfinite(truth_ra) & np.isfinite(truth_dec) & np.isfinite(truth_w1)

    # Optional subsample for speed
    if args.max_truth is not None and int(args.max_truth) < int(ok.sum()):
        rng = np.random.default_rng(int(args.seed))
        idx_ok = np.where(ok)[0]
        choose = rng.choice(idx_ok, size=int(args.max_truth), replace=False)
        keep = np.zeros_like(ok)
        keep[choose] = True
        ok &= keep

    truth_ra = truth_ra[ok]
    truth_dec = truth_dec[ok]
    truth_w1 = truth_w1[ok]
    truth_w2 = None if truth_w2 is None else truth_w2[ok]

    # Compute galactic coords for mask + depth map lookup.
    sc_gal = SkyCoord(truth_ra * u.deg, truth_dec * u.deg, frame="icrs").galactic
    truth_l = sc_gal.l.deg.astype(float)
    truth_b = sc_gal.b.deg.astype(float)
    truth_ipix = hp.ang2pix(
        int(args.nside),
        np.deg2rad(90.0 - truth_b),
        np.deg2rad(truth_l),
        nest=False,
    )
    in_seen = seen[truth_ipix]

    # External map-level depth proxy (map-level; independent of CatWISE count fluctuations).
    depth_map = hp.read_map(str(args.depth_map_fits), verbose=False)
    if int(hp.get_nside(depth_map)) != int(args.nside):
        depth_map = hp.ud_grade(depth_map, nside_out=int(args.nside), order_in="RING", order_out="RING", power=0)
    depth_map = np.asarray(depth_map, dtype=float)
    if args.depth_map_transform == "log":
        depth_map = np.log(np.clip(depth_map, 1e-12, np.inf))
    elif args.depth_map_transform == "log10":
        depth_map = np.log10(np.clip(depth_map, 1e-12, np.inf))

    depth_val = depth_map[truth_ipix]
    ok_depth = np.isfinite(depth_val) & (depth_val != hp.UNSEEN)

    # Apply the analysis footprint (seen pixels only)
    keep = in_seen & ok_depth
    truth_ra = truth_ra[keep]
    truth_dec = truth_dec[keep]
    truth_w1 = truth_w1[keep]
    truth_w2 = None if truth_w2 is None else truth_w2[keep]
    truth_l = truth_l[keep]
    truth_b = truth_b[keep]
    truth_ipix = truth_ipix[keep]
    depth_val = depth_val[keep]

    # Ecliptic geometry proxies (computed from RA/Dec; not from CatWISE).
    elon, elat = ecl_from_radec_deg(truth_ra, truth_dec)
    abs_elat = np.abs(elat)
    sin_elon = np.sin(np.deg2rad(elon))
    cos_elon = np.cos(np.deg2rad(elon))

    # -------------------- Cross-match: DR16Q -> CatWISE --------------------
    match = match_truth_to_catwise(
        truth_ra=truth_ra,
        truth_dec=truth_dec,
        cat_ra=cat_ra_k,
        cat_dec=cat_dec_k,
        max_sep_arcsec=float(args.match_radius_arcsec),
    )
    y = match.matched.astype(int)

    # -------------------- Fit a completeness model --------------------
    # Core features: W1 + logNexp (+ W1*logNexp interaction).
    cols = [truth_w1, depth_val, truth_w1 * depth_val]
    feature_names = ["w1", "depth", "w1_x_depth"]

    if args.feature_set == "depth_plus_ecliptic":
        cols.extend([abs_elat, sin_elon, cos_elon])
        feature_names.extend(["abs_elat_deg", "sin_elon", "cos_elon"])

    X = np.column_stack(cols)

    # Sky-holdout CV groups: ecliptic longitude wedges.
    # 24 groups of 15 degrees each to reduce leakage of spatial structure.
    groups = np.floor((elon % 360.0) / 15.0).astype(int)
    gkf = GroupKFold(n_splits=5)
    p_cv = np.zeros(len(y), dtype=float)

    # Use a mildly regularized logistic regression; max_iter bumped for stability.
    base_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)

    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        lr.fit(X[tr_idx], y[tr_idx])
        p_cv[te_idx] = lr.predict_proba(X[te_idx])[:, 1]

    auc = float(roc_auc_score(y, p_cv))
    brier = float(brier_score_loss(y, p_cv))

    # Fit final model on full dataset for map generation.
    base_lr.fit(X, y)
    coef = base_lr.coef_.reshape(-1).astype(float)
    intercept = float(base_lr.intercept_.reshape(-1)[0])

    # -------------------- Convert to δm(pix) and edge completeness maps --------------------
    npix = hp.nside2npix(int(args.nside))
    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)
    if args.feature_set == "depth_plus_ecliptic":
        # Convert pixel centers (galactic lon/lat) -> ICRS ra/dec -> ecliptic
        sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic").icrs
        elon_p, elat_p = ecl_from_radec_deg(sc_pix.ra.deg, sc_pix.dec.deg)
        abs_elat_p = np.abs(elat_p)
        sin_elon_p = np.sin(np.deg2rad(elon_p))
        cos_elon_p = np.cos(np.deg2rad(elon_p))
    else:
        abs_elat_p = sin_elon_p = cos_elon_p = None

    depth_pix = depth_map.copy()
    ok_pix = seen & np.isfinite(depth_pix) & (depth_pix != hp.UNSEEN)

    # m50 map solves logit(p)=0 for W1:
    # 0 = b0 + b_w1*m + b_x*x + b_w1x*m*x + ...  =>  m = -(b0 + b_x*x + ...)/(b_w1 + b_w1x*x)
    b_w1 = coef[0]
    b_x = coef[1]
    b_w1x = coef[2]
    if args.feature_set == "depth_plus_ecliptic":
        b_abs = coef[3]
        b_sin = coef[4]
        b_cos = coef[5]
    else:
        b_abs = b_sin = b_cos = 0.0

    denom = b_w1 + b_w1x * depth_pix
    denom_safe = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6 + 1e-6, denom)
    num = intercept + b_x * depth_pix
    if args.feature_set == "depth_plus_ecliptic":
        num = num + b_abs * abs_elat_p + b_sin * sin_elon_p + b_cos * cos_elon_p
    m50 = -num / denom_safe

    m50_map = np.full(npix, hp.UNSEEN, dtype=float)
    m50_map[ok_pix] = m50[ok_pix]
    med_m50 = float(np.median(m50_map[ok_pix]))

    dm_map = np.full(npix, hp.UNSEEN, dtype=float)
    dm_map[ok_pix] = m50_map[ok_pix] - med_m50

    # Edge completeness at a representative cut (for visualization)
    cut_ref = float(args.alpha_edge_cut)
    lin_edge = intercept + b_w1 * cut_ref + b_x * depth_pix + b_w1x * cut_ref * depth_pix
    if args.feature_set == "depth_plus_ecliptic":
        lin_edge = lin_edge + b_abs * abs_elat_p + b_sin * sin_elon_p + b_cos * cos_elon_p
    p_edge = sigmoid(lin_edge)
    p_edge_map = np.full(npix, hp.UNSEEN, dtype=float)
    p_edge_map[ok_pix] = p_edge[ok_pix]

    # -------------------- Write outputs --------------------
    meta = {
        "dr16q_fits": str(dr16q_path),
        "catwise_catalog": str(args.catwise_catalog),
        "exclude_mask_fits": None if args.exclude_mask_fits is None else str(args.exclude_mask_fits),
        "depth_map_fits": str(args.depth_map_fits),
        "depth_map_name": str(args.depth_map_name),
        "depth_map_transform": str(args.depth_map_transform),
        "nside": int(args.nside),
        "b_cut_deg": float(args.b_cut),
        "w1cov_min": float(args.w1cov_min),
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "feature_set": str(args.feature_set),
        "truth_n_used": int(len(y)),
        "truth_positive_rate": float(np.mean(y)),
        "features": feature_names,
        "logreg": {"intercept": intercept, "coef": [float(x) for x in coef]},
        "cv": {
            "scheme": "GroupKFold on ecliptic longitude wedges (15 deg bins)",
            "n_splits": 5,
            "roc_auc": auc,
            "brier": brier,
        },
        "alpha_edge_reference": {
            "cut": float(args.alpha_edge_cut),
            "delta": float(args.alpha_edge_delta),
            "alpha_edge": alpha_edge_ref,
            "source": "Estimated from CatWISE W1 histogram in the seen footprint (global slope).",
        },
        "maps": {
            "m50_map_fits": "m50_map_nside64.fits",
            "delta_m_map_fits": "delta_m_map_nside64.fits",
            "p_edge_map_fits": f"p_edge_map_w1{str(cut_ref).replace('.', 'p')}.fits",
            "p_edge_w1_ref": cut_ref,
        },
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    hp.write_map(str(outdir / "m50_map_nside64.fits"), m50_map, overwrite=True, dtype=float)
    hp.write_map(str(outdir / "delta_m_map_nside64.fits"), dm_map, overwrite=True, dtype=float)
    hp.write_map(
        str(outdir / f"p_edge_map_w1{str(cut_ref).replace('.', 'p')}.fits"),
        p_edge_map,
        overwrite=True,
        dtype=float,
    )

    # Small training table snapshot for debugging (not huge; keep just a sample)
    rng = np.random.default_rng(int(args.seed))
    samp_n = min(20000, len(y))
    samp_idx = rng.choice(len(y), size=samp_n, replace=False)
    snap = {
        "truth_sample_n": int(samp_n),
        "columns": ["w1", "depth", "abs_elat_deg", "sin_elon", "cos_elon", "y", "p_cv", "sep_arcsec"],
        "rows": [
            [
                float(truth_w1[i]),
                float(depth_val[i]),
                float(abs_elat[i]),
                float(sin_elon[i]),
                float(cos_elon[i]),
                int(y[i]),
                float(p_cv[i]),
                float(match.sep_arcsec[i]),
            ]
            for i in samp_idx
        ],
    }
    (outdir / "training_snapshot.json").write_text(json.dumps(snap, indent=2))

    # -------------------- Optional plots --------------------
    if args.make_plots:
        import matplotlib.pyplot as plt

        # Calibration curve (binned)
        nb = 20
        bins = np.linspace(0.0, 1.0, nb + 1)
        which = np.digitize(p_cv, bins) - 1
        p_mean = np.full(nb, np.nan)
        y_mean = np.full(nb, np.nan)
        for bidx in range(nb):
            m = which == bidx
            if np.any(m):
                p_mean[bidx] = float(np.mean(p_cv[m]))
                y_mean[bidx] = float(np.mean(y[m]))

        plt.figure(figsize=(5.0, 4.0))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        plt.plot(p_mean, y_mean, "o-", label="DR16Q holdout")
        plt.xlabel("Predicted P(in CatWISE)")
        plt.ylabel("Empirical fraction")
        plt.title(f"Calibration (AUC={auc:.3f}, Brier={brier:.3f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "calibration_curve.png", dpi=200)
        plt.close()

        # Mollweide map of δm (Galactic)
        plt.figure(figsize=(10.0, 4.0))
        hp.mollview(dm_map, title="External completeness δm map (SDSS DR16Q)", unit="mag", fig=plt.gcf().number)
        plt.savefig(outdir / "delta_m_mollweide.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10.0, 4.0))
        hp.mollview(
            p_edge_map,
            title=f"External completeness P(in CatWISE) at W1={cut_ref:.2f}",
            unit="prob",
            min=0.0,
            max=1.0,
            fig=plt.gcf().number,
        )
        plt.savefig(outdir / "p_edge_mollweide.png", dpi=200)
        plt.close()

    # stdout summary
    print(json.dumps({**meta["cv"], "truth_n_used": int(len(y)), "truth_positive_rate": float(np.mean(y))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
