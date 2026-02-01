#!/usr/bin/env python3
"""
Vector convergence test: Poisson-GLM systematics cleaning + held-out residual dipole direction.

Goal (Gemini-style):
  Fit counts with survey/systematics templates (coverage + dust + scan strategy),
  subtract those contributions, and measure the *residual* dipole direction.
  If the residual direction moves toward the SN anisotropy axis and is stable
  under reasonable choices, that would support a "shared signal" hypothesis.

This script implements a conservative, testable version:
  - Build HEALPix pixel count maps from the CatWISE/Secrest catalog
  - Fit a Poisson GLM (log link) for counts using systematics templates
    (and optionally a dipole term) on training pixels
  - On held-out pixels, compute fractional residuals:
        y = N/mu - 1
    and fit a dipole to y with WLS weights ~mu (approx var(y)=1/mu)
  - Repeat over K folds and report stability + axis angles vs:
      * Secrest raw dipole axis
      * SN horizon-anisotropy axis (from scan_summary.json)

Notes:
    - Coverage/depth is approximated using the catalog's per-source `w1cov` column
    aggregated to pixels. This is instrument-derived, but sampled at source positions;
    it is not a perfect independent exposure map. The CV step helps guard overfit.
  - A true state-of-the-art version would ingest an independent WISE/unWISE depth map
    (Nexp) in HEALPix and use it as an offset; this plumbing supports swapping that in.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def unitvec_from_lb(l_deg: float, b_deg: float) -> np.ndarray:
    return lb_to_unitvec(np.array([l_deg], dtype=float), np.array([b_deg], dtype=float))[0]


def vec_to_lb(vec: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size != 3:
        raise ValueError("expected 3-vector")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return float("nan")
    dot = abs(float(np.dot(a, b)) / (da * db))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(math.acos(dot)))


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x[valid])) if np.any(valid) else 0.0
    s = float(np.std(x[valid])) if np.any(valid) else 1.0
    if s == 0.0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


def poisson_glm_irls(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray | None = None,
    max_iter: int = 100,
    *,
    max_eta: float = 20.0,
    damping: float = 0.5,
) -> np.ndarray:
    """
    Fit Poisson GLM with log link.
      mu = exp(offset + X beta)
    Returns beta.

    Implementation note:
      A hand-rolled IRLS can be numerically fragile in this application (high dynamic range in
      offsets, strong template correlations). We therefore use a small, stable L-BFGS fit of the
      Poisson negative log-likelihood. This keeps dependencies light (SciPy is already required by
      the project) while behaving predictably across template/offset variants.
    """

    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    off = np.zeros(n, dtype=float) if offset is None else np.asarray(offset, dtype=float)

    # Initialize at a reasonable scale (intercept-only approximation).
    mu0 = np.clip(float(np.mean(y)), 1.0, np.inf)
    x0 = np.zeros(p, dtype=float)
    x0[0] = math.log(mu0) - float(np.mean(off))  # assumes X[:,0] is intercept

    max_eta_f = float(max_eta)

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = np.asarray(beta, dtype=float)
        eta = off + X @ beta
        # Prevent overflow; clipping is acceptable here since eta excursions are unphysical for counts/pixels.
        eta = np.clip(eta, -max_eta_f, max_eta_f)
        mu = np.exp(eta)
        # NLL up to additive constant: sum(mu - y*eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        return nll, np.asarray(grad, dtype=float)

    def f(beta: np.ndarray) -> float:
        return fun_and_grad(beta)[0]

    def g(beta: np.ndarray) -> np.ndarray:
        return fun_and_grad(beta)[1]

    res = minimize(
        f,
        x0,
        jac=g,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-10},
    )
    beta_hat = np.asarray(res.x, dtype=float)
    if damping != 1.0:
        # Keep the signature compatible; damping is unused in L-BFGS mode but we accept it.
        pass
    return beta_hat


def fit_residual_dipole_fractional(pix_unit: np.ndarray, y_frac: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Fit y_frac = a + b·n with WLS weights w, return dipole vector b (fractional units).
    """

    y_frac = np.asarray(y_frac, dtype=float)
    w = np.asarray(w, dtype=float)
    X = np.column_stack([np.ones_like(y_frac), pix_unit[:, 0], pix_unit[:, 1], pix_unit[:, 2]])
    sw = np.sqrt(np.clip(w, 0.0, np.inf))
    Xw = X * sw[:, None]
    yw = y_frac * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    b = np.asarray(beta[1:4], dtype=float)
    return b


def load_sn_axis(sn_scan_json: str) -> Tuple[float, float]:
    d = json.load(open(sn_scan_json, "r"))
    best = d.get("best_axis") or {}
    return float(best["axis_l_deg"]), float(best["axis_b_deg"])


@dataclass
class FoldResult:
    fold: int
    n_train_pix: int
    n_test_pix: int
    b_vec: np.ndarray


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--nvss-crossmatch", default=None, help="Optional: NVSS crossmatch FITS to remove matches.")
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
        help="Optional: Secrest-style exclude regions (FITS table with ra/dec/radius/use).",
    )

    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-max", type=float, default=16.4)
    ap.add_argument("--nside", type=int, default=64)

    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--template-set", choices=["basic", "ecliptic_harmonics", "poly", "full"], default="ecliptic_harmonics")
    ap.add_argument(
        "--glm-include-dipole",
        action="store_true",
        help="Include a dipole term (n_x,n_y,n_z) in the GLM fit in addition to templates.",
    )
    ap.add_argument(
        "--unwise-tiles-fits",
        default="data/external/unwise/tiles.fits",
        help="unWISE tiles table (ra/dec/coadd_id) used to map sky positions to coadd IDs.",
    )
    ap.add_argument(
        "--nexp-tile-stats-json",
        default=None,
        help=(
            "Optional: JSON mapping {coadd_id: nexp_stat} where nexp_stat is e.g. median W1 exposures "
            "from unWISE -w1-n-m.fits.gz for the chosen release. When provided, uses log(Nexp) as a Poisson-GLM offset "
            "and drops the w1cov template by default (unless --keep-w1cov-template is set)."
        ),
    )
    ap.add_argument(
        "--keep-w1cov-template",
        action="store_true",
        help="If set, keep the catalog-derived w1cov template even when --nexp-tile-stats-json is provided.",
    )
    ap.add_argument("--sn-scan-json", default="outputs/horizon_anisotropy_fullscan_null100_dipoleT_field_axispar_nside4_surveyz_20260131_225012UTC/scan_summary.json")
    ap.add_argument("--secrest-json", default="Q_D_RES/secrest_reproduction_dipole.json")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/quasar_vector_convergence_glmcv_{args.template_set}_nside{args.nside}_w1{args.w1_max}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Heavy deps late.
    import healpy as hp
    from astropy.table import Table

    tab = Table.read(args.catalog, memmap=True)
    # Mandatory columns.
    source_id = np.asarray(tab["source_id"]).astype(str)
    l = np.asarray(tab["l"], dtype=float)
    b = np.asarray(tab["b"], dtype=float)
    w1 = np.asarray(tab["w1"], dtype=float)
    w1cov = np.asarray(tab["w1cov"], dtype=float)
    ebv = np.asarray(tab["ebv"], dtype=float)
    elon = np.asarray(tab["elon"], dtype=float)
    elat = np.asarray(tab["elat"], dtype=float)
    Tb = np.asarray(tab["Tb"], dtype=float)
    alpha = np.asarray(tab["alpha"], dtype=float)
    logS = np.asarray(tab["logS"], dtype=float)

    base = (
        np.isfinite(l)
        & np.isfinite(b)
        & np.isfinite(w1)
        & np.isfinite(w1cov)
        & np.isfinite(ebv)
        & np.isfinite(elon)
        & np.isfinite(elat)
        & (np.abs(b) > float(args.b_cut))
        & (w1cov >= float(args.w1cov_min))
    )

    # Optional NVSS removal.
    if args.nvss_crossmatch:
        nv = Table.read(args.nvss_crossmatch, memmap=True)
        nv_ids = np.asarray(nv["source_id"]).astype(str)
        base &= ~np.isin(source_id, nv_ids)

    # Apply faint cut.
    base &= (w1 <= float(args.w1_max))

    l = l[base]
    b = b[base]
    w1cov = w1cov[base]
    ebv = ebv[base]
    elon = elon[base]
    elat = elat[base]
    Tb = Tb[base]
    alpha = alpha[base]
    logS = logS[base]

    # Build pixelization.
    nside = int(args.nside)
    theta = np.deg2rad(90.0 - b)
    phi = np.deg2rad(l % 360.0)
    pix = hp.ang2pix(nside, theta, phi, nest=True)
    npix = hp.nside2npix(nside)

    # Apply Secrest exclude regions (pixel-level) if provided.
    exclude_pix = None
    if args.exclude_mask_fits:
        ex_path = Path(args.exclude_mask_fits)
        if ex_path.exists():
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            ex = Table.read(ex_path, memmap=True)
            use = np.asarray(ex["use"], dtype=bool) if "use" in ex.colnames else np.ones(len(ex), dtype=bool)
            ex = ex[use]
            if len(ex) > 0:
                # Convert disc centers to Galactic to match our l,b pixelization.
                g = SkyCoord(ra=np.asarray(ex["ra"], dtype=float) * u.deg, dec=np.asarray(ex["dec"], dtype=float) * u.deg, frame="icrs").galactic
                l0 = g.l.deg
                b0 = g.b.deg
                rdeg = np.asarray(ex["radius"], dtype=float)
                exclude = set()
                for ll, bb, rr in zip(l0, b0, rdeg, strict=True):
                    vec = hp.ang2vec(np.deg2rad(90.0 - bb), np.deg2rad(ll))
                    disc = hp.query_disc(nside, vec, np.deg2rad(rr), nest=True)
                    exclude.update(int(x) for x in disc)
                exclude_pix = np.fromiter(sorted(exclude), dtype=np.int64)
                keep = ~np.isin(pix, exclude_pix)
                pix = pix[keep]
                l = l[keep]
                b = b[keep]
                w1cov = w1cov[keep]
                ebv = ebv[keep]
                elon = elon[keep]
                elat = elat[keep]
                Tb = Tb[keep]
                alpha = alpha[keep]
                logS = logS[keep]

    # Pixel unit vectors.
    pix_unit = np.zeros((npix, 3), dtype=float)
    for p in range(npix):
        th, ph = hp.pix2ang(nside, p, nest=True)
        pix_unit[p] = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))

    # Counts per pixel.
    Np = np.bincount(pix, minlength=npix).astype(float)
    valid = Np > 0

    # Optional: independent unWISE depth-of-coverage (Nexp) used as GLM offset.
    offset = None
    if args.nexp_tile_stats_json:
        tile_stats_path = Path(args.nexp_tile_stats_json)
        if not tile_stats_path.exists():
            raise FileNotFoundError(f"--nexp-tile-stats-json not found: {tile_stats_path}")
        tile_stats = json.load(open(tile_stats_path, "r"))
        # Map each HEALPix pixel to the nearest unWISE tile center (in Galactic coordinates).
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.table import Table as ATable
        from scipy.spatial import cKDTree

        tiles = ATable.read(args.unwise_tiles_fits, memmap=True)
        coadd_id = np.asarray(tiles["coadd_id"]).astype(str)
        ra = np.asarray(tiles["ra"], dtype=float)
        dec = np.asarray(tiles["dec"], dtype=float)
        g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
        lg = np.deg2rad(g.l.deg % 360.0)
        bg = np.deg2rad(g.b.deg)
        cosb = np.cos(bg)
        tile_vec = np.column_stack([cosb * np.cos(lg), cosb * np.sin(lg), np.sin(bg)])
        tree = cKDTree(tile_vec)

        _, nn_idx = tree.query(pix_unit, k=1)
        pix_coadd = coadd_id[nn_idx]

        nexp = np.full(npix, np.nan, dtype=float)
        v_idx = np.flatnonzero(valid)
        for p in v_idx:
            nexp[p] = float(tile_stats.get(str(pix_coadd[p]), float("nan")))

        ok = valid & np.isfinite(nexp) & (nexp > 0.0)
        # If tile stats are missing for any valid pixels, drop them.
        valid = ok
        offset = np.zeros(npix, dtype=float)
        offset[valid] = np.log(nexp[valid])

    # Pixel means for templates. Use circular-safe ecliptic longitude moments.
    elon_rad = np.deg2rad(elon % 360.0)
    sin_elon = np.sin(elon_rad)
    cos_elon = np.cos(elon_rad)
    sin_2elon = np.sin(2.0 * elon_rad)
    cos_2elon = np.cos(2.0 * elon_rad)

    def pix_mean(val: np.ndarray) -> np.ndarray:
        s = np.bincount(pix, weights=val, minlength=npix)
        m = np.zeros(npix, dtype=float)
        m[valid] = s[valid] / np.clip(Np[valid], 1.0, np.inf)
        return m

    t_ebv = pix_mean(ebv)
    t_w1cov = pix_mean(w1cov)
    t_abs_elat = pix_mean(np.abs(elat))
    t_elat = pix_mean(elat)
    t_sin_elon = pix_mean(sin_elon)
    t_cos_elon = pix_mean(cos_elon)
    t_sin_2elon = pix_mean(sin_2elon)
    t_cos_2elon = pix_mean(cos_2elon)
    t_Tb = pix_mean(Tb)
    t_alpha = pix_mean(alpha)
    t_logS = pix_mean(logS)

    # Assemble template matrix.
    tpl_cols: List[np.ndarray] = []
    tpl_names: List[str] = []

    tpl_names += ["ebv", "abs_elat"]
    tpl_cols += [t_ebv, t_abs_elat]

    include_w1cov_tpl = True
    if args.nexp_tile_stats_json and not args.keep_w1cov_template:
        include_w1cov_tpl = False
    if include_w1cov_tpl:
        tpl_names += ["w1cov"]
        tpl_cols += [t_w1cov]

    if args.template_set in ("ecliptic_harmonics", "poly", "full"):
        tpl_names += ["sin_elon", "cos_elon", "sin_2elon", "cos_2elon", "elat"]
        tpl_cols += [t_sin_elon, t_cos_elon, t_sin_2elon, t_cos_2elon, t_elat]

    if args.template_set in ("poly", "full"):
        tpl_names += ["ebv2"]
        tpl_cols += [t_ebv * t_ebv]
        if include_w1cov_tpl:
            tpl_names += ["w1cov2", "log_w1cov"]
            tpl_cols += [t_w1cov * t_w1cov, np.log(np.clip(t_w1cov, 1.0, np.inf))]

    if args.template_set == "full":
        tpl_names += ["Tb", "alpha", "logS"]
        tpl_cols += [t_Tb, t_alpha, t_logS]

    Z = np.column_stack([zscore(c, valid) for c in tpl_cols])

    # K-fold split across valid pixels.
    rng = np.random.default_rng(int(args.seed))
    valid_idx = np.flatnonzero(valid)
    rng.shuffle(valid_idx)
    k = int(args.kfold)
    folds = np.array_split(valid_idx, k)

    # Load reference axes.
    sn_l, sn_b = load_sn_axis(args.sn_scan_json)
    sn_axis = unitvec_from_lb(sn_l, sn_b)
    sec = json.load(open(args.secrest_json, "r"))
    se_l, se_b = float(sec["dipole"]["l_deg"]), float(sec["dipole"]["b_deg"])
    se_axis = unitvec_from_lb(se_l, se_b)

    fold_results: List[Dict] = []
    for i, test_idx in enumerate(folds):
        test_mask = np.zeros(npix, dtype=bool)
        test_mask[test_idx] = True
        train_mask = valid & ~test_mask
        test_mask = valid & test_mask

        y_train = Np[train_mask]
        if args.glm_include_dipole:
            X_train = np.column_stack(
                [
                    np.ones_like(y_train),
                    pix_unit[train_mask, 0],
                    pix_unit[train_mask, 1],
                    pix_unit[train_mask, 2],
                    Z[train_mask],
                ]
            )
        else:
            X_train = np.column_stack([np.ones_like(y_train), Z[train_mask]])

        off_train = None if offset is None else offset[train_mask]
        beta = poisson_glm_irls(X_train, y_train, offset=off_train, max_iter=80, max_eta=20.0, damping=0.5)

        # Predict mu for test pixels.
        if args.glm_include_dipole:
            X_test = np.column_stack(
                [
                    np.ones(int(test_mask.sum())),
                    pix_unit[test_mask, 0],
                    pix_unit[test_mask, 1],
                    pix_unit[test_mask, 2],
                    Z[test_mask],
                ]
            )
        else:
            X_test = np.column_stack([np.ones(int(test_mask.sum())), Z[test_mask]])
        eta_test = X_test @ beta
        if offset is not None:
            eta_test = offset[test_mask] + eta_test
        eta_test = np.clip(eta_test, -20.0, 20.0)
        mu_test = np.exp(eta_test)
        mu_test = np.clip(mu_test, 1e-8, 1e15)
        y_test = Np[test_mask]

        # Fractional residual field and its dipole on test set.
        frac = y_test / mu_test - 1.0
        w = mu_test  # approx var(frac)=1/mu
        b_vec = fit_residual_dipole_fractional(pix_unit[test_mask], frac, w)

        # Also report the GLM dipole coefficient direction (if enabled).
        glm_b_vec = None
        glm_b_l = None
        glm_b_b = None
        glm_b_amp = None
        glm_ang_sn = None
        glm_ang_sec = None
        if args.glm_include_dipole:
            glm_b = np.asarray(beta[1:4], dtype=float)
            glm_b_vec = [float(x) for x in glm_b]
            glm_b_amp = float(np.linalg.norm(glm_b))
            glm_b_l, glm_b_b = vec_to_lb(glm_b)
            glm_ang_sn = axis_angle_deg(glm_b, sn_axis)
            glm_ang_sec = axis_angle_deg(glm_b, se_axis)

        fold_results.append(
            {
                "fold": int(i),
                "n_train_pix": int(train_mask.sum()),
                "n_test_pix": int(test_mask.sum()),
                "b_vec": [float(x) for x in b_vec],
                "b_amp": float(np.linalg.norm(b_vec)),
                "b_l_deg": vec_to_lb(b_vec)[0],
                "b_b_deg": vec_to_lb(b_vec)[1],
                "axis_angle_to_sn_deg": axis_angle_deg(b_vec, sn_axis),
                "axis_angle_to_secrest_deg": axis_angle_deg(b_vec, se_axis),
                "glm_dipole_vec": glm_b_vec,
                "glm_dipole_amp": glm_b_amp,
                "glm_dipole_l_deg": glm_b_l,
                "glm_dipole_b_deg": glm_b_b,
                "glm_axis_angle_to_sn_deg": glm_ang_sn,
                "glm_axis_angle_to_secrest_deg": glm_ang_sec,
            }
        )

    # Aggregate direction by vector-averaging b_vec.
    b_mean = np.mean([np.array(fr["b_vec"], dtype=float) for fr in fold_results], axis=0)

    glm_mean = None
    if args.glm_include_dipole:
        glm_vecs = [np.array(fr["glm_dipole_vec"], dtype=float) for fr in fold_results if fr["glm_dipole_vec"] is not None]
        if glm_vecs:
            glm_mean = np.mean(glm_vecs, axis=0)

    out = {
        "inputs": {
            "catalog": args.catalog,
            "nvss_crossmatch": args.nvss_crossmatch,
            "exclude_mask_fits": args.exclude_mask_fits,
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_max": float(args.w1_max),
            "nside": int(args.nside),
            "kfold": int(args.kfold),
            "seed": int(args.seed),
            "template_set": args.template_set,
            "glm_include_dipole": bool(args.glm_include_dipole),
            "unwise_tiles_fits": args.unwise_tiles_fits,
            "nexp_tile_stats_json": args.nexp_tile_stats_json,
            "keep_w1cov_template": bool(args.keep_w1cov_template),
            "sn_scan_json": args.sn_scan_json,
            "secrest_json": args.secrest_json,
        },
        "reference_axes": {
            "sn_best_axis": {"l_deg": sn_l, "b_deg": sn_b},
            "secrest_raw_axis": {"l_deg": se_l, "b_deg": se_b},
        },
        "templates": {"names": tpl_names},
        "folds": fold_results,
        "aggregate": {
            "b_vec_mean": [float(x) for x in b_mean],
            "b_amp_mean": float(np.linalg.norm(b_mean)),
            "b_l_deg_mean": vec_to_lb(b_mean)[0],
            "b_b_deg_mean": vec_to_lb(b_mean)[1],
            "axis_angle_to_sn_deg_mean": axis_angle_deg(b_mean, sn_axis),
            "axis_angle_to_secrest_deg_mean": axis_angle_deg(b_mean, se_axis),
        },
        "notes": (
            "This is a held-out residual dipole direction after fitting a Poisson GLM with systematics templates "
            "on training pixels. If the residual direction is stable across folds and shifts toward the SN axis, "
            "that supports a 'shared signal' hypothesis; if it wanders, the quasar dipole remains systematics-dominated."
        ),
    }

    if args.glm_include_dipole and glm_mean is not None:
        out["glm_aggregate"] = {
            "glm_dipole_vec_mean": [float(x) for x in glm_mean],
            "glm_dipole_amp_mean": float(np.linalg.norm(glm_mean)),
            "glm_dipole_l_deg_mean": vec_to_lb(glm_mean)[0],
            "glm_dipole_b_deg_mean": vec_to_lb(glm_mean)[1],
            "glm_axis_angle_to_sn_deg_mean": axis_angle_deg(glm_mean, sn_axis),
            "glm_axis_angle_to_secrest_deg_mean": axis_angle_deg(glm_mean, se_axis),
        }

    with open(outdir / "glm_cv_summary.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    if args.make_plots:
        import matplotlib.pyplot as plt

        # Scatter of fold axes on a simple lon-lat plane (not Mollweide).
        ls = [fr["b_l_deg"] for fr in fold_results]
        bs = [fr["b_b_deg"] for fr in fold_results]
        plt.figure(figsize=(6.5, 4.0), dpi=200)
        plt.scatter(ls, bs, s=40, label="fold residual dipole")
        plt.scatter([sn_l], [sn_b], marker="*", s=120, label="SN axis")
        plt.scatter([se_l], [se_b], marker="x", s=80, label="Secrest raw axis")
        plt.scatter([out["aggregate"]["b_l_deg_mean"]], [out["aggregate"]["b_b_deg_mean"]], marker="D", s=80, label="mean")
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        plt.xlabel("l [deg]")
        plt.ylabel("b [deg]")
        plt.title("Residual dipole directions (held-out) after GLM cleaning")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / "glm_cv_axes.png")
        plt.close()

        # Angles to SN per fold.
        ang_sn = [fr["axis_angle_to_sn_deg"] for fr in fold_results]
        plt.figure(figsize=(6.0, 3.6), dpi=200)
        plt.plot(range(len(ang_sn)), ang_sn, "o-", lw=2)
        plt.axhline(out["aggregate"]["axis_angle_to_sn_deg_mean"], color="k", ls="--", alpha=0.7, label="mean")
        plt.ylabel("axis angle to SN [deg]")
        plt.xlabel("fold")
        plt.title("Fold-to-fold axis angle to SN after cleaning")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / "glm_cv_angles_to_sn.png")
        plt.close()

        if args.glm_include_dipole and glm_mean is not None:
            # Fold directions of the fitted GLM dipole coefficient (train-only).
            gls = [fr["glm_dipole_l_deg"] for fr in fold_results if fr["glm_dipole_l_deg"] is not None]
            gbs = [fr["glm_dipole_b_deg"] for fr in fold_results if fr["glm_dipole_b_deg"] is not None]
            plt.figure(figsize=(6.5, 4.0), dpi=200)
            plt.scatter(gls, gbs, s=40, label="fold GLM dipole (train)")
            plt.scatter([sn_l], [sn_b], marker="*", s=120, label="SN axis")
            plt.scatter([se_l], [se_b], marker="x", s=80, label="Secrest raw axis")
            plt.scatter([out["glm_aggregate"]["glm_dipole_l_deg_mean"]], [out["glm_aggregate"]["glm_dipole_b_deg_mean"]], marker="D", s=80, label="GLM mean")
            plt.xlim(0, 360)
            plt.ylim(-90, 90)
            plt.xlabel("l [deg]")
            plt.ylabel("b [deg]")
            plt.title("GLM dipole coefficient directions (train) across folds")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(outdir / "glm_cv_glm_dipole_axes.png")
            plt.close()

    print(json.dumps(out.get("glm_aggregate") or out["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
