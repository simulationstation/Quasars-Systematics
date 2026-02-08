#!/usr/bin/env python3
"""Apply CatWISE-style robustness logic to NVSS radio dipole maps.

This is an in-repo reproducibility script focused on one locally available survey (NVSS).
It runs five of the six stress tests discussed in chat:
  1) Flux-cut stability + correlated-cut Monte Carlo (nested lower-flux cuts)
  2) Template sensitivity (declination, RA harmonics, quadrupole-like terms)
  3) Injection/recovery under omitted-vs-included nuisance templates
  4) Proxy hierarchical/common-dipole test across independent NVSS flux bands
     (true leave-one-survey-out requires additional survey catalogs)
  5) Component-merging sensitivity versus merge radius
  6) Geometry jackknife (RA sectors x Dec bins)

Outputs:
  - summary JSON
  - compact markdown report
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
from scipy.spatial import cKDTree


R_EQ_TO_GAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ],
    dtype=float,
)


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def vec_to_radec_deg(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return float("nan"), float("nan")
    u = v / n
    ra = float(np.degrees(np.arctan2(u[1], u[0])) % 360.0)
    dec = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
    return ra, dec


def radec_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    c = np.cos(dec)
    return np.column_stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)])


def axis_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=float).reshape(3)
    b = np.asarray(v2, dtype=float).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or (not np.isfinite(na)) or (not np.isfinite(nb)):
        return float("nan")
    c = abs(float(np.dot(a, b) / (na * nb)))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def ang_sep_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=float).reshape(3)
    b = np.asarray(v2, dtype=float).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or (not np.isfinite(na)) or (not np.isfinite(nb)):
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return np.zeros_like(x)
    m = float(np.mean(x[valid]))
    s = float(np.std(x[valid]))
    if s == 0.0 or (not np.isfinite(s)):
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


def residualize_against(base: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove the component of template t explained by columns of base."""
    b = np.asarray(base, dtype=float)
    y = np.asarray(t, dtype=float).reshape(-1)
    coef, *_ = np.linalg.lstsq(b, y, rcond=None)
    return y - b @ coef


def fit_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    offset: np.ndarray | None = None,
    max_iter: int = 300,
    beta_init: np.ndarray | None = None,
    prior_prec_diag: np.ndarray | None = None,
    compute_cov: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    off = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)

    if beta_init is None:
        mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
        b0 = np.zeros(X.shape[1], dtype=float)
        b0[0] = math.log(mu0)
    else:
        b0 = np.asarray(beta_init, dtype=float).reshape(X.shape[1])

    if prior_prec_diag is None:
        prior = np.zeros(X.shape[1], dtype=float)
    else:
        prior = np.asarray(prior_prec_diag, dtype=float).reshape(X.shape[1])

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(off + X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta) + 0.5 * np.sum(prior * (beta * beta)))
        grad = X.T @ (mu - y) + prior * beta
        return nll, np.asarray(grad, dtype=float)

    res = minimize(
        lambda b: fun_and_grad(b)[0],
        b0,
        jac=lambda b: fun_and_grad(b)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    beta = np.asarray(res.x, dtype=float)

    if not bool(compute_cov):
        return beta, None
    try:
        eta = np.clip(off + X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        fisher = X.T @ (mu[:, None] * X)
        fisher += np.diag(prior)
        cov = np.linalg.inv(fisher)
    except Exception:
        cov = None
    return beta, cov


@dataclass(frozen=True)
class PixGeom:
    vec: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    glat_deg: np.ndarray


def build_pix_geom(nside: int) -> PixGeom:
    npix = hp.nside2npix(int(nside))
    vx, vy, vz = hp.pix2vec(int(nside), np.arange(npix), nest=False)
    vec = np.column_stack([vx, vy, vz]).astype(float)
    ra = (np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(np.clip(vec[:, 2], -1.0, 1.0)))
    gvec = vec @ R_EQ_TO_GAL.T
    glat = np.degrees(np.arcsin(np.clip(gvec[:, 2], -1.0, 1.0)))
    return PixGeom(vec=vec, ra_deg=ra, dec_deg=dec, glat_deg=glat)


def source_gal_lat_deg(src_vec: np.ndarray) -> np.ndarray:
    gvec = src_vec @ R_EQ_TO_GAL.T
    return np.degrees(np.arcsin(np.clip(gvec[:, 2], -1.0, 1.0)))


def build_design(
    vec_seen: np.ndarray,
    ra_seen_deg: np.ndarray,
    dec_seen_deg: np.ndarray,
    *,
    model: str,
    prior_strength_l2: float,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    valid = np.ones(vec_seen.shape[0], dtype=bool)

    base_cols: list[np.ndarray] = [
        np.ones(vec_seen.shape[0], dtype=float),
        vec_seen[:, 0],
        vec_seen[:, 1],
        vec_seen[:, 2],
    ]
    cols: list[np.ndarray] = list(base_cols)
    names = ["intercept", "dip_x", "dip_y", "dip_z"]
    base = np.column_stack(base_cols)

    def add_template(raw: np.ndarray, name: str) -> None:
        t = zscore(raw, valid)
        t = residualize_against(base, t)
        t = zscore(t, valid)
        cols.append(t)
        names.append(name)

    if model in {"dipole_dec", "dipole_dec_ra", "dipole_dec_ra_l2"}:
        add_template(dec_seen_deg, "dec_z")
        add_template(dec_seen_deg**2, "dec2_z")

    if model in {"dipole_ra", "dipole_dec_ra", "dipole_dec_ra_l2"}:
        rar = np.deg2rad(ra_seen_deg)
        add_template(np.sin(rar), "sin_ra_z")
        add_template(np.cos(rar), "cos_ra_z")

    prior = None
    if model == "dipole_dec_ra_l2":
        nx, ny, nz = vec_seen[:, 0], vec_seen[:, 1], vec_seen[:, 2]
        l2_terms = [
            nx * ny,
            nx * nz,
            ny * nz,
            nx * nx - ny * ny,
            3.0 * nz * nz - 1.0,
        ]
        qnames = ["q_xy", "q_xz", "q_yz", "q_x2my2", "q_3z2m1"]
        for t, n in zip(l2_terms, qnames, strict=True):
            add_template(t, n)
        prior = np.zeros(4 + 2 + 2 + 5, dtype=float)
        # Only regularize quadrupole terms.
        prior[-5:] = float(prior_strength_l2)

    X = np.column_stack(cols)
    return X, names, prior


def fit_model(
    y_map: np.ndarray,
    seen: np.ndarray,
    geom: PixGeom,
    *,
    model: str,
    max_iter: int,
    prior_strength_l2: float,
) -> dict[str, Any]:
    y = np.asarray(y_map[seen], dtype=float)
    vec_seen = geom.vec[seen]
    ra_seen = geom.ra_deg[seen]
    dec_seen = geom.dec_deg[seen]

    X, names, prior = build_design(
        vec_seen,
        ra_seen,
        dec_seen,
        model=model,
        prior_strength_l2=float(prior_strength_l2),
    )
    beta, cov = fit_poisson_glm(X, y, max_iter=int(max_iter), prior_prec_diag=prior)
    bvec = np.asarray(beta[1:4], dtype=float)
    D = float(np.linalg.norm(bvec))
    ra, dec = vec_to_radec_deg(bvec)
    out: dict[str, Any] = {
        "model": model,
        "n_pix": int(np.sum(seen)),
        "n_src": int(np.sum(y)),
        "D": D,
        "ra_deg": float(ra),
        "dec_deg": float(dec),
        "b_vec": [float(x) for x in bvec],
        "coef_names": names,
        "coef": [float(x) for x in beta],
    }
    if cov is not None:
        cov_b = np.asarray(cov[1:4, 1:4], dtype=float)
        out["cov_b"] = [[float(x) for x in row] for row in cov_b.tolist()]
    else:
        out["cov_b"] = None
    return out


def build_counts_by_cut(
    ipix: np.ndarray,
    flux: np.ndarray,
    cuts: list[float],
    npix: int,
) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    for c in cuts:
        m = flux >= float(c)
        out[float(c)] = np.bincount(ipix[m], minlength=npix).astype(np.int64)
    return out


def drift_metrics_from_vectors(vs: np.ndarray) -> dict[str, float]:
    n = vs.shape[0]
    path = 0.0
    for i in range(n - 1):
        path += axis_angle_deg(vs[i], vs[i + 1])
    end_to_end = axis_angle_deg(vs[0], vs[-1])
    max_pair = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            max_pair = max(max_pair, axis_angle_deg(vs[i], vs[j]))
    return {
        "path_deg": float(path),
        "end_to_end_deg": float(end_to_end),
        "max_pair_deg": float(max_pair),
    }


def run_correlated_cut_mc(
    *,
    counts_by_cut: dict[float, np.ndarray],
    cuts: list[float],
    seen: np.ndarray,
    geom: PixGeom,
    nsim: int,
    seed: int,
    model: str,
    max_iter: int,
    prior_strength_l2: float,
) -> dict[str, Any]:
    cuts_sorted = sorted(float(c) for c in cuts)

    # Differential bins for nested lower-cut maps: [c0,c1), [c1,c2), ..., [c_last, inf)
    y_diffs: list[np.ndarray] = []
    for i, c in enumerate(cuts_sorted):
        y_hi = counts_by_cut[c]
        if i + 1 < len(cuts_sorted):
            y_next = counts_by_cut[cuts_sorted[i + 1]]
            y_diff = y_hi - y_next
        else:
            y_diff = y_hi.copy()
        y_diff = np.clip(y_diff, 0, None)
        y_diffs.append(y_diff.astype(np.int64, copy=False))

    # Observed drift.
    b_obs = []
    for c in cuts_sorted:
        fit = fit_model(
            counts_by_cut[c],
            seen,
            geom,
            model=model,
            max_iter=max_iter,
            prior_strength_l2=prior_strength_l2,
        )
        b_obs.append(np.asarray(fit["b_vec"], dtype=float))
    b_obs_arr = np.vstack(b_obs)
    obs = drift_metrics_from_vectors(b_obs_arr)

    rng = np.random.default_rng(int(seed))
    ge_path = 0
    ge_end = 0
    ge_maxp = 0

    for _ in range(int(nsim)):
        sim_diffs = [rng.poisson(lam=yd).astype(np.int64) for yd in y_diffs]
        # cumulative lower-cut maps in ascending cuts.
        y_cums: list[np.ndarray] = []
        acc = np.zeros_like(sim_diffs[0])
        for j in range(len(sim_diffs) - 1, -1, -1):
            acc = acc + sim_diffs[j]
            y_cums.append(acc.copy())
        y_cums = list(reversed(y_cums))

        b_sim = []
        for y in y_cums:
            fit = fit_model(
                y,
                seen,
                geom,
                model=model,
                max_iter=max_iter,
                prior_strength_l2=prior_strength_l2,
            )
            b_sim.append(np.asarray(fit["b_vec"], dtype=float))
        met = drift_metrics_from_vectors(np.vstack(b_sim))
        ge_path += int(met["path_deg"] >= obs["path_deg"])
        ge_end += int(met["end_to_end_deg"] >= obs["end_to_end_deg"])
        ge_maxp += int(met["max_pair_deg"] >= obs["max_pair_deg"])

    return {
        "cuts_mJy": cuts_sorted,
        "observed": obs,
        "nsim": int(nsim),
        "p_path_ge_obs": float(ge_path / max(1, nsim)),
        "p_end_to_end_ge_obs": float(ge_end / max(1, nsim)),
        "p_max_pair_ge_obs": float(ge_maxp / max(1, nsim)),
    }


def run_template_sensitivity(
    *,
    y_map: np.ndarray,
    seen: np.ndarray,
    geom: PixGeom,
    model_list: list[str],
    max_iter: int,
    prior_strength_l2: float,
    cmb_vec: np.ndarray,
) -> dict[str, Any]:
    rows = []
    baseline_b = None
    for m in model_list:
        fit = fit_model(
            y_map,
            seen,
            geom,
            model=m,
            max_iter=max_iter,
            prior_strength_l2=prior_strength_l2,
        )
        b = np.asarray(fit["b_vec"], dtype=float)
        if baseline_b is None:
            baseline_b = b
        fit["angle_to_cmb_deg"] = float(ang_sep_deg(b, cmb_vec))
        fit["axis_angle_to_baseline_deg"] = float(axis_angle_deg(b, baseline_b))
        rows.append(fit)
    return {"models": rows}


def run_injection_recovery(
    *,
    y_data: np.ndarray,
    seen: np.ndarray,
    geom: PixGeom,
    nsim: int,
    seed: int,
    max_iter: int,
    prior_strength_l2: float,
    d_inj: float,
    cmb_vec: np.ndarray,
) -> dict[str, Any]:
    # Build generation model from no-dipole + fitted RA systematic amplitude.
    y = np.asarray(y_data[seen], dtype=float)
    decz = zscore(geom.dec_deg, seen)[seen]
    dec2z = zscore(geom.dec_deg**2, seen)[seen]
    sinraz = zscore(np.sin(np.deg2rad(geom.ra_deg)), seen)[seen]
    cosraz = zscore(np.cos(np.deg2rad(geom.ra_deg)), seen)[seen]
    nx, ny, nz = geom.vec[seen].T
    base = np.column_stack([np.ones_like(y), nx, ny, nz])

    decz = zscore(residualize_against(base, decz), np.ones_like(decz, dtype=bool))
    dec2z = zscore(residualize_against(base, dec2z), np.ones_like(dec2z, dtype=bool))
    sinraz = zscore(residualize_against(base, sinraz), np.ones_like(sinraz, dtype=bool))
    cosraz = zscore(residualize_against(base, cosraz), np.ones_like(cosraz, dtype=bool))

    X_nodip = np.column_stack([np.ones_like(y), decz, dec2z])
    b_nodip, _ = fit_poisson_glm(X_nodip, y, max_iter=max_iter)

    X_full = np.column_stack([np.ones_like(y), nx, ny, nz, decz, dec2z, sinraz, cosraz])
    b_full, _ = fit_poisson_glm(X_full, y, max_iter=max_iter)
    beta_sin = float(b_full[6])
    beta_cos = float(b_full[7])

    eta_base = np.clip(X_nodip @ b_nodip, -25.0, 25.0)
    b_inj = float(d_inj) * (cmb_vec / np.linalg.norm(cmb_vec))
    eta_inj = eta_base + (nx * b_inj[0] + ny * b_inj[1] + nz * b_inj[2]) + beta_sin * sinraz + beta_cos * cosraz
    mu = np.exp(np.clip(eta_inj, -25.0, 25.0))

    rng = np.random.default_rng(int(seed))
    D_dip: list[float] = []
    D_omit: list[float] = []
    D_with: list[float] = []
    ang_dip: list[float] = []
    ang_omit: list[float] = []
    ang_with: list[float] = []

    for _ in range(int(nsim)):
        ys = rng.poisson(mu)

        # Dipole only
        X_dip = np.column_stack([np.ones_like(ys), nx, ny, nz])
        bd, _ = fit_poisson_glm(X_dip, ys, max_iter=max_iter)
        bvec_d = np.asarray(bd[1:4], dtype=float)

        # Omit RA templates
        X_omit = np.column_stack([np.ones_like(ys), nx, ny, nz, decz, dec2z])
        bo, _ = fit_poisson_glm(X_omit, ys, max_iter=max_iter)
        bvec_o = np.asarray(bo[1:4], dtype=float)

        # Include RA templates
        X_with = np.column_stack([np.ones_like(ys), nx, ny, nz, decz, dec2z, sinraz, cosraz])
        bw, _ = fit_poisson_glm(X_with, ys, max_iter=max_iter)
        bvec_w = np.asarray(bw[1:4], dtype=float)

        D_dip.append(float(np.linalg.norm(bvec_d)))
        D_omit.append(float(np.linalg.norm(bvec_o)))
        D_with.append(float(np.linalg.norm(bvec_w)))
        ang_dip.append(float(ang_sep_deg(bvec_d, b_inj)))
        ang_omit.append(float(ang_sep_deg(bvec_o, b_inj)))
        ang_with.append(float(ang_sep_deg(bvec_w, b_inj)))

    def summarize(x: list[float]) -> dict[str, float]:
        a = np.asarray(x, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "p16": float(np.percentile(a, 16)),
            "p50": float(np.percentile(a, 50)),
            "p84": float(np.percentile(a, 84)),
        }

    return {
        "nsim": int(nsim),
        "d_inj": float(d_inj),
        "cmb_ra_deg": float(vec_to_radec_deg(cmb_vec)[0]),
        "cmb_dec_deg": float(vec_to_radec_deg(cmb_vec)[1]),
        "fitted_ra_template_coeff_from_data": {"sin_ra_z": beta_sin, "cos_ra_z": beta_cos},
        "recovery_dipole_only": {
            "description": "Recover with dipole-only model (omit declination/RA nuisance templates).",
            "D": summarize(D_dip),
            "ang_to_inj_deg": summarize(ang_dip),
        },
        "recovery_omit_ra_templates": {
            "D": summarize(D_omit),
            "ang_to_inj_deg": summarize(ang_omit),
        },
        "recovery_with_ra_templates": {
            "D": summarize(D_with),
            "ang_to_inj_deg": summarize(ang_with),
        },
    }


def union_find_components(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        p = parent[x]
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        while x != p:
            nxt = parent[x]
            parent[x] = p
            x = nxt
        return p

    for i, j in pairs:
        ri = find(int(i))
        rj = find(int(j))
        if ri == rj:
            continue
        if rank[ri] < rank[rj]:
            parent[ri] = rj
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
        else:
            parent[rj] = ri
            rank[ri] += 1

    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        roots[i] = find(i)
    return roots


def run_component_merging(
    *,
    src_vec: np.ndarray,
    src_flux: np.ndarray,
    cutsel: np.ndarray,
    merge_radii_arcsec: list[float],
    nside: int,
    seen: np.ndarray,
    geom: PixGeom,
    max_iter: int,
    prior_strength_l2: float,
) -> dict[str, Any]:
    vec = np.asarray(src_vec[cutsel], dtype=float)
    flux = np.asarray(src_flux[cutsel], dtype=float)

    rows = []
    for r_arcsec in merge_radii_arcsec:
        r = float(r_arcsec)
        if r <= 0.0:
            keep = np.arange(vec.shape[0], dtype=np.int64)
        else:
            theta = np.deg2rad(r / 3600.0)
            chord = 2.0 * np.sin(theta / 2.0)
            tree = cKDTree(vec)
            pairs = tree.query_pairs(float(chord), output_type="ndarray")
            if pairs.size == 0:
                keep = np.arange(vec.shape[0], dtype=np.int64)
            else:
                roots = union_find_components(vec.shape[0], pairs)
                _, keep = np.unique(roots, return_index=True)
                keep = np.sort(keep)
        ip = hp.vec2pix(int(nside), vec[keep, 0], vec[keep, 1], vec[keep, 2], nest=False)
        y = np.bincount(ip, minlength=hp.nside2npix(int(nside))).astype(np.int64)
        fit = fit_model(
            y,
            seen,
            geom,
            model="dipole_dec",
            max_iter=max_iter,
            prior_strength_l2=prior_strength_l2,
        )
        rows.append(
            {
                "merge_radius_arcsec": r,
                "n_components_before": int(vec.shape[0]),
                "n_components_after": int(keep.size),
                "retained_fraction": float(keep.size / max(1, vec.shape[0])),
                "fit": fit,
            }
        )
    return {"rows": rows}


def run_jackknife(
    *,
    y_map: np.ndarray,
    seen: np.ndarray,
    geom: PixGeom,
    ra_bins: int,
    dec_bins: int,
    max_iter: int,
    prior_strength_l2: float,
) -> dict[str, Any]:
    ra = geom.ra_deg
    dec = geom.dec_deg

    dec_edges = np.quantile(dec[seen], np.linspace(0.0, 1.0, int(dec_bins) + 1))
    # avoid duplicate edges
    dec_edges[0] -= 1e-9
    dec_edges[-1] += 1e-9
    dec_id = np.clip(np.digitize(dec, dec_edges) - 1, 0, int(dec_bins) - 1)

    ra_id = np.floor((ra % 360.0) / (360.0 / float(ra_bins))).astype(int)
    ra_id = np.clip(ra_id, 0, int(ra_bins) - 1)

    region_id = dec_id * int(ra_bins) + ra_id
    nreg = int(ra_bins) * int(dec_bins)

    full = fit_model(
        y_map,
        seen,
        geom,
        model="dipole_dec",
        max_iter=max_iter,
        prior_strength_l2=prior_strength_l2,
    )
    b_full = np.asarray(full["b_vec"], dtype=float)

    rows = []
    b_rows = []
    for r in range(nreg):
        seen_r = seen & (region_id != r)
        if np.sum(seen_r) < 100:
            continue
        fit = fit_model(
            y_map,
            seen_r,
            geom,
            model="dipole_dec",
            max_iter=max_iter,
            prior_strength_l2=prior_strength_l2,
        )
        b = np.asarray(fit["b_vec"], dtype=float)
        rows.append(
            {
                "region": int(r),
                "n_pix": int(np.sum(seen_r)),
                "D": float(fit["D"]),
                "ra_deg": float(fit["ra_deg"]),
                "dec_deg": float(fit["dec_deg"]),
                "axis_angle_to_full_deg": float(axis_angle_deg(b, b_full)),
            }
        )
        b_rows.append(b)

    axis_scatter = float(np.nan)
    if b_rows:
        angs = np.array([axis_angle_deg(b, b_full) for b in b_rows], dtype=float)
        axis_scatter = float(np.nanstd(angs, ddof=1)) if angs.size > 1 else float(angs[0])

    return {
        "full_fit": full,
        "n_regions": int(nreg),
        "leave_one_out": rows,
        "axis_scatter_deg": axis_scatter,
    }


def run_hierarchical_fluxband_proxy(
    *,
    ipix: np.ndarray,
    flux: np.ndarray,
    bands: list[tuple[float, float | None]],
    npix: int,
    seen: np.ndarray,
    geom: PixGeom,
    max_iter: int,
    prior_strength_l2: float,
) -> dict[str, Any]:
    rows = []
    b_list = []
    C_list = []

    for lo, hi in bands:
        if hi is None:
            m = flux >= lo
            tag = f"{lo:g}+"
        else:
            m = (flux >= lo) & (flux < hi)
            tag = f"{lo:g}-{hi:g}"
        y = np.bincount(ipix[m], minlength=npix).astype(np.int64)
        fit = fit_model(
            y,
            seen,
            geom,
            model="dipole_dec",
            max_iter=max_iter,
            prior_strength_l2=prior_strength_l2,
        )
        rows.append({"band_mJy": tag, "n_src": int(np.sum(m)), "fit": fit})
        covb = fit.get("cov_b")
        if covb is not None:
            C = np.asarray(covb, dtype=float)
            if np.all(np.isfinite(C)):
                b_list.append(np.asarray(fit["b_vec"], dtype=float))
                C_list.append(C)

    out: dict[str, Any] = {"bands": rows}

    def combine(ids: list[int]) -> dict[str, Any] | None:
        if not ids:
            return None
        Cinvs = []
        rhs = np.zeros(3, dtype=float)
        for i in ids:
            C = C_list[i]
            try:
                Ci = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                continue
            Cinvs.append(Ci)
            rhs += Ci @ b_list[i]
        if not Cinvs:
            return None
        Csum = np.sum(Cinvs, axis=0)
        Cc = np.linalg.inv(Csum)
        bc = Cc @ rhs
        ra, dec = vec_to_radec_deg(bc)
        return {
            "b_vec": [float(x) for x in bc],
            "D": float(np.linalg.norm(bc)),
            "ra_deg": float(ra),
            "dec_deg": float(dec),
            "cov_b": [[float(x) for x in row] for row in Cc.tolist()],
        }

    idx_all = list(range(len(b_list)))
    out["common_all_bands"] = combine(idx_all)

    loo = []
    for i in idx_all:
        ids = [j for j in idx_all if j != i]
        loo.append({"leave_out_band_index": int(i), "common": combine(ids)})
    out["leave_one_band_out"] = loo
    out["note"] = (
        "Proxy for hierarchical common-dipole consistency within NVSS flux bands. "
        "True leave-one-survey-out requires additional survey catalogs (RACS-low/LoTSS)."
    )
    return out


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    c = summary["config"]
    ts = summary["template_sensitivity"]["models"]
    drift = summary["flux_cut_drift_mc"]
    inj = summary["injection_recovery"]
    merge = summary["component_merging"]["rows"]
    jk = summary["jackknife"]

    lines: list[str] = []
    lines.append("# NVSS Same-Logic Robustness Audit\n")
    lines.append(f"Generated: `{summary['meta']['created_utc']}`\n")
    lines.append("## Configuration\n")
    lines.append(f"- Catalog: `{c['catalog']}`")
    lines.append(f"- nside: `{c['nside']}`")
    lines.append(f"- Flux cuts (mJy): `{c['flux_cuts_mJy']}`")
    lines.append(f"- Sky mask: `dec >= {c['dec_min_deg']} deg`, `|b_gal| >= {c['gal_b_cut_deg']} deg`\n")

    lines.append("## 1) Flux-cut drift + correlated-cut null\n")
    lines.append(f"- Observed path/end/max axis drift (deg): `{drift['observed']}`")
    lines.append(f"- Null p-values (`nsim={drift['nsim']}`): path `{drift['p_path_ge_obs']:.4g}`, end-to-end `{drift['p_end_to_end_ge_obs']:.4g}`, max-pair `{drift['p_max_pair_ge_obs']:.4g}`\n")

    lines.append("## 2) Template sensitivity (fiducial cut)\n")
    for r in ts:
        lines.append(
            f"- `{r['model']}`: D=`{r['D']:.5f}`, (RA,Dec)=(`{r['ra_deg']:.1f}`, `{r['dec_deg']:.1f}`), "
            f"angle-to-CMB=`{r['angle_to_cmb_deg']:.1f} deg`, axis-shift-vs-baseline=`{r['axis_angle_to_baseline_deg']:.1f} deg`"
        )
    lines.append("")

    lines.append("## 3) Injection/recovery (template omission test)\n")
    lines.append(
        f"- Injected dipole: `D_inj={inj['d_inj']:.5f}` along CMB axis; injected RA-template coeffs from data fit: `{inj['fitted_ra_template_coeff_from_data']}`"
    )
    lines.append(
        f"- Dipole-only recovery: median D=`{inj['recovery_dipole_only']['D']['p50']:.5f}`, "
        f"median angle-to-inj=`{inj['recovery_dipole_only']['ang_to_inj_deg']['p50']:.2f} deg`"
    )
    lines.append(
        f"- Omit RA templates: median D=`{inj['recovery_omit_ra_templates']['D']['p50']:.5f}`, "
        f"median angle-to-inj=`{inj['recovery_omit_ra_templates']['ang_to_inj_deg']['p50']:.2f} deg`"
    )
    lines.append(
        f"- Include RA templates: median D=`{inj['recovery_with_ra_templates']['D']['p50']:.5f}`, "
        f"median angle-to-inj=`{inj['recovery_with_ra_templates']['ang_to_inj_deg']['p50']:.2f} deg`\n"
    )

    lines.append("## 4) Common-dipole proxy across NVSS flux bands\n")
    lines.append(f"- {summary['hierarchical_fluxband_proxy']['note']}\n")

    lines.append("## 5) Component-merging sensitivity\n")
    for r in merge:
        f = r["fit"]
        lines.append(
            f"- radius `{r['merge_radius_arcsec']}" + '"`: ' +
            f"N `{r['n_components_before']}` -> `{r['n_components_after']}`; "
            f"D=`{f['D']:.5f}`, (RA,Dec)=(`{f['ra_deg']:.1f}`, `{f['dec_deg']:.1f}`)"
        )
    lines.append("")

    lines.append("## 6) Geometry jackknife\n")
    lines.append(f"- Regions: `{jk['n_regions']}` leave-one-out fits")
    lines.append(f"- Axis scatter vs full fit: `{jk['axis_scatter_deg']:.3f} deg`\n")

    lines.append("## Caveat\n")
    lines.append(
        "This run is NVSS-only because additional radio catalogs were not local in this workspace. "
        "The same script logic can be extended to RACS-low and LoTSS once those catalogs/maps are staged."
    )

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/nvss/reference/NVSS.fit",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--flux-cuts", default="10,20,30,40,50")
    ap.add_argument("--fiducial-cut", type=float, default=20.0)
    ap.add_argument("--dec-min-deg", type=float, default=-20.0)
    ap.add_argument("--gal-b-cut-deg", type=float, default=10.0)
    ap.add_argument("--nsim-drift", type=int, default=600)
    ap.add_argument("--nsim-inject", type=int, default=400)
    ap.add_argument("--merge-radii-arcsec", default="0,22.5,45")
    ap.add_argument("--ra-jk-bins", type=int, default=8)
    ap.add_argument("--dec-jk-bins", type=int, default=4)
    ap.add_argument("--prior-strength-l2", type=float, default=20.0)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260208)
    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/nvss_same_logic_audit_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    cuts = sorted(float(x) for x in args.flux_cuts.split(",") if x.strip())
    merge_radii = sorted(float(x) for x in args.merge_radii_arcsec.split(",") if x.strip())
    if float(args.fiducial_cut) not in cuts:
        cuts.append(float(args.fiducial_cut))
        cuts = sorted(cuts)

    rng = np.random.default_rng(int(args.seed))

    # CMB dipole direction in equatorial coordinates used by the paper.
    cmb_ra = 167.94
    cmb_dec = -6.94
    cmb_vec = radec_to_unitvec(np.array([cmb_ra]), np.array([cmb_dec]))[0]

    with fits.open(args.catalog, memmap=True) as hdul:
        d = hdul[1].data
        ra = np.asarray(d["RAJ2000"], dtype=float)
        dec = np.asarray(d["DEJ2000"], dtype=float)
        flux = np.asarray(d["S1_4"], dtype=float)

    src_vec = radec_to_unitvec(ra, dec)
    glat = source_gal_lat_deg(src_vec)

    src_mask = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(flux)
        & (dec >= float(args.dec_min_deg))
        & (np.abs(glat) >= float(args.gal_b_cut_deg))
        & (flux >= min(cuts))
    )

    src_vec = src_vec[src_mask]
    flux = flux[src_mask]

    npix = hp.nside2npix(int(args.nside))
    ipix = hp.vec2pix(int(args.nside), src_vec[:, 0], src_vec[:, 1], src_vec[:, 2], nest=False)

    geom = build_pix_geom(int(args.nside))
    seen = (geom.dec_deg >= float(args.dec_min_deg)) & (np.abs(geom.glat_deg) >= float(args.gal_b_cut_deg))

    counts_by_cut = build_counts_by_cut(ipix=ipix, flux=flux, cuts=cuts, npix=npix)

    # 1) Flux-cut series and correlated-cut drift MC.
    flux_scan = []
    for c in cuts:
        fit = fit_model(
            counts_by_cut[c],
            seen,
            geom,
            model="dipole_dec",
            max_iter=int(args.max_iter),
            prior_strength_l2=float(args.prior_strength_l2),
        )
        b = np.asarray(fit["b_vec"], dtype=float)
        fit["cut_mJy"] = float(c)
        fit["angle_to_cmb_deg"] = float(ang_sep_deg(b, cmb_vec))
        flux_scan.append(fit)

    drift_mc = run_correlated_cut_mc(
        counts_by_cut=counts_by_cut,
        cuts=cuts,
        seen=seen,
        geom=geom,
        nsim=int(args.nsim_drift),
        seed=int(rng.integers(0, 2**31 - 1)),
        model="dipole_dec",
        max_iter=int(args.max_iter),
        prior_strength_l2=float(args.prior_strength_l2),
    )

    # 2) Template sensitivity at fiducial cut.
    y_fid = counts_by_cut[float(args.fiducial_cut)]
    template_sens = run_template_sensitivity(
        y_map=y_fid,
        seen=seen,
        geom=geom,
        model_list=["dipole", "dipole_dec", "dipole_ra", "dipole_dec_ra", "dipole_dec_ra_l2"],
        max_iter=int(args.max_iter),
        prior_strength_l2=float(args.prior_strength_l2),
        cmb_vec=cmb_vec,
    )

    # 3) Injection/recovery.
    inj = run_injection_recovery(
        y_data=y_fid,
        seen=seen,
        geom=geom,
        nsim=int(args.nsim_inject),
        seed=int(rng.integers(0, 2**31 - 1)),
        max_iter=int(args.max_iter),
        prior_strength_l2=float(args.prior_strength_l2),
        d_inj=0.00435,
        cmb_vec=cmb_vec,
    )

    # 4) Hierarchical common-dipole proxy across independent NVSS flux bands.
    hier = run_hierarchical_fluxband_proxy(
        ipix=ipix,
        flux=flux,
        bands=[(20.0, 30.0), (30.0, 50.0), (50.0, None)],
        npix=npix,
        seen=seen,
        geom=geom,
        max_iter=int(args.max_iter),
        prior_strength_l2=float(args.prior_strength_l2),
    )

    # 5) Component-merging sensitivity at fiducial cut.
    cutsel = flux >= float(args.fiducial_cut)
    merge = run_component_merging(
        src_vec=src_vec,
        src_flux=flux,
        cutsel=cutsel,
        merge_radii_arcsec=merge_radii,
        nside=int(args.nside),
        seen=seen,
        geom=geom,
        max_iter=int(args.max_iter),
        prior_strength_l2=float(args.prior_strength_l2),
    )

    # 6) Geometry jackknife.
    jk = run_jackknife(
        y_map=y_fid,
        seen=seen,
        geom=geom,
        ra_bins=int(args.ra_jk_bins),
        dec_bins=int(args.dec_jk_bins),
        max_iter=int(args.max_iter),
        prior_strength_l2=float(args.prior_strength_l2),
    )

    summary = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(args.seed),
        },
        "config": {
            "catalog": str(args.catalog),
            "outdir": str(outdir),
            "nside": int(args.nside),
            "flux_cuts_mJy": [float(c) for c in cuts],
            "fiducial_cut_mJy": float(args.fiducial_cut),
            "dec_min_deg": float(args.dec_min_deg),
            "gal_b_cut_deg": float(args.gal_b_cut_deg),
            "nsim_drift": int(args.nsim_drift),
            "nsim_inject": int(args.nsim_inject),
            "merge_radii_arcsec": [float(x) for x in merge_radii],
            "ra_jk_bins": int(args.ra_jk_bins),
            "dec_jk_bins": int(args.dec_jk_bins),
            "prior_strength_l2": float(args.prior_strength_l2),
            "max_iter": int(args.max_iter),
            "n_sources_after_mask_and_mincut": int(src_vec.shape[0]),
        },
        "flux_scan": flux_scan,
        "flux_cut_drift_mc": drift_mc,
        "template_sensitivity": template_sens,
        "injection_recovery": inj,
        "hierarchical_fluxband_proxy": hier,
        "component_merging": merge,
        "jackknife": jk,
    }

    (outdir / "nvss_same_logic_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(outdir / "master_report.md", summary)

    print(json.dumps({
        "outdir": str(outdir),
        "json": str(outdir / "nvss_same_logic_audit.json"),
        "report": str(outdir / "master_report.md"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
