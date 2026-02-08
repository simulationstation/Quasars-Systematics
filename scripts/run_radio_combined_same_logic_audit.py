#!/usr/bin/env python3
"""Combined radio dipole robustness audit on NVSS + RACS-low + LoTSS.

This script ports the "same-logic" robustness approach to a three-survey radio
stack using publicly available catalogs staged locally:
  - NVSS (1.4 GHz)
  - RACS-low DR1 (888 MHz; CASDA TAP export)
  - LoTSS-DR2 source catalog (144 MHz)

Core outputs:
  1) Per-survey flux-cut stability scans.
  2) Joint shared-dipole fit with survey-specific nuisance templates.
  3) Leave-one-survey-out shared-dipole fits.
  4) Injection/recovery under omitted-vs-included nuisance templates.
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


def radec_to_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    c = np.cos(dec)
    return np.column_stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)])


def vec_to_radec_deg(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return float("nan"), float("nan")
    u = v / n
    ra = float(np.degrees(np.arctan2(u[1], u[0])) % 360.0)
    dec = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
    return ra, dec


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
    b = np.asarray(base, dtype=float)
    y = np.asarray(t, dtype=float).reshape(-1)
    coef, *_ = np.linalg.lstsq(b, y, rcond=None)
    return y - b @ coef


def fit_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 300,
    beta_init: np.ndarray | None = None,
    prior_prec_diag: np.ndarray | None = None,
    compute_cov: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
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
        eta = np.clip(X @ beta, -25.0, 25.0)
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
        eta = np.clip(X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        fisher = X.T @ (mu[:, None] * X)
        fisher += np.diag(prior)
        cov = np.linalg.inv(fisher)
    except Exception:
        cov = None
    return beta, cov


@dataclass(frozen=True)
class SurveyRaw:
    name: str
    freq_mhz: float
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    flux_mjy: np.ndarray


@dataclass(frozen=True)
class SurveyMap:
    name: str
    freq_mhz: float
    cut_mjy: float
    y_seen: np.ndarray  # counts on seen pixels
    nvec_seen: np.ndarray  # unit vectors on seen pixels
    ra_seen: np.ndarray
    dec_seen: np.ndarray
    pix_seen: np.ndarray
    npix_total: int
    n_src: int


def load_nvss(path: Path) -> SurveyRaw:
    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        ra = np.asarray(d["RAJ2000"], dtype=float)
        dec = np.asarray(d["DEJ2000"], dtype=float)
        flux = np.asarray(d["S1_4"], dtype=float)
    return SurveyRaw(name="NVSS", freq_mhz=1400.0, ra_deg=ra, dec_deg=dec, flux_mjy=flux)


def load_lotss(path: Path) -> SurveyRaw:
    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        ra = np.asarray(d["RA"], dtype=float)
        dec = np.asarray(d["DEC"], dtype=float)
        flux = np.asarray(d["Total_flux"], dtype=float)
    return SurveyRaw(name="LoTSS-DR2", freq_mhz=144.0, ra_deg=ra, dec_deg=dec, flux_mjy=flux)


def load_racs_csv(path: Path) -> SurveyRaw:
    import gzip

    ra = []
    dec = []
    flux = []
    with gzip.open(path, "rt", newline="") as f:
        # header: ra,dec,total_flux_source,gal_lat
        _ = f.readline()
        for line in f:
            s = line.strip().split(",")
            if len(s) < 3:
                continue
            try:
                r = float(s[0])
                d = float(s[1])
                fl = float(s[2])
            except ValueError:
                continue
            ra.append(r)
            dec.append(d)
            flux.append(fl)
    return SurveyRaw(
        name="RACS-low",
        freq_mhz=888.0,
        ra_deg=np.asarray(ra, dtype=float),
        dec_deg=np.asarray(dec, dtype=float),
        flux_mjy=np.asarray(flux, dtype=float),
    )


def gal_lat_from_vec(v: np.ndarray) -> np.ndarray:
    gv = v @ R_EQ_TO_GAL.T
    return np.degrees(np.arcsin(np.clip(gv[:, 2], -1.0, 1.0)))


def build_survey_map(
    raw: SurveyRaw,
    *,
    cut_mjy: float,
    nside: int,
    gal_b_cut_deg: float,
) -> SurveyMap:
    v = radec_to_unitvec(raw.ra_deg, raw.dec_deg)
    glat = gal_lat_from_vec(v)
    m = (
        np.isfinite(raw.ra_deg)
        & np.isfinite(raw.dec_deg)
        & np.isfinite(raw.flux_mjy)
        & (raw.flux_mjy >= float(cut_mjy))
        & (np.abs(glat) >= float(gal_b_cut_deg))
    )
    v = v[m]
    ra = raw.ra_deg[m]
    dec = raw.dec_deg[m]
    ipix = hp.vec2pix(int(nside), v[:, 0], v[:, 1], v[:, 2], nest=False)
    npix = hp.nside2npix(int(nside))
    y = np.bincount(ipix, minlength=npix).astype(np.int64)
    seen = y > 0

    vx, vy, vz = hp.pix2vec(int(nside), np.arange(npix), nest=False)
    nvec = np.column_stack([vx, vy, vz]).astype(float)
    ra_pix = (np.degrees(np.arctan2(nvec[:, 1], nvec[:, 0])) + 360.0) % 360.0
    dec_pix = np.degrees(np.arcsin(np.clip(nvec[:, 2], -1.0, 1.0)))

    return SurveyMap(
        name=raw.name,
        freq_mhz=raw.freq_mhz,
        cut_mjy=float(cut_mjy),
        y_seen=y[seen].astype(np.int64, copy=False),
        nvec_seen=nvec[seen],
        ra_seen=ra_pix[seen],
        dec_seen=dec_pix[seen],
        pix_seen=np.where(seen)[0].astype(np.int64),
        npix_total=int(npix),
        n_src=int(np.sum(m)),
    )


def build_templates(
    nvec: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    model: str,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    n = nvec.shape[0]
    if model == "dipole_only":
        return np.zeros((n, 0), dtype=float), [], None

    base = np.column_stack([np.ones(n), nvec])
    valid = np.ones(n, dtype=bool)

    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(raw: np.ndarray, name: str) -> None:
        t = zscore(raw, valid)
        t = residualize_against(base, t)
        t = zscore(t, valid)
        cols.append(t)
        names.append(name)

    add(dec_deg, "dec_z")
    add(dec_deg**2, "dec2_z")
    rr = np.deg2rad(ra_deg)
    add(np.sin(rr), "sin_ra_z")
    add(np.cos(rr), "cos_ra_z")

    prior = np.zeros(len(cols), dtype=float)
    if model == "dec_ra_l2":
        nx, ny, nz = nvec[:, 0], nvec[:, 1], nvec[:, 2]
        q_terms = [
            (nx * ny, "q_xy"),
            (nx * nz, "q_xz"),
            (ny * nz, "q_yz"),
            (nx * nx - ny * ny, "q_x2my2"),
            (3.0 * nz * nz - 1.0, "q_3z2m1"),
        ]
        for raw, name in q_terms:
            add(raw, name)
            prior = np.append(prior, 20.0)

    T = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
    if T.shape[1] == 0:
        prior = None
    return T, names, prior


def fit_joint_shared_dipole(
    surveys: list[SurveyMap],
    *,
    model: str,
    max_iter: int,
    l2_ridge: float,
) -> dict[str, Any]:
    # Precompute templates by survey.
    Ts = []
    tnames = []
    priors = []
    for s in surveys:
        T, names, prior = build_templates(s.nvec_seen, s.ra_seen, s.dec_seen, model=model)
        if prior is not None and prior.size:
            prior = prior.copy()
            if model == "dec_ra_l2" and prior.size >= 9:
                prior[-5:] = float(l2_ridge)
        Ts.append(T)
        tnames.append(names)
        priors.append(prior)

    n_surv = len(surveys)
    n_t = [T.shape[1] for T in Ts]
    p_total = 3 + n_surv + sum(n_t)

    # parameter indexing
    # [b(3)] [a_s (n_surv)] [g_s blocks]
    g_start = 3 + n_surv
    g_offsets = []
    cur = g_start
    for k in n_t:
        g_offsets.append(cur)
        cur += k

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        b = theta[:3]
        a = theta[3 : 3 + n_surv]
        g = [theta[g_offsets[i] : g_offsets[i] + n_t[i]] for i in range(n_surv)]
        return b, a, g

    theta0 = np.zeros(p_total, dtype=float)
    for i, s in enumerate(surveys):
        m = max(1.0, float(np.mean(s.y_seen)))
        theta0[3 + i] = math.log(m)

    prior_diag = np.zeros(p_total, dtype=float)
    for i in range(n_surv):
        if priors[i] is not None and priors[i].size:
            prior_diag[g_offsets[i] : g_offsets[i] + n_t[i]] = priors[i]

    def fun_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b, a, g = unpack(theta)
        nll = 0.0
        grad = np.zeros_like(theta)
        gb = np.zeros(3, dtype=float)
        ga = np.zeros(n_surv, dtype=float)
        gg = [np.zeros(n_t[i], dtype=float) for i in range(n_surv)]
        for i, s in enumerate(surveys):
            eta = a[i] + s.nvec_seen @ b
            if n_t[i]:
                eta = eta + Ts[i] @ g[i]
            eta = np.clip(eta, -25.0, 25.0)
            mu = np.exp(eta)
            r = mu - s.y_seen
            nll += float(np.sum(mu - s.y_seen * eta))
            gb += s.nvec_seen.T @ r
            ga[i] += float(np.sum(r))
            if n_t[i]:
                gg[i] += Ts[i].T @ r

        # ridge prior
        nll += float(0.5 * np.sum(prior_diag * (theta * theta)))
        grad[:3] = gb
        grad[3 : 3 + n_surv] = ga
        for i in range(n_surv):
            if n_t[i]:
                grad[g_offsets[i] : g_offsets[i] + n_t[i]] = gg[i]
        grad += prior_diag * theta
        return float(nll), np.asarray(grad, dtype=float)

    res = minimize(
        lambda th: fun_and_grad(th)[0],
        theta0,
        jac=lambda th: fun_and_grad(th)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    th = np.asarray(res.x, dtype=float)
    b, a, g = unpack(th)
    ra, dec = vec_to_radec_deg(b)
    out = {
        "model": model,
        "converged": bool(res.success),
        "message": str(res.message),
        "nll": float(res.fun),
        "b_vec": [float(x) for x in b],
        "D": float(np.linalg.norm(b)),
        "ra_deg": float(ra),
        "dec_deg": float(dec),
        "survey_intercepts": {surveys[i].name: float(a[i]) for i in range(n_surv)},
        "survey_template_names": {surveys[i].name: tnames[i] for i in range(n_surv)},
        "survey_template_coefs": {
            surveys[i].name: [float(x) for x in g[i]] for i in range(n_surv)
        },
    }
    return out


def fit_single_survey(s: SurveyMap, *, model: str, max_iter: int, l2_ridge: float) -> dict[str, Any]:
    T, names, prior_t = build_templates(s.nvec_seen, s.ra_seen, s.dec_seen, model=model)
    X = np.column_stack([np.ones(s.y_seen.size), s.nvec_seen, T])
    prior = np.zeros(X.shape[1], dtype=float)
    if prior_t is not None and prior_t.size:
        prior[4:] = prior_t
        if model == "dec_ra_l2" and prior_t.size >= 9:
            prior[-5:] = float(l2_ridge)
    beta, cov = fit_poisson_glm(X, s.y_seen, max_iter=max_iter, prior_prec_diag=prior, compute_cov=True)
    b = np.asarray(beta[1:4], dtype=float)
    ra, dec = vec_to_radec_deg(b)
    out: dict[str, Any] = {
        "survey": s.name,
        "model": model,
        "cut_mjy": float(s.cut_mjy),
        "n_src": int(s.n_src),
        "n_pix_seen": int(s.y_seen.size),
        "D": float(np.linalg.norm(b)),
        "ra_deg": float(ra),
        "dec_deg": float(dec),
        "b_vec": [float(x) for x in b],
        "template_names": names,
        "coef": [float(x) for x in beta],
    }
    if cov is not None:
        cb = np.asarray(cov[1:4, 1:4], dtype=float)
        out["cov_b"] = [[float(x) for x in row] for row in cb.tolist()]
    else:
        out["cov_b"] = None
    return out


def summarize_series(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "p16": float(np.percentile(a, 16)),
        "p50": float(np.percentile(a, 50)),
        "p84": float(np.percentile(a, 84)),
    }


def run_injection_recovery_joint(
    surveys: list[SurveyMap],
    *,
    d_inj: float,
    cmb_vec: np.ndarray,
    nsim: int,
    seed: int,
    max_iter: int,
    l2_ridge: float,
) -> dict[str, Any]:
    # Generation model: survey-specific nuisance without dipole + injected common dipole.
    rng = np.random.default_rng(int(seed))
    b_inj = float(d_inj) * (cmb_vec / np.linalg.norm(cmb_vec))

    gen_params = []
    for s in surveys:
        T, names, _ = build_templates(s.nvec_seen, s.ra_seen, s.dec_seen, model="dec_ra")
        X = np.column_stack([np.ones(s.y_seen.size), T])
        b0, _ = fit_poisson_glm(X, s.y_seen, max_iter=max_iter, compute_cov=False)
        eta_base = X @ b0
        gen_params.append((T, names, b0, eta_base))

    D_dip = []
    D_decra = []
    D_decra_l2 = []
    ang_dip = []
    ang_decra = []
    ang_decra_l2 = []

    for _ in range(int(nsim)):
        sim_maps = []
        for i, s in enumerate(surveys):
            eta = gen_params[i][3] + s.nvec_seen @ b_inj
            mu = np.exp(np.clip(eta, -25.0, 25.0))
            y = rng.poisson(mu)
            sim_maps.append(
                SurveyMap(
                    name=s.name,
                    freq_mhz=s.freq_mhz,
                    cut_mjy=s.cut_mjy,
                    y_seen=y,
                    nvec_seen=s.nvec_seen,
                    ra_seen=s.ra_seen,
                    dec_seen=s.dec_seen,
                    pix_seen=s.pix_seen,
                    npix_total=s.npix_total,
                    n_src=int(np.sum(y)),
                )
            )

        fit_dip = fit_joint_shared_dipole(sim_maps, model="dipole_only", max_iter=max_iter, l2_ridge=l2_ridge)
        fit_decra = fit_joint_shared_dipole(sim_maps, model="dec_ra", max_iter=max_iter, l2_ridge=l2_ridge)
        fit_l2 = fit_joint_shared_dipole(sim_maps, model="dec_ra_l2", max_iter=max_iter, l2_ridge=l2_ridge)

        b_dip = np.asarray(fit_dip["b_vec"], dtype=float)
        b_decra = np.asarray(fit_decra["b_vec"], dtype=float)
        b_l2 = np.asarray(fit_l2["b_vec"], dtype=float)

        D_dip.append(float(np.linalg.norm(b_dip)))
        D_decra.append(float(np.linalg.norm(b_decra)))
        D_decra_l2.append(float(np.linalg.norm(b_l2)))
        ang_dip.append(float(ang_sep_deg(b_dip, b_inj)))
        ang_decra.append(float(ang_sep_deg(b_decra, b_inj)))
        ang_decra_l2.append(float(ang_sep_deg(b_l2, b_inj)))

    return {
        "nsim": int(nsim),
        "d_inj": float(d_inj),
        "cmb_ra_deg": float(vec_to_radec_deg(cmb_vec)[0]),
        "cmb_dec_deg": float(vec_to_radec_deg(cmb_vec)[1]),
        "recover_dipole_only": {"D": summarize_series(D_dip), "ang_to_inj_deg": summarize_series(ang_dip)},
        "recover_dec_ra": {"D": summarize_series(D_decra), "ang_to_inj_deg": summarize_series(ang_decra)},
        "recover_dec_ra_l2": {"D": summarize_series(D_decra_l2), "ang_to_inj_deg": summarize_series(ang_decra_l2)},
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    cmb = np.asarray(summary["meta"]["cmb_b_vec"], dtype=float)
    lines: list[str] = []
    lines.append("# Combined Radio Same-Logic Audit\n")
    lines.append(f"Generated: `{summary['meta']['created_utc']}`\n")
    lines.append("## Surveys\n")
    for s in summary["inputs"]["surveys"]:
        lines.append(
            f"- `{s['name']}` `{s['freq_mhz']:.0f} MHz`, cut `{s['cut_mjy']} mJy`, nsrc `{s['n_src']}`"
        )
    lines.append("")

    lines.append("## Per-survey Flux-Cut Scans (`dec_ra` model)\n")
    for name, rows in summary["per_survey_flux_scans"].items():
        lines.append(f"- `{name}`")
        for r in rows:
            b = np.asarray(r["b_vec"], dtype=float)
            lines.append(
                f"  cut `{r['cut_mjy']}` mJy: D `{r['D']:.5f}`, (RA,Dec)=(`{r['ra_deg']:.1f}`,`{r['dec_deg']:.1f}`), "
                f"angle-to-CMB `{ang_sep_deg(b, cmb):.1f}` deg"
            )
    lines.append("")

    lines.append("## Joint Shared-Dipole Fits\n")
    for r in summary["joint_template_sensitivity"]:
        b = np.asarray(r["b_vec"], dtype=float)
        lines.append(
            f"- `{r['model']}`: D `{r['D']:.5f}`, (RA,Dec)=(`{r['ra_deg']:.1f}`,`{r['dec_deg']:.1f}`), "
            f"angle-to-CMB `{ang_sep_deg(b, cmb):.1f}` deg"
        )
    lines.append("")

    lines.append("## Leave-One-Survey-Out (joint `dec_ra`)\n")
    for r in summary["leave_one_survey_out"]:
        b = np.asarray(r["fit"]["b_vec"], dtype=float)
        lines.append(
            f"- leave out `{r['left_out']}`: D `{r['fit']['D']:.5f}`, "
            f"(RA,Dec)=(`{r['fit']['ra_deg']:.1f}`,`{r['fit']['dec_deg']:.1f}`), "
            f"angle-to-CMB `{ang_sep_deg(b, cmb):.1f}` deg"
        )
    lines.append("")

    inj = summary["injection_recovery_joint"]
    lines.append("## Injection/Recovery (joint)\n")
    lines.append(f"- injected D `{inj['d_inj']:.5f}` along CMB direction")
    lines.append(
        f"- `dipole_only`: D p50 `{inj['recover_dipole_only']['D']['p50']:.5f}`, "
        f"angle p50 `{inj['recover_dipole_only']['ang_to_inj_deg']['p50']:.2f}` deg"
    )
    lines.append(
        f"- `dec_ra`: D p50 `{inj['recover_dec_ra']['D']['p50']:.5f}`, "
        f"angle p50 `{inj['recover_dec_ra']['ang_to_inj_deg']['p50']:.2f}` deg"
    )
    lines.append(
        f"- `dec_ra_l2`: D p50 `{inj['recover_dec_ra_l2']['D']['p50']:.5f}`, "
        f"angle p50 `{inj['recover_dec_ra_l2']['ang_to_inj_deg']['p50']:.2f}` deg"
    )
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nvss-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/nvss/reference/NVSS.fit",
    )
    ap.add_argument(
        "--racs-csv-gz",
        default="data/external/radio_dipole/racs_low/racs_low_dr1_sources_galacticcut_v2021_08_v02_mincols.csv.gz",
    )
    ap.add_argument(
        "--lotss-fits",
        default="data/external/radio_dipole/lotss_dr2/LoTSS_DR2_v110_masked.srl.fits",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--gal-b-cut-deg", type=float, default=10.0)

    ap.add_argument("--nvss-cuts", default="10,20,30,40,50")
    ap.add_argument("--racs-cuts", default="10,20,30")
    ap.add_argument("--lotss-cuts", default="5,10,20")
    ap.add_argument("--nvss-fid", type=float, default=20.0)
    ap.add_argument("--racs-fid", type=float, default=20.0)
    ap.add_argument("--lotss-fid", type=float, default=5.0)

    ap.add_argument("--nsim-inj", type=int, default=1000)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--l2-ridge", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=20260208)
    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/radio_combined_same_logic_audit_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    # CMB dipole direction (equatorial) used in the target paper.
    cmb_ra = 167.94
    cmb_dec = -6.94
    cmb_vec = radec_to_unitvec(np.array([cmb_ra]), np.array([cmb_dec]))[0]

    nvss_raw = load_nvss(Path(args.nvss_fits))
    racs_raw = load_racs_csv(Path(args.racs_csv_gz))
    lotss_raw = load_lotss(Path(args.lotss_fits))

    def parse_cuts(s: str) -> list[float]:
        return sorted(float(x) for x in s.split(",") if x.strip())

    nvss_cuts = parse_cuts(args.nvss_cuts)
    racs_cuts = parse_cuts(args.racs_cuts)
    lotss_cuts = parse_cuts(args.lotss_cuts)

    # Per-survey scans.
    scans: dict[str, list[dict[str, Any]]] = {"NVSS": [], "RACS-low": [], "LoTSS-DR2": []}
    for c in nvss_cuts:
        m = build_survey_map(nvss_raw, cut_mjy=c, nside=args.nside, gal_b_cut_deg=args.gal_b_cut_deg)
        scans["NVSS"].append(fit_single_survey(m, model="dec_ra", max_iter=args.max_iter, l2_ridge=args.l2_ridge))
    for c in racs_cuts:
        m = build_survey_map(racs_raw, cut_mjy=c, nside=args.nside, gal_b_cut_deg=args.gal_b_cut_deg)
        scans["RACS-low"].append(fit_single_survey(m, model="dec_ra", max_iter=args.max_iter, l2_ridge=args.l2_ridge))
    for c in lotss_cuts:
        m = build_survey_map(lotss_raw, cut_mjy=c, nside=args.nside, gal_b_cut_deg=args.gal_b_cut_deg)
        scans["LoTSS-DR2"].append(fit_single_survey(m, model="dec_ra", max_iter=args.max_iter, l2_ridge=args.l2_ridge))

    # Fiducial combined maps.
    s_nvss = build_survey_map(nvss_raw, cut_mjy=args.nvss_fid, nside=args.nside, gal_b_cut_deg=args.gal_b_cut_deg)
    s_racs = build_survey_map(racs_raw, cut_mjy=args.racs_fid, nside=args.nside, gal_b_cut_deg=args.gal_b_cut_deg)
    s_lotss = build_survey_map(lotss_raw, cut_mjy=args.lotss_fid, nside=args.nside, gal_b_cut_deg=args.gal_b_cut_deg)
    surveys = [s_nvss, s_racs, s_lotss]

    joint_rows = []
    for model in ["dipole_only", "dec_ra", "dec_ra_l2"]:
        joint_rows.append(
            fit_joint_shared_dipole(surveys, model=model, max_iter=args.max_iter, l2_ridge=args.l2_ridge)
        )

    # Leave-one-survey-out on the dec_ra model.
    loo_rows = []
    for i, s in enumerate(surveys):
        keep = [x for j, x in enumerate(surveys) if j != i]
        fit = fit_joint_shared_dipole(keep, model="dec_ra", max_iter=args.max_iter, l2_ridge=args.l2_ridge)
        loo_rows.append({"left_out": s.name, "fit": fit})

    inj = run_injection_recovery_joint(
        surveys,
        d_inj=0.00435,
        cmb_vec=cmb_vec,
        nsim=int(args.nsim_inj),
        seed=int(rng.integers(0, 2**31 - 1)),
        max_iter=int(args.max_iter),
        l2_ridge=float(args.l2_ridge),
    )

    summary = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(args.seed),
            "cmb_ra_deg": cmb_ra,
            "cmb_dec_deg": cmb_dec,
            "cmb_b_vec": [float(x) for x in cmb_vec],
        },
        "config": {
            "nside": int(args.nside),
            "gal_b_cut_deg": float(args.gal_b_cut_deg),
            "nvss_cuts": nvss_cuts,
            "racs_cuts": racs_cuts,
            "lotss_cuts": lotss_cuts,
            "nvss_fid": float(args.nvss_fid),
            "racs_fid": float(args.racs_fid),
            "lotss_fid": float(args.lotss_fid),
            "nsim_inj": int(args.nsim_inj),
            "max_iter": int(args.max_iter),
            "l2_ridge": float(args.l2_ridge),
        },
        "inputs": {
            "nvss_fits": str(args.nvss_fits),
            "racs_csv_gz": str(args.racs_csv_gz),
            "lotss_fits": str(args.lotss_fits),
            "surveys": [
                {"name": s.name, "freq_mhz": s.freq_mhz, "cut_mjy": s.cut_mjy, "n_src": s.n_src}
                for s in surveys
            ],
        },
        "per_survey_flux_scans": scans,
        "joint_template_sensitivity": joint_rows,
        "leave_one_survey_out": loo_rows,
        "injection_recovery_joint": inj,
    }

    jpath = outdir / "radio_combined_same_logic_audit.json"
    rpath = outdir / "master_report.md"
    jpath.write_text(json.dumps(summary, indent=2) + "\n")
    write_report(rpath, summary)

    print(
        json.dumps(
            {"outdir": str(outdir), "json": str(jpath), "report": str(rpath)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

