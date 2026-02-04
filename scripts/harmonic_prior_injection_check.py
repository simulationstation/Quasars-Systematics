#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        out = np.zeros_like(x)
        return out
    m = float(np.mean(x[valid]))
    s = float(np.std(x[valid]))
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


@dataclass(frozen=True)
class Mask:
    mask: np.ndarray
    seen: np.ndarray


def build_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> Mask:
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    # Pixels with zero sources (plus neighbors), matching the Secrest masking behavior.
    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        indices = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            indices[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[indices] = True  # -1 neighbor indices map to last pixel; keep for behavior-compatibility

    # Optional exclusion discs.
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

    # Galactic latitude cut.
    _lon_pix, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return Mask(mask=mask, seen=~mask)


def fit_poisson_glm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    offset: np.ndarray | None,
    max_iter: int,
    prior_prec_diag: np.ndarray | None = None,
) -> np.ndarray:
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    off = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)

    mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
    beta0 = np.zeros(X.shape[1], dtype=float)
    beta0[0] = math.log(mu0)

    prec = None if prior_prec_diag is None else np.asarray(prior_prec_diag, dtype=float).reshape(X.shape[1])

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = off + X @ beta
        eta = np.clip(eta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        if prec is not None:
            nll = float(nll + 0.5 * np.sum(prec * beta * beta))
            grad = grad + prec * beta
        return nll, np.asarray(grad, dtype=float)

    res = minimize(
        lambda b: fun_and_grad(b)[0],
        beta0,
        jac=lambda b: fun_and_grad(b)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    return np.asarray(res.x, dtype=float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Dipole injection check for the harmonic-prior method under low-ell (ell>=2) contamination. "
            "Fits baseline vs free harmonics vs harmonics with a Gaussian C_ell prior."
        )
    )
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1-max", type=float, default=16.6)

    ap.add_argument("--harmonic-lmax", type=int, default=5)
    ap.add_argument(
        "--cl-json",
        default="REPORTS/Q_D_RES_2_2/data/lognormal_cov_w1max16p6_n500/lognormal_mocks_cov.json",
        help="C_ell prior source (expects cl_estimate.cl_signal).",
    )
    ap.add_argument("--true-cl-scale", type=float, default=10.0, help="Scale applied to the injected low-ell field.")
    ap.add_argument("--fit-prior-scale", type=float, default=1.0, help="Scale applied to the harmonic prior in the fit.")
    ap.add_argument("--prior-min-cl", type=float, default=1e-12, help="Floor for non-positive C_ell values.")

    ap.add_argument("--n-mocks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-iter", type=int, default=200)

    ap.add_argument("--dipole-amp", type=float, default=0.01678)
    ap.add_argument("--dipole-axis-lb", default="264.021,48.253", help="Injected dipole axis (l,b) in degrees.")

    ap.add_argument(
        "--out-json",
        default="REPORTS/arxiv_amplitude_multipole_prior_injection/data/lowell_injection_validation.json",
    )
    ap.add_argument(
        "--out-fig",
        default="REPORTS/arxiv_amplitude_multipole_prior_injection/figures/lowell_injection_validation.png",
    )

    args = ap.parse_args(argv)

    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits

    nside = int(args.nside)
    npix = hp.nside2npix(nside)
    w1_max = float(args.w1_max)

    # Load catalog (minimal columns).
    with fits.open(args.catalog, memmap=True) as hdul:
        d = hdul[1].data
        w1 = np.asarray(d["w1"], dtype=float)
        w1cov = np.asarray(d["w1cov"], dtype=float)
        l = np.asarray(d["l"], dtype=float)
        b = np.asarray(d["b"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    base &= w1cov >= float(args.w1cov_min)

    th = np.deg2rad(90.0 - b[base])
    ph = np.deg2rad(l[base])
    ipix_base = hp.ang2pix(nside, th, ph, nest=False).astype(np.int64)

    # Mask and pixel geometry.
    m = build_mask(
        nside=nside,
        ipix_base=ipix_base,
        exclude_mask_fits=str(args.exclude_mask_fits) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )
    seen = m.seen

    lon_pix, lat_pix = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    pix_unit = lb_to_unitvec(lon_pix, lat_pix)

    # Ecliptic latitude template.
    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    elat_deg = sc_pix.barycentricmeanecliptic.lat.deg.astype(float)
    abs_elat_z_seen = zscore(np.abs(elat_deg), seen)[seen]

    # Build real-data counts at W1<=W1_max.
    keep = base & (w1 <= w1_max)
    thk = np.deg2rad(90.0 - b[keep])
    phk = np.deg2rad(l[keep])
    ipix = hp.ang2pix(nside, thk, phk, nest=False).astype(np.int64)
    counts = np.bincount(ipix, minlength=npix).astype(float)
    y_data = counts[seen]

    n_seen = pix_unit[seen]

    # Base intensity map for mocks: fit WITHOUT a dipole term so the injected dipole is identifiable.
    X_mu = np.column_stack([np.ones_like(y_data), abs_elat_z_seen])
    beta_mu = fit_poisson_glm(X_mu, y_data, offset=None, max_iter=int(args.max_iter))
    mu_hat = np.exp(np.clip(X_mu @ beta_mu, -25.0, 25.0))

    # Fit design for recovery.
    X_fit_base = np.column_stack([np.ones_like(y_data), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2], abs_elat_z_seen])

    # Harmonic templates (fixed for all mocks).
    lmax = int(args.harmonic_lmax)
    if lmax < 2:
        raise SystemExit("--harmonic-lmax must be >= 2")

    try:
        from scipy.special import sph_harm as _sph_harm  # type: ignore

        def sph_harm(l_: int, m_: int, th_: np.ndarray, ph_: np.ndarray) -> np.ndarray:
            return _sph_harm(m_, l_, ph_, th_)

    except Exception:
        from scipy.special import sph_harm_y as _sph_harm_y  # type: ignore

        def sph_harm(l_: int, m_: int, th_: np.ndarray, ph_: np.ndarray) -> np.ndarray:
            return _sph_harm_y(l_, m_, th_, ph_)

    th_pix = np.deg2rad(90.0 - lat_pix)
    ph_pix = np.deg2rad(lon_pix % 360.0)

    harm_cols_seen: list[np.ndarray] = []
    harm_ell: list[int] = []
    harm_std_raw: list[float] = []

    def harm_col(raw_full: np.ndarray) -> tuple[np.ndarray, float] | None:
        raw_full = np.asarray(raw_full, dtype=float)
        raw0 = raw_full - float(np.mean(raw_full[seen]))
        std = float(np.std(raw0[seen]))
        if not np.isfinite(std) or std <= 0.0:
            return None
        return (raw0 / std)[seen], std

    for ell in range(2, lmax + 1):
        y0 = sph_harm(ell, 0, th_pix, ph_pix).real.astype(float)
        col = harm_col(y0)
        if col is not None:
            harm_cols_seen.append(col[0])
            harm_ell.append(int(ell))
            harm_std_raw.append(float(col[1]))
        for m_ in range(1, ell + 1):
            ylm = sph_harm(ell, m_, th_pix, ph_pix)
            yr = (np.sqrt(2.0) * ylm.real).astype(float)
            yi = (np.sqrt(2.0) * ylm.imag).astype(float)
            colr = harm_col(yr)
            if colr is not None:
                harm_cols_seen.append(colr[0])
                harm_ell.append(int(ell))
                harm_std_raw.append(float(colr[1]))
            coli = harm_col(yi)
            if coli is not None:
                harm_cols_seen.append(coli[0])
                harm_ell.append(int(ell))
                harm_std_raw.append(float(coli[1]))

    X_fit_harm = np.column_stack([X_fit_base] + harm_cols_seen)

    # Load C_ell prior.
    cl_obj = json.loads(Path(str(args.cl_json)).read_text())
    cl = np.asarray(cl_obj["cl_estimate"]["cl_signal"], dtype=float)
    if cl.size <= lmax:
        raise SystemExit(f"C_ell array too short for lmax={lmax}: size={cl.size}")

    cl_floor = float(args.prior_min_cl)
    if not np.isfinite(cl_floor) or cl_floor <= 0.0:
        raise SystemExit("--prior-min-cl must be finite and > 0")

    # Build injection C_ell (only low-ell).
    cl_inj = np.zeros(lmax + 1, dtype=float)
    for ell in range(2, lmax + 1):
        c = float(cl[ell])
        if not np.isfinite(c) or c <= 0.0:
            c = cl_floor
        cl_inj[ell] = c * float(args.true_cl_scale)

    # Prior precision for fit (only harmonics regularized; others unpenalized).
    prior_prec = np.zeros(X_fit_harm.shape[1], dtype=float)
    for j, (ell, std_raw) in enumerate(zip(harm_ell, harm_std_raw, strict=True)):
        c = float(cl[ell])
        if not np.isfinite(c) or c <= 0.0:
            c = cl_floor
        var_beta = float(max(cl_floor, c) * float(args.fit_prior_scale) * (float(std_raw) ** 2))
        prior_prec[5 + j] = 1.0 / float(max(1e-20, var_beta))

    # Injected dipole vector.
    inj_l, inj_b = (float(x) for x in str(args.dipole_axis_lb).split(","))
    inj_axis = lb_to_unitvec(np.array([inj_l]), np.array([inj_b]))[0]
    b_inj = float(args.dipole_amp) * inj_axis
    delta_dip = n_seen @ b_inj

    rng = np.random.default_rng(int(args.seed))

    def fit_case(y: np.ndarray, mode: str) -> np.ndarray:
        if mode == "baseline":
            beta = fit_poisson_glm(X_fit_base, y, offset=None, max_iter=int(args.max_iter))
        elif mode == "free":
            beta = fit_poisson_glm(X_fit_harm, y, offset=None, max_iter=int(args.max_iter))
        elif mode == "prior":
            beta = fit_poisson_glm(
                X_fit_harm, y, offset=None, max_iter=int(args.max_iter), prior_prec_diag=prior_prec
            )
        else:
            raise ValueError(mode)
        return np.asarray(beta[1:4], dtype=float)

    out_b: dict[str, list[np.ndarray]] = {k: [] for k in ("baseline", "free", "prior")}
    for i in range(int(args.n_mocks)):
        try:
            g_full = hp.synfast(cl_inj, nside=nside, lmax=lmax, new=True, verbose=False)
        except TypeError:
            g_full = hp.synfast(cl_inj, nside=nside, lmax=lmax)
        g_full = np.asarray(g_full, dtype=float)
        g_full = g_full - float(np.mean(g_full[seen]))
        g_seen = g_full[seen]

        loglam = np.log(np.clip(mu_hat, 1e-12, np.inf)) + delta_dip + g_seen
        lam = np.exp(np.clip(loglam, -25.0, 25.0))
        y = rng.poisson(lam).astype(float)

        for mode in out_b:
            out_b[mode].append(fit_case(y, mode))

        if (i + 1) % 20 == 0 or (i + 1) == int(args.n_mocks):
            print(f"{i+1}/{int(args.n_mocks)} mocks")

    def summarize(vs: list[np.ndarray]) -> dict[str, Any]:
        bvec = np.asarray(vs, dtype=float)
        D = np.linalg.norm(bvec, axis=1)
        bmean = np.mean(bvec, axis=0)
        return {
            "n": int(bvec.shape[0]),
            "b_mean": [float(x) for x in bmean],
            "D_mean": float(np.mean(D)),
            "D_std": float(np.std(D)),
            "D_p16": float(np.percentile(D, 16)),
            "D_p50": float(np.percentile(D, 50)),
            "D_p84": float(np.percentile(D, 84)),
        }

    payload: dict[str, Any] = {
        "meta": {
            "catalog": str(args.catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": nside,
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_max": w1_max,
            "harmonic_lmax": lmax,
            "cl_json": str(args.cl_json),
            "true_cl_scale": float(args.true_cl_scale),
            "fit_prior_scale": float(args.fit_prior_scale),
            "prior_min_cl": cl_floor,
            "n_mocks": int(args.n_mocks),
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "dipole_amp_inj": float(args.dipole_amp),
            "dipole_axis_lb_inj": [inj_l, inj_b],
        },
        "inj": {"b_inj": [float(x) for x in b_inj], "D_inj": float(np.linalg.norm(b_inj))},
        "summary": {k: summarize(v) for k, v in out_b.items()},
    }

    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bins = 30
    for mode, color in (("baseline", "C0"), ("prior", "C1"), ("free", "C2")):
        bvec = np.asarray(out_b[mode], dtype=float)
        D = np.linalg.norm(bvec, axis=1)
        ax.hist(D, bins=bins, density=True, histtype="step", lw=2, color=color, label=mode)
    ax.axvline(float(np.linalg.norm(b_inj)), color="k", ls="--", lw=1.5, label="injected")
    ax.set_xlabel("Recovered dipole amplitude D")
    ax.set_ylabel("density")
    ax.set_title(
        f"Dipole injection check (W1_max={w1_max}; ell<={lmax}; injClx={args.true_cl_scale:g}; priorx={args.fit_prior_scale:g})"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out_fig = Path(str(args.out_fig))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_fig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
