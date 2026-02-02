#!/usr/bin/env python3
"""
RvMP Fig.5-style CatWISE dipole scan using a Poisson maximum-likelihood model.

This is the "likelihood" counterpart to `scripts/reproduce_rvmp_fig5_catwise.py`,
which reproduces the Secrest+22 weighted linear dipole estimator.

Model
-----
For each unmasked HEALPix pixel p:

  N_p ~ Poisson(mu_p)
  log mu_p = beta0 + b · n_p + Σ_k c_k T_{k,p} + offset_p

where:
  - n_p is the unit vector of the pixel center in Galactic coordinates,
  - b is a 3-vector; for small dipoles D ≈ |b|,
  - T_{k,p} are nuisance templates (ecliptic latitude trend, dust proxy, depth proxy),
  - offset_p is an optional fixed offset (e.g. log depth).

This matches the RvMP wording "marginalize over the ecliptic trend" more literally:
we include the trend as nuisance parameters in the likelihood and (approximately)
marginalize via a Gaussian approximation to the fitted parameter covariance.

Masking footprint
-----------------
Matches Secrest+22:
  - mask_zeros on the full W1cov>=80 map (zeros + neighbours),
  - exclude_master_revised.fits discs (use==True),
  - Galactic plane cut |b| < b_cut (pixel centers).

Optional injection
------------------
Supports an injected dipolar faint-limit modulation (pure selection effect):
  W1_eff = W1 - delta_m * cos(theta_axis)
and then apply W1_eff < W1_max (and optional W1_eff > W1_min).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def skyarea_deg2() -> float:
    return float(4.0 * math.pi * (180.0 / math.pi) ** 2)


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def vec_to_lb(vec: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x[valid])) if np.any(valid) else 0.0
    s = float(np.std(x[valid])) if np.any(valid) else 1.0
    if s == 0.0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


@dataclass(frozen=True)
class SecrestMask:
    mask: np.ndarray  # True = masked
    seen: np.ndarray  # True = unmasked


def build_secrest_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> SecrestMask:
    """Implements SkyMap.mask_zeros + fits2mask + galactic plane cut."""
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    # mask_zeros(tbl) on the W1cov>=80 full sample
    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        indices = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            indices[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        # Match Secrest behaviour (includes -1 neighbour indexing last pixel).
        mask[indices] = True

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
                disc = hp.query_disc(nside=int(nside), vec=vec, radius=float(rad), inclusive=True, nest=False)
                mask[disc] = True

    # galactic plane cut
    lon_pix, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return SecrestMask(mask=mask, seen=~mask)


def fit_poisson_glm(
    X: np.ndarray, y: np.ndarray, *, offset: np.ndarray | None, max_iter: int = 300
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Poisson GLM (log link) via L-BFGS.
    Returns (beta, cov_beta_approx) where cov is Fisher^{-1} if invertible.
    """
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    off = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)

    mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
    beta0 = np.zeros(X.shape[1], dtype=float)
    beta0[0] = math.log(mu0)

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = off + X @ beta
        eta = np.clip(eta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        return nll, np.asarray(grad, dtype=float)

    res = minimize(
        lambda b: fun_and_grad(b)[0],
        beta0,
        jac=lambda b: fun_and_grad(b)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    beta = np.asarray(res.x, dtype=float)

    # Fisher / covariance approximation: (X^T diag(mu) X)^{-1}
    try:
        eta = np.clip(off + X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        fisher = X.T @ (mu[:, None] * X)
        cov = np.linalg.inv(fisher)
    except Exception:  # noqa: BLE001
        cov = None
    return beta, cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--mask-catalog",
        default=None,
        help=(
            "Optional separate catalog used ONLY to build the footprint mask and any per-pixel template means. "
            "Use this when the analysis catalog is a filtered/subsampled file (e.g. NVSS-removed)."
        ),
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1-min", type=float, default=None, help="Optional bright cut (keep only W1 > w1_min).")
    ap.add_argument("--w1-grid", default="15.5,16.6,0.05", help="start,stop,step (inclusive end).")
    ap.add_argument("--max-iter", type=int, default=300)

    ap.add_argument("--eclip-template", choices=["none", "abs_elat", "abs_sin_elat"], default="abs_elat")
    ap.add_argument("--dust-template", choices=["none", "ebv_mean"], default="none")
    ap.add_argument("--depth-mode", choices=["none", "w1cov_covariate", "w1cov_offset"], default="none")

    ap.add_argument("--mc-draws", type=int, default=400, help="Approx marginalization draws from N(beta,cov).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--make-plot", action="store_true")

    # Injection
    ap.add_argument("--inject-delta-m-mag", type=float, default=0.0)
    ap.add_argument("--inject-axis", default="cmb", help="'cmb' or 'l,b' in degrees for injection axis.")

    args = ap.parse_args()

    import healpy as hp
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    outdir = Path(args.outdir or f"outputs/catwise_rvmp_fig5_poisson_glm_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.catalog, memmap=True) as hdul:
        data = hdul[1].data
        w1 = np.asarray(data["w1"], dtype=float)
        w1cov = np.asarray(data["w1cov"], dtype=float)
        l = np.asarray(data["l"], dtype=float)
        b = np.asarray(data["b"], dtype=float)
        ebv = None
        if args.dust_template != "none":
            if "ebv" not in data.names:
                raise SystemExit(f"catalog missing ebv but --dust-template={args.dust_template}")
            ebv = np.asarray(data["ebv"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    if ebv is not None:
        base &= np.isfinite(ebv)
    base &= w1cov >= float(args.w1cov_min)

    theta = np.deg2rad(90.0 - b[base])
    phi = np.deg2rad(l[base])
    ipix_base = hp.ang2pix(int(args.nside), theta, phi, nest=False)
    ipix_mask_base = np.asarray(ipix_base, dtype=np.int64)

    # Optional injection effective faint-limit modulation.
    inject_delta_m = float(args.inject_delta_m_mag)
    if args.inject_axis.strip().lower() == "cmb":
        axis_l, axis_b = 264.021, 48.253
    else:
        parts = args.inject_axis.split(",")
        if len(parts) != 2:
            raise SystemExit("--inject-axis must be 'cmb' or 'l,b'")
        axis_l, axis_b = float(parts[0]), float(parts[1])

    if inject_delta_m != 0.0:
        n_src = lb_to_unitvec(l[base], b[base])
        n_axis = lb_to_unitvec(np.array([axis_l]), np.array([axis_b]))[0]
        cos_theta = n_src @ n_axis
        w1_eff = w1[base] - inject_delta_m * cos_theta
    else:
        w1_eff = w1[base].copy()

    # Apply bright-side cut on the same effective magnitude.
    if args.w1_min is not None:
        keep = w1_eff > float(args.w1_min)
        w1_eff = w1_eff[keep]
        ipix_eff = ipix_base[keep]
        ebv_eff = None if ebv is None else ebv[base][keep]
        w1cov_eff = w1cov[base][keep]
        l_eff = l[base][keep]
        b_eff = b[base][keep]
    else:
        ipix_eff = ipix_base
        ebv_eff = None if ebv is None else ebv[base]
        w1cov_eff = w1cov[base]
        l_eff = l[base]
        b_eff = b[base]

    # Sort by W1_eff for incremental cumulative updates.
    order = np.argsort(w1_eff)
    w1_eff_sorted = w1_eff[order]
    ipix_sorted = np.asarray(ipix_eff, dtype=np.int64)[order]

    # Parse grid.
    w1_start, w1_stop, w1_step = (float(x) for x in args.w1_grid.split(","))
    n_steps = int(round((w1_stop - w1_start) / w1_step)) + 1
    cuts = [w1_start + i * w1_step for i in range(n_steps)]

    # Optionally, build the footprint (mask_zeros) and template means from a separate catalog.
    if args.mask_catalog is not None:
        with fits.open(args.mask_catalog, memmap=True) as hdul:
            dm = hdul[1].data
            w1cov_m = np.asarray(dm["w1cov"], dtype=float)
            l_m = np.asarray(dm["l"], dtype=float)
            b_m = np.asarray(dm["b"], dtype=float)
            base_m = np.isfinite(w1cov_m) & np.isfinite(l_m) & np.isfinite(b_m)
            base_m &= w1cov_m >= float(args.w1cov_min)
            theta_m = np.deg2rad(90.0 - b_m[base_m])
            phi_m = np.deg2rad(l_m[base_m])
            ipix_mask_base = hp.ang2pix(int(args.nside), theta_m, phi_m, nest=False).astype(np.int64)

            if args.dust_template == "ebv_mean":
                if "ebv" not in dm.names:
                    raise SystemExit(f"mask-catalog missing ebv but --dust-template={args.dust_template}")
                ebv_m_base = np.asarray(dm["ebv"], dtype=float)[base_m]
            else:
                ebv_m_base = None

            w1cov_m_base = w1cov_m[base_m]
    else:
        ebv_m_base = None if ebv is None else ebv[base]
        w1cov_m_base = w1cov[base]

    npix = hp.nside2npix(int(args.nside))

    # Mask + pixel geometry.
    secrest_mask = build_secrest_mask(
        nside=int(args.nside),
        ipix_base=ipix_mask_base,
        exclude_mask_fits=args.exclude_mask_fits,
        b_cut_deg=float(args.b_cut),
    )
    mask = secrest_mask.mask
    seen = secrest_mask.seen

    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)  # galactic l,b
    pix_unit = lb_to_unitvec(lon_pix, lat_pix)

    # Pixel-center ecliptic lat (templates).
    sc_pix = SkyCoord(lon_pix * u.deg, lat_pix * u.deg, frame="galactic")
    elat_deg = sc_pix.barycentricmeanecliptic.lat.deg.astype(float)
    abs_elat = np.abs(elat_deg)
    abs_sin_elat = np.abs(np.sin(np.deg2rad(elat_deg)))

    # Base per-pixel means for EBV and W1COV.
    cnt_base = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), minlength=npix).astype(float)
    sum_w1cov = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), weights=np.asarray(w1cov_m_base, dtype=float), minlength=npix).astype(float)
    w1cov_mean = np.divide(sum_w1cov, cnt_base, out=np.zeros_like(sum_w1cov), where=cnt_base != 0.0)
    if ebv_m_base is None:
        ebv_mean = np.zeros(npix, dtype=float)
    else:
        sum_ebv = np.bincount(np.asarray(ipix_mask_base, dtype=np.int64), weights=np.asarray(ebv_m_base, dtype=float), minlength=npix).astype(float)
        ebv_mean = np.divide(sum_ebv, cnt_base, out=np.zeros_like(sum_ebv), where=cnt_base != 0.0)

    rng = np.random.default_rng(int(args.seed))
    counts = np.zeros(npix, dtype=np.int64)
    cursor = 0

    rows: list[dict[str, Any]] = []
    for w1_cut in cuts:
        nxt = int(np.searchsorted(w1_eff_sorted, float(w1_cut), side="left"))
        if nxt > cursor:
            delta = ipix_sorted[cursor:nxt]
            counts += np.bincount(delta, minlength=npix).astype(np.int64)
            cursor = nxt

        y = counts[seen].astype(float)
        n_seen = pix_unit[seen]

        # Templates.
        templates: list[np.ndarray] = []
        if args.eclip_template == "abs_elat":
            templates.append(zscore(abs_elat, seen)[seen])
        elif args.eclip_template == "abs_sin_elat":
            templates.append(zscore(abs_sin_elat, seen)[seen])

        if args.dust_template == "ebv_mean":
            templates.append(zscore(ebv_mean, seen)[seen])

        offset = None
        if args.depth_mode == "w1cov_covariate":
            templates.append(zscore(np.log(np.clip(w1cov_mean, 1.0, np.inf)), seen)[seen])
        elif args.depth_mode == "w1cov_offset":
            # An offset has an implicit fixed coefficient of +1, so *do not z-score* it.
            # Instead normalize by a typical value so the offset is O(1) and centered.
            logw = np.log(np.clip(w1cov_mean, 1.0, np.inf))
            ref = float(np.median(logw[seen]))
            offset = (logw - ref)[seen]

        cols = [np.ones_like(y), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2]]
        cols.extend(templates)
        X = np.column_stack(cols)

        beta, cov = fit_poisson_glm(X, y, offset=offset, max_iter=int(args.max_iter))
        bvec = np.asarray(beta[1:4], dtype=float)
        D_hat = float(np.linalg.norm(bvec))
        l_hat, b_hat = vec_to_lb(bvec)

        # Approx marginalization by sampling beta ~ N(beta_hat, cov_hat).
        if cov is None or not np.all(np.isfinite(cov)):
            draws = None
        else:
            draws = rng.multivariate_normal(beta, cov, size=int(args.mc_draws))

        if draws is None:
            D16 = D50 = D84 = float("nan")
            l16 = l50 = l84 = float("nan")
            b16 = b50 = b84 = float("nan")
        else:
            b_draw = draws[:, 1:4]
            D_draw = np.linalg.norm(b_draw, axis=1)
            # Direction from the b-vector; undefined when b~0, but that's not our regime.
            lb = np.array([vec_to_lb(v) for v in b_draw])
            l_draw = lb[:, 0]
            b_draw_lat = lb[:, 1]

            def pct(a: np.ndarray, q: float) -> float:
                return float(np.nanpercentile(a, q))

            D16, D50, D84 = pct(D_draw, 16), pct(D_draw, 50), pct(D_draw, 84)
            l16, l50, l84 = pct(l_draw, 16), pct(l_draw, 50), pct(l_draw, 84)
            b16, b50, b84 = pct(b_draw_lat, 16), pct(b_draw_lat, 50), pct(b_draw_lat, 84)

        rows.append(
            {
                "w1_cut": float(w1_cut),
                "N_total": int(counts.sum()),
                "N_seen": int(np.sum(counts[seen])),
                "dipole": {
                    "D_hat": D_hat,
                    "l_hat_deg": l_hat,
                    "b_hat_deg": b_hat,
                    "D_p16": D16,
                    "D_p50": D50,
                    "D_p84": D84,
                    "l_p16": l16,
                    "l_p50": l50,
                    "l_p84": l84,
                    "b_p16": b16,
                    "b_p50": b50,
                    "b_p84": b84,
                },
                "beta_hat": [float(x) for x in beta],
            }
        )

    payload = {
        "meta": {
            "catalog": str(args.catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": int(args.nside),
            "w1cov_min": float(args.w1cov_min),
            "b_cut": float(args.b_cut),
            "w1_min": None if args.w1_min is None else float(args.w1_min),
            "w1_grid": args.w1_grid,
            "eclip_template": args.eclip_template,
            "dust_template": args.dust_template,
            "depth_mode": args.depth_mode,
            "mc_draws": int(args.mc_draws),
            "seed": int(args.seed),
            "inject_delta_m_mag": float(args.inject_delta_m_mag),
            "inject_axis_lb": [axis_l, axis_b],
        },
        "rows": rows,
    }
    (outdir / "rvmp_fig5_poisson_glm.json").write_text(json.dumps(payload, indent=2))

    if args.make_plot:
        import matplotlib.pyplot as plt

        w1c = np.array([r["w1_cut"] for r in rows], dtype=float)
        D50 = np.array([r["dipole"]["D_p50"] for r in rows], dtype=float)
        D16 = np.array([r["dipole"]["D_p16"] for r in rows], dtype=float)
        D84 = np.array([r["dipole"]["D_p84"] for r in rows], dtype=float)
        l50 = np.array([r["dipole"]["l_p50"] for r in rows], dtype=float)
        l16 = np.array([r["dipole"]["l_p16"] for r in rows], dtype=float)
        l84 = np.array([r["dipole"]["l_p84"] for r in rows], dtype=float)
        b50 = np.array([r["dipole"]["b_p50"] for r in rows], dtype=float)
        b16 = np.array([r["dipole"]["b_p16"] for r in rows], dtype=float)
        b84 = np.array([r["dipole"]["b_p84"] for r in rows], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)
        ax = axes[0]
        ax.fill_between(w1c, D16, D84, alpha=0.25, color="C0", lw=0)
        ax.plot(w1c, D50, color="C0")
        ax.set_ylabel("D ≈ |b| (Poisson GLM)")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.fill_between(w1c, l16, l84, alpha=0.25, color="C1", lw=0)
        ax.plot(w1c, l50, color="C1")
        ax.axhline(264.021, color="k", ls="--", lw=1)
        ax.set_ylabel("l [deg]")
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.fill_between(w1c, b16, b84, alpha=0.25, color="C2", lw=0)
        ax.plot(w1c, b50, color="C2")
        ax.axhline(48.253, color="k", ls="--", lw=1)
        ax.set_ylabel("b [deg]")
        ax.set_xlabel("W1_max (effective, if injected)")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(outdir / "rvmp_fig5_poisson_glm.png", dpi=200)
        plt.close(fig)

    D_last = rows[-1]["dipole"]["D_p50"]
    l_last = rows[-1]["dipole"]["l_p50"]
    b_last = rows[-1]["dipole"]["b_p50"]
    print(f"w1_cut_max={cuts[-1]:.3f}: D~{D_last:.4g}, (l,b)~({l_last:.1f},{b_last:+.1f})")
    print(f"Wrote: {outdir}/rvmp_fig5_poisson_glm.json")
    if args.make_plot:
        print(f"Wrote: {outdir}/rvmp_fig5_poisson_glm.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
