#!/usr/bin/env python3
"""
Fit a quasar number-count dipole while marginalizing simple contaminant templates.

Goal (Step 2 of the "Gangster" validation path):
  Separate a true dipole-like modulation from known survey/systematics directions
  (dust/extinction, ecliptic scan strategy, coverage).

We do this in HEALPix-binned space using a *count-level* weighted least squares (WLS) model:

  N_pix ≈ A + B · n_hat + Σ_k c_k T_k

where n_hat is the (Galactic) unit-vector direction for each pixel, and T_k are template fields.

With Poisson counts Var(N_pix) ≈ N_pix, a convenient WLS choice is weight w ≈ 1/N_pix.
The dipole amplitude is then estimated as:

  D ≈ |B| / A

Outputs:
  - JSON summary with:
      * raw catalog dipole via the vector-sum estimator,
      * dipole-only WLS estimate,
      * dipole+templates WLS estimate,
      * templates-only "de-templated" reweighted-dipole estimate.
  - Optional diagnostic Mollweide PNG of:
      * delta counts map (N/mean - 1),
      * residual after subtracting the templates-only model.

Notes:
  - This is not a full, publication-grade systematics model.
  - It is a defensible next-step diagnostic for whether EBV/ecliptic/coverage can soak up the dipole.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import healpy as hp
import numpy as np
from astropy.table import Table

from secrest_utils import apply_baseline_cuts, compute_dipole


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def vec_to_lb(vec: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size != 3:
        raise ValueError("expected 3-vector")
    n = float(np.linalg.norm(v))
    if n == 0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def solve_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Weighted least squares using row-scaling (sqrt(w)).
    Returns (beta, cov_beta, chi2, dof).
    """
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    sw = np.sqrt(np.clip(w, 0.0, np.inf))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    chi2 = float(np.sum(w * resid * resid))
    dof = int(max(0, X.shape[0] - X.shape[1]))
    # Estimate a scale; for perfect Poisson weights this should be ~1.
    sigma2 = chi2 / dof if dof > 0 else float("nan")
    XtWX = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(XtWX) * sigma2
    return beta, cov, chi2, dof


@dataclass
class DipoleFit:
    amplitude: float
    amplitude_sigma: float
    l_deg: float
    b_deg: float
    chi2: float
    dof: int


def dipole_from_beta(beta: np.ndarray, cov: np.ndarray, idx0: int) -> DipoleFit:
    d = np.asarray(beta[idx0 : idx0 + 3], dtype=float)
    cov_d = np.asarray(cov[idx0 : idx0 + 3, idx0 : idx0 + 3], dtype=float)
    amp = float(np.linalg.norm(d))
    if amp > 0 and np.all(np.isfinite(cov_d)):
        u = d / amp
        amp_var = float(u @ cov_d @ u)
        amp_sig = float(math.sqrt(max(0.0, amp_var)))
    else:
        amp_sig = float("nan")
    l, b = vec_to_lb(d)
    return DipoleFit(amplitude=amp, amplitude_sigma=amp_sig, l_deg=l, b_deg=b, chi2=float("nan"), dof=0)


def dipole_from_counts_beta(beta: np.ndarray, cov: np.ndarray) -> DipoleFit:
    """Interpret beta from a count-level model N = A + B·n + ... as D = |B|/A."""
    A = float(beta[0])
    B = np.asarray(beta[1:4], dtype=float)
    amp = float(np.linalg.norm(B) / A) if A != 0.0 else float("nan")
    if A != 0.0 and np.all(np.isfinite(cov[:4, :4])):
        # delta-method variance for D = |B|/A
        covA = float(cov[0, 0])
        covB = np.asarray(cov[1:4, 1:4], dtype=float)
        covAB = np.asarray(cov[0, 1:4], dtype=float)
        normB = float(np.linalg.norm(B))
        if normB > 0:
            u = B / normB
            dD_dA = -normB / (A * A)
            dD_dB = u / A
            var = (dD_dA ** 2) * covA + float(dD_dB @ covB @ dD_dB) + 2.0 * float(dD_dA * (covAB @ dD_dB))
            amp_sig = float(math.sqrt(max(0.0, var)))
        else:
            amp_sig = float("nan")
    else:
        amp_sig = float("nan")
    l, b = vec_to_lb(B)
    return DipoleFit(amplitude=amp, amplitude_sigma=amp_sig, l_deg=l, b_deg=b, chi2=float("nan"), dof=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Secrest/CatWISE FITS (expects l,b,w1,w1cov,ebv,elat).")
    ap.add_argument("--outdir", default=None, help="Output directory.")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-min", type=float, default=None)
    ap.add_argument("--w1-max", type=float, default=16.4)
    ap.add_argument("--mask-file", type=str, default=None, help="Optional exclusion mask FITS (exclude_master_revised.fits).")
    ap.add_argument("--make-plots", action="store_true", help="Write diagnostic Mollweide plots.")
    args = ap.parse_args()

    outdir = Path(args.outdir or "outputs/quasar_template_fit")
    outdir.mkdir(parents=True, exist_ok=True)

    tbl = Table.read(args.catalog)

    # Optional exclusion mask (slow, but 292 regions only).
    mask = np.ones(len(tbl), dtype=bool)
    cuts: List[Dict] = []
    if args.mask_file:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        mask_tbl = Table.read(args.mask_file)
        ra_all = np.asarray(tbl["ra"], float)
        dec_all = np.asarray(tbl["dec"], float)
        cat_coords = SkyCoord(ra=ra_all * u.deg, dec=dec_all * u.deg, frame="icrs")

        exclude_mask = np.zeros(len(tbl), dtype=bool)
        for row in mask_tbl:
            if not bool(row["use"]):
                center = SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg, frame="icrs")
                sep = cat_coords.separation(center).deg
                exclude_mask |= sep < float(row["radius"])

        mask &= ~exclude_mask
        cuts.append({"name": "exclusion_mask", "N_excluded": int(exclude_mask.sum()), "N_after": int(mask.sum())})

    # Apply baseline cuts (including w1_max).
    mask, base_cuts = apply_baseline_cuts(
        tbl,
        b_cut=float(args.b_cut),
        w1cov_min=float(args.w1cov_min),
        w1_max=float(args.w1_max),
        w1_min=args.w1_min,
        existing_mask=mask,
    )
    cuts.extend(base_cuts)

    l = np.asarray(tbl["l"], float)[mask]
    b = np.asarray(tbl["b"], float)[mask]
    ebv = np.asarray(tbl["ebv"], float)[mask]
    elat = np.asarray(tbl["elat"], float)[mask]
    w1cov = np.asarray(tbl["w1cov"], float)[mask]

    # Raw (catalog-level) dipole for reference.
    D_raw, l_raw, b_raw, _ = compute_dipole(l, b)

    # HEALPix binning
    nside = int(args.nside)
    npix = hp.nside2npix(nside)
    theta = np.deg2rad(90.0 - b)
    phi = np.deg2rad(l % 360.0)
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    Np = np.bincount(pix, minlength=npix).astype(float)
    valid = Np > 0
    n_valid = int(valid.sum())

    # Mean unit-vector per pixel (use object locations, not pixel centers).
    nvec = lb_to_unitvec(l, b)
    sx = np.bincount(pix, weights=nvec[:, 0], minlength=npix)
    sy = np.bincount(pix, weights=nvec[:, 1], minlength=npix)
    sz = np.bincount(pix, weights=nvec[:, 2], minlength=npix)
    mx = np.zeros(npix); my = np.zeros(npix); mz = np.zeros(npix)
    mx[valid] = sx[valid] / Np[valid]
    my[valid] = sy[valid] / Np[valid]
    mz[valid] = sz[valid] / Np[valid]

    # Templates: per-pixel means
    def mean_by_pix(values: np.ndarray) -> np.ndarray:
        s = np.bincount(pix, weights=values, minlength=npix)
        out = np.zeros(npix, dtype=float)
        out[valid] = s[valid] / Np[valid]
        return out

    t_ebv = mean_by_pix(ebv)
    t_abs_elat = np.abs(mean_by_pix(elat))
    t_w1cov = mean_by_pix(w1cov)

    # Response for plotting (fractional count fluctuation)
    mean_N = float(Np[valid].mean())
    delta_map = np.full(npix, hp.UNSEEN, dtype=float)
    delta_map[valid] = Np[valid] / mean_N - 1.0

    # Standardize templates (on valid pixels)
    def zscore(t: np.ndarray) -> np.ndarray:
        tv = t[valid]
        m = float(tv.mean())
        s = float(tv.std(ddof=0))
        if s == 0:
            return np.zeros_like(t)
        out = (t - m) / s
        return out

    z_ebv = zscore(t_ebv)
    z_abs_elat = zscore(t_abs_elat)
    z_w1cov = zscore(t_w1cov)

    # Build design matrices on valid pixels only.
    # Count-level regression: y = Np, weights w ~ 1/Np.
    y = Np[valid]
    w = 1.0 / np.clip(y, 1.0, np.inf)
    X_dip = np.column_stack([np.ones(n_valid), mx[valid], my[valid], mz[valid]])
    X_tmp = np.column_stack([np.ones(n_valid), z_ebv[valid], z_abs_elat[valid], z_w1cov[valid]])
    X_both = np.column_stack(
        [np.ones(n_valid), mx[valid], my[valid], mz[valid], z_ebv[valid], z_abs_elat[valid], z_w1cov[valid]]
    )

    # Solve models
    beta_d, cov_d, chi2_d, dof_d = solve_wls(X_dip, y, w)
    dip_d = dipole_from_counts_beta(beta_d, cov_d)
    dip_d.chi2, dip_d.dof = chi2_d, dof_d

    beta_t, cov_t, chi2_t, dof_t = solve_wls(X_tmp, y, w)
    # No dipole in this model.

    beta_b, cov_b, chi2_b, dof_b = solve_wls(X_both, y, w)
    dip_b = dipole_from_counts_beta(beta_b, cov_b)
    dip_b.chi2, dip_b.dof = chi2_b, dof_b

    # Templates-only residual map (counts) for plotting.
    yhat_t = X_tmp @ beta_t
    resid_counts = y - yhat_t
    resid_map = np.full(npix, hp.UNSEEN, dtype=float)
    resid_map[valid] = resid_counts / mean_N  # normalize for plotting scale

    # De-templated reweighting: weight objects inversely by templates-only predicted counts per pixel.
    # This keeps weights positive and yields a dipole estimate comparable to compute_dipole().
    pred_counts_full = np.zeros(npix, dtype=float)
    pred_counts_full[valid] = np.clip(yhat_t, 1e-6, np.inf)
    pred_mean = float(pred_counts_full[valid].mean())
    obj_weights = pred_mean / pred_counts_full[pix]  # shape (n_obj,)

    D_w, l_w, b_w, _ = compute_dipole(l, b, weights=obj_weights)

    result = {
        "inputs": {
            "catalog": args.catalog,
            "mask_file": args.mask_file,
            "nside": nside,
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_min": args.w1_min,
            "w1_max": float(args.w1_max),
        },
        "cuts": cuts,
        "counts": {"N_objects": int(len(l)), "nside": nside, "n_valid_pix": n_valid, "mean_N_per_pix": mean_N},
        "raw_catalog_dipole": {"amplitude": float(D_raw), "l_deg": float(l_raw), "b_deg": float(b_raw)},
        "regression": {
            "dipole_only": asdict(dip_d),
            "templates_only": {
                "chi2": chi2_t,
                "dof": dof_t,
                "beta": beta_t.tolist(),
            },
            "dipole_plus_templates": asdict(dip_b),
            "detemplated_reweighted_dipole": {"amplitude": float(D_w), "l_deg": float(l_w), "b_deg": float(b_w)},
            "templates": ["ebv(z)", "|elat|(z)", "w1cov(z)"],
        },
    }

    with open(outdir / "template_fit_summary.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    if args.make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        hp.mollview(delta_map, title="CatWISE counts: delta (binned)", sub=(1, 2, 1), cmap="coolwarm", min=-0.3, max=0.3)
        hp.mollview(resid_map, title="Residual after (EBV,|elat|,w1cov)", sub=(1, 2, 2), cmap="coolwarm", min=-0.3, max=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "template_fit_maps.png", dpi=200)
        plt.close()

    print(json.dumps(result["regression"], indent=2))
    print(f"Wrote: {outdir / 'template_fit_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
