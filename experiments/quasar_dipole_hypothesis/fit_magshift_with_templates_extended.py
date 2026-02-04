#!/usr/bin/env python3
"""
Extended version of fit_magshift_with_templates.py with richer systematics templates.

Purpose:
  The earlier "delta_m fit" demonstrates that a tiny dipolar magnitude modulation can
  reduce the CatWISE/Secrest number-count dipole, especially after marginalizing a few
  nuisance templates (EBV, |elat|, w1cov).

  This script pushes the *attribution* question:
    "Does the required delta_m survive once we include a more realistic set of scan/dust templates?"

Template sets:
  - basic:
      EBV, |elat|, w1cov
  - ecliptic_harmonics:
      basic + sin(elon), cos(elon), sin(2elon), cos(2elon), elat (signed)
  - poly:
      ecliptic_harmonics + EBV^2, w1cov^2, log(w1cov)
  - full:
      poly + Tb, alpha, logS   (these are *catalog columns*; interpret with care)

Outputs:
  - magshift_with_templates_extended.json
  - magshift_with_templates_extended.png
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


def solve_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
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
    sigma2 = chi2 / dof if dof > 0 else float("nan")
    XtWX = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(XtWX) * sigma2
    return beta, cov, chi2, dof


def dipole_from_unit(unit: np.ndarray, sel: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    sel = np.asarray(sel, dtype=bool)
    n = int(sel.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), np.array([np.nan, np.nan, np.nan])
    sum_vec = unit[sel].sum(axis=0)
    dip_vec = 3.0 * sum_vec / float(n)
    amp = float(np.linalg.norm(dip_vec))
    l, b = vec_to_lb(dip_vec)
    return amp, l, b, dip_vec


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x[valid])) if np.any(valid) else 0.0
    s = float(np.std(x[valid])) if np.any(valid) else 1.0
    if s == 0.0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


def load_axis_from_secrest_json(path: str) -> Tuple[float, float]:
    with open(path, "r") as f:
        d = json.load(f)
    return float(d["dipole"]["l_deg"]), float(d["dipole"]["b_deg"])


@dataclass
class FitRow:
    delta_m: float
    N: int
    raw_amp: float
    raw_l: float
    raw_b: float
    fit_amp: float
    fit_amp_sigma: float
    fit_l: float
    fit_b: float
    proj_axis: float
    chi2: float
    dof: int


def make_template_matrix(
    name: str,
    ebv: np.ndarray,
    sin_elon: np.ndarray,
    cos_elon: np.ndarray,
    sin_2elon: np.ndarray,
    cos_2elon: np.ndarray,
    elat: np.ndarray,
    w1cov: np.ndarray,
    Tb: np.ndarray,
    alpha: np.ndarray,
    logS: np.ndarray,
    valid: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """
    Return (names, Z) where Z is [n_pix, n_templates] with z-scored template columns.
    """

    names: List[str] = []
    cols: List[np.ndarray] = []

    # Always available.
    names += ["ebv", "abs_elat", "w1cov"]
    cols += [ebv, np.abs(elat), w1cov]

    if name in ("ecliptic_harmonics", "poly", "full"):
        names += ["sin_elon", "cos_elon", "sin_2elon", "cos_2elon", "elat"]
        cols += [sin_elon, cos_elon, sin_2elon, cos_2elon, elat]

    if name in ("poly", "full"):
        names += ["ebv2", "w1cov2", "log_w1cov"]
        cols += [ebv * ebv, w1cov * w1cov, np.log(np.clip(w1cov, 1.0, np.inf))]

    if name == "full":
        names += ["Tb", "alpha", "logS"]
        cols += [Tb, alpha, logS]

    Z = np.column_stack([zscore(c, valid) for c in cols])
    return names, Z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Secrest/CatWISE FITS (expects l,b,w1,w1cov,ebv,elon,elat).")
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-max", type=float, default=16.4)

    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--delta-m-max", type=float, default=0.03)
    ap.add_argument("--grid-n", type=int, default=61)
    ap.add_argument("--sign", choices=["+", "-"], default="-", help="Sign convention for W1_corr = W1 - sign*delta_m*cos.")

    ap.add_argument("--template-set", choices=["basic", "ecliptic_harmonics", "poly", "full"], default="basic")

    ap.add_argument("--axis-from", choices=["secrest", "custom"], default="secrest")
    ap.add_argument("--axis-l", type=float, default=None)
    ap.add_argument("--axis-b", type=float, default=None)
    ap.add_argument(
        "--secrest-json",
        default="REPORTS/Q_D_RES/secrest_reproduction_dipole.json",
        help="Axis source when --axis-from=secrest.",
    )
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir or "outputs/quasar_magshift_with_templates_extended")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.axis_from == "custom":
        if args.axis_l is None or args.axis_b is None:
            raise SystemExit("--axis-l/--axis-b required for --axis-from=custom")
        axis_l, axis_b = float(args.axis_l), float(args.axis_b)
    else:
        axis_l, axis_b = load_axis_from_secrest_json(args.secrest_json)

    # Heavy imports late.
    import healpy as hp
    from astropy.table import Table

    tab = Table.read(args.catalog, memmap=True)
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

    l = l[base]
    b = b[base]
    w1 = w1[base]
    w1cov = w1cov[base]
    ebv = ebv[base]
    elon = elon[base]
    elat = elat[base]
    Tb = Tb[base]
    alpha = alpha[base]
    logS = logS[base]

    unit = lb_to_unitvec(l, b)
    axis_unit = unitvec_from_lb(axis_l, axis_b)
    cos_axis = unit @ axis_unit

    # Use circular-safe ecliptic-longitude templates (mean(sin) / mean(cos) at pixel level),
    # not sin(mean(elon)) which is wrong near the 0/360 wrap.
    elon_rad = np.deg2rad(elon % 360.0)
    sin_elon = np.sin(elon_rad)
    cos_elon = np.cos(elon_rad)
    sin_2elon = np.sin(2.0 * elon_rad)
    cos_2elon = np.cos(2.0 * elon_rad)

    # HEALPix pixelization.
    nside = int(args.nside)
    theta = np.deg2rad(90.0 - b)
    phi = np.deg2rad(l % 360.0)
    pix = hp.ang2pix(nside, theta, phi, nest=True)
    npix = hp.nside2npix(nside)

    # Pixel unit vectors for the dipole model.
    pix_unit = np.zeros((npix, 3), dtype=float)
    for p in range(npix):
        th, ph = hp.pix2ang(nside, p, nest=True)
        # Convert to Cartesian (Galactic coordinates).
        x = math.sin(th) * math.cos(ph)
        y = math.sin(th) * math.sin(ph)
        z = math.cos(th)
        pix_unit[p] = (x, y, z)

    sign = 1.0 if args.sign == "+" else -1.0
    dm_grid = np.linspace(0.0, float(args.delta_m_max), int(args.grid_n))

    rows: List[FitRow] = []
    for dm in dm_grid:
        w1_corr = w1 - sign * float(dm) * cos_axis
        sel = w1_corr <= float(args.w1_max)
        N = int(sel.sum())

        raw_amp, raw_l, raw_b, raw_vec = dipole_from_unit(unit, sel)

        # Pixel counts for selected sample.
        Np = np.bincount(pix[sel], minlength=npix).astype(float)
        pix_valid = Np > 0

        # Pixel-mean templates for selected sample.
        def pix_mean(val: np.ndarray) -> np.ndarray:
            s = np.bincount(pix[sel], weights=val[sel], minlength=npix)
            m = np.zeros(npix, dtype=float)
            m[pix_valid] = s[pix_valid] / Np[pix_valid]
            return m

        t_ebv = pix_mean(ebv)
        t_sin_elon = pix_mean(sin_elon)
        t_cos_elon = pix_mean(cos_elon)
        t_sin_2elon = pix_mean(sin_2elon)
        t_cos_2elon = pix_mean(cos_2elon)
        t_elat = pix_mean(elat)
        t_w1cov = pix_mean(w1cov)
        t_Tb = pix_mean(Tb)
        t_alpha = pix_mean(alpha)
        t_logS = pix_mean(logS)

        tpl_names, Z = make_template_matrix(
            args.template_set,
            t_ebv,
            t_sin_elon,
            t_cos_elon,
            t_sin_2elon,
            t_cos_2elon,
            t_elat,
            t_w1cov,
            t_Tb,
            t_alpha,
            t_logS,
            pix_valid,
        )

        y = Np[pix_valid]
        w = 1.0 / np.clip(y, 1.0, np.inf)

        X = np.column_stack([np.ones_like(y), pix_unit[pix_valid, 0], pix_unit[pix_valid, 1], pix_unit[pix_valid, 2], Z[pix_valid]])
        beta, cov, chi2, dof = solve_wls(X, y, w)

        A = float(beta[0])
        B = np.asarray(beta[1:4], dtype=float)
        dvec = B / A if A != 0.0 else np.array([np.nan, np.nan, np.nan])
        fit_amp = float(np.linalg.norm(dvec))
        fit_l, fit_b = vec_to_lb(dvec)

        # Delta-method sigma for |B|/A.
        fit_sigma = float("nan")
        if A != 0.0 and np.all(np.isfinite(cov[:4, :4])):
            covA = float(cov[0, 0])
            covB = np.asarray(cov[1:4, 1:4], dtype=float)
            covAB = np.asarray(cov[0, 1:4], dtype=float)
            normB = float(np.linalg.norm(B))
            if normB > 0:
                u = B / normB
                dD_dA = -normB / (A * A)
                dD_dB = u / A
                var = (dD_dA**2) * covA + float(dD_dB @ covB @ dD_dB) + 2.0 * float(dD_dA * (covAB @ dD_dB))
                fit_sigma = float(math.sqrt(max(0.0, var)))

        proj_axis = float(np.dot(dvec, axis_unit))

        rows.append(
            FitRow(
                delta_m=float(dm),
                N=N,
                raw_amp=float(raw_amp),
                raw_l=float(raw_l),
                raw_b=float(raw_b),
                fit_amp=float(fit_amp),
                fit_amp_sigma=float(fit_sigma),
                fit_l=float(fit_l),
                fit_b=float(fit_b),
                proj_axis=proj_axis,
                chi2=float(chi2),
                dof=int(dof),
            )
        )

    best = min(rows, key=lambda r: (r.fit_amp if np.isfinite(r.fit_amp) else 1e9))

    out = {
        "inputs": {
            "catalog": args.catalog,
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_max": float(args.w1_max),
            "nside": int(args.nside),
            "delta_m_max": float(args.delta_m_max),
            "grid_n": int(args.grid_n),
            "sign": args.sign,
            "template_set": args.template_set,
            "axis_l_deg": float(axis_l),
            "axis_b_deg": float(axis_b),
            "secrest_json": args.secrest_json,
        },
        "templates": {"names": tpl_names, "count": len(tpl_names)},
        "best": {
            "delta_m": best.delta_m,
            "N": best.N,
            "raw_amp": best.raw_amp,
            "raw_l_deg": best.raw_l,
            "raw_b_deg": best.raw_b,
            "fit_amp": best.fit_amp,
            "fit_amp_sigma": best.fit_amp_sigma,
            "fit_l_deg": best.fit_l,
            "fit_b_deg": best.fit_b,
            "proj_axis": best.proj_axis,
            "chi2": best.chi2,
            "dof": best.dof,
        },
        "grid": [r.__dict__ for r in rows],
        "notes": "fit_amp is the residual dipole amplitude after marginalizing the specified templates in a pixel-count WLS model.",
    }

    with open(outdir / "magshift_with_templates_extended.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    if args.make_plots:
        import matplotlib.pyplot as plt

        xs = [r.delta_m for r in rows]
        ys = [r.fit_amp for r in rows]
        ys_raw = [r.raw_amp for r in rows]
        yerr = [r.fit_amp_sigma for r in rows]

        plt.figure(figsize=(7.5, 4.2), dpi=200)
        plt.plot(xs, ys_raw, color="#999999", lw=1.5, label="raw (vector-sum) dipole")
        plt.errorbar(xs, ys, yerr=yerr, fmt="o-", lw=2, ms=3, capsize=2, label="after templates (WLS)")
        plt.axvline(best.delta_m, color="k", ls="--", alpha=0.7, label=f"best dm={best.delta_m:.4f}")
        plt.xlabel("delta_m (mag)")
        plt.ylabel("dipole amplitude")
        plt.title(f"delta_m scan with templates ({args.template_set})")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / "magshift_with_templates_extended.png")
        plt.close()

    print(json.dumps(out["best"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
