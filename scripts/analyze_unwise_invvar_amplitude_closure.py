#!/usr/bin/env python3
"""
Amplitude-closure diagnostic using an imaging-derived unWISE W1 invvar depth proxy.

Core idea
---------
If the dominant dipole amplitude in a flux-limited catalog is induced by a spatially varying
effective limiting magnitude m_lim(n), then (to first order):

  δN/N ≈ α_edge(m_max) * δm_lim(n),

where α_edge ≡ d ln N(<m) / dm evaluated at the faint cut m_max.

This script builds a *map-level* proxy for δm_lim(n) from unWISE W1 inverse-variance maps:

  m_lim ∝ + 1.25 * log10(invvar)   (since σ ∝ 1/sqrt(invvar))

So the dipole of the invvar-derived δm map predicts a selection-driven dipole vector that should
scale with α_edge across W1_max.

Deliverables
------------
Given:
  - an invvar HEALPix map (Galactic; RING), and
  - a Poisson-GLM scan JSON for the CatWISE sample,

we compute:
  - invvar-derived δm dipole vector (mag),
  - predicted selection dipole vector vs W1 cut: β_sel(W1_max) = α_edge(W1_max) * d_vec,
  - observed GLM dipole vector β_obs(W1_max) from the scan JSON,
  - a residual vector β_res = β_obs - β_sel, and its amplitude.

This is a "closure attempt": if β_sel accounts for most of β_obs, the amplitude is plausibly
selection-driven by depth variations.

Limitations
-----------
- invvar is a depth/noise proxy; CatWISE "accepted" selection is more complex than a pure
  detection threshold, so this is diagnostic rather than a proof.
- Tile-median invvar maps smear small-scale structure; this targets large-scale (l=1) effects.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def unitvec_to_lb(v: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        return float("nan"), float("nan")
    u = v / n
    l = math.degrees(math.atan2(u[1], u[0])) % 360.0
    b = math.degrees(math.asin(np.clip(u[2], -1.0, 1.0)))
    return l, b


def fit_dipole_least_squares(y: np.ndarray, nhat: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fit y ≈ a0 + a·n using OLS. Returns (a0, a_vec, cov_a_vec)."""

    y = np.asarray(y, dtype=float)
    nhat = np.asarray(nhat, dtype=float)
    if y.ndim != 1 or nhat.ndim != 2 or nhat.shape[1] != 3 or nhat.shape[0] != y.size:
        raise ValueError("shape mismatch for dipole fit")

    X = np.column_stack([np.ones_like(y), nhat[:, 0], nhat[:, 1], nhat[:, 2]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ beta
    dof = max(1, y.size - X.shape[1])
    s2 = float(np.dot(resid, resid) / dof)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = s2 * XtX_inv
    return float(beta[0]), np.asarray(beta[1:4], dtype=float), np.asarray(cov[1:4, 1:4], dtype=float)


def alpha_edge(w1: np.ndarray, cut: float, *, delta: float) -> float:
    w1 = np.asarray(w1, dtype=float)
    n1 = int(np.sum(w1 <= float(cut)))
    n0 = int(np.sum(w1 <= float(cut) - float(delta)))
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    return float((math.log(n1) - math.log(n0)) / float(delta))


@dataclass(frozen=True)
class Row:
    w1_cut: float
    beta_obs: np.ndarray
    D_obs: float
    alpha_edge: float
    beta_sel: np.ndarray
    D_sel: float
    beta_res: np.ndarray
    D_res: float

    def as_dict(self) -> Dict[str, Any]:
        l_obs, b_obs = unitvec_to_lb(self.beta_obs)
        l_sel, b_sel = unitvec_to_lb(self.beta_sel)
        l_res, b_res = unitvec_to_lb(self.beta_res)
        return {
            "w1_cut": float(self.w1_cut),
            "alpha_edge": float(self.alpha_edge),
            "D_obs": float(self.D_obs),
            "beta_obs_x": float(self.beta_obs[0]),
            "beta_obs_y": float(self.beta_obs[1]),
            "beta_obs_z": float(self.beta_obs[2]),
            "l_obs_deg": float(l_obs),
            "b_obs_deg": float(b_obs),
            "D_sel": float(self.D_sel),
            "beta_sel_x": float(self.beta_sel[0]),
            "beta_sel_y": float(self.beta_sel[1]),
            "beta_sel_z": float(self.beta_sel[2]),
            "l_sel_deg": float(l_sel),
            "b_sel_deg": float(b_sel),
            "D_res": float(self.D_res),
            "beta_res_x": float(self.beta_res[0]),
            "beta_res_y": float(self.beta_res[1]),
            "beta_res_z": float(self.beta_res[2]),
            "l_res_deg": float(l_res),
            "b_res_deg": float(b_res),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glm-scan-json",
        default="REPORTS/Q_D_RES_2_2/data/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json",
        help="Poisson GLM scan JSON (expects 'rows' with 'w1_cut' and 'beta_hat').",
    )
    ap.add_argument(
        "--invvar-map-fits",
        default="data/cache/unwise_invvar/neo7/invvar_healpix_nside64.fits",
        help="HEALPix map of (log) invvar values (Galactic coords).",
    )
    ap.add_argument("--invvar-map-is-log", action="store_true", help="Interpret --invvar-map-fits as log(invvar).")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument(
        "--catwise-catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--alpha-edge-delta", type=float, default=0.05)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    import healpy as hp
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    outdir = Path(args.outdir or "outputs/unwise_invvar_amplitude_closure")
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------- Load GLM scan (observed dipole vectors) --------------------
    scan = json.loads(Path(args.glm_scan_json).read_text())
    rows = scan.get("rows", [])
    if not rows:
        raise SystemExit(f"No 'rows' found in {args.glm_scan_json}")

    w1_cuts = [float(r["w1_cut"]) for r in rows]
    beta_obs = [np.asarray(r["beta_hat"][1:4], dtype=float) for r in rows]

    # -------------------- Load CatWISE (for alpha_edge) --------------------
    with fits.open(str(args.catwise_catalog), memmap=True) as hdul:
        c = hdul[1].data
        w1 = np.asarray(c["w1"], dtype=float)
        w1cov = np.asarray(c["w1cov"], dtype=float)
        l = np.asarray(c["l"], dtype=float)
        b = np.asarray(c["b"], dtype=float)

    base = np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b)
    base &= (w1cov >= float(args.w1cov_min)) & (np.abs(b) >= float(args.b_cut))
    w1_base = w1[base]

    # -------------------- Secrest-style footprint mask for map dipole fit --------------------
    # Build mask_zeros on the W1cov>=80 parent sample, then apply exclude discs and |b| cut.
    theta_base = np.deg2rad(90.0 - b[base])
    phi_base = np.deg2rad(l[base] % 360.0)
    ipix_base = hp.ang2pix(int(args.nside), theta_base, phi_base, nest=False)

    npix = hp.nside2npix(int(args.nside))
    mask = np.zeros(npix, dtype=bool)

    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        neigh = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            neigh[i] = hp.pixelfunc.get_all_neighbours(int(args.nside), int(ip))
        mask[idx0] = True
        mask[neigh] = True  # includes -1 indexing per Secrest utilities

    # Exclusion discs
    tmask = Table.read(str(args.exclude_mask_fits), memmap=True)
    if "use" in tmask.colnames:
        tmask = tmask[np.asarray(tmask["use"], dtype=bool)]
    if len(tmask):
        sc = SkyCoord(tmask["ra"], tmask["dec"], unit=u.deg, frame="icrs").galactic
        radius = np.deg2rad(np.asarray(tmask["radius"], dtype=float))
        for lon, lat, rad in zip(sc.l.deg, sc.b.deg, radius, strict=True):
            vec = hp.ang2vec(np.deg2rad(90.0 - float(lat)), np.deg2rad(float(lon)))
            disc = hp.query_disc(int(args.nside), vec=vec, radius=float(rad), inclusive=True, nest=False)
            mask[disc] = True

    # Galactic plane cut on pixel centers
    _, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(args.b_cut)

    seen = ~mask

    # -------------------- Load invvar map and build delta_m map --------------------
    inv = hp.read_map(str(args.invvar_map_fits), verbose=False)
    inv = np.asarray(inv, dtype=float)
    if int(hp.get_nside(inv)) != int(args.nside):
        inv = hp.ud_grade(inv, nside_out=int(args.nside), order_in="RING", order_out="RING", power=0)

    if args.invvar_map_is_log:
        inv = np.exp(inv)

    inv_ok = seen & np.isfinite(inv) & (inv > 0)
    if not np.any(inv_ok):
        raise SystemExit("No valid invvar pixels in seen mask.")
    inv_med = float(np.median(inv[inv_ok]))

    # Effective limiting-magnitude shift (up to an additive constant that cancels in the dipole).
    delta_m = np.full_like(inv, hp.UNSEEN, dtype=float)
    delta_m[inv_ok] = 1.25 * np.log10(inv[inv_ok] / inv_med)

    # Dipole fit of delta_m on seen pixels.
    lon_pix, lat_pix = hp.pix2ang(int(args.nside), np.arange(npix), lonlat=True)
    nhat = lb_to_unitvec(lon_pix[inv_ok], lat_pix[inv_ok])
    _, d_vec, cov_d = fit_dipole_least_squares(delta_m[inv_ok], nhat)
    d_amp = float(np.linalg.norm(d_vec))
    d_l, d_b = unitvec_to_lb(d_vec)

    # -------------------- Build predicted selection component per cut --------------------
    out_rows: List[Row] = []
    for w1_cut, bobs in zip(w1_cuts, beta_obs, strict=True):
        aedge = alpha_edge(w1_base, float(w1_cut), delta=float(args.alpha_edge_delta))
        bsel = d_vec * float(aedge) if np.isfinite(aedge) else np.array([float("nan")] * 3)
        Dobs = float(np.linalg.norm(bobs))
        Dsel = float(np.linalg.norm(bsel))
        bres = bobs - bsel
        Dres = float(np.linalg.norm(bres))
        out_rows.append(
            Row(
                w1_cut=float(w1_cut),
                beta_obs=np.asarray(bobs, dtype=float),
                D_obs=Dobs,
                alpha_edge=float(aedge),
                beta_sel=np.asarray(bsel, dtype=float),
                D_sel=Dsel,
                beta_res=np.asarray(bres, dtype=float),
                D_res=Dres,
            )
        )

    out_json = outdir / "invvar_amplitude_closure.json"
    payload = {
        "meta": {
            "glm_scan_json": str(args.glm_scan_json),
            "invvar_map_fits": str(args.invvar_map_fits),
            "invvar_map_is_log": bool(args.invvar_map_is_log),
            "catwise_catalog": str(args.catwise_catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "nside": int(args.nside),
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "alpha_edge_delta": float(args.alpha_edge_delta),
            "invvar_median_seen": float(inv_med),
            "delta_m_dipole_vec": [float(x) for x in d_vec],
            "delta_m_dipole_amp": float(d_amp),
            "delta_m_dipole_l_deg": float(d_l),
            "delta_m_dipole_b_deg": float(d_b),
            "delta_m_dipole_cov": cov_d.tolist(),
        },
        "rows": [r.as_dict() for r in out_rows],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out_json}")

    if args.make_plots:
        import matplotlib.pyplot as plt

        w = np.array([r.w1_cut for r in out_rows], dtype=float)
        Dobs = np.array([r.D_obs for r in out_rows], dtype=float)
        Dsel = np.array([r.D_sel for r in out_rows], dtype=float)
        Dres = np.array([r.D_res for r in out_rows], dtype=float)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(w, Dobs, "k-", lw=2, label="Observed (GLM)")
        plt.plot(w, Dsel, "C0--", lw=2, label="Predicted selection (invvar δm × α_edge)")
        plt.plot(w, Dres, "C3-", lw=2, label="Residual (obs − sel)")
        plt.xlabel(r"$W1_{\\max}$")
        plt.ylabel("Dipole amplitude")
        plt.title("Amplitude closure diagnostic (invvar-derived δm dipole)")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        fig.tight_layout()
        out_png = outdir / "invvar_amplitude_closure.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Wrote: {out_png}")

        # Mollweide of delta_m map
        fig = plt.figure(figsize=(10, 6))
        hp.mollview(
            delta_m,
            fig=fig.number,
            coord=["G"],
            title="unWISE W1 invvar-derived δm (tile-median proxy)",
            unit="mag (relative)",
            min=-0.2,
            max=0.2,
        )
        out_png2 = outdir / "invvar_delta_m_mollweide.png"
        fig.savefig(out_png2, dpi=200)
        plt.close(fig)
        print(f"Wrote: {out_png2}")

    # Also write a short markdown summary for easy copy/paste.
    md = outdir / "master_report.md"
    md.write_text(
        "\n".join(
            [
                "# invvar-based amplitude closure diagnostic",
                "",
                "This report uses an imaging-derived unWISE W1 inverse-variance depth proxy to build an effective",
                "limiting-magnitude shift map δm(n) and predicts the selection-driven dipole amplitude across W1 cuts.",
                "",
                f"- invvar map: `{args.invvar_map_fits}`",
                f"- GLM scan: `{args.glm_scan_json}`",
                "",
                "## Key numbers",
                "",
                f"- δm dipole amplitude: `{d_amp:.6g}` mag",
                f"- δm dipole axis (Galactic): `(l,b)=({d_l:.2f}°, {d_b:.2f}°)`",
                "",
                "## Files",
                "",
                f"- JSON: `{out_json}`",
                f"- Plot: `{outdir / 'invvar_amplitude_closure.png'}`",
                f"- Map: `{outdir / 'invvar_delta_m_mollweide.png'}`",
                "",
                "## Interpretation (one line)",
                "",
                "If the predicted selection curve tracks the observed GLM dipole amplitude, this supports a depth-driven",
                "selection origin for the amplitude. If not, substantial residual amplitude remains unexplained by this",
                "simple depth proxy.",
                "",
            ]
        )
    )
    print(f"Wrote: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
