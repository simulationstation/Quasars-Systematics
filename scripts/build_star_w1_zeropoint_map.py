#!/usr/bin/env python3
"""
Build an independent W1 zero-point / δm map from bright AllWISE stars.

Method
------
1) Download a bright, clean star sample from IRSA TAP (see `download_allwise_star_sample.py`).
2) Fit a simple color model to predict W1 from 2MASS colors:
     y = (W1 - K)  ~  c0 + c1*(J-K) + c2*(H-K)
3) Define the residual:
     r = (W1 - K) - y_hat
   This is an approximate per-source W1 zero-point residual (plus astrophysical color-model error).
4) Bin r into a HEALPix map in Galactic coordinates, producing δm(pix).

This gives an *independent* map-level systematic proxy that does not use quasar counts.

Optionally, you can build multiple maps in time slices using `w1mjdmean` bins. This does *not*
reconstruct a quasar catalog per epoch; it measures (star-based) photometric residual structure
as a function of time, which can be used to predict amplitude-only dipole contamination.
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


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return np.zeros_like(x)
    m = float(np.mean(x[valid]))
    s = float(np.std(x[valid]))
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)]).astype(float)


def fit_dipole_ls(map_vals: np.ndarray, seen: np.ndarray, n_unit_seen: np.ndarray) -> dict[str, Any]:
    """Fit map_vals ~ a0 + b·n on seen pixels; return b and its norm."""
    y = np.asarray(map_vals, dtype=float)[seen]
    X = np.column_stack([np.ones_like(y), n_unit_seen[:, 0], n_unit_seen[:, 1], n_unit_seen[:, 2]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b = np.asarray(beta[1:4], dtype=float)
    D = float(np.linalg.norm(b))
    return {"b_vec": [float(x) for x in b], "D": D}


def fit_dipole_wls(map_vals: np.ndarray, seen: np.ndarray, n_unit: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Weighted LS fit map_vals ~ a0 + b·n on seen pixels; returns b and its norm."""
    seen = np.asarray(seen, dtype=bool)
    y = np.asarray(map_vals, dtype=float)[seen]
    n_seen = np.asarray(n_unit, dtype=float)[seen]
    w = np.asarray(weights, dtype=float)[seen]
    w = np.clip(w, 0.0, np.inf)
    ok = np.isfinite(y) & np.all(np.isfinite(n_seen), axis=1) & np.isfinite(w) & (w > 0.0)
    if int(np.sum(ok)) < 10:
        return {"b_vec": [float("nan")] * 3, "D": float("nan"), "n_pix_used": int(np.sum(ok))}
    y = y[ok]
    n_seen = n_seen[ok]
    w = w[ok]

    X = np.column_stack([np.ones_like(y), n_seen[:, 0], n_seen[:, 1], n_seen[:, 2]])
    # Solve (X^T W X) beta = X^T W y
    XtW = X.T * w
    beta = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
    b = np.asarray(beta[1:4], dtype=float)
    return {"b_vec": [float(x) for x in b], "D": float(np.linalg.norm(b)), "n_pix_used": int(y.size)}


@dataclass(frozen=True)
class StarArrays:
    glon: np.ndarray
    glat: np.ndarray
    w1: np.ndarray
    w1mjdmean: np.ndarray
    j: np.ndarray
    h: np.ndarray
    k: np.ndarray


def load_chunks(sample_dir: Path) -> StarArrays:
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    chunks = manifest.get("chunks", {})
    paths = [Path(v["path"]) for v in chunks.values()]
    if not paths:
        raise RuntimeError(f"No chunks found in {sample_dir}")

    glon_list: list[np.ndarray] = []
    glat_list: list[np.ndarray] = []
    w1_list: list[np.ndarray] = []
    mjd_list: list[np.ndarray] = []
    j_list: list[np.ndarray] = []
    h_list: list[np.ndarray] = []
    k_list: list[np.ndarray] = []

    for p in sorted(paths):
        # CSV with header; columns are known from the downloader.
        arr = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding=None)
        if arr.size == 0:
            continue
        glon_list.append(np.asarray(arr["glon"], float))
        glat_list.append(np.asarray(arr["glat"], float))
        w1_list.append(np.asarray(arr["w1mpro"], float))
        mjd_list.append(np.asarray(arr["w1mjdmean"], float))
        j_list.append(np.asarray(arr["j_m_2mass"], float))
        h_list.append(np.asarray(arr["h_m_2mass"], float))
        k_list.append(np.asarray(arr["k_m_2mass"], float))

    glon = np.concatenate(glon_list)
    glat = np.concatenate(glat_list)
    w1 = np.concatenate(w1_list)
    w1mjdmean = np.concatenate(mjd_list)
    j = np.concatenate(j_list)
    h = np.concatenate(h_list)
    k = np.concatenate(k_list)
    return StarArrays(glon=glon, glat=glat, w1=w1, w1mjdmean=w1mjdmean, j=j, h=h, k=k)


def parse_comma_floats(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def make_healpix_mean_map(*, npix: int, ipix: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ipix = np.asarray(ipix, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    cnt = np.bincount(ipix, minlength=int(npix)).astype(np.int64)
    sum_v = np.bincount(ipix, weights=values, minlength=int(npix)).astype(float)
    sum_v2 = np.bincount(ipix, weights=values * values, minlength=int(npix)).astype(float)
    seen = cnt > 0
    mean_v = np.zeros(int(npix), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_v[seen] = sum_v[seen] / cnt[seen]
    fill = float(np.median(mean_v[seen])) if np.any(seen) else 0.0
    mean_v[~seen] = fill
    # Per-pixel variance estimate of values (unbiased for cnt>1).
    var_v = np.full(int(npix), float("nan"), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = sum_v2[seen] / cnt[seen]
        var = m2 - mean_v[seen] ** 2
        var = np.clip(var, 0.0, np.inf)
        var_v[seen] = var
    # Standard error of the mean (heuristic; uses within-pixel variance when available).
    se_mean = np.full(int(npix), float("nan"), dtype=float)
    ok = seen & (cnt > 1) & np.isfinite(var_v)
    se_mean[ok] = np.sqrt(var_v[ok] / cnt[ok])
    return mean_v, seen, cnt, fill, se_mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, help="Directory created by download_allwise_star_sample.py")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--time-bins-mjd",
        default=None,
        help=(
            "Optional comma-separated MJD edges for time-sliced maps, e.g. '55240,55300,55360,55420'. "
            "Bins are [lo,hi). If omitted, only an all-times map is built."
        ),
    )
    ap.add_argument(
        "--alpha-edge",
        type=float,
        default=None,
        help=(
            "Optional alpha_edge = d ln N(<m)/dm used to convert δm dipole amplitude (mag) into a predicted "
            "count-dipole amplitude via D_counts ≈ alpha_edge * D_mag."
        ),
    )
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    import healpy as hp

    sample_dir = Path(str(args.sample_dir))
    outdir = Path(args.outdir or f"outputs/star_w1_zeropoint_map_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    stars = load_chunks(sample_dir)
    n_total = int(stars.w1.size)
    print(f"Loaded stars: n={n_total}")

    # Fit color model: (W1 - K) ~ 1 + (J-K) + (H-K).
    y = stars.w1 - stars.k
    x1 = stars.j - stars.k
    x2 = stars.h - stars.k
    X = np.column_stack([np.ones_like(y), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    # HEALPix binning in Galactic coords (RING).
    nside = int(args.nside)
    npix = int(hp.nside2npix(nside))
    ipix = hp.ang2pix(nside, np.deg2rad(90.0 - stars.glat), np.deg2rad(stars.glon % 360.0), nest=False)
    lon_pix, lat_pix = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    n_unit = lb_to_unitvec(lon_pix, lat_pix)

    alpha_edge = None if args.alpha_edge is None else float(args.alpha_edge)

    def write_bundle(*, label: str, sel: np.ndarray) -> dict[str, Any]:
        """Write map/meta/plots for a subset selection; return a summary dict."""
        sel = np.asarray(sel, dtype=bool)
        if sel.shape != stars.w1.shape:
            raise ValueError("Selection mask has wrong shape.")
        n_sel = int(np.sum(sel))
        if n_sel == 0:
            raise RuntimeError(f"Empty selection for label={label}")

        m, seen, cnt, fill, se_mean = make_healpix_mean_map(npix=npix, ipix=ipix[sel], values=resid[sel])
        dip = fit_dipole_ls(m, seen, n_unit[seen])
        dip_w_cnt = fit_dipole_wls(m, seen, n_unit, cnt.astype(float))
        dip_w_se = fit_dipole_wls(m, seen, n_unit, np.where(np.isfinite(se_mean) & (se_mean > 0), 1.0 / (se_mean * se_mean), 0.0))
        map_std_seen = float(np.std(m[seen])) if np.any(seen) else float("nan")

        meta: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sample_dir": str(sample_dir),
            "label": str(label),
            "n_stars": int(n_sel),
            "nside": int(nside),
            "model": {"y": "w1-k", "X": ["1", "j-k", "h-k"], "beta": [float(x) for x in beta]},
            "resid_mean": float(np.mean(resid[sel])),
            "resid_std": float(np.std(resid[sel])),
            "map_fill_value": float(fill),
            "map_seen_frac": float(np.mean(seen)),
            "map_std_seen": map_std_seen,
            "dipole_fit_mag": dip,
            "dipole_fit_mag_wls_count": dip_w_cnt,
            "dipole_fit_mag_wls_se": dip_w_se,
        }
        if alpha_edge is not None and np.isfinite(alpha_edge):
            meta["alpha_edge"] = float(alpha_edge)
            meta["dipole_pred_counts"] = float(alpha_edge * float(dip["D"]))
            meta["dipole_pred_counts_wls_count"] = float(alpha_edge * float(dip_w_cnt["D"]))
            meta["dipole_pred_counts_wls_se"] = float(alpha_edge * float(dip_w_se["D"]))

        out_fits = outdir / f"star_w1_resid_map_nside{nside}_{label}.fits"
        hp.write_map(str(out_fits), m.astype(np.float32), overwrite=True, dtype=np.float32)
        out_cnt = outdir / f"star_w1_resid_count_nside{nside}_{label}.fits"
        hp.write_map(str(out_cnt), cnt.astype(np.int32), overwrite=True, dtype=np.int32)
        out_meta = outdir / f"star_w1_resid_map_{label}.meta.json"
        out_meta.write_text(json.dumps(meta, indent=2))
        print(f"Wrote: {out_fits}")
        print(f"Wrote: {out_cnt}")
        print(f"Wrote: {out_meta}")

        if bool(args.make_plots):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10.5, 5.0), dpi=160)
            hp.mollview(m, title=f"AllWISE star residual map  δm (mag)  [{label}]", unit="mag", cmap="coolwarm", fig=fig.number)
            hp.graticule(dpar=30, dmer=30, alpha=0.25)
            fig.savefig(outdir / f"star_w1_resid_moll_{label}.png", bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0), dpi=160)
            ax.hist(m[seen], bins=80, alpha=0.85, color="#4C72B0")
            ax.set_title(f"Per-pixel mean residuals (seen pixels) [{label}]")
            ax.set_xlabel("δm [mag]")
            ax.set_ylabel("pixels")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(outdir / f"star_w1_resid_hist_{label}.png")
            plt.close(fig)

        return {
            "label": str(label),
            "n_stars": n_sel,
            "map_seen_frac": float(np.mean(seen)),
            "map_std_seen": map_std_seen,
            "D_mag": float(dip["D"]),
            "D_mag_wls_count": float(dip_w_cnt["D"]),
            "D_mag_wls_se": float(dip_w_se["D"]),
            "D_counts_pred": None if alpha_edge is None else float(alpha_edge * float(dip["D"])),
            "D_counts_pred_wls_count": None if alpha_edge is None else float(alpha_edge * float(dip_w_cnt["D"])),
            "D_counts_pred_wls_se": None if alpha_edge is None else float(alpha_edge * float(dip_w_se["D"])),
        }

    # Always write the full (all-times) map.
    summaries: list[dict[str, Any]] = []
    summaries.append(write_bundle(label="all", sel=np.ones_like(stars.w1, dtype=bool)))

    # Optional time slicing.
    if args.time_bins_mjd:
        edges = parse_comma_floats(str(args.time_bins_mjd))
        if len(edges) < 2:
            raise SystemExit("--time-bins-mjd must provide at least 2 comma-separated edges.")
        edges = sorted(edges)
        mjd = np.asarray(stars.w1mjdmean, dtype=float)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            sel = (mjd >= float(lo)) & (mjd < float(hi))
            if int(np.sum(sel)) == 0:
                print(f"SKIP: empty MJD bin [{lo},{hi})")
                continue
            lab = f"mjd{lo:.0f}_{hi:.0f}".replace(".", "p")
            summaries.append(write_bundle(label=lab, sel=sel))

    # Write a short report snippet.
    rep = outdir / "summary.md"
    lines: list[str] = [
        "# Star-based W1 residual maps (AllWISE)",
        "",
        f"- Sample dir: `{sample_dir}`",
        f"- Stars total: `{n_total}`",
        f"- HEALPix: nside `{nside}` (RING, Galactic)",
        "- Residual model: `w1-k ~ 1 + (j-k) + (h-k)`",
        f"- Residual std (per star, all): `{float(np.std(resid)):.4f}` mag",
    ]
    if alpha_edge is not None:
        lines.append(f"- alpha_edge (for D_counts prediction): `{alpha_edge:.6g}`")
    lines += [
        "",
        "## Map summaries",
        "",
        "| label | n_stars | seen_frac | map_std_seen [mag] | D_mag [mag] | D_mag (wls cnt) | D_mag (wls se) | D_counts_pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        dcounts = s["D_counts_pred"]
        dcounts_str = "—" if dcounts is None else f"{float(dcounts):.6f}"
        lines.append(
            f"| `{s['label']}` | {int(s['n_stars'])} | {float(s['map_seen_frac']):.4f} | {float(s['map_std_seen']):.6e} | {float(s['D_mag']):.6f} | {float(s['D_mag_wls_count']):.6f} | {float(s['D_mag_wls_se']):.6f} | {dcounts_str} |"
        )
    rep.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {rep}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
