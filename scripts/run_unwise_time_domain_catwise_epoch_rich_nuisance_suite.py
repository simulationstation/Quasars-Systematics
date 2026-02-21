#!/usr/bin/env python3
"""Run richer-nuisance robustness checks for CatWISE-parent epoch dipole stability.

This add-on suite extends the minimal epoch systematics stack with a richer
template basis (dust + stellar-density + depth-quality proxy), then evaluates:

1) Constant-amplitude rejection with richer nuisance control (base mask).
2) The same test on common-support + matched-depth pixels.
3) Covariance-sensitivity of the constant-D test under equicorrelated epochs.
4) Vector-sum sensitivity diagnostic against nuisance-attribution strength.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from run_unwise_time_domain_catwise_epoch_systematics_suite import (
    CATWISE_DEFAULT,
    DEPTH_DEFAULT,
    EXCLUDE_DEFAULT,
    build_seen_mask,
    constant_amplitude_chi2,
    dipole_metrics,
    fit_poisson_glm,
    read_healpix_map_fits,
    weighted_quantile,
    zscore,
)


STAR_COUNT_DEFAULT = (
    "outputs/star_w1_zeropoint_map_allwise_mod8_snr20_msig0p1_full/"
    "star_w1_resid_count_nside64_all.fits"
)
INVVAR_DEFAULT = "data/cache/unwise_invvar/neo7/invvar_healpix_nside64.fits"
MINIMAL_SUITE_DEFAULT = "REPORTS/unwise_time_domain_catwise_epoch_systematics_suite"
AMPLITUDE_TABLE_DEFAULT = "REPORTS/unwise_time_domain_catwise_epoch_amplitude/data/epoch_amplitude_table.csv"


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def read_epoch_dates(summary_json: dict[str, Any]) -> dict[int, str]:
    from astropy.time import Time

    out: dict[int, str] = {}
    for row in summary_json.get("epochs", []):
        e = int(row.get("epoch", -1))
        mjd = float(row.get("mjd_mean", np.nan))
        if e < 0:
            continue
        out[e] = "" if not np.isfinite(mjd) else str(Time(mjd, format="mjd").utc.isot)
    return out


def build_ebv_mean_map(
    *,
    catwise_catalog: Path,
    nside: int,
    w1cov_min: float,
) -> np.ndarray:
    import healpy as hp
    from astropy.io import fits

    npix = hp.nside2npix(int(nside))
    with fits.open(str(catwise_catalog), memmap=True) as hdul:
        d = hdul[1].data
        w1cov = np.asarray(d["w1cov"], dtype=np.float64)
        l = np.asarray(d["l"], dtype=np.float64)
        b = np.asarray(d["b"], dtype=np.float64)
        ebv = np.asarray(d["ebv"], dtype=np.float64)

    sel = (
        np.isfinite(w1cov)
        & np.isfinite(l)
        & np.isfinite(b)
        & np.isfinite(ebv)
        & (w1cov >= float(w1cov_min))
    )
    ipix = hp.ang2pix(
        int(nside),
        np.deg2rad(90.0 - b[sel]),
        np.deg2rad(l[sel] % 360.0),
        nest=False,
    ).astype(np.int64)

    cnt = np.bincount(ipix, minlength=npix).astype(np.float64)
    s = np.bincount(ipix, weights=ebv[sel], minlength=npix).astype(np.float64)
    out = np.full(npix, np.nan, dtype=np.float64)
    np.divide(s, cnt, out=out, where=cnt > 0.0)
    return out


def prepare_map_template(
    *,
    values: np.ndarray,
    seen: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(values, dtype=np.float64).copy()
    if transform == "log":
        arr = np.log(np.clip(arr, 1e-12, np.inf))
    elif transform == "log1p":
        arr = np.log1p(np.clip(arr, 0.0, np.inf))
    elif transform != "none":
        raise ValueError(f"Unknown transform={transform}")

    finite_seen = seen & np.isfinite(arr)
    if np.count_nonzero(finite_seen) == 0:
        arr = np.zeros_like(arr)
    else:
        fill = float(np.nanmedian(arr[finite_seen]))
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = fill

    z, mu, sd = zscore(arr, seen)
    return z, {"mean_seen": float(mu), "std_seen": float(sd)}


def run_epoch_fits_dynamic(
    *,
    counts: np.ndarray,
    parent_counts: np.ndarray,
    seen_mask: np.ndarray,
    nhat: np.ndarray,
    template_names: list[str],
    template_arrays: list[np.ndarray],
    epochs: list[int],
    date_iso_by_epoch: dict[int, str],
    fit_label: str,
) -> dict[str, Any]:
    mask = np.asarray(seen_mask, dtype=bool)
    if np.count_nonzero(mask) == 0:
        raise RuntimeError(f"{fit_label}: empty mask")

    nh = np.asarray(nhat[mask], dtype=np.float64)
    ymask_parent = np.asarray(parent_counts[mask], dtype=np.float64)
    tmat = np.column_stack([np.asarray(t[mask], dtype=np.float64) for t in template_arrays])

    X0 = np.ones((nh.shape[0], 1), dtype=np.float64)
    X_dip = np.column_stack([np.ones(nh.shape[0]), nh[:, 0], nh[:, 1], nh[:, 2]])
    X_nuis = np.column_stack([np.ones(nh.shape[0]), tmat])
    X_full = np.column_stack([np.ones(nh.shape[0]), nh[:, 0], nh[:, 1], nh[:, 2], tmat])

    rows: list[dict[str, Any]] = []
    b_init_null = None
    b_init_dip = None
    b_init_nuis = None
    b_init_full = None

    for e in epochs:
        y = np.asarray(counts[e, mask], dtype=np.float64)
        N = float(np.sum(y))
        if N <= 0.0:
            rows.append({"epoch": int(e), "date_utc": date_iso_by_epoch.get(int(e), ""), "N": 0})
            continue

        b0, _c0, _mu0, dev0 = fit_poisson_glm(X0, y, beta_init=b_init_null)
        b_init_null = b0
        bd, cd, _mud, devd = fit_poisson_glm(X_dip, y, beta_init=b_init_dip)
        b_init_dip = bd
        bn, _cn, _mun, devn = fit_poisson_glm(X_nuis, y, beta_init=b_init_nuis)
        b_init_nuis = bn
        bf, cf, _muf, devf = fit_poisson_glm(X_full, y, beta_init=b_init_full)
        b_init_full = bf

        md = dipole_metrics(bd, cd, dip_idx=(1, 2, 3))
        mf = dipole_metrics(bf, cf, dip_idx=(1, 2, 3))

        n_pix_unmasked = int(np.count_nonzero(mask))
        n_pix_active = int(np.count_nonzero(y > 0))
        f_pix_active = float(n_pix_active / n_pix_unmasked) if n_pix_unmasked > 0 else float("nan")

        parent_valid = ymask_parent > 0
        frac = np.zeros_like(y, dtype=np.float64)
        frac[parent_valid] = y[parent_valid] / ymask_parent[parent_valid]
        fq = weighted_quantile(frac[parent_valid], ymask_parent[parent_valid], [0.25, 0.5, 0.75])

        rows.append(
            {
                "fit_label": fit_label,
                "epoch": int(e),
                "date_utc": date_iso_by_epoch.get(int(e), ""),
                "N": int(N),
                "n_pix_unmasked": n_pix_unmasked,
                "n_pix_active": n_pix_active,
                "f_pix_active": f_pix_active,
                "parent_frac_q25": float(fq[0]),
                "parent_frac_q50": float(fq[1]),
                "parent_frac_q75": float(fq[2]),
                "D_dip_only": float(md["D"]),
                "sigma_D_dip_only": float(md["sigma_D"]),
                "D_rich": float(mf["D"]),
                "sigma_D_rich": float(mf["sigma_D"]),
                "l_dip_only_deg": float(md["l_deg"]),
                "b_dip_only_deg": float(md["b_deg"]),
                "l_rich_deg": float(mf["l_deg"]),
                "b_rich_deg": float(mf["b_deg"]),
                "dev_null": float(dev0),
                "dev_dip_only": float(devd),
                "dev_nuis_only": float(devn),
                "dev_rich": float(devf),
                "frac_dev_explained_nuis_only": float((dev0 - devn) / dev0) if dev0 > 0 else float("nan"),
                "frac_dev_explained_dip_after_nuis": float((devn - devf) / devn) if devn > 0 else float("nan"),
                "beta_rich": [float(x) for x in bf],
            }
        )

    D_d = np.array([rr.get("D_dip_only", np.nan) for rr in rows], dtype=float)
    s_d = np.array([rr.get("sigma_D_dip_only", np.nan) for rr in rows], dtype=float)
    D_r = np.array([rr.get("D_rich", np.nan) for rr in rows], dtype=float)
    s_r = np.array([rr.get("sigma_D_rich", np.nan) for rr in rows], dtype=float)

    return {
        "fit_label": fit_label,
        "template_names": list(template_names),
        "rows": rows,
        "summary": {
            "fit_label": fit_label,
            "n_epoch_rows": int(len(rows)),
            "constant_D_dip_only": constant_amplitude_chi2(D_d, s_d),
            "constant_D_rich": constant_amplitude_chi2(D_r, s_r),
            "n_unmasked_pix": int(np.count_nonzero(mask)),
        },
    }


def constant_D_correlated_equicorr(
    D: np.ndarray,
    sD: np.ndarray,
    *,
    rho: float,
) -> dict[str, float]:
    from scipy.stats import chi2 as chi2_dist

    d = np.asarray(D, dtype=np.float64)
    s = np.asarray(sD, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(s) & (s > 0.0)
    d = d[ok]
    s = s[ok]
    n = int(d.size)
    if n < 2:
        return {"n": n, "rho": float(rho), "chi2": float("nan"), "dof": 0, "p_value": float("nan")}

    if rho <= -1.0 / (n - 1) or rho >= 1.0:
        return {"n": n, "rho": float(rho), "chi2": float("nan"), "dof": n - 1, "p_value": float("nan")}

    S = np.outer(s, s) * float(rho)
    np.fill_diagonal(S, s * s)
    try:
        W = np.linalg.inv(S)
    except Exception:  # noqa: BLE001
        return {"n": n, "rho": float(rho), "chi2": float("nan"), "dof": n - 1, "p_value": float("nan")}

    one = np.ones(n, dtype=np.float64)
    denom = float(one @ W @ one)
    if not np.isfinite(denom) or denom <= 0.0:
        return {"n": n, "rho": float(rho), "chi2": float("nan"), "dof": n - 1, "p_value": float("nan")}
    d0 = float((one @ W @ d) / denom)
    resid = d - d0
    chi2 = float(resid @ W @ resid)
    dof = n - 1
    p = float(chi2_dist.sf(chi2, dof))
    return {"n": n, "rho": float(rho), "D_weighted_mean": d0, "chi2": chi2, "dof": dof, "p_value": p}


def vector_sum_sensitivity_diagnostic(
    *,
    amplitude_table_csv: Path,
    minimal_epoch_fits_csv: Path,
    epochs: list[int],
) -> dict[str, float]:
    dglm: dict[int, float] = {}
    dvec: dict[int, float] = {}
    with amplitude_table_csv.open() as f:
        for row in csv.DictReader(f):
            e = int(row["epoch"])
            if e not in epochs:
                continue
            dglm[e] = float(row["D_glm"])
            dvec[e] = float(row["D_vecsum"])

    frac_nuis: dict[int, float] = {}
    with minimal_epoch_fits_csv.open() as f:
        for row in csv.DictReader(f):
            e = int(row["epoch"])
            if e not in epochs:
                continue
            frac_nuis[e] = float(row["frac_dev_explained_nuis_only"])

    e_ok = [e for e in epochs if e in dglm and e in dvec and e in frac_nuis]
    if len(e_ok) < 3:
        return {
            "n": int(len(e_ok)),
            "corr_diff_vs_frac_nuis": float("nan"),
            "diff_mean": float("nan"),
            "diff_std": float("nan"),
            "max_diff_epoch": -1,
            "max_diff_value": float("nan"),
        }

    diff = np.array([dvec[e] - dglm[e] for e in e_ok], dtype=np.float64)
    frac = np.array([frac_nuis[e] for e in e_ok], dtype=np.float64)
    corr = float(np.corrcoef(diff, frac)[0, 1]) if np.std(diff) > 0 and np.std(frac) > 0 else float("nan")
    imax = int(np.argmax(diff))
    return {
        "n": int(len(e_ok)),
        "corr_diff_vs_frac_nuis": corr,
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "max_diff_epoch": int(e_ok[imax]),
        "max_diff_value": float(diff[imax]),
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def make_figure(
    *,
    out_png: Path,
    base_rows: list[dict[str, Any]],
    common_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
        return np.array([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)

    e = np.array([int(r.get("epoch", -1)) for r in base_rows], dtype=np.int64)
    d_min = arr(base_rows, "D_dip_only")
    s_min = arr(base_rows, "sigma_D_dip_only")
    d_rich = arr(base_rows, "D_rich")
    s_rich = arr(base_rows, "sigma_D_rich")

    ec = np.array([int(r.get("epoch", -1)) for r in common_rows], dtype=np.int64)
    d_rich_c = arr(common_rows, "D_rich")
    s_rich_c = arr(common_rows, "sigma_D_rich")

    plt.figure(figsize=(8.4, 4.8))
    plt.errorbar(e, d_min, yerr=s_min, fmt="o-", ms=4, lw=1.1, label="Dipole-only (base)")
    plt.errorbar(e, d_rich, yerr=s_rich, fmt="s-", ms=3.8, lw=1.1, label="Rich nuisance (base)")
    plt.errorbar(ec, d_rich_c, yerr=s_rich_c, fmt="^-", ms=3.6, lw=1.0, label="Rich nuisance (common+depth)")
    plt.xlabel("Epoch")
    plt.ylabel("Dipole amplitude D")
    plt.title("Epoch amplitudes under richer nuisance control")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def write_master_report(path: Path, payload: dict[str, Any]) -> None:
    s_base = payload["summaries"]["base"]["constant_D_rich"]
    s_cd = payload["summaries"]["common_support_depth"]["constant_D_rich"]
    s_min_base = payload["summaries"]["base"]["constant_D_dip_only"]
    vec_diag = payload["vector_sum_diagnostic"]
    cov_rows = payload["covariance_sensitivity"]["base_rich"]

    cov_lines = "\n".join(
        [
            f"- rho={row['rho']:.2f}: chi2={row['chi2']:.3f}, dof={row['dof']}, p={row['p_value']:.3e}"
            for row in cov_rows
        ]
    )

    text = f"""# CatWISE-parent epoch richer-nuisance robustness suite ({payload['meta']['fit_label']})

Date: {payload['meta']['date_utc']} (UTC)

## Rich nuisance template basis

Dipole + minimal basis (`abs(elat)`, `sin(elon)`, `cos(elon)`, `logNexp`) plus:
- `ebv_mean_z` (pixel EBV mean from CatWISE catalog)
- `log1p_star_count_z` (ALLWISE star-count proxy)
- `log_invvar_z` (unWISE W1 inverse-variance depth-quality proxy)

## Constant-D test (epochs 0–{payload['config']['epochs_max']})

Base mask:
- Dipole-only: chi2={s_min_base['chi2']:.3f}, dof={s_min_base['dof']}, p={s_min_base['p_value']:.3e}
- Rich nuisance: chi2={s_base['chi2']:.3f}, dof={s_base['dof']}, p={s_base['p_value']:.3e}

Common-support + matched-depth mask:
- Rich nuisance: chi2={s_cd['chi2']:.3f}, dof={s_cd['dof']}, p={s_cd['p_value']:.3e}

## Covariance sensitivity (equicorrelated-epoch approximation; rich nuisance, base mask)

{cov_lines}

## Vector-sum sensitivity diagnostic (epochs 0–{payload['config']['epochs_max']})

- corr[(D_vecsum-D_glm), frac_dev_explained_nuis_only] = {vec_diag['corr_diff_vs_frac_nuis']:.4f}
- mean(D_vecsum-D_glm) = {vec_diag['diff_mean']:.5f} ± {vec_diag['diff_std']:.5f}
- max epoch diff: e={vec_diag['max_diff_epoch']} with D_vecsum-D_glm={vec_diag['max_diff_value']:.5f}

## Files

- data/summary.json
- data/epoch_fits_base_rich.csv
- data/epoch_fits_common_support_depth_rich.csv
- data/covariance_sensitivity.json
- figures/D_vs_epoch_rich_nuisance.png
"""
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-outdir",
        default="outputs/epoch_dipole_time_domain_catwise_parent_20260204_232537UTC",
    )
    ap.add_argument("--report-dir", default="REPORTS/unwise_time_domain_catwise_epoch_rich_nuisance_suite")
    ap.add_argument("--catwise-catalog", default=CATWISE_DEFAULT)
    ap.add_argument("--exclude-mask-fits", default=EXCLUDE_DEFAULT)
    ap.add_argument("--depth-map-fits", default=DEPTH_DEFAULT)
    ap.add_argument("--star-count-map-fits", default=STAR_COUNT_DEFAULT)
    ap.add_argument("--invvar-map-fits", default=INVVAR_DEFAULT)
    ap.add_argument("--minimal-suite-dir", default=MINIMAL_SUITE_DEFAULT)
    ap.add_argument("--amplitude-table-csv", default=AMPLITUDE_TABLE_DEFAULT)
    ap.add_argument("--epochs-max", type=int, default=15)
    args = ap.parse_args()

    run_outdir = Path(args.run_outdir)
    report_dir = Path(args.report_dir)
    data_dir = report_dir / "data"
    fig_dir = report_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run_outdir / "run_config.json").read_text())
    run_summary = json.loads((run_outdir / "summary.json").read_text())
    date_iso_by_epoch = read_epoch_dates(run_summary)
    counts = np.load(run_outdir / "counts_by_epoch.npy")

    with np.load(run_outdir / "parent_index.npz", allow_pickle=False) as z:
        parent_pix64 = np.asarray(z["parent_pix64"], dtype=np.int64)

    import healpy as hp

    nside = int(cfg["nside"])
    npix = hp.nside2npix(int(nside))
    if counts.shape[1] != npix:
        raise RuntimeError(f"counts npix mismatch: {counts.shape[1]} vs {npix}")

    parent_counts = np.bincount(parent_pix64, minlength=npix).astype(np.int64)
    seen = build_seen_mask(
        nside=nside,
        catwise_catalog=Path(args.catwise_catalog),
        exclude_mask_fits=Path(args.exclude_mask_fits),
        b_cut_deg=float(cfg["b_cut_deg"]),
        w1cov_min=float(cfg["w1cov_min"]),
    )

    xpix, ypix, zpix = hp.pix2vec(nside, np.arange(npix, dtype=np.int64), nest=False)
    nhat = np.column_stack([xpix, ypix, zpix]).astype(np.float64)

    depth_map = read_healpix_map_fits(Path(args.depth_map_fits))
    invvar_map = read_healpix_map_fits(Path(args.invvar_map_fits))
    star_count_map = read_healpix_map_fits(Path(args.star_count_map_fits))
    ebv_mean_map = build_ebv_mean_map(
        catwise_catalog=Path(args.catwise_catalog),
        nside=nside,
        w1cov_min=float(cfg["w1cov_min"]),
    )

    if depth_map.shape[0] != npix or invvar_map.shape[0] != npix or star_count_map.shape[0] != npix:
        raise RuntimeError("Template map nside/npix mismatch")

    # Minimal template basis.
    from run_unwise_time_domain_catwise_epoch_systematics_suite import build_templates

    t_min = build_templates(nside=nside, seen=seen, depth_map=depth_map)
    ebv_z, ebv_stats = prepare_map_template(values=ebv_mean_map, seen=seen, transform="none")
    star_count_z, star_stats = prepare_map_template(values=star_count_map, seen=seen, transform="log1p")
    invvar_z, invvar_stats = prepare_map_template(values=invvar_map, seen=seen, transform="log")

    template_names = [
        "abs_elat_z",
        "sin_elon_z",
        "cos_elon_z",
        "depth_z",
        "ebv_mean_z",
        "log1p_star_count_z",
        "log_invvar_z",
    ]
    template_arrays = [
        np.asarray(t_min["abs_elat_z"], dtype=np.float64),
        np.asarray(t_min["sin_elon_z"], dtype=np.float64),
        np.asarray(t_min["cos_elon_z"], dtype=np.float64),
        np.asarray(t_min["depth_z"], dtype=np.float64),
        ebv_z,
        star_count_z,
        invvar_z,
    ]

    epochs_max = int(args.epochs_max)
    epochs = [int(e) for e in range(min(counts.shape[0], epochs_max + 1))]

    print("Running base-mask richer nuisance fits...", flush=True)
    base_fit = run_epoch_fits_dynamic(
        counts=counts,
        parent_counts=parent_counts,
        seen_mask=seen,
        nhat=nhat,
        template_names=template_names,
        template_arrays=template_arrays,
        epochs=epochs,
        date_iso_by_epoch=date_iso_by_epoch,
        fit_label="base_mask_rich",
    )

    print("Building common-support + matched-depth mask...", flush=True)
    seen_parent = seen & (parent_counts > 0)
    common_support = seen_parent & np.all(counts[epochs, :] > 0, axis=0)
    depth_raw = np.asarray(t_min["depth_raw"], dtype=np.float64)
    q10_list: list[float] = []
    q90_list: list[float] = []
    for e in epochs:
        w = counts[e, common_support].astype(np.float64)
        q = weighted_quantile(depth_raw[common_support], w, [0.10, 0.90])
        q10_list.append(float(q[0]))
        q90_list.append(float(q[1]))
    qlo = float(np.nanmax(np.asarray(q10_list, dtype=np.float64)))
    qhi = float(np.nanmin(np.asarray(q90_list, dtype=np.float64)))
    if (not np.isfinite(qlo)) or (not np.isfinite(qhi)) or (qhi <= qlo):
        q = weighted_quantile(depth_raw[common_support], parent_counts[common_support].astype(np.float64), [0.05, 0.95])
        qlo, qhi = float(q[0]), float(q[1])
    common_support_depth = common_support & np.isfinite(depth_raw) & (depth_raw >= qlo) & (depth_raw <= qhi)

    print("Running common-support + matched-depth richer nuisance fits...", flush=True)
    common_depth_fit = run_epoch_fits_dynamic(
        counts=counts,
        parent_counts=parent_counts,
        seen_mask=common_support_depth,
        nhat=nhat,
        template_names=template_names,
        template_arrays=template_arrays,
        epochs=epochs,
        date_iso_by_epoch=date_iso_by_epoch,
        fit_label="common_support_depth_rich",
    )

    print("Computing covariance sensitivity...", flush=True)
    D_base = np.array([float(r.get("D_rich", np.nan)) for r in base_fit["rows"]], dtype=np.float64)
    s_base = np.array([float(r.get("sigma_D_rich", np.nan)) for r in base_fit["rows"]], dtype=np.float64)
    D_cd = np.array([float(r.get("D_rich", np.nan)) for r in common_depth_fit["rows"]], dtype=np.float64)
    s_cd = np.array([float(r.get("sigma_D_rich", np.nan)) for r in common_depth_fit["rows"]], dtype=np.float64)

    rho_grid = [0.0, 0.2, 0.4, 0.6]
    cov_base = [constant_D_correlated_equicorr(D_base, s_base, rho=rho) for rho in rho_grid]
    cov_cd = [constant_D_correlated_equicorr(D_cd, s_cd, rho=rho) for rho in rho_grid]

    print("Computing vector-sum sensitivity diagnostic...", flush=True)
    vec_diag = vector_sum_sensitivity_diagnostic(
        amplitude_table_csv=Path(args.amplitude_table_csv),
        minimal_epoch_fits_csv=Path(args.minimal_suite_dir) / "data/epoch_fits_base.csv",
        epochs=epochs,
    )

    print("Writing report products...", flush=True)
    fields = [
        "fit_label",
        "epoch",
        "date_utc",
        "N",
        "n_pix_unmasked",
        "n_pix_active",
        "f_pix_active",
        "parent_frac_q25",
        "parent_frac_q50",
        "parent_frac_q75",
        "D_dip_only",
        "sigma_D_dip_only",
        "D_rich",
        "sigma_D_rich",
        "l_dip_only_deg",
        "b_dip_only_deg",
        "l_rich_deg",
        "b_rich_deg",
        "dev_null",
        "dev_dip_only",
        "dev_nuis_only",
        "dev_rich",
        "frac_dev_explained_nuis_only",
        "frac_dev_explained_dip_after_nuis",
    ]
    write_rows_csv(data_dir / "epoch_fits_base_rich.csv", base_fit["rows"], fields)
    write_rows_csv(data_dir / "epoch_fits_common_support_depth_rich.csv", common_depth_fit["rows"], fields)
    write_rows_csv(
        data_dir / "epoch_comparability_table_rich.csv",
        base_fit["rows"],
        ["epoch", "date_utc", "N", "n_pix_unmasked", "n_pix_active", "f_pix_active", "parent_frac_q25", "parent_frac_q50", "parent_frac_q75"],
    )

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "date_utc": utc_date(),
            "fit_label": f"epoch_rich_nuisance_suite_{utc_tag()}",
            "run_outdir": str(run_outdir),
            "report_dir": str(report_dir),
        },
        "config": {
            "nside": nside,
            "epochs_max": epochs_max,
            "epochs_used": epochs,
            "depth_map_fits": str(args.depth_map_fits),
            "star_count_map_fits": str(args.star_count_map_fits),
            "invvar_map_fits": str(args.invvar_map_fits),
            "minimal_suite_dir": str(args.minimal_suite_dir),
        },
        "templates": {
            "names": template_names,
            "transforms": {
                "abs_elat_z": "zscore",
                "sin_elon_z": "zscore",
                "cos_elon_z": "zscore",
                "depth_z": "zscore(logNexp)",
                "ebv_mean_z": "zscore(pixel mean EBV)",
                "log1p_star_count_z": "zscore(log1p(star_count))",
                "log_invvar_z": "zscore(log(invvar))",
            },
            "stats_seen": {
                "ebv_mean_z": ebv_stats,
                "log1p_star_count_z": star_stats,
                "log_invvar_z": invvar_stats,
            },
        },
        "common_support": {
            "n_seen_pix": int(np.count_nonzero(seen)),
            "n_common_support_depth_pix": int(np.count_nonzero(common_support_depth)),
            "matched_depth_definition": {"depth_q_lo": qlo, "depth_q_hi": qhi},
        },
        "summaries": {
            "base": base_fit["summary"],
            "common_support_depth": common_depth_fit["summary"],
        },
        "covariance_sensitivity": {
            "base_rich": cov_base,
            "common_support_depth_rich": cov_cd,
            "note": "equicorrelated epoch covariance approximation with fixed dof=n-1",
        },
        "vector_sum_diagnostic": vec_diag,
    }

    (data_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (data_dir / "covariance_sensitivity.json").write_text(
        json.dumps(payload["covariance_sensitivity"], indent=2) + "\n"
    )
    (data_dir / "template_definitions.json").write_text(json.dumps(payload["templates"], indent=2) + "\n")

    make_figure(
        out_png=fig_dir / "D_vs_epoch_rich_nuisance.png",
        base_rows=base_fit["rows"],
        common_rows=common_depth_fit["rows"],
    )
    write_master_report(report_dir / "master_report.md", payload)

    print(f"Wrote richer-nuisance report -> {report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

