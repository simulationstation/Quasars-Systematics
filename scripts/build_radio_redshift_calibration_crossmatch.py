#!/usr/bin/env python3
"""Build radio redshift calibration priors via DR16Q crossmatches."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dr16q-fits", default="data/external/sdss_dr16q/DR16Q_v4.fits")
    ap.add_argument("--dr16q-ra-col", default="RA")
    ap.add_argument("--dr16q-dec-col", default="DEC")
    ap.add_argument("--dr16q-z-col", default="Z")
    ap.add_argument("--dr16q-z-min", type=float, default=0.0)
    ap.add_argument("--dr16q-z-max", type=float, default=7.0)

    ap.add_argument(
        "--radio-audit-json",
        default="outputs/radio_combined_same_logic_audit_20260208_060931UTC/radio_combined_same_logic_audit.json",
        help="Used to read canonical flux cuts per survey.",
    )
    ap.add_argument(
        "--nvss-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/nvss/reference/NVSS.fit",
    )
    ap.add_argument("--nvss-ra-col", default="RAJ2000")
    ap.add_argument("--nvss-dec-col", default="DEJ2000")
    ap.add_argument("--nvss-flux-col", default="S1_4")
    ap.add_argument("--nvss-match-arcsec", type=float, default=15.0)
    ap.add_argument("--nvss-cut-ref-mjy", type=float, default=20.0)

    ap.add_argument(
        "--lotss-fits",
        default="data/external/radio_dipole/lotss_dr2/LoTSS_DR2_v110_masked.srl.fits",
    )
    ap.add_argument("--lotss-ra-col", default="RA")
    ap.add_argument("--lotss-dec-col", default="DEC")
    ap.add_argument("--lotss-flux-col", default="Total_flux")
    ap.add_argument("--lotss-match-arcsec", type=float, default=3.0)
    ap.add_argument("--lotss-cut-ref-mjy", type=float, default=5.0)

    ap.add_argument(
        "--racs-csv-gz",
        default="data/external/radio_dipole/racs_low/racs_low_dr1_sources_galacticcut_v2021_08_v02_mincols.csv.gz",
    )
    ap.add_argument("--racs-ra-col", default="ra")
    ap.add_argument("--racs-dec-col", default="dec")
    ap.add_argument("--racs-flux-col", default="total_flux_source")
    ap.add_argument("--racs-match-arcsec", type=float, default=3.0)
    ap.add_argument("--racs-cut-ref-mjy", type=float, default=20.0)

    ap.add_argument("--min-matches-per-cut", type=int, default=30)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument(
        "--base-calibration-json",
        default="configs/shared_redshift_calibration_from_dr16q.json",
        help="Optional existing calibration JSON to merge into output.",
    )
    ap.add_argument(
        "--out-calibration",
        default="configs/shared_redshift_calibration_qso_radio_xmatch.json",
    )
    return ap.parse_args()


def _read_fits_cols(path: Path, ra_col: str, dec_col: str, flux_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        cols = {c.name.upper(): c.name for c in hdul[1].columns}
        ra_key = cols.get(ra_col.upper())
        dec_key = cols.get(dec_col.upper())
        flux_key = cols.get(flux_col.upper())
        if ra_key is None or dec_key is None or flux_key is None:
            raise KeyError(f"{path}: missing required columns ({ra_col}, {dec_col}, {flux_col})")
        d = hdul[1].data
        ra = np.asarray(d[ra_key], dtype=float)
        dec = np.asarray(d[dec_key], dtype=float)
        flux = np.asarray(d[flux_key], dtype=float)
    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(flux) & (flux > 0.0)
    return ra[mask], dec[mask], flux[mask]


def _read_racs_csv(path: Path, ra_col: str, dec_col: str, flux_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra_list: list[float] = []
    dec_list: list[float] = []
    flux_list: list[float] = []
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ra = float(row[ra_col])
                dec = float(row[dec_col])
                flux = float(row[flux_col])
            except Exception:
                continue
            if not (math.isfinite(ra) and math.isfinite(dec) and math.isfinite(flux)):
                continue
            if flux <= 0.0:
                continue
            ra_list.append(ra)
            dec_list.append(dec)
            flux_list.append(flux)
    return np.asarray(ra_list, dtype=float), np.asarray(dec_list, dtype=float), np.asarray(flux_list, dtype=float)


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    w = 1.0 / np.asarray(sigma, dtype=float)
    xw = x * w[:, None]
    yw = y * w
    beta, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = y - x @ beta
    chi2 = float(np.sum((resid / sigma) ** 2))
    dof = int(y.size - x.shape[1])
    fisher = xw.T @ xw
    cov = np.linalg.inv(fisher)
    if dof > 0:
        cov = cov * max(1.0, chi2 / dof)
    return beta, cov, chi2, dof


def _unique_nearest(
    query_idx: np.ndarray,
    target_idx: np.ndarray,
    sep_arcsec: np.ndarray,
) -> np.ndarray:
    """Return mask over query rows keeping only nearest unique target matches."""
    order = np.argsort(sep_arcsec)
    target_seen: set[int] = set()
    keep = np.zeros(query_idx.size, dtype=bool)
    for j in order:
        t = int(target_idx[j])
        if t in target_seen:
            continue
        target_seen.add(t)
        keep[j] = True
    return keep


def _curve_from_flux_cuts(
    z_match: np.ndarray,
    s_match: np.ndarray,
    cuts: list[float],
    min_matches: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cut in sorted(cuts):
        m = s_match >= float(cut)
        n = int(np.sum(m))
        if n < int(min_matches):
            continue
        z_sub = z_match[m]
        z16, z50, z84 = [float(x) for x in np.percentile(z_sub, [16.0, 50.0, 84.0])]
        robust_sigma = 0.5 * max(0.0, z84 - z16)
        sigma_med = 1.253 * robust_sigma / math.sqrt(max(1, n))
        sigma_med = max(sigma_med, 1e-4)
        rows.append(
            {
                "cut_mjy": float(cut),
                "n_match": n,
                "z_p16": z16,
                "z_p50": z50,
                "z_p84": z84,
                "sigma_median_est": float(sigma_med),
            }
        )
    return rows


def _fit_log_model(curve: list[dict[str, Any]], cut_ref_mjy: float) -> dict[str, Any]:
    x = np.asarray([math.log(float(cut_ref_mjy) / float(r["cut_mjy"])) for r in curve], dtype=float)
    y = np.asarray([math.log(max(1e-6, float(r["z_p50"]))) for r in curve], dtype=float)
    sigma_z = np.asarray([max(1e-6, float(r["sigma_median_est"])) for r in curve], dtype=float)
    z_med = np.asarray([max(1e-6, float(r["z_p50"])) for r in curve], dtype=float)
    sigma_y = sigma_z / z_med
    design = np.column_stack([np.ones_like(x), x])
    beta, cov, chi2, dof = _weighted_lstsq(design, y, sigma_y)
    ln_z_ref = float(beta[0])
    eta = float(beta[1])
    ln_z_ref_sigma = float(math.sqrt(max(0.0, cov[0, 0])))
    eta_sigma = float(math.sqrt(max(0.0, cov[1, 1])))

    z_ref = float(math.exp(ln_z_ref))
    z_ref_sigma = float(z_ref * ln_z_ref_sigma)
    z_lo = max(0.01, z_ref - 5.0 * max(z_ref_sigma, 1e-4))
    z_hi = z_ref + 5.0 * max(z_ref_sigma, 1e-4)
    if z_hi <= z_lo:
        z_hi = z_lo + 0.1
    eta_lo = max(0.0, eta - 5.0 * max(eta_sigma, 1e-4))
    eta_hi = eta + 5.0 * max(eta_sigma, 1e-4)
    if eta_hi <= eta_lo:
        eta_hi = eta_lo + 0.1

    return {
        "z_ref": {"mu": z_ref, "sigma": z_ref_sigma, "bounds": [z_lo, z_hi]},
        "eta": {"mu": eta, "sigma": eta_sigma, "bounds": [eta_lo, eta_hi]},
        "fit_quality": {
            "chi2": float(chi2),
            "dof": int(dof),
            "chi2_red": float(chi2 / dof) if dof > 0 else float("nan"),
        },
        "log_model": {
            "ln_z_ref_mu": ln_z_ref,
            "ln_z_ref_sigma": ln_z_ref_sigma,
            "eta_mu": eta,
            "eta_sigma": eta_sigma,
        },
    }


def _extract_cuts(radio_audit_json: Path) -> dict[str, list[float]]:
    obj = json.loads(radio_audit_json.read_text())
    scans = obj.get("per_survey_flux_scans", {})
    out: dict[str, list[float]] = {}
    for survey, rows in scans.items():
        cuts = sorted({float(r["cut_mjy"]) for r in rows})
        out[str(survey)] = cuts
    return out


def _merge_base(base_path: Path | None) -> dict[str, Any]:
    if base_path is None:
        return {"schema_version": "1.0"}
    if not base_path.exists():
        return {"schema_version": "1.0"}
    return json.loads(base_path.read_text())


def _plot_survey_curve(
    out_png: Path,
    survey: str,
    curve: list[dict[str, Any]],
    cut_ref: float,
    fit: dict[str, Any],
) -> None:
    cuts = np.asarray([r["cut_mjy"] for r in curve], dtype=float)
    z50 = np.asarray([r["z_p50"] for r in curve], dtype=float)
    sig = np.asarray([r["sigma_median_est"] for r in curve], dtype=float)
    grid = np.linspace(float(np.min(cuts)), float(np.max(cuts)), 200)
    z_ref = float(fit["z_ref"]["mu"])
    eta = float(fit["eta"]["mu"])
    y_fit = z_ref * ((float(cut_ref) / grid) ** eta)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.errorbar(cuts, z50, yerr=sig, fmt="o", label="Crossmatch medians")
    ax.plot(grid, y_fit, lw=2.0, label="Log-linear fit")
    ax.set_xlabel("Flux cut [mJy]")
    ax.set_ylabel("Matched DR16Q median z")
    ax.set_title(f"{survey} redshift-vs-flux calibration")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / (
        args.run_tag if args.run_tag else f"shared_redshift_radio_calibration_{utc_tag()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dr16q_path = Path(args.dr16q_fits).resolve()
    audit_path = Path(args.radio_audit_json).resolve()
    cuts_by_survey = _extract_cuts(audit_path)

    with fits.open(dr16q_path, memmap=True) as hdul:
        cols = {c.name.upper(): c.name for c in hdul[1].columns}
        ra_key = cols.get(args.dr16q_ra_col.upper())
        dec_key = cols.get(args.dr16q_dec_col.upper())
        z_key = cols.get(args.dr16q_z_col.upper())
        if ra_key is None or dec_key is None or z_key is None:
            raise KeyError("DR16Q columns not found")
        d = hdul[1].data
        q_ra = np.asarray(d[ra_key], dtype=float)
        q_dec = np.asarray(d[dec_key], dtype=float)
        q_z = np.asarray(d[z_key], dtype=float)
    q_mask = np.isfinite(q_ra) & np.isfinite(q_dec) & np.isfinite(q_z)
    q_mask &= (q_z >= float(args.dr16q_z_min)) & (q_z <= float(args.dr16q_z_max))
    q_ra = q_ra[q_mask]
    q_dec = q_dec[q_mask]
    q_z = q_z[q_mask]
    q_coord = SkyCoord(ra=q_ra * u.deg, dec=q_dec * u.deg)

    surveys = {
        "NVSS": {
            "loader": lambda: _read_fits_cols(
                Path(args.nvss_fits), args.nvss_ra_col, args.nvss_dec_col, args.nvss_flux_col
            ),
            "match_arcsec": float(args.nvss_match_arcsec),
            "cut_ref_mjy": float(args.nvss_cut_ref_mjy),
        },
        "RACS-low": {
            "loader": lambda: _read_racs_csv(
                Path(args.racs_csv_gz), args.racs_ra_col, args.racs_dec_col, args.racs_flux_col
            ),
            "match_arcsec": float(args.racs_match_arcsec),
            "cut_ref_mjy": float(args.racs_cut_ref_mjy),
        },
        "LoTSS-DR2": {
            "loader": lambda: _read_fits_cols(
                Path(args.lotss_fits), args.lotss_ra_col, args.lotss_dec_col, args.lotss_flux_col
            ),
            "match_arcsec": float(args.lotss_match_arcsec),
            "cut_ref_mjy": float(args.lotss_cut_ref_mjy),
        },
    }

    radio_out: dict[str, Any] = {}
    survey_reports: dict[str, Any] = {}

    for survey, meta in surveys.items():
        ra, dec, flux = meta["loader"]()
        sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        idx, sep2d, _ = q_coord.match_to_catalog_sky(sc)
        sep_arcsec = sep2d.arcsec
        within = sep_arcsec <= float(meta["match_arcsec"])

        if np.any(within):
            q_idx = np.where(within)[0]
            t_idx = np.asarray(idx[within], dtype=int)
            sep_sel = np.asarray(sep_arcsec[within], dtype=float)
            keep = _unique_nearest(q_idx, t_idx, sep_sel)
            q_idx = q_idx[keep]
            t_idx = t_idx[keep]
            sep_sel = sep_sel[keep]
            z_match = q_z[q_idx]
            s_match = flux[t_idx]
        else:
            q_idx = np.asarray([], dtype=int)
            t_idx = np.asarray([], dtype=int)
            sep_sel = np.asarray([], dtype=float)
            z_match = np.asarray([], dtype=float)
            s_match = np.asarray([], dtype=float)

        cuts = cuts_by_survey.get(survey, [])
        curve = _curve_from_flux_cuts(
            z_match=z_match,
            s_match=s_match,
            cuts=cuts,
            min_matches=int(args.min_matches_per_cut),
        )
        if len(curve) < 2:
            survey_reports[survey] = {
                "status": "insufficient_matches",
                "n_match": int(z_match.size),
                "n_cuts_usable": len(curve),
                "cuts": cuts,
            }
            continue

        fit = _fit_log_model(curve=curve, cut_ref_mjy=float(meta["cut_ref_mjy"]))
        radio_out[survey] = {
            "z_ref": fit["z_ref"],
            "eta": fit["eta"],
            "fit_quality": fit["fit_quality"],
            "match_info": {
                "n_radio_rows": int(ra.size),
                "n_dr16q_rows": int(q_z.size),
                "n_matches_unique": int(z_match.size),
                "median_sep_arcsec": float(np.median(sep_sel)) if sep_sel.size else float("nan"),
                "match_arcsec": float(meta["match_arcsec"]),
                "cut_ref_mjy": float(meta["cut_ref_mjy"]),
            },
            "curve": curve,
        }
        survey_reports[survey] = {
            "status": "ok",
            "n_match": int(z_match.size),
            "n_cuts_usable": len(curve),
            "fit": fit,
        }
        _plot_survey_curve(
            out_png=out_dir / f"{survey}_redshift_flux_calibration.png",
            survey=survey,
            curve=curve,
            cut_ref=float(meta["cut_ref_mjy"]),
            fit=fit,
        )

        with (out_dir / f"{survey}_curve.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cut_mjy", "n_match", "z_p16", "z_p50", "z_p84", "sigma_median_est"])
            for r in curve:
                w.writerow([r["cut_mjy"], r["n_match"], r["z_p16"], r["z_p50"], r["z_p84"], r["sigma_median_est"]])

    combined = _merge_base(Path(args.base_calibration_json).resolve() if args.base_calibration_json else None)
    combined["created_utc"] = datetime.now(timezone.utc).isoformat()
    combined["radio"] = radio_out
    combined["source_radio_calibration"] = {
        "dr16q_fits": str(dr16q_path),
        "radio_audit_json": str(audit_path),
        "cuts_by_survey": cuts_by_survey,
    }

    out_cal = Path(args.out_calibration)
    out_cal.parent.mkdir(parents=True, exist_ok=True)
    out_cal.write_text(json.dumps(combined, indent=2) + "\n")

    (out_dir / "radio_calibration_only.json").write_text(
        json.dumps({"radio": radio_out, "survey_reports": survey_reports}, indent=2) + "\n"
    )
    (out_dir / "combined_calibration.json").write_text(json.dumps(combined, indent=2) + "\n")

    md = []
    md.append("# Radio Redshift Calibration via DR16Q Crossmatch")
    md.append("")
    md.append(f"- DR16Q source: `{dr16q_path}`")
    md.append(f"- Radio audit source: `{audit_path}`")
    md.append(f"- Output combined calibration: `{out_cal.resolve()}`")
    md.append("")
    for survey in ["NVSS", "RACS-low", "LoTSS-DR2"]:
        r = survey_reports.get(survey, {})
        if r.get("status") != "ok":
            md.append(f"- {survey}: insufficient matches (usable cuts={r.get('n_cuts_usable', 0)}).")
            continue
        f = r["fit"]
        md.append(
            f"- {survey}: `z_ref={f['z_ref']['mu']:.4f}±{f['z_ref']['sigma']:.4f}`, "
            f"`eta={f['eta']['mu']:.4f}±{f['eta']['sigma']:.4f}`, "
            f"`chi2_red={f['fit_quality']['chi2_red']:.4f}`, "
            f"`n_match={r['n_match']}`"
        )
    (out_dir / "master_report.md").write_text("\n".join(md) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "out_calibration": str(out_cal.resolve()),
                "surveys_calibrated": sorted(radio_out.keys()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

