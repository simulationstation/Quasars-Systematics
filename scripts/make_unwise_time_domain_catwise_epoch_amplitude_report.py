#!/usr/bin/env python3
"""Generate a REPORTS bundle for the CatWISE-parent unWISE time-domain epoch amplitude test."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_secrest_seen_mask(
    *,
    nside: int,
    catwise_catalog: Path,
    exclude_mask_fits: Path,
    b_cut_deg: float,
    w1cov_min: float,
) -> np.ndarray:
    import healpy as hp
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    npix = hp.nside2npix(int(nside))

    with fits.open(str(catwise_catalog), memmap=True) as hdul:
        d = hdul[1].data
        w1cov = np.asarray(d["w1cov"], dtype=float)
        l = np.asarray(d["l"], dtype=float)
        b = np.asarray(d["b"], dtype=float)

    sel = np.isfinite(w1cov) & np.isfinite(l) & np.isfinite(b) & (w1cov >= float(w1cov_min))
    ipix_base = hp.ang2pix(
        int(nside),
        np.deg2rad(90.0 - b[sel]),
        np.deg2rad(l[sel] % 360.0),
        nest=False,
    ).astype(np.int64)

    cnt = np.bincount(ipix_base, minlength=npix)
    mask = np.zeros(npix, dtype=bool)
    idx0 = np.where(cnt == 0)[0]
    if idx0.size:
        neigh = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            neigh[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        mask[neigh] = True

    tmask = Table.read(str(exclude_mask_fits), memmap=True)
    if "use" in tmask.colnames:
        tmask = tmask[np.asarray(tmask["use"], dtype=bool)]
    if len(tmask):
        sc = SkyCoord(tmask["ra"], tmask["dec"], unit=u.deg, frame="icrs").galactic
        radius = np.deg2rad(np.asarray(tmask["radius"], dtype=float))
        for lon, lat, rad in zip(sc.l.deg, sc.b.deg, radius, strict=True):
            vec = hp.ang2vec(np.deg2rad(90.0 - float(lat)), np.deg2rad(float(lon)))
            disc = hp.query_disc(int(nside), vec=vec, radius=float(rad), inclusive=True, nest=False)
            mask[disc] = True

    _lon_pix, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return ~mask


def vector_sum_amplitude(counts_seen: np.ndarray, nhat_seen: np.ndarray) -> float:
    y = np.asarray(counts_seen, dtype=np.float64)
    N = float(np.sum(y))
    if not np.isfinite(N) or N <= 0.0:
        return float("nan")
    sum_vec = np.sum(nhat_seen * y[:, None], axis=0)
    return 3.0 * float(np.linalg.norm(sum_vec)) / N


def _stats(vals: list[float]) -> dict[str, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"min": float("nan"), "max": float("nan"), "range": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(v.min()),
        "max": float(v.max()),
        "range": float(v.max() - v.min()),
        "mean": float(v.mean()),
        "std": float(v.std()),
    }


@dataclass(frozen=True)
class EpochRow:
    epoch: int
    mjd_mean: float
    date_utc: str
    n_selected: int
    D_glm: float
    sigma_D_glm: float
    D_vecsum: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "mjd_mean": float(self.mjd_mean),
            "date_utc": str(self.date_utc),
            "N": int(self.n_selected),
            "D_glm": float(self.D_glm),
            "sigma_D_glm": float(self.sigma_D_glm),
            "D_vecsum": float(self.D_vecsum),
        }


def write_master_report(report_dir: Path, out_json: dict) -> None:
    fig_dir = report_dir / "figures"
    stats = out_json["stats"]
    cfg = out_json["config"]
    pm = cfg.get("parent_meta", {})

    # Keep this short; the tables/figures carry details.
    text = f"""# CatWISE-parent unWISE time-domain epoch-resolved dipole amplitude

Date: {utc_date()} (UTC)

This report summarizes an **epoch-resolved, amplitude-only** dipole stability test using the
unWISE Time-Domain Catalog (IRSA580) **restricted to the published CatWISE/Secrest accepted AGN parent sample**.

If the dipole amplitude is cosmological/kinematic, it should be stable across epochs. If it is selection-driven,
it can vary with epoch/coverage/background.

## Exact definition

Parent catalog:
- Secrest+22 accepted CatWISE AGN catalog (Zenodo 6784602), filtered to:
  - `W1 <= {pm.get('w1_max', 'NA')}` (Vega)
  - `W1-W2 >= {pm.get('w1w2_min', 'NA')}` (Vega)
  - `W1cov >= {pm.get('w1cov_min', 'NA')}`
  - Secrest-style footprint mask (`mask_zeros` + exclusion discs + `|b| > {pm.get('b_cut_deg', 'NA')}°`)
- Parent size: `N_parent = {pm.get('n_parent', 'NA')}`

Epoch selection (time-domain):
- Object must have a matched (W1,W2) time-domain measurement in that epoch passing:
  - `primary==1`, `flags_unwise==0`, `flags_info==0`, `flux>0`, `dflux>0`
  - `W1 <= {cfg.get('w1_max_vega','NA')}` via flux threshold
  - `SNR_W1 >= {cfg.get('snr_w1_min','NA')}`; `SNR_W2 >= {cfg.get('snr_w2_min','NA')}`
  - `apply_color_cut = {cfg.get('apply_color_cut', False)}`
- Matching to parent uses a `match_radius = {cfg.get('match_radius_arcsec','NA')} arcsec` nearest-neighbor in ICRS.

Footprint mask:
- Fixed across epochs (same mask as the parent definition), HEALPix `nside={cfg['nside']}`, Galactic, RING.

Estimator:
- Primary: Poisson GLM dipole amplitude `D = |b|` on masked HEALPix maps.
- Cross-check: vector-sum amplitude on the same masked maps.

## Headline results (epochs 0–15)

Poisson GLM amplitude:
- `D_min = {stats['D_glm_epochs_0_15']['min']:.5f}`
- `D_max = {stats['D_glm_epochs_0_15']['max']:.5f}`
- `D_range = {stats['D_glm_epochs_0_15']['range']:.5f}`

Vector-sum amplitude:
- `D_min = {stats['D_vecsum_epochs_0_15']['min']:.5f}`
- `D_max = {stats['D_vecsum_epochs_0_15']['max']:.5f}`
- `D_range = {stats['D_vecsum_epochs_0_15']['range']:.5f}`

Per-epoch sample size (0–15):
- `N_min = {stats['N_epochs_0_15']['min']:.3g}`
- `N_max = {stats['N_epochs_0_15']['max']:.3g}`

## Figures

![]({(fig_dir / 'D_vs_epoch_glm.png').as_posix()})

![]({(fig_dir / 'D_vs_epoch_compare.png').as_posix()})

![]({(fig_dir / 'N_vs_epoch.png').as_posix()})

## Reproduce

Run directory:
- `{out_json['meta']['run_outdir']}`

Report directory:
- `{out_json['meta']['report_dir']}`
"""
    (report_dir / "master_report.md").write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-outdir",
        required=True,
        help="Finished run directory from scripts/run_unwise_time_domain_catwise_epoch_dipole.py",
    )
    ap.add_argument(
        "--report-dir",
        default="REPORTS/unwise_time_domain_catwise_epoch_amplitude",
        help="Report folder root (writes data/ and figures/).",
    )
    ap.add_argument(
        "--catwise-catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    args = ap.parse_args()

    run_outdir = Path(args.run_outdir)
    report_dir = Path(args.report_dir)
    data_dir = report_dir / "data"
    fig_dir = report_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run_outdir / "run_config.json").read_text())
    summ = json.loads((run_outdir / "summary.json").read_text())
    epochs = summ["epochs"]
    counts = np.load(run_outdir / "counts_by_epoch.npy")

    import healpy as hp
    from astropy.time import Time

    nside = int(cfg["nside"])
    npix = hp.nside2npix(nside)

    seen = build_secrest_seen_mask(
        nside=nside,
        catwise_catalog=Path(args.catwise_catalog),
        exclude_mask_fits=Path(args.exclude_mask_fits),
        b_cut_deg=float(cfg["b_cut_deg"]),
        w1cov_min=float(cfg["w1cov_min"]),
    )

    xpix, ypix, zpix = hp.pix2vec(nside, np.arange(npix), nest=False)
    nhat_seen = np.column_stack([xpix, ypix, zpix])[seen]

    rows: list[EpochRow] = []
    for r in epochs:
        e = int(r["epoch"])
        mjd = float(r["mjd_mean"])
        date_utc = Time(mjd, format="mjd").utc.isot
        n_selected = int(r["N"])
        D_glm = float(r["D"])
        sigma_D_glm = float(r["sigma_D"])

        y_seen = counts[e][seen]
        D_vec = vector_sum_amplitude(y_seen, nhat_seen)

        rows.append(
            EpochRow(
                epoch=e,
                mjd_mean=mjd,
                date_utc=date_utc,
                n_selected=n_selected,
                D_glm=D_glm,
                sigma_D_glm=sigma_D_glm,
                D_vecsum=D_vec,
            )
        )

    rows_015 = [rr for rr in rows if rr.epoch <= 15]
    stats = {
        "D_glm_epochs_0_15": _stats([rr.D_glm for rr in rows_015]),
        "D_vecsum_epochs_0_15": _stats([rr.D_vecsum for rr in rows_015]),
        "N_epochs_0_15": _stats([float(rr.n_selected) for rr in rows_015]),
    }

    out_json = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "date_utc": utc_date(),
            "run_outdir": str(run_outdir),
            "report_dir": str(report_dir),
        },
        "config": cfg,
        "stats": stats,
        "epochs": [rr.as_dict() for rr in rows],
    }

    (data_dir / "epoch_amplitude.json").write_text(json.dumps(out_json, indent=2) + "\n")

    csv_path = data_dir / "epoch_amplitude_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["epoch", "mjd_mean", "date_utc", "N", "D_glm", "sigma_D_glm", "D_vecsum"],
        )
        w.writeheader()
        for rr in rows:
            w.writerow(rr.as_dict())

    # Figures
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mjd = np.array([rr.mjd_mean for rr in rows], dtype=float)
        d_glm = np.array([rr.D_glm for rr in rows], dtype=float)
        sd_glm = np.array([rr.sigma_D_glm for rr in rows], dtype=float)
        d_vec = np.array([rr.D_vecsum for rr in rows], dtype=float)
        nsel = np.array([rr.n_selected for rr in rows], dtype=float)
        epoch = np.array([rr.epoch for rr in rows], dtype=int)

        ok = np.isfinite(mjd) & np.isfinite(d_glm)

        plt.figure(figsize=(7.8, 4.6))
        plt.errorbar(mjd[ok], d_glm[ok], yerr=sd_glm[ok], fmt="o-", lw=1.2, ms=4)
        for x, y, e in zip(mjd[ok], d_glm[ok], epoch[ok], strict=True):
            plt.text(x, y, str(int(e)), fontsize=8, ha="center", va="bottom")
        plt.xlabel("Mean MJD (per epoch)")
        plt.ylabel("Dipole amplitude D (Poisson GLM)")
        plt.title("CatWISE-parent: epoch-resolved dipole amplitude")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(fig_dir / "D_vs_epoch_glm.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7.8, 4.6))
        plt.plot(mjd[ok], d_glm[ok], "o-", lw=1.2, ms=4, label="Poisson GLM")
        ok2 = np.isfinite(mjd) & np.isfinite(d_vec)
        plt.plot(mjd[ok2], d_vec[ok2], "s-", lw=1.1, ms=3.5, label="Vector-sum")
        plt.xlabel("Mean MJD (per epoch)")
        plt.ylabel("Dipole amplitude")
        plt.title("Epoch-resolved amplitude: estimator cross-check")
        plt.grid(True, alpha=0.35)
        plt.legend(loc="best", frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "D_vs_epoch_compare.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7.8, 4.6))
        plt.plot(mjd, nsel, "o-", lw=1.2, ms=4)
        for x, y, e in zip(mjd, nsel, epoch, strict=True):
            plt.text(x, y, str(int(e)), fontsize=8, ha="center", va="bottom")
        plt.yscale("log")
        plt.xlabel("Mean MJD (per epoch)")
        plt.ylabel("Selected parent objects per epoch (log)")
        plt.title("Per-epoch sample size (CatWISE parent)")
        plt.grid(True, alpha=0.35, which="both")
        plt.tight_layout()
        plt.savefig(fig_dir / "N_vs_epoch.png", dpi=180)
        plt.close()

        m015 = epoch <= 15
        plt.figure(figsize=(5.2, 5.0))
        plt.scatter(d_glm[m015], d_vec[m015], s=28)
        for x, y, e in zip(d_glm[m015], d_vec[m015], epoch[m015], strict=True):
            plt.text(x, y, str(int(e)), fontsize=8, ha="left", va="bottom")
        lo = float(min(np.nanmin(d_glm[m015]), np.nanmin(d_vec[m015])))
        hi = float(max(np.nanmax(d_glm[m015]), np.nanmax(d_vec[m015])))
        pad = 0.02 * (hi - lo) if hi > lo else 0.01
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.4)
        plt.xlim(lo - pad, hi + pad)
        plt.ylim(lo - pad, hi + pad)
        plt.xlabel("D (Poisson GLM)")
        plt.ylabel("D (Vector-sum)")
        plt.title("Estimator cross-check (epochs 0–15)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "D_glm_vs_vecsum.png", dpi=180)
        plt.close()

    except Exception as e:  # noqa: BLE001
        print(f"Plotting failed: {type(e).__name__}: {e}")

    write_master_report(report_dir, out_json)

    print(f"Wrote report -> {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

