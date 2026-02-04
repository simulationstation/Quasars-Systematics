#!/usr/bin/env python3
"""
All-sky external validation (Gaia DR3 QSO candidates) for the CatWISE completeness model.

Goal
----
Use an all-sky, external quasar-like sample (Gaia DR3 QSO candidates; CDS I/356 "qsocand.dat")
to validate *spatial* structure in CatWISE "accepted" selection on the Secrest+22 footprint.

We compute, per HEALPix pixel:
  - N_gaia(pix): number of Gaia QSO candidates passing a PQSO threshold in the footprint
  - N_match(pix): number of those with a CatWISE "accepted" match within a radius
  - f_accept(pix) = N_match / N_gaia

We then compare f_accept against:
  - unWISE logNexp exposure-count depth proxy (map-level; independent of catalog counts)
  - the SDSS DR16Q–trained depth-only δm(pix) map (from the external completeness model)

Important limitation
--------------------
Gaia qsocand does not provide W1; this is a *spatial* validation of predicted depth-linked
selection patterns, not a full magnitude-dependent completeness calibration.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

# ICRS/J2000 -> Galactic rotation matrix (same numerical values used widely, e.g. Hipparcos/IAU)
_R_EQ_TO_GAL = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=float,
)


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def unitvec_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float) % 360.0)
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cosd = np.cos(dec)
    return np.column_stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)]).astype(float)


def gal_ipix_from_unitvec_eq(v_eq: np.ndarray, *, nside: int) -> np.ndarray:
    import healpy as hp

    v_eq = np.asarray(v_eq, dtype=float)
    v_gal = v_eq @ _R_EQ_TO_GAL.T
    lon = np.arctan2(v_gal[:, 1], v_gal[:, 0]) % (2.0 * math.pi)
    lat = np.arcsin(np.clip(v_gal[:, 2], -1.0, 1.0))
    theta = (0.5 * math.pi) - lat
    phi = lon
    return hp.ang2pix(int(nside), theta, phi, nest=False)


def find_first_col(names: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def safe_float_col(arr: object) -> np.ndarray:
    return np.asarray(arr, dtype=float)


def build_secrest_mask(
    *,
    nside: int,
    ipix_base: np.ndarray,
    exclude_mask_fits: str | None,
    b_cut_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Secrest-style HEALPix footprint mask (mask_zeros + exclude discs + |b| cut).

    Returns (mask, seen) where mask=True means masked and seen=True means unmasked.
    """
    import healpy as hp

    npix = hp.nside2npix(int(nside))
    mask = np.zeros(npix, dtype=bool)

    # mask_zeros on the "base" W1cov>=80 parent map
    cnt_base = np.bincount(np.asarray(ipix_base, dtype=np.int64), minlength=npix)
    idx0 = np.where(cnt_base == 0)[0]
    if idx0.size:
        neigh = np.empty((idx0.size, 8), dtype=int)
        for i, ip in enumerate(idx0):
            neigh[i] = hp.pixelfunc.get_all_neighbours(int(nside), int(ip))
        mask[idx0] = True
        # Match Secrest behaviour (includes -1 neighbour indexing last pixel).
        mask[neigh] = True

    # exclude discs
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
                disc = hp.query_disc(
                    nside=int(nside),
                    vec=vec,
                    radius=float(rad),
                    inclusive=True,
                    nest=False,
                )
                mask[disc] = True

    # galactic plane cut on pixel centers
    _, lat_pix = hp.pix2ang(int(nside), np.arange(npix), lonlat=True)
    mask |= np.abs(lat_pix) < float(b_cut_deg)

    return mask, ~mask


def _ensure_nside(map_in: np.ndarray, *, nside: int) -> np.ndarray:
    import healpy as hp

    m = np.asarray(map_in, dtype=float)
    if int(hp.get_nside(m)) != int(nside):
        m = hp.ud_grade(m, nside_out=int(nside), order_in="RING", order_out="RING", power=0)
    return np.asarray(m, dtype=float)


@dataclass(frozen=True)
class GaiaParseConfig:
    # CDS ReadMe: Byte-by-byte Description of file: qsocand.dat
    # Bytes are 1-based and inclusive. Python slices are 0-based and end-exclusive.
    pqso_slice: slice = slice(186, 200)  # Bytes 187-200
    ra_slice: slice = slice(948, 971)  # Bytes 949-971
    dec_slice: slice = slice(972, 995)  # Bytes 973-995


def _parse_float_field(line: str, s: slice) -> float | None:
    txt = line[s].strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--qsocand-gz",
        default="data/external/gaia_dr3_extragal/qsocand.dat.gz",
        help="CDS I/356 qsocand.dat.gz (Gaia DR3 QSO candidates).",
    )
    ap.add_argument(
        "--catwise-catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits",
    )
    ap.add_argument(
        "--unwise-lognexp-map",
        default="data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits",
        help="unWISE logNexp HEALPix map (Galactic coords).",
    )
    ap.add_argument(
        "--sdss-delta-m-map",
        default="REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits",
        help="SDSS DR16Q–trained depth-only δm map (tracked in REPORTS).",
    )
    ap.add_argument(
        "--dr16q-fits",
        default="data/external/sdss_dr16q/DR16Q_v4.fits",
        help="Optional: DR16Q FITS used only to define a coarse SDSS footprint map for inside/outside comparisons.",
    )
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--match-radius-arcsec", type=float, default=2.0)
    ap.add_argument("--pqso-min", type=float, default=0.8)
    ap.add_argument(
        "--min-gaia-per-pix",
        type=int,
        default=20,
        help="Minimum Gaia-QSO-candidate count per pixel to include in correlation/quantile diagnostics.",
    )
    ap.add_argument("--chunk-lines", type=int, default=250000, help="Chunk size in kept (PQSO-filtered) rows.")
    ap.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap on input lines read (for smoke tests).",
    )
    ap.add_argument("--n-records", type=int, default=6649162, help="Expected record count for percent progress.")
    ap.add_argument("--workers", type=int, default=-1, help="cKDTree query workers (threads).")
    ap.add_argument("--progress-every", type=int, default=250000, help="Progress print cadence (input lines).")
    ap.add_argument("--outdir", default=None, help="Output directory for this run (defaults to outputs/*).")
    ap.add_argument("--export-report-dir", default=None, help="If set, copy final artifacts into this report folder.")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    from astropy.io import fits
    import healpy as hp
    from scipy.spatial import cKDTree

    qsocand_path = Path(str(args.qsocand_gz))
    if not qsocand_path.exists():
        raise SystemExit(
            "Missing Gaia qsocand.dat.gz.\n\n"
            "Download (resumable) from CDS:\n"
            "  https://cdsarc.cds.unistra.fr/ftp/I/356/qsocand.dat.gz\n\n"
            f"Expected at: {qsocand_path}"
        )

    outdir = Path(args.outdir or f"outputs/external_validation_gaia_qsocand_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------- Load CatWISE + build Secrest footprint --------------------
    with fits.open(str(args.catwise_catalog), memmap=True) as hdul:
        c = hdul[1].data
        cat_ra = safe_float_col(c["ra"])
        cat_dec = safe_float_col(c["dec"])
        cat_l = safe_float_col(c["l"])
        cat_b = safe_float_col(c["b"])
        cat_w1cov = safe_float_col(c["w1cov"])

    cat_base = (
        np.isfinite(cat_ra)
        & np.isfinite(cat_dec)
        & np.isfinite(cat_l)
        & np.isfinite(cat_b)
        & np.isfinite(cat_w1cov)
        & (cat_w1cov >= float(args.w1cov_min))
    )
    theta_base = np.deg2rad(90.0 - cat_b[cat_base])
    phi_base = np.deg2rad(cat_l[cat_base])
    ipix_cat_base = hp.ang2pix(int(args.nside), theta_base, phi_base, nest=False)
    _, seen = build_secrest_mask(
        nside=int(args.nside),
        ipix_base=ipix_cat_base,
        exclude_mask_fits=str(args.exclude_mask_fits) if args.exclude_mask_fits else None,
        b_cut_deg=float(args.b_cut),
    )

    # For matching: restrict CatWISE points to the seen footprint + w1cov cut (keeps tree smaller)
    ipix_cat_all = hp.ang2pix(
        int(args.nside),
        np.deg2rad(90.0 - cat_b),
        np.deg2rad(cat_l),
        nest=False,
    )
    cat_keep = cat_base & seen[ipix_cat_all]
    cat_ra_k = cat_ra[cat_keep]
    cat_dec_k = cat_dec[cat_keep]

    cat_v = unitvec_from_radec_deg(cat_ra_k, cat_dec_k)
    tree = cKDTree(cat_v)

    theta_max = np.deg2rad(float(args.match_radius_arcsec) / 3600.0)
    dmax = float(2.0 * np.sin(theta_max / 2.0))

    npix = hp.nside2npix(int(args.nside))
    gaia_total = np.zeros(npix, dtype=np.int64)
    gaia_match = np.zeros(npix, dtype=np.int64)

    parse_cfg = GaiaParseConfig()

    # -------------------- Stream parse + match Gaia qsocand --------------------
    kept_ra: list[float] = []
    kept_dec: list[float] = []

    n_lines_read = 0
    n_lines_kept = 0

    def process_chunk(ra_chunk: np.ndarray, dec_chunk: np.ndarray) -> None:
        nonlocal gaia_total, gaia_match, n_lines_kept

        if ra_chunk.size == 0:
            return

        v_eq = unitvec_from_radec_deg(ra_chunk, dec_chunk)
        ipix = gal_ipix_from_unitvec_eq(v_eq, nside=int(args.nside))
        keep = seen[ipix]
        if not np.any(keep):
            return

        v_eq = v_eq[keep]
        ipix = ipix[keep]

        dist, _ = tree.query(v_eq, k=1, workers=int(args.workers))
        dist = np.asarray(dist, dtype=float)
        matched = dist <= dmax

        gaia_total += np.bincount(ipix, minlength=npix)
        gaia_match += np.bincount(ipix, weights=matched.astype(int), minlength=npix).astype(np.int64)
        n_lines_kept += int(ra_chunk.size)

    with gzip.open(str(qsocand_path), mode="rt", encoding="ascii", errors="ignore") as f:
        for line in f:
            n_lines_read += 1
            if args.max_lines is not None and n_lines_read > int(args.max_lines):
                break
            if len(line) < 995:
                continue

            pqso = _parse_float_field(line, parse_cfg.pqso_slice)
            if pqso is None or not np.isfinite(pqso) or pqso < float(args.pqso_min):
                continue

            ra = _parse_float_field(line, parse_cfg.ra_slice)
            dec = _parse_float_field(line, parse_cfg.dec_slice)
            if ra is None or dec is None:
                continue
            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue

            kept_ra.append(float(ra))
            kept_dec.append(float(dec))

            if len(kept_ra) >= int(args.chunk_lines):
                process_chunk(np.asarray(kept_ra, dtype=float), np.asarray(kept_dec, dtype=float))
                kept_ra.clear()
                kept_dec.clear()

            if (n_lines_read % int(args.progress_every)) == 0:
                denom = float(args.max_lines or args.n_records or 1)
                frac = min(1.0, float(n_lines_read) / denom) if denom else 0.0
                print(
                    f"[gaia-qsocand] lines={n_lines_read:,} "
                    f"kept={n_lines_kept:,} "
                    f"progress={100.0 * frac:5.1f}%",
                    flush=True,
                )

    # Final chunk
    process_chunk(np.asarray(kept_ra, dtype=float), np.asarray(kept_dec, dtype=float))

    # -------------------- Build maps --------------------
    accept_frac = np.full(npix, hp.UNSEEN, dtype=float)
    ok = gaia_total > 0
    accept_frac[ok] = gaia_match[ok] / gaia_total[ok]

    hp.write_map(str(outdir / "gaia_qso_total_nside64.fits"), gaia_total.astype(float), overwrite=True, dtype=float)
    hp.write_map(str(outdir / "gaia_qso_match_nside64.fits"), gaia_match.astype(float), overwrite=True, dtype=float)
    hp.write_map(str(outdir / "gaia_qso_accept_frac_nside64.fits"), accept_frac, overwrite=True, dtype=float)

    # -------------------- Optional SDSS footprint map (for inside/outside diagnostics) --------------------
    dr16q_cnt = None
    dr16q_path = Path(str(args.dr16q_fits)) if args.dr16q_fits else None
    if dr16q_path and dr16q_path.exists():
        with fits.open(str(dr16q_path), memmap=True) as hdul:
            t = hdul[1].data
            names = list(t.names)
            ra_col = find_first_col(names, ["ra", "RA", "RAJ2000"])
            dec_col = find_first_col(names, ["dec", "DEC", "DEJ2000"])
            if ra_col and dec_col:
                ra = safe_float_col(t[ra_col])
                dec = safe_float_col(t[dec_col])
                ok2 = np.isfinite(ra) & np.isfinite(dec)
                v_eq = unitvec_from_radec_deg(ra[ok2], dec[ok2])
                ipix = gal_ipix_from_unitvec_eq(v_eq, nside=int(args.nside))
                ipix = ipix[seen[ipix]]
                dr16q_cnt = np.bincount(ipix.astype(np.int64), minlength=npix).astype(np.int64)
                hp.write_map(
                    str(outdir / "sdss_dr16q_count_nside64.fits"),
                    dr16q_cnt.astype(float),
                    overwrite=True,
                    dtype=float,
                )

    # -------------------- Correlation diagnostics --------------------
    lognexp = _ensure_nside(hp.read_map(str(args.unwise_lognexp_map), verbose=False), nside=int(args.nside))
    delta_m = _ensure_nside(hp.read_map(str(args.sdss_delta_m_map), verbose=False), nside=int(args.nside))

    def is_seen_finite(m: np.ndarray) -> np.ndarray:
        return seen & np.isfinite(m) & (m != hp.UNSEEN)

    ok_pix = (
        is_seen_finite(lognexp)
        & is_seen_finite(delta_m)
        & (gaia_total >= int(args.min_gaia_per_pix))
        & np.isfinite(accept_frac)
        & (accept_frac != hp.UNSEEN)
    )

    def spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        from scipy.stats import spearmanr

        r, p = spearmanr(x, y)
        return {"rho": float(r), "p_value": float(p)}

    corr_all = {
        "accept_frac_vs_lognexp": spearman(lognexp[ok_pix], accept_frac[ok_pix]),
        "accept_frac_vs_delta_m": spearman(delta_m[ok_pix], accept_frac[ok_pix]),
    }

    corr_in = corr_out = None
    if dr16q_cnt is not None:
        in_sdss = ok_pix & (dr16q_cnt > 0)
        out_sdss = ok_pix & (dr16q_cnt == 0)
        if np.any(in_sdss):
            corr_in = {
                "accept_frac_vs_lognexp": spearman(lognexp[in_sdss], accept_frac[in_sdss]),
                "accept_frac_vs_delta_m": spearman(delta_m[in_sdss], accept_frac[in_sdss]),
                "n_pix": int(np.sum(in_sdss)),
            }
        if np.any(out_sdss):
            corr_out = {
                "accept_frac_vs_lognexp": spearman(lognexp[out_sdss], accept_frac[out_sdss]),
                "accept_frac_vs_delta_m": spearman(delta_m[out_sdss], accept_frac[out_sdss]),
                "n_pix": int(np.sum(out_sdss)),
            }
    else:
        in_sdss = out_sdss = None

    meta = {
        "gaia_qsocand": {
            "path": str(qsocand_path),
            "pqso_min": float(args.pqso_min),
            "lines_read": int(n_lines_read),
            "lines_kept_pqso": int(n_lines_kept),
        },
        "catwise": {
            "catalog": str(args.catwise_catalog),
            "exclude_mask_fits": str(args.exclude_mask_fits),
            "w1cov_min": float(args.w1cov_min),
            "b_cut_deg": float(args.b_cut),
            "n_catwise_in_tree": int(len(cat_ra_k)),
        },
        "match": {"radius_arcsec": float(args.match_radius_arcsec), "workers": int(args.workers)},
        "maps": {
            "nside": int(args.nside),
            "unwise_lognexp_map": str(args.unwise_lognexp_map),
            "sdss_delta_m_map": str(args.sdss_delta_m_map),
            "sdss_dr16q_fits_for_footprint": None if dr16q_path is None else str(dr16q_path),
        },
        "analysis": {"min_gaia_per_pix": int(args.min_gaia_per_pix), "n_pix_used": int(np.sum(ok_pix))},
        "spearman": {"all": corr_all, "in_sdss": corr_in, "out_sdss": corr_out},
        "outputs": {
            "gaia_qso_total_map": "gaia_qso_total_nside64.fits",
            "gaia_qso_match_map": "gaia_qso_match_nside64.fits",
            "gaia_qso_accept_frac_map": "gaia_qso_accept_frac_nside64.fits",
        },
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    # -------------------- Plots --------------------
    if args.make_plots:
        import matplotlib.pyplot as plt

        def plot_moll(m: np.ndarray, *, title: str, unit: str, fname: str, vmin=None, vmax=None) -> None:
            plt.figure(figsize=(10.0, 4.0))
            hp.mollview(
                m,
                title=title,
                unit=unit,
                min=vmin,
                max=vmax,
                fig=plt.gcf().number,
            )
            plt.savefig(outdir / fname, dpi=200)
            plt.close()

        plot_moll(
            accept_frac,
            title=f"Gaia QSOs: CatWISE acceptance fraction (PQSO>={args.pqso_min})",
            unit="fraction",
            fname="gaia_accept_frac_mollweide.png",
            vmin=0.0,
            vmax=1.0,
        )
        plot_moll(
            delta_m,
            title="SDSS DR16Q depth-only δm map (external completeness)",
            unit="mag",
            fname="sdss_delta_m_mollweide.png",
        )
        if dr16q_cnt is not None:
            dr16q_map = np.full(npix, hp.UNSEEN, dtype=float)
            dr16q_map[seen] = dr16q_cnt[seen].astype(float)
            plot_moll(
                dr16q_map,
                title="SDSS DR16Q footprint proxy (counts per pixel)",
                unit="count",
                fname="sdss_dr16q_count_mollweide.png",
            )

        def quantile_binned_curve(
            x_map: np.ndarray,
            *,
            label: str,
            subset: np.ndarray,
            nbins: int = 20,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = x_map[subset]
            q = np.quantile(x, np.linspace(0.0, 1.0, nbins + 1))
            q[0] -= 1e-12
            q[-1] += 1e-12
            centers: list[float] = []
            pvals: list[float] = []
            perr: list[float] = []
            for lo, hi in zip(q[:-1], q[1:], strict=True):
                m = subset & (x_map >= lo) & (x_map < hi)
                n_tot = int(np.sum(gaia_total[m]))
                n_hit = int(np.sum(gaia_match[m]))
                if n_tot <= 0:
                    continue
                p = n_hit / n_tot
                err = math.sqrt(max(0.0, p * (1.0 - p) / n_tot))
                centers.append(float(np.median(x_map[m])))
                pvals.append(float(p))
                perr.append(float(err))
            return np.asarray(centers), np.asarray(pvals), np.asarray(perr)

        # Single figure with inside/outside SDSS if available.
        def make_binned_plot(x_map: np.ndarray, *, xlabel: str, base: np.ndarray, fname: str) -> None:
            plt.figure(figsize=(6.0, 4.0))
            if dr16q_cnt is None:
                xc, p, e = quantile_binned_curve(x_map, label="all", subset=base)
                plt.errorbar(xc, p, yerr=e, fmt="o-", label="all pixels")
            else:
                xc, p, e = quantile_binned_curve(x_map, label="in_sdss", subset=base & (dr16q_cnt > 0))
                if xc.size:
                    plt.errorbar(xc, p, yerr=e, fmt="o-", label="inside SDSS footprint")
                xc, p, e = quantile_binned_curve(x_map, label="out_sdss", subset=base & (dr16q_cnt == 0))
                if xc.size:
                    plt.errorbar(xc, p, yerr=e, fmt="o-", label="outside SDSS footprint")
            plt.xlabel(xlabel)
            plt.ylabel("CatWISE acceptance fraction (Gaia QSOs)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / fname, dpi=200)
            plt.close()

        make_binned_plot(delta_m, xlabel="SDSS DR16Q depth-only δm (mag)", base=ok_pix, fname="accept_frac_vs_delta_m.png")
        make_binned_plot(lognexp, xlabel="unWISE logNexp", base=ok_pix, fname="accept_frac_vs_lognexp.png")

    # -------------------- Optional export to report dir --------------------
    if args.export_report_dir:
        report_dir = Path(str(args.export_report_dir))
        (report_dir / "data").mkdir(parents=True, exist_ok=True)
        (report_dir / "figures").mkdir(parents=True, exist_ok=True)

        (report_dir / "data" / "meta.json").write_text((outdir / "meta.json").read_text())
        for fn in [
            "gaia_qso_total_nside64.fits",
            "gaia_qso_match_nside64.fits",
            "gaia_qso_accept_frac_nside64.fits",
        ]:
            (report_dir / "data" / fn).write_bytes((outdir / fn).read_bytes())
        if (outdir / "sdss_dr16q_count_nside64.fits").exists():
            (report_dir / "data" / "sdss_dr16q_count_nside64.fits").write_bytes(
                (outdir / "sdss_dr16q_count_nside64.fits").read_bytes()
            )
        for fn in [
            "gaia_accept_frac_mollweide.png",
            "sdss_delta_m_mollweide.png",
            "sdss_dr16q_count_mollweide.png",
            "accept_frac_vs_delta_m.png",
            "accept_frac_vs_lognexp.png",
        ]:
            p = outdir / fn
            if p.exists():
                (report_dir / "figures" / fn).write_bytes(p.read_bytes())

    print(json.dumps(meta["spearman"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
