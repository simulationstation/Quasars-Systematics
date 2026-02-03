from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
import time
from pathlib import Path
from typing import Literal

import numpy as np

from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em


@dataclass(frozen=True)
class O3InjectionSet:
    """Minimal injection set for computing a distance-sensitive selection proxy alpha(model).

    Supports both the older GWTC-3 population injection release (O3a BBHpop) and the
    GWTC-3-era O3 search-sensitivity injection summaries (LIGO-T2100113).
    """

    path: str
    ifar_threshold_yr: float
    z: np.ndarray  # (N,)
    dL_mpc_fid: np.ndarray  # (N,) fiducial luminosity distance used for injection
    snr_net_opt: np.ndarray  # (N,) optimal network SNR (H+L)
    found_ifar: np.ndarray  # (N,) boolean
    sampling_pdf: np.ndarray  # (N,) injection sampling pdf (not used by default)
    mixture_weight: np.ndarray  # (N,) additional weight for mixture-model injection sets (defaults to 1)
    m1_source: np.ndarray  # (N,) Msun
    m2_source: np.ndarray  # (N,) Msun
    ra_rad: np.ndarray | None  # (N,) optional
    dec_rad: np.ndarray | None  # (N,) optional
    total_generated: int
    analysis_time_s: float


def load_o3_injections(path: str | Path, *, ifar_threshold_yr: float = 1.0) -> O3InjectionSet:
    """Load an O3 injection summary file into a common minimal format.

    We treat an injection as "found" if any available `ifar_*` dataset exceeds the given threshold.
    """
    from h5py import File

    path = Path(path).expanduser().resolve()
    with File(path, "r") as f:
        inj = f["injections"]
        z = np.asarray(inj["redshift"][()], dtype=float)
        dL = np.asarray(inj["distance"][()], dtype=float)

        # Prefer a precomputed network SNR if available.
        if "optimal_snr_net" in inj:
            snr_net = np.asarray(inj["optimal_snr_net"][()], dtype=float)
        else:
            snr_h = np.asarray(inj["optimal_snr_h"][()], dtype=float)
            snr_l = np.asarray(inj["optimal_snr_l"][()], dtype=float)
            snr_net = np.sqrt(np.clip(snr_h, 0.0, np.inf) ** 2 + np.clip(snr_l, 0.0, np.inf) ** 2)

        # Build found mask from all available iFAR datasets.
        ifar_keys = [str(k) for k in inj.keys() if str(k).startswith("ifar_")]
        if not ifar_keys:
            raise KeyError(f"{path}: no injections/ifar_* datasets found.")
        found = np.zeros_like(z, dtype=bool)
        for k in ifar_keys:
            vals = np.asarray(inj[k][()], dtype=float)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            found |= vals > float(ifar_threshold_yr)

        sampling_pdf = np.asarray(inj["sampling_pdf"][()], dtype=float) if "sampling_pdf" in inj else np.ones_like(z, dtype=float)
        mixture_weight = np.asarray(inj["mixture_weight"][()], dtype=float) if "mixture_weight" in inj else np.ones_like(z, dtype=float)

        m1 = np.asarray(inj["mass1_source"][()], dtype=float)
        m2 = np.asarray(inj["mass2_source"][()], dtype=float)

        ra = np.asarray(inj["right_ascension"][()], dtype=float) if "right_ascension" in inj else None
        dec = np.asarray(inj["declination"][()], dtype=float) if "declination" in inj else None

        total_generated = f.attrs.get("total_generated", inj.attrs.get("total_generated"))
        if total_generated is None:
            n_acc = int(inj.attrs.get("n_accepted", z.size))
            n_rej = int(inj.attrs.get("n_rejected", 0))
            total_generated = int(n_acc + n_rej)
        total_generated = int(total_generated)

        analysis_time_s = f.attrs.get("analysis_time_s", inj.attrs.get("analysis_time_s"))
        if analysis_time_s is None:
            raise KeyError(f"{path}: missing analysis_time_s attribute.")
        analysis_time_s = float(analysis_time_s)

    return O3InjectionSet(
        path=str(path),
        ifar_threshold_yr=float(ifar_threshold_yr),
        z=z,
        dL_mpc_fid=dL,
        snr_net_opt=snr_net,
        found_ifar=np.asarray(found, dtype=bool),
        sampling_pdf=sampling_pdf,
        mixture_weight=mixture_weight,
        m1_source=m1,
        m2_source=m2,
        ra_rad=ra,
        dec_rad=dec,
        total_generated=total_generated,
        analysis_time_s=analysis_time_s,
    )


def load_o3a_bbhpop_injections(path: str | Path, *, ifar_threshold_yr: float = 1.0) -> O3InjectionSet:
    """Backward-compatible alias for `load_o3_injections`."""
    return load_o3_injections(path, ifar_threshold_yr=ifar_threshold_yr)


# Backward-compatible type name (used across older scripts/modules).
O3aBbhInjectionSet = O3InjectionSet


def infer_observing_segment_from_event_name(event: str) -> Literal["o3a", "o3b", "other", "unknown"]:
    """Infer O3 segment from the event name `GWYYMMDD_HHMMSS`."""
    s = str(event)
    if not (s.startswith("GW") and "_" in s and len(s) >= 15):
        return "unknown"
    yymmdd = s[2:8]
    if not yymmdd.isdigit():
        return "unknown"

    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = 2000 + yy
    try:
        d = date(year, mm, dd)
    except Exception:
        return "unknown"

    o3a_start = date(2019, 4, 1)
    o3a_end = date(2019, 10, 1)
    o3b_start = date(2019, 11, 1)
    o3b_end = date(2020, 3, 27)

    if o3a_start <= d < o3a_end:
        return "o3a"
    if o3b_start <= d <= o3b_end:
        return "o3b"
    return "other"


def resolve_o3_sensitivity_injection_file(
    *,
    events: list[str],
    base_dir: str | Path = "data/cache/gw/zenodo",
    record_id: int = 7890437,
    population: Literal["mixture", "bbhpop"] = "mixture",
    auto_download: bool = True,
) -> Path:
    """Resolve (and optionally download) an O3 sensitivity injection summary file."""
    base_dir = Path(base_dir).expanduser().resolve()
    rec_dir = base_dir / str(int(record_id))
    rec_dir.mkdir(parents=True, exist_ok=True)

    segs = {infer_observing_segment_from_event_name(ev) for ev in events}
    if segs == {"o3a"}:
        seg = "o3a"
    elif segs == {"o3b"}:
        seg = "o3b"
    else:
        seg = "o3"

    gps_tag = {"o3a": "1238166018-15843600", "o3b": "1256655642-12905976"}

    pop = str(population)
    base = f"endo3_{pop}-LIGO-T2100113-v12"
    filename = f"{base}.hdf5" if seg == "o3" else f"{base}-{gps_tag[seg]}.hdf5"
    dest = rec_dir / filename
    if dest.exists():
        return dest

    if not auto_download:
        raise FileNotFoundError(f"Missing injection file {dest} (auto_download=False).")

    url = f"https://zenodo.org/records/{int(record_id)}/files/{filename}?download=1"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        from urllib.request import urlretrieve  # noqa: S310

        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                urlretrieve(url, tmp)  # type: ignore[misc]
                tmp.replace(dest)
                last_err = None
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(2.0 * attempt)
        if last_err is not None:
            raise last_err
        (rec_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "record_id": int(record_id),
                    "source": "zenodo",
                    "url": url,
                    "file": filename,
                    "population": pop,
                    "segment": seg,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

    if not dest.exists():
        raise FileNotFoundError(f"Download failed: {dest} was not created.")
    return dest


def calibrate_snr_threshold_match_count(*, snr_net_opt: np.ndarray, found_ifar: np.ndarray) -> float:
    """Pick an SNR threshold so that `snr > thresh` matches the IFAR-found count (fiducial cosmology)."""
    snr_net_opt = np.asarray(snr_net_opt, dtype=float)
    found_ifar = np.asarray(found_ifar, dtype=bool)
    if snr_net_opt.shape != found_ifar.shape:
        raise ValueError("snr_net_opt and found_ifar must have matching shapes.")
    n = int(snr_net_opt.size)
    k = int(np.sum(found_ifar))
    if n == 0 or k == 0:
        raise ValueError("No injections / no found injections; cannot calibrate SNR threshold.")

    q = 1.0 - float(k) / float(n)
    return float(np.quantile(snr_net_opt, q, method="higher"))


def calibrate_snr_threshold_match_found_fraction(*, snr_net_opt: np.ndarray, found_ifar: np.ndarray, weights: np.ndarray) -> float:
    """Pick an SNR threshold so that weighted frac(snr > thresh) matches weighted found_ifar fraction."""
    snr_net_opt = np.asarray(snr_net_opt, dtype=float)
    found_ifar = np.asarray(found_ifar, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if snr_net_opt.shape != found_ifar.shape or snr_net_opt.shape != weights.shape:
        raise ValueError("snr_net_opt, found_ifar, and weights must have matching shapes.")
    if snr_net_opt.size == 0:
        raise ValueError("No injections; cannot calibrate SNR threshold.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must be finite and strictly positive.")

    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        raise ValueError("Sum of weights is non-positive.")
    target = float(np.sum(weights * found_ifar.astype(float)) / wsum)
    if target <= 0.0:
        raise ValueError("No weighted found injections; cannot calibrate SNR threshold.")

    q = 1.0 - target
    order = np.argsort(snr_net_opt)
    snr_sorted = snr_net_opt[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted) / wsum
    i = int(np.searchsorted(cdf, q, side="left"))
    i = max(0, min(i, snr_sorted.size - 1))
    return float(snr_sorted[i])


@dataclass(frozen=True)
class SelectionAlphaResult:
    """Per-draw selection normalization proxy."""

    method: str
    convention: str
    det_model: str
    weight_mode: str
    mu_det_distance: str
    z_max: float
    snr_threshold: float | None
    snr_offset: float
    n_injections_used: int
    alpha_mu: np.ndarray  # (n_draws,)
    alpha_gr: np.ndarray  # (n_draws,)

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if k not in ("alpha_mu", "alpha_gr")}, indent=2)


def compute_selection_alpha_from_injections(
    *,
    post: MuForwardPosterior,
    injections: O3InjectionSet,
    convention: Literal["A", "B"] = "A",
    z_max: float,
    snr_threshold: float | None = None,
    det_model: Literal["threshold", "snr_binned"] = "snr_binned",
    snr_offset: float = 0.0,
    snr_binned_nbins: int = 200,
    weight_mode: Literal["none", "inv_sampling_pdf"] = "none",
    mu_det_distance: Literal["gw", "em"] = "gw",
    # Optional fixed-axis anisotropy for μ (requires injection RA/Dec):
    axis_icrs: np.ndarray | None = None,  # (3,)
    g_aniso: float = 0.0,
) -> SelectionAlphaResult:
    """Compute alpha(model) via distance-rescaled injection SNRs.

    This is an intentionally simple, explicit selection proxy:
      - Calibrate an empirical p_det(snr) curve from IFAR-found injections at fiducial distances.
      - For each posterior draw, rescale SNR by dL_fid / dL_model(z) and recompute the found fraction.

    If `axis_icrs` is provided and `g_aniso!=0`, apply an extra μ-only anisotropy:
      dL_gw(z, n) -> dL_gw(z) * exp(g_aniso * cos(theta)),
    where cos(theta) is computed per injection from its (RA,Dec).
    """
    z_max = float(z_max)
    if z_max <= 0.0:
        raise ValueError("z_max must be positive.")
    snr_offset = float(snr_offset)
    if not np.isfinite(snr_offset):
        raise ValueError("snr_offset must be finite.")
    mu_det_distance = str(mu_det_distance)
    if mu_det_distance not in ("gw", "em"):
        raise ValueError("mu_det_distance must be one of {'gw','em'}.")

    z = np.asarray(injections.z, dtype=float)
    dL_fid = np.asarray(injections.dL_mpc_fid, dtype=float)
    snr = np.asarray(injections.snr_net_opt, dtype=float)
    found_ifar = np.asarray(injections.found_ifar, dtype=bool)
    m1 = np.asarray(injections.m1_source, dtype=float)
    m2 = np.asarray(injections.m2_source, dtype=float)

    z_hi = float(min(z_max, float(post.z_grid[-1])))
    m = (
        np.isfinite(z)
        & (z > 0.0)
        & (z <= z_hi)
        & np.isfinite(dL_fid)
        & (dL_fid > 0.0)
        & np.isfinite(snr)
        & (snr > 0.0)
        & np.isfinite(m1)
        & np.isfinite(m2)
        & (m1 > 0.0)
        & (m2 > 0.0)
        & (m2 <= m1)
    )
    if not np.any(m):
        raise ValueError(f"No injections remain after z/dL/SNR/mass cuts (z_hi={z_hi}).")
    z = z[m]
    dL_fid = dL_fid[m]
    snr = snr[m]
    found_ifar = found_ifar[m]

    # Build per-injection weights w_i for estimating alpha(model).
    w = np.ones_like(z, dtype=float)
    mw = np.asarray(getattr(injections, "mixture_weight", np.ones_like(injections.z)), dtype=float)[m]
    if mw.shape != z.shape:
        raise ValueError("injections.mixture_weight must match injections.z shape.")
    w = w * mw

    if weight_mode == "none":
        pass
    elif weight_mode == "inv_sampling_pdf":
        pdf = np.asarray(injections.sampling_pdf, dtype=float)[m]
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise ValueError("sampling_pdf contains non-finite or non-positive values; cannot use inv_sampling_pdf weighting.")
        w = w / pdf
    else:
        raise ValueError("Unknown weight_mode.")

    good_w = np.isfinite(w) & (w > 0.0)
    if not np.any(good_w):
        raise ValueError("All injections received zero/invalid weights.")
    if not np.all(good_w):
        z = z[good_w]
        dL_fid = dL_fid[good_w]
        snr = snr[good_w]
        w = w[good_w]
        found_ifar = found_ifar[good_w]
        m = np.asarray(m, dtype=bool)
        m[np.where(m)[0][~good_w]] = False

    # Optional per-injection cos(theta) for anisotropy (ICRS).
    g_aniso = float(g_aniso)
    do_aniso = axis_icrs is not None and np.isfinite(g_aniso) and g_aniso != 0.0
    cos_theta: np.ndarray | None = None
    if do_aniso:
        axis = np.asarray(axis_icrs, dtype=float).reshape(3)
        n = float(np.linalg.norm(axis))
        if not np.isfinite(n) or n == 0.0:
            raise ValueError("axis_icrs is invalid.")
        axis = axis / n

        if injections.ra_rad is None or injections.dec_rad is None:
            raise ValueError("axis_icrs provided but injections lack (ra,dec).")
        ra = np.asarray(injections.ra_rad, dtype=float)[m]
        dec = np.asarray(injections.dec_rad, dtype=float)[m]
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        zc = np.sin(dec)
        cos_theta = axis[0] * x + axis[1] * y + axis[2] * zc
        cos_theta = np.asarray(cos_theta, dtype=float)

    if snr_threshold is None:
        if det_model == "threshold":
            snr_threshold = calibrate_snr_threshold_match_count(snr_net_opt=snr, found_ifar=found_ifar)
        elif det_model == "snr_binned":
            snr_threshold = None
        else:
            raise ValueError("Unknown det_model.")

    # Build a monotone empirical detection-probability curve p_det(snr).
    pdet_edges: np.ndarray | None = None
    pdet_vals: np.ndarray | None = None
    if det_model == "snr_binned":
        nb = int(snr_binned_nbins)
        if nb < 20:
            raise ValueError("snr_binned_nbins too small (need >= 20).")
        edges = np.quantile(snr, np.linspace(0.0, 1.0, nb + 1))
        edges = np.unique(edges)
        if edges.size < 10:
            raise ValueError("Too few unique SNR edges for snr_binned; injection SNR distribution seems degenerate.")
        bin_idx = np.clip(np.digitize(snr, edges) - 1, 0, edges.size - 2)
        p = np.zeros(edges.size - 1, dtype=float)
        for i in range(p.size):
            m_i = bin_idx == i
            if not np.any(m_i):
                p[i] = p[i - 1] if i > 0 else 0.0
                continue
            p[i] = float(np.mean(found_ifar[m_i].astype(float)))
        p = np.maximum.accumulate(np.clip(p, 0.0, 1.0))
        pdet_edges = edges
        pdet_vals = p

    # Precompute model distances on the posterior z_grid for each draw, then interpolate to injection z.
    z_grid = np.asarray(post.z_grid, dtype=float)
    dL_em_grid = predict_dL_em(post, z_eval=z_grid)  # (n_draws, n_z)
    _, R_grid = predict_r_gw_em(post, z_eval=z_grid, convention=convention, allow_extrapolation=False)
    dL_gw_grid = dL_em_grid * np.asarray(R_grid, dtype=float)

    n_draws = int(dL_em_grid.shape[0])
    alpha_mu = np.empty((n_draws,), dtype=float)
    alpha_gr = np.empty((n_draws,), dtype=float)

    for j in range(n_draws):
        dL_em = np.interp(z, z_grid, dL_em_grid[j])
        dL_gw = np.interp(z, z_grid, dL_gw_grid[j])

        dL_mu_det = dL_gw if mu_det_distance == "gw" else dL_em
        if do_aniso and mu_det_distance == "gw" and cos_theta is not None:
            dL_mu_det = dL_mu_det * np.exp(g_aniso * cos_theta)

        snr_gr = snr * (dL_fid / np.clip(dL_em, 1e-6, np.inf))
        snr_mu = snr * (dL_fid / np.clip(dL_mu_det, 1e-6, np.inf))
        snr_gr_eff = snr_gr - snr_offset
        snr_mu_eff = snr_mu - snr_offset

        wsum = float(np.sum(w))
        if wsum <= 0.0:
            raise ValueError("Sum of selection weights is non-positive.")

        if det_model == "threshold":
            if snr_threshold is None:
                raise ValueError("snr_threshold must be provided or calibratable for det_model='threshold'.")
            alpha_gr[j] = float(np.sum(w * (snr_gr_eff > float(snr_threshold))) / wsum)
            alpha_mu[j] = float(np.sum(w * (snr_mu_eff > float(snr_threshold))) / wsum)
        elif det_model == "snr_binned":
            assert pdet_edges is not None and pdet_vals is not None
            idx_gr = np.clip(np.digitize(snr_gr_eff, pdet_edges) - 1, 0, pdet_vals.size - 1)
            idx_mu = np.clip(np.digitize(snr_mu_eff, pdet_edges) - 1, 0, pdet_vals.size - 1)
            p_gr = pdet_vals[idx_gr]
            p_mu = pdet_vals[idx_mu]
            alpha_gr[j] = float(np.sum(w * p_gr) / wsum)
            alpha_mu[j] = float(np.sum(w * p_mu) / wsum)
        else:
            raise ValueError("Unknown det_model.")

    alpha_mu = np.clip(alpha_mu, 1e-300, 1.0)
    alpha_gr = np.clip(alpha_gr, 1e-300, 1.0)
    return SelectionAlphaResult(
        method="o3_injections_snr_rescale",
        convention=str(convention),
        det_model=str(det_model),
        weight_mode=str(weight_mode),
        mu_det_distance=str(mu_det_distance),
        z_max=float(z_hi),
        snr_threshold=float(snr_threshold) if snr_threshold is not None else None,
        snr_offset=float(snr_offset),
        n_injections_used=int(z.size),
        alpha_mu=np.asarray(alpha_mu, dtype=float),
        alpha_gr=np.asarray(alpha_gr, dtype=float),
    )


def compute_selection_alpha_from_injections_g_grid(
    *,
    post: MuForwardPosterior,
    injections: O3InjectionSet,
    convention: Literal["A", "B"] = "A",
    z_max: float,
    g_grid: np.ndarray,
    snr_threshold: float | None = None,
    det_model: Literal["threshold", "snr_binned"] = "snr_binned",
    snr_offset: float = 0.0,
    snr_binned_nbins: int = 200,
    weight_mode: Literal["none", "inv_sampling_pdf"] = "none",
    mu_det_distance: Literal["gw", "em"] = "gw",
    axis_icrs: np.ndarray | None = None,  # (3,)
    progress_every_draws: int = 0,
) -> dict[str, object]:
    """Compute alpha(model) for multiple g values on a fixed axis in one pass.

    This avoids recomputing per-draw distance interpolations for each g. It is otherwise
    algebraically identical to calling `compute_selection_alpha_from_injections` repeatedly.
    """
    z_max = float(z_max)
    if z_max <= 0.0:
        raise ValueError("z_max must be positive.")
    snr_offset = float(snr_offset)
    if not np.isfinite(snr_offset):
        raise ValueError("snr_offset must be finite.")
    mu_det_distance = str(mu_det_distance)
    if mu_det_distance not in ("gw", "em"):
        raise ValueError("mu_det_distance must be one of {'gw','em'}.")

    g_grid = np.asarray(g_grid, dtype=float).reshape(-1)
    if g_grid.size == 0 or not np.all(np.isfinite(g_grid)):
        raise ValueError("g_grid must be a non-empty finite array.")

    z = np.asarray(injections.z, dtype=float)
    dL_fid = np.asarray(injections.dL_mpc_fid, dtype=float)
    snr = np.asarray(injections.snr_net_opt, dtype=float)
    found_ifar = np.asarray(injections.found_ifar, dtype=bool)
    m1 = np.asarray(injections.m1_source, dtype=float)
    m2 = np.asarray(injections.m2_source, dtype=float)

    z_hi = float(min(z_max, float(post.z_grid[-1])))
    m = (
        np.isfinite(z)
        & (z > 0.0)
        & (z <= z_hi)
        & np.isfinite(dL_fid)
        & (dL_fid > 0.0)
        & np.isfinite(snr)
        & (snr > 0.0)
        & np.isfinite(m1)
        & np.isfinite(m2)
        & (m1 > 0.0)
        & (m2 > 0.0)
        & (m2 <= m1)
    )
    if not np.any(m):
        raise ValueError(f"No injections remain after z/dL/SNR/mass cuts (z_hi={z_hi}).")
    z = z[m]
    dL_fid = dL_fid[m]
    snr = snr[m]
    found_ifar = found_ifar[m]

    w = np.ones_like(z, dtype=float)
    mw = np.asarray(getattr(injections, "mixture_weight", np.ones_like(injections.z)), dtype=float)[m]
    if mw.shape != z.shape:
        raise ValueError("injections.mixture_weight must match injections.z shape.")
    w = w * mw

    if weight_mode == "none":
        pass
    elif weight_mode == "inv_sampling_pdf":
        pdf = np.asarray(injections.sampling_pdf, dtype=float)[m]
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise ValueError("sampling_pdf contains non-finite or non-positive values; cannot use inv_sampling_pdf weighting.")
        w = w / pdf
    else:
        raise ValueError("Unknown weight_mode.")

    good_w = np.isfinite(w) & (w > 0.0)
    if not np.any(good_w):
        raise ValueError("All injections received zero/invalid weights.")
    if not np.all(good_w):
        z = z[good_w]
        dL_fid = dL_fid[good_w]
        snr = snr[good_w]
        w = w[good_w]
        found_ifar = found_ifar[good_w]
        m = np.asarray(m, dtype=bool)
        m[np.where(m)[0][~good_w]] = False

    # Optional per-injection cos(theta) for anisotropy (ICRS), shared across all g.
    do_aniso = axis_icrs is not None and mu_det_distance == "gw" and np.any(np.abs(g_grid) > 0.0)
    cos_theta: np.ndarray | None = None
    if do_aniso:
        axis = np.asarray(axis_icrs, dtype=float).reshape(3)
        n = float(np.linalg.norm(axis))
        if not np.isfinite(n) or n == 0.0:
            raise ValueError("axis_icrs is invalid.")
        axis = axis / n

        if injections.ra_rad is None or injections.dec_rad is None:
            raise ValueError("axis_icrs provided but injections lack (ra,dec).")
        ra = np.asarray(injections.ra_rad, dtype=float)[m]
        dec = np.asarray(injections.dec_rad, dtype=float)[m]
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        zc = np.sin(dec)
        cos_theta = axis[0] * x + axis[1] * y + axis[2] * zc
        cos_theta = np.asarray(cos_theta, dtype=float)

    if snr_threshold is None:
        if det_model == "threshold":
            snr_threshold = calibrate_snr_threshold_match_count(snr_net_opt=snr, found_ifar=found_ifar)
        elif det_model == "snr_binned":
            snr_threshold = None
        else:
            raise ValueError("Unknown det_model.")

    pdet_edges: np.ndarray | None = None
    pdet_vals: np.ndarray | None = None
    if det_model == "snr_binned":
        nb = int(snr_binned_nbins)
        if nb < 20:
            raise ValueError("snr_binned_nbins too small (need >= 20).")
        edges = np.quantile(snr, np.linspace(0.0, 1.0, nb + 1))
        edges = np.unique(edges)
        if edges.size < 10:
            raise ValueError("Too few unique SNR edges for snr_binned; injection SNR distribution seems degenerate.")
        bin_idx = np.clip(np.digitize(snr, edges) - 1, 0, edges.size - 2)
        p = np.zeros(edges.size - 1, dtype=float)
        for i in range(p.size):
            m_i = bin_idx == i
            if not np.any(m_i):
                p[i] = p[i - 1] if i > 0 else 0.0
                continue
            p[i] = float(np.mean(found_ifar[m_i].astype(float)))
        p = np.maximum.accumulate(np.clip(p, 0.0, 1.0))
        pdet_edges = edges
        pdet_vals = p

    # Precompute model distances on the posterior z_grid for each draw, then interpolate to injection z.
    z_grid = np.asarray(post.z_grid, dtype=float)
    dL_em_grid = predict_dL_em(post, z_eval=z_grid)  # (n_draws, n_z)
    _, R_grid = predict_r_gw_em(post, z_eval=z_grid, convention=convention, allow_extrapolation=False)
    dL_gw_grid = dL_em_grid * np.asarray(R_grid, dtype=float)

    n_draws = int(dL_em_grid.shape[0])
    alpha_gr = np.empty((n_draws,), dtype=float)
    alpha_mu_by_g = np.empty((g_grid.size, n_draws), dtype=float)

    fac_by_g: np.ndarray | None = None
    if do_aniso and cos_theta is not None:
        fac_by_g = np.exp((-g_grid.reshape((-1, 1))) * cos_theta.reshape((1, -1)))

    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise ValueError("Sum of selection weights is non-positive.")

    progress_every = int(progress_every_draws)
    if progress_every < 0:
        raise ValueError("progress_every_draws must be >= 0.")
    t0 = time.monotonic()
    if progress_every > 0:
        grid_str = ", ".join(f"{float(g):.3g}" for g in g_grid.tolist())
        print(
            f"[selection_alpha] start n_draws={n_draws} n_inj={int(z.size)} "
            f"det_model={det_model} mu_det_distance={mu_det_distance} g_grid=[{grid_str}]",
            flush=True,
        )

    for j in range(n_draws):
        dL_em = np.interp(z, z_grid, dL_em_grid[j])
        dL_gw = np.interp(z, z_grid, dL_gw_grid[j])

        dL_mu_det_iso = dL_gw if mu_det_distance == "gw" else dL_em

        snr_gr_eff = snr * (dL_fid / np.clip(dL_em, 1e-6, np.inf)) - snr_offset
        snr_mu_iso = snr * (dL_fid / np.clip(dL_mu_det_iso, 1e-6, np.inf))

        if det_model == "threshold":
            if snr_threshold is None:
                raise ValueError("snr_threshold must be provided or calibratable for det_model='threshold'.")
            alpha_gr[j] = float(np.sum(w * (snr_gr_eff > float(snr_threshold))) / wsum)
            if fac_by_g is None:
                snr_mu_eff = snr_mu_iso - snr_offset
                v = float(np.sum(w * (snr_mu_eff > float(snr_threshold))) / wsum)
                alpha_mu_by_g[:, j] = v
            else:
                for i in range(g_grid.size):
                    snr_mu_eff = snr_mu_iso * fac_by_g[i] - snr_offset
                    alpha_mu_by_g[i, j] = float(np.sum(w * (snr_mu_eff > float(snr_threshold))) / wsum)
            continue

        if det_model != "snr_binned":
            raise ValueError("Unknown det_model.")
        assert pdet_edges is not None and pdet_vals is not None

        idx_gr = np.clip(np.digitize(snr_gr_eff, pdet_edges) - 1, 0, pdet_vals.size - 1)
        alpha_gr[j] = float(np.sum(w * pdet_vals[idx_gr]) / wsum)

        if fac_by_g is None:
            snr_mu_eff = snr_mu_iso - snr_offset
            idx_mu = np.clip(np.digitize(snr_mu_eff, pdet_edges) - 1, 0, pdet_vals.size - 1)
            v = float(np.sum(w * pdet_vals[idx_mu]) / wsum)
            alpha_mu_by_g[:, j] = v
        else:
            for i in range(g_grid.size):
                snr_mu_eff = snr_mu_iso * fac_by_g[i] - snr_offset
                idx_mu = np.clip(np.digitize(snr_mu_eff, pdet_edges) - 1, 0, pdet_vals.size - 1)
                alpha_mu_by_g[i, j] = float(np.sum(w * pdet_vals[idx_mu]) / wsum)

        if progress_every > 0 and ((j + 1) % progress_every == 0 or (j + 1) == n_draws):
            elapsed_s = time.monotonic() - t0
            rate = float(j + 1) / elapsed_s if elapsed_s > 0.0 else 0.0
            eta_s = float(n_draws - (j + 1)) / rate if rate > 0.0 else float("inf")
            print(
                f"[selection_alpha] {j+1}/{n_draws} ({100.0*(j+1)/max(1,n_draws):.1f}%) "
                f"elapsed={elapsed_s:.1f}s eta={eta_s:.1f}s",
                flush=True,
            )

    alpha_gr = np.clip(alpha_gr, 1e-300, 1.0)
    alpha_mu_by_g = np.clip(alpha_mu_by_g, 1e-300, 1.0)
    return {
        "meta": {
            "method": "o3_injections_snr_rescale",
            "convention": str(convention),
            "det_model": str(det_model),
            "weight_mode": str(weight_mode),
            "mu_det_distance": str(mu_det_distance),
            "z_max": float(z_hi),
            "snr_threshold": float(snr_threshold) if snr_threshold is not None else None,
            "snr_offset": float(snr_offset),
            "n_injections_used": int(z.size),
        },
        "g_grid": np.asarray(g_grid, dtype=float),
        "alpha_gr": np.asarray(alpha_gr, dtype=float),
        "alpha_mu_by_g": {float(g): np.asarray(alpha_mu_by_g[i], dtype=float) for i, g in enumerate(g_grid.tolist())},
    }
