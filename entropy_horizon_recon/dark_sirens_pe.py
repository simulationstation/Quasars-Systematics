from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import healpy as hp
import numpy as np
from scipy.special import logsumexp

from .gw_distance_priors import GWDistancePrior
from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em


@dataclass(frozen=True)
class PePixelDistanceHistogram:
    """Binned approximation to p(Ω, dL | d) built from PE samples."""

    nside: int
    nest: bool
    p_credible: float
    pix_sel: np.ndarray  # (n_pix_sel,)
    prob_pix: np.ndarray  # (n_pix_sel,)
    dL_edges: np.ndarray  # (n_bins+1,)
    pdf_bins: np.ndarray  # (n_pix_sel, n_bins)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "nside": int(self.nside),
            "nest": bool(self.nest),
            "p_credible": float(self.p_credible),
            "n_pix_sel": int(self.pix_sel.size),
            "dl_min_mpc": float(self.dL_edges[0]),
            "dl_max_mpc": float(self.dL_edges[-1]),
            "n_dl_bins": int(self.dL_edges.size - 1),
        }


def compute_dark_siren_logL_draws_from_pe_hist(
    *,
    event: str,
    pe: PePixelDistanceHistogram,
    post: MuForwardPosterior,
    z_gal: np.ndarray,
    w_gal: np.ndarray,
    ipix_gal: np.ndarray,
    convention: Literal["A", "B"] = "A",
    gw_distance_prior: GWDistancePrior | None = None,
    distance_mode: Literal["full", "spectral_only"] = "full",
    gal_chunk_size: int = 50_000,
    g_aniso: float = 0.0,
    cos_theta_gal: np.ndarray | None = None,
    compute_gr: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (logL_mu, logL_gr) per posterior draw using PE posterior histograms.

    Anisotropy option:
      Apply a fixed-axis 1-parameter modulation to the HE GW distance prediction:
        dL_gw -> dL_gw * exp(g_aniso * cos_theta)
      where cos_theta is precomputed per-galaxy relative to the chosen axis.
    """
    if not bool(pe.nest):
        raise ValueError("PE pixel histogram must use NESTED ordering.")
    z = np.asarray(z_gal, dtype=float)
    w = np.asarray(w_gal, dtype=float)
    ipix = np.asarray(ipix_gal, dtype=np.int64)
    if z.ndim != 1 or w.ndim != 1 or ipix.ndim != 1 or not (z.shape == w.shape == ipix.shape):
        raise ValueError("z_gal/w_gal/ipix_gal must be 1D arrays with matching shapes.")
    if z.size == 0:
        raise ValueError("No galaxies provided.")
    if cos_theta_gal is not None:
        cos_theta_gal = np.asarray(cos_theta_gal, dtype=float)
        if cos_theta_gal.shape != z.shape:
            raise ValueError("cos_theta_gal must match z_gal shape.")

    # Map pixels to row indices in the selected set.
    npix = int(hp.nside2npix(int(pe.nside)))
    pix_to_row = np.full((npix,), -1, dtype=np.int32)
    pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
    row = pix_to_row[ipix]
    good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
    if not np.any(good):
        raise ValueError("All galaxies map outside the PE credible region (or have invalid z/w).")
    z = z[good]
    w = w[good]
    row = row[good].astype(np.int64, copy=False)
    cos_theta = cos_theta_gal[good] if cos_theta_gal is not None else None

    prob = np.asarray(pe.prob_pix, dtype=float)[row]

    prior = gw_distance_prior or GWDistancePrior()
    edges = np.asarray(pe.dL_edges, dtype=float)
    nb = int(edges.size - 1)

    pdf_flat: np.ndarray | None = None
    pdf_1d: np.ndarray | None = None
    if distance_mode == "full":
        pdf_flat = np.asarray(pe.pdf_bins, dtype=float).reshape(-1)
    elif distance_mode == "spectral_only":
        p_pix = np.asarray(pe.prob_pix, dtype=float)
        pdf_bins = np.asarray(pe.pdf_bins, dtype=float)
        p_sum = float(np.sum(p_pix))
        if not (np.isfinite(p_sum) and p_sum > 0.0):
            raise ValueError("Invalid prob_pix sum while building spectral_only distance density.")
        pdf_1d = np.sum(p_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum
        pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)
        widths = np.diff(edges)
        norm = float(np.sum(pdf_1d * widths))
        if not (np.isfinite(norm) and norm > 0.0):
            raise ValueError("Invalid sky-marginal distance density normalization in spectral_only mode.")
        pdf_1d = pdf_1d / norm
    else:
        raise ValueError("distance_mode must be 'full' or 'spectral_only'.")

    chunk = int(gal_chunk_size)
    if chunk <= 0:
        raise ValueError("gal_chunk_size must be positive.")

    n_draws = int(post.H_samples.shape[0])
    logL_mu = np.full((n_draws,), -np.inf, dtype=float)
    logL_gr = np.full((n_draws,), -np.inf, dtype=float) if compute_gr else np.full((n_draws,), np.nan, dtype=float)

    g_aniso = float(g_aniso)
    do_aniso = (cos_theta is not None) and (g_aniso != 0.0) and np.isfinite(g_aniso)

    for a in range(0, z.size, chunk):
        b = min(z.size, a + chunk)
        z_c = np.asarray(z[a:b], dtype=float)
        w_c = np.asarray(w[a:b], dtype=float)
        row_c = np.asarray(row[a:b], dtype=np.int64)
        prob_c = np.asarray(prob[a:b], dtype=float)
        cos_c = np.asarray(cos_theta[a:b], dtype=float) if do_aniso else None

        z_u, inv = np.unique(z_c, return_inverse=True)
        dL_em_u = predict_dL_em(post, z_eval=z_u)
        _, R_u = predict_r_gw_em(post, z_eval=z_u, convention=convention, allow_extrapolation=False)
        dL_gw_u = dL_em_u * np.asarray(R_u, dtype=float)

        dL_em = dL_em_u[:, inv]
        dL_gw = dL_gw_u[:, inv]
        if do_aniso and cos_c is not None:
            dL_gw = dL_gw * np.exp(g_aniso * cos_c.reshape((1, -1)))

        logw = np.log(np.clip(w_c, 1e-30, np.inf))[None, :]
        logprob = np.log(np.clip(prob_c, 1e-300, np.inf))[None, :]

        def _chunk_logL(dL: np.ndarray) -> np.ndarray:
            dL = np.asarray(dL, dtype=float)
            bin_idx = np.searchsorted(edges, dL, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL) & (dL > 0.0)
            if distance_mode == "full":
                assert pdf_flat is not None
                lin = row_c.reshape((1, -1)) * nb + np.clip(bin_idx, 0, nb - 1)
                pdf = pdf_flat[lin]
                pdf = np.where(valid, pdf, 0.0)
            else:
                assert pdf_1d is not None
                pdf = pdf_1d[np.clip(bin_idx, 0, nb - 1)]
                pdf = np.where(valid, pdf, 0.0)

            logpdf = np.log(np.clip(pdf, 1e-300, np.inf))
            logprior = prior.log_pi_dL(np.clip(dL, 1e-6, np.inf))
            logterm = logw + logprob + logpdf - logprior
            logterm = np.where(np.isfinite(logprior), logterm, -np.inf)
            return logsumexp(logterm, axis=1)

        logL_mu = np.logaddexp(logL_mu, _chunk_logL(dL_gw))
        if compute_gr:
            logL_gr = np.logaddexp(logL_gr, _chunk_logL(dL_em))

    return np.asarray(logL_mu, dtype=float), np.asarray(logL_gr, dtype=float)

