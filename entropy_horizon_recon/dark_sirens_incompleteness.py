from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .constants import PhysicalConstants
from .gw_distance_priors import GWDistancePrior
from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em


@dataclass(frozen=True)
class MissingHostPriorPrecompute:
    """Precomputed draw-wise background quantities for the missing-host (out-of-catalog) term.

    This is designed to be computed once per `post` and then reused across many events.
    """

    z_grid: np.ndarray  # (n_z,)
    dL_em: np.ndarray  # (n_draws, n_z)
    dL_gw: np.ndarray  # (n_draws, n_z)  (HE/μ propagation distance; isotropic baseline)
    base_z: np.ndarray  # (n_draws, n_z) proportional to rho_host(z) * dV/dz/dOmega
    ddLdz_em: np.ndarray  # (n_draws, n_z)
    ddLdz_gw: np.ndarray  # (n_draws, n_z)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "z_grid_min": float(self.z_grid[0]),
            "z_grid_max": float(self.z_grid[-1]),
            "n_draws": int(self.dL_em.shape[0]),
            "n_z": int(self.z_grid.size),
        }


def precompute_missing_host_prior(
    post: MuForwardPosterior,
    *,
    convention: Literal["A", "B"] = "A",
    z_max: float,
    host_prior_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "comoving_uniform",
    host_prior_z_k: float = 0.0,
    constants: PhysicalConstants | None = None,
) -> MissingHostPriorPrecompute:
    """Precompute draw-wise arrays needed to evaluate the missing-host integral.

    The missing-host likelihood uses a simple host density prior in comoving coordinates:
      rho_host(z) ∝ 1                      (comoving_uniform / none)
      rho_host(z) ∝ (1+z)^k               (comoving_powerlaw)

    and a geometric Jacobian:
      dV/dz/dOmega = (c/H(z)) * D_M(z)^2.
    """
    constants = constants or PhysicalConstants()
    z_max = float(z_max)
    if z_max <= 0.0:
        raise ValueError("z_max must be positive.")

    z_grid_full = np.asarray(post.z_grid, dtype=float)
    if z_grid_full.ndim != 1 or z_grid_full.size < 2 or float(z_grid_full[0]) != 0.0:
        raise ValueError("post.z_grid must be a 1D array starting at z=0 with at least two points.")

    z_hi = float(min(z_max, float(z_grid_full[-1])))
    m = z_grid_full <= z_hi
    z_grid = z_grid_full[m]
    if z_grid.size < 2:
        raise ValueError("z_max too small relative to posterior grid.")

    dL_em = predict_dL_em(post, z_eval=z_grid, constants=constants)  # (n_draws, n_z)
    _, R = predict_r_gw_em(post, z_eval=z_grid, convention=convention, allow_extrapolation=False)
    dL_gw = dL_em * np.asarray(R, dtype=float)

    z = z_grid.reshape((1, -1))
    H = np.asarray(post.H_samples, dtype=float)[:, m]  # (n_draws, n_z)
    if H.shape != dL_em.shape:
        raise ValueError("Unexpected shape mismatch between H_samples and dL_em.")

    Dm = dL_em / np.clip(1.0 + z, 1e-12, np.inf)
    dVdz = (constants.c_km_s / np.clip(H, 1e-12, np.inf)) * (Dm**2)

    if host_prior_z_mode in ("none", "comoving_uniform"):
        rho = np.ones_like(z, dtype=float)
    elif host_prior_z_mode == "comoving_powerlaw":
        rho = np.clip(1.0 + z, 1e-12, np.inf) ** float(host_prior_z_k)
    else:
        raise ValueError("Unknown host_prior_z_mode.")

    base_z = dVdz * rho

    ddLdz_em = np.gradient(dL_em, z_grid, axis=1)
    ddLdz_gw = np.gradient(dL_gw, z_grid, axis=1)

    return MissingHostPriorPrecompute(
        z_grid=z_grid,
        dL_em=dL_em,
        dL_gw=dL_gw,
        base_z=base_z,
        ddLdz_em=ddLdz_em,
        ddLdz_gw=ddLdz_gw,
    )


def _host_prior_matrix_from_precompute(
    pre: MissingHostPriorPrecompute,
    *,
    dL_grid: np.ndarray,
    model: Literal["mu", "gr"],
    gw_distance_prior: GWDistancePrior,
) -> np.ndarray:
    """Build per-draw host prior factors on a dL grid (isotropic baseline).

    Returns an array of shape (n_draws, n_dL) with:

      host(dL) = [rho(z) * dV/dz/dOmega] * (dz/ddL) * (1/π(dL)),

    where z=z(dL) is defined by:
      - model='gr': dL(dz) = dL_EM(z)
      - model='mu': dL(dz) = dL_GW(z) (propagation-modified; isotropic baseline)
    """
    dL_grid = np.asarray(dL_grid, dtype=float)
    if dL_grid.ndim != 1 or dL_grid.size < 2:
        raise ValueError("dL_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(dL_grid)) or np.any(np.diff(dL_grid) <= 0.0):
        raise ValueError("dL_grid must be finite and strictly increasing.")

    if model == "gr":
        dL_of_z = pre.dL_em
        ddLdz = pre.ddLdz_em
    elif model == "mu":
        dL_of_z = pre.dL_gw
        ddLdz = pre.ddLdz_gw
    else:
        raise ValueError("model must be 'mu' or 'gr'.")

    z_grid = pre.z_grid
    n_draws = int(dL_of_z.shape[0])
    out = np.zeros((n_draws, dL_grid.size), dtype=float)

    log_pi = gw_distance_prior.log_pi_dL(dL_grid)
    inv_pi = np.zeros_like(log_pi, dtype=float)
    ok_pi = np.isfinite(log_pi)
    inv_pi[ok_pi] = np.exp(-log_pi[ok_pi])

    for j in range(n_draws):
        dL_j = np.asarray(dL_of_z[j], dtype=float)
        if not np.all(np.isfinite(dL_j)) or np.any(np.diff(dL_j) <= 0.0):
            raise ValueError("Non-monotone or invalid dL(z) encountered; cannot invert for missing-host term.")

        dL_min = float(dL_j[0])
        dL_max = float(dL_j[-1])
        m = (dL_grid >= dL_min) & (dL_grid <= dL_max) & (dL_grid > 0.0)
        if not np.any(m):
            continue

        z_of_dL = np.interp(dL_grid[m], dL_j, z_grid)
        base = np.interp(z_of_dL, z_grid, pre.base_z[j])
        dd = np.interp(z_of_dL, z_grid, ddLdz[j])
        dd = np.clip(dd, 1e-12, np.inf)
        out[j, m] = base / dd * inv_pi[m]

    return out


def compute_missing_host_logL_draws_from_histogram(
    *,
    prob_pix: np.ndarray,
    pdf_bins: np.ndarray,
    dL_edges: np.ndarray,
    pre: MissingHostPriorPrecompute,
    gw_distance_prior: GWDistancePrior,
    distance_mode: Literal["full", "spectral_only"] = "full",
    pixel_chunk_size: int = 5_000,
    # Optional fixed-axis anisotropy for the μ-model mapping:
    g_aniso: float = 0.0,
    cos_theta_pix: np.ndarray | None = None,
    compute_gr: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Missing-host logL vectors using a binned per-pixel distance posterior.

    This evaluates:
      L_missing(draw) = Σ_pix prob_pix * ∫ ddL p(dL | pix, data) * host(draw; dL, pix).

    In the isotropic baseline, host(draw; dL, pix) is independent of pix. If `g_aniso!=0` and
    `cos_theta_pix` is provided, we apply a directional modulation to the μ-model distance mapping:

      dL_gw(z, n) = dL_gw_iso(z) * exp(g_aniso * cos(theta)),

    which induces a pixel-dependent host term for μ. GR remains isotropic.
    """
    prob_pix = np.asarray(prob_pix, dtype=float)
    pdf_bins = np.asarray(pdf_bins, dtype=float)
    dL_edges = np.asarray(dL_edges, dtype=float)
    if prob_pix.ndim != 1:
        raise ValueError("prob_pix must be 1D.")
    if pdf_bins.ndim != 2:
        raise ValueError("pdf_bins must be 2D (n_pix, n_bins).")
    if dL_edges.ndim != 1 or dL_edges.size < 3:
        raise ValueError("dL_edges must be 1D with >=3 entries (>=2 bins).")
    if np.any(~np.isfinite(dL_edges)) or np.any(np.diff(dL_edges) <= 0.0):
        raise ValueError("dL_edges must be finite and strictly increasing.")
    if pdf_bins.shape[0] != prob_pix.size:
        raise ValueError("pdf_bins.shape[0] must match prob_pix.size.")
    if pdf_bins.shape[1] != dL_edges.size - 1:
        raise ValueError("pdf_bins.shape[1] must equal len(dL_edges)-1.")
    if prob_pix.size == 0:
        raise ValueError("No valid sky pixels provided for missing-host term.")

    g_aniso = float(g_aniso)
    do_aniso = (cos_theta_pix is not None) and (np.isfinite(g_aniso)) and (g_aniso != 0.0)
    if do_aniso:
        cos_theta_pix = np.asarray(cos_theta_pix, dtype=float)
        if cos_theta_pix.shape != prob_pix.shape:
            raise ValueError("cos_theta_pix must match prob_pix shape.")

    widths = np.diff(dL_edges)
    dL_mid = 0.5 * (dL_edges[:-1] + dL_edges[1:])

    compute_gr = bool(compute_gr)
    host_gr_w: np.ndarray | None = None
    if compute_gr:
        host_gr = _host_prior_matrix_from_precompute(pre, dL_grid=dL_mid, model="gr", gw_distance_prior=gw_distance_prior)
        host_gr_w = host_gr * widths.reshape((1, -1))

    # For isotropic μ, precompute once; for anisotropic μ, host depends on pixel cos(theta).
    host_mu_w: np.ndarray | None = None
    if not do_aniso:
        host_mu = _host_prior_matrix_from_precompute(pre, dL_grid=dL_mid, model="mu", gw_distance_prior=gw_distance_prior)
        host_mu_w = host_mu * widths.reshape((1, -1))

    n_draws = int(pre.dL_em.shape[0])
    L_mu = np.zeros((n_draws,), dtype=float)
    L_gr = np.zeros((n_draws,), dtype=float) if compute_gr else np.full((n_draws,), np.nan, dtype=float)

    if distance_mode == "spectral_only":
        # Sky-marginal distance density:
        p_sum = float(np.sum(prob_pix))
        if not (np.isfinite(p_sum) and p_sum > 0.0):
            raise ValueError("Invalid prob_pix sum while building spectral_only missing-host term.")

        pdf_1d = np.sum(prob_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum  # (n_bins,)
        pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)
        norm = float(np.sum(pdf_1d * widths))
        if not (np.isfinite(norm) and norm > 0.0):
            raise ValueError("Invalid sky-marginal distance density normalization in spectral_only missing-host term.")
        pdf_1d = pdf_1d / norm

        # GR:
        if compute_gr:
            assert host_gr_w is not None
            L_gr = np.clip(host_gr_w @ pdf_1d, 1e-300, np.inf)

        # μ:
        if not do_aniso:
            assert host_mu_w is not None
            L_mu = np.clip(host_mu_w @ pdf_1d, 1e-300, np.inf)
            return np.log(L_mu), np.log(L_gr)

        # Anisotropic μ: average host over pixels with weights prob_pix.
        # This keeps distance-spectrum fixed but preserves sky-dependent host mapping.
        inv_pi = np.exp(-gw_distance_prior.log_pi_dL(dL_mid))
        inv_pi = np.where(np.isfinite(inv_pi), inv_pi, 0.0)
        scale = np.exp(g_aniso * np.asarray(cos_theta_pix, dtype=float))
        scale = np.clip(scale, 1e-12, np.inf)

        # Precompute per-pixel weight for the sky sum (normalize prob_pix to sum=1 over selected pixels).
        p_norm = prob_pix / p_sum
        pdf = np.asarray(pdf_1d, dtype=float).reshape((1, -1))  # (1, n_bins)

        # Compute L_mu per draw via pixel-chunk loops.
        n_pix = int(prob_pix.size)
        chunk = int(pixel_chunk_size)
        if chunk <= 0:
            raise ValueError("pixel_chunk_size must be positive.")
        for a in range(0, n_pix, chunk):
            b = min(n_pix, a + chunk)
            p = p_norm[a:b]
            s = scale[a:b]
            dL_scaled = dL_mid.reshape((1, -1)) / s.reshape((-1, 1))  # (chunk, n_bins)
            dL_scaled_flat = dL_scaled.reshape(-1)
            for j in range(n_draws):
                dL_iso = np.asarray(pre.dL_gw[j], dtype=float)
                fz = np.asarray(pre.base_z[j] / np.clip(pre.ddLdz_gw[j], 1e-12, np.inf), dtype=float)
                f_interp = np.interp(dL_scaled_flat, dL_iso, fz, left=0.0, right=0.0).reshape((b - a, -1))
                host = (f_interp / s.reshape((-1, 1))) * inv_pi.reshape((1, -1))
                # L_mu += Σ_pix p * ∫ ddL host(dL,pix) * pdf_1d(dL)
                L_mu[j] += float(np.sum(p * np.sum(host * widths.reshape((1, -1)) * pdf, axis=1)))

        L_mu = np.clip(L_mu, 1e-300, np.inf)
        if compute_gr:
            L_gr = np.clip(L_gr, 1e-300, np.inf)
        return np.log(L_mu), (np.log(L_gr) if compute_gr else np.asarray(L_gr, dtype=float))

    if distance_mode != "full":
        raise ValueError("distance_mode must be 'full' or 'spectral_only'.")

    n_pix = int(prob_pix.size)
    chunk = int(pixel_chunk_size)
    if chunk <= 0:
        raise ValueError("pixel_chunk_size must be positive.")

    # Precompute 1/pi(dL) for anisotropic μ case, since host must be built per pixel.
    inv_pi_mid = None
    if do_aniso:
        log_pi = gw_distance_prior.log_pi_dL(dL_mid)
        inv_pi_mid = np.zeros_like(log_pi, dtype=float)
        ok_pi = np.isfinite(log_pi)
        inv_pi_mid[ok_pi] = np.exp(-log_pi[ok_pi])

    for a in range(0, n_pix, chunk):
        b = min(n_pix, a + chunk)
        p = prob_pix[a:b]
        pdf = np.asarray(pdf_bins[a:b, :], dtype=float)  # (chunk, n_bins)
        pdf = np.clip(pdf, 0.0, np.inf)

        # GR contribution for this chunk (pixel-independent host).
        if compute_gr:
            assert host_gr_w is not None
            proj_gr = host_gr_w @ pdf.T  # (n_draws, chunk)
            L_gr += proj_gr @ p

        if not do_aniso:
            assert host_mu_w is not None
            proj_mu = host_mu_w @ pdf.T
            L_mu += proj_mu @ p
            continue

        # Anisotropic μ: host depends on pixel scale exp(g cosθ).
        assert inv_pi_mid is not None
        cos = np.asarray(cos_theta_pix[a:b], dtype=float)
        scale = np.exp(g_aniso * cos)
        scale = np.clip(scale, 1e-12, np.inf)

        # (chunk, n_bins): dL_scaled = dL / scale
        dL_scaled = dL_mid.reshape((1, -1)) / scale.reshape((-1, 1))
        dL_scaled_flat = dL_scaled.reshape(-1)

        # For each draw: build host on these scaled distances and contract with the per-pixel pdf.
        for j in range(n_draws):
            dL_iso = np.asarray(pre.dL_gw[j], dtype=float)
            if not np.all(np.isfinite(dL_iso)) or np.any(np.diff(dL_iso) <= 0.0):
                raise ValueError("Non-monotone dL_gw(z) encountered in missing-host anisotropy.")

            fz = np.asarray(pre.base_z[j] / np.clip(pre.ddLdz_gw[j], 1e-12, np.inf), dtype=float)
            f_interp = np.interp(dL_scaled_flat, dL_iso, fz, left=0.0, right=0.0).reshape((b - a, -1))
            host = (f_interp / scale.reshape((-1, 1))) * inv_pi_mid.reshape((1, -1))
            # Integrate over dL bins and sum over pixels with prob weights.
            dot_pix = np.sum(host * widths.reshape((1, -1)) * pdf, axis=1)  # (chunk,)
            L_mu[j] += float(dot_pix @ p)

    L_mu = np.clip(L_mu, 1e-300, np.inf)
    if compute_gr:
        L_gr = np.clip(L_gr, 1e-300, np.inf)
    return np.log(L_mu), (np.log(L_gr) if compute_gr else np.asarray(L_gr, dtype=float))
