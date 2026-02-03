from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import PhysicalConstants


@dataclass(frozen=True)
class BackgroundCosmology:
    """FLRW background built from an H(z) function."""

    z_grid: np.ndarray
    H_grid: np.ndarray
    Dc_grid: np.ndarray
    constants: PhysicalConstants

    def H(self, z: np.ndarray) -> np.ndarray:
        return np.interp(z, self.z_grid, self.H_grid)

    def Dc(self, z: np.ndarray) -> np.ndarray:
        return np.interp(z, self.z_grid, self.Dc_grid)


def spline_logH_from_knots(z_knots: np.ndarray, logH_knots: np.ndarray) -> CubicSpline:
    if z_knots.ndim != 1 or logH_knots.ndim != 1:
        raise ValueError("z_knots and logH_knots must be 1D arrays.")
    if z_knots.shape != logH_knots.shape:
        raise ValueError("z_knots and logH_knots must have the same shape.")
    if np.any(~np.isfinite(z_knots)) or np.any(~np.isfinite(logH_knots)):
        raise ValueError("z_knots/logH_knots contain non-finite values.")
    if np.any(np.diff(z_knots) <= 0):
        raise ValueError("z_knots must be strictly increasing.")
    return CubicSpline(z_knots, logH_knots, bc_type="natural", extrapolate=True)


def build_background_from_H_grid(
    z_grid: np.ndarray,
    H_grid: np.ndarray,
    *,
    constants: PhysicalConstants,
) -> BackgroundCosmology:
    z_grid = np.asarray(z_grid, dtype=float)
    H_grid = np.asarray(H_grid, dtype=float)
    if z_grid.ndim != 1 or H_grid.ndim != 1 or z_grid.shape != H_grid.shape:
        raise ValueError("z_grid and H_grid must be 1D arrays with matching shape.")
    if z_grid.size < 2:
        raise ValueError("Need at least two grid points.")
    if np.any(np.diff(z_grid) <= 0):
        raise ValueError("z_grid must be strictly increasing.")
    if np.any(H_grid <= 0) or not np.all(np.isfinite(H_grid)):
        raise ValueError("H_grid must be positive and finite.")

    invH = 1.0 / H_grid
    Dc_grid = np.empty_like(z_grid)
    Dc_grid[0] = 0.0
    dz = np.diff(z_grid)
    Dc_grid[1:] = constants.c_km_s * np.cumsum(0.5 * dz * (invH[:-1] + invH[1:]))
    return BackgroundCosmology(z_grid=z_grid, H_grid=H_grid, Dc_grid=Dc_grid, constants=constants)

