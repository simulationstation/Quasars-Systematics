"""Shared helpers for Secrest+22 dipole analyses.

This module centralizes the baseline catalog cuts and dipole estimators
used by the reproduction and slicing scripts so they remain consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from astropy.table import Table

# Default catalog path from Stage 3 reproduction
SECREST_CATALOG_DEFAULT = "./data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits"

# Published Secrest+22 values for comparison
SECREST_PUBLISHED = {
    "amplitude": 0.0154,
    "amplitude_sigma": 0.0015,
    "l_deg": 238.2,
    "b_deg": 28.8,
    "N_sources": 1355352,
    "reference": "Secrest et al. 2022, ApJL 937 L31",
}

# CMB dipole direction for comparison
CMB_L_DEG = 264.021
CMB_B_DEG = 48.253


@dataclass
class BootstrapResult:
    amplitude_q16: float
    amplitude_q50: float
    amplitude_q84: float
    amplitude_std: float
    l_median: float
    b_median: float
    direction_scatter_q16: float
    direction_scatter_q50: float
    direction_scatter_q84: float
    direction_sigma_deg: float
    n_bootstrap: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_bootstrap": self.n_bootstrap,
            "amplitude_q16": self.amplitude_q16,
            "amplitude_q50": self.amplitude_q50,
            "amplitude_q84": self.amplitude_q84,
            "amplitude_std": self.amplitude_std,
            "l_median": self.l_median,
            "b_median": self.b_median,
            "direction_scatter_q16": self.direction_scatter_q16,
            "direction_scatter_q50": self.direction_scatter_q50,
            "direction_scatter_q84": self.direction_scatter_q84,
            "direction_sigma_deg": self.direction_sigma_deg,
        }


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def lb_to_unitvec(l_deg: Iterable[float], b_deg: Iterable[float]) -> np.ndarray:
    l_rad = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b_rad = np.deg2rad(np.asarray(b_deg, dtype=float))
    cos_b = np.cos(b_rad)
    return np.column_stack(
        [cos_b * np.cos(l_rad), cos_b * np.sin(l_rad), np.sin(b_rad)]
    )


def unitvec_to_lb(vec: np.ndarray) -> Tuple[float, float]:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return float("nan"), float("nan")
    v = vec / norm
    l = math.degrees(math.atan2(v[1], v[0])) % 360.0
    b = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    return l, b


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    # Some call sites pass (1, 3) arrays when a single direction is created
    # from scalar (l,b). Normalize those to 1D 3-vectors so np.dot works.
    v1 = np.asarray(v1, dtype=float).reshape(-1)
    v2 = np.asarray(v2, dtype=float).reshape(-1)
    if v1.size != 3 or v2.size != 3:
        raise ValueError(f"angle_deg expects 3-vectors; got shapes {v1.shape} and {v2.shape}")

    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return float("nan")
    cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def angular_difference_deg(l1: float, l2: float) -> float:
    return ((l1 - l2 + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Dipole estimators
# ---------------------------------------------------------------------------


def compute_dipole(
    l_deg: np.ndarray, b_deg: np.ndarray, weights: Optional[np.ndarray] = None
) -> Tuple[float, float, float, np.ndarray]:
    if weights is None:
        weights = np.ones(len(l_deg), dtype=float)
    weights = np.asarray(weights, dtype=float)
    unit = lb_to_unitvec(l_deg, b_deg)
    sum_vec = np.sum(unit * weights[:, None], axis=0)
    wtot = float(np.sum(weights))
    amp = 3.0 * float(np.linalg.norm(sum_vec)) / wtot if wtot > 0 else float("nan")
    l_dip, b_dip = unitvec_to_lb(sum_vec)
    return amp, l_dip, b_dip, sum_vec


def bootstrap_dipole(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
    weights: Optional[np.ndarray] = None,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    l = np.asarray(l_deg, dtype=float)
    b = np.asarray(b_deg, dtype=float)
    w = np.ones_like(l) if weights is None else np.asarray(weights, dtype=float)
    n = len(l)
    if n == 0:
        return BootstrapResult(
            amplitude_q16=float("nan"),
            amplitude_q50=float("nan"),
            amplitude_q84=float("nan"),
            amplitude_std=float("nan"),
            l_median=float("nan"),
            b_median=float("nan"),
            direction_scatter_q16=float("nan"),
            direction_scatter_q50=float("nan"),
            direction_scatter_q84=float("nan"),
            direction_sigma_deg=float("nan"),
            n_bootstrap=n_bootstrap,
        )

    probs = w / w.sum()
    amp_samples: List[float] = []
    l_samples: List[float] = []
    b_samples: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True, p=probs)
        D, l_d, b_d, _ = compute_dipole(l[idx], b[idx], w[idx])
        amp_samples.append(D)
        l_samples.append(l_d)
        b_samples.append(b_d)

    amps = np.asarray(amp_samples)
    lons = np.asarray(l_samples)
    lats = np.asarray(b_samples)

    med_l = float(np.median(lons))
    med_b = float(np.median(lats))
    med_vec = lb_to_unitvec(np.array([med_l]), np.array([med_b]))[0]

    seps = [angle_deg(lb_to_unitvec([lv], [bv])[0], med_vec) for lv, bv in zip(lons, lats)]
    seps_arr = np.asarray(seps)

    return BootstrapResult(
        amplitude_q16=float(np.percentile(amps, 16)),
        amplitude_q50=float(np.percentile(amps, 50)),
        amplitude_q84=float(np.percentile(amps, 84)),
        amplitude_std=float(np.std(amps)),
        l_median=med_l,
        b_median=med_b,
        direction_scatter_q16=float(np.percentile(seps_arr, 16)),
        direction_scatter_q50=float(np.percentile(seps_arr, 50)),
        direction_scatter_q84=float(np.percentile(seps_arr, 84)),
        direction_sigma_deg=float(np.percentile(seps_arr, 68)),
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Catalog + cuts
# ---------------------------------------------------------------------------


def apply_baseline_cuts(
    tbl: Table,
    b_cut: float = 30.0,
    w1cov_min: float = 80.0,
    w1_max: float = 16.4,
    w1_min: Optional[float] = None,
    existing_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    n_rows = len(tbl)
    mask = np.ones(n_rows, dtype=bool) if existing_mask is None else existing_mask.copy()
    cuts: List[Dict[str, Any]] = []

    w1 = np.asarray(tbl["w1"], dtype=float)
    b = np.asarray(tbl["b"], dtype=float)
    w1cov = np.asarray(tbl["w1cov"], dtype=float) if "w1cov" in tbl.colnames else None

    if w1cov is not None and w1cov_min is not None:
        cut = w1cov >= w1cov_min
        mask &= cut
        cuts.append(
            {
                "name": "W1 coverage",
                "condition": f"w1cov >= {w1cov_min}",
                "N_after": int(mask.sum()),
            }
        )

    gal_cut = np.abs(b) > b_cut
    mask &= gal_cut
    cuts.append(
        {
            "name": "Galactic latitude",
            "condition": f"|b| > {b_cut} deg",
            "N_after": int(mask.sum()),
        }
    )

    if w1_min is not None:
        min_cut = w1 >= w1_min
        mask &= min_cut
        cuts.append(
            {
                "name": "W1 minimum",
                "condition": f"W1 >= {w1_min}",
                "N_after": int(mask.sum()),
            }
        )

    if w1_max is not None:
        max_cut = w1 <= w1_max
        mask &= max_cut
        cuts.append(
            {
                "name": "W1 maximum",
                "condition": f"W1 <= {w1_max}",
                "N_after": int(mask.sum()),
            }
        )

    cuts.append(
        {
            "name": "Summary",
            "condition": "Baseline cuts (|b|, w1cov, W1 limits)",
            "N_before": n_rows,
            "N_after": int(mask.sum()),
        }
    )

    return mask, cuts
