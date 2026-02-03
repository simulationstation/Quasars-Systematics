from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used throughout the pipeline.

    Units:
      - c_km_s: km / s
    """

    c_km_s: float = 299_792.458

