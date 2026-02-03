#!/usr/bin/env python3
"""
Quick smoke test for `scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py`.

Creates a tiny synthetic FITS catalog (uniform sky, no masks) and runs the Poisson
GLM scan in both cumulative and differential modes, with jackknife enabled.

This is meant to be fast and to catch obvious runtime/syntax regressions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


@dataclass(frozen=True)
class RunResult:
    outdir: Path
    json_path: Path


def write_synth_catalog(path: Path, *, n: int = 8000, seed: int = 0) -> None:
    from astropy.table import Table

    rng = np.random.default_rng(int(seed))
    # Uniform on sphere.
    u = rng.uniform(0.0, 1.0, size=n)
    v = rng.uniform(0.0, 1.0, size=n)
    l = 360.0 * u
    b = np.degrees(np.arcsin(2.0 * v - 1.0))
    w1 = rng.uniform(15.0, 16.0, size=n)
    w1cov = np.full(n, 100.0)
    tab = Table(
        {
            "w1": w1.astype(np.float32),
            "w1cov": w1cov.astype(np.float32),
            "l": l.astype(np.float32),
            "b": b.astype(np.float32),
        }
    )
    tab.write(path, overwrite=True)


def run_scan(*, catalog: Path, outdir: Path, w1_mode: str) -> RunResult:
    cmd = [
        sys.executable,
        "scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py",
        "--catalog",
        str(catalog),
        "--exclude-mask-fits",
        "",
        "--outdir",
        str(outdir),
        "--nside",
        "2",
        "--b-cut",
        "0",
        "--w1cov-min",
        "0",
        "--w1-grid",
        "15.2,15.6,0.2",
        "--mc-draws",
        "20",
        "--max-iter",
        "80",
        "--w1-mode",
        str(w1_mode),
        "--jackknife-nside",
        "1",
        "--jackknife-stride",
        "1",
        "--jackknife-max-iter",
        "40",
        "--jackknife-max-regions",
        "12",
    ]
    subprocess.run(cmd, check=True)
    json_path = outdir / "rvmp_fig5_poisson_glm.json"
    if not json_path.exists():
        raise RuntimeError(f"Missing output JSON: {json_path}")
    return RunResult(outdir=outdir, json_path=json_path)


def sanity_check(json_path: Path) -> None:
    d = json.loads(json_path.read_text())
    rows = d.get("rows") or []
    if not rows:
        raise RuntimeError("No rows produced.")
    r0 = rows[0]
    for key in ("dipole", "dipole_quasi", "fit_diag", "template_dipoles"):
        if key not in r0:
            raise RuntimeError(f"Missing key in first row: {key}")
    if r0["fit_diag"].get("pearson_over_dof") is None:
        raise RuntimeError("Missing pearson_over_dof in fit_diag.")


def main() -> int:
    root = Path.cwd()
    outroot = root / "outputs" / f"smoke_rvmp_fig5_poisson_glm_{utc_tag()}"
    outroot.mkdir(parents=True, exist_ok=True)
    catalog = outroot / "synth_catwise.fits"
    write_synth_catalog(catalog, n=8000, seed=0)

    res1 = run_scan(catalog=catalog, outdir=outroot / "cumulative", w1_mode="cumulative")
    sanity_check(res1.json_path)

    res2 = run_scan(catalog=catalog, outdir=outroot / "differential", w1_mode="differential")
    sanity_check(res2.json_path)

    print(f"OK: {res1.json_path}")
    print(f"OK: {res2.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

