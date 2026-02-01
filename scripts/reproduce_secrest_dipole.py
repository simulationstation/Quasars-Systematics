#!/usr/bin/env python3
"""Reproduce a Secrest-style CatWISE quasar number-count dipole from the released catalog.

This script is intentionally simple and self-contained:
  - loads the Secrest+22 accepted CatWISE AGN catalog (FITS)
  - applies the standard baseline cuts (|b| > 30 deg, w1cov >= 80, w1 <= W1_MAX)
  - computes the vector-sum dipole estimator (3 * mean(unit_vector))
  - optionally bootstraps the dipole for a quick internal uncertainty estimate

It writes a `dipole.json` artifact compatible with the other analysis scripts in this repository.

Reference:
  Secrest et al. 2022, ApJL 937 L31, DOI: 10.3847/2041-8213/ac88c0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from secrest_utils import (
    SECREST_PUBLISHED,
    angle_deg,
    apply_baseline_cuts,
    bootstrap_dipole,
    compute_dipole,
    lb_to_unitvec,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--catalog",
        required=True,
        help="Path to Secrest+22 accepted CatWISE AGN FITS (expects l,b,w1,w1cov and related columns).",
    )
    ap.add_argument("--outdir", default="outputs/secrest_reproduction", help="Output directory.")
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-min", type=float, default=None)
    ap.add_argument("--w1-max", type=float, default=16.4)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tbl = Table.read(args.catalog, memmap=True)
    mask, cuts = apply_baseline_cuts(
        tbl,
        b_cut=float(args.b_cut),
        w1cov_min=float(args.w1cov_min),
        w1_max=float(args.w1_max),
        w1_min=args.w1_min,
        existing_mask=None,
    )

    l = np.asarray(tbl["l"], float)[mask]
    b = np.asarray(tbl["b"], float)[mask]

    D, l_d, b_d, sum_vec = compute_dipole(l, b)
    boot = bootstrap_dipole(l, b, n_bootstrap=int(args.bootstrap), seed=int(args.seed)).as_dict()

    pub_vec = lb_to_unitvec(SECREST_PUBLISHED["l_deg"], SECREST_PUBLISHED["b_deg"])[0]
    rec_vec = np.asarray(sum_vec, float).reshape(3)
    sep_from_pub = angle_deg(rec_vec, pub_vec)

    out = {
        "catalog": str(args.catalog),
        "cuts": {
            "b_cut_deg": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_min": args.w1_min,
            "w1_max": float(args.w1_max),
            "cuts_applied": cuts,
        },
        "N_final": int(mask.sum()),
        "dipole": {"amplitude": float(D), "l_deg": float(l_d), "b_deg": float(b_d)},
        "bootstrap": boot,
        "comparison_to_published": {
            "published": SECREST_PUBLISHED,
            "angular_separation_from_published_deg": float(sep_from_pub),
        },
        "notes": (
            "This is a simple vector-sum dipole estimator for internal consistency checks. "
            "Secrest et al. use a specific estimator/weighting; do not interpret small amplitude "
            "differences as physical without matching the full published pipeline."
        ),
    }

    with open(outdir / "dipole.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

