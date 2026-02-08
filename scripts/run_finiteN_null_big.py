#!/usr/bin/env python3
"""Large finite-N null test with periodic checkpoints.

Runs two nulls for CatWISE-parent epoch amplitudes (epochs 0..15):
1) Gaussian constant-D null using reported per-epoch sigma_D values.
2) Map-level multinomial + Poisson-GLM null using pooled sky pattern.

Writes a progress JSON repeatedly so partial results are preserved if
execution is interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_unwise_time_domain_catwise_epoch_amplitude_report import (
    build_secrest_seen_mask,
)
from scripts.run_unwise_time_domain_catwise_epoch_dipole import fit_poisson_glm_dipole


def _clopper_upper_zero(n: int, alpha: float = 0.05) -> float:
    return 1.0 - (alpha ** (1.0 / float(n)))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report-json",
        default="REPORTS/unwise_time_domain_catwise_epoch_amplitude/data/epoch_amplitude.json",
    )
    ap.add_argument(
        "--run-dir",
        default="outputs/epoch_dipole_time_domain_catwise_parent_20260204_232537UTC",
    )
    ap.add_argument(
        "--out-json",
        default="outputs/epoch_dipole_time_domain_catwise_parent_20260204_232537UTC/finiteN_null_big_v2.json",
    )
    ap.add_argument(
        "--progress-json",
        default="outputs/epoch_dipole_time_domain_catwise_parent_20260204_232537UTC/finiteN_null_big_v2.progress.json",
    )
    ap.add_argument("--seed", type=int, default=20260207)
    ap.add_argument("--nsim-gauss", type=int, default=10_000_000)
    ap.add_argument("--chunk-gauss", type=int, default=250_000)
    ap.add_argument("--gauss-checkpoint", type=int, default=1_000_000)
    ap.add_argument("--nsim-glm", type=int, default=20_000)
    ap.add_argument("--glm-checkpoint", type=int, default=250)
    args = ap.parse_args()

    report_json = Path(args.report_json)
    run_dir = Path(args.run_dir)
    out_json = Path(args.out_json)
    progress_json = Path(args.progress_json)

    d = json.loads(report_json.read_text())
    rows = [r for r in d["epochs"] if int(r["epoch"]) <= 15]
    N = np.array([int(r["N"]) for r in rows], dtype=np.int64)
    D_obs = np.array([float(r["D_glm"]) for r in rows], dtype=float)
    s_obs = np.array([float(r.get("sigma_D_glm", r.get("sigma_D"))) for r in rows], dtype=float)

    w = 1.0 / (s_obs**2)
    wsum = float(w.sum())
    D0 = float((w * D_obs).sum() / wsum)
    obs_chi2 = float(((D_obs - D0) ** 2 * w).sum())
    obs_dof = int(D_obs.size - 1)
    obs_p = float(chi2.sf(obs_chi2, obs_dof))
    obs_range = float(D_obs.max() - D_obs.min())
    obs_std = float(D_obs.std())
    i_obs_min = int(np.argmin(D_obs))
    i_obs_max = int(np.argmax(D_obs))
    obs_pairz = float(obs_range / np.sqrt(s_obs[i_obs_min] ** 2 + s_obs[i_obs_max] ** 2))

    rng = np.random.default_rng(int(args.seed))

    g_count_range = 0
    g_count_std = 0
    g_count_chi2 = 0
    g_done = 0
    t0 = time.time()

    while g_done < int(args.nsim_gauss):
        m = min(int(args.chunk_gauss), int(args.nsim_gauss) - g_done)
        z = rng.normal(size=(m, D_obs.size))
        Dm = D0 + z * s_obs[None, :]
        ranges = Dm.max(axis=1) - Dm.min(axis=1)
        stds = Dm.std(axis=1)
        means_w = (Dm * w[None, :]).sum(axis=1) / wsum
        ch = ((Dm - means_w[:, None]) ** 2 * w[None, :]).sum(axis=1)

        g_count_range += int(np.count_nonzero(ranges >= obs_range))
        g_count_std += int(np.count_nonzero(stds >= obs_std))
        g_count_chi2 += int(np.count_nonzero(ch >= obs_chi2))
        g_done += m

        if g_done % int(args.gauss_checkpoint) == 0 or g_done == int(args.nsim_gauss):
            dt = time.time() - t0
            payload = {
                "stage": "gaussian",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seed": int(args.seed),
                "observed": {
                    "D_range": obs_range,
                    "D_std": obs_std,
                    "chi2_const": obs_chi2,
                    "dof": obs_dof,
                    "p_const": obs_p,
                },
                "gaussian_progress": {
                    "done": int(g_done),
                    "total": int(args.nsim_gauss),
                    "seconds": float(dt),
                    "sim_per_sec": float(g_done / max(dt, 1e-9)),
                    "count_range_ge_obs": int(g_count_range),
                    "count_std_ge_obs": int(g_count_std),
                    "count_chi2_ge_obs": int(g_count_chi2),
                    "p_range_ge_obs": float(g_count_range / g_done),
                    "p_std_ge_obs": float(g_count_std / g_done),
                    "p_chi2_ge_obs": float(g_count_chi2 / g_done),
                    "p_upper95_if_zero": float(_clopper_upper_zero(g_done)),
                },
            }
            _write_json(progress_json, payload)
            print(
                f"[gauss] {g_done}/{int(args.nsim_gauss)} "
                f"p_range={g_count_range/g_done:.3e} p_chi2={g_count_chi2/g_done:.3e}",
                flush=True,
            )

    cfg = json.loads((run_dir / "run_config.json").read_text())
    counts = np.load(run_dir / "counts_by_epoch.npy")
    seen = build_secrest_seen_mask(
        nside=int(cfg["nside"]),
        catwise_catalog=Path(
            "data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits"
        ),
        exclude_mask_fits=Path(
            "data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits"
        ),
        b_cut_deg=float(cfg["b_cut_deg"]),
        w1cov_min=float(cfg["w1cov_min"]),
    )
    npix = hp.nside2npix(int(cfg["nside"]))
    xyz = np.vstack(hp.pix2vec(int(cfg["nside"]), np.arange(npix), nest=False)).T
    nhat = xyz[seen]
    Y = counts[:16][:, seen]
    p_pix = Y.sum(axis=0).astype(float)
    p_pix /= p_pix.sum()

    m_count_range = 0
    m_count_std = 0
    m_count_chi2 = 0
    m_count_pairz = 0
    m_done = 0
    t1 = time.time()

    for i in range(1, int(args.nsim_glm) + 1):
        Dsim = np.empty(16, dtype=float)
        for k, n in enumerate(N):
            y = rng.multinomial(int(n), p_pix)
            beta, _ = fit_poisson_glm_dipole(y, nhat)
            Dsim[k] = float(np.linalg.norm(beta[1:]))

        r = float(Dsim.max() - Dsim.min())
        sd = float(Dsim.std())
        Dbar = float((Dsim * w).sum() / wsum)
        ch = float(((Dsim - Dbar) ** 2 * w).sum())
        i0 = int(np.argmin(Dsim))
        i1 = int(np.argmax(Dsim))
        pairz = float((Dsim[i1] - Dsim[i0]) / np.sqrt(s_obs[i0] ** 2 + s_obs[i1] ** 2))

        m_count_range += int(r >= obs_range)
        m_count_std += int(sd >= obs_std)
        m_count_chi2 += int(ch >= obs_chi2)
        m_count_pairz += int(pairz >= obs_pairz)
        m_done = i

        if i % int(args.glm_checkpoint) == 0 or i == int(args.nsim_glm):
            dt = time.time() - t1
            payload = {
                "stage": "glm_map",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seed": int(args.seed),
                "observed": {
                    "D_range": obs_range,
                    "D_std": obs_std,
                    "chi2_const": obs_chi2,
                    "dof": obs_dof,
                    "p_const": obs_p,
                    "pairz_minmax": obs_pairz,
                },
                "gaussian_final": {
                    "done": int(g_done),
                    "total": int(args.nsim_gauss),
                    "count_range_ge_obs": int(g_count_range),
                    "count_std_ge_obs": int(g_count_std),
                    "count_chi2_ge_obs": int(g_count_chi2),
                    "p_range_ge_obs": float(g_count_range / max(g_done, 1)),
                    "p_std_ge_obs": float(g_count_std / max(g_done, 1)),
                    "p_chi2_ge_obs": float(g_count_chi2 / max(g_done, 1)),
                    "p_upper95_if_zero": float(_clopper_upper_zero(max(g_done, 1))),
                },
                "glm_progress": {
                    "done": int(m_done),
                    "total": int(args.nsim_glm),
                    "seconds": float(dt),
                    "sim_per_sec": float(m_done / max(dt, 1e-9)),
                    "count_range_ge_obs": int(m_count_range),
                    "count_std_ge_obs": int(m_count_std),
                    "count_chi2_ge_obs": int(m_count_chi2),
                    "count_pairz_ge_obs": int(m_count_pairz),
                    "p_range_ge_obs": float(m_count_range / m_done),
                    "p_std_ge_obs": float(m_count_std / m_done),
                    "p_chi2_ge_obs": float(m_count_chi2 / m_done),
                    "p_pairz_ge_obs": float(m_count_pairz / m_done),
                    "p_upper95_if_zero": float(_clopper_upper_zero(m_done)),
                },
            }
            _write_json(progress_json, payload)
            print(
                f"[glm] {m_done}/{int(args.nsim_glm)} "
                f"p_range={m_count_range/m_done:.3e} p_chi2={m_count_chi2/m_done:.3e}",
                flush=True,
            )

    final = {
        "meta": {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": int(args.seed),
        },
        "observed": {
            "n_epochs": int(D_obs.size),
            "N_min": int(N.min()),
            "N_max": int(N.max()),
            "D_min": float(D_obs.min()),
            "D_max": float(D_obs.max()),
            "D_range": obs_range,
            "D_std": obs_std,
            "D0_weighted": D0,
            "chi2_const": obs_chi2,
            "dof": obs_dof,
            "p_const": obs_p,
            "pairz_minmax": obs_pairz,
            "sigma_mean": float(s_obs.mean()),
            "sqrt3_over_N_mean": float(np.sqrt(3.0 / N).mean()),
        },
        "gaussian_null": {
            "nsim": int(g_done),
            "count_range_ge_obs": int(g_count_range),
            "count_std_ge_obs": int(g_count_std),
            "count_chi2_ge_obs": int(g_count_chi2),
            "p_range_ge_obs": float(g_count_range / max(g_done, 1)),
            "p_std_ge_obs": float(g_count_std / max(g_done, 1)),
            "p_chi2_ge_obs": float(g_count_chi2 / max(g_done, 1)),
            "p_upper95_if_zero": float(_clopper_upper_zero(max(g_done, 1))),
        },
        "glm_map_null": {
            "nsim": int(m_done),
            "count_range_ge_obs": int(m_count_range),
            "count_std_ge_obs": int(m_count_std),
            "count_chi2_ge_obs": int(m_count_chi2),
            "count_pairz_ge_obs": int(m_count_pairz),
            "p_range_ge_obs": float(m_count_range / max(m_done, 1)),
            "p_std_ge_obs": float(m_count_std / max(m_done, 1)),
            "p_chi2_ge_obs": float(m_count_chi2 / max(m_done, 1)),
            "p_pairz_ge_obs": float(m_count_pairz / max(m_done, 1)),
            "p_upper95_if_zero": float(_clopper_upper_zero(max(m_done, 1))),
        },
    }
    _write_json(out_json, final)
    _write_json(progress_json, final)
    print(f"wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
