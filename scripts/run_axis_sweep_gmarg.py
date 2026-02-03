#!/usr/bin/env python3
"""Run a large fixed-axis g-scan sweep with g-marginalization.

This is a convenience launcher that:
  1) Generates an axis list (named + random) and writes axes.json
  2) Runs `scripts/run_darksiren_fixed_axis_gscan_full.py` for each axis with bounded concurrency
  3) Aggregates results via `scripts/aggregate_axis_sweep_gmarg.py`
  4) Optionally re-runs axes that hit g-grid boundaries (posterior edge mass) with an extended grid

Designed for long runs: write per-axis logs and be resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class AxisJob:
    kind: Literal["special", "random"]
    name: str
    preset: str | None
    frame: str
    lon_deg: float
    lat_deg: float


def utc_tag() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def parse_g_grid(spec: str) -> list[float]:
    s = str(spec).strip()
    if not s:
        raise ValueError("g-grid spec is empty")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 3:
        gmin, gmax, step = (float(x) for x in parts)
        if step <= 0:
            raise ValueError("g step must be > 0")
        n = int(np.floor((gmax - gmin) / step + 0.5)) + 1
        grid = gmin + step * np.arange(n, dtype=float)
        grid = grid[(grid >= gmin - 1e-12) & (grid <= gmax + 1e-12)]
        return [float(x) for x in np.unique(grid).tolist()]
    return [float(x) for x in np.unique(np.array([float(x) for x in parts], dtype=float)).tolist()]


def make_axes(*, n_random: int, seed: int) -> list[AxisJob]:
    rng = np.random.default_rng(int(seed))

    axes: list[AxisJob] = [
        AxisJob(kind="special", name="cmb", preset="cmb", frame="galactic", lon_deg=264.021, lat_deg=48.253),
        AxisJob(kind="special", name="secrest", preset="secrest", frame="galactic", lon_deg=236.01, lat_deg=28.77),
        AxisJob(kind="special", name="ecliptic_north", preset="ecliptic_north", frame="barycentricmeanecliptic", lon_deg=0.0, lat_deg=90.0),
    ]

    u = rng.uniform(-1.0, 1.0, size=int(n_random))
    dec = np.degrees(np.arcsin(u))
    ra = rng.uniform(0.0, 360.0, size=int(n_random))

    for i in range(int(n_random)):
        axes.append(
            AxisJob(
                kind="random",
                name=f"rand{i+1:04d}",
                preset=None,
                frame="icrs",
                lon_deg=float(ra[i]),
                lat_deg=float(dec[i]),
            )
        )
    return axes


def axis_outdir(root: Path, name: str) -> Path:
    return root / name


def build_cmd(
    *,
    axis: AxisJob,
    outdir: Path,
    g_grid: str,
    nproc: int,
    g_prior_type: str,
    g_prior_mu: float,
    g_prior_sigma: float,
    g_prior_uniform_min: float,
    g_prior_uniform_max: float,
    max_draws: int,
) -> list[str]:
    cmd = [
        os.path.join(os.getcwd(), ".venv", "bin", "python"),
        "scripts/run_darksiren_fixed_axis_gscan_full.py",
        f"--g-grid={str(g_grid)}",
        "--nproc",
        str(int(nproc)),
        "--g-prior-type",
        str(g_prior_type),
        "--g-prior-mu",
        str(float(g_prior_mu)),
        "--g-prior-sigma",
        str(float(g_prior_sigma)),
        "--g-prior-uniform-min",
        str(float(g_prior_uniform_min)),
        "--g-prior-uniform-max",
        str(float(g_prior_uniform_max)),
        "--outdir",
        str(outdir),
    ]
    if int(max_draws) > 0:
        cmd += ["--max-draws", str(int(max_draws))]
    if axis.preset:
        cmd += ["--axis", axis.preset]
    else:
        cmd += [
            "--axis",
            "",
            "--axis-name",
            axis.name,
            "--axis-frame",
            axis.frame,
            "--axis-lon-deg",
            str(float(axis.lon_deg)),
            "--axis-lat-deg",
            str(float(axis.lat_deg)),
        ]
    return cmd


def should_extend_axis(json_path: Path, *, edge_thresh: float) -> bool:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    gm = obj.get("g_marginalization", {})
    if not gm:
        raise RuntimeError(f"Missing g_marginalization in {json_path}")
    if bool(gm.get("best_is_edge", False)):
        return True
    if float(gm.get("edge_mass_low", 0.0)) > float(edge_thresh):
        return True
    if float(gm.get("edge_mass_high", 0.0)) > float(edge_thresh):
        return True
    return False


def run_many(jobs: list[AxisJob], *, root: Path, cmd_kwargs: dict[str, Any], max_in_flight: int) -> None:
    pending = list(jobs)
    running: list[tuple[AxisJob, subprocess.Popen[bytes], Path]] = []
    done = 0
    total = len(jobs)

    def _start_one(job: AxisJob) -> None:
        nonlocal running
        outdir = axis_outdir(root, job.name)
        outdir.mkdir(parents=True, exist_ok=True)
        out_json = outdir / "fixed_axis_gscan_full.json"
        if out_json.exists():
            print(f"[axis_sweep] SKIP {job.name} (exists)", flush=True)
            return

        cmd = build_cmd(axis=job, outdir=outdir, **cmd_kwargs)
        log_path = outdir / "run.log"
        f = open(log_path, "wb")  # noqa: P201
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)  # noqa: S603
        running.append((job, p, log_path))
        print(f"[axis_sweep] START {job.name} pid={p.pid}", flush=True)

    # Prime the queue.
    while pending and len(running) < int(max_in_flight):
        _start_one(pending.pop(0))

    while running:
        # Poll for completion.
        time.sleep(1.0)
        still = []
        for job, p, log_path in running:
            rc = p.poll()
            if rc is None:
                still.append((job, p, log_path))
                continue
            done += 1
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[axis_sweep] DONE {done}/{total} {job.name} {status} log={log_path}", flush=True)
            if rc != 0:
                raise RuntimeError(f"Axis job failed: {job.name} (rc={rc}) log={log_path}")
        running = still
        while pending and len(running) < int(max_in_flight):
            _start_one(pending.pop(0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="Sweep output root directory.")
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-in-flight", type=int, default=6, help="Concurrent axis runs.")

    ap.add_argument("--g-grid", type=str, default="-0.6,0.6,0.1")
    ap.add_argument("--g-grid-extend", type=str, default="-1.0,1.0,0.1")
    ap.add_argument("--edge-mass-thresh", type=float, default=0.01)

    ap.add_argument("--nproc", type=int, default=20, help="Per-axis event-parallel workers.")
    ap.add_argument("--max-draws", type=int, default=0, help="0 means full draw set from summary.")

    ap.add_argument("--g-prior-type", type=str, default="normal", choices=["normal", "uniform"])
    ap.add_argument("--g-prior-mu", type=float, default=0.0)
    ap.add_argument("--g-prior-sigma", type=float, default=0.2)
    ap.add_argument("--g-prior-uniform-min", type=float, default=-1.0)
    ap.add_argument("--g-prior-uniform-max", type=float, default=1.0)

    args = ap.parse_args()

    root = Path(args.outdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    axes = make_axes(n_random=int(args.n_random), seed=int(args.seed))
    write_json(
        root / "axes.json",
        {
            "timestamp_utc": utc_tag(),
            "seed": int(args.seed),
            "n_random": int(args.n_random),
            "axes": [
                {
                    "kind": a.kind,
                    "name": a.name,
                    "preset": a.preset,
                    "frame": a.frame,
                    "lon_deg": float(a.lon_deg),
                    "lat_deg": float(a.lat_deg),
                }
                for a in axes
            ],
        },
    )
    print(f"[axis_sweep] Wrote {root/'axes.json'}", flush=True)

    cmd_kwargs = dict(
        g_grid=str(args.g_grid),
        nproc=int(args.nproc),
        g_prior_type=str(args.g_prior_type),
        g_prior_mu=float(args.g_prior_mu),
        g_prior_sigma=float(args.g_prior_sigma),
        g_prior_uniform_min=float(args.g_prior_uniform_min),
        g_prior_uniform_max=float(args.g_prior_uniform_max),
        max_draws=int(args.max_draws),
    )

    # Pass 1.
    print(f"[axis_sweep] PASS1 g_grid={args.g_grid}", flush=True)
    run_many(axes, root=root, cmd_kwargs=cmd_kwargs, max_in_flight=int(args.max_in_flight))

    # Aggregate pass 1.
    subprocess.run(  # noqa: S603
        [
            os.path.join(os.getcwd(), ".venv", "bin", "python"),
            "scripts/aggregate_axis_sweep_gmarg.py",
            "--root",
            str(root),
            "--out",
            "axis_sweep_summary_pass1.json",
            "--edge-mass-thresh",
            str(float(args.edge_mass_thresh)),
        ],
        check=True,
    )

    # Boundary extension pass (only if needed).
    edge_thresh = float(args.edge_mass_thresh)
    need_ext = []
    for ax in axes:
        out_json = axis_outdir(root, ax.name) / "fixed_axis_gscan_full.json"
        if not out_json.exists():
            raise RuntimeError(f"Missing output for axis {ax.name}: {out_json}")
        if should_extend_axis(out_json, edge_thresh=edge_thresh):
            need_ext.append(ax)

    if need_ext:
        print(f"[axis_sweep] PASS2 extend {len(need_ext)} axes with g_grid={args.g_grid_extend}", flush=True)
        # Move existing json/png aside for reproducibility.
        for ax in need_ext:
            d = axis_outdir(root, ax.name)
            src = d / "fixed_axis_gscan_full.json"
            dst = d / "fixed_axis_gscan_full_initial.json"
            if src.exists() and not dst.exists():
                src.replace(dst)
            srcp = d / "fixed_axis_gscan_full.png"
            dstp = d / "fixed_axis_gscan_full_initial.png"
            if srcp.exists() and not dstp.exists():
                srcp.replace(dstp)
        cmd_kwargs_ext = dict(cmd_kwargs)
        cmd_kwargs_ext["g_grid"] = str(args.g_grid_extend)
        run_many(need_ext, root=root, cmd_kwargs=cmd_kwargs_ext, max_in_flight=int(args.max_in_flight))

    # Final aggregate.
    subprocess.run(  # noqa: S603
        [
            os.path.join(os.getcwd(), ".venv", "bin", "python"),
            "scripts/aggregate_axis_sweep_gmarg.py",
            "--root",
            str(root),
            "--out",
            "axis_sweep_summary.json",
            "--edge-mass-thresh",
            str(float(args.edge_mass_thresh)),
        ],
        check=True,
    )
    print(f"[axis_sweep] DONE root={root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
