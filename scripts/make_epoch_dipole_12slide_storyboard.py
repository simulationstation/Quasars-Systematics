#!/usr/bin/env python3
"""Build a 12-panel storyboard of epoch dipole direction + amplitude.

Each panel ("slide") shows:
- Dipole direction in Galactic coordinates for one epoch.
- Overlap trajectory lines (all epochs in light gray, cumulative in blue).
- Timestamp and amplitude metadata.
- Inset amplitude-vs-time trace up to the current slide.
- Marker color/size encoded by a luminosity proxy (log10 of selected N).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def mjd_to_datestr(mjd: float) -> str:
    t0 = datetime(1858, 11, 17, tzinfo=timezone.utc)
    dt = t0 + timedelta(days=float(mjd))
    return dt.strftime("%Y-%m-%d")


def beta_to_gal_lonlat(beta: list[float]) -> tuple[float, float, float]:
    v = np.asarray(beta[1:4], dtype=float)
    amp = float(np.linalg.norm(v))
    if not np.isfinite(amp) or amp <= 0.0:
        return float("nan"), float("nan"), float("nan")
    x, y, z = v / amp
    lon = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    # Wrap longitude to [-180, 180] for compact plotting.
    lon = ((lon + 180.0) % 360.0) - 180.0
    return float(lon), float(lat), amp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary-json",
        default="outputs/epoch_dipole_time_domain_catwise_parent_20260204_232537UTC/summary.json",
        help="Path to run summary JSON that contains per-epoch beta/D/N/MJD.",
    )
    ap.add_argument(
        "--out",
        default="REPORTS/unwise_time_domain_catwise_epoch_amplitude/figures/dipole_12slide_storyboard.png",
        help="Output image path.",
    )
    ap.add_argument("--n-slides", type=int, default=12, help="Number of panels/slides to render.")
    ap.add_argument("--max-epoch", type=int, default=11, help="Highest epoch index to include.")
    args = ap.parse_args()

    summary_path = Path(args.summary_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(summary_path.read_text())
    rows = sorted(data.get("epochs", []), key=lambda r: int(r["epoch"]))
    rows = [r for r in rows if int(r["epoch"]) <= int(args.max_epoch)]
    rows = rows[: int(args.n_slides)]
    if len(rows) == 0:
        raise SystemExit(f"No usable rows found in {summary_path}")

    epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
    mjd = np.array([float(r["mjd_mean"]) for r in rows], dtype=float)
    date = np.array([mjd_to_datestr(m) for m in mjd], dtype=object)
    d_amp = np.array([float(r["D"]) for r in rows], dtype=float)
    n_sel = np.array([int(r["N"]) for r in rows], dtype=float)

    lon = np.empty(len(rows), dtype=float)
    lat = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        lo, la, _ = beta_to_gal_lonlat(r["beta"])
        lon[i] = lo
        lat[i] = la

    lum_proxy = np.log10(np.clip(n_sel, 1.0, np.inf))
    lum_min = float(np.nanmin(lum_proxy))
    lum_max = float(np.nanmax(lum_proxy))
    if np.isclose(lum_min, lum_max):
        lum_max = lum_min + 1.0
    lum_norm = (lum_proxy - lum_min) / (lum_max - lum_min)

    n = len(rows)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 14), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    cmap = plt.get_cmap("plasma")

    y0 = float(np.nanmin(d_amp) - 0.01)
    y1 = float(np.nanmax(d_amp) + 0.01)
    x_all = np.arange(n, dtype=int)

    for idx in range(nrows * ncols):
        ax = axes.flat[idx]
        if idx >= n:
            ax.axis("off")
            continue

        # Full overlap trajectory (all 12 epochs) in each panel.
        ax.plot(lon, lat, color="0.82", lw=1.2, alpha=1.0, zorder=0)
        ax.scatter(lon, lat, color="0.85", s=16, zorder=0)

        # Cumulative trajectory up to this slide.
        ax.plot(lon[: idx + 1], lat[: idx + 1], color="#1f77b4", lw=2.0, zorder=2)
        size = 80.0 + 210.0 * float(lum_norm[idx])
        ax.scatter(
            lon[idx],
            lat[idx],
            c=[lum_norm[idx]],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=size,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

        ax.set_xlim(180.0, -180.0)  # sky-style direction convention
        ax.set_ylim(-90.0, 90.0)
        ax.set_xticks([180, 120, 60, 0, -60, -120, -180])
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.grid(alpha=0.25, lw=0.6)

        if idx // ncols == nrows - 1:
            ax.set_xlabel("Galactic longitude l [deg]")
        if idx % ncols == 0:
            ax.set_ylabel("Galactic latitude b [deg]")

        ax.text(
            0.02,
            0.96,
            f"Slide {idx+1}/{n}  |  Epoch {epoch[idx]}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            weight="bold",
        )
        ax.text(
            0.02,
            0.86,
            f"{date[idx]}  (MJD {mjd[idx]:.1f})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.text(
            0.02,
            0.77,
            f"D = {d_amp[idx]:.5f}\nN = {int(n_sel[idx]):,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

        # Inset: amplitude-vs-time with current epoch highlighted.
        inset = ax.inset_axes([0.54, 0.05, 0.43, 0.28])
        inset.plot(x_all, d_amp, color="0.75", lw=1.0, zorder=0)
        inset.scatter(x_all, d_amp, color="0.65", s=10, zorder=1)
        inset.plot(x_all[: idx + 1], d_amp[: idx + 1], color="#d62728", lw=1.6, zorder=2)
        inset.scatter([idx], [d_amp[idx]], color="#d62728", s=20, zorder=3)
        inset.set_xlim(-0.3, n - 0.7)
        inset.set_ylim(y0, y1)
        inset.set_title("Amplitude D vs time", fontsize=6, pad=1.5)
        inset.set_xticks([0, n - 1])
        inset.set_xticklabels([str(epoch[0]), str(epoch[-1])], fontsize=6)
        inset.tick_params(axis="y", labelsize=6, length=2)
        inset.tick_params(axis="x", labelsize=6, length=2)
        inset.grid(alpha=0.2, lw=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lum_min, vmax=lum_max))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Luminosity indicator proxy: log10(N selected per epoch)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Epoch Dipole Storyboard (12 slides): direction drift, amplitude timeline, timestamps, and overlap lines",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Light-gray path in every panel = full 12-epoch trajectory. Blue path = trajectory up to current slide.",
        ha="center",
        fontsize=10,
    )

    fig.savefig(out_path, dpi=220)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
