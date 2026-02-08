#!/usr/bin/env python3
"""
Analyze the NEOWISE explanatory supplement "Table 2" photometric-stability time series.

This is a *global-in-time* diagnostic (not a sky map): it estimates the typical W1/W2
zero-point drift amplitude in magnitudes over the mission timeline. It is useful context
for amplitude-only dipole discussions, but it does not by itself identify spatial patterns.

Expected input (not tracked in git):
  data/external/wise_calib/neowise_table2.tbl
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def fit_harmonics(t: np.ndarray, y: np.ndarray, yerr: np.ndarray, periods_days: list[float]) -> dict[str, Any]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    w = 1.0 / np.clip(yerr, 1e-12, np.inf) ** 2

    cols = [np.ones_like(t)]
    names = ["c0"]
    for p in periods_days:
        omega = 2.0 * math.pi / float(p)
        cols.append(np.sin(omega * t))
        cols.append(np.cos(omega * t))
        names.append(f"sin_{p:g}d")
        names.append(f"cos_{p:g}d")

    X = np.column_stack(cols).astype(float)
    # Weighted least squares via normal equations.
    XtW = X.T * w
    beta = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat

    out: dict[str, Any] = {"coef": {n: float(b) for n, b in zip(names, beta, strict=True)}}
    out["rms_resid"] = float(np.sqrt(np.mean(resid**2)))

    amps: dict[str, float] = {}
    for p in periods_days:
        a = float(out["coef"][f"sin_{p:g}d"])
        b = float(out["coef"][f"cos_{p:g}d"])
        amps[f"A_{p:g}d"] = float(math.hypot(a, b))
    out["amps"] = amps
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/external/wise_calib/neowise_table2.tbl")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--periods-days", default="365.25,182.625", help="Comma-separated harmonic periods (days).")
    ap.add_argument("--make-plot", action="store_true")
    args = ap.parse_args()

    table_path = Path(str(args.table))
    if not table_path.exists():
        raise SystemExit(f"Missing table: {table_path}")

    outdir = Path(args.outdir or f"outputs/neowise_table2_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    from astropy.table import Table

    t = Table.read(str(table_path), format="ascii.ipac")
    mjd = np.asarray(t["mjd"], dtype=float)
    w1 = np.asarray(t["w1dmag"], dtype=float)
    w1e = np.asarray(t["w1dmagerr"], dtype=float)
    w2 = np.asarray(t["w2dmag"], dtype=float)
    w2e = np.asarray(t["w2dmagerr"], dtype=float)

    valid = np.isfinite(mjd) & np.isfinite(w1) & np.isfinite(w1e) & np.isfinite(w2) & np.isfinite(w2e)
    mjd = mjd[valid]
    w1 = w1[valid]
    w1e = w1e[valid]
    w2 = w2[valid]
    w2e = w2e[valid]

    periods = [float(x) for x in str(args.periods_days).split(",") if str(x).strip()]
    res_w1 = fit_harmonics(mjd, w1, w1e, periods)
    res_w2 = fit_harmonics(mjd, w2, w2e, periods)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "table": str(table_path),
        "n_rows": int(mjd.size),
        "mjd_min": float(np.min(mjd)) if mjd.size else float("nan"),
        "mjd_max": float(np.max(mjd)) if mjd.size else float("nan"),
        "periods_days": periods,
        "fit_w1": res_w1,
        "fit_w2": res_w2,
    }
    out_meta = outdir / "neowise_table2_fit.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Wrote: {out_meta}")

    if bool(args.make_plot):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Rebuild model for plotting.
        def predict(beta: dict[str, float], tvals: np.ndarray) -> np.ndarray:
            yhat = np.full_like(tvals, float(beta["c0"]), dtype=float)
            for p in periods:
                omega = 2.0 * math.pi / float(p)
                yhat += float(beta[f"sin_{p:g}d"]) * np.sin(omega * tvals)
                yhat += float(beta[f"cos_{p:g}d"]) * np.cos(omega * tvals)
            return yhat

        order = np.argsort(mjd)
        mjd_o = mjd[order]
        w1_o = w1[order]
        w2_o = w2[order]
        w1_hat = predict(res_w1["coef"], mjd_o)
        w2_hat = predict(res_w2["coef"], mjd_o)

        fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.0), dpi=160, sharex=True)
        ax[0].plot(mjd_o, w1_o, lw=0.7, alpha=0.55, color="#4C72B0", label="W1 data")
        ax[0].plot(mjd_o, w1_hat, lw=1.2, color="#C44E52", label="harmonic fit")
        ax[0].set_ylabel("w1dmag [mag]")
        ax[0].grid(alpha=0.25)
        ax[0].legend(loc="best")

        ax[1].plot(mjd_o, w2_o, lw=0.7, alpha=0.55, color="#55A868", label="W2 data")
        ax[1].plot(mjd_o, w2_hat, lw=1.2, color="#C44E52", label="harmonic fit")
        ax[1].set_ylabel("w2dmag [mag]")
        ax[1].set_xlabel("MJD")
        ax[1].grid(alpha=0.25)
        ax[1].legend(loc="best")

        title = "NEOWISE Table 2 photometric drift (global)\n"
        for p in periods:
            title += f"  A_W1({p:g}d)={res_w1['amps'][f'A_{p:g}d']:.4g} mag  "
            title += f"A_W2({p:g}d)={res_w2['amps'][f'A_{p:g}d']:.4g} mag"
            title += "\n"
        fig.suptitle(title.strip())
        fig.tight_layout()
        fig.savefig(outdir / "neowise_table2_fit.png", bbox_inches="tight")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

