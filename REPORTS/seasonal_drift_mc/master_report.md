# Correlated-cut drift Monte Carlo (seasonal/scan proxy)

Date: 2026-02-04 (UTC)

Purpose: a **robust** version of the “time/season” diagnostic.

- The “time” piece is proxied by **ecliptic longitude**: WISE observing conditions vary with time, and the scan strategy ties many time-dependent effects to fixed structure in ecliptic coordinates (especially ecliptic longitude).
- The “robust” piece is that the common faint-cut scan uses **nested cuts** (`W1 < 15.7` is a subset of `W1 < 15.8`), so adjacent points are *correlated*. We therefore simulate **differential W1 bins** and **cumulatively sum** them to form correlated cut-series, rather than generating independent maps per cut.

This report summarizes two runs:

1) **Null drift test** (no seasonal injection): quantify how often correlated nested cuts produce direction drift like the real scan purely from Poisson counting noise.
2) **Seasonal injection + kinematic dipole**: show that a scan/season-linked longitude pattern can inflate the recovered dipole to `~10^-2` if omitted, and that including `sin(elon)/cos(elon)` restores the injected kinematic-scale dipole.

---

## 1) Null drift test (correlated cuts)

Inputs:
- Real scan used for “observed drift” reference:
  - `REPORTS/amplitude_physical_predictors_suite/bcut30_baseline/rvmp_fig5_poisson_glm.json`
  - Model: Poisson GLM per cut with `dipole + abs_elat`.

Monte Carlo setup:
- Simulate **independent Poisson counts per differential W1 bin**, then cumulatively sum to get correlated `W1 < cut` maps.
- Per-bin baseline mean intensity is fit on the real data using a Poisson GLM with **no dipole** (`intercept + abs_elat`).
- Inject a fixed true dipole (log-intensity dipole) with amplitude set to the real fitted value at `W1_max=16.6`:
  - `D_true = 0.0167785` toward `(l,b)=(236.5769°, 21.8059°)`.

Run outputs:
- `REPORTS/seasonal_drift_mc/data/drift_mc_null_summary.json`
- `REPORTS/seasonal_drift_mc/figures/drift_mc_null_hist.png`

Observed drift metrics from the real scan:
- path length: **50.78°**
- end-to-end: **35.41°**
- max pair separation: **35.83°**

Null p-values (fraction of mocks with metric ≥ observed):
- path length: **p = 0.564**
- end-to-end: **p = 0.00498**
- max pair: **p = 0.00853**

Interpretation:
- The *overall* “wander amount” (path length) is typical under the correlated-cut null.
- But the *span* (end-to-end, and max pair separation) is in the **~0.5–0.9% tail**, i.e. larger than expected from Poisson noise alone under this null.

---

## 2) Seasonal longitude-pattern injection + kinematic dipole (correlated cuts)

Goal: demonstrate that a scan/season-linked longitude pattern can bias the dipole if omitted, **in a correlated-cut simulation**.

Setup:
- Inject a fixed true dipole:
  - `D_true = 0.0046` toward the CMB dipole `(l,b)=(264.021°, 48.253°)`.
- Inject a longitude pattern using per-cut `sin(elon)/cos(elon)` coefficients from the real cumulative fit:
  - `outputs/rvmp_fig5_poisson_glm_eclip_sincos/rvmp_fig5_poisson_glm.json`
- Convert the per-cut coefficients into **per-bin** coefficients so that, in a small-amplitude linearization, the cumulative injected coefficient matches the scan sequence.

Recovery models fit per cut (Poisson GLM):
- **baseline fit**: `dipole + abs_elat` (omits the longitude templates)
- **lon fit**: `dipole + abs_elat + sin(elon) + cos(elon)`

Run outputs:
- `REPORTS/seasonal_drift_mc/data/drift_mc_inject_summary.json`
- `REPORTS/seasonal_drift_mc/figures/drift_mc_inject_bias.png`

Headline at `W1_max=16.6` (mean over mocks):
- baseline fit (omits lon templates): `|mean(b)| ≈ 0.0239` (spurious, wrong direction)
- lon fit (includes lon templates): `|mean(b)| ≈ 0.00480` at direction `(l,b)≈(261.6°, 47.8°)` (close to injected)

Interpretation:
- In a fully correlated-cut setup, a modest scan/season-linked longitude pattern can drive an apparent dipole at the `~10^-2` level if not modeled.
- Adding the longitude templates restores the injected kinematic-scale dipole.

---

## Reproduce

The run launcher used:
- `runs/2026-02-04_seasonal_drift_mc_full/job.sh`

Core command:
```bash
.venv/bin/python scripts/seasonal_drift_mc.py --help
```
