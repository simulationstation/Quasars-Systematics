# 2-3-EEE: Secrest-accepted validation checks (CatWISE quasar dipole)

## Goal
Run the **same kinds of checks Secrest et al. rely on** (and that a referee would consider “standard”) and confirm that:
1) our implementation reproduces the published baseline dipole at the fiducial cut, and
2) our masking + ecliptic-trend correction behave sensibly under Secrest-style residual systematics tests.

This bundle is meant to support statements like “we match the published Secrest pipeline” before introducing
our additional robustness results (GLM, template sensitivity, LSS covariance, injections, CMB-projection decomposition).

## What “Secrest-accepted tools” means (concretely)
From the released Secrest+22 Zenodo bundle (record `6784602`), the WISE-side pipeline relies on:
- **Footprint masking:** `mask_zeros` (no-coverage pixels + neighbours) on the `W1cov>=80` parent sample,
  + exclusion discs (`exclude_master_revised.fits`),
  + Galactic plane cut `|b|>30°` (pixel-center).
- **Ecliptic-latitude trend correction:** fit a **linear** density trend versus `|β|` (ecliptic latitude) using binned regression,
  then **force intercept=0** and subtract the fitted slope term.
- **Dipole estimator:** least-squares dipole fit on the masked sky via the `fit_dipole` linear solve (healpy/Secrest style),
  applied to the **count map** after applying the Secrest ecliptic weight
  `w = 1 - p0 * (|β| / count)` (with `p0` derived from the density-vs-|β| fit).
- **Residual systematics checks:** subtract the best-fit dipole from the corrected density map and verify that binned residuals
  show no strong trends versus common systematics proxies (EBV, synchrotron, declination, ecliptic latitude, etc.), quantified
  via reduced `χ²/ν`.

In this repo these checks are implemented in:
- `scripts/reproduce_rvmp_fig5_catwise.py` (Secrest-style dipole scan; includes the ecliptic-weight logic)
- `scripts/run_secrest_systematics_audit.py` (new; reproduces Secrest-style residual `χ²/ν` diagnostics)

## Inputs
- WISE/CatWISE accepted sample:
  - `data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits`
  - `data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits`
- Optional (for the no-NVSS/homogenized variant):
  - built locally by `scripts/build_catwise_no_nvss_homogenized.py` (not committed; too large)

## Outputs in this folder

### Dipole reproductions (single-cut, Secrest-style)
- `REPORTS/2-3-EEE/data/secrest_style_single_w1max16p4_rvmp_fig5_repro.json`
  - Baseline (“WISE-only”) at `W1_max=16.4`.
- `REPORTS/2-3-EEE/data/secrest_style_no_nvss_single_w1max16p5_rvmp_fig5_repro.json`
  - Secrest+22-style independence control: NVSS-matched sources removed + homogenization, at `W1_max=16.5`.

### Residual systematics audits (Secrest-style `χ²/ν`)
- `REPORTS/2-3-EEE/data/systematics_full_w1max16p4.json`
- `REPORTS/2-3-EEE/figures/systematics_grid_full_w1max16p4.png`
- `REPORTS/2-3-EEE/data/systematics_no_nvss_w1max16p5.json`
- `REPORTS/2-3-EEE/figures/systematics_grid_no_nvss_w1max16p5.png`

## Headline results (copy-ready)

### A) Baseline, Secrest-style estimator (WISE-only; `W1_max=16.4`)
From `REPORTS/2-3-EEE/data/systematics_full_w1max16p4.json`:
- `D ≈ 0.01610`, `(l,b) ≈ (238.8°, +28.3°)`

This is close to the published Secrest quasar dipole direction at the fiducial cut (the exact amplitude depends on implementation details like
the precise trend-correction fit, mask revision, and whether NVSS-removal is applied).

### B) NVSS-removed + homogenized (Secrest+22 independence control; `W1_max=16.5`)
From `REPORTS/2-3-EEE/data/systematics_no_nvss_w1max16p5.json`:
- `D ≈ 0.01531`, `(l,b) ≈ (239.5°, +30.1°)`

This is consistent with the Secrest+22 WISE dipole numbers reported for the NVSS-independent WISE sample (order `~1.5×10^-2` and similar direction).

### C) Residual systematics tests (`χ²/ν`)
For both the baseline and no-NVSS cases, the Secrest-style binned residual tests against common systematics proxies return `χ²/ν` values
generally of order unity (see the grid plots), indicating that **the pipeline behaves as expected under the same diagnostics used in the Secrest workflow**.

## Why this matters for the paper
These checks establish that:
- we are not using an exotic estimator; we match the Secrest fit-dipole + ecliptic correction logic,
- our “drift/degeneracy” claims are therefore not artifacts of a nonstandard pipeline,
- we can responsibly present *additional* robustness results (GLM template dependence, LSS covariance, injections, `D_∥/D_⊥`) as extensions on top of a validated baseline.

## Repro commands

### 1) Baseline, Secrest-style single-cut reproduction (`W1_max=16.4`)
```bash
./.venv/bin/python scripts/reproduce_rvmp_fig5_catwise.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --exclude-mask-fits data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits \
  --w1cov-min 80 --b-cut 30 \
  --w1-grid 16.4,16.4,0.1 \
  --nsim 400 \
  --outdir outputs/secrest_style_single_w1max16p4
```

### 2) Build NVSS-removed + homogenized CatWISE sample (optional)
```bash
./.venv/bin/python scripts/build_catwise_no_nvss_homogenized.py \
  --outdir outputs/catwise_no_nvss_homog \
  --w1cut-max 16.5 \
  --nvss-scut-mjy 10 --nvss-tbcut-K 50
```

### 3) Secrest-style residual systematics audit
```bash
./.venv/bin/python scripts/run_secrest_systematics_audit.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --exclude-mask-fits data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits \
  --w1-cut 16.4 \
  --xyfit-binsize 200 \
  --outdir outputs/secrest_systematics_full_w1max16p4 \
  --make-plots
```

No-NVSS variant:
```bash
./.venv/bin/python scripts/run_secrest_systematics_audit.py \
  --catalog outputs/catwise_no_nvss_homog/catwise_no_nvss_homog.fits \
  --mask-catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --exclude-mask-fits data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits \
  --w1-cut 16.5 \
  --xyfit-binsize 200 \
  --outdir outputs/secrest_systematics_no_nvss_w1max16p5 \
  --make-plots
```
