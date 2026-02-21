# Quasars-Systematics

This repository contains a reproducible CatWISE dipole audit and paper-ready bundles supporting a PRD-style
manuscript (CatWISE dipole sensitivity to magnitude cuts, depth templates, and LSS covariance).

Main PRD manuscript sources are intentionally **not** tracked here; the repo focuses on code + reproducible
figures/tables and audit reports. A compact MNRAS Letter bundle is included under `mnras_letter/`.
(Exception: a few short note-style manuscripts live under `radio_a/`.)

## Reviewer seed (headline verification)

This repository includes a lightweight “run-one seed” command that reproduces the headline numbers quoted in
`mnras_letter/main.tex` using **vendored** small artifacts (no large external downloads and no heavy reruns).

```bash
make reproduce
```

This writes a timestamped folder under `outputs/` containing:

- `report.md` (human-readable headline summary)
- `summary.json` (machine-readable values and input provenance)

Notes:

- This is a quick verifier for reviewers; it is intentionally not a substitute for the full end-to-end reruns.
- The vendored inputs used by the seed live under `artifacts/`.

## Scope and limitations (read first)

This repo is designed to support the following (core, referee-facing) claims:
- The CatWISE ``accepted'' number-count dipole **amplitude** is robustly non-zero at the few×10⁻² level.
- Low-ell harmonic nuisance + C_ell priors provide a conservative amplitude robustness check under the survey mask.
- The inferred dipole **axis/direction is not stable** under plausible magnitude-limit and depth/completeness modeling choices.
- Selection/completeness systematics can generate smooth magnitude-limit scans and bias both amplitude and direction in injection tests.

What this repo does **not** currently claim or provide:
- A fully closed, image-level injection/recovery **end-to-end completeness pipeline** that “solves” the full amplitude all-sky.
- A validated, all-sky **W1-conditioned** completeness map (the all-sky external proxy uses Gaia qsocand, which has no W1).

## PRD paper figure pack

The PRD manuscript expects the following PNGs in your Overleaf project root:

- `rvmp_fig5_repro_baseline.png`
- `rvmp_fig5_poisson_glm_ecliponly.png`
- `rvmp_fig5_repro_inject_dm0125cmb.png`
- `glm_cv_axes_nexp_offset.png`
- `lss_cov_D_hist_w1max16p6.png`
- `validate_depth_systematic_recovery.png`
- `systematics_grid_full_w1max16p4.png`
- `systematics_grid_no_nvss_w1max16p5.png`
- `cmb_projection_plot.png`

They are tracked in this repo at:
- `REPORTS/Q_D_RES_2_2/figures/` (first 6 files above)
- `REPORTS/2-3-EEE/figures/` (the `systematics_grid_*.png` pair)
- `REPORTS/2-3-DDD/artifacts_main/` (`cmb_projection_plot.png`)

Convenience staging command (creates a folder you can upload to Overleaf):

```bash
mkdir -p outputs/overleaf_prd_figs
cp REPORTS/Q_D_RES_2_2/figures/rvmp_fig5_repro_baseline.png outputs/overleaf_prd_figs/
cp REPORTS/Q_D_RES_2_2/figures/rvmp_fig5_poisson_glm_ecliponly.png outputs/overleaf_prd_figs/
cp REPORTS/Q_D_RES_2_2/figures/rvmp_fig5_repro_inject_dm0125cmb.png outputs/overleaf_prd_figs/
cp REPORTS/Q_D_RES_2_2/figures/glm_cv_axes_nexp_offset.png outputs/overleaf_prd_figs/
cp REPORTS/Q_D_RES_2_2/figures/lss_cov_D_hist_w1max16p6.png outputs/overleaf_prd_figs/
cp REPORTS/Q_D_RES_2_2/figures/validate_depth_systematic_recovery.png outputs/overleaf_prd_figs/
cp REPORTS/2-3-EEE/figures/systematics_grid_full_w1max16p4.png outputs/overleaf_prd_figs/
cp REPORTS/2-3-EEE/figures/systematics_grid_no_nvss_w1max16p5.png outputs/overleaf_prd_figs/
cp REPORTS/2-3-DDD/artifacts_main/cmb_projection_plot.png outputs/overleaf_prd_figs/
```

## Key results bundles

### PRD audit bundle (main)

Key artifacts for the PRD audit are in `REPORTS/Q_D_RES_2_2/`:

- `REPORTS/Q_D_RES_2_2/master_report.md` (paper update bundle; figures + data + key numbers)
- `REPORTS/Q_D_RES_2_2/figures/` (exact PRD PNG filenames)

Additional PRD appendices / validation bundles:

- `REPORTS/2-3-EEE/master_report.md` (Secrest-accepted validation suite: baseline reproduction + residual systematics χ²/ν)
- `REPORTS/2-3-DDD/master_report.md` (CMB-parallel/perpendicular decomposition of the GLM scan dipole vectors)
- `completeness_validation.md` (end-to-end completeness validation checklist/plan)
- `REPORTS/external_completeness_sdss_dr16q/master_report.md` (externally validated, map-level completeness model using SDSS DR16Q + unWISE logNexp)
- `REPORTS/external_validation_gaia_qsocand/master_report.md` (all-sky external validation using Gaia DR3 QSO candidates)
- `REPORTS/external_completeness_gaia_qsocand_externalonly/master_report.md` (all-sky externally trained spatial completeness proxy from Gaia qsocand + map predictors)
- `REPORTS/end_to_end_completeness_correction/master_report.md` (end-to-end correction attempt: impact of Gaia external completeness template on CMB-perp drift)
  - figures: `REPORTS/end_to_end_completeness_correction/figures/cmb_projection_compare_baseline_vs_gaia_extonly.png`,
    `REPORTS/end_to_end_completeness_correction/figures/gaia_extonly_cov_rvmp_fig5_poisson_glm.png`
  - table: `REPORTS/end_to_end_completeness_correction/data/cmb_projection_compare_baseline_vs_gaia_extonly.csv`
- `REPORTS/dipole_direction_report/master_report.md` (fast “seasonal imprint” proxy via ecliptic-longitude wedges + `sinλ/cosλ`)
- `REPORTS/seasonal_selection_injection_check/master_report.md` (injection test: how ecliptic-longitude selection can bias dipole amplitude)
- `REPORTS/seasonal_drift_mc/master_report.md` (correlated-cut drift Monte Carlo + correlated seasonal injection)
- `REPORTS/unwise_time_domain_epoch_amplitude/master_report.md` (true epoch-resolved amplitude stability test using the unWISE time-domain catalog)
- `REPORTS/unwise_time_domain_catwise_epoch_amplitude/master_report.md` (epoch-resolved amplitude stability test using the unWISE time-domain catalog, restricted to the published CatWISE accepted parent sample)
  - Figure sync: `mnras_letter/D_vs_epoch_compare.png` is generated for epochs `0–15` only (epoch 16 excluded as partial), with fixed y-axis ticks so the vector-sum max (~0.155) is auditable by eye.
- `outputs/epoch_dipole_time_domain_20260204_222039UTC/finiteN_null_big_fullsample_v1.json` (highest-stat finite-`N` null on the large epoch sample, with ~3.9e7--5.4e7 selected objects per epoch)
  - headline: observed epoch variability is far beyond finite-count expectations; `10,000,000` Gaussian-null draws and `20,000` map-level multinomial+GLM null draws both produced `0` exceedances of the observed spread/chi2 metrics.
- `REPORTS/seasonal_update/update.md` (paper-ready writeup tying the ecliptic-longitude proxy to Secrest-style residual checks)
- `REPORTS/arxiv_amplitude_multipole_prior_injection/master_report.md` (harmonic-prior dipole injection check under low-ell contamination)
- `REPORTS/arxiv_amplitude_multipole_mode_coupling_prior/master_report.md` (harmonic-prior amplitude sweep vs prior scale)
- `REPORTS/amplitude_physical_predictors_suite/master_report.md` (physical scan/depth predictor control sweep)
- `REPORTS/ecllon_proxy_with_depth/master_report.md` (ecliptic-longitude proxy + physical depth/coverage covariates)
- `REPORTS/selection_sim_depthmap/master_report.md` (end-to-end selection simulation driven by a real depth/completeness map)
- `REPORTS/selection_sim_depthmap_plus_lon/master_report.md` (selection simulation: depth map + ecliptic-longitude pattern)

Other folders under `REPORTS/` are legacy/exploratory and are not required for the CatWISE dipole audit.

### Radio NB dipole identifiability note

This repo also contains a short audit note on the negative-binomial counts-in-cells radio dipole estimator
advocated in arXiv:2509.16732:

- Manuscript + figures: `radio_a/`
- Reproduction + stress tests: `scripts/run_radio_nb_dipole_audit.py`

Headline (joint LoTSS+RACS-low+NVSS, $N_{\rm side}=32$, footprint-defined cells including zeros):
adding physically motivated survey templates yields large likelihood gains and large direction shifts relative
to the dipole-only fit; in this configuration the extended model is preferred even under BIC, indicating the
dipole-only interpretation is not robustly identifiable without explicit systematics modeling and
injection/recovery validation on the relevant footprints.

## References / DOIs used by this repository

- Quasar-dipole reproducibility archive (Zenodo), DOI: `10.5281/zenodo.18643926`
- Supplementary assets archive (Zenodo), DOI: `10.5281/zenodo.18489200`
- Legacy repository archive (Zenodo), DOI: `10.5281/zenodo.18476711`
- Secrest et al. 2022, ApJL 937 L31, DOI: `10.3847/2041-8213/ac88c0`
- Secrest+22 accepted CatWISE AGN catalog (Zenodo record), DOI: `10.5281/zenodo.6784602`
- CatWISE2020 (Marocco et al. 2021, ApJS 253, 8), DOI: `10.3847/1538-4365/abd805`
- unWISE Time-Domain Catalog (IRSA580), DOI: `10.26131/IRSA580`
- unWISE coadds:
  - Lang 2014, AJ 147, 108, DOI: `10.1088/0004-6256/147/5/108`
  - Meisner et al. 2017, AJ 153, 38, DOI: `10.3847/1538-3881/153/1/38`
- WISE mission (optional context):
  - Wright et al. 2010, AJ 140, 1868, DOI: `10.1088/0004-6256/140/6/1868`

## Environment setup

Recommended (Conda, easiest for `healpy`):

```bash
conda env create -f environment.yml
conda activate quasars-systematics
```

Alternative (pip/venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data requirements

### 1) Secrest+22 CatWISE AGN catalog (required)

Download the Secrest+22 accepted bundle from Zenodo:

- DOI: `10.5281/zenodo.6784602`

Place and extract it under `data/external/zenodo_6784602/` so the expected catalog path exists:

```text
data/external/zenodo_6784602/
  secrest+22_accepted.tgz
  secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits
  secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits
  secrest_extracted/secrest+22_accepted/wise/reference/nvss_crossmatch.fits   (if present in the release)
```

Notes:
- The tarball is ~2.3 GiB.
- `exclude_master_revised.fits` is used for the official Secrest exclusion mask in some workflows.

### 2) unWISE depth proxy (optional; "independent depth" robustness)

For the GLM+CV robustness test we use an *independent*, imaging-derived depth statistic based on the
unWISE W1 exposure-count maps (`w1-n-m`).

To keep this repository lightweight, we **include a derived per-tile statistic**:

- `data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json`
- `data/external/unwise/tiles.fits` (used to map sky positions to unWISE tile IDs)

This file is sufficient to reproduce the `Nexp`-offset GLM results shown in `REPORTS/Q_D_RES/`.

#### Map-level depth (recommended for identifiability)

For template/dipole-degeneracy studies, it is often cleaner to work with a **map-level** depth proxy
that does not depend on the catalog’s realized source positions.

This repo now includes a small precomputed HEALPix map (Galactic coords):

- `data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits`

You can regenerate it from the included tile table + per-tile stats:

```bash
python3 scripts/build_unwise_nexp_healpix_map.py --nside 64 --value-mode log_nexp
```

If you want to regenerate this JSON from the full unWISE coadds, use:

```bash
python3 scripts/build_unwise_nexp_tile_stats.py --help
```

Warning: regenerating from raw `w1-n-m` maps requires downloading a very large volume of FITS files
(multi-TB scale depending on the tile set and caching).

### 3) Gaia DR3 QSO candidates (optional; all-sky external validation)

For the all-sky external validation report in `REPORTS/external_validation_gaia_qsocand/`, download:
- CDS catalog: `I/356` (Gaia DR3 Part 2: Extra-galactic)
- file: `qsocand.dat.gz`

Place it at:
- `data/external/gaia_dr3_extragal/qsocand.dat.gz`

Example (resumable):
```bash
mkdir -p data/external/gaia_dr3_extragal
wget -c -O data/external/gaia_dr3_extragal/qsocand.dat.gz.part https://cdsarc.cds.unistra.fr/ftp/I/356/qsocand.dat.gz
mv data/external/gaia_dr3_extragal/qsocand.dat.gz.part data/external/gaia_dr3_extragal/qsocand.dat.gz
```

### 4) SDSS DR16Q (optional; footprint-limited external completeness model)

For the SDSS DR16Q external completeness model in `REPORTS/external_completeness_sdss_dr16q/`, download:
- `DR16Q_v4.fits`

Place it at:
- `data/external/sdss_dr16q/DR16Q_v4.fits`

This file is **not** tracked in git (it is multi-GB scale).

## Reproducing the headline results

All commands below assume you have the Secrest+22 catalog extracted at:

`data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits`

## New: RvMP Fig. 5 audit (Secrest-style vs Poisson-likelihood + injection)

This repository now includes a “straightforward path” verification suite motivated by:
- Secrest’s critique that (i) naive linear dipole estimators can be biased on a masked sky and
  (ii) the strong ecliptic-latitude trend must be modeled, and
- the 2025 RvMP review (2025RvMP...97d1001S) summarizing CatWISE dipole stability vs `W1_max`.

These scripts:
- reproduce the Secrest+22 masking + ecliptic correction logic in a self-contained way,
- implement a Poisson maximum-likelihood analogue with the same footprint mask, and
- include an injection test showing that “stability vs faint cut” does **not** by itself rule out
  a selection-gradient mechanism.

### A) Secrest-style Fig.5 scan (weighted linear dipole fit)

```bash
python3 scripts/reproduce_rvmp_fig5_catwise.py \
  --make-plot \
  --w1-grid 15.5,16.6,0.05 \
  --nsim 400 \
  --outdir outputs/rvmp_fig5_secrest_style
```

Outputs:
- `outputs/rvmp_fig5_secrest_style/rvmp_fig5_repro.json`
- `outputs/rvmp_fig5_secrest_style/rvmp_fig5_repro.png`

### B) Poisson-likelihood Fig.5 scan (template-marginalized GLM)

```bash
python3 scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
  --eclip-template abs_elat \
  --dust-template none \
  --depth-mode none \
  --w1-mode cumulative \
  --jackknife-nside 2 --jackknife-stride 1 --jackknife-max-regions 48 \
  --make-plot \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_poisson_glm
```

The output JSON now includes:
- `fit_diag`: deviance/DOF, Pearson χ²/DOF (overdispersion diagnostics),
- `dipole_quasi`: a conservative quasi-Poisson uncertainty variant (covariance scaled by dispersion),
- `corr_b_templates` / `template_dipoles`: explicit dipole–template degeneracy summaries,
- `jackknife` (if enabled): leave-one-region-out sky jackknife of the fitted dipole vector.

Paper-ready cached real-data runs:
- PRD figure version: `REPORTS/Q_D_RES_2_2/figures/rvmp_fig5_poisson_glm_ecliponly.png`
  (scan JSON: `REPORTS/Q_D_RES_2_2/data/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json`)
- Conservative jackknife-annotated version: `REPORTS/Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.png`
  (scan JSON: `REPORTS/Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json`)

Key numbers at `W1_max=16.6` in the ecliptic-template-only scan:
- `D ≃ 1.6×10^{-2}` across the scan, but the best-fit axis drifts strongly with depth:
  angle-to-CMB ≈ 1.5° at `W1_max=15.5`, ≈ 28.2° at `16.5`, and ≈ 34.3° at `16.6`.

#### Low-ell harmonic nuisance + C_ell prior (amplitude robustness)

Enable real spherical-harmonic nuisance templates for `ell>=2` via `--harmonic-lmax`.
Optionally regularize those coefficients with a Gaussian prior using `C_ell` estimated from the clustered (lognormal) mocks.

```bash
python3 scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
  --eclip-template abs_elat \
  --dust-template none \
  --depth-mode none \
  --harmonic-lmax 5 \
  --harmonic-prior lognormal_cl \
  --harmonic-prior-cl-json REPORTS/Q_D_RES_2_2/data/lognormal_cov_w1max16p6_n500/lognormal_mocks_cov.json \
  --harmonic-prior-scale 1 \
  --w1-mode cumulative \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_poisson_glm_harmprior_lmax5
```

#### Dipole injection check for the harmonic-prior fit

This reproduces the injection check bundled at `REPORTS/arxiv_amplitude_multipole_prior_injection/`.

```bash
python3 scripts/harmonic_prior_injection_check.py \
  --w1-max 16.6 \
  --harmonic-lmax 5 \
  --n-mocks 200 \
  --true-cl-scale 10 \
  --fit-prior-scale 1 \
  --seed 123
```

Outputs:
- `REPORTS/arxiv_amplitude_multipole_prior_injection/data/lowell_injection_validation.json`
- `REPORTS/arxiv_amplitude_multipole_prior_injection/figures/lowell_injection_validation.png`
- `REPORTS/arxiv_amplitude_multipole_prior_injection/master_report.md`

#### Seasonal selection/completeness injection check (ecliptic longitude)

This runs controlled Poisson-map injections of `sinλ/cosλ` patterns to quantify how a scan-linked selection term can bias dipole amplitude when omitted.

```bash
python3 scripts/seasonal_selection_injection_check.py \
  --w1-max 16.6 \
  --n-mocks 200 \
  --lon-amps 0,0.01,0.02,0.03,0.04 \
  --lon-phase-deg 0 \
  --dipole-amp 0 \
  --outdir REPORTS/seasonal_selection_injection_check/dipole0
```

See: `REPORTS/seasonal_selection_injection_check/master_report.md`

#### Differential-bin diagnostic (recommended add-on)

```bash
python3 scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
  --eclip-template abs_elat \
  --dust-template none \
  --depth-mode none \
  --w1-mode differential \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_poisson_glm_differential
```

#### Independent depth variant (recommended): unWISE Nexp as a depth covariate

This uses the included `data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json` (a per-tile statistic)
and maps each HEALPix pixel to the nearest unWISE tile center to build a depth proxy.

```bash
python3 scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
  --eclip-template abs_elat \
  --dust-template none \
  --depth-mode unwise_nexp_covariate \
  --nexp-tile-stats-json data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json \
  --make-plot \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_poisson_glm_unwise_nexp
```

Outputs:
- `outputs/rvmp_fig5_poisson_glm/rvmp_fig5_poisson_glm.json`
- `outputs/rvmp_fig5_poisson_glm/rvmp_fig5_poisson_glm.png`

#### Map-level depth variant (HEALPix depth map; not catalog-proxy)

```bash
python3 scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
  --eclip-template abs_elat \
  --dust-template none \
  --depth-mode depth_map_covariate \
  --depth-map-fits data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits \
  --depth-map-name unwise_lognexp_nside64 \
  --make-plot \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_poisson_glm_depthmap
```

## Seasonal (ecliptic-longitude) proxy (optional)

The PRD paper’s main story is about ecliptic-latitude scan structure (`|β|`) and depth templates.
As an additional **fast proxy** for time/season systematics projecting into ecliptic coordinates, we also run
an ecliptic-longitude diagnostic at `W1_max=16.6`:

- Poisson-GLM fit on the full sky with/without `sinλ/cosλ` nuisance templates, and
- longitude-wedge fits (directional sensitivity across ecliptic longitude).

Report bundle:
- `REPORTS/dipole_direction_report/master_report.md`

Re-run:

```bash
.venv/bin/python scripts/run_ecliptic_lon_proxy.py \
  --w1-max 16.6 \
  --lambda-edges 0,90,180,270,360 \
  --make-plot \
  --outdir outputs/ecllon_proxy_run
```

### Quick smoke test (no external data)

```bash
python3 scripts/smoke_rvmp_fig5_poisson_glm.py
```

## LSS covariance: clustered (lognormal) mocks

Poisson-only errors are optimistic because they neglect large-scale structure (sample variance).
This script generates quick clustered mocks and returns a dipole covariance that includes LSS+shot noise:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 scripts/run_catwise_lognormal_mocks.py \
  --w1-max 16.6 \
  --n-mocks 500 \
  --n-proc 0 \
  --mp-start spawn \
  --eclip-template abs_elat \
  --dust-template none \
  --depth-mode none \
  --outdir outputs/lognormal_mocks_cov
```

The covariance is written to `outputs/lognormal_mocks_cov/lognormal_mocks_cov.json` and includes:
- an estimated clustering `C_ell` from data residuals (with a crude f_sky correction),
- the recovered dipole-vector covariance across mocks (`cov_b`),
- optional injection recovery diagnostics (`--inject-dipole-amp ...`).

### End-to-end completeness validation (recommended)

Generate mocks with both (i) an injected dipole and (ii) a depth-tied selection systematic, then compare fits
with and without a depth template:

```bash
python3 scripts/run_catwise_lognormal_mocks.py \
  --w1-max 16.6 \
  --n-mocks 500 \
  --n-proc 0 \
  --inject-dipole-amp 0.005 \
  --inject-axis cmb \
  --mock-depth-alpha 0.04 \
  --mock-depth-map-fits data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits \
  --mock-zero-base-dipole \
  --depth-mode none \
  --outdir outputs/lognormal_mocks_inj_depth_misspecified

python3 scripts/run_catwise_lognormal_mocks.py \
  --w1-max 16.6 \
  --n-mocks 500 \
  --n-proc 0 \
  --inject-dipole-amp 0.005 \
  --inject-axis cmb \
  --mock-depth-alpha 0.04 \
  --mock-depth-map-fits data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits \
  --mock-zero-base-dipole \
  --depth-mode none \
  --fit-depth-mode depth_map_covariate \
  --depth-map-fits data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits \
  --outdir outputs/lognormal_mocks_inj_depth_modeled
```

### C) Mechanism-killing injection test (dipolar faint-limit modulation)

This injects a purely selection-driven dipolar modulation of the effective faint limit:
`W1_eff = W1 - delta_m * cos(theta_axis)` and re-runs the scan.

```bash
python3 scripts/reproduce_rvmp_fig5_catwise.py \
  --inject-delta-m-mag 0.0125 \
  --inject-axis cmb \
  --make-plot \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_secrest_style_injected
```

### D) NVSS removal + homogenization (Secrest 2_rm_nvss analogue; optional)

This step uses additional files from the Zenodo bundle:
- `.../nvss/reference/NVSS.fit`
- `.../nvss/reference/nvss_artifacts.fits`
- `.../nvss/reference/NVSS_CatWISE2020_40arcsec_best_symmetric.fits`
- `.../reference/haslam408_dsds_Remazeilles2014_512.fits`

Build the homogenized “no-NVSS” CatWISE sample:

```bash
python3 scripts/build_catwise_no_nvss_homogenized.py \
  --outdir outputs/catwise_no_nvss_homog \
  --w1cut-max 16.5 \
  --nvss-scut-mjy 10 --nvss-tbcut-K 50
```

Then re-run either scan using:
- `--catalog outputs/catwise_no_nvss_homog/catwise_no_nvss_homog.fits`
- `--mask-catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits`

Important: `--mask-catalog` avoids spuriously masking pixels as “zero coverage” when the analysis
catalog is a filtered/subsampled file.

### A) Baseline dipole reproduction (Secrest-style; recommended)

This uses the **Secrest footprint + ecliptic correction + fit\_dipole linear solve** (matching the released
Secrest pipeline logic more closely than a simple vector-sum estimator).

```bash
python3 scripts/reproduce_rvmp_fig5_catwise.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --exclude-mask-fits data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits \
  --w1cov-min 80 --b-cut 30 \
  --w1-grid 16.4,16.4,0.1 \
  --nsim 400 \
  --outdir outputs/secrest_style_single_w1max16p4
```

Optional: Secrest-style residual systematics audit (what a referee will expect):

```bash
python3 scripts/run_secrest_systematics_audit.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --exclude-mask-fits data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits \
  --w1-cut 16.4 \
  --xyfit-binsize 200 \
  --outdir outputs/secrest_systematics_full_w1max16p4 \
  --make-plots
```

Paper-ready archived outputs from this audit live in `REPORTS/2-3-EEE/`.

### B) Faint-limit scaling diagnostic (optional)

This produces `fixed_axis_scaling_fit.png` and `fixed_axis_scaling_fit.json`.

```bash
python3 experiments/quasar_dipole_hypothesis/fit_intrinsic_plus_selection_fixed_axis.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --axis-from secrest \
  --secrest-json REPORTS/Q_D_RES/secrest_reproduction_dipole.json \
  --w1cov-min 80 --b-cut 30 \
  --w1max-grid 15.6,16.6,0.05 \
  --alpha-dm 0.05 \
  --make-plots \
  --outdir outputs/fixed_axis_scaling_fit
```

### C) Additional mechanism diagnostics (optional)

This produces `dipole_vs_w1max.png`, `dipole_vs_w1covmin.png`, and hemisphere count-match plots.

```bash
python3 experiments/quasar_dipole_hypothesis/magshift_mechanism_diagnostics.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --secrest-json REPORTS/Q_D_RES/secrest_reproduction_dipole.json \
  --w1-max 16.4 --w1-max-for-cdf 17.0 \
  --make-plots \
  --outdir outputs/magshift_mechanisms
```

### D) GLM+CV robustness: template-cleaned residual dipole with independent depth (optional)

This is the main robustness point: the inferred "cleaned" direction depends strongly on which
depth proxy is used. To run with the included unWISE per-tile depth stats:

```bash
python3 experiments/quasar_dipole_hypothesis/vector_convergence_glm_cv.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --exclude-mask-fits data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits \
  --nvss-crossmatch data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/nvss_crossmatch.fits \
  --template-set ecliptic_harmonics \
  --nside 64 --kfold 5 --seed 123 \
  --nexp-tile-stats-json data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json \
  --outdir outputs/glmcv_nexp_offset \
  --make-plots
```
