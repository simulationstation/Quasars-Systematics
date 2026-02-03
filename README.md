# Quasars-Systematics

This repository contains a reproducible analysis and an ApJL letter draft arguing that the
CatWISE/Secrest quasar number-count dipole is **dominated by survey selection/systematics** tied to
the faint $W1$ magnitude boundary (rather than requiring a large intrinsic/cosmological dipole).

Key artifacts for reviewers are in `Q_D_RES/`:

- `Q_D_RES/Resolution.md` (ApJL letter draft in AASTeX; paste into Overleaf)
- `Q_D_RES/fixed_axis_scaling_fit.png` (main result figure used in the letter)
- `Q_D_RES/dipole_master_tests.md` (detailed run log + additional diagnostics and figures)
- `Q_D_RES/rvmp_fig5_audit.md` (RvMP Fig. 5 / ecliptic-trend + estimator audit; includes injection test)
- `Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.png` (Poisson GLM scan + conservative jackknife)
- `Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json` (scan table + fit diagnostics/templates/jackknife)
- `Q_D_RES/*.json` (small machine-readable summaries used for numbers/plots)

## References / DOIs used by this repository

- Secrest et al. 2022, ApJL 937 L31, DOI: `10.3847/2041-8213/ac88c0`
- Secrest+22 accepted CatWISE AGN catalog (Zenodo record), DOI: `10.5281/zenodo.6784602`
- CatWISE2020 (Marocco et al. 2021, ApJS 253, 8), DOI: `10.3847/1538-4365/abd805`
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

This file is sufficient to reproduce the `Nexp`-offset GLM results shown in `Q_D_RES/`.

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

Latest cached real-data run (included in `Q_D_RES/`):
- `D ≃ 1.6×10^{-2}` across the scan, but the best-fit axis drifts strongly with depth:
  angle-to-CMB ≈ 1.5° at `W1_max=15.5`, ≈ 28.2° at `16.5`, and ≈ 34.3° at `16.6`
  (see `Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json`).

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

### A) Baseline dipole reproduction (sanity check)

```bash
python3 scripts/reproduce_secrest_dipole.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --outdir outputs/secrest_reproduction \
  --b-cut 30 --w1cov-min 80 --w1-max 16.4 --bootstrap 200
```

### B) Figure 1: faint-limit scaling diagnostic (main ApJL figure)

This produces `fixed_axis_scaling_fit.png` and `fixed_axis_scaling_fit.json`.

```bash
python3 experiments/quasar_dipole_hypothesis/fit_intrinsic_plus_selection_fixed_axis.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --axis-from secrest \
  --secrest-json Q_D_RES/secrest_reproduction_dipole.json \
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
  --secrest-json Q_D_RES/secrest_reproduction_dipole.json \
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

## ApJL draft + figures

The ApJL draft is in `Q_D_RES/Resolution.md` and references the PNGs by filename.
If you copy the figures in `Q_D_RES/` into your Overleaf project root, the TeX block in that file
should compile without modification.
