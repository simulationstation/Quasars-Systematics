# RvMP Fig. 5 Audit: CatWISE Dipole vs Faint Cut (Secrest-style + Poisson GLM)

This note records a “straightforward path” audit motivated by:
- Secrest’s email critique that (i) linear dipole estimators can be biased and (ii) the ecliptic-latitude trend must be modeled, and
- the 2025 RvMP review (2025RvMP...97d1001S) stating that the CatWISE dipole remains stable vs `W1_max` after marginalizing the ecliptic trend.

The goal is **not** to claim new physics; it is to check whether the CatWISE signal is consistent with a survey/selection mechanism once the estimator and scan-depth nuisance structure are treated carefully.

## What we implemented (in this repo)

Scripts:
- `scripts/reproduce_rvmp_fig5_catwise.py`  
  Self-contained reproduction of the **Secrest+22** masking + ecliptic correction pipeline (including the exact linear dipole solve used in `hpmap_utilities.SkyMap.fit_dipole`), scanned vs faint cut `W1_max`.
- `scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py`  
  A **Poisson maximum-likelihood** analogue on HEALPix counts with the same footprint mask and nuisance templates in the log intensity (ecliptic latitude, optional dust and depth proxies).
- `scripts/build_catwise_no_nvss_homogenized.py`  
  Optional NVSS-removal + homogenization step following Secrest+22 `wise/code/2_rm_nvss.py`, producing a “no-NVSS” CatWISE sample for re-running the scans.

## Summary of findings (qualitative)

Across both the Secrest-style estimator and the Poisson-likelihood estimator:
- The **dipole amplitude** stays at the expected level `D ~ O(10^-2)` across the scanned `W1_max` range.
- The **dipole direction drifts substantially** as `W1_max` is pushed fainter (even when modeling the ecliptic latitude trend).

This makes it difficult to interpret the full-range faint-cut scan as robust evidence for a single stable cosmological/kinematic dipole direction without additional, independent depth/completeness modeling.

## 2026-02 update: conservative errors + explicit degeneracy diagnostics

The Poisson-GLM scan script now records:
- **Overdispersion diagnostics** (`fit_diag`: deviance/DOF, Pearson χ²/DOF), plus a conservative **quasi-Poisson** uncertainty variant (`dipole_quasi`) that inflates the Fisher covariance by the estimated dispersion.
- A sky **jackknife** option that reports dipole-amplitude uncertainty and axis scatter across disjoint sky regions (`jackknife`).
- **Template–dipole degeneracy** summaries via the fitted covariance (`corr_b_templates`) and the intrinsic dipole vector of each nuisance template field (`template_dipoles`).
- A **differential-bin** mode (`--w1-mode differential`) to avoid the “cumulative cuts make smoothness inevitable” failure mode.

Included small artifacts from a full real-data run (all < 0.2 MB each):
- `Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.png` (main Poisson scan figure)
- `Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json` (scan table + diagnostics)
- `Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_differential.json` (differential-bin diagnostic)
- `Q_D_RES/rvmp_fig5_poisson_glm_w1cov_covariate.json` and `Q_D_RES/rvmp_fig5_poisson_glm_w1cov_offset.json` (depth-proxy modeling sensitivity)

Numerically (ecliptic-template-only Poisson GLM; cumulative mode), the recovered axis remains close to the CMB dipole at the bright end but drifts strongly with depth:
- angle-to-CMB ≈ 1.5° at `W1_max=15.5`, ≈ 28.2° at `16.5`, and ≈ 34.3° at `16.6` (see the JSON above).
The sky jackknife scatter at fixed cut is small (∼degree-level) compared to the ∼tens-of-degrees drift across cuts, supporting the interpretation that the drift is not just noise.

## Mechanism test: injected faint-limit modulation

Both scan scripts include an injection mode that applies a purely selection-driven faint-limit modulation:

`W1_eff = W1 - delta_m * cos(theta_axis)` and then selects `W1_eff < W1_max`.

This is a direct test of the statement “stability vs faint cut rules out selection systematics”.

In practice, the injection can produce an apparently “stable vs cut” dipole while being entirely selection-induced, so “stability vs cut” is **not** a sufficient disproof of selection gradients.

## How to reproduce (commands)

### Secrest-style scan

```bash
python3 scripts/reproduce_rvmp_fig5_catwise.py \
  --make-plot \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_secrest_style
```

### Poisson-likelihood scan

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

### Differential-bin diagnostic (recommended add-on)

```bash
python3 scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
  --eclip-template abs_elat \
  --dust-template none \
  --depth-mode none \
  --w1-mode differential \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_poisson_glm_differential
```

### Independent depth test (unWISE Nexp)

Use an imaging-derived depth proxy based on unWISE exposure maps. This repo includes a per-tile
statistic file (`data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json`). The script maps each
HEALPix pixel to the nearest unWISE tile center and uses `log(Nexp)` as a **covariate** (recommended)
or as a fixed-coefficient offset (more aggressive).

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

### Injection (selection gradient) scan

```bash
python3 scripts/reproduce_rvmp_fig5_catwise.py \
  --inject-delta-m-mag 0.0125 \
  --inject-axis cmb \
  --make-plot \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_secrest_style_injected
```

### Optional: NVSS removal + homogenization (Secrest analogue)

```bash
python3 scripts/build_catwise_no_nvss_homogenized.py \
  --outdir outputs/catwise_no_nvss_homog \
  --w1cut-max 16.5 \
  --nvss-scut-mjy 10 --nvss-tbcut-K 50
```

Then rerun either scan using:
- `--catalog outputs/catwise_no_nvss_homog/catwise_no_nvss_homog.fits`
- `--mask-catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits`

(`--mask-catalog` ensures the “zero coverage” mask is built from the full catalog, not the filtered one.)
