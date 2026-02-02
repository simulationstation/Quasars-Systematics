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
  --make-plot \
  --w1-grid 15.5,16.6,0.05 \
  --outdir outputs/rvmp_fig5_poisson_glm
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

