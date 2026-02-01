# Q_D_RES (CatWISE/Secrest Quasar Dipole: Results Snapshot)

This folder is a curated snapshot of our current evidence about the **Secrest et al. quasar dipole** (CatWISE AGN sample).

## Bottom line (my honest take)

- **Yes:** the results we have so far strongly support the conclusion that the **observed CatWISE dipole is dominated by selection / scan-depth / calibration systematics**, not a cosmological overdensity/bulk-flow dipole.
- **But:** I would not write “mathematical proof” yet. To make this fully peer-review-hard, we still need a clean *end-to-end* model showing that after controlling for depth/scan/dust the residual dipole is consistent with noise *and* that this conclusion is stable to reasonable alternative depth/coverage templates (and/or a mixture model for “missingness” as Gemini suggested).

The most important robustness point we now have is that swapping a catalog-derived coverage proxy (`w1cov`) for an **independent imaging-derived depth proxy** (unWISE `Nexp`) materially changes the “cleaned” dipole direction — classic sign the signal is not “fundamental”.

## What’s inside

### Narrative report + key plots

- `Q_D_RES/dipole_master_tests.md`
- `Q_D_RES/axis_alignment_mollweide.png`
- `Q_D_RES/dipole_vs_w1max.png`
- `Q_D_RES/fixed_axis_scaling_fit.png`
- `Q_D_RES/glm_cv_axes_nexp_offset.png`
- `Q_D_RES/glm_cv_angles_to_sn_nexp_offset.png`

### Key small JSON artifacts (easy to cite)

- `Q_D_RES/secrest_reproduction_dipole.json`  
  Local reproduction of Secrest-style dipole on the CatWISE AGN catalog.

- `Q_D_RES/fixed_axis_scaling_fit.json`  
  Scaling fit across W1 magnitude cuts (the “derivative” style evidence).

- `Q_D_RES/glm_cv_summary_w1cov_glm_dipole_NVSS.json`  
  Earlier GLM+CV result (scan templates; catalog-derived w1cov).

- `Q_D_RES/glm_cv_summary_nexp_offset_nodipole.json`  
  GLM+CV with **independent unWISE Nexp offset** + Secrest exclude mask (primary robustness upgrade).

## Where the heavy stuff lives (not copied here)

Large cached data products live under `data/` and `outputs/` (both are gitignored).

In particular the independent depth proxy is built from unWISE `neo7` W1 exposure-count maps and cached here:

- `data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json`

Built by:

- `scripts/build_unwise_nexp_tile_stats.py`

and consumed by:

- `experiments/quasar_dipole_hypothesis/vector_convergence_glm_cv.py`

