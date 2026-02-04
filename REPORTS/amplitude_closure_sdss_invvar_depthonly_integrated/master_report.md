# Amplitude closure attempt (SDSS DR16Q depth-only × invvar depth)

This report trains an SDSS DR16Q external completeness model using an invvar-based unWISE depth proxy (depth-only features),
then applies its integrated completeness prediction as a fixed offset in the CatWISE Poisson GLM dipole scan.

## Artifacts

- SDSS model report: `REPORTS/external_completeness_sdss_dr16q_invvar_depth_only/`
- Corrected GLM scan JSON: `REPORTS/amplitude_closure_sdss_invvar_depthonly_integrated/data/rvmp_fig5_poisson_glm_sdss_depthonly_integrated_offset.json`
- Corrected GLM scan PNG: `REPORTS/amplitude_closure_sdss_invvar_depthonly_integrated/figures/rvmp_fig5_poisson_glm_sdss_depthonly_integrated_offset.png`
- CMB projection compare: `REPORTS/amplitude_closure_sdss_invvar_depthonly_integrated/figures/cmb_projection_compare.png`

## Key numbers (W1_max=16.6)

- Baseline (eclip-only): D_hat=0.0167785
- Corrected (depth-only integrated offset): D_hat=0.018325  (+9.2% vs baseline)

Notes:
- This is an *externally trained* model (DR16Q truth) but not necessarily *all-sky validated* (DR16Q sky coverage is not uniform).
- If the corrected amplitude drops materially, that supports a selection/completeness origin for amplitude; if it does not, significant residual remains.