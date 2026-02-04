# Amplitude closure attempt (SDSS DR16Q × invvar depth)

This report builds an SDSS DR16Q external completeness model using an invvar-based unWISE depth proxy,
exports a δm(n) map, and applies it as a fixed GLM offset (scaled by α_edge) in the CatWISE dipole scan.

## Artifacts

- SDSS model report: `REPORTS/external_completeness_sdss_dr16q_invvar/`
- Corrected GLM scan JSON: `REPORTS/amplitude_closure_sdss_invvar/data/rvmp_fig5_poisson_glm_sdss_invvar_dm_offset.json`
- Corrected GLM scan PNG: `REPORTS/amplitude_closure_sdss_invvar/figures/rvmp_fig5_poisson_glm_sdss_invvar_dm_offset.png`
- CMB projection compare: `REPORTS/amplitude_closure_sdss_invvar/figures/cmb_projection_compare.png`

## Key number (faintest cut in this scan)

- W1_max=16.6: corrected D_hat=0.0722078, (l,b)=(347.63°, 28.53°)

Interpretation: if this corrected scan’s amplitude is substantially reduced vs the baseline scan,
that supports a selection/completeness origin for the amplitude. If not, significant residual amplitude remains.