# PRD figure pack (Overleaf staging)

This repo also supports a larger PRD-style figure pack workflow. The PRD manuscript sources are
intentionally **not** tracked here; instead, this document records which PNGs to stage into an Overleaf
project and where they live in this repository.

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

