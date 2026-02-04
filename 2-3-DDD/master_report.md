# 2-3-DDD: CMB-projection decomposition of the Poisson-GLM dipole scan

## What this is
This is a **post-processing diagnostic** of already-produced Poisson GLM dipole-scan outputs.
It rewrites each best-fit dipole vector as:
- a component **parallel** to the CMB dipole direction, and
- a component **perpendicular** to the CMB dipole direction.

This addresses the “axis vs direction” ambiguity: a kinematic dipole is a *directional* prediction (sign matters), while our drift plots often use **axis** angles (sign-invariant).

## Inputs
All inputs are JSON scan products already tracked in the repo:
- `Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json`
- `Q_D_RES/rvmp_fig5_poisson_glm_w1cov_covariate.json`
- `Q_D_RES/rvmp_fig5_poisson_glm_w1cov_offset.json` *(included only as a cautionary variant; see note below)*

## Method
For each magnitude cut, we take the recovered dipole amplitude and direction `(D, \hat d)` and compute:

- `cosθ = \hat d · \hat d_CMB`
- `D_∥ = D cosθ`  (signed component along the CMB dipole direction)
- `D_⊥ = D sqrt(1 - cos^2θ)` (magnitude of the component orthogonal to the CMB direction)

So, if the recovered dipole were purely kinematic **and** correctly estimated, we would expect `D_⊥ ≈ 0` (up to noise/mode-coupling).

## Outputs
Main, interpretable bundle (excludes the offset variant so the plot scale is readable):
- `2-3-DDD/artifacts_main/cmb_projection_plot.png`
- `2-3-DDD/artifacts_main/cmb_projection_summary.csv`
- `2-3-DDD/artifacts_main/cmb_projection_summary.json`

Supplementary bundle including the `w1cov_offset` variant:
- `2-3-DDD/artifacts/cmb_projection_plot.png`
- `2-3-DDD/artifacts/cmb_projection_summary.csv`
- `2-3-DDD/artifacts/cmb_projection_summary.json`

## Key numbers (cumulative scan)

| scan | `W1_max` | `D` | `θ` to CMB (deg) | `D_∥` | `D_⊥` |
|---|---:|---:|---:|---:|---:|
| ecliponly | 15.5 | 0.01700 | 1.35 | 0.01700 | 0.00040 |
| ecliponly | 16.6 | 0.01678 | 34.33 | 0.01386 | 0.00946 |
| w1cov_covariate | 15.5 | 0.01640 | 8.27 | 0.01623 | 0.00236 |
| w1cov_covariate | 16.6 | 0.01586 | 33.99 | 0.01315 | 0.00887 |

## Interpretation (why this is useful for the paper)
- The previously-reported **axis drift** with `W1_max` can be reframed as: as the sample is pushed fainter, a **large non-CMB (perpendicular) component** grows.
- The **CMB-parallel component stays large** (it does not vanish), so “direction drift” should not be described as “the dipole becomes unrelated to the CMB”; it is better described as “a substantial additional component appears.”
- This supports the systematics/completeness interpretation: a depth/selection-linked dipole **not aligned with the CMB** can grow with `W1_max` and rotate the best-fit direction even when the total amplitude stays near `~1.6e-2`.

### Note on `w1cov_offset`
The `w1cov_offset` scan produces very large dipoles (order `~0.1`), and should be treated as a **modeling/identifiability caution** rather than a physically interpretable result. It is included only to document behavior under an “offset vs covariate” choice.

## Repro command
Main bundle:
```bash
./.venv/bin/python scripts/analyze_cmb_projection_from_glm_scan.py \
  --inputs \
    Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json \
    Q_D_RES/rvmp_fig5_poisson_glm_w1cov_covariate.json \
  --labels ecliponly w1cov_covariate \
  --outdir 2-3-DDD/artifacts_main
```

Supplementary (includes `w1cov_offset`):
```bash
./.venv/bin/python scripts/analyze_cmb_projection_from_glm_scan.py \
  --inputs \
    Q_D_RES/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json \
    Q_D_RES/rvmp_fig5_poisson_glm_w1cov_covariate.json \
    Q_D_RES/rvmp_fig5_poisson_glm_w1cov_offset.json \
  --labels ecliponly w1cov_covariate w1cov_offset \
  --outdir 2-3-DDD/artifacts
```
