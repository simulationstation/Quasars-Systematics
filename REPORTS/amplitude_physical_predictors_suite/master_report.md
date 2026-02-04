# Physical predictor control sweep (amplitude)

Goal: test whether adding *physical* scan/completeness predictors (coverage/depth proxies) drives the CatWISE dipole amplitude down toward the kinematic expectation (≈0.0046).

All runs: nside=64, W1_max=16.6, W1_grid=15.5..16.6 step 0.05, w1cov_min=80.

## Summary at W1_max=16.6

| run | D_hat | approx ± (from p16/p84) | (l,b) deg |
|---|---:|---:|---:|
| `bcut25_baseline` | 0.016852 | 0.00129 | (229.91, 21.11) |
| `bcut25_sdss_depthonly_offset` | 0.018532 | 0.00133 | (239.19, 14.41) |
| `bcut25_sdss_offset_plus_lon` | 0.013825 | 0.00242 | (248.94, -28.00) |
| `bcut25_unwise_invvar_cov` | 0.017042 | 0.00143 | (230.90, 20.38) |
| `bcut25_unwise_nexp_cov` | 0.016674 | 0.00139 | (228.55, 22.03) |
| `bcut25_w1cov_cov` | 0.016610 | 0.00141 | (228.08, 22.32) |
| `bcut30_baseline` | 0.016779 | 0.00147 | (236.58, 21.81) |
| `bcut30_sdss_depthonly_offset` | 0.018525 | 0.00147 | (244.41, 15.00) |
| `bcut30_sdss_offset_plus_lon` | 0.017686 | 0.00432 | (292.33, -50.54) |
| `bcut30_unwise_invvar_cov` | 0.016457 | 0.00156 | (235.02, 23.33) |
| `bcut30_unwise_nexp_cov` | 0.015806 | 0.00134 | (229.20, 27.53) |
| `bcut30_w1cov_cov` | 0.015864 | 0.00153 | (230.16, 26.70) |
| `bcut35_baseline` | 0.015502 | 0.00167 | (240.28, 21.38) |
| `bcut35_sdss_depthonly_offset` | 0.017281 | 0.00171 | (247.28, 14.07) |
| `bcut35_sdss_offset_plus_lon` | 0.024367 | 0.00630 | (254.08, -63.09) |
| `bcut35_unwise_invvar_cov` | 0.014553 | 0.00170 | (235.39, 27.41) |
| `bcut35_unwise_nexp_cov` | 0.014221 | 0.00167 | (230.93, 29.93) |
| `bcut35_w1cov_cov` | 0.014296 | 0.00164 | (232.31, 28.71) |

## Takeaway

- Across these masks (`b_cut`=25/30/35) and depth/coverage controls (mean `w1cov`, unWISE `logNexp`, unWISE `invvar`, SDSS-trained depth-only completeness offset), the recovered amplitude remains at the **~(1.4–1.9)×10^-2** level.
- In this suite, **it does not collapse to ~0.0046**, so these particular physical proxies do not, by themselves, explain the amplitude.

Outputs live under `REPORTS/amplitude_physical_predictors_suite/` (each subfolder contains `rvmp_fig5_poisson_glm.json` + `rvmp_fig5_poisson_glm.png`).
