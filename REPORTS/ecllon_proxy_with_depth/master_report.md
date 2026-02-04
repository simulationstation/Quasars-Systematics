# Ecliptic-longitude proxy with physical depth/coverage covariates

This is a *season/time-linked* proxy: WISE scan conditions project onto fixed structure in ecliptic longitude λ.
We repeat the ecliptic-longitude proxy fits while also including an additional *physical* scan/depth covariate.

All runs: nside=64, W1_max=16.6, b_cut=30, w1cov_min=80, Secrest-style mask.

## Full-sky fits

| depth covariate | baseline D (dipole+abs_elat+depth) | +lon D (adds sinλ/cosλ) |
|---|---:|---:|
| `none` | 0.016779 | 0.012890 |
| `lognexp` | 0.015806 | 0.010254 |
| `invvar` | 0.016145 | 0.009568 |
| `w1cov_mean` | 0.015864 | 0.010373 |

Notes:
- Adding a depth/coverage covariate slightly reduces the baseline full-sky amplitude (from ~0.0168 to ~0.0158–0.0161 in these runs).
- Adding `sinλ/cosλ` still drives the best-fit dipole amplitude down, but increases degeneracy (fit becomes much less stable); interpret the drop as evidence of mode/covariate competition, not a clean “correction”.

## Outputs

- `REPORTS/ecllon_proxy_with_depth/none/ecllon_proxy_with_depth.json`
- `REPORTS/ecllon_proxy_with_depth/none/ecllon_proxy_with_depth.png`
- `REPORTS/ecllon_proxy_with_depth/lognexp/ecllon_proxy_with_depth.json`
- `REPORTS/ecllon_proxy_with_depth/lognexp/ecllon_proxy_with_depth.png`
- `REPORTS/ecllon_proxy_with_depth/invvar/ecllon_proxy_with_depth.json`
- `REPORTS/ecllon_proxy_with_depth/invvar/ecllon_proxy_with_depth.png`
- `REPORTS/ecllon_proxy_with_depth/w1cov_mean/ecllon_proxy_with_depth.json`
- `REPORTS/ecllon_proxy_with_depth/w1cov_mean/ecllon_proxy_with_depth.png`
