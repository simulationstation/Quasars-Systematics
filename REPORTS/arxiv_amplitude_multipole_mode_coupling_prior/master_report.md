# ArXiv amplitude check: low-ℓ multipole mode coupling **with harmonic priors** (Abghari+24-inspired)

Motivation: Abghari et al. (arXiv:2405.09762) argue the CatWISE quasar sky is not a pure dipole and contains low-ℓ multipoles comparable to the dipole.
On a partial-sky mask, these modes couple and can inflate dipole uncertainty and/or bias the recovered amplitude.

This folder upgrades the earlier “free low-ℓ nuisance” test (`REPORTS/arxiv_amplitude_multipole_mode_coupling/`) by adding a **physically motivated Gaussian prior**
for ℓ≥2 harmonic coefficients using **Cℓ estimated from clustered (lognormal) mocks**.

## What was run

We fit a Poisson GLM on HEALPix counts (Secrest-style footprint mask; `nside=64`; `|b|>30°`; `W1cov>=80`; exclusion discs), at a single representative faint limit:

- `W1_max = 16.6`

Model pieces:
- baseline model: dipole + `abs(ecliptic latitude)` template
- harmonic nuisance extension: add real spherical-harmonic templates for ℓ=2..ℓ_max
  - **free** coefficients (worst-case degeneracy), or
  - **Gaussian-prior** coefficients with `Var(a_{ℓm}) = scale * C_ℓ`

Prior source (Cℓ):
- `REPORTS/Q_D_RES_2_2/data/lognormal_cov_w1max16p6_n500/lognormal_mocks_cov.json` (`cl_estimate.cl_signal`)

## Outputs

- Data: `REPORTS/arxiv_amplitude_multipole_mode_coupling_prior/data/*.json`
- Summary: `REPORTS/arxiv_amplitude_multipole_mode_coupling_prior/data/summary.json`
- Figure: `REPORTS/arxiv_amplitude_multipole_mode_coupling_prior/figures/D_vs_prior_scale.png`

## Key amplitude results (W1_max = 16.6)

From `REPORTS/arxiv_amplitude_multipole_mode_coupling_prior/data/summary.json`.

| case | ℓ_max | prior | scale | D_hat | approx σ_D |
|---|---:|---|---:|---:|---:|
| baseline | 1 | none | – | 0.01678 | 0.00152 |
| free nuisance | 5 | none | – | 0.02347 | 0.00638 |
| harmonic prior | 5 | lognormal Cℓ | 1 | 0.01783 | 0.00216 |

Kinematic expectation reference: `D_kin ≈ 0.0046`.

Approx “(D_hat − D_kin)/σ_D” (purely a scale-of-tension number):
- baseline: ~8.0σ
- harmonic prior (ℓ≤5, scale=1): ~6.1σ

## Interpretation

- Allowing low-ℓ modes to float freely can materially change the fitted amplitude and inflate its uncertainty, consistent with Abghari+24’s basic identifiability warning.
- Under an LSS-motivated harmonic prior anchored to a clustered-mock Cℓ, the recovered amplitude stays close to the baseline value, with only modest uncertainty inflation.

## Injection validation

A small end-to-end injection check for the harmonic-prior method lives in:
- `REPORTS/arxiv_amplitude_multipole_prior_injection/`

