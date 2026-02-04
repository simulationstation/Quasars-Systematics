# Harmonic-prior injection check

This folder contains a small end-to-end injection check used to sanity-test the harmonic-prior method under a masked sky with added low-ell modes.

## Run configuration

- nside: `64`
- W1_max: `16.6`
- mask b-cut (deg): `30.0`
- harmonic lmax: `5`
- injected low-ell scale (multiplier): `10.0`
- fit prior scale (multiplier): `1.0`
- n_mocks: `200`
- seed: `123`

Injected dipole amplitude: `D_inj = 0.016780`

## Results

Recovered dipole amplitude distribution (summary):

| fit | D_p50 | D_p16 | D_p84 | D_std |
|---|---:|---:|---:|---:|
| `baseline` | 0.018616 | 0.014938 | 0.022985 | 0.004248 |
| `free` | 0.018704 | 0.014800 | 0.023901 | 0.004632 |
| `prior` | 0.018031 | 0.014850 | 0.022585 | 0.003636 |

Notes:
- `free` is the over-parameterized limit (harmonics with no regularization).
- `prior` applies the Gaussian C_ell prior to the harmonic coefficients.

## Outputs

- JSON: `REPORTS/arxiv_amplitude_multipole_prior_injection/data/lowell_injection_validation.json`
- Figure: `REPORTS/arxiv_amplitude_multipole_prior_injection/figures/lowell_injection_validation.png`
