# CatWISE-parent unWISE time-domain epoch-resolved dipole amplitude

Date: 2026-02-20 (UTC)

This report summarizes an **epoch-resolved, amplitude-only** dipole stability test using the
unWISE Time-Domain Catalog (IRSA580) **restricted to the published CatWISE/Secrest accepted AGN parent sample**.

If the dipole amplitude is cosmological/kinematic, it should be stable across epochs. If it is selection-driven,
it can vary with epoch/coverage/background.

## Exact definition

Parent catalog:
- Secrest+22 accepted CatWISE AGN catalog (Zenodo 6784602), filtered to:
  - `W1 <= 16.4` (Vega)
  - `W1-W2 >= 0.8` (Vega)
  - `W1cov >= 80.0`
  - Secrest-style footprint mask (`mask_zeros` + exclusion discs + `|b| > 30.0°`)
- Parent size: `N_parent = 1359698`

Epoch selection (time-domain):
- Object must have a matched (W1,W2) time-domain measurement in that epoch passing:
  - `primary==1`, `flags_unwise==0`, `flags_info==0`, `flux>0`, `dflux>0`
  - `W1 <= 16.4` via flux threshold
  - `SNR_W1 >= 5.0`; `SNR_W2 >= 0.0`
  - `apply_color_cut = True`
- Matching to parent uses a `match_radius = 2.0 arcsec` nearest-neighbor in ICRS.

Footprint mask:
- Fixed across epochs (same mask as the parent definition), HEALPix `nside=64`, Galactic, RING.

Estimator:
- Primary: Poisson GLM dipole amplitude `D = |b|` on masked HEALPix maps.
- Cross-check: vector-sum amplitude on the same masked maps.

## Headline results (epochs 0–15)

Poisson GLM amplitude:
- `D_min = 0.06722`
- `D_max = 0.11765`
- `D_range = 0.05043`

Vector-sum amplitude:
- `D_min = 0.05834`
- `D_max = 0.15529`
- `D_range = 0.09695`

Per-epoch sample size (0–15):
- `N_min = 2.08e+05`
- `N_max = 2.3e+05`

## Figures

![](REPORTS/unwise_time_domain_catwise_epoch_amplitude/figures/D_vs_epoch_glm.png)

![](REPORTS/unwise_time_domain_catwise_epoch_amplitude/figures/D_vs_epoch_compare.png)

![](REPORTS/unwise_time_domain_catwise_epoch_amplitude/figures/N_vs_epoch.png)

## Reproduce

Run directory:
- `outputs/epoch_dipole_time_domain_catwise_parent_20260204_232537UTC`

Report directory:
- `REPORTS/unwise_time_domain_catwise_epoch_amplitude`
