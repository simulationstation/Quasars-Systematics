# Q\_D\_RES\_2\_2 (Master Report for paper update)

Date: 2026-02-03 (UTC)

This folder is a “paper update bundle” that consolidates:
- the core reproduction figures used in the ApJL draft,
- the Poisson-GLM scan tables (including jackknife + template–dipole diagnostics),
- new **clustered-mock (lognormal) LSS covariance** results for the dipole amplitude,
- an **end-to-end completeness validation** showing how depth-linked selection systematics can bias **both**
  direction and amplitude when the depth model is misspecified.

## Contents

### Figures (drop-in)

Core figures used by the current draft:
- `figures/rvmp_fig5_repro_baseline.png`  
  Secrest-style linear reproduction scan vs `W1_max`.
- `figures/rvmp_fig5_poisson_glm_ecliponly.png`  
  Poisson GLM scan vs `W1_max` using only an ecliptic-latitude template (this copy is the jackknife-cumulative plot).
- `figures/rvmp_fig5_repro_inject_dm0125cmb.png`  
  Injection test: dipolar effective faint-limit modulation (`delta_m=0.0125` mag) along the CMB axis.
- `figures/glm_cv_axes_nexp_offset.png`  
  Depth-template sensitivity (example axis shifts under plausible depth/coverage modeling changes).

Extra robustness/diagnostic figures:
- `figures/glm_cv_angles_to_sn_nexp_offset.png` (depth-template sensitivity diagnostic)
- `figures/template_fit_maps.png` (template/dipole visual diagnostics)
- `figures/rvmp_fig5_poisson_glm_inject_dm0125cmb.png` (Poisson-GLM version of the injection test)
- `figures/rvmp_fig5_poisson_glm_unwiseNexp_cov.png` (unWISE depth proxy variant used previously)

New figures for the paper update (LSS covariance + validated completeness):
- `figures/lss_cov_D_hist_w1max16p6.png`  
  Amplitude distribution from clustered (lognormal) mocks at `W1_max=16.6` including LSS+shot noise.
- `figures/validate_depth_systematic_recovery.png`  
  End-to-end validation: inject (i) a known dipole and (ii) a depth-linked selection systematic, then compare fits
  **without** vs **with** a depth-map template.
- `figures/ecllon_proxy.png`  
  Fast “seasonal imprint” proxy: fit the dipole in bins of ecliptic longitude and compare a full-sky fit with/without
  low-order ecliptic-longitude templates (`sinλ`, `cosλ`).

### Data products (tables + covariances)

Poisson GLM scan tables:
- `data/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json` (main scan table; includes jackknife + degeneracy diagnostics)
- `data/rvmp_fig5_poisson_glm_ecliponly_differential.json` (differential-bin diagnostic)
- `data/rvmp_fig5_poisson_glm_w1cov_covariate.json` and `data/rvmp_fig5_poisson_glm_w1cov_offset.json` (depth-proxy sensitivity)

LSS covariance (clustered mocks):
- `data/lognormal_cov_w1max16p6_n500/lognormal_mocks_cov.json`
- `data/lognormal_cov_w1max16p6_n500/b_est.npy`, `D_est.npy` (per-mock recovered dipole vectors/amplitudes)

Validated completeness (known-truth injection + depth systematic):
- `data/validate_inj_depth_misspecified/lognormal_mocks_cov.json` (+ `b_est.npy`, `D_est.npy`, `axis_angle_to_inj_deg.npy`)
- `data/validate_inj_depth_modeled/lognormal_mocks_cov.json` (+ `b_est.npy`, `D_est.npy`, `axis_angle_to_inj_deg.npy`)

Map-level depth proxy used by the validation:
- `data/depth_maps/lognexp_healpix_nside64.fits` (+ `.meta.json`)

Fast proxy (“seasonal imprint” in ecliptic longitude):
- `data/ecllon_proxy.json`

## Headline numbers (copy-ready)

### Direction drift (Poisson GLM; ecliptic-template-only)

From `data/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json` (using `l_hat_deg,b_hat_deg` and the Secrest+22 CMB axis):
- At `W1_max=15.5`: angle-to-CMB ≈ **1.35°**
- At `W1_max=16.5`: angle-to-CMB ≈ **27.70°**
- At `W1_max=16.6`: angle-to-CMB ≈ **34.33°**

The sky jackknife axis scatter at fixed cut is small compared to the drift across cuts:
- At `W1_max=16.6`: jackknife axis-angle-to-full median ≈ **1.22°**

### Amplitude significance with LSS covariance (clustered mocks)

From `data/lognormal_cov_w1max16p6_n500/lognormal_mocks_cov.json`:
- Best-fit (real data; Poisson GLM): **D\_hat = 0.01678**
- Mock percentiles: **p16/p50/p84 = 0.01463 / 0.01715 / 0.01922**
  - Rough 1σ: **σ\_D ≈ (p84−p16)/2 ≈ 0.00230**
  - Amplitude-only significance: **D/σ\_D ≈ 7.3**
  - Dipole-vector Mahalanobis S/N (using mock `cov_b`): **≈ 9.3**

Interpretation for the paper: **including LSS does not erase “non-zero amplitude”**, but it changes the error model
compared to Poisson-only MC.

### End-to-end completeness validation (known truth + misspecification)

These two runs inject:
- a known dipole: `inject_dipole_amp = 0.005` along the CMB axis, and
- a depth-linked selection systematic: `mu -> mu * exp(alpha * depth_z)` with **alpha = 0.04**,
  where `depth_z` is a z-scored map-level depth proxy (`lognexp_healpix_nside64.fits`).

Calibration note: the z-scored depth map has an intrinsic dipole amplitude `|a| ≈ 0.389`, so `alpha≈0.041`
corresponds to a selection-induced dipole at the observed `D~0.016` scale.

Results (500 mocks each):
- **Misspecified fit** (no depth template): `data/validate_inj_depth_misspecified/lognormal_mocks_cov.json`
  - D\_p50 ≈ **0.01270** (inflated relative to injected 0.005)
  - axis-angle-to-injected p50 ≈ **83.48°** (strong direction bias)
- **Modeled fit** (depth-map covariate included): `data/validate_inj_depth_modeled/lognormal_mocks_cov.json`
  - D\_p50 ≈ **0.00586** (close to injected 0.005)
  - axis-angle-to-injected p50 ≈ **28.50°** (improved recovery but still broad scatter at this injected amplitude)

Interpretation for the paper: **a slightly misspecified depth/completeness model can bias both direction and amplitude**,
and adding the correct map-level depth template materially improves recovery in known-truth tests.

### Fast proxy: ecliptic-longitude dependence (seasonal imprint proxy)

From `data/ecllon_proxy.json` (Poisson GLM at `W1_max=16.6`, `nside=64`, `abs_elat` template; bins of ecliptic longitude):
- Full-sky baseline (abs\_elat only): **D\_hat = 0.01678**, angle-to-CMB ≈ **34.33°**
- Full-sky with additional low-order longitude templates (`sinλ`, `cosλ`): **D\_hat = 0.01289**, angle-to-CMB ≈ **79.77°**

Ecliptic-longitude-bin fits (abs\_elat template only) show large variation in recovered axis across longitude ranges:
- λ∈[180°,270°]: angle-to-CMB ≈ **3.7°**
- λ∈[0°,90°]: angle-to-CMB ≈ **68.8°**
- λ∈[90°,180°]: angle-to-CMB ≈ **34.4°**
- λ∈[270°,360°]: angle-to-CMB ≈ **56.4°**

Important caveat for the paper: **partial-sky wedge fits inflate D and are not directly comparable to the full-sky D**;
the key diagnostic is the strong *directional* dependence on ecliptic longitude and the sensitivity of the full-sky
fit to adding simple ecliptic-longitude templates.

## Suggested paper edits (minimal, referee-proof)

1) **Upgrade the error model language**:
   - Add a short paragraph noting that Poisson-only errors are optimistic and quoting the LSS+shot-noise result above.
   - Point to `figures/lss_cov_D_hist_w1max16p6.png` as the compact visual.

2) **Make identifiability explicit (direction + amplitude)**:
   - Add 2–3 sentences explaining that depth-template misspecification can bias **both** the recovered axis and D.
   - Cite the end-to-end validation figure `figures/validate_depth_systematic_recovery.png`.

3) **Keep the main claim focused**:
   - “Amplitude robustly non-zero” still holds under clustered mocks.
   - “Direction degenerate with depth modeling” is strengthened by the validation, without overstating a cosmological conclusion.

## Repro commands (repo-local)

Run from the repository root. See `README.md` for full details.

- Poisson GLM scan (with jackknife):
  - `python3 scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py --help`
- Clustered mocks (LSS covariance):
  - `python3 scripts/run_catwise_lognormal_mocks.py --help`
