# Selection simulation (depth map + ecliptic-longitude pattern)

Goal: build an end-to-end *selection/completeness* simulation that can induce a spurious dipole if the selection term is omitted, and that becomes unbiased once the correct selection model is included.

This is a “time/season-linked” proxy: WISE observing conditions vary in time, but because the scan strategy is tied to the ecliptic, many time-dependent effects project onto the sky as fixed structure in ecliptic longitude λ.

## Real-data reference (W1_max=16.6)

From `REPORTS/dipole_direction_report/data/ecllon_proxy.json`:
- Full-sky baseline fit (dipole + abs_elat): `D_hat = 0.0167785`
- Full-sky fit including `sinλ/cosλ`: `D_hat = 0.0128897`
- Best-fit lon-template coefficients (z-scored basis):
  - `sinλ = +0.0053068`
  - `cosλ = −0.0126857`

Per-λ-wedge dipoles (each wedge fit is dipole + abs_elat):
- λ∈[0,90):   `D_hat=0.05416` at `(l,b)≈(215.7, −55.3)`
- λ∈[90,180): `D_hat=0.06836` at `(l,b)≈(210.8, +49.3)`
- λ∈[180,270):`D_hat=0.05767` at `(l,b)≈(263.4, +51.9)`
- λ∈[270,360):`D_hat=0.06555` at `(l,b)≈(333.0, −62.7)`

## Lon-pattern-only injection (end-to-end)

We generate Poisson mock maps on the masked sky using:
- a per-cut baseline mean `log(mu_hat)` fit to the **real** map using only `abs_elat` (no dipole)
- an injected selection offset built from `sinλ/cosλ` templates

**Across-cut run (40 mocks per cut)**
- Output: `REPORTS/selection_sim_depthmap_plus_lon/lon_only_scan/selection_sim_depthmap_plus_lon.json`
- Figure: `REPORTS/selection_sim_depthmap_plus_lon/lon_only_scan/selection_sim_depthmap_plus_lon.png`
- The injected `sinλ/cosλ` coefficients are taken per-cut from:
  - `outputs/rvmp_fig5_poisson_glm_eclip_sincos/rvmp_fig5_poisson_glm.json`

At `W1_max=16.6`, this produces:
- baseline_fit (dipole+abs_elat, selection omitted): `|mean(b)| ≈ 0.0203`
- with_lon_templates (dipole+abs_elat+sinλ/cosλ, still no offset): `|mean(b)| ≈ 0.00168`
- with_true_offset (dipole+abs_elat, with the injected selection offset): `|mean(b)| ≈ 0.00063`

Interpretation: a scan-linked lon pattern at the level implied by the fitted lon coefficients can generate an apparent dipole of order `~10^-2` if it is not modeled.

**Wedge diagnostics at W1_max=16.6 (120 mocks)**
- Output: `REPORTS/selection_sim_depthmap_plus_lon/lon_only_diag/selection_sim_depthmap_plus_lon_diagnostics.json`

This run injects `sinλ=+0.0053068`, `cosλ=−0.0126857` and recovers:
- baseline_fit dipole bias: `|mean(b)| ≈ 0.01986`
- with_lon_templates bias: `|mean(b)| ≈ 0.000657`
- with_true_offset bias: `|mean(b)| ≈ 0.000104`

Wedge dipole amplitudes (mean over mocks) are ~0.02–0.036 here, i.e. same order as real but not a full match in direction.

## Depth+lon combined injection (example)

- Output: `REPORTS/selection_sim_depthmap_plus_lon/depth1_lon_diag/selection_sim_depthmap_plus_lon_diagnostics.json`

This adds a depth-map selection term using the SDSS-trained depth-only `delta_m` map:
- `REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits`

with `depth_sel_scale=1.0` (and the same injected lon coefficients as above at 16.6).

Result: the wedge amplitudes can be pushed closer to the real per-wedge `D_hat` scale, but the wedge *directions* still do not match the data.

## Bottom line

- Selection patterns tied to ecliptic longitude can **create** an apparent dipole amplitude at the `~10^-2` level if unmodeled, and can be removed in controlled mocks.
- A depth-linked map term can be tuned to match the **full-sky amplitude**, but matching the **λ-wedge directions** and the full set of observed behaviors simultaneously still requires a richer physical selection model.
