# Selection simulation from a real depth map (end-to-end)

This folder contains an end-to-end *selection simulation* that uses a real sky depth/completeness map to modulate the expected counts and then measures how much dipole amplitude is induced if that selection term is omitted from the fit.

## Setup

- Sky: nside=64, Secrest-style mask (w1cov>=80 parent, exclude discs, b_cut=30).
- Baseline mean model per cut: fit the real map with **no dipole**, only `intercept + abs_elat`.
- Selection term: use the SDSS-trained depth-only **delta-m** map `REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits`.
- For each W1 cut, compute `alpha_edge = d ln N(<m) / dm` from the seen-footprint W1 histogram and apply
  `selection_offset(pix) = sel_scale * alpha_edge(W1_max) * (delta_m(pix) - median(delta_m))`.

We then generate Poisson maps and fit dipoles two ways:
- **baseline_fit**: dipole + abs_elat (omits the selection term)
- **with_true_offset**: dipole + abs_elat with the *true injected selection offset* included as a fixed offset

## Real-data reference (W1_max=16.6)

- Baseline full-sky dipole amplitude (dipole+abs_elat): `D_hat = 0.016779`.
- Lon-template coefficients in the full-sky fit that includes `sinλ/cosλ`: `sinλ=+0.0053068`, `cosλ=-0.0126857`.

## Results at W1_max=16.6 (no true dipole injected)

| sel_scale | alpha_edge | baseline_fit |mean b| | with_true_offset |mean b| |
|---:|---:|---:|---:|
| 1.0 | 1.779 | 0.006257 | 0.000297 |
| 2.7 | 1.779 | 0.016468 | 0.000156 |

Interpretation:
- With `sel_scale=1`, the depth-map selection term induces a spurious dipole of order ~6×10^-3 at W1_max=16.6.
- Scaling the same selection term up to `sel_scale≈2.7` makes the induced amplitude ~1.65×10^-2, i.e. close to the observed baseline amplitude at W1_max=16.6.
- In both cases, when the **true** injected selection offset is included in the fit, the dipole bias collapses to ~0 (as it must in this controlled setup).

## Lon-template / wedge diagnostics (W1_max=16.6)

We also measured what `sinλ/cosλ` coefficients and λ-wedge dipoles appear in the simulated maps (200 mocks).

- Diagnostics (+2.7): `REPORTS/selection_sim_depthmap/delta_m_scale2p7_diag/selection_sim_depthmap_diagnostics.json`
- Diagnostics (-2.7): `REPORTS/selection_sim_depthmap/delta_m_scaleMinus2p7_diag/selection_sim_depthmap_diagnostics.json`

Key takeaway from the diagnostics:
- The **depth-map-only** selection simulation can be tuned to match the *amplitude* at one cut, but it does **not** reproduce the real-data `sinλ/cosλ` phase or the λ-wedge behavior out of the box.
- Therefore, this is evidence that *a depth-linked selection term can in principle create amplitude*, not a proof that this specific delta-m map (by itself) is the full explanation of the anomaly.

