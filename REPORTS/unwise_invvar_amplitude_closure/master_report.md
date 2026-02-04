# invvar-based amplitude closure diagnostic

This report uses an imaging-derived unWISE W1 inverse-variance depth proxy to build an effective
limiting-magnitude shift map δm(n) and predicts the selection-driven dipole amplitude across W1 cuts.

- invvar map: `data/cache/unwise_invvar/neo7/invvar_healpix_nside64.fits`
- GLM scan: `REPORTS/Q_D_RES_2_2/data/rvmp_fig5_poisson_glm_ecliponly_cumulative_jk.json`

## Key numbers

- δm dipole amplitude: `0.0724852` mag
- δm dipole axis (Galactic): `(l,b)=(106.51°, 28.20°)`

- At `W1_max=16.60`: `D_obs=0.01678`, `D_sel=0.12826`, `D_res=0.13507`, `alpha_edge=1.769`

## Files

- JSON: `outputs/unwise_invvar_amp_closure_20260204_091900UTC/closure/invvar_amplitude_closure.json`
- Plot: `outputs/unwise_invvar_amp_closure_20260204_091900UTC/closure/invvar_amplitude_closure.png`
- Map: `outputs/unwise_invvar_amp_closure_20260204_091900UTC/closure/invvar_delta_m_mollweide.png`

## Interpretation (one line)

If `D_sel(W1_max)` tracks `D_obs(W1_max)`, this supports a depth-driven selection origin for the amplitude.
If `D_sel` is far larger/smaller than `D_obs`, then the depth proxy is *not* directly equal to an effective
magnitude-cut modulation δm_eff, and a calibrated completeness roll-off model is required.
