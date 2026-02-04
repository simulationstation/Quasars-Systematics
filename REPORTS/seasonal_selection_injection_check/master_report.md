# Seasonal selection injection check (ecliptic longitude)

Purpose: quantify how much a scan/season-linked **selection/completeness pattern** can bias the recovered dipole *amplitude* if it is **omitted** from the fit.

The key idea is simple: WISE observing conditions vary in time, but because the scan strategy is tied to the ecliptic, many time-dependent effects project onto the sky as fixed structure in ecliptic coordinates (especially ecliptic longitude λ). A dipole-only fit can absorb that structure.

## Model

We generate Poisson mock maps on the masked sky using

- `log λ(pix) = log μ_hat(pix) + (b · n_hat)(pix) + A * [cos(φ) * cos(λ)_z + sin(φ) * sin(λ)_z]`

where:

- `μ_hat` is a baseline fit to the real map using only `abs(ecliptic latitude)` (no injected dipole).
- `(b · n_hat)` is an optional injected dipole in log-intensity.
- `sin(λ)_z` and `cos(λ)_z` are z-scored ecliptic-longitude templates on *seen* pixels.
- `A, φ` specify the injected longitude-pattern amplitude/phase (in the z-scored basis).

We then recover a dipole with two fits:

- **baseline_fit**: dipole + `abs_elat`
- **with_lon_templates**: dipole + `abs_elat` + `sinλ`/`cosλ`

Important: dipole amplitude `D=|b|` is positive-definite, so its median can remain nonzero even when the dipole vector is unbiased. For *bias*, we track

- `D_of_b_mean = |mean(b_vec)|`.

## Real-data lon-template scale (W1_max=16.6)

From `REPORTS/dipole_direction_report/data/ecllon_proxy.json` (the full-sky fit **including** `sinλ/cosλ`), the best-fit lon-template coefficients are

- `sinλ` coeff = `+0.0053068`
- `cosλ` coeff = `−0.0126857`

so the implied lon-template amplitude/phase are

- `A_data = sqrt(sin^2 + cos^2) = 0.0137509`
- `φ_data = atan2(sin, cos) = 157.299°`.

Caveat: these coefficients are moderately significant and can be partially degenerate with the dipole itself; treat `(A_data, φ_data)` as an *order-of-magnitude guide* for a plausible scan-linked pattern, not a calibrated physical completeness amplitude.

## Injection results (W1_max=16.6)

### Data-like phase/amplitude injection (φ=157.299°, A=0.0137509)

- **No true dipole injected** (`dipole0_phase157/`, `D_inj=0`):
  - baseline_fit: `D_of_b_mean ≈ 0.019888` (spurious dipole)
  - with_lon_templates: `D_of_b_mean ≈ 0.000512`

- **Kinematic-scale dipole injected** (`dipole0p0046_phase157/`, `D_inj=0.0046` toward (l,b)=(264.021°,48.253°)):
  - baseline_fit: `D_of_b_mean ≈ 0.023806`
  - with_lon_templates: `D_of_b_mean ≈ 0.004789` (close to injected)

Interpretation: in this controlled mock setup, a scan-linked longitude pattern at the level suggested by the `sinλ/cosλ` coefficients can generate an *apparent* dipole amplitude of order `~10^{-2}` if it is not modeled.

### Phase=0 sweep (sanity trend)

The `dipole0/` run (phase=0°, `A∈{0,0.01,0.02,0.03,0.04}`) shows an approximately linear relationship between injected lon-template amplitude and the recovered dipole bias when the lon templates are omitted.

## Outputs

Each run subfolder contains:

- `seasonal_injection.json`
- `seasonal_injection.png`

Run subfolders created in this repo:

- `dipole0/`, `dipole0p0046/` (phase=0 sweep + kinematic injection)
- `dipole0_phase157/`, `dipole0p0046_phase157/` (data-like lon-template amplitude/phase)

## Reproduce

```bash
python3 scripts/seasonal_selection_injection_check.py \
  --w1-max 16.6 \
  --n-mocks 400 \
  --lon-amps 0,0.013750934538540795 \
  --lon-phase-deg 157.29896537036782 \
  --dipole-amp 0.0046 \
  --outdir REPORTS/seasonal_selection_injection_check/dipole0p0046_phase157
```
