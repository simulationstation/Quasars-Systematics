# Seasonal selection injection check (ecliptic longitude)

This folder contains an injection check meant to quantify how much a scan-linked selection/completeness pattern can bias the recovered dipole amplitude if it is *omitted* from the fit.

## Model

We generate Poisson mock maps from

- `log λ(pix) = log μ_hat(pix) + (b · n_hat)(pix) + A * [cos(φ) * cos(λ)_z + sin(φ) * sin(λ)_z]`

where:

- `μ_hat` is a baseline fit to the real data using only `abs(ecliptic latitude)` (no injected dipole).
- `b · n_hat` is an optional injected dipole in log-intensity.
- `cos(λ)_z` and `sin(λ)_z` are z-scored ecliptic-longitude templates on *seen* pixels.
- `A` is the injected ecliptic-longitude modulation amplitude (in the z-scored basis).

We then recover a dipole with two models:

- **baseline_fit**: dipole + `abs_elat`
- **with_lon_templates**: dipole + `abs_elat` + `sinλ`/`cosλ`

Important: because dipole amplitude is positive-definite, the median `D` can remain large even when the *mean dipole vector* is unbiased (especially when adding extra templates). For bias, the relevant diagnostic is `D_of_b_mean = |mean(b_vec)|`.

## Key results (W1_max=16.6)

### Data-like phase/amplitude

The real-data ecliptic-longitude proxy fit at `W1_max=16.6` (see `REPORTS/dipole_direction_report/data/ecllon_proxy.json`) returns

- `A_data ≈ 0.00819` (z-scored basis amplitude)
- `φ_data ≈ 166.26°`

Injection runs using that `(A, φ)` show:

- **No true dipole injected** (`dipole0_phase166/`):
  - baseline_fit: `D_of_b_mean ≈ 0.01114`
  - with_lon_templates: `D_of_b_mean ≈ 0.00049`

- **Kinematic-scale dipole injected** (`dipole0p0046_phase166/`, `D_inj=0.0046` toward (l,b)=(264.021°,48.253°)):
  - baseline_fit: `D_of_b_mean ≈ 0.01526`
  - with_lon_templates: `D_of_b_mean ≈ 0.00439`

So, at least in this controlled mock setup, a percent-level scan-linked modulation consistent with the fitted lon templates can move a kinematic-scale dipole toward the observed all-sky amplitude.

### Phase=0 sweep (sanity trend)

The `dipole0/` run (phase=0°, `A∈{0,0.01,0.02,0.03,0.04}`) shows an approximately linear relationship between injected lon-template amplitude and the recovered dipole bias when the lon templates are omitted.

## Outputs

Each run subfolder contains:

- `seasonal_injection.json`
- `seasonal_injection.png`

Run subfolders created in this session:

- `dipole0/` and `dipole0p0046/`
- `dipole0_phase166/` and `dipole0p0046_phase166/`

## Reproduce

```bash
python3 scripts/seasonal_selection_injection_check.py \
  --w1-max 16.6 \
  --n-mocks 200 \
  --lon-amps 0,0.01,0.02,0.03,0.04 \
  --lon-phase-deg 0 \
  --dipole-amp 0 \
  --outdir REPORTS/seasonal_selection_injection_check/dipole0

python3 scripts/seasonal_selection_injection_check.py \
  --w1-max 16.6 \
  --n-mocks 400 \
  --lon-amps 0,0.008192567026925831 \
  --lon-phase-deg 166.2604027634997 \
  --dipole-amp 0.0046 \
  --outdir REPORTS/seasonal_selection_injection_check/dipole0p0046_phase166
```
