# 2-3-F — Full dark-siren fixed-axis anisotropy scan (3 axes)

**Date (UTC):** 2026-02-03  
**Artifacts:** `2-3-F/artifacts/` (copied from `outputs/darksiren_axis_triplet_full_20260203_230334UTC/`)  
**Machine-readable summary:** `2-3-F/summary_metrics.json`

## Goal

Run the **full likelihood** fixed-axis anisotropy scan discussed in `EntropyPaper.tex`, but only for three
physically motivated axes:

- `cmb` (CMB dipole axis)
- `secrest` (Secrest+22 fitted axis)
- `ecliptic_north` (scan/seasonal-systematics proxy axis)

The model tested is a **1-parameter directional modulation** of the GW luminosity distance prediction:

> `dL_gw(z, n) → dL_gw(z, n) * exp(g * cosθ)`  (θ measured from the chosen fixed axis)

We compare:

- **GR** (baseline, isotropic)
- **μ0** (entropy/HE model, isotropic; equivalent to `g=0`)
- **μ(g)** (entropy/HE model with anisotropy parameter `g`, marginalized with an explicit prior)

## What “full likelihood pieces” means here

Each axis run uses:

- **Catalog (“cat”) term** from the galaxy catalog likelihood
- **Missing-host term** (out-of-catalog integral)
- **Global `f_miss` marginalization** (shared nuisance) using the Beta-prior metadata stored in the production summary
- **Selection normalization** `α(model)` from O3 injections, applied at the correct combined-draw level

## Inputs / configuration

- **Event bundle:** `data/dark_sirens/2-1-c-m/production_36events/event_scores_M0_start101.json` (36 events)
- **Posterior:** `data/entropy_posteriors/M0_start101/samples/mu_forward_posterior.npz`
- **Draws:** 256 (from `data/dark_sirens/2-1-c-m/production_36events/summary_M0_start101.json: draw_idx`)
- **HEALPix / PE settings:** `Nside=64`, `p_credible=0.9` (from cached production artifacts)
- **g grid:** `[-0.6, …, 0.6]` in steps of `0.1` (13 points)
- **Prior on g:** Normal(0, 0.2)
- **Compute:** 3 axes in parallel, `--nproc 36` per axis (event-parallel workers)

The exact launcher used is in `2-3-F/artifacts/job.sh`.

## Results (g marginalization)

All three axes strongly prefer the **μ-model over GR** (this is largely axis-independent, since it is driven by the
isotropic μ0 piece).
The **extra anisotropy parameter `g`** is only **weakly** preferred over μ0.

| Axis | logBF(μ/GR) | BF(μ/GR) | logBF(μ(g)/μ0) | BF(μ(g)/μ0) | g (posterior mean ± std) | best g (grid) | ΔLPD(g=0) | best ΔLPD | gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cmb | 3.271 | 26.3 | 0.243 | 1.27 | +0.152 ± 0.124 | +0.2 | 3.028 | 4.179 | 1.151 |
| secrest | 3.244 | 25.6 | 0.216 | 1.24 | +0.149 ± 0.129 | +0.3 | 3.028 | 4.163 | 1.135 |
| ecliptic_north | 3.150 | 23.3 | 0.121 | 1.13 | −0.123 ± 0.158 | −0.3 | 3.028 | 3.952 | 0.924 |

Notes:
- `ΔLPD(g=0)` is the μ0 vs GR log-posterior-density gain (same across axes because `g=0` is isotropic).
- “gain” is `best ΔLPD − ΔLPD(g=0)` on the scanned grid (how much the best anisotropic point improves over μ0).
- None of these runs had evidence mass piling up at the grid edges (edge-mass diagnostics are in the JSONs).
- Axis preference among these 3 is *small*: ΔlogZ between the best axis (CMB) and the others is ≤ 0.12.

## Plots

Per-axis ΔLPD(g) curves:

- `2-3-F/artifacts/cmb/fixed_axis_gscan_full.png`
- `2-3-F/artifacts/secrest/fixed_axis_gscan_full.png`
- `2-3-F/artifacts/ecliptic_north/fixed_axis_gscan_full.png`

## Plain-English interpretation

- The data **clearly prefers the μ/entropy model over GR** in this setup (Bayes factors ~25×).
- Allowing a directional “speed”/distance modulation parameter `g` **does not buy much**:
  the Bayes factor for μ(g) over μ0 is only ~1.1–1.3× (marginal).
- The inferred `g` is **not far from zero** (roughly ~1σ-scale shifts), and its sign flips depending on axis.
- With only 3 axes, this does **not** identify a uniquely preferred axis; a larger random-axis sweep would be needed
  to quantify how “special” any one axis is relative to chance.

## Re-running

If you have a compatible production cache directory with:

- `<CACHE_OUTDIR>/cache/`
- `<CACHE_OUTDIR>/cache_terms/`
- `<CACHE_OUTDIR>/cache_missing/`

you can reproduce an axis run (example: CMB) via:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 scripts/run_darksiren_fixed_axis_gscan_full.py \
  --axis cmb \
  --g-grid=-0.6,0.6,0.1 \
  --nproc 36 \
  --g-prior-type normal --g-prior-mu 0.0 --g-prior-sigma 0.2 \
  --cache-outdir <CACHE_OUTDIR> \
  --outdir outputs/darksiren_fixed_axis_full_cmb \
  --make-plot
```

