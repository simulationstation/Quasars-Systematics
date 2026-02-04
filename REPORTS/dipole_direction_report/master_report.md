# Dipole Direction Report (Fast Ecliptic-Longitude Proxy)

Date: 2026-02-03 (UTC)

This folder contains a self-contained bundle for the **fast “seasonal imprint” proxy test**: checking whether the
inferred CatWISE dipole direction depends strongly on **ecliptic longitude** and/or is absorbed by simple
low-order ecliptic-longitude templates.

## What this test is (plain language)

WISE observing conditions vary with time/season, but those time-dependent effects can project onto the sky as
fixed spatial patterns in **ecliptic coordinates**. If the recovered dipole direction changes a lot when we:
1) look at different **ecliptic-longitude slices** of the sky, or
2) add simple **sin/cos(ecliptic longitude)** nuisance templates,
then the dipole direction is likely being driven (at least in part) by scan/season-related systematics rather than
a single stable cosmological axis.

This is a **proxy**: it does not “prove time dependence,” but it is a strong diagnostic for ecliptic-structured
systematics.

## Exact analysis definition

Data and cuts (same baseline used elsewhere in this repo):
- CatWISE accepted catalog + Secrest exclusion mask.
- Coverage cut: `W1cov >= 80`
- Galactic cut: `|b| > 30°`
- Fixed footprint mask built from the parent (`W1cov>=80`) selection (mask-zeros + neighbors + exclude discs + |b| cut).
- Faint cut: `W1 <= 16.6`
- HEALPix: `nside = 64` (RING)

Estimator:
- Poisson GLM on HEALPix counts:
  - baseline nuisance template: `abs(ecliptic latitude)` (z-scored)
  - optional longitude templates: `sin(λ)` and `cos(λ)` (z-scored), where `λ` is barycentric mean ecliptic longitude

Implementation:
- Script: `scripts/run_ecliptic_lon_proxy.py`

## Outputs in this folder

- Figure:
  - `REPORTS/dipole_direction_report/figures/ecllon_proxy.png`
- Data:
  - `REPORTS/dipole_direction_report/data/ecllon_proxy.json`

## Headline results (W1_max=16.6)

From `REPORTS/dipole_direction_report/data/ecllon_proxy.json`:

Full-sky fits:
- Baseline (abs_elat template only):
  - `D_hat = 0.01678`
  - axis angle to CMB dipole axis: `34.33°`
- Baseline + simple ecliptic-longitude templates (`sinλ`, `cosλ`):
  - `D_hat = 0.01289`
  - axis angle to CMB dipole axis: `79.77°`

Ecliptic-longitude-bin fits (4 wedges; abs_elat only):
- λ∈[0°,90°]:  `D_hat = 0.05416`, angle-to-CMB `68.8°`
- λ∈[90°,180°]: `D_hat = 0.06836`, angle-to-CMB `34.4°`
- λ∈[180°,270°]: `D_hat = 0.05767`, angle-to-CMB `3.7°`
- λ∈[270°,360°]: `D_hat = 0.06555`, angle-to-CMB `56.4°`

Important caveat:
- These wedge/partial-sky fits **inflate** `D_hat` and are not directly comparable to the full-sky amplitude.
- The key diagnostic is the **large directional variation** across ecliptic longitude and the strong sensitivity of
  the full-sky solution to adding low-order longitude templates.

## Interpretation (what this implies)

This test shows that the inferred dipole direction at `W1_max=16.6` is **strongly dependent on ecliptic longitude**
and is **highly sensitive** to adding simple ecliptic-longitude nuisance terms.

That pattern is consistent with the idea that the dipole direction is being influenced by **ecliptic-structured
survey/scan systematics** (which can originate from time/season observing conditions) rather than representing one
stable physical sky axis.

## Prior literature sanity check (is this already in “their” papers?)

I checked several of the main CatWISE dipole papers for an analysis that matches this specific proxy test
(longitude wedges + `sinλ/cosλ` nuisance terms).

- **Closest precedent (but not the same test):**
  - **Secrest et al. (2022; arXiv:2206.05624)** explicitly discuss narrow stripes of reduced sensitivity at specific
    **ecliptic longitudes** (citing Singal 2021) and test a **masking** of four short λ ranges, reporting an amplitude
    change. This is a longitude-related systematics check, but it is *not* a longitude-binned direction-drift study
    and it is *not* a `sinλ/cosλ` nuisance-template fit.
- **No clear match found:**
  - **Secrest et al. (2020; arXiv:2009.14826)**: emphasizes ecliptic-*latitude* scan structure; I did not find
    longitude wedge/template tests.
  - **Dam et al. (2022; arXiv:2212.07733)**, **Abghari et al. (2024; arXiv:2405.09762)**, and
    **von Hausegger et al. (2025; arXiv:2510.23769)**: discuss ecliptic-latitude trends / depth systematics, but I
    did not find this specific longitude-proxy analysis.

## How to reproduce

Run from the repo root:

```bash
python3 scripts/run_ecliptic_lon_proxy.py \
  --catalog /home/primary/PROJECT/data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --exclude-mask-fits /home/primary/PROJECT/data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits \
  --w1cov-min 80 \
  --b-cut 30 \
  --nside 64 \
  --w1-max 16.6 \
  --lambda-edges 0,90,180,270,360 \
  --make-plot \
  --outdir outputs/ecllon_proxy_run
```
