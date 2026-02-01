# Quasars-Systematics

This repository contains a reproducible analysis and an ApJL letter draft arguing that the
CatWISE/Secrest quasar number-count dipole is **dominated by survey selection/systematics** tied to
the faint $W1$ magnitude boundary (rather than requiring a large intrinsic/cosmological dipole).

Key artifacts for reviewers are in `Q_D_RES/`:

- `Q_D_RES/Resolution.md` (ApJL letter draft in AASTeX; paste into Overleaf)
- `Q_D_RES/fixed_axis_scaling_fit.png` (main result figure used in the letter)
- `Q_D_RES/dipole_master_tests.md` (detailed run log + additional diagnostics and figures)
- `Q_D_RES/*.json` (small machine-readable summaries used for numbers/plots)

## References / DOIs used by this repository

- Secrest et al. 2022, ApJL 937 L31, DOI: `10.3847/2041-8213/ac88c0`
- Secrest+22 accepted CatWISE AGN catalog (Zenodo record), DOI: `10.5281/zenodo.6784602`
- CatWISE2020 (Marocco et al. 2021, ApJS 253, 8), DOI: `10.3847/1538-4365/abd805`
- unWISE coadds:
  - Lang 2014, AJ 147, 108, DOI: `10.1088/0004-6256/147/5/108`
  - Meisner et al. 2017, AJ 153, 38, DOI: `10.3847/1538-3881/153/1/38`
- WISE mission (optional context):
  - Wright et al. 2010, AJ 140, 1868, DOI: `10.1088/0004-6256/140/6/1868`

## Environment setup

Recommended (Conda, easiest for `healpy`):

```bash
conda env create -f environment.yml
conda activate quasars-systematics
```

Alternative (pip/venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data requirements

### 1) Secrest+22 CatWISE AGN catalog (required)

Download the Secrest+22 accepted bundle from Zenodo:

- DOI: `10.5281/zenodo.6784602`

Place and extract it under `data/external/zenodo_6784602/` so the expected catalog path exists:

```text
data/external/zenodo_6784602/
  secrest+22_accepted.tgz
  secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits
  secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits
  secrest_extracted/secrest+22_accepted/wise/reference/nvss_crossmatch.fits   (if present in the release)
```

Notes:
- The tarball is ~2.3 GiB.
- `exclude_master_revised.fits` is used for the official Secrest exclusion mask in some workflows.

### 2) unWISE depth proxy (optional; "independent depth" robustness)

For the GLM+CV robustness test we use an *independent*, imaging-derived depth statistic based on the
unWISE W1 exposure-count maps (`w1-n-m`).

To keep this repository lightweight, we **include a derived per-tile statistic**:

- `data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json`

This file is sufficient to reproduce the `Nexp`-offset GLM results shown in `Q_D_RES/`.

If you want to regenerate this JSON from the full unWISE coadds, use:

```bash
python3 scripts/build_unwise_nexp_tile_stats.py --help
```

Warning: regenerating from raw `w1-n-m` maps requires downloading a very large volume of FITS files
(multi-TB scale depending on the tile set and caching).

## Reproducing the headline results

All commands below assume you have the Secrest+22 catalog extracted at:

`data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits`

### A) Baseline dipole reproduction (sanity check)

```bash
python3 scripts/reproduce_secrest_dipole.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --outdir outputs/secrest_reproduction \
  --b-cut 30 --w1cov-min 80 --w1-max 16.4 --bootstrap 200
```

### B) Figure 1: faint-limit scaling diagnostic (main ApJL figure)

This produces `fixed_axis_scaling_fit.png` and `fixed_axis_scaling_fit.json`.

```bash
python3 experiments/quasar_dipole_hypothesis/fit_intrinsic_plus_selection_fixed_axis.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --axis-from secrest \
  --secrest-json Q_D_RES/secrest_reproduction_dipole.json \
  --w1cov-min 80 --b-cut 30 \
  --w1max-grid 15.6,16.6,0.05 \
  --alpha-dm 0.05 \
  --make-plots \
  --outdir outputs/fixed_axis_scaling_fit
```

### C) Additional mechanism diagnostics (optional)

This produces `dipole_vs_w1max.png`, `dipole_vs_w1covmin.png`, and hemisphere count-match plots.

```bash
python3 experiments/quasar_dipole_hypothesis/magshift_mechanism_diagnostics.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --secrest-json Q_D_RES/secrest_reproduction_dipole.json \
  --w1-max 16.4 --w1-max-for-cdf 17.0 \
  --make-plots \
  --outdir outputs/magshift_mechanisms
```

### D) GLM+CV robustness: template-cleaned residual dipole with independent depth (optional)

This is the main robustness point: the inferred "cleaned" direction depends strongly on which
depth proxy is used. To run with the included unWISE per-tile depth stats:

```bash
python3 experiments/quasar_dipole_hypothesis/vector_convergence_glm_cv.py \
  --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --exclude-mask-fits data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits \
  --nvss-crossmatch data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/nvss_crossmatch.fits \
  --template-set ecliptic_harmonics \
  --nside 64 --kfold 5 --seed 123 \
  --nexp-tile-stats-json data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json \
  --outdir outputs/glmcv_nexp_offset \
  --make-plots
```

## ApJL draft + figures

The ApJL draft is in `Q_D_RES/Resolution.md` and references the PNGs by filename.
If you copy the figures in `Q_D_RES/` into your Overleaf project root, the TeX block in that file
should compile without modification.
