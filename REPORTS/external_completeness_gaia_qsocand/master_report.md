# External completeness model (all-sky; Gaia DR3 QSO candidates) — legacy bundle

This report documents an **all-sky, externally trained, map-level completeness proxy** for the Secrest+22
CatWISE “accepted” sample using **Gaia DR3 QSO candidates** (`I/356`, `qsocand.dat`).

Unlike the SDSS DR16Q model (which is magnitude-dependent in W1 but footprint-limited), this Gaia model is
**all-sky** but **does not condition on W1** (Gaia has no W1).

## Status / recommendation

This folder is kept for provenance from an earlier iteration of the Gaia model.

Use the **external-only** bundle instead:
- `REPORTS/external_completeness_gaia_qsocand_externalonly/master_report.md`

That version intentionally avoids CatWISE-derived binned covariates inside the external model.

## What is modeled?

For Gaia qsocand objects (with `PQSO >= 0.8`) inside the Secrest footprint, define:

- `y=1` if the Gaia QSO candidate has a CatWISE “accepted” match within 2 arcsec  
- `y=0` otherwise

We fit a logistic model for `P(y=1 | x)` using global predictors:
- `logNexp(pix)` from the unWISE exposure-count map (map-level depth proxy)
- ecliptic geometry terms from RA/Dec: `|β|`, `sinλ`, `cosλ`

The fitted model is exported as a **HEALPix map** of:
- `p_map_nside64.fits`: predicted `P(in CatWISE accepted)` at each pixel center
- `logp_offset_map_nside64.fits`: `log(p)` centered on its seen-pixel median (usable as a GLM offset)

## Artifacts

Data:
- `REPORTS/external_completeness_gaia_qsocand/data/meta.json`
- `REPORTS/external_completeness_gaia_qsocand/data/p_map_nside64.fits`
- `REPORTS/external_completeness_gaia_qsocand/data/logp_offset_map_nside64.fits`

Figures:
- `REPORTS/external_completeness_gaia_qsocand/figures/calibration_curve.png`
- `REPORTS/external_completeness_gaia_qsocand/figures/logp_offset_mollweide.png`

## How to reproduce

1) Download Gaia qsocand (CDS I/356) to:
- `data/external/gaia_dr3_extragal/qsocand.dat.gz`

2) Run:
```bash
python3 scripts/run_gaia_dr3_qsocand_completeness_model.py \\
  --qsocand-gz data/external/gaia_dr3_extragal/qsocand.dat.gz \\
  --pqso-min 0.8 \\
  --make-plots \\
  --export-report-dir REPORTS/external_completeness_gaia_qsocand
```
