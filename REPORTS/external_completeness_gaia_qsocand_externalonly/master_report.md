# External completeness model (all-sky; Gaia DR3 QSO candidates, external-only)

This bundle contains the **recommended** all-sky, externally trained spatial completeness proxy built from:
- Gaia DR3 QSO candidates (CDS `I/356`, file `qsocand.dat`)
- unWISE `logNexp` depth map (map-level)
- ecliptic geometry (`|β|`, `sinλ`, `cosλ`)

It intentionally **does not** use CatWISE-derived binned covariates inside the external model to avoid
re-introducing catalogue-dependent structure.

Artifacts:
- `data/meta.json`
- `data/p_map_nside64.fits`
- `data/logp_offset_map_nside64.fits`
- `figures/calibration_curve.png`
- `figures/logp_offset_mollweide.png`

Reproduce:
```bash
python3 scripts/run_gaia_dr3_qsocand_completeness_model.py \\
  --feature-set external_only \\
  --pqso-min 0.8 \\
  --make-plots \\
  --export-report-dir REPORTS/external_completeness_gaia_qsocand_externalonly
```

