# Case-closed maximal-nuisance dipole suite

Date: 2026-02-21 (UTC)

Goal: a single, auditable bundle testing whether the **percent-level CatWISE dipole** can be
explained by **survey systematics** under a maximal reasonable nuisance basis, with held-out
validation and external (cross-catalog) completeness offsets.

This suite is designed to be *hard to hand-wave away*: it includes a ridge-regularized maximal
nuisance basis, an explicit CMB-fixed dipole fit (physical mode), a held-out sky check, and a
leave-one-template-out ranking for attribution.

## Headline (representative cut)

- Representative cut: `W1_max = 16.60`
- Kinematic reference (repo convention): `D_kin ≈ 0.0046` toward CMB `(l,b)=(264.021,48.253)`
- Baseline (free axis): `D=0.01678`, axis sep to CMB `=34.34°`
- Maximal nuis (free axis): `D=0.00917`, axis sep to CMB `=83.28°`
- Maximal nuis (templates ⟂ {1,nx,ny,nz}): `D=0.01768`, axis sep to CMB `=34.79°`
- CMB-fixed (templates ⟂ {1,u·n}): `D_par=0.01096` (signed)
- Held-out CMB-fixed test (2-fold wedge split): `D_par,test=0.01160`, `|D|=0.01160`
- LOTO attribution (top Δdeviance drivers):
  - `cos_elon_z`: `Δdev=4.61`
  - `log1p_starcount_z`: `Δdev=3.20`
  - `sin_elon_z`: `Δdev=3.01`
  - `log_invvar_z`: `Δdev=2.23`

## Outputs

- Summary JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/summary.json`
- Scan suite JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/scan_suite.json`
- Held-out JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/heldout_validation.json`
- LOTO JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/loto_attribution.json`
- LOTO CSV: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/loto_attribution.csv`
- Coefficients CSV (rep cut): `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/coefficients_representative_cut.csv`
- Key figure: `REPORTS/case_closed_maximal_nuisance_suite/figures/case_closed_4panel.png`
- CMB-fixed scan: `REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_fixed_amplitude_scan.png`

![](REPORTS/case_closed_maximal_nuisance_suite/figures/case_closed_4panel.png)

![](REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_fixed_amplitude_scan.png)

## Reproduce

Run from repo root:

```bash
./.venv/bin/python scripts/run_case_closed_maximal_nuisance_suite.py
```

