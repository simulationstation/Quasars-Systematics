# Case-closed maximal-nuisance dipole suite

Date: 2026-02-21 (UTC)

Goal: a single, auditable bundle testing whether the **percent-level CatWISE dipole** can be
explained by **survey systematics** under a maximal reasonable nuisance basis, with held-out
validation and external (cross-catalog) completeness templates.

This suite is designed to be *hard to hand-wave away*: it includes a ridge-regularized maximal
nuisance basis, an explicit CMB-fixed dipole fit (physical mode), a held-out sky check, and a
leave-one-template-out ranking for attribution.

## Headline (representative cut)

- Representative cut: `W1_max = 16.60`
- Kinematic reference (repo convention): `D_kin ≈ 0.0046` toward CMB `(l,b)=(264.021,48.253)`
- Baseline (free axis): `D=0.01678`, axis sep to CMB `=34.33°`
- Maximal nuis (free axis): `D=0.00578`, axis sep to CMB `=73.43°`
- Maximal nuis (templates ⟂ {1,nx,ny,nz}): `D=0.01776`, axis sep to CMB `=34.81°`
- CMB-fixed (maximal nuis; priors): `D_par=-0.00815` (signed)
- Held-out CMB-fixed test (2-fold wedge split): `D_par,test=-0.00250`, `|D|=0.00250`
- LOTO attribution (top Δdeviance drivers):
  - `Y2_0_re_z`: `Δdev=23.07`
  - `Y2_1_re_z`: `Δdev=16.06`
  - `log1p_starcount_z`: `Δdev=13.31`
  - `sin2_elon_z`: `Δdev=11.58`
- Bootstrap calibration (constrained D_true=0.0046): `p(D_par,sim ≥ D_par,obs)=0.744`

## Outputs

- Summary JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/summary.json`
- Scan suite JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/scan_suite.json`
- Held-out JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/heldout_validation.json`
- LOTO JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/loto_attribution.json`
- LOTO CSV: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/loto_attribution.csv`
- Coefficients CSV (rep cut): `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/coefficients_representative_cut.csv`
- Bootstrap JSON: `/home/primary/QS/REPORTS/case_closed_maximal_nuisance_suite/data/bootstrap_dpar.json`
- Key figure: `REPORTS/case_closed_maximal_nuisance_suite/figures/case_closed_4panel.png`
- CMB-fixed scan: `REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_fixed_amplitude_scan.png`
- D_par/D_perp scan: `REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_projection_par_perp_scan.png`
- Bootstrap hist: `REPORTS/case_closed_maximal_nuisance_suite/figures/bootstrap_dpar_hist.png`

![](REPORTS/case_closed_maximal_nuisance_suite/figures/case_closed_4panel.png)

![](REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_fixed_amplitude_scan.png)

![](REPORTS/case_closed_maximal_nuisance_suite/figures/cmb_projection_par_perp_scan.png)

![](REPORTS/case_closed_maximal_nuisance_suite/figures/bootstrap_dpar_hist.png)

## Reproduce

Run from repo root:

```bash
./.venv/bin/python scripts/run_case_closed_maximal_nuisance_suite.py
```

