# CatWISE quasar dipole systematics (reproducibility bundle)

This repository supports the compact MNRAS Letter in `mnras_letter/` and the accompanying reproducibility
bundle archived on Zenodo (`10.5281/zenodo.18719277`). It contains code, small vendored reviewer-seed
artifacts, and auditable report outputs under `REPORTS/`.

A larger PRD-style figure pack workflow is also supported; see `docs/OVERLEAF_PRD.md`. (Short note-style
manuscripts also live under `radio_a/`.)

## Quick summary (what matters)

| Question / claim | Key quantitative summary | Where to verify | One-command reproduce |
|---|---:|---|---|
| Magnitude-limit stability diagnostic (CatWISE) | `D ≈ 1.6e-2` while axis drifts to `~34°` by `W1_max=16.6` | `mnras_letter/main.tex`, `REPORTS/Q_D_RES_2_2/master_report.md` | `make reproduce` |
| Time-domain (epoch) diagnostic (unWISE; CatWISE parent) | Dipole-only epoch amplitudes span `D≈0.067–0.118`, but a constrained rich bootstrap gives `P(range_sim≥range_obs)=0.974` and `P(χ²_sim≥χ²_obs)=0.983` | `REPORTS/unwise_time_domain_catwise_epoch_systematics_suite/`, `REPORTS/unwise_time_domain_catwise_parametric_bootstrap_constrained_constantD_rich_20260220_231118UTC/` | `./.venv/bin/python scripts/run_unwise_time_domain_catwise_epoch_systematics_suite.py` |
| “Case-closed” robustness package (maximal nuisance + calibration) | At `W1_max=16.6`: baseline `D=0.01678` → maximal nuis `D=0.00578` → orthog `D=0.01776`; CMB-fixed `D_par=-0.00815`; constrained-null bootstrap `p_abs=P(|D_par,sim|≥|D_par,obs|)=0.707` for `D_true=0.0046` | `REPORTS/case_closed_maximal_nuisance_suite/` (notably `data/scan_rep_consistency.json`, `data/bootstrap_dpar.json`) | `./.venv/bin/python scripts/run_case_closed_maximal_nuisance_suite.py --bootstrap-nsim 1000` |

### Why this work should not be dismissed

- **Controlled falsifiers:** time-domain stability and faint-limit axis drift are diagnostics that a kinematic/physical interpretation must be stable under, absent selection/estimation biases.
- **Calibrated inference:** key variability statements are calibrated with constrained parametric bootstraps and injection-through-measured-systematics tests, not just analytic χ².
- **Multiple estimators + templates:** results are checked with regression, Poisson GLMs, and cross-check estimators, reducing “it’s just the method” failure modes.
- **Held-out validation:** where applicable, nuisance fields are trained on one sky subset and evaluated on another (wedge GroupKFold), to reduce overfit arguments.
- **Internal self-audits:** the maximal-nuisance suite includes a scan-vs-single-cut consistency check (`scan_rep_consistency.json`) so optimizer artefacts cannot silently change headline numbers.
- **Auditable artifacts:** each suite writes a self-contained report directory with JSON summaries + plots + exact command lines for independent verification.

## Quickstart

### 1) Verify headline values (reviewer seed; no external downloads)

This lightweight command reproduces the headline numbers quoted in `mnras_letter/main.tex` using vendored
artifacts under `artifacts/`. It **does not** require the Secrest+22 tarball or other large external downloads.

```bash
make reproduce
```

This writes a timestamped folder under `outputs/` containing `report.md` and `summary.json`.

### 2) Run the case-closed suite (maximal nuisance + calibration)

```bash
./.venv/bin/python scripts/run_case_closed_maximal_nuisance_suite.py \
  --out-report-dir REPORTS/case_closed_maximal_nuisance_suite_$(date -u +%Y%m%d_%H%M%SUTC) \
  --bootstrap-nsim 1000
```

### 3) Run the epoch-slicing suite (unWISE time domain; CatWISE parent)

```bash
./.venv/bin/python scripts/run_unwise_time_domain_catwise_epoch_systematics_suite.py
```

## Scope and limitations (read first)

This repo is designed to support the following (core, referee-facing) claims:

- The CatWISE “accepted” number-count dipole **amplitude** is non-zero at the few×10⁻² level.
- The inferred dipole **axis/direction is not stable** under plausible magnitude-limit and depth/completeness modeling choices.
- Selection/completeness systematics can generate smooth magnitude-limit scans and bias both amplitude and direction in injection tests.
- unWISE epoch slicing provides a controlled time-domain diagnostic showing that dipole-only recovery is systematics-dominated.

This repo does **not** currently claim or provide:

- A fully closed, image-level injection/recovery **end-to-end completeness pipeline** that “solves” the all-sky amplitude.
- A validated, all-sky **W1-conditioned** completeness model (the all-sky external proxy uses Gaia qsocand, which has no W1).

## How to cite (copy/paste)

If you use this repository, please cite:

- The MNRAS Letter in `mnras_letter/` (source: `mnras_letter/main.tex`).
- The Zenodo reproducibility archive: `10.5281/zenodo.18719277`.
- Supplementary assets archive (where applicable): `10.5281/zenodo.18489200`.

## Data requirements (minimal vs full)

- Minimal (reviewer seed): none (uses vendored artifacts under `artifacts/`).
- Full end-to-end reruns: require the Secrest+22 accepted CatWISE AGN bundle (`10.5281/zenodo.6784602`) and
  additional optional external data (details: `docs/ADVANCED.md`).

## Key report bundles (source of truth)

These directories are the “source of truth” for the numbers quoted in the paper and table above:

- `REPORTS/Q_D_RES_2_2/` (magnitude-limit scan + core diagnostics)
- `REPORTS/unwise_time_domain_catwise_epoch_systematics_suite/` (epoch slicing + nuisance controls)
- `REPORTS/unwise_time_domain_catwise_parametric_bootstrap_constrained_constantD_rich_20260220_231118UTC/` (epoch constrained null calibration)
- `REPORTS/case_closed_maximal_nuisance_suite/` (maximal nuisance + CMB-fixed + held-out + bootstrap)

For additional bundles and “how to run everything”, see `docs/ADVANCED.md`.

## References / DOIs used by this repository

- Quasar-dipole reproducibility archive (Zenodo), DOI: `10.5281/zenodo.18719277`
- Supplementary assets archive (Zenodo), DOI: `10.5281/zenodo.18489200`
- Legacy repository archive (Zenodo), DOI: `10.5281/zenodo.18476711`
- Secrest et al. 2022, ApJL 937 L31, DOI: `10.3847/2041-8213/ac88c0`
- Secrest+22 accepted CatWISE AGN catalog (Zenodo record), DOI: `10.5281/zenodo.6784602`
- CatWISE2020 (Marocco et al. 2021, ApJS 253, 8), DOI: `10.3847/1538-4365/abd805`
- unWISE Time-Domain Catalog (IRSA580), DOI: `10.26131/IRSA580`

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

## Advanced usage

See `docs/ADVANCED.md`.

