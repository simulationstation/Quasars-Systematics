# End-to-end selection correction attempt (externally trained; all-sky)

This report bundles the current “best-effort” **externally trained, map-level selection/completeness correction**
we can apply to the CatWISE dipole analysis *without using catalog count fluctuations to define the correction*.

It is intended as the closest thing (in this repository, with publicly available data products) to an
“end-to-end corrected” dipole analysis.

## Summary (plain language)

We use **Gaia DR3 QSO candidates** (all-sky, independent of WISE) to learn where CatWISE “accepted” is more/less
complete on the sky, and we feed that learned map into the **Poisson GLM** dipole scan as an externally motivated
nuisance template.

Result: this **substantially reduces the CMB-perpendicular (“direction drift”) component** at faint cuts, but it
**does not eliminate the dipole amplitude** (the measured amplitude remains \~1–2%).

So: this is strong additional evidence that *directional* conclusions are selection-model dependent, but it is **not**
a complete “amplitude solved” closure.

---

## 1) External completeness model used (Gaia; external-only predictors)

Model bundle:
- `REPORTS/external_completeness_gaia_qsocand_externalonly/`

Key data products copied into this report:
- `data/gaia_extonly_logp_offset_map_nside64.fits`
- `data/gaia_extonly_meta.json`

This Gaia model is trained on:
- Gaia DR3 QSO candidates (`PQSO>=0.8`)
- label `y=1` if matched to CatWISE accepted within 2 arcsec, else `0`

Predictors (external-only):
- unWISE `logNexp(pix)` depth proxy (map-level)
- ecliptic geometry terms from RA/Dec: `|β|`, `sinλ`, `cosλ`

Cross-validation (sky-holdout by 15° ecliptic-longitude wedges) yields AUC \~0.56 (weak but nonzero predictive
power; see `data/gaia_extonly_meta.json`).

---

## 2) Dipole scan integration (Poisson GLM)

### 2.1 Baseline scan (for comparison)
- `data/baseline_rvmp_fig5_poisson_glm.json`
- `figures/baseline_rvmp_fig5_poisson_glm.png`

### 2.2 “Externally informed” scan (Gaia completeness template)
We include the Gaia-derived map as a **free nuisance template** (z-scored) in the GLM scan:
- `data/gaia_extonly_cov_rvmp_fig5_poisson_glm.json`
- `figures/gaia_extonly_cov_rvmp_fig5_poisson_glm.png`

### 2.3 CMB-parallel/perpendicular comparison
- `figures/cmb_projection_compare_baseline_vs_gaia_extonly.png`
- `data/cmb_projection_compare_baseline_vs_gaia_extonly.csv`

Representative numbers (axis-angle to CMB; sign-invariant):
- At `W1_max=16.4`: baseline `Δθ_axis≈28.00°` → Gaia-template `Δθ_axis≈3.86°`
- At `W1_max=16.6`: baseline `Δθ_axis≈34.33°` → Gaia-template `Δθ_axis≈13.16°`

Corresponding CMB-perpendicular component at `W1_max=16.6`:
- baseline `D_perp≈0.00946`
- Gaia-template `D_perp≈0.00381`

Amplitude at `W1_max=16.6`:
- baseline `D≈0.01678`
- Gaia-template `D≈0.01674`

Interpretation:
- The Gaia-derived external template can absorb a large fraction of the **non-CMB component** that drives the
  magnitude-limit direction drift.
- The **overall amplitude** remains at the percent level, indicating that either (i) the remaining amplitude is not
  captured by this completeness proxy, or (ii) there is a real residual dipole + LSS contribution, or (iii) other
  unmodeled selection effects remain.

---

## 3) Reproducibility pointers

Gaia completeness model:
- `scripts/run_gaia_dr3_qsocand_completeness_model.py` (use `--feature-set external_only`)

Dipole scans:
- `scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py`

The CMB projection summary/plot:
- `scripts/analyze_cmb_projection_from_glm_scan.py`

