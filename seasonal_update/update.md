# Seasonal / ecliptic-longitude update (how it relates to the paper)

This folder documents two **related but distinct** “seasonal imprint” diagnostics:

1) a **Secrest-style residual systematics test** (pipeline-congruent validation),
2) a **Poisson-GLM ecliptic-longitude proxy test** (more aggressive, mechanism-oriented).

Your current PRD draft (the one you pasted) **does not explicitly include** either test yet; it mainly focuses on
ecliptic-*latitude* correction, GLM axis drift with `W1_max`, depth-template sensitivity, LSS covariance, and
known-truth completeness injections.

The goal of this update is to give you *paper-ready* material to add (typically as an Appendix or a short “additional systematics proxy” subsection),
without derailing the main story.

---

## Test 1 (pipeline-congruent): Secrest-style residual systematics, including ecliptic longitude

**What it is:**  
Using Secrest-style masking + ecliptic-lat trend correction + dipole subtraction, test whether the **residual**
density map shows statistically significant trends versus standard proxy maps, quantified via binned reduced χ²/ν.
This is the kind of diagnostic Secrest et al. use to argue “systematics are under control.”

**Implementation:**  
This is exactly what the Secrest-accepted validation suite does:
- Script: `scripts/run_secrest_systematics_audit.py`
- Paper-ready bundle: `2-3-EEE/master_report.md`

**Paper-ready figures (copied into this folder):**
- `systematics_grid_full_w1max16p4.png`
- `systematics_grid_no_nvss_w1max16p5.png`

**Headline results (numbers are in `2-3-EEE/data/*.json`):**

Secrest-style dipole reproductions (weighted-counts estimator):
- Baseline (WISE-only; `W1_max=16.4`): `D≈0.01610`, `(ℓ,b)≈(238.77°, +28.35°)`
- NVSS-removed + homogenized (`W1_max=16.5`): `D≈0.01531`, `(ℓ,b)≈(239.45°, +30.10°)`

Residual trend vs **ecliptic longitude** proxy (`elon_deg`), reduced χ²/ν:
- Baseline: χ²/ν(`elon_deg`) ≈ **1.61**
- No-NVSS: χ²/ν(`elon_deg`) ≈ **1.64**

**Interpretation:**  
Even under Secrest-style correction and dipole subtraction, there is a **mild but non-negligible residual dependence on ecliptic longitude**.
This is consistent with some remaining scan/season-related structure in ecliptic coordinates.
It does *not* by itself quantify “how much dipole bias remains,” but it supports the idea that ecliptic-*latitude* correction alone does not remove all
ecliptic-structured systematics.

**How to add to the PRD paper:**  
Recommended: short Appendix paragraph + one figure (baseline grid), with a sentence noting the no-NVSS variant is similar.
This strengthens referee confidence that you are using “their” style diagnostics.

---

## Test 2 (mechanism-oriented): fast ecliptic-longitude proxy (GLM + sinλ/cosλ templates + longitude wedges)

**What it is (plain language):**  
WISE observing conditions vary with time/season. Those time-dependent effects can project onto the sky as **fixed spatial patterns in ecliptic coordinates**.
A fast proxy is therefore to check whether the recovered dipole:
- varies strongly across **ecliptic longitude wedges**, and/or
- is absorbed/rotated by adding simple low-order longitude templates (`sinλ`, `cosλ`) in a Poisson GLM.

This is a **proxy** (it does not prove time dependence), but strong sensitivity is a red flag for scan/season structure.

**Implementation:**  
- Script: `scripts/run_ecliptic_lon_proxy.py`
- Paper-ready bundle: `dipole_direction_report/master_report.md`
- Figure (copied into this folder): `ecllon_proxy.png`

**Exact cut and model (as run):**
- Data/cuts: CatWISE accepted + Secrest exclusion mask, `W1cov≥80`, `|b|>30°`, **fixed Secrest footprint**, `W1≤16.6`, `nside=64`
- Estimator: Poisson GLM on HEALPix counts
  - baseline nuisance: `abs(ecliptic latitude)` (z-scored)
  - optional: add `sin(λ)` and `cos(λ)` (z-scored), where `λ` is barycentric mean ecliptic longitude (pixel-center)

**Headline results (`W1_max=16.6`):**

Full-sky fits:
- Baseline (`abs_elat` only):
  - `D_hat ≈ 0.01678`
  - angle to CMB dipole axis ≈ `34.33°`
  - `(ℓ,b) ≈ (236.58°, +21.81°)`
- Baseline + (`sinλ`, `cosλ`) longitude templates:
  - `D_hat ≈ 0.01289`
  - angle to CMB dipole axis ≈ `79.77°`
  - `(ℓ,b) ≈ (283.74°, −50.54°)`
  - amplitude reduction ≈ **23.2%** relative to the baseline full-sky fit

Longitude-wedge fits (4 wedges; `abs_elat` only):
- λ∈[0°,90°]:   `D_hat ≈ 0.0542`, angle-to-CMB ≈ `68.8°`
- λ∈[90°,180°]: `D_hat ≈ 0.0684`, angle-to-CMB ≈ `34.4°`
- λ∈[180°,270°]:`D_hat ≈ 0.0577`, angle-to-CMB ≈ `3.7°`
- λ∈[270°,360°]:`D_hat ≈ 0.0655`, angle-to-CMB ≈ `56.4°`

**Important caveat:**  
The wedge/partial-sky amplitudes are **inflated** and are not directly comparable to the full-sky `D_hat`.
The key diagnostic is the **large directional variation** across wedges and the strong sensitivity of the full-sky solution to adding low-order longitude terms.

**Interpretation:**  
This is consistent with the dipole direction (and some of its amplitude) being influenced by **ecliptic-structured scan/season systematics**.
It does *not* “disprove” a kinematic component — but it shows that **simple seasonal/scan proxies can absorb ~O(20%) of the amplitude and strongly rotate the best-fit direction** at `W1_max=16.6`.

**How to add to the PRD paper:**  
Recommended: Appendix-only (or one-paragraph “additional proxy” subsection) to avoid scope creep.
Suggested take-away sentence:
> “A fast ecliptic-longitude proxy test shows strong dependence of the inferred dipole direction on ecliptic longitude and strong sensitivity to low-order `sinλ/cosλ` templates, consistent with residual scan/season structure beyond the dominant `|β|` trend.”

---

## Suggested minimal patch to the PRD draft (structure)

If you want to include both tests without expanding the main narrative:

- Add a short **Appendix: Ecliptic-longitude (seasonal) proxy diagnostics** with:
  - Fig. A1: `ecllon_proxy.png`
  - 1 paragraph describing the GLM test + the 23% amplitude reduction + direction sensitivity.

- Add a short **Appendix: Secrest-style residual systematics validation** with:
  - Fig. A2: `systematics_grid_full_w1max16p4.png`
  - 1 sentence noting the no-NVSS variant is similar; optionally point to the second grid.

This keeps the main text focused while pre-empting “did you check ecliptic longitude / seasonal effects?” questions.

