# End-to-end completeness validation plan (CatWISE quasar dipole)

This document is a **work plan** for validating a *completeness/selection model* end-to-end, at the level needed to support (or refute) a kinematic/cosmological interpretation of a catalog dipole.

It is written to be actionable in this repo (`/home/primary/QS`) and tied to existing scripts where possible.

---

## 0) What “validated completeness” means here

For this project, a “validated completeness model” means:

1) You can take **independent survey metadata** (exposure/depth/coverage maps, masks, scan-geometry proxies, etc.) and build a model that predicts the **selection/completeness modulation** relevant to the flux limit.
2) When you include that model in the dipole inference (as an offset or explicit covariate set), the inferred dipole vector becomes **stable** under:
   - changes in the faint cut (`W1_max`) and **differential** magnitude bins,
   - reasonable modeling variations (template swaps, map smoothing choices),
   - sky splits/jackknife (statistical),
   - time/seasonal proxies (systematic).
3) The full pipeline passes **known-truth injection/recovery tests** where completeness is *deliberately misspecified* vs *correctly specified* (to quantify bias and show the model prevents it).

This is stronger than “the amplitude is significant.” It’s the level needed to say “this is (mostly) kinematic” rather than “this is a catalog/systematics dipole.”

---

## 1) Key principle: independence and non-circularity

To avoid circularity:

- **Footprint mask** must be defined from *coverage/exclusion products* (or exposure maps), not from “pixels that happened to contain sources after a cut.”
- “Depth proxies” used for validation should be **map-level**, derived from survey imaging metadata, not catalog counts (catalog proxies can *inherit* the selection imprint you’re trying to model).

This repo already contains a map-level unWISE exposure proxy (`data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits`) and a known-truth injection framework (`scripts/run_catwise_lognormal_mocks.py`).

---

## 2) Deliverables (what we want at the end)

### Paper-grade
- A figure/table showing `D`, and **CMB-parallel** / **CMB-perpendicular** components (`D_∥`, `D_⊥`) vs `W1_max` **before** and **after** completeness modeling.
- A figure showing **axis/direction stability** under:
  - magnitude-bin choice (cumulative + differential),
  - template swaps among *independent* depth maps,
  - jackknife (statistical).

### Repo-grade
- A reproducible “completeness model build” script that produces:
  - `completeness_map.fits` (or a small set of maps),
  - a `meta.json` capturing inputs, smoothing, normalization, and model form.
- An injection/recovery report bundle (like `REPORTS/Q_D_RES_2_2/`) containing:
  - mock configs,
  - recovered dipole distributions,
  - bias metrics.

---

## 3) Build the completeness model (map-level)

### 3.1 Covariate stack (independent maps)

Start with a minimal covariate stack that is plausibly related to depth/completeness:

- unWISE exposure proxy: `logNexp` (already in repo)
- CatWISE coverage map proxy (`W1cov`, `W2cov`) **only if** it can be used as an *independent* metadata map (be careful: catalog-derived summary maps can still encode selection).
- dust proxy: `E(B−V)` (external map)
- scan geometry: `|β|` (abs ecliptic latitude), plus optional low-order seasonal proxies:
  - `sin(λ)`, `cos(λ)` where `λ` is ecliptic longitude (fast “seasonal imprint” proxy)
- optional: Moon contamination / backgrounds / bright-star density proxies (if available as maps)

Standardize each template (z-score on unmasked pixels) and store the template dipole vectors (ℓ=1) as a diagnostic: if a template has a strong intrinsic dipole, it can directly trade off with the dipole fit.

### 3.2 Choose a completeness parameterization (start simple)

You need a model that maps covariates → selection modulation near the faint cut.

Recommended starting point (because it matches the mechanism you already demonstrated):

**Effective limiting-magnitude modulation model**
> “In direction `n`, the effective faint limit is shifted by `δm(n)`.”

Implementation options:

- **Linear template model:** `δm(n) = Σ_k a_k t_k(n)`  
  (good for diagnosing degeneracies; easy to fit)
- **Low-rank + harmonic model:** add a small spherical-harmonic component in ecliptic coordinates to capture scan-geometry modes not explained by depth maps.

Once `δm(n)` is defined, the induced counts modulation at fixed `W1_max` is controlled by the observed slope `d ln N(<m) / dm` (your injection argument). That’s exactly why smooth scans don’t rule it out.

### 3.3 How to “fit” the completeness model without circularity

There is no single perfect method; the plan is to use multiple, consistent constraints:

**A) External-truth cross-match (best, if feasible)**
- Cross-match CatWISE quasars to an *independent* deeper/cleaner quasar sample (e.g., SDSS/eBOSS/DR16Q/WISE+optical selections) and measure completeness as a function of sky position and `W1` near the cut.
- Fit `δm(n)` (or `m_lim(n)`) to reproduce the spatial variation in completeness.

**B) Internal “turnover” calibration (use with caution)**
- In sky regions binned by depth (e.g., quantiles of `logNexp`), compare the faint-end rollover of `dN/dm` and infer an effective `m_lim` shift.
- This uses the catalog counts, so it’s not fully independent, but it can be used as a *consistency* check when combined with (A) and injection tests.

**C) Imaging-level injection/recovery (gold standard, hardest)**
- Inject synthetic sources into unWISE coadds (or use an existing injection dataset) and run the detection/measurement pipeline to measure completeness directly.
- This is the most defensible end-to-end approach, but it is expensive and requires tooling outside the current catalog-only workflow.

---

## 4) Put completeness into the dipole inference (offset vs covariate)

Key identifiability rule:

- If you think the completeness modulation is **known** (or tightly constrained), it should enter as an **offset** (fixed term).
- If it is **uncertain**, it should enter as **templates** with nuisance parameters that you marginalize over.

In practice:

1) Start with GLM fits where the depth/completeness model enters as **covariates** to diagnose degeneracies.
2) Once the completeness model is externally calibrated, transition the calibrated piece into an **offset** and keep only residual flexibility (e.g., small template coefficients) as nuisance parameters.

Deliverable: report the dipole–template coefficient covariance (or correlations) so the “degeneracy” claim is quantitative.

---

## 5) End-to-end validation suite (what to run)

### 5.1 Known-truth injection/recovery (already supported here)

Use `scripts/run_catwise_lognormal_mocks.py` to run ensembles where truth is known:

- Inject a true dipole vector (e.g. along CMB) at a controlled amplitude
- Inject a depth-linked selection systematic using an independent depth map
- Run two analysis modes:
  - **misspecified** (no depth template / wrong map)
  - **modeled** (correct depth map as covariate and/or calibrated offset)

Acceptance criteria:
- Modeled runs recover (within errors) the injected dipole amplitude and direction.
- Misspecified runs show quantified bias (so you can say what goes wrong when completeness is wrong).

This is the minimal “referee-proof” demonstration that (i) the estimator works, and (ii) depth misspecification can produce the failure modes seen in real data.

### 5.2 Magnitude-bin stability (cumulative vs differential)

Re-run the real-data scan in **differential bins** (already present in the repo outputs) and verify:
- the completeness-modeled dipole direction is stable across bins, especially in the faintest bin where completeness sensitivity is highest.

Acceptance criteria:
- Reduced `D_⊥CMB` growth with `W1_max`.
- Reduced bin-to-bin direction drift (compared to Poisson+JK scatter).

### 5.3 Time/seasonal proxy nulls (fast)

Even if you can’t get full time-resolved exposure metadata, you can run fast proxies:
- include `sinλ, cosλ` templates (ecliptic longitude) and see how much of the dipole is absorbed
- do longitude-wedge fits as a diagnostic (direction dependence)

Acceptance criteria:
- If these proxies absorb a large fraction of `D_⊥`, that argues the drift is scan-geometry/seasonal, not cosmological.
- If they do not, seasonal imprint is less likely to be dominant (but not excluded).

### 5.4 LSS covariance (so completeness isn’t blamed on statistics)

Use clustered mocks (lognormal) to get a dipole-vector covariance that includes LSS+shot noise:
- verify that claimed stability improvements are not within expected statistical scatter.

Acceptance criteria:
- Any “before vs after completeness modeling” change in `D_⊥` should be large compared to the LSS+shot-noise uncertainty.

---

## 6) “Ready to claim kinematic?” decision checklist

You can *start* making a kinematic case only if all are true:

1) With the calibrated completeness model, `D_⊥CMB` is consistent with ~0 within realistic covariance, **across** the `W1_max` scan and in differential bins.
2) The fitted dipole **direction** is stable under:
   - jackknife,
   - template swaps among independent depth maps,
   - modest changes in smoothing/resolution of the completeness map.
3) Injection/recovery with the same model class shows unbiased recovery at amplitudes relevant to the data.
4) Residual correlations with scan-geometry proxies (|β|, sinλ/cosλ) are small or explicitly modeled.

If any fail, the correct paper-level statement remains:
> “The catalog dipole amplitude is robustly non-zero, but the direction is model-dependent and plausibly dominated by selection/completeness systematics near the faint cut.”

---

## 7) Concrete next implementation steps in this repo

1) Add a “completeness model builder” script:
   - Input: `lognexp_healpix_nside64.fits` (+ optional maps)
   - Output: `outputs/completeness_model_<tag>/completeness_map.fits` + `meta.json`
2) Add a “before/after” scan postprocess:
   - Plot `D`, `D_∥CMB`, `D_⊥CMB` vs `W1_max` for:
     - baseline (`abs_elat` only)
     - modeled completeness (depth-map covariate / calibrated offset)
3) Expand known-truth injections:
   - vary injected dipole amplitude (0.002, 0.005, 0.010)
   - vary depth systematic strength (α grid)
   - show bias curves (axis bias and amplitude bias vs α)

---

## 8) Notes on scope and realism

- The **gold standard** is imaging-level injection/recovery. If that is not feasible, the next best is an external-truth cross-match plus the mock-based known-truth validation already implemented here.
- A completeness model that is “good enough” for amplitude significance is often **not** good enough for **direction** (because direction is what gets rotated by unmodeled dipole-like systematics).
