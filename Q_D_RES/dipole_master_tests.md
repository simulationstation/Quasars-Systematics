# Quasar Dipole Master Tests (Q\_D)

Date: 2026-02-01  
Project root: `/home/primary/PROJECT`

This document is a detailed, reproducible record of the CatWISE/Secrest quasar dipole tests we ran,
and of the *mechanism checks* we performed to see whether a tiny direction-dependent magnitude bias,
combined with the hard faint cut used in the selection, can (partly) generate or remove the observed
number-count dipole.

It is **not** (yet) a demonstration that the horizon-entropy reconstruction predicts the quasar
anomaly. It is a professional-level “does this even make sense?” and “what would it take?” audit.

---

## 0) What Question Are We Testing?

We want to test a hypothesis of the form:

> The observed CatWISE quasar number-count dipole could be (partly) explained by a direction-dependent
> photometric/selection effect that is physically connected to our broader “horizon / anisotropy”
> program (or to some other real sky-dependent mechanism).

Because the CatWISE/Secrest catalog is dominated by number-count selection and has limited per-object
redshift information, the **most direct** mechanism test we can do right now is:

1. Keep the standard selection (including the hard faint cut `W1 <= 16.4`).
2. Apply a small dipole-like modulation in magnitude, aligned to a chosen axis:
   - `W1_corr = W1 - sign * delta_m * cos(theta_axis)`
3. Re-apply the hard faint cut, re-measure the dipole.
4. Ask whether a *plausibly small* `delta_m` (millimagnitudes to ~0.01 mag) could materially change the dipole.

If the answer is “yes”, then:
- the anomaly could plausibly be driven by subtle systematics, **or**
- if we can derive `delta_m` from a physical model (instead of fitting it), that would give a path
  to a *predictive* explanation.

---

## 1) Inputs (Data + Code)

### 1.1 External “mature” dipole code

We used the mature repo you provided:

- Repo path: `external/quasar-dipole-fun`

This repo provides:
- catalog-level dipole estimators,
- a “Secrest reproduction” workflow,
- slicing utilities and saved intermediate products.

Sanity check (repo’s own test):
- Ran: `external/quasar-dipole-fun/run_mock_test.py` (PASS earlier in this session)

### 1.2 CatWISE/Secrest data (Zenodo)

We downloaded the Secrest+22 accepted bundle:

- Download artifact: `data/external/zenodo_6784602/secrest+22_accepted.tgz` (~2.3 GiB)
- Extracted into: `data/external/zenodo_6784602/secrest_extracted/`

Key FITS used:
- CatWISE AGN catalog:  
  `data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits`

The catalog contains (relevant columns):
- `l, b` (Galactic coords)
- `ra, dec` (ICRS coords)
- `w1` (magnitude-like quantity used for the faint cut)
- `w1cov` (coverage)
- `ebv` (dust proxy)
- `elon, elat` (ecliptic coords, a proxy for scan strategy)

Note: The extracted tree also contains `exclude_master_revised.fits`, but the baseline reproduction
in `quasar-dipole-fun` uses the baseline cuts only unless that mask is explicitly provided.

---

## 2) The “Compass Check”: Axis Alignment Between Experiments

Before trying to “explain” Secrest, we compared directions:

- CMB dipole: (l,b) = (264.021, 48.253)
- Secrest/CatWISE dipole (as reproduced by `quasar-dipole-fun`): (236.01, 28.77)
- Our SN horizon-anisotropy best hemisphere axis (from our scan summary): (168.75, 41.81)

Computed comparisons (from `outputs/quasar_dipole_hypothesis_20260201_013039UTC/axis_alignment_summary.json`):

- Secrest vs CMB: angle ~ **29.0 deg** (axis-alignment chance ~0.126)
- Secrest vs SN best axis: angle ~ **55.0 deg** (axis-alignment chance ~0.427)

Interpretation:
- Secrest is not “mysteriously aligned” with our current SN anisotropy best axis; the separation is
  large (~55 deg) and not rare geometrically.

Figure:
- `axis_alignment_mollweide.png`

---

## 3) Step 1 (Baseline): Reproduce the Catalog Dipole from the Real FITS

We ran the external reproduction script against the extracted FITS:

Command (exact):
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python external/quasar-dipole-fun/reproduce_secrest_dipole.py \
    --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
    --outdir outputs/secrest_reproduction_local_20260201_0300UTC \
    --bootstrap 200 \
    --seed 42
```

Important implementation note:
- The reproduction script hit a small `angle_deg()` shape bug when running locally (it was mixing
  `(1,3)` and `(3,)` vectors for a dot-product). We fixed that in:
  - `external/quasar-dipole-fun/secrest_utils.py`
  - Only affects *printing/comparison*; the dipole estimate itself is unchanged.

Baseline cuts (as used by the external pipeline by default):
- `w1cov >= 80`
- `|b| > 30 deg`
- `w1 <= 16.4`

Baseline results:
- Final sample size: **N = 1,401,166**
- Vector-sum dipole amplitude: **D = 0.02099**
- Dipole direction: **(l,b) = (236.01, 28.77)**
- Bootstrap (200 resamples): amplitude std ~ **0.00142**, direction sigma ~ **6.33 deg**

Outputs:
- `outputs/secrest_reproduction_local_20260201_0300UTC/dipole.json`
- `outputs/secrest_reproduction_local_20260201_0300UTC/cuts_used.json`
- `outputs/secrest_reproduction_local_20260201_0300UTC/coverage.json`

Published Secrest numbers vs this pipeline:
- Secrest (paper): D ~ 0.0154 +/- 0.0015 at (238.2, 28.8).
- This pipeline’s simple vector estimator yields D ~ 0.021.
- We treated the pipeline’s D as the *internal baseline* for mechanism tests, and we explicitly
  avoid claiming “we reproduced the paper estimator exactly.”

---

## 4) Step 1.5 (Order-of-Magnitude): “How Big a Mag Dipole Would Be Needed?”

We computed a first-order response:

> If counts are truncated at a hard faint cut, then a small sky-dependent shift in magnitude
> can cause a dipole in number counts:
> `deltaN/N ~ (d ln N / d m_max) * delta_m`

Using the external repo’s quick W1 slicing bins and a 200k-source subset, we estimated:

- Faint-edge slope (two edge bins): `alpha_edge = d ln N / d m ~= 1.2609 per mag`
- For an observed D ~ 0.02099, the implied magnitude modulation amplitude is:
  - `delta_m_amp ~ 0.0166 mag` (peak-to-trough ~0.0333 mag)

This is just a “back of envelope” plausibility check.

Outputs:
- `outputs/quasar_dipole_magshift_20260201_015936UTC/required_magshift_summary.json`
- `w1_density_log.png`

Figure:
- `w1_density_log.png`

---

## 5) Step 2 (Templates): Does Dust/Ecliptic/Coverage Explain It?

We implemented and ran a template-marginalized count-level regression on a HEALPix map:

Model (per pixel):
- `N_pix = A + B·n_hat + c1*EBV + c2*|elat| + c3*w1cov`
- WLS weights: `w ~ 1/N_pix` (Poisson counts)
- Report dipole as `D = |B|/A`

Command:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python experiments/quasar_dipole_hypothesis/fit_dipole_with_templates.py \
    --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
    --outdir outputs/quasar_template_fit_20260201_0316UTC \
    --nside 64 \
    --make-plots
```

Results:
- Raw catalog dipole (vector-sum): **0.02099** at (236.01, 28.77)
- Dipole-only fit (count-level WLS): **0.02035 +/- 0.00266**
- Dipole + templates: **0.01929 +/- 0.00267** (small reduction, ~5%)
- “De-templated” reweighted dipole: **0.02030**

Interpretation:
- These basic templates improve the fit but **do not explain away** the dipole.

Figure:
- `template_fit_maps.png`

---

## 6) Step 3 (Fit delta_m): What delta_m Minimizes the Dipole?

We then fit (by scanning) a dipole-like magnitude modulation aligned to a chosen axis.

Axis chosen here: the Secrest/CatWISE dipole axis (the one measured from the catalog itself).

The correction:
- `W1_corr = W1 - sign * delta_m * cos(theta_axis)`
- Then select: `W1_corr <= 16.4`, re-measure the dipole.

Command:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python experiments/quasar_dipole_hypothesis/fit_magshift_amp.py \
    --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
    --axis-from secrest \
    --delta-m-max 0.05 \
    --grid-n 101 \
    --refine \
    --make-plots \
    --outdir outputs/quasar_magshift_fit_secrest_20260201_0327UTC
```

Results:
- Baseline (no correction): **D = 0.02099**
- Best correction (by dipole amplitude):
  - sign = **-**
  - `delta_m ~= 0.0113 mag`
  - dipole drops to **D ~= 0.0100**
  - but the dipole direction rotates to roughly **(l,b) ~= (235.82, -33.03)**
  - importantly, the *projection onto the Secrest axis* becomes small (~0.0047)

Figure:
- `magshift_fit.png`

Interpretation:
- A dipolar magnitude modulation at the **~0.01 mag** level can reduce the observed dipole by about half
  via the hard faint cut (a selection-effect mechanism).

---

## 7) Step 3 (delta_m + templates): Does delta_m Survive Marginalization?

This is the critical “next sensible step” you requested:

> Fit delta_m while also marginalizing EBV / |elat| / w1cov in the count-level regression.

Implementation:
- For each delta_m:
  - selection is applied as a position-dependent faint threshold:
    - `W1 <= 16.4 + sign * delta_m * cos(theta_axis)`
  - build a HEALPix counts map for the selected sample
  - fit `N_pix = A + B·n + templates`
  - report residual dipole `D = |B|/A` after templates

Command:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python experiments/quasar_dipole_hypothesis/fit_magshift_with_templates.py \
    --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
    --axis-from secrest \
    --nside 64 \
    --delta-m-max 0.03 \
    --grid-n 61 \
    --refine \
    --make-plots \
    --outdir outputs/quasar_magshift_with_templates_secrest_20260201_0340UTC
```

Results (key):
- Baseline (delta_m=0): residual dipole after templates
  - **D_fit ~= 0.01925 +/- 0.00267**
- Best (min residual dipole after templates):
  - sign = **-**
  - `delta_m ~= 0.0115 mag`
  - residual dipole after templates:
    - **D_fit ~= 0.00569 +/- 0.00237**
  - projection onto Secrest axis is essentially zero (~|proj| < 1e-3)

Figure:
- `magshift_with_templates.png`

Interpretation:
- The required `delta_m` (~0.01 mag) does **not** disappear when you marginalize the basic
  “obvious” sky templates. The mechanism remains plausible and becomes *more effective*
  (residual D after templates gets very small).

---

## 8) Summary of What We Learned

## 8.0 New: Concrete Mechanism Diagnostics (Count-Match + Cut Sensitivity)

You asked specifically for a “mechanism link” (not just “delta_m can be fit to remove the dipole”).
We therefore ran an additional diagnostic suite that does *not* fit delta_m from the dipole vector
sum directly.

Command:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python experiments/quasar_dipole_hypothesis/magshift_mechanism_diagnostics.py \
    --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
    --outdir outputs/quasar_magshift_mechanisms_secrest_20260201_044750UTC \
    --b-cut 30 \
    --w1cov-min 80 \
    --w1-max 16.4 \
    --w1-max-for-cdf 17.0 \
    --make-plots
```

Key outputs:
- `outputs/quasar_magshift_mechanisms_secrest_20260201_044750UTC/mechanism_diagnostics.json`
- `outputs/quasar_magshift_mechanisms_secrest_20260201_044750UTC/hemisphere_w1_cdf.png`
- `outputs/quasar_magshift_mechanisms_secrest_20260201_044750UTC/hemisphere_w1_ratio.png`
- `outputs/quasar_magshift_mechanisms_secrest_20260201_044750UTC/dipole_vs_w1max.png`
- `outputs/quasar_magshift_mechanisms_secrest_20260201_044750UTC/dipole_vs_w1covmin.png`
- `outputs/quasar_magshift_mechanisms_secrest_20260201_044750UTC/w1cov_hemisphere_hist.png`

### 8.0.1 Count-match “delta_m” (no dipole fitting)

We split the sky into two hemispheres about the *Secrest axis*:
- Fore: `cos(theta_axis) > 0`
- Aft: `cos(theta_axis) < 0`

Then we asked:
> How much would you have to shift the faint cut `W1_max` in one hemisphere so that its cumulative
> counts match the other hemisphere’s cumulative counts at the same nominal `W1_max`?

Result at `W1_max = 16.4`:
- Fore has **708,202** objects (at the cut), aft has **692,964**.
- To match aft’s count, fore’s cut would need to be **brighter** by ~`0.0130 mag`.
- To match fore’s count, aft’s cut would need to be **fainter** by ~`0.0140 mag`.

So the empirical hemispheric “cut shift” is of order:
- `|delta_m| ~ 0.0135 mag` (peak-to-trough would be ~0.027 mag, depending on sign convention)

Interpretation:
- This is a *direct* demonstration that hemispheric count differences about the Secrest axis can be
  produced by an O(10 millimag) shift in the effective magnitude threshold.
- This magnitude scale is consistent with:
  - the earlier “required delta_m from slope” back-of-envelope,
  - and the fitted `delta_m ~ 0.0115 mag` that minimized the dipole after basic templates.

### 8.0.2 Sensitivity to W1_max (faint cut)

We measured the catalog dipole as a function of the faint cut:
- `W1_max` swept from 16.0 to 16.8.

Observed behavior:
- D decreases from ~0.024 at `W1_max=16.0` down to ~0.020 at `W1_max=16.4`.
- Past ~16.6, the sample stops changing (the catalog effectively has no fainter objects under the
  other fixed cuts), and D saturates.

Interpretation:
- Strong dependence on the selection cut location is what you expect for a selection/photometry
  mechanism interacting with a steep faint-end slope.
- It is not what you’d expect for a clean, intrinsic cosmic dipole that is stable under small
  changes to the limiting magnitude.

### 8.0.3 Sensitivity to w1cov_min (coverage / scan-depth)

We measured the catalog dipole as a function of the minimum coverage threshold:
- `w1cov_min` swept from 50 to 200 (keeping `W1_max=16.4` fixed).

Observed behavior:
- For `w1cov_min <= 80`, nothing changes (because the baseline sample already enforces `w1cov>=80`).
- As `w1cov_min` increases beyond ~110, the measured dipole becomes **enormous** (D~0.04 → 0.4) while
  the sample size collapses (you are selecting a very non-uniform sky footprint).

Interpretation:
- This is a concrete “survey mechanism” indicator: coverage is highly anisotropic and selection on
  it can create very large apparent dipoles (purely by changing the sky mask/footprint).
- It strongly suggests that any paper-grade analysis must model scan-depth / completeness far more
  explicitly than a single `w1cov>=80` threshold.

### 8.0.4 Important sanity check: is delta_m a *real* W1 zero-point dipole?

If the fitted `delta_m ~ 0.01 mag` were literally a sky-dependent W1 photometric zero-point shift,
we would expect to see a mean W1 offset between hemispheres in fixed magnitude bins.

We checked mean W1 in several bins and found hemisphere mean differences of only:
- O(1e-4) mag (0.0001 mag) across bins

Interpretation:
- There is **no direct evidence** in the catalog that W1 itself is shifted by ~0.01 mag between the
  hemispheres.
- Therefore, the fitted `delta_m` is best interpreted as an **effective selection/completeness
  modulation** (a way to parameterize how the hard cut responds to a sky-dependent missingness /
  completeness mechanism), not as a literal measured W1 zero-point dipole.

### 8.1 What is “good news”

- The Secrest/CatWISE number-count dipole can be **strongly reduced** by a very small
  direction-dependent magnitude modulation (~0.01 mag) interacting with the hard faint cut.
- This remains true even after marginalizing basic systematics proxies (EBV, |elat|, w1cov)
  in a Poisson WLS count regression.

### 8.2 What this does *not* prove

- This does not prove a horizon-entropy / modified gravity origin for the quasar dipole.
- In these scans, `delta_m` is being *fit* to cancel the dipole (mechanism check), not predicted.
- The best axis used here is the Secrest axis itself; we have not shown the same effect along
  our SN anisotropy axis (and in fact, earlier quick checks suggested it does not help).

### 8.3 One-line interpretation

> A selection + tiny photometric bias explanation is mechanically plausible at the ~0.01 mag level,
> even after removing the most obvious large-scale templates.

---

## 8.4 A “Mixture Model” Step: Separate an alpha-scaling (selection) dipole from a non-scaling dipole

To push beyond “fit delta\_m at one cut”, we implemented a two-component scaling model across many
choices of the faint cut `W1_max`.

Core idea:
- In a magnitude-limited sample, selection/missingness that acts like a *magnitude-cut shift*
  produces a number-count dipole that scales with the faint-edge slope
  `alpha_edge(W1_max) = d ln N / d m_max` (units: 1/mag).
- Other survey systematics (mask/coverage geometry, residual dust, etc.) do **not** necessarily
  scale with this slope.

So we fit (vector form) across multiple `W1_max` values:

`d_obs(W1_max) ≈ d0 + alpha_edge(W1_max) * delta_m_amp * n_axis`

where:
- `d_obs` is the measured 3-vector dipole (vector-sum estimator)
- `n_axis` is fixed to the Secrest dipole direction
- `delta_m_amp` is an effective magnitude-cut dipole amplitude (mag)
- `d0` is the remaining (non-alpha-scaling) dipole vector

Implementation:
- Script: `experiments/quasar_dipole_hypothesis/fit_intrinsic_plus_selection_fixed_axis.py`
- Output: `outputs/quasar_intrinsic_plus_selection_fixedaxis_20260201_051136UTC/`

Command:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python experiments/quasar_dipole_hypothesis/fit_intrinsic_plus_selection_fixed_axis.py \
    --catalog data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits \
    --outdir outputs/quasar_intrinsic_plus_selection_fixedaxis_20260201_051136UTC \
    --axis-from secrest \
    --b-cut 30 \
    --w1cov-min 80 \
    --w1max-grid 15.6,16.6,0.05 \
    --alpha-dm 0.05 \
    --min-N 200000 \
    --make-plots
```

Key result:
- `delta_m_amp = -0.01249 ± 0.00271` mag

Interpretation:
- This is consistent with the single-cut cancellation fit (`delta_m ~ -0.0115 mag`), but it is
  estimated in a way that is explicitly tied to the magnitude-cut response physics and uses many
  cuts (not a single cut).
- The remaining `d0` term should be interpreted as “everything not captured by this alpha-scaling
  selection model”, not necessarily a physical/cosmic dipole.

Artifacts copied into `Q_D/` for convenience:
- `Q_D/fixed_axis_scaling_fit.json`
- `Q_D/fixed_axis_scaling_fit.png`

---

## 8.5 The “Vector Convergence” Test (Poisson GLM + aggressive scan/dust/coverage templates)

Gemini’s idea was:

> The raw Secrest vector is likely biased by survey footprint / scan strategy. If we do a more
> aggressive systematics clean (coverage + dust + scan templates) and the cleaned dipole direction
> rotates toward the SN anisotropy axis, that would support a “shared signal” hypothesis.

We implemented a cross-validated Poisson-GLM version:

1) Build HEALPix pixel counts from the Secrest/CatWISE catalog.
2) Fit a **Poisson GLM** (log link) on *training* pixels including:
   - an explicit dipole term (n\_x, n\_y, n\_z), and
   - a pre-chosen set of systematics templates (dust + scan/coverage proxies).
3) Repeat across K folds and report the fitted GLM dipole direction per fold and in aggregate.

Implementation:
- Script: `experiments/quasar_dipole_hypothesis/vector_convergence_glm_cv.py`
- Primary run output: `outputs/quasar_glmcv_dipole_NVSS_20260201_054735UTC/`
- Copied into `Q_D/` for convenience:
  - `Q_D/glm_cv_summary_glm_dipole_NVSS.json`
  - `Q_D/glm_cv_glm_dipole_axes.png`
  - `Q_D/glm_cv_angles_to_sn.png`

Configuration (primary run):
- NVSS removal: enabled (Secrest-provided crossmatch join by source\_id)
- Templates: `template_set = ecliptic_harmonics`:
  - EBV (dust proxy)
  - w1cov (coverage proxy)
  - |elat| and low-order ecliptic-longitude harmonics (sin/cos of elon and 2*elon)
- Pixelization: nside=64
- CV: 6 folds

### 8.5.1 Result: cleaned dipole direction moves toward the SN axis (under this model)

Aggregate (vector-mean across folds) from `outputs/quasar_glmcv_dipole_NVSS_20260201_054735UTC/glm_cv_summary.json`:
- Cleaned GLM dipole axis: **(l,b) ≈ (148.28°, 45.48°)**
- Axis-angle to SN best axis (168.75°, 41.81°): **≈ 15.2°**
- Axis-angle to Secrest raw dipole axis (236.01°, 28.77°): **≈ 68.4°**

Fold-to-fold SN-axis angles (same run):
- min ≈ 1.7°
- max ≈ 28.8°

Interpretation:
- Under this aggressive scan/dust/coverage template model, the fitted “cleaned” dipole direction is
  substantially closer to the SN anisotropy axis than the raw Secrest direction is.

### 8.5.2 Template-set dependence (critical caveat)

This convergence is not automatic:
- With `template_set=basic` (EBV + w1cov + |elat| only), the fitted dipole direction stays stable but
  does **not** move toward the SN axis.
- Adding explicit ecliptic-longitude harmonics is what moves the fitted dipole direction into the SN-axis neighborhood.

This is both:
- good news (scan strategy matters exactly as expected), and
- a warning label (template degeneracies can rotate the inferred dipole direction if the model is misspecified).

### 8.5.3 What this is not yet

This is still not “publication-grade convergence” because:
- our “coverage map” is still proxied by catalog `w1cov` aggregated to pixels (instrument-derived but sampled at source positions),
- we have not propagated full directional uncertainties beyond fold scatter,
- and we have not repeated this across a grid of `W1_max` values and masks.

The next professional upgrade is to ingest an **independent** full-sky WISE/unWISE depth-of-coverage map (Nexp)
as an offset/template and repeat the same GLM+CV workflow.

We did this upgrade in Section 8.6. The headline is that the apparent “convergence toward the SN axis”
is **not** robust to swapping `w1cov` for an independent unWISE depth template.

---

## 8.6 Independent depth-of-coverage (unWISE Nexp) + Secrest exclude mask

This section implements the exact requested “hardening” of the vector-convergence test:
- Replace the catalog-derived `w1cov` proxy with an **independent** depth-of-coverage statistic derived from
  unWISE W1 exposure-count maps (`w1-n-m`).
- Apply the official Secrest circular exclusion mask `exclude_master_revised.fits`.

### 8.6.1 Data product used (independent depth)

We use the unWISE `neo7` full-depth coadds, and for each unWISE tile we download:

- `unwise-<coadd_id>-w1-n-m.fits.gz`

This is a per-pixel exposure-count map for W1. For each tile, we compute a robust scalar depth proxy:

- `Nexp_tile = median(nexp_pixels > 0)`  (median of positive pixels)

We cache all tile files under:

- `data/cache/unwise_nexp/neo7/`

and store the per-tile summary map:

- `data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json`

These artifacts are **not** tracked in git (they live under `data/`, which is ignored).

Implementation script (resumable, parallel):

- `scripts/build_unwise_nexp_tile_stats.py`

### 8.6.2 How Nexp enters the GLM

We do **not** force a linear relationship between quasars and depth. Instead of treating `Nexp` as a free template
coefficient, we use it as a **Poisson offset**:

- `log(mu_p) = log(Nexp_tile(p)) + X_p beta`

So the expected quasar count in pixel `p` scales with depth through the fixed `+log(Nexp)` term, while the remaining
systematics enter through `X` (dust, scan strategy, etc).

We then:
- fit the GLM on training pixels,
- compute fractional residuals on held-out pixels `y = N/mu - 1`,
- and fit a dipole to `y` (WLS, weights ~ mu).

### 8.6.3 Result (no-dipole GLM; measure residual dipole on held-out pixels)

Run:
- Output: `outputs/quasar_glmcv_nexp_offset_NVSS_exmask_nodipole_fix_20260201_065131UTC/`
- Templates: EBV + |elat| + ecliptic harmonics (sin/cos elon, sin/cos 2*elon, elat)
- Masking: |b|>30 deg, `w1cov>=80`, Secrest `exclude_master_revised.fits` applied, NVSS matches removed.

Aggregate held-out residual dipole:
- `(l,b) = (54.85, +5.60)` deg
- `|b_vec| = 0.00552`  (fractional units)
- Axis angle to Secrest raw: `34.40` deg
- Axis angle to SN axis: `76.38` deg

Interpretation:
- After switching to an independent depth template, the cleaned residual dipole direction does **not** move into the
  SN-axis neighborhood.
- The residual amplitude is small (~0.6% level).

### 8.6.4 Result (GLM includes a dipole term; residual dipole becomes tiny as expected)

Run:
- Output: `outputs/quasar_glmcv_nexp_offset_NVSS_exmask_glmdipole_fix_20260201_065154UTC/`

When the GLM itself contains a dipole term, the held-out residual dipole is (by design) very small:
- residual `|b_vec| ~ 1.1e-4`

The fitted GLM dipole coefficient direction (train-only) is:
- `(l,b) = (93.21, +8.81)` deg
- Axis angle to SN axis: `73.38` deg

So the “dipole-in-the-GLM” direction also does **not** align with the SN axis under the Nexp offset.

### 8.6.5 Takeaway for the “vector convergence” hypothesis

This is a major robustness checkpoint:
- The earlier “convergence toward the SN axis” was conditional on a particular coverage proxy (`w1cov`) and template set.
- Replacing depth with a genuinely independent imaging-derived statistic (unWISE `Nexp`) changes the inferred direction
  materially and removes the apparent SN-axis convergence.

This does **not** automatically mean “no shared physics”; it means the directional inference is **template dominated**
at the current level of modeling, and the *only defensible claim* is:

- the quasar dipole direction is highly sensitive to scan/depth systematics modeling, and
- any alignment claim must be shown to survive swaps of independent depth templates.

---

## 9) Figures in This Folder

All figures are stored in `Q_D/`:

- `Q_D/axis_alignment_mollweide.png`  
  Secrest vs CMB vs our SN anisotropy axis (geometric “compass check”).

- `Q_D/w1_density_log.png`  
  Faint-end W1 density used to estimate the first-order count response at the faint cut.

- `Q_D/template_fit_maps.png`  
  HEALPix binned delta map and residual after subtracting EBV/|elat|/w1cov templates.

- `Q_D/magshift_fit.png`  
  Dipole amplitude vs delta_m for the simple “apply correction and re-cut” scan.

- `Q_D/magshift_with_templates.png`  
  Dipole amplitude vs delta_m when fitting/marginalizing templates in the count model.

- `Q_D/dipole_amplitude_summary.png`  
  Compact visual summary of how D changes through the successive tests.

- `Q_D/hemisphere_w1_cdf.png`  
  Fore vs aft W1 CDFs about the Secrest axis, and the implied count-match delta_m at W1_max.

- `Q_D/hemisphere_w1_ratio.png`  
  Fore/Aft W1 histogram ratio about the Secrest axis (shows where in magnitude the asymmetry lives).

- `Q_D/dipole_vs_w1max.png`  
  Sensitivity of the measured dipole amplitude to the faint cut W1_max.

- `Q_D/dipole_vs_w1covmin.png`  
  Sensitivity of the measured dipole amplitude to the coverage threshold w1cov_min.

- `Q_D/w1cov_hemisphere_hist.png`  
  w1cov distribution by hemisphere about the Secrest axis (a simple coverage/selection proxy).

- `Q_D/fixed_axis_scaling_fit.png`  
  Two-component scaling separation across many W1\_max cuts: non-scaling dipole + alpha\_edge-scaled
  magnitude-cut dipole (axis fixed to Secrest).

- `Q_D/glm_cv_glm_dipole_axes.png`  
  Fold-by-fold GLM dipole directions (train-only) for the vector-convergence test, with SN and Secrest axes marked.

- `Q_D/glm_cv_angles_to_sn.png`  
  Fold-by-fold axis angle to the SN axis in the GLM vector-convergence test.

- `Q_D/glm_cv_axes_nexp_offset.png`  
  Same as `glm_cv_axes.png` but for the independent unWISE Nexp offset run (Section 8.6).

- `Q_D/glm_cv_angles_to_sn_nexp_offset.png`  
  Same as `glm_cv_angles_to_sn.png` but for the independent unWISE Nexp offset run (Section 8.6).

- `Q_D/glm_cv_glm_dipole_axes_nexp_offset.png`  
  Same as `glm_cv_glm_dipole_axes.png` but for the independent unWISE Nexp offset run (Section 8.6).

---

## 10) Exact Output Artifacts (for reproducibility)

- Axis alignment summary:  
  `outputs/quasar_dipole_hypothesis_20260201_013039UTC/axis_alignment_summary.json`

- Baseline reproduction:  
  `outputs/secrest_reproduction_local_20260201_0300UTC/dipole.json`

- Template fit:  
  `outputs/quasar_template_fit_20260201_0316UTC/template_fit_summary.json`

- Fit delta_m (no templates):  
  `outputs/quasar_magshift_fit_secrest_20260201_0327UTC/magshift_fit.json`

- Fit delta_m + templates:  
  `outputs/quasar_magshift_with_templates_secrest_20260201_0340UTC/magshift_with_templates.json`

---

## 11) Recommended Next Steps (not executed here)

To turn this into a paper-grade statement, the next logical steps are:

1. Expand the template set beyond EBV/|elat|/w1cov (e.g., include low-frequency foreground proxies,
   bright-star masks, spatial completeness models, and scan-depth maps if available).
2. Fit the dipole with a mixture model:
   - “true dipole” + “missingness/selection” (Gemini’s `L = L_cat + L_missing` suggestion).
3. Try the same workflow on alternative samples/cuts (e.g., different W1 max values) and show the
   fitted `delta_m` scales as expected from the faint-end slope.
4. Only then attempt to connect to the horizon program:
   - derive a predicted sky-dependent magnitude modulation from the horizon model (if possible),
     and compare *predicted* axis/amplitude to the data without fitting.
