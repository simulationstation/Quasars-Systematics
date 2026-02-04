# External completeness model (map-level, SDSS DR16Q–validated)

This report documents an **externally validated, map-level completeness/selection model** for the Secrest+22 “accepted” CatWISE AGN candidate sample, built in `/home/primary/QS`.

The goal is to predict (and then model) sky-dependent selection near the faint WISE limit **using external information**, rather than relying on CatWISE number-count fluctuations themselves.

---

## 1) Inputs (external + independent maps)

### 1.1 CatWISE “accepted” catalog (target to be “complete in”)
- `data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits`
- This includes columns: `ra`, `dec`, `l`, `b`, `w1`, `w1cov`, `ebv`, and several coordinate proxies.

### 1.2 Secrest exclusion discs (mask geometry)
- `data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits`

### 1.3 Map-level depth proxy (independent imaging metadata)
- `data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits`
- This is a **HEALPix map** (Galactic coords) derived from unWISE exposure-count metadata (not catalog counts).

### 1.4 External truth set (SDSS spectroscopic quasars)
- SDSS DR16Q FITS from SDSS SAS:
  - `data/external/sdss_dr16q/DR16Q_v4.fits` (downloaded; not git-tracked)
  - checksum validated against `data/external/sdss_dr16q/eboss_qso_DR16Q.sha1sum`

Note: I checked `/home/primary/PROJECT` for an existing DR16Q FITS before downloading; it was not present.

---

## 2) Footprint + label definition (no circularity)

### 2.1 Footprint mask (Secrest-style)
We restrict the truth set to the same analysis footprint used in the dipole work:
- parent coverage: `W1cov >= 80`
- Secrest “mask_zeros” on the parent map (zeros + neighbour pixels)
- exclusion discs from `exclude_master_revised.fits` (where `use==True`)
- Galactic plane cut on pixel centers: `|b| > 30°`

### 2.2 Label: “in CatWISE accepted”
For each truth quasar, define:
- `y = 1` if there exists a CatWISE accepted source within **2 arcsec** (nearest-neighbour in ICRS, matched via a 3D unit-vector KD-tree)
- `y = 0` otherwise

This gives a directly interpretable empirical quantity:
> “Given an externally confirmed quasar at this sky position and W1 magnitude, what is the probability it appears in the CatWISE accepted catalog?”

---

## 3) Model (externally trained + sky-holdout validated)

### 3.1 Feature set used (depth-only; avoids SDSS-footprint confounding)
We use a **depth-only** predictor stack:
- `W1` (from DR16Q WISE-associated magnitude columns)
- `logNexp(pix)` from the unWISE exposure map
- interaction: `W1 * logNexp`

This avoids using coordinate-geometry templates (ecliptic latitude/longitude) inside the *external* model, which can otherwise soak up SDSS footprint selection and extrapolate badly outside SDSS coverage.

### 3.2 Estimator
Logistic regression:
\[
P(y=1\mid\mathbf{x}) = \mathrm{sigmoid}(\beta_0 + \beta\cdot \mathbf{x})
\]

### 3.3 Cross-validation
We validate with a sky-holdout scheme:
- `GroupKFold(n_splits=5)` where groups are **15° ecliptic-longitude wedges** (computed from RA/Dec)
- This reduces leakage of large-scale scan/geometry structure between train and test.

### 3.4 CV performance (full dataset)
From `REPORTS/external_completeness_sdss_dr16q/data/sdss_dr16q_depthonly_meta.json`:
- `truth_n_used = 671,134`
- `truth_positive_rate = 0.348`
- `ROC AUC ≈ 0.933`
- `Brier ≈ 0.1056`

Calibration visualization:
- `REPORTS/external_completeness_sdss_dr16q/figures/calibration_curve.png`

---

## 4) Map products (what the model exports)

### 4.1 “m50” and “δm” maps
From the fitted logistic model we construct a per-pixel “midpoint magnitude”:
- `m50(pix)`: the W1 where the model predicts `P(in CatWISE)=0.5` at that pixel’s `logNexp`.
- `δm(pix) = m50(pix) - median(m50)` (so it is centered).

Files:
- `REPORTS/external_completeness_sdss_dr16q/data/m50_map_nside64.fits`
- `REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits`
- visualization: `REPORTS/external_completeness_sdss_dr16q/figures/delta_m_mollweide.png`

Empirical scale (nside=64, seen pixels):
- `std(δm) ≈ 0.0109 mag`
- dipole amplitude of `δm` (linear a·n fit) is ≈ **0.0043 mag** (few mmag), i.e. a plausible depth-linked threshold modulation scale.

### 4.2 Edge-completeness map at a reference faint cut
We also export a map of predicted “in CatWISE” probability at a representative faint cut:
- `P(in CatWISE | W1=16.6, pix)`

Files:
- `REPORTS/external_completeness_sdss_dr16q/data/p_edge_map_w116p6.fits`
- `REPORTS/external_completeness_sdss_dr16q/figures/p_edge_mollweide.png`

---

## 5) Integration test: does this reduce CatWISE dipole axis drift?

This is a *diagnostic integration*, not a claim of “final correction.”

### 5.1 Baseline GLM scan (for comparison)
Command:
```bash
python scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
  --depth-mode none \
  --make-plot \
  --outdir outputs/rvmp_fig5_poisson_glm_baseline_sdsscomp
```

### 5.2 With SDSS-validated δm map as a nuisance covariate
We include the map as a **template** (free coefficient), scaled by the global faint-edge slope `alpha_edge(W1_max)` per cut:
```bash
python scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
  --depth-mode delta_m_map_covariate_alpha_edge \
  --depth-map-fits REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits \
  --depth-map-name sdss_dr16q_depthonly_delta_m \
  --make-plot \
  --outdir outputs/rvmp_fig5_poisson_glm_sdss_depthonly_cov
```

We then compare CMB-parallel/perpendicular components:
```bash
python scripts/analyze_cmb_projection_from_glm_scan.py \
  --inputs \
    outputs/rvmp_fig5_poisson_glm_baseline_sdsscomp/rvmp_fig5_poisson_glm.json \
    outputs/rvmp_fig5_poisson_glm_sdss_depthonly_cov/rvmp_fig5_poisson_glm.json \
  --labels baseline sdss_depthonly_cov \
  --outdir outputs/cmb_projection_sdss_depthonly_compare
```

Figure:
- `REPORTS/external_completeness_sdss_dr16q/figures/cmb_projection_compare_baseline_vs_sdss_depthonly.png`

Summary table:
- `REPORTS/external_completeness_sdss_dr16q/data/cmb_projection_compare_baseline_vs_sdss_depthonly.csv`

### 5.3 What this implies
This depth-only external completeness model:
- **does not eliminate** the magnitude-limit–driven axis drift by itself,
- and only modestly changes the inferred CMB-perpendicular component at the faint end in this quick integration.

Interpretation: whatever drives the large axis drift in the CatWISE dipole is **not captured by unWISE exposure-count depth alone**, consistent with the broader paper narrative that scan-geometry/completeness systematics can dominate direction inference near the faint threshold.

---

## 6) Limitations (important for “legitimacy”)

1) **SDSS footprint limitation:** DR16Q is not all-sky. The model is externally validated on the DR16Q footprint; applying it over the full CatWISE footprint is an extrapolation based on the assumed relation between completeness and the depth proxy.
2) **Truth-set W1 systematics:** DR16Q W1 magnitudes are WISE-associated and may share some WISE photometric systematics. This is still far less circular than using CatWISE number counts to infer completeness, but it is not as “gold standard” as image-level injection–recovery.
3) **Missing covariates:** a more complete model should add additional independent maps (e.g., dust/extinction, stellar density/confusion, background proxies, artifact/bright-star masks).

---

## 7) Next legit upgrade

To make this closer to “closure-grade”:
- add an all-sky external truth set (e.g., Gaia quasars) as a complementary validator (even if shallower),
- add independent map predictors beyond exposure counts,
- validate “before vs after” stability in **differential W1 bins** (not only cumulative),
- quantify dipole–template degeneracy (covariances) for the new completeness templates the same way we do for depth maps in the paper.

