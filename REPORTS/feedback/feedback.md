# Feedback on PRD draft (CatWISE dipole audit)

Overall: the narrative is coherent and reads like a careful systematics audit (stable catalog dipole amplitude, but drifting/template-dependent axis; LSS covariance does not erase amplitude significance; known-truth injections show misspecification can bias both amplitude+axis).

## High-value edits to strengthen referee robustness

### 1) Make “fixed footprint” explicitly non–data-dependent
In Sec. *Data and masking*, add one sentence clarifying that the valid-pixel footprint is defined from **coverage/exclusion products** (and the parent `W1_cov≥80` coverage requirement), **not** from “pixels with nonzero counts after applying `W1≤W1_max`”.
This preempts the common objection that the footprint could be shot-noise/data-dependent.

### 2) Add Secrest-baseline validation + residual-systematics check (recommended to include)
Add a short paragraph (either late in *Data and masking* or early in *Results*) stating that your pipeline reproduces the Secrest-style baseline at the fiducial cut and passes the standard “residual systematics vs maps” checks used in Secrest’s workflow.

Paper-ready validation figure(s) copied here:
- `systematics_grid_full_w1max16p4.png` (WISE-only baseline)
- `systematics_grid_no_nvss_w1max16p5.png` (NVSS-removed + homogenized variant)

Suggested wording points (numbers are already in `REPORTS/2-3-EEE/master_report.md`):
- Baseline (WISE-only; `W1_max=16.4`): `D≈0.01610`, `(l,b)≈(238.8°, +28.3°)`
- NVSS-removed + homogenized (`W1_max=16.5`): `D≈0.01531`, `(l,b)≈(239.5°, +30.1°)`
- Secrest-style binned residual χ²/ν vs common proxies is generally O(1) (largest excursions are scan-geometry/ecliptic-longitude-type proxies), indicating the pipeline behaves sensibly under the same diagnostics Secrest uses.

### 3) Add CMB-parallel / CMB-perpendicular decomposition (tightens the “axis vs direction” story)
You already use sign-invariant axis angles in the scan (appropriate for drift diagnostics), but referees will ask “is there still a CMB-aligned component?”
Adding a short decomposition plot and 1–2 sentences makes this crisp:

Paper-ready figure copied here:
- `cmb_projection_plot.png`

Key interpretive point to state:
- The drift is well described as a **growing non-CMB (perpendicular) component** with fainter `W1_max`, rather than “the dipole becomes unrelated to the CMB”.

### 4) Lognormal covariance: keep the disclaimer, and be explicit about scope
You already call the LSS covariance “approximate/diagnostic”. Consider adding one more sentence clarifying it is mainly used to support the amplitude S/N statement at a representative cut (e.g., `W1_max=16.6`), not as a final precision covariance for cosmological parameter inference.

### 5) Template degeneracy: make it quantitative (1 sentence if possible)
Where you claim “degenerate with depth modeling”, it helps to say explicitly that:
- the depth/coverage templates have nontrivial ℓ=1 content and/or correlate with the fitted dipole coefficient vector,
and that you report/inspect dipole–template correlations/covariances (these are already produced in the Poisson-GLM scan JSON outputs).

## Where I’d put these in the paper
- Main text: keep the current figures (scan, GLM, depth sensitivity, LSS cov, injection, validation).
- Appendix or short Results add-on: add the Secrest residual-systematics grid(s) + the CMB projection plot, and cite them as “baseline validation / interpretive diagnostic” rather than as new headline claims.
