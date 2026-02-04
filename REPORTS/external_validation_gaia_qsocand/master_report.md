# All-sky external validation (Gaia DR3 QSO candidates)

This report documents an **all-sky external validation** of the CatWISE “accepted” selection using **Gaia DR3 QSO
candidates** (CDS `I/356`, file `qsocand.dat`).

This is complementary to (and substantially less WISE-circular than) SDSS-only validations because Gaia’s QSO
candidates are **all-sky** and are not defined by WISE scan depth.

Important limitation: Gaia qsocand does **not** provide `W1`, so this is a **spatial** validation of map-level
structure (not a full magnitude-dependent completeness calibration).

---

## 1) Inputs

### 1.1 Gaia DR3 QSO candidates (external, all-sky)
- Source: CDS `I/356` (Gaia DR3 Part 2: Extra-galactic)
- File: `qsocand.dat.gz` (not git-tracked; large download)
- Local path used:
  - `data/external/gaia_dr3_extragal/qsocand.dat.gz`
- We apply a purity threshold:
  - `PQSO >= 0.8` (column `classprob_dsc_combmod_quasar` in the CDS ReadMe)

### 1.2 CatWISE “accepted” catalog (target selection)
- `data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits`

### 1.3 Secrest exclusion discs (mask geometry)
- `data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/exclude_master_revised.fits`

### 1.4 Independent map-level depth proxy (unWISE imaging metadata)
- `data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits` (Galactic coords)

### 1.5 SDSS DR16Q footprint proxy (for *inside/outside SDSS* splits)
- `data/external/sdss_dr16q/DR16Q_v4.fits` (not git-tracked; large)
- Used only to build a coarse “SDSS footprint” map as `count(pix)>0`.

---

## 2) Method (all-sky external “acceptance fraction” map)

### 2.1 Footprint (Secrest-style)
We define the analysis footprint exactly as in the main dipole work:
- parent coverage: `W1cov >= 80`
- Secrest “mask_zeros” on the parent map (zeros + neighbour pixels)
- exclusion discs from `exclude_master_revised.fits` (where `use==True`)
- Galactic plane cut on pixel centers: `|b| > 30°`

### 2.2 Matching
For each Gaia qsocand source we ask if it is present in CatWISE “accepted”:
- nearest-neighbour match in ICRS using a 3D unit-vector KD-tree
- match radius: **2 arcsec**

### 2.3 Per-pixel acceptance fraction
On HEALPix (`Nside=64`, Galactic coords), we compute:
- `N_gaia(pix)` = Gaia qsocand count in the footprint
- `N_match(pix)` = number of those with a CatWISE accepted match
- `f_accept(pix) = N_match / N_gaia`

Artifacts:
- `REPORTS/external_validation_gaia_qsocand/data/gaia_qso_total_nside64.fits`
- `REPORTS/external_validation_gaia_qsocand/data/gaia_qso_match_nside64.fits`
- `REPORTS/external_validation_gaia_qsocand/data/gaia_qso_accept_frac_nside64.fits`

Visualization:
- `REPORTS/external_validation_gaia_qsocand/figures/gaia_accept_frac_mollweide.png`

Summary (this run; `PQSO>=0.8`, within the Secrest footprint):
- Gaia QSOs in footprint: **1,595,844**
- matched to CatWISE accepted: **627,103**
- overall acceptance fraction: **0.393**

---

## 3) Comparison to the SDSS-trained depth-only completeness map

We compare `f_accept(pix)` against the SDSS DR16Q–trained **depth-only** map-level completeness proxy (a δm map),
and equivalently against `logNexp(pix)` (because this δm model is a function of logNexp only).

Reference maps:
- `REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits`
- `data/cache/unwise_nexp/neo7/lognexp_healpix_nside64.fits`

Diagnostics:
- binned quantile curves: `f_accept` vs δm (inside/outside SDSS footprint)
  - `REPORTS/external_validation_gaia_qsocand/figures/accept_frac_vs_delta_m.png`
- binned quantile curves: `f_accept` vs logNexp (inside/outside SDSS footprint)
  - `REPORTS/external_validation_gaia_qsocand/figures/accept_frac_vs_lognexp.png`
- SDSS footprint proxy map (DR16Q counts per pixel)
  - `REPORTS/external_validation_gaia_qsocand/figures/sdss_dr16q_count_mollweide.png`

### 3.1 Spearman correlation (per-pixel; pixels with ≥20 Gaia QSOs)
From `REPORTS/external_validation_gaia_qsocand/data/meta.json`:
- **All pixels:** ρ ≈ **−0.070** (p ≈ 1.8e−27)
- **Inside SDSS footprint (DR16Q count>0):** ρ ≈ **−0.265** (p ≈ 1.8e−191)
- **Outside SDSS footprint:** ρ ≈ **+0.111** (p ≈ 8.5e−34)

Interpretation: under this Gaia proxy, the relationship between the SDSS-trained depth-only map and all-sky CatWISE
acceptance is **not globally stable** and flips sign between the SDSS region and the rest of the sky.

This is consistent with the broader theme of the paper: a single “depth-only” proxy is *not* sufficient to validate or
stabilize dipole-direction inferences all-sky, and completeness modeling remains an identifiability bottleneck.

---

## 4) Reproducibility

This report was produced by:
- `scripts/run_gaia_dr3_qsocand_external_validation.py`

One-shot command (writes this report’s `data/` + `figures/` folders):
```bash
python3 scripts/run_gaia_dr3_qsocand_external_validation.py \\
  --qsocand-gz data/external/gaia_dr3_extragal/qsocand.dat.gz \\
  --pqso-min 0.8 \\
  --make-plots \\
  --export-report-dir REPORTS/external_validation_gaia_qsocand
```

Notes:
- The raw Gaia file is large and is intentionally excluded from git via `.gitignore`.
- The runtime is dominated by streaming parse + KD-tree queries; the script supports `--workers` for multi-core.
