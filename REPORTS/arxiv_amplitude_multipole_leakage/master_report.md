# ArXiv amplitude check: low-ℓ multipole leakage (Abghari+24-inspired)

Motivation: Abghari et al. (arXiv:2405.09762) argue the CatWISE quasar map is not a pure dipole and contains low-ℓ multipoles comparable to the dipole; on a ~50% sky mask this can couple modes and inflate dipole uncertainty / bias amplitude.

Test here: rerun the **Poisson GLM** scan but include real spherical-harmonic nuisance templates (free coefficients) for ℓ=2..ℓ_max, on top of the usual dipole + ecliptic-latitude template.

## Outputs

- `REPORTS/arxiv_amplitude_multipole_leakage/data/rvmp_fig5_poisson_glm_harmonic_lmax3.json`
- `REPORTS/arxiv_amplitude_multipole_leakage/figures/rvmp_fig5_poisson_glm_harmonic_lmax3.png`
- `REPORTS/arxiv_amplitude_multipole_leakage/data/rvmp_fig5_poisson_glm_harmonic_lmax5.json`
- `REPORTS/arxiv_amplitude_multipole_leakage/figures/rvmp_fig5_poisson_glm_harmonic_lmax5.png`
- `REPORTS/arxiv_amplitude_multipole_leakage/figures/cmb_projection_compare.png` (includes D_hat curves)

## Key amplitude numbers (W1_max=16.6)

- baseline (eclip-only): D_hat=0.0167785
- harmonic nuisance ℓ<=3: D_hat=0.0216548 (+29.1% vs baseline)
- harmonic nuisance ℓ<=5: D_hat=0.0234688 (+39.9% vs baseline)

Interpretation:
- If adding low-ℓ nuisance modes drives D_hat down or blows up its uncertainty, that supports the “multipole leakage” concern for amplitude claims.
- If D_hat is stable, that argues the measured amplitude is not primarily an artifact of low-ℓ leakage (though other systematics can still exist).