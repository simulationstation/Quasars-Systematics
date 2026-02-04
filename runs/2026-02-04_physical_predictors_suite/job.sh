#!/usr/bin/env bash
set -euo pipefail

cd /home/primary/QS

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BASE_OUT="REPORTS/amplitude_physical_predictors_suite"
mkdir -p "$BASE_OUT"

W1_GRID="15.5,16.6,0.05"

run_case() {
  local bcut="$1"
  local eclip="$2"
  local depth_mode="$3"
  local tag="$4"

  local outdir="${BASE_OUT}/bcut${bcut}_${tag}"
  mkdir -p "$outdir"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] bcut=${bcut} eclip=${eclip} depth=${depth_mode} -> ${outdir}"

  local extra=()
  if [[ "$depth_mode" == "unwise_nexp_covariate" || "$depth_mode" == "unwise_nexp_offset" ]]; then
    extra+=(--nexp-tile-stats-json data/cache/unwise_nexp/neo7/w1_n_m_tile_stats_median.json)
  fi
  if [[ "$depth_mode" == "depth_map_covariate" || "$depth_mode" == "depth_map_offset" ]]; then
    extra+=(--depth-map-fits data/cache/unwise_invvar/neo7/invvar_healpix_nside64.fits --depth-map-name unwise_invvar_nside64)
  fi
  if [[ "$depth_mode" == "external_logreg_integrated_offset" ]]; then
    extra+=(--external-logreg-meta REPORTS/external_completeness_sdss_dr16q/data/sdss_dr16q_depthonly_meta.json)
  fi

  .venv/bin/python scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py \
    --nside 64 \
    --w1cov-min 80 \
    --b-cut "${bcut}" \
    --w1-mode cumulative \
    --w1-grid "${W1_GRID}" \
    --eclip-template "${eclip}" \
    --dust-template none \
    --depth-mode "${depth_mode}" \
    --make-plot \
    --outdir "${outdir}" \
    "${extra[@]}"
}

# Sweep over masks and physical depth/coverage models.
for bcut in 25 30 35; do
  run_case "$bcut" abs_elat none baseline
  run_case "$bcut" abs_elat w1cov_covariate w1cov_cov
  run_case "$bcut" abs_elat unwise_nexp_covariate unwise_nexp_cov
  run_case "$bcut" abs_elat depth_map_covariate unwise_invvar_cov
  run_case "$bcut" abs_elat external_logreg_integrated_offset sdss_depthonly_offset

  # Include sin/cos(ecliptic lon) as a comparison (still with physical completeness offset).
  run_case "$bcut" abs_elat_sincos_elon external_logreg_integrated_offset sdss_offset_plus_lon
  echo
done

# Summarize: extract D_hat at W1_max=16.6 for all runs.
.venv/bin/python - <<'PY'
import json
from pathlib import Path

base = Path('REPORTS/amplitude_physical_predictors_suite')
rows = []
for d in sorted(p for p in base.iterdir() if p.is_dir()):
    j = d / 'rvmp_fig5_poisson_glm.json'
    if not j.exists():
        continue
    data = json.loads(j.read_text())
    # Find the entry at W1_cut=16.6 (float comparisons: use string formatting)
    found = None
    for r in data.get('results', []):
        if abs(float(r.get('w1_cut')) - 16.6) < 1e-9:
            found = r
            break
    if found is None:
        continue
    rows.append({
        'run': d.name,
        'D_hat': float(found['dipole']['D_hat']),
        'l_hat_deg': float(found['dipole']['l_hat_deg']),
        'b_hat_deg': float(found['dipole']['b_hat_deg']),
        'angle_to_cmb_deg': float(found['dipole'].get('angle_to_cmb_deg', float('nan'))),
        'dispersion_pearson_chi2_per_dof': float(found.get('fit_diag', {}).get('pearson_chi2_over_dof', float('nan'))),
    })

out = base / 'summary_w1max16p6.json'
out.write_text(json.dumps({'rows': rows}, indent=2))
print('Wrote:', out)
PY
