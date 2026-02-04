#!/usr/bin/env bash
set -euo pipefail

cd /home/primary/QS

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTDIR="REPORTS/selection_sim_depthmap/delta_m_scale2p7_diag"
mkdir -p "$OUTDIR"

.venv/bin/python scripts/selection_sim_depthmap_diagnostics.py \
  --w1-cut 16.6 \
  --depth-map-fits REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits \
  --depth-map-kind delta_m_mag \
  --sel-scale 2.7 \
  --n-mocks 200 \
  --seed 123 \
  --lambda-edges 0,90,180,270,360 \
  --outdir "$OUTDIR"
