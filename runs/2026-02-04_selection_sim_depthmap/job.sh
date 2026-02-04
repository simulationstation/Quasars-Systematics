#!/usr/bin/env bash
set -euo pipefail

cd /home/primary/QS

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTDIR="REPORTS/selection_sim_depthmap/delta_m_scale1"
mkdir -p "$OUTDIR"

.venv/bin/python scripts/run_selection_sim_depthmap.py \
  --depth-map-fits REPORTS/external_completeness_sdss_dr16q/data/delta_m_map_nside64.fits \
  --depth-map-kind delta_m_mag \
  --sel-scale 1.0 \
  --n-mocks 100 \
  --seed 123 \
  --outdir "$OUTDIR"
