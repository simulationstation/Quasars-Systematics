#!/usr/bin/env bash
set -euo pipefail
cd /home/primary/QS

.venv/bin/python scripts/run_selection_sim_depthmap_plus_lon.py \
  --w1-grid 15.5,16.6,0.05 \
  --n-mocks 40 \
  --depth-sel-scale 0.0 \
  --lon-coeffs-scan-json outputs/rvmp_fig5_poisson_glm_eclip_sincos/rvmp_fig5_poisson_glm.json \
  --outdir REPORTS/selection_sim_depthmap_plus_lon/lon_only_scan
