#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Use most cores, but leave a little headroom.
NPROC=$(python3 - <<'PY'
import os
n=os.cpu_count() or 2
print(max(1, n-8))
PY
)

BASE_SCAN="REPORTS/amplitude_physical_predictors_suite/bcut30_baseline/rvmp_fig5_poisson_glm.json"
LON_SCAN="outputs/rvmp_fig5_poisson_glm_eclip_sincos/rvmp_fig5_poisson_glm.json"

# 1) Null: fixed true dipole, no seasonal injection.
.venv/bin/python scripts/seasonal_drift_mc.py \
  --n-mocks 100000 \
  --n-proc "$NPROC" \
  --real-scan-json "$BASE_SCAN" \
  --dipole-amp 0.016778520598772835 \
  --dipole-axis-lb "236.5768885465019,21.805941893933664" \
  --lon-sin 0 \
  --lon-cos 0 \
  --outdir outputs/seasonal_drift_mc_null_20260204_1

# 2) Seasonal injection: inject lon pattern (from scan) + a kinematic-scale true dipole.
.venv/bin/python scripts/seasonal_drift_mc.py \
  --n-mocks 50000 \
  --n-proc "$NPROC" \
  --real-scan-json "$BASE_SCAN" \
  --dipole-amp 0.0046 \
  --dipole-axis-lb "264.021,48.253" \
  --lon-from-scan "$LON_SCAN" \
  --lon-scale 1.0 \
  --include-lon-fit \
  --outdir outputs/seasonal_drift_mc_inject_20260204_1

printf "\nDONE: %s\n" "$(date -u --iso-8601=seconds)"
