#!/usr/bin/env bash
set -euo pipefail

cd /home/primary/QS

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_case() {
  local tag="$1"
  local dip="$2"
  local outdir="REPORTS/seasonal_selection_injection_check/${tag}"
  mkdir -p "$outdir"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running: tag=${tag} dipole_amp=${dip}"
  .venv/bin/python scripts/seasonal_selection_injection_check.py \
    --n-mocks 200 \
    --lon-amps 0,0.01,0.02,0.03,0.04 \
    --lon-phase-deg 0 \
    --dipole-amp "${dip}" \
    --outdir "${outdir}"
}

run_case dipole0 0
run_case dipole0p0046 0.0046
