#!/usr/bin/env bash
set -euo pipefail

cd /home/primary/QS

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PHASE_DEG="166.2604027634997"
LON_AMPS="0,0.008192567026925831"

run_case() {
  local tag="$1"
  local dip="$2"
  local outdir="REPORTS/seasonal_selection_injection_check/${tag}"
  mkdir -p "$outdir"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running: tag=${tag} dipole_amp=${dip} phase_deg=${PHASE_DEG} lon_amps=${LON_AMPS}"
  .venv/bin/python scripts/seasonal_selection_injection_check.py \
    --n-mocks 400 \
    --lon-amps "${LON_AMPS}" \
    --lon-phase-deg "${PHASE_DEG}" \
    --dipole-amp "${dip}" \
    --outdir "${outdir}"
}

run_case dipole0_phase166 0
run_case dipole0p0046_phase166 0.0046
