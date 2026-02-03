#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${1:?need OUTROOT arg}"
cd /home/primary/QS

# Keep BLAS/OpenMP from oversubscribing; we use process-level parallelism.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NPROC=36
GGRID="-0.6,0.6,0.1"
PROGRESS_S=30
SEL_PROGRESS_EVERY=10

run_one() {
  local name="$1"; shift
  local preset="$1"; shift

  mkdir -p "$OUTROOT/$name"
  echo "[triplet] START ${name} preset=${preset} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  ( \
    .venv/bin/python scripts/run_darksiren_fixed_axis_gscan_full.py \
      --axis "${preset}" \
      --g-grid="${GGRID}" \
      --nproc "${NPROC}" \
      --g-prior-type normal \
      --g-prior-mu 0.0 \
      --g-prior-sigma 0.2 \
      --progress-seconds "${PROGRESS_S}" \
      --selection-progress-every "${SEL_PROGRESS_EVERY}" \
      --outdir "$OUTROOT/$name" \
      --make-plot \
  ) > "$OUTROOT/$name/run.log" 2>&1 &

  echo $! > "$OUTROOT/$name/pid.txt"
}

run_one cmb cmb
run_one secrest secrest
run_one ecliptic_north ecliptic_north

pids=( $(cat "$OUTROOT"/*/pid.txt) )
names=( cmb secrest ecliptic_north )

status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "${pid}"; then
    echo "[triplet] DONE ${name} ok utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "[triplet] DONE ${name} FAIL utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    status=1
  fi
done

echo "[triplet] ALL_DONE status=${status} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "${status}"
