#!/usr/bin/env bash
set -euo pipefail

cd /home/primary/QS

.venv/bin/python scripts/seasonal_selection_injection_check.py \
  --w1cov-min 80 \
  --b-cut 30 \
  --nside 64 \
  --w1-max 16.6 \
  --n-mocks 200 \
  --seed 123 \
  --lon-amps 0,0.01,0.02,0.03,0.04 \
  --lon-phase-deg 0
