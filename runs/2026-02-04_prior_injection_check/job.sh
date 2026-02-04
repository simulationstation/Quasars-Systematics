#!/usr/bin/env bash
set -euo pipefail

cd /home/primary/QS

.venv/bin/python scripts/harmonic_prior_injection_check.py \
  --w1-max 16.6 \
  --harmonic-lmax 5 \
  --n-mocks 200 \
  --true-cl-scale 10 \
  --fit-prior-scale 1 \
  --seed 123 \
  --out-json REPORTS/arxiv_amplitude_multipole_prior_injection/data/lowell_injection_validation.json \
  --out-fig REPORTS/arxiv_amplitude_multipole_prior_injection/figures/lowell_injection_validation.png
