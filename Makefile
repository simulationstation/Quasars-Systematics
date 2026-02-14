.PHONY: help reproduce reproduce-out

PY ?= python3

help:
	@echo "Targets:"
	@echo "  make reproduce            Generate a timestamped reviewer-seed report (headlines only)."
	@echo "  make reproduce-out OUT=...  Same as reproduce, but write to explicit OUT dir."

reproduce:
	@$(PY) scripts/reproduce_reviewer_seed.py

reproduce-out:
	@if [ -z "$(OUT)" ]; then echo "OUT is required, e.g. make reproduce-out OUT=outputs/reviewer_seed_custom"; exit 2; fi
	@$(PY) scripts/reproduce_reviewer_seed.py --out "$(OUT)"
