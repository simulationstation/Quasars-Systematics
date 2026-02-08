# Radio NB Dipole Audit Note (PRD / REVTeX)

This folder contains a short REVTeX (PRD-style) manuscript and figures summarizing the
negative-binomial counts-in-cells radio dipole robustness tests implemented in
`scripts/run_radio_nb_dipole_audit.py`.

## Contents

- `main.tex`: REVTeX 4.2 manuscript (APS/PRD style).
- `make_figures.py`: regenerates the paper figures from the pinned JSON outputs.
- `fig1_direction_compare.png`: fitted dipole directions under model variants.
- `fig2_delta_ic.png`: delta AIC/BIC (vs pure) for joint fits.
- `fig3_template_coefficients.png`: per-survey template coefficients in the exp(+phys) fit.
- `fig4_injection_recovery.png`: injection/recovery summary (axis error + amplitude).
- `figure_summary.json`: small machine-readable figure bookkeeping.

## Rebuild Figures

```bash
.venv/bin/python radio_a/make_figures.py
```

## Compile

From repo root:

```bash
pdflatex -interaction=nonstopmode radio_a/main.tex
pdflatex -interaction=nonstopmode radio_a/main.tex
```

Or from inside `radio_a/`:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

