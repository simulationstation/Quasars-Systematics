# Entropy ↔ Quasar Bridge (Fast Dark-Siren Hemisphere Proxy)

Date: 2026-02-03 (UTC)

This folder bundles a **fast proxy test** that ties the **“corrected quasar axis” idea** to the **dark-siren
preference** for the entropy propagation model described in `EntropyPaper.tex`.

The motivation is the Gemini suggestion: if there were a real, sky-dependent “entropy gradient” (or any other
directional propagation/selection texture), then the **dark-siren score** might concentrate preferentially in one
hemisphere about a physically motivated axis (e.g., the CMB dipole / kinematic axis, or a quasar-derived axis).

This is **not** an end-to-end anisotropic model fit. It is a **diagnostic**: “is the score directional at all?”

## What is being tested (plain language)

We ask: “Do the GW events in the ‘head’ half of the sky (relative to a chosen axis) provide noticeably more support
for the entropy propagation model than the events in the ‘tail’ half?”

If the answer were “yes (strongly),” it would motivate building a *real* anisotropic extension of the model.
If the answer is “no,” it supports the interpretation that the current preference is mostly sky-independent
(consistent with the sky-rotation control described in the paper).

## Inputs (imported into this repo; *not* modifying `/home/primary/PROJECT`)

From the paper run bundle `2-1-c-m`:
- Per-event score table: `data/dark_sirens/2-1-c-m/production_36events/event_scores_M0_start101.json`
  - This is the seed/run that matches the paper headline `ΔLPD_tot ≈ 3.03` (but note the caveat below).
- Summary/diagnostics: `data/dark_sirens/2-1-c-m/production_36events/summary_M0_start101.json`,
  `data/dark_sirens/2-1-c-m/diagnostics/diagnostic_summary.json`

Public GWTC-3 sky maps (UNIQ + PROBDENSITY multi-order FITS):
- `data/external/zenodo_5546663/skymaps/` (36 files; ~24 MB total)

## Method (what the script does)

Script: `scripts/run_darksiren_axis_split_proxy.py`

For each event sky map, it computes:
- `P_head = ∫_{axis·n>0} p(n) dΩ`

Then it splits the **per-event** score `ΔLPD_i` (HE − GR) into:
- `ΔLPD_head = Σ ΔLPD_i * P_head,i`
- `ΔLPD_tail = Σ ΔLPD_i * (1 − P_head,i)`

To quantify “is this bigger than chance?”, it uses a **permutation test**:
- shuffle `ΔLPD_i` across events (holding `P_head,i` fixed) and recompute `ΔLPD_head − ΔLPD_tail`

## Additional directionality check: “dipole in the per-event score”

The hemisphere test is a step-function split about a chosen axis. To also test for a smoother, dipole-like pattern
in the *per-event preference*, we run a second proxy:

- Script: `scripts/run_darksiren_score_dipole_proxy.py`

For each event, this computes the sky-posterior mean direction (ICRS Cartesian)
`m_i = ∫ n p_i(n) dΩ / ∫ p_i(n) dΩ` from the public GWTC-3 sky maps, then fits:

`ΔLPD_i ≈ a + d·m_i`

and estimates the dipole amplitude `|d|` significance via a permutation test (shuffle ΔLPD across events).

## Results (this proxy run)

Figures:
- `entropy_quasar_bridge_report/figures/axis_split_proxy_cmb.png`
- `entropy_quasar_bridge_report/figures/axis_split_proxy_secrest.png`
- `entropy_quasar_bridge_report/figures/axis_split_proxy_ecliptic_north.png`

JSON outputs:
- `entropy_quasar_bridge_report/data/axis_split_proxy_cmb.json`
- `entropy_quasar_bridge_report/data/axis_split_proxy_secrest.json`
- `entropy_quasar_bridge_report/data/axis_split_proxy_ecliptic_north.json`

Headline (two-sided permutation p-values for directionality of the per-event score):
- **CMB axis:** `Δ(ΔLPD)=+1.57`, `p≈0.40`
- **Secrest axis:** `Δ(ΔLPD)=+1.80`, `p≈0.36`
- **Ecliptic north:** `Δ(ΔLPD)=−1.06`, `p≈0.60`

Interpretation (plain):
- Under this fast proxy, the dark-siren preference is **not strongly concentrated** in a single hemisphere about
  these axes. The score looks **consistent with sky-independent behavior**.

Dipole-regression proxy result:
- Outputs:
  - `entropy_quasar_bridge_report/data/score_dipole_proxy.json`
  - `entropy_quasar_bridge_report/figures/score_dipole_proxy.png`
- Best-fit dipole axis (Galactic): `(l,b) ≈ (277.9°, 16.9°)` **but** with **no significance**:
  - permutation (one-sided) p-value for large `|d|`: **p ≈ 0.77** (i.e., no evidence for a dipole in ΔLPD).

Extra note (not directionality, but useful context):
- The per-event ΔLPD shows a moderate correlation with localization / catalog complexity proxies
  (`corr(ΔLPD, log10 sky area) ≈ 0.32`, `corr(ΔLPD, log10 n_gal) ≈ 0.31`), suggesting the per-event preference is
  more about event informativeness / mixture-model details than sky direction.

## Critical caveat (why ΔLPD_tot here ≠ paper ΔLPD_tot)

The production score in the paper is a joint marginalization over shared model draws (global parameters), i.e.
it uses `logmeanexp(sum_events logL_draw)` rather than `sum_events logmeanexp(logL_draw)`.

The `event_scores_*.json` file contains **per-event** marginal LPDs, so:
- `Σ ΔLPD_i` from the event table is not expected to match the paper’s joint `ΔLPD_tot`.

This proxy test is therefore:
- appropriate for a **quick “is anything directional?”** diagnostic,
- but **not** a substitute for an anisotropic full likelihood evaluation.

## How to reproduce (repo-local)

Run from the repo root:

```bash
.venv/bin/python scripts/run_darksiren_axis_split_proxy.py \
  --axis cmb \
  --n-perm 5000 \
  --seed 1 \
  --make-plot \
  --outdir outputs/darksiren_axis_proxy_cmb
```

Try alternate axes:

```bash
.venv/bin/python scripts/run_darksiren_axis_split_proxy.py --axis secrest --make-plot --outdir outputs/darksiren_axis_proxy_secrest
.venv/bin/python scripts/run_darksiren_axis_split_proxy.py --axis ecliptic_north --make-plot --outdir outputs/darksiren_axis_proxy_eclnorth
```

## What this implies for the Gemini “entropy-gradient” idea

This fast proxy does **not** (currently) show a clean hemispherical signature of the dark-siren preference along the
CMB/quasar axes. That leans toward:
- the siren preference being **mostly spectral / population-level** (sky-independent), consistent with the
  paper’s PE-sky-rotation control, and
- quasar dipole systematics being **a separate sky-structured selection problem** unless/until a more explicit
  anisotropic propagation model is built and validated.

If you still want a stronger, “directly tied” test, the next step would be to extend the siren model with an
explicit dipole parameter (e.g., `d_L^GW(z,n)=d_L^GW(z)[1+g cosθ]`) and fit `g` along a fixed axis (CMB or a
depth-controlled quasar residual axis). That would be a new model/fit, not just a proxy.
