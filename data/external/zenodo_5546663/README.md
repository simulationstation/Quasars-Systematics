# GWTC-3 Sky Maps (Subset)

This folder contains a small subset of public GWTC-3 sky-localization maps (multi-order HEALPix FITS) used only for
the **axis/hemisphere proxy** in:

- `scripts/run_darksiren_axis_split_proxy.py`
- `entropy_quasar_bridge_report/`

Source:
- LIGO/Virgo/KAGRA public data release (GWTC-3 sky maps; Zenodo record `5546663`).

Notes:
- Files are stored as-is in `skymaps/` (UNIQ + PROBDENSITY representation).
- Only the 36 events referenced by the `2-1-c-m` production score bundle are included here.

