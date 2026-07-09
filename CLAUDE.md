# CAPS Analysis Dashboard — agent notes

Streamlit + Polars + Plotly dashboard for Aerodyne CAPS instrument output
files. See README.md for layout and setup. Entry point: `app.py`.
This is a Python port of an older R/Quarto dashboard (`caps_functions.R` /
`caps_info.R`); `parser.py` mirrors that logic.

## Conventions to preserve

- **Units live in `config/<instrument>.yaml` under `Units:`, not in code.**
  `app.py` falls back to name-substring inference (`UNIT_PATTERNS`) only for
  columns/instruments without a config entry. Unit strings may contain HTML
  (`Mm<sup>-1</sup>`) because they render in Plotly axis titles.
- **Plotting: columns sharing a unit share a y-axis; each distinct unit gets
  its own axis** (left, then right, then autoshifted right). Gridlines and
  zero-lines are deliberately drawn by the primary axis only — do not enable
  them on secondary axes; multiple grids make the plot unreadable.
  If a request ever needs 4+ units at once, prefer switching to stacked
  subplots with a shared x-axis over adding more overlaid axes.
- **Instrument detection**: first digit of the serial number at position 32
  (1-indexed) of the `%`-prefixed parameter block (see `INSTRUMENT_TYPES` in
  `parser.py`). The parameter block is kept as raw string tokens because some
  positions hold dates — do not convert it to floats.
- Traces use `Scattergl` above `WEBGL_THRESHOLD` rows (files are often
  50k+ rows at 1 Hz); keep that when touching the plot code.
- `example_data/` contains `.log` files the app never lists (it globs
  `**/*.dat` only); they are raw instrument logs kept for reference.
- **`baseline.py` is a faithful port of
  `reference/baseline_recalculation_functions_v2.R`** — validated bit-for-bit
  (~1e-13) against R on `example_data/Baseline_Recalc/`. It deliberately
  mirrors R quirks (e.g. the per-period IQR window filters all rows, not just
  baseline rows; baseline detection always uses the instrument-level status
  column). Don't "fix" those without re-validating against the R output.
  Per-instrument knobs live in `config/<instrument>.yaml` under
  `Baseline_Recalculation:`; physics constants (Rayleigh table, LED
  wavelength codes) stay in `baseline.py`. Bad baselines are *flagged*
  (`baseline_good_<species>`), not removed — the recalc math still excludes
  them internally, so numeric outputs stay identical to R.

## Workflow notes

- The user typically has their own Streamlit instance running on port 8501
  (started outside the agent). Don't kill it; run verification servers on
  another port. Streamlit ignores the `PORT` env var — pass
  `--server.port` explicitly.
- Python 3.11 venv at `.venv/`; on this machine bare `python` is not on
  PATH — use `.venv/Scripts/python.exe`.
