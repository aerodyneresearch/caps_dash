# CAPS Analysis Dashboard

Streamlit dashboard for exploring Aerodyne CAPS instrument output files
(NO2, NOx, extinction, SSA). Load a single `.dat` output file and explore
it across a Data tab (unit-aware time-series plots, averaging, summary
statistics) and a Metadata tab (instrument header and configuration).

## Layout

```
caps_dashboard/
├── app.py            # Streamlit entry point
├── parser.py         # CAPS .dat file parser (header, parameter block, data table)
├── config/           # Per-instrument dashboard config (plots, units, baseline recalc)
│   ├── no2.yaml
│   └── nox.yaml
├── example_data/     # Sample instrument output files (only *.dat files are listed)
└── requirements.txt
```

The instrument type is detected from the serial number embedded in the
file's parameter block; the matching `config/<type>.yaml` then supplies
default plot columns and per-column units. Columns that share a unit share
a y-axis; each additional unit gets its own axis.

## Setup

Requires Python 3.11+.

```
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on Unix)
pip install -r requirements.txt
```

## Run

```
streamlit run app.py
```

Then open http://localhost:8501 and load a file from the sidebar
(upload, or pick one from `example_data/`).

## Notes

- The virtual environment is not portable: after moving this folder,
  delete `.venv` and recreate it with the setup steps above.
- `example_data/` includes raw `.log` files alongside the `.dat` files;
  the dashboard only lists `.dat` files. Delete the logs (~54 MB) if you
  want a leaner repository.
