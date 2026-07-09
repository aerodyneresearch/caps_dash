"""CAPS Analysis Dashboard — Streamlit + Polars + Plotly.

Load a single CAPS instrument output file (one analysis day/session) and
explore it across a Data tab and a Metadata tab.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from baseline import BaselineError, apply_baseline_recalc, baseline_period_stats
from lod import LOD_SIGMA_FACTOR, allan_deviation, baseline_lod_series
from parser import CapsFile, read_caps_file

EXAMPLE_DIR = Path(__file__).parent / "example_data"
AVERAGING_OPTIONS = [1, 2, 5, 10, 30, 60, 300, 600, 900, 1800, 3600]
NUMERIC_DTYPES = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)

# Fallback unit inference by substring match on the column name, checked in
# order. Config files can override per column via a `Units:` mapping.
UNIT_PATTERNS = [
    ("concentration", "ppb"),
    ("loss", "Mm<sup>-1</sup>"),
    ("baseline_period", ""),
    ("baseline_number", ""),
    ("baseline", "Mm<sup>-1</sup>"),
    ("rayleigh", "Mm<sup>-1</sup>"),
    ("extinction", "Mm<sup>-1</sup>"),
    ("pressure", "Torr"),
    ("temperature", "K"),
    ("signal", "counts"),
    ("igortime", "s"),
]
# Plotly's default qualitative palette, used to color axis titles to match traces.
PLOTLY_COLORS = [
    "#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
    "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]
WEBGL_THRESHOLD = 20_000


@st.cache_data(show_spinner="Parsing CAPS file...", max_entries=8)
def load_caps_file(source) -> CapsFile:
    return read_caps_file(source)


@st.cache_data(show_spinner="Recalculating baselines...", max_entries=8)
def recalc_baselines(
    _caps_file: CapsFile, cache_key: str, sd_filter: float | None = None
) -> pl.DataFrame:
    return apply_baseline_recalc(_caps_file.data, _caps_file.config, sd_filter)


def column_unit(col: str, config: dict | None) -> str:
    units = (config or {}).get("Units") or {}
    if col in units:
        return str(units[col])
    low = col.lower()
    return next((unit for pattern, unit in UNIT_PATTERNS if pattern in low), "")


def build_timeseries_figure(
    plot_df: pl.DataFrame, selected_cols: list[str], config: dict | None
) -> go.Figure:
    """Line plot with one y-axis per distinct unit among the selected columns.

    First unit goes on the left axis, second on the right, and any further
    units stack outward on the right via Plotly's autoshift.
    """
    unit_groups: dict[str, list[str]] = {}
    for c in selected_cols:
        unit_groups.setdefault(column_unit(c, config), []).append(c)

    trace_cls = go.Scattergl if plot_df.height > WEBGL_THRESHOLD else go.Scatter
    fig = go.Figure()
    trace_idx = 0
    for axis_idx, (unit, group_cols) in enumerate(unit_groups.items()):
        axis_name = "yaxis" if axis_idx == 0 else f"yaxis{axis_idx + 1}"
        group_colors = []
        for c in group_cols:
            color = PLOTLY_COLORS[trace_idx % len(PLOTLY_COLORS)]
            group_colors.append(color)
            fig.add_trace(
                trace_cls(
                    x=plot_df["Timestamp"],
                    y=plot_df[c],
                    mode="lines",
                    name=c,
                    line={"color": color},
                    yaxis=f"y{axis_idx + 1}" if axis_idx else "y",
                )
            )
            trace_idx += 1

        if len(group_cols) == 1:
            title = f"{group_cols[0]} ({unit})" if unit else group_cols[0]
        else:
            title = unit or "Value"
        axis: dict = {"title": {"text": title}}
        # Color the axis to match its trace when unambiguous.
        if len(group_cols) == 1:
            axis["title"]["font"] = {"color": group_colors[0]}
            axis["tickfont"] = {"color": group_colors[0]}
        # Only the primary axis draws gridlines; more than one grid gets busy fast.
        if axis_idx == 1:
            axis.update(overlaying="y", side="right", showgrid=False, zeroline=False)
        elif axis_idx > 1:
            axis.update(
                overlaying="y",
                side="right",
                anchor="free",
                autoshift=True,
                showgrid=False,
                zeroline=False,
            )
        fig.update_layout({axis_name: axis})

    fig.update_layout(
        height=550,
        xaxis_title="Time (UTC)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 60},
    )
    return fig


def pick_data_source() -> CapsFile | None:
    st.sidebar.title("CAPS Dashboard")
    mode = st.sidebar.radio("Data source", ["Upload file", "Example data"])

    if mode == "Upload file":
        uploaded = st.sidebar.file_uploader("Select CAPS output file", type=["dat", "txt", "csv"])
        return load_caps_file(uploaded) if uploaded is not None else None

    if not EXAMPLE_DIR.exists():
        st.sidebar.info("No example_data folder found next to app.py.")
        return None

    examples = sorted(EXAMPLE_DIR.glob("**/*.dat"))
    if not examples:
        st.sidebar.info("No .dat files found in example_data.")
        return None

    choice = st.sidebar.selectbox(
        "Example file",
        examples,
        format_func=lambda p: p.relative_to(EXAMPLE_DIR).as_posix(),
    )
    return load_caps_file(str(choice)) if choice else None


def numeric_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in df.schema.items() if dt in NUMERIC_DTYPES]


def render_data_tab(caps_file: CapsFile) -> None:
    df = caps_file.data
    recalc_available = bool((caps_file.config or {}).get("Baseline_Recalculation"))

    control_col, plot_col = st.columns([1, 4])
    with control_col:
        if recalc_available and st.checkbox(
            "Recalculate baselines",
            help="Re-derive baselines and concentrations from the raw loss data "
            "(port of baseline_recalculation_functions_v2.R; see reference/).",
        ):
            try:
                df = recalc_baselines(caps_file, f"{caps_file.name}:{df.height}")
            except BaselineError as exc:
                st.warning(f"Baseline recalculation failed: {exc}")

        cols = numeric_columns(df)
        default_cols = []
        if caps_file.config:
            default_cols = [c for c in caps_file.config.get("Main_Plots", []) if c in cols]
        if not default_cols:
            default_cols = cols[:2]

        selected_cols = st.multiselect("Columns to plot", cols, default=default_cols)
        avg_seconds = st.selectbox("Averaging interval (s)", AVERAGING_OPTIONS, index=0)

    if not selected_cols:
        st.warning("Select at least one column to plot.")
        return

    plot_df = df
    if "Timestamp" in df.columns and avg_seconds > 1:
        plot_df = (
            df.drop_nulls("Timestamp")
            .sort("Timestamp")
            .group_by_dynamic("Timestamp", every=f"{avg_seconds}s")
            .agg([pl.col(c).mean() for c in selected_cols])
        )

    fig = build_timeseries_figure(plot_df, selected_cols, caps_file.config)
    with plot_col:
        st.plotly_chart(fig, width="stretch")

    st.subheader("Summary statistics")
    stats_rows = [
        {
            "Column": c,
            "Unit": column_unit(c, caps_file.config).replace("<sup>-1</sup>", "⁻¹"),
            "Mean": plot_df[c].mean(),
            "SD": plot_df[c].std(),
            "Min": plot_df[c].min(),
            "Max": plot_df[c].max(),
            "N": plot_df[c].drop_nulls().len(),
        }
        for c in selected_cols
    ]
    st.dataframe(stats_rows, width="stretch")

    st.subheader("Raw data preview")
    preview_rows = 1000
    st.dataframe(df.head(preview_rows), width="stretch", height=350)
    st.caption(f"Showing first {min(preview_rows, df.height):,} of {df.height:,} rows.")


def build_baseline_scan_figure(
    df: pl.DataFrame, stats: pl.DataFrame, baseline_number: int, loss_col: str
) -> go.Figure:
    """Loss during one baseline period, with its IQR window and filtered mean."""
    row = stats.filter(pl.col("baseline_number") == baseline_number).row(0, named=True)
    seg = df.filter(
        (pl.col("baseline_number") == baseline_number) & (pl.col("baseline_period") == 1)
    )
    x = seg["Timestamp"] if "Timestamp" in seg.columns else list(range(seg.height))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=seg[loss_col],
            mode="lines+markers",
            name=loss_col,
            line={"color": PLOTLY_COLORS[0], "width": 2},
            marker={"size": 6},
        )
    )
    fig.add_hline(
        y=row["min_threshold"],
        line={"dash": "dash", "color": "#999999", "width": 1},
        annotation_text="IQR window",
        annotation_position="bottom left",
    )
    fig.add_hline(y=row["max_threshold"], line={"dash": "dash", "color": "#999999", "width": 1})
    if row["mean_loss_filtered"] is not None:
        fig.add_hline(
            y=row["mean_loss_filtered"],
            line={"color": PLOTLY_COLORS[1], "width": 2},
            annotation_text="filtered mean",
            annotation_position="top right",
        )
    sd_txt = f"{row['sd_loss']:.3f}" if row["sd_loss"] is not None else "n/a"
    verdict = "good" if row["good"] else "flagged bad"
    fig.update_layout(
        title=f"Baseline #{baseline_number} — SD {sd_txt} Mm⁻¹ — {verdict}",
        height=340,
        xaxis_title="Time (UTC)",
        yaxis_title="Loss (Mm<sup>-1</sup>)",
        showlegend=False,
        margin={"t": 60},
    )
    return fig


def build_baseline_pdf_figure(
    df: pl.DataFrame, conc_orig: str, conc_recalc: str
) -> go.Figure:
    """Probability density of concentration during baseline (zero-air) periods."""
    base = df.filter(pl.col("baseline_period") == 1)
    fig = go.Figure()
    for col, label, color in [
        (conc_orig, f"Instrument ({conc_orig})", PLOTLY_COLORS[0]),
        (conc_recalc, f"Recalculated ({conc_recalc})", PLOTLY_COLORS[1]),
    ]:
        fig.add_trace(
            go.Histogram(
                x=base[col].drop_nulls(),
                histnorm="probability density",
                name=label,
                marker={"color": color},
                opacity=0.6,
                nbinsx=120,
            )
        )
    fig.update_layout(
        barmode="overlay",
        height=400,
        xaxis_title="Concentration (ppb)",
        yaxis_title="Probability density",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 40},
    )
    return fig


def render_baseline_tab(caps_file: CapsFile) -> None:
    settings = (caps_file.config or {}).get("Baseline_Recalculation") or {}
    cells = settings.get("Cells") or {}

    species_col, sd_col, _spacer = st.columns([1, 1, 2])
    with species_col:
        species = st.selectbox("Cell / species", list(cells))
    with sd_col:
        sd_filter = st.number_input(
            "SD filter (Mm⁻¹)",
            min_value=0.01,
            value=float(settings.get("Baseline_sd_filter", 0.3)),
            step=0.05,
            help="Baseline periods whose loss SD is at or above this are flagged "
            "bad and excluded from the recalculation.",
        )

    try:
        df = recalc_baselines(caps_file, f"{caps_file.name}:{caps_file.data.height}", sd_filter)
    except BaselineError as exc:
        st.warning(f"Baseline recalculation failed: {exc}")
        return

    cell = cells[species]
    loss_col = cell["Loss_col"]
    stats = baseline_period_stats(
        df, loss_col, sd_filter, float(settings.get("IQR_multiplier", 1.5))
    )
    bad = stats.filter(~pl.col("good"))
    st.caption(
        f"{stats.height} baseline periods — {stats.height - bad.height} good, "
        f"{bad.height} flagged bad (loss SD ≥ {sd_filter:g} Mm⁻¹)."
    )

    scan_all_col, scan_bad_col = st.columns(2)
    with scan_all_col:
        st.subheader("All baselines")
        nums = stats["baseline_number"].to_list()
        num = (
            st.select_slider("Baseline #", options=nums, key=f"scan_all_{species}")
            if len(nums) > 1
            else nums[0]
        )
        st.plotly_chart(
            build_baseline_scan_figure(df, stats, num, loss_col),
            width="stretch",
            key=f"fig_scan_all_{species}",
        )
    with scan_bad_col:
        st.subheader("Flagged baselines")
        bad_nums = bad["baseline_number"].to_list()
        if not bad_nums:
            st.info("No baselines flagged bad at this threshold.")
        else:
            bad_num = (
                st.select_slider("Flagged baseline #", options=bad_nums, key=f"scan_bad_{species}")
                if len(bad_nums) > 1
                else bad_nums[0]
            )
            st.plotly_chart(
                build_baseline_scan_figure(df, stats, bad_num, loss_col),
                width="stretch",
                key=f"fig_scan_bad_{species}",
            )

    st.subheader("Baseline-period concentration distribution")
    conc_orig = cell.get("Concentration_col") or f"Concentration_{species}"
    conc_recalc = f"concentration_{species}_interp"
    missing = [c for c in (conc_orig, conc_recalc) if c not in df.columns]
    if missing:
        st.info(f"Columns not available for the distribution plot: {missing}")
        return
    st.plotly_chart(
        build_baseline_pdf_figure(df, conc_orig, conc_recalc),
        width="stretch",
        key=f"fig_pdf_{species}",
    )
    st.caption(
        "Concentration reported during baseline (zero-air) periods, where the true "
        "value is zero. A tighter distribution centered on zero means better "
        "baseline correction."
    )


def build_lod_timeseries_figure(series: dict[str, pl.DataFrame]) -> go.Figure:
    """LOD (3*SD of baseline concentration) per baseline period, over time."""
    fig = go.Figure()
    for idx, (label, s) in enumerate(series.items()):
        x = s["start"] if "start" in s.columns else s["baseline_number"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=s["lod"],
                mode="lines+markers",
                name=label,
                line={"color": PLOTLY_COLORS[idx], "width": 2},
                marker={"size": 6},
            )
        )
    fig.update_layout(
        height=380,
        xaxis_title="Time (UTC)",
        yaxis_title="LOD, 3σ (ppb)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 40},
    )
    return fig


def build_allan_figure(curves: dict[str, pl.DataFrame], as_lod: bool) -> go.Figure:
    """Allan-Werle deviation vs averaging time, log-log, with a white-noise guide."""
    factor = LOD_SIGMA_FACTOR if as_lod else 1.0
    fig = go.Figure()
    first = next(iter(curves.values()))
    ref_y = (first["adev"][0] * factor) * (first["tau"][0] / first["tau"]).sqrt()
    fig.add_trace(
        go.Scatter(
            x=first["tau"],
            y=ref_y,
            mode="lines",
            name="white noise (τ⁻¹ᐟ²)",
            line={"dash": "dash", "color": "#999999", "width": 1},
        )
    )
    for idx, (label, curve) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=curve["tau"],
                y=curve["adev"] * factor,
                mode="lines+markers",
                name=label,
                line={"color": PLOTLY_COLORS[idx], "width": 2},
                marker={"size": 6},
            )
        )
    y_title = "LOD, 3σ (ppb)" if as_lod else "Allan deviation, σ (ppb)"
    fig.update_layout(
        height=420,
        xaxis={"type": "log", "title": {"text": "Averaging time (s)"}},
        yaxis={"type": "log", "title": {"text": y_title}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 40},
    )
    return fig


def render_lod_tab(caps_file: CapsFile) -> None:
    settings = (caps_file.config or {}).get("Baseline_Recalculation") or {}
    cells = settings.get("Cells") or {}

    species = st.selectbox("Cell / species", list(cells), key="lod_species")
    cell = cells[species]
    conc_orig = cell.get("Concentration_col") or f"Concentration_{species}"
    conc_recalc = f"concentration_{species}_interp"

    df = caps_file.data
    recalc_error = None
    try:
        df = recalc_baselines(caps_file, f"{caps_file.name}:{caps_file.data.height}")
    except BaselineError as exc:
        recalc_error = exc
    have_baselines = "baseline_period" in df.columns and bool(df["baseline_period"].sum())

    conc_series = [
        (label, col)
        for label, col in [("Instrument", conc_orig), ("Recalculated", conc_recalc)]
        if col in df.columns
    ]

    st.subheader("LOD time series")
    if not have_baselines:
        msg = f" ({recalc_error})" if recalc_error else ""
        st.info(f"No baseline periods found in this file{msg}.")
    else:
        series = {}
        for label, col in conc_series:
            s = baseline_lod_series(df, col)
            if s.height:
                series[f"{label} ({col})"] = s
        if not series:
            st.info("No baseline periods with enough points for an SD.")
        else:
            st.plotly_chart(
                build_lod_timeseries_figure(series), width="stretch", key=f"fig_lod_ts_{species}"
            )
            medians = ", ".join(
                f"{label}: {s['lod'].median():.3f} ppb" for label, s in series.items()
            )
            st.caption(
                f"LOD = {LOD_SIGMA_FACTOR:g}× the SD of concentration within each "
                f"baseline (zero-air) period. Median LOD — {medians}."
            )

    st.subheader("Allan-Werle deviation")
    mode_col, dur_col, opt_col = st.columns([1.4, 1, 1])
    with mode_col:
        segment_mode = st.radio(
            "Segment",
            ["Long baseline period", "Entire file / time range"],
            key=f"allan_mode_{species}",
            help="Pick a long zero-air stretch: either an extended baseline period, "
            "or the whole file if the instrument sampled zero air throughout.",
        )
    with dur_col:
        min_minutes = st.number_input(
            "Min baseline duration (min)", min_value=1.0, value=60.0, step=5.0,
            key=f"allan_minutes_{species}",
        )
    with opt_col:
        as_lod = st.checkbox("Show as LOD (3σ)", value=True, key=f"allan_lod_{species}")

    if segment_mode == "Long baseline period":
        if not have_baselines or "Timestamp" not in df.columns:
            st.info("No baseline periods (with timestamps) available in this file.")
            return
        durations = (
            df.filter(pl.col("baseline_period") == 1)
            .group_by("baseline_number")
            .agg(
                pl.col("Timestamp").min().alias("start"),
                (pl.col("Timestamp").max() - pl.col("Timestamp").min())
                .dt.total_seconds()
                .alias("seconds"),
            )
            .sort("baseline_number")
        )
        long_periods = durations.filter(pl.col("seconds") >= min_minutes * 60)
        if long_periods.is_empty():
            longest = durations["seconds"].max() or 0
            st.info(
                f"No baseline period lasts ≥ {min_minutes:g} min "
                f"(longest is {longest / 60:.1f} min). Lower the minimum duration, "
                "or use a dedicated long zero-air run with the "
                "'Entire file / time range' mode."
            )
            return
        labels = {
            f"#{r['baseline_number']} — {r['start']} ({r['seconds'] / 60:.1f} min)": r[
                "baseline_number"
            ]
            for r in long_periods.iter_rows(named=True)
        }
        choice = st.selectbox("Baseline period", list(labels), key=f"allan_period_{species}")
        seg = df.filter(
            (pl.col("baseline_number") == labels[choice]) & (pl.col("baseline_period") == 1)
        )
    else:
        seg = df
        if "Timestamp" in df.columns:
            ts = df["Timestamp"].drop_nulls()
            t_min, t_max = ts.min(), ts.max()
            start, end = st.slider(
                "Time range",
                min_value=t_min,
                max_value=t_max,
                value=(t_min, t_max),
                key=f"allan_range_{species}",
            )
            seg = df.filter(pl.col("Timestamp").is_between(start, end))
        st.caption(
            "Assumes the selected range is a zero-air run — on ambient data the "
            "Allan deviation reflects real atmospheric variability, not the LOD."
        )

    if "Timestamp" in seg.columns and seg.height > 1:
        dt = seg["Timestamp"].diff().median().total_seconds()
    else:
        dt = 1.0
    curves = {}
    for label, col in conc_series:
        try:
            curves[f"{label} ({col})"] = allan_deviation(seg[col].to_numpy(), dt)
        except ValueError as exc:
            st.warning(f"{label} ({col}): {exc}")
    if not curves:
        return
    st.plotly_chart(
        build_allan_figure(curves, as_lod), width="stretch", key=f"fig_allan_{species}"
    )
    factor = LOD_SIGMA_FACTOR if as_lod else 1.0
    best_parts = []
    for label, curve in curves.items():
        best = curve.sort("adev").row(0, named=True)
        best_parts.append(
            f"{label}: {best['adev'] * factor:.4f} ppb at τ = {best['tau']:.0f} s"
        )
    st.caption(
        f"{seg.height:,} samples at {dt:g} s spacing. "
        f"Optimum ({'LOD' if as_lod else 'σ'}) — {'; '.join(best_parts)}. "
        "The dashed guide is the pure white-noise slope; where the curve departs "
        "upward from it, drift outweighs further averaging."
    )


def render_metadata_tab(caps_file: CapsFile) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("File summary")
        summary = {
            "File name": caps_file.name,
            "Detected instrument type": caps_file.instrument_type or "Unknown",
            "Rows": f"{caps_file.data.height:,}",
            "Columns": str(caps_file.data.width),
        }
        if "Timestamp" in caps_file.data.columns:
            ts = caps_file.data["Timestamp"].drop_nulls()
            if ts.len():
                summary["Start"] = str(ts.min())
                summary["End"] = str(ts.max())
                summary["Duration"] = str(ts.max() - ts.min())
        st.table(summary)

    with right:
        st.subheader("Instrument header")
        if caps_file.header_info:
            st.table(caps_file.header_info)
        else:
            st.info("No descriptive header fields found.")

    st.subheader("Detected configuration")
    if caps_file.config:
        st.json(caps_file.config)
    else:
        st.warning("No matching config file found for this instrument type.")

    with st.expander("Raw metadata parameter block"):
        st.write(caps_file.parameters)


def main() -> None:
    st.set_page_config(page_title="CAPS Dashboard", layout="wide")

    caps_file = pick_data_source()
    if caps_file is None:
        st.title("CAPS Analysis Dashboard")
        st.info("Load a CAPS output file from the sidebar to begin.")
        return

    st.title(f"CAPS Analysis — {caps_file.name}")
    recalc_available = bool(
        ((caps_file.config or {}).get("Baseline_Recalculation") or {}).get("Cells")
    )
    if recalc_available:
        tab_data, tab_baseline, tab_lod, tab_metadata = st.tabs(
            ["Data", "Baseline Recalc", "LOD", "Metadata"]
        )
        with tab_baseline:
            render_baseline_tab(caps_file)
        with tab_lod:
            render_lod_tab(caps_file)
    else:
        tab_data, tab_metadata = st.tabs(["Data", "Metadata"])
    with tab_data:
        render_data_tab(caps_file)
    with tab_metadata:
        render_metadata_tab(caps_file)


main()
