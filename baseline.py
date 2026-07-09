"""Baseline recalculation for CAPS files.

Python/Polars port of reference/baseline_recalculation_functions_v2.R.
Step order and output column names mirror the R code so results can be
compared file-for-file:

  1. ``baseline_period`` — 1 where the instrument status starts with the
     configured baseline prefix (e.g. "32"), else 0.
  2. ``baseline_number`` — each baseline period and the measurement rows
     that follow it share a number; rows before the first baseline share
     number 1 with the first period.
  3. ``baseline_good_<species>`` — baseline periods whose loss SD is at or
     above the configured filter are flagged bad. (The R code demoted them
     to measurement rows outright; here they stay identifiable, but the
     recalculation math still excludes them, so the numeric outputs are
     unchanged.)
  4. ``rayleigh_<nm>`` — per-row Rayleigh scattering loss for the cell's
     LED wavelength (last digit of the cell status value), scaled from STP
     to cell temperature and pressure.
  5. ``LastBaseline_<loss>_recalc`` — per-period mean loss minus mean
     Rayleigh loss, keeping only rows inside the IQR window of that
     period's baseline loss.
  6. ``LastBaseline_<loss>_recalc_interp`` — the recalculated baseline
     linearly interpolated across measurement rows; the first and last
     periods keep the stepwise value.
  7. ``concentration_<species>_recalc`` / ``_interp`` — corrected loss
     converted to a mixing ratio at STP via the span factor.

Baseline period detection always uses the instrument-level status column;
the per-cell status column only selects the LED wavelength. (The R code
did the same, by hardcoding ``Status`` inside its helpers.)
"""

from __future__ import annotations

from typing import Any

import polars as pl

STP_TEMP = 273.15  # K
STP_PRES = 760.0  # Torr

# Last digit of the status value encodes the cell's LED wavelength (nm).
LED_WAVELENGTHS = {"2": 365, "3": 405, "4": 450, "5": 530, "6": 630, "7": 660, "8": 780}
# Rayleigh scattering loss of air at STP (Mm^-1), by wavelength.
RAYLEIGH_CONSTANTS = {365: 64.3, 405: 42.4, 450: 27.6, 530: 14.1, 630: 6.96, 660: 5.98, 780: 3.07}

DEFAULT_BASELINE_STATUS_PREFIX = "32"
DEFAULT_BASELINE_SD_FILTER = 0.3  # Mm^-1
DEFAULT_IQR_MULTIPLIER = 1.5


class BaselineError(ValueError):
    """Raised when a file cannot be baseline-recalculated."""


def _is_baseline(status_col: str, prefix: str) -> pl.Expr:
    return pl.col(status_col).cast(pl.String).str.starts_with(prefix).fill_null(False)


def find_led_color(df: pl.DataFrame, status_col: str) -> int:
    status = df[status_col].drop_nulls()
    if not status.len():
        raise BaselineError(f"Status column {status_col!r} has no values")
    code = str(status[0])[-1]
    if code not in LED_WAVELENGTHS:
        raise BaselineError(
            f"Unrecognized LED code {code!r} (last digit of {status_col}={status[0]})"
        )
    return LED_WAVELENGTHS[code]


def assign_rayleigh(
    df: pl.DataFrame, status_col: str, temperature_col: str, pressure_col: str
) -> tuple[pl.DataFrame, str]:
    led_color = find_led_color(df, status_col)
    rayleigh_col = f"rayleigh_{led_color}"
    expr = RAYLEIGH_CONSTANTS[led_color] * (pl.col(pressure_col) / STP_PRES) * (
        STP_TEMP / pl.col(temperature_col)
    )
    return df.with_columns(expr.alias(rayleigh_col)), rayleigh_col


def assign_baseline_period(df: pl.DataFrame, status_col: str, prefix: str) -> pl.DataFrame:
    return df.with_columns(_is_baseline(status_col, prefix).cast(pl.Int8).alias("baseline_period"))


def assign_baseline_number(df: pl.DataFrame) -> pl.DataFrame:
    if df["baseline_period"].sum() == 0:
        raise BaselineError("No usable baseline periods found")
    bp = pl.col("baseline_period")
    entering = (bp == 1) & (bp.shift(1).fill_null(0) == 0)
    return df.with_columns(
        entering.cast(pl.UInt32).cum_sum().clip(lower_bound=1).alias("baseline_number")
    )


def flag_bad_baselines(
    df: pl.DataFrame, loss_col: str, species: str, sd_filter: float
) -> pl.DataFrame:
    """Flag each baseline period good/bad by the SD of its loss.

    Adds ``sd_loss_baseline_<loss>`` (per-period SD broadcast to rows) and
    ``baseline_good_<species>`` (True/False on baseline rows, null elsewhere).
    Bad periods stay in the data — only the recalculation math skips them.
    """
    sd_col = f"sd_loss_baseline_{loss_col}"
    flag_col = f"baseline_good_{species}"
    sds = (
        df.filter(pl.col("baseline_period") == 1)
        .group_by("baseline_number")
        .agg(pl.col(loss_col).std().alias(sd_col))
    )
    df = df.drop(sd_col, flag_col, strict=False).join(sds, on="baseline_number", how="left")
    return df.with_columns(
        pl.when(pl.col("baseline_period") == 1)
        .then((pl.col(sd_col) < sd_filter).fill_null(False))
        .otherwise(None)
        .alias(flag_col)
    )


def recalculate_baseline_loss(
    df: pl.DataFrame, loss_col: str, rayleigh_col: str, iqr_multiplier: float
) -> pl.DataFrame:
    out_col = f"LastBaseline_{loss_col}_recalc"
    q25 = pl.col(loss_col).quantile(0.25, interpolation="linear")
    q75 = pl.col(loss_col).quantile(0.75, interpolation="linear")
    iqr = q75 - q25
    thresholds = (
        df.filter(pl.col("baseline_period") == 1)
        .group_by("baseline_number")
        .agg(
            (q25 - iqr_multiplier * iqr).alias("min_threshold"),
            (q75 + iqr_multiplier * iqr).alias("max_threshold"),
        )
    )
    # As in the R version, the IQR window filters every row in the group,
    # not just baseline rows; measurement rows only survive it when their
    # loss falls inside the baseline's own spread.
    means = (
        df.join(thresholds, on="baseline_number", how="left")
        .filter(pl.col(loss_col).is_between(pl.col("min_threshold"), pl.col("max_threshold")))
        .group_by("baseline_number")
        .agg((pl.col(loss_col).mean() - pl.col(rayleigh_col).mean()).alias(out_col))
    )
    return df.drop(out_col, strict=False).join(means, on="baseline_number", how="left")


def baseline_interpolation(df: pl.DataFrame, baseline_col: str) -> pl.DataFrame:
    keep = (
        (pl.col("baseline_period") == 1)
        | (pl.col("baseline_number") == 1)
        | (pl.col("baseline_number") == pl.col("baseline_number").max())
    )
    return df.with_columns(
        pl.when(keep)
        .then(pl.col(baseline_col))
        .otherwise(None)
        .interpolate()
        .alias(f"{baseline_col}_interp")
    )


def recalculate_concentration(
    df: pl.DataFrame,
    loss_col: str,
    span_col: str,
    rayleigh_col: str,
    temperature_col: str,
    pressure_col: str,
    species: str,
    interp: bool,
) -> pl.DataFrame:
    baseline_col = f"LastBaseline_{loss_col}_recalc" + ("_interp" if interp else "")
    name = f"concentration_{species}_" + ("interp" if interp else "recalc")
    conc = (
        ((pl.col(loss_col) - pl.col(rayleigh_col)) - pl.col(baseline_col))
        / pl.col(span_col)
        * (STP_PRES / pl.col(pressure_col) * pl.col(temperature_col) / STP_TEMP)
    )
    return df.with_columns(conc.alias(name))


def baseline_recalc(
    df: pl.DataFrame,
    *,
    status_col: str,
    loss_col: str,
    span_col: str,
    temperature_col: str,
    pressure_col: str,
    species: str,
    led_status_col: str | None = None,
    prefix: str = DEFAULT_BASELINE_STATUS_PREFIX,
    sd_filter: float = DEFAULT_BASELINE_SD_FILTER,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> pl.DataFrame:
    """Run the full recalculation pipeline for one measurement cell."""
    df = assign_baseline_period(df, status_col, prefix)
    df = assign_baseline_number(df)
    flag_col = f"baseline_good_{species}"
    df = flag_bad_baselines(df, loss_col, species, sd_filter)

    # The recalculation math sees only good baselines (matching the R code,
    # which demoted bad periods outright); the all-baseline period/number
    # columns are restored afterwards so flagged periods stay inspectable.
    all_period = df.get_column("baseline_period")
    all_number = df.get_column("baseline_number")
    df = df.with_columns(
        (pl.col("baseline_period") * pl.col(flag_col).fill_null(False).cast(pl.Int8)).alias(
            "baseline_period"
        )
    )
    try:
        df = assign_baseline_number(df)
    except BaselineError:
        raise BaselineError(
            f"All {species} baseline periods were flagged bad "
            f"(loss SD >= {sd_filter:g} Mm^-1); nothing to recalculate from"
        ) from None
    df, rayleigh_col = assign_rayleigh(df, led_status_col or status_col, temperature_col, pressure_col)
    df = recalculate_baseline_loss(df, loss_col, rayleigh_col, iqr_multiplier)
    df = recalculate_concentration(
        df, loss_col, span_col, rayleigh_col, temperature_col, pressure_col, species, interp=False
    )
    df = baseline_interpolation(df, f"LastBaseline_{loss_col}_recalc")
    df = recalculate_concentration(
        df, loss_col, span_col, rayleigh_col, temperature_col, pressure_col, species, interp=True
    )
    return df.with_columns(all_period, all_number)


def baseline_period_stats(
    df: pl.DataFrame, loss_col: str, sd_filter: float, iqr_multiplier: float
) -> pl.DataFrame:
    """Per-baseline-period diagnostics (SD, IQR window, filtered mean, verdict).

    Operates on a dataframe already processed by baseline_recalc; one row per
    baseline period, sorted by baseline_number.
    """
    base = df.filter(pl.col("baseline_period") == 1)
    if base.is_empty():
        raise BaselineError("No baseline periods found")
    q25 = pl.col(loss_col).quantile(0.25, interpolation="linear")
    q75 = pl.col(loss_col).quantile(0.75, interpolation="linear")
    iqr = q75 - q25
    aggs = [
        pl.len().alias("n"),
        pl.col(loss_col).mean().alias("mean_loss"),
        pl.col(loss_col).std().alias("sd_loss"),
        (q25 - iqr_multiplier * iqr).alias("min_threshold"),
        (q75 + iqr_multiplier * iqr).alias("max_threshold"),
    ]
    if "Timestamp" in base.columns:
        aggs.append(pl.col("Timestamp").min().alias("start"))
    stats = base.group_by("baseline_number").agg(aggs)
    in_window = (
        base.join(
            stats.select("baseline_number", "min_threshold", "max_threshold"),
            on="baseline_number",
        )
        .filter(pl.col(loss_col).is_between(pl.col("min_threshold"), pl.col("max_threshold")))
        .group_by("baseline_number")
        .agg(pl.col(loss_col).mean().alias("mean_loss_filtered"))
    )
    return (
        stats.join(in_window, on="baseline_number", how="left")
        .with_columns((pl.col("sd_loss") < sd_filter).fill_null(False).alias("good"))
        .sort("baseline_number")
    )


def apply_baseline_recalc(
    df: pl.DataFrame, config: dict[str, Any] | None, sd_filter: float | None = None
) -> pl.DataFrame:
    """Run baseline_recalc for every cell declared in the instrument config."""
    settings = (config or {}).get("Baseline_Recalculation") or {}
    cells = settings.get("Cells")
    if not cells:
        raise BaselineError("Config has no Baseline_Recalculation.Cells section")
    missing = [
        c
        for c in (settings.get("Status_col"), settings.get("Temperature_col"), settings.get("Pressure_col"))
        if c and c not in df.columns
    ]
    if missing:
        raise BaselineError(f"Columns not found in data: {missing}")

    for species, cell in cells.items():
        df = baseline_recalc(
            df,
            status_col=settings["Status_col"],
            led_status_col=cell.get("Status_col"),
            loss_col=cell["Loss_col"],
            span_col=cell["Span_col"],
            temperature_col=settings["Temperature_col"],
            pressure_col=settings["Pressure_col"],
            species=species,
            prefix=str(settings.get("Baseline_status_prefix", DEFAULT_BASELINE_STATUS_PREFIX)),
            sd_filter=float(
                sd_filter
                if sd_filter is not None
                else settings.get("Baseline_sd_filter", DEFAULT_BASELINE_SD_FILTER)
            ),
            iqr_multiplier=float(settings.get("IQR_multiplier", DEFAULT_IQR_MULTIPLIER)),
        )
    return df
