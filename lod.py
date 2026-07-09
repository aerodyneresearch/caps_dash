"""Limit-of-detection analysis for CAPS files.

LOD is defined as 3x the standard deviation of the background (zero-air)
signal. Two views, both fed by a dataframe already processed by
baseline.apply_baseline_recalc:

- Per-baseline LOD time series: the SD of concentration within each baseline
  period gives one LOD estimate at that time.
- Allan-Werle deviation: for long zero-air segments, the Allan deviation of
  concentration versus averaging time shows how far averaging improves the
  LOD before instrument drift takes over (Werle et al., Appl. Phys. B 57,
  131, 1993).
"""

from __future__ import annotations

import numpy as np
import polars as pl

LOD_SIGMA_FACTOR = 3.0


def baseline_lod_series(
    df: pl.DataFrame, conc_col: str, min_points: int = 3
) -> pl.DataFrame:
    """LOD (3*SD of conc_col) per baseline period; one row per period.

    Periods with fewer than min_points samples are dropped — a two-point SD
    is not an LOD estimate.
    """
    base = df.filter(pl.col("baseline_period") == 1)
    aggs = [pl.len().alias("n"), pl.col(conc_col).std().alias("sd")]
    if "Timestamp" in base.columns:
        aggs.append(pl.col("Timestamp").min().alias("start"))
    return (
        base.group_by("baseline_number")
        .agg(aggs)
        .filter(pl.col("n") >= min_points)
        .with_columns((LOD_SIGMA_FACTOR * pl.col("sd")).alias("lod"))
        .sort("baseline_number")
    )


def allan_deviation(values: np.ndarray, dt: float, n_taus: int = 40) -> pl.DataFrame:
    """Non-overlapping Allan deviation of an evenly sampled series.

    Averaging windows are log-spaced from one sample up to a quarter of the
    record, so the longest averaging time still spans at least four windows.
    Returns columns tau (s), adev (signal units), n_windows.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 16:
        raise ValueError(f"Need at least 16 samples for an Allan deviation, got {n}")
    m_max = n // 4
    ms = np.unique(np.round(np.logspace(0, np.log10(m_max), n_taus)).astype(int))
    taus, adevs, counts = [], [], []
    for m in ms:
        bins = n // m
        means = x[: bins * m].reshape(bins, m).mean(axis=1)
        diffs = np.diff(means)
        adevs.append(float(np.sqrt(0.5 * np.mean(diffs**2))))
        taus.append(float(m * dt))
        counts.append(bins)
    return pl.DataFrame({"tau": taus, "adev": adevs, "n_windows": counts})
