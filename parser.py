"""Parsing utilities for Aerodyne CAPS instrument output files.

CAPS .dat files have three sections, all of which are needed to make sense
of a file:
  1. Descriptive header lines, e.g. "% Serial Number: 625002"
  2. A positional block of numeric tokens (also "%"-prefixed) that encodes
     instrument configuration. Field position is meaningful — e.g. token 32
     (1-indexed) is the instrument serial number, whose leading digit tells
     us the instrument type.
  3. A CSV table of timestamped measurements.

This mirrors the logic in the original R dashboard's caps_functions.R /
caps_info.R, translated to Python.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Union

import polars as pl
import yaml

CONFIG_DIR = Path(__file__).parent / "config"

# First digit of the serial number indicates instrument type.
INSTRUMENT_TYPES = {
    "1": "no2",
    "2": "extinction",
    "3": "ssa",
    "6": "nox",
}

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S%.f"

SourceType = Union[str, Path, BinaryIO]


@dataclass
class CapsFile:
    name: str
    instrument_type: str | None
    header_info: dict[str, str]
    parameters: list[str]
    data: pl.DataFrame
    config: dict[str, Any] | None


def _load_bytes(source: SourceType) -> tuple[bytes, str]:
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
        name = getattr(source, "name", "uploaded_file")
        return data, name
    path = Path(source)
    return path.read_bytes(), path.name


def _parse_header_info(comment_lines: list[str]) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in comment_lines:
        text = line.lstrip("%").strip()
        if not text:
            continue
        if ":" in text:
            key, _, value = text.partition(":")
        elif "=" in text:
            key, _, value = text.partition("=")
        else:
            continue
        key, value = key.strip(), value.strip()
        if key and value:
            info[key] = value
    return info


def _parse_parameter_block(lines: list[str]) -> list[str]:
    """Flatten the numeric metadata block into an ordered list of raw tokens.

    Kept as strings (not floats) because a handful of positions in some
    instrument files hold dates rather than numbers, and dropping those
    tokens would shift every position after them.
    """
    comment_lines = [l for l in lines if l.lstrip().startswith("%")]
    start_idx = next(
        (i for i, l in enumerate(comment_lines) if "exact sample time" in l.lower()),
        None,
    )
    if start_idx is None:
        return []
    # The CSV header row is the first non-comment line; everything before it
    # is contiguous "%" lines, so this index lines up with comment_lines too.
    header_idx = next(
        (i for i, l in enumerate(lines) if "igor" in l.lower()),
        len(lines),
    )
    block = comment_lines[start_idx + 1 : header_idx]
    tokens: list[str] = []
    for line in block:
        tokens.extend(line.lstrip("%").split())
    return tokens


def detect_instrument_type(parameters: list[str]) -> str | None:
    if len(parameters) < 32:
        return None
    serial = parameters[31]
    return INSTRUMENT_TYPES.get(serial[0]) if serial else None


def load_config(instrument_type: str) -> dict[str, Any] | None:
    path = CONFIG_DIR / f"{instrument_type}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def read_caps_file(source: SourceType) -> CapsFile:
    raw_bytes, name = _load_bytes(source)
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    comment_lines = [l for l in lines if l.lstrip().startswith("%")]
    header_info = _parse_header_info(comment_lines)
    parameters = _parse_parameter_block(lines)
    instrument_type = detect_instrument_type(parameters)
    config = load_config(instrument_type) if instrument_type else None

    df = pl.read_csv(io.BytesIO(raw_bytes), comment_prefix="%", infer_schema_length=10000)
    if "Timestamp" in df.columns:
        df = df.with_columns(
            pl.col("Timestamp").str.strptime(pl.Datetime, TIMESTAMP_FORMAT, strict=False)
        )

    return CapsFile(
        name=name,
        instrument_type=instrument_type,
        header_info=header_info,
        parameters=parameters,
        data=df,
        config=config,
    )
