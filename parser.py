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
import re
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

# Logged .dat files use slashes; direct-instrument .log files use dashes.
TIMESTAMP_FORMATS = [
    "%Y/%m/%d %H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
]

# Direct-instrument logs write fractional seconds with a comma decimal
# separator ("00:00:00,256"), which collides with the CSV delimiter and makes
# every data row one field longer than the header.
_COMMA_FRACTION = re.compile(r"(\d{2}:\d{2}:\d{2}),(\d+)\s*$", re.MULTILINE)

SourceType = Union[str, Path, BinaryIO]


@dataclass
class CapsFile:
    name: str
    instrument_type: str | None
    header_info: dict[str, str]
    parameters: list[str]
    data: pl.DataFrame
    config: dict[str, Any] | None
    # How instrument_type was determined: "parameter_block" (authoritative),
    # "header_match" (exact), "column_similarity" (guess), or None (unknown).
    detection_method: str | None = None
    detection_note: str | None = None


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


def load_all_configs() -> dict[str, dict[str, Any]]:
    configs = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        with open(path) as f:
            configs[path.stem] = yaml.safe_load(f) or {}
    return configs


def _config_header_columns(config: dict[str, Any]) -> list[str]:
    header = config.get("Header") or ""
    return [c.strip() for c in str(header).split(",") if c.strip()]


def detect_from_header(
    columns: list[str], configs: dict[str, dict[str, Any]]
) -> tuple[str, str] | None:
    """Exact match of the file's column names against a config's Header."""
    cols = set(columns)
    matches = [
        itype
        for itype, cfg in configs.items()
        if cols and set(_config_header_columns(cfg)) == cols
    ]
    if len(matches) == 1:
        return matches[0], f"column names exactly match the {matches[0]} config header"
    return None


def detect_from_structure(
    columns: list[str], configs: dict[str, dict[str, Any]]
) -> tuple[str, str] | None:
    """Best-guess match on column structure when names don't match exactly.

    First by column-name overlap (Jaccard similarity, needs a unique best
    score of at least 0.5), then by a unique column-count match.
    """
    cols = set(columns)
    scored = sorted(
        (
            (len(cols & hdr) / len(cols | hdr), itype)
            for itype, cfg in configs.items()
            if (hdr := set(_config_header_columns(cfg)))
        ),
        reverse=True,
    )
    if scored and scored[0][0] >= 0.5 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        similarity, itype = scored[0]
        return itype, (
            f"{similarity:.0%} column-name overlap with the {itype} config header (best guess)"
        )
    count_matches = [
        itype
        for itype, cfg in configs.items()
        if _config_header_columns(cfg) and len(_config_header_columns(cfg)) == len(columns)
    ]
    if len(count_matches) == 1:
        return count_matches[0], (
            f"column count ({len(columns)}) matches only the {count_matches[0]} config "
            "(weak guess — column names did not match)"
        )
    return None


def read_caps_file(source: SourceType) -> CapsFile:
    raw_bytes, name = _load_bytes(source)
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    comment_lines = [l for l in lines if l.lstrip().startswith("%")]
    header_info = _parse_header_info(comment_lines)
    parameters = _parse_parameter_block(lines)

    # Blank lines must go: polars (1.42) takes a blank line that follows a
    # comment line as the schema row, breaking the whole read. Direct
    # instrument logs have exactly that between parameter block and header.
    csv_text = _COMMA_FRACTION.sub(r"\1.\2", text.replace("\r\n", "\n"))
    csv_text = "\n".join(line for line in csv_text.split("\n") if line.strip())
    df = pl.read_csv(
        io.BytesIO(csv_text.encode("utf-8")), comment_prefix="%", infer_schema_length=10000
    )
    if "Timestamp" in df.columns and df["Timestamp"].dtype == pl.String:
        df = df.with_columns(
            pl.coalesce(
                pl.col("Timestamp").str.strptime(pl.Datetime, fmt, strict=False)
                for fmt in TIMESTAMP_FORMATS
            ).alias("Timestamp")
        )

    # Detection chain: the metadata parameter block is authoritative; without
    # it, fall back to the CSV header, then to a structural best guess.
    detection_method = detection_note = None
    instrument_type = detect_instrument_type(parameters)
    if instrument_type:
        detection_method = "parameter_block"
        detection_note = (
            f"serial number {parameters[31]} at position 32 of the metadata parameter block"
        )
    else:
        configs = load_all_configs()
        if result := detect_from_header(df.columns, configs):
            instrument_type, detection_note = result
            detection_method = "header_match"
        elif result := detect_from_structure(df.columns, configs):
            instrument_type, detection_note = result
            detection_method = "column_similarity"

    config = load_config(instrument_type) if instrument_type else None

    # Direct-instrument files name some columns differently; the config's
    # Rename mapping normalizes them to the canonical names everything else
    # (Units, Baseline_Recalculation, plots) refers to.
    if config and (rename := config.get("Rename")):
        mapping = {
            src: dst
            for src, dst in rename.items()
            if src in df.columns and dst not in df.columns
        }
        if mapping:
            df = df.rename(mapping)

    return CapsFile(
        name=name,
        instrument_type=instrument_type,
        header_info=header_info,
        parameters=parameters,
        data=df,
        config=config,
        detection_method=detection_method,
        detection_note=detection_note,
    )
