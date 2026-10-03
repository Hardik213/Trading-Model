from __future__ import annotations

"""Strict, bounded-memory ingestion for Dukascopy XAUUSD quote ticks.

The adapter preserves source order and duplicates. It does not sort, deduplicate,
repair, or silently skip rows. Output batches use the ``timestamp,bid,ask``
interface expected by ``TickAggregator`` and retain source provenance alongside
canonical values.
"""

import csv
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .tick_aggregation import validate_tick_frame


RAW_TICK_COLUMNS = ("timestamp", "askPrice", "bidPrice")
TICK_BATCH_COLUMNS = (
    "timestamp",
    "bid",
    "ask",
    "source_file",
    "source_row",
    "source_line",
    "source_order",
    "raw_timestamp_ms",
    "raw_askPrice",
    "raw_bidPrice",
)
_INTEGER_MILLISECONDS = re.compile(r"^[0-9]+$")
DEFAULT_CHUNK_SIZE = 65_536


def discover_dukascopy_tick_files(root: str | Path) -> list[Path]:
    """Discover named monthly Dukascopy quote-tick files in lexical order.

    Record order inside files is never changed. Explicit path iterables passed to
    :func:`iter_dukascopy_ticks` retain the caller's supplied order instead.
    """
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return (
            [path]
            if path.suffix.lower() == ".csv" and path.name.lower().startswith("xauusd-tick-")
            else []
        )
    return sorted(
        candidate
        for candidate in path.rglob("*.csv")
        if candidate.is_file() and candidate.name.lower().startswith("xauusd-tick-")
    )


def _canonical_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        selected = discover_dukascopy_tick_files(path)
    else:
        selected = [Path(path) for path in paths]
    if not selected:
        raise ValueError("No Dukascopy tick CSV files supplied.")
    return selected


def _row_error(path: Path, row: int, line: int, message: str) -> ValueError:
    return ValueError(f"{path}: row {row} (physical line {line}): {message}")


def _build_batch(
    records: list[dict],
    *,
    previous_timestamp: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    raw = pd.DataFrame.from_records(records)
    timestamp_text = raw["raw_timestamp_ms"].astype("string")
    valid_timestamp_syntax = timestamp_text.str.fullmatch(_INTEGER_MILLISECONDS)
    if not valid_timestamp_syntax.all():
        position = int(np.flatnonzero(~valid_timestamp_syntax.to_numpy())[0])
        record = records[position]
        raise _row_error(
            Path(record["source_file"]), record["source_row"], record["source_line"],
            "timestamp must be a non-negative integer Unix-millisecond value",
        )

    numeric_timestamps = pd.to_numeric(timestamp_text, errors="coerce")
    try:
        timestamps = pd.to_datetime(
            numeric_timestamps.astype("int64"), unit="ms", utc=True, errors="coerce"
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Tick timestamp is outside the supported UTC range.") from exc
    valid_timestamps = ~timestamps.isna()
    if not valid_timestamps.all():
        position = int(np.flatnonzero(~valid_timestamps.to_numpy())[0])
        record = records[position]
        raise _row_error(
            Path(record["source_file"]), record["source_row"], record["source_line"],
            f"invalid Unix-millisecond timestamp {record['raw_timestamp_ms']!r}",
        )

    ask = pd.to_numeric(raw["raw_askPrice"], errors="coerce").to_numpy(dtype=float)
    bid = pd.to_numeric(raw["raw_bidPrice"], errors="coerce").to_numpy(dtype=float)
    valid_quotes = np.isfinite(ask) & np.isfinite(bid)
    if not valid_quotes.all():
        position = int(np.flatnonzero(~valid_quotes)[0])
        record = records[position]
        raise _row_error(
            Path(record["source_file"]), record["source_row"], record["source_line"],
            "askPrice and bidPrice must be numeric and finite",
        )
    positive_quotes = (ask > 0.0) & (bid > 0.0)
    if not positive_quotes.all():
        position = int(np.flatnonzero(~positive_quotes)[0])
        record = records[position]
        raise _row_error(
            Path(record["source_file"]), record["source_row"], record["source_line"],
            "askPrice and bidPrice must be positive",
        )
    uncrossed_quotes = ask >= bid
    if not uncrossed_quotes.all():
        position = int(np.flatnonzero(~uncrossed_quotes)[0])
        record = records[position]
        raise _row_error(
            Path(record["source_file"]), record["source_row"], record["source_line"],
            "crossed quote: askPrice is below bidPrice",
        )

    timestamp_values = pd.DatetimeIndex(timestamps)
    boundary_decreasing = bool(
        len(timestamp_values)
        and previous_timestamp is not None
        and timestamp_values[0] < previous_timestamp
    )
    internal_decreases = np.flatnonzero(timestamp_values[1:] < timestamp_values[:-1])
    if boundary_decreasing or len(internal_decreases):
        if boundary_decreasing:
            position = 0
            prior = previous_timestamp
        else:
            position = int(internal_decreases[0]) + 1
            prior = timestamp_values[position - 1]
        record = records[position]
        raise _row_error(
            Path(record["source_file"]), record["source_row"], record["source_line"],
            f"decreasing timestamp {timestamp_values[position].isoformat()} after {prior.isoformat()}",
        )

    batch = pd.DataFrame(
        {
            "timestamp": timestamp_values,
            "bid": bid,
            "ask": ask,
            "source_file": raw["source_file"],
            "source_row": raw["source_row"].astype("int64"),
            "source_line": raw["source_line"].astype("int64"),
            "source_order": raw["source_order"].astype("int64"),
            "raw_timestamp_ms": timestamp_text,
            "raw_askPrice": raw["raw_askPrice"],
            "raw_bidPrice": raw["raw_bidPrice"],
        },
        columns=TICK_BATCH_COLUMNS,
    )
    validate_tick_frame(batch)
    return batch, pd.Timestamp(timestamp_values[-1])


def iter_dukascopy_ticks(
    paths: str | Path | Iterable[str | Path],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield validated canonical tick batches using bounded memory.

    Input CSV schema is exactly ``timestamp,askPrice,bidPrice``. Timestamps are
    non-negative Unix milliseconds converted to timezone-aware UTC. The emitted
    columns are ``timestamp,bid,ask`` plus source file, 1-based data-row and
    physical-line identifiers, a global source ordinal, and the original raw
    field strings. Equal timestamps are retained in stable source order; a
    strictly decreasing timestamp anywhere in the ordered file stream is an
    error.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    selected_paths = _canonical_paths(paths)
    previous_timestamp: pd.Timestamp | None = None
    source_order = 0

    for path in selected_paths:
        records: list[dict] = []
        data_row = 0
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader, None)
                if header != list(RAW_TICK_COLUMNS):
                    raise ValueError(
                        f"{path}: expected exact schema {','.join(RAW_TICK_COLUMNS)!r}; "
                        f"got {header!r}."
                    )

                for fields in reader:
                    data_row += 1
                    physical_line = reader.line_num
                    if len(fields) != len(RAW_TICK_COLUMNS):
                        raise _row_error(
                            path,
                            data_row,
                            physical_line,
                            f"expected {len(RAW_TICK_COLUMNS)} fields, got {len(fields)}",
                        )

                    records.append(
                        {
                            "raw_timestamp_ms": fields[0],
                            "raw_askPrice": fields[1],
                            "raw_bidPrice": fields[2],
                            "source_file": str(path),
                            "source_row": data_row,
                            "source_line": physical_line,
                            "source_order": source_order,
                        }
                    )
                    source_order += 1

                    if len(records) == chunk_size:
                        batch, previous_timestamp = _build_batch(
                            records,
                            previous_timestamp=previous_timestamp,
                        )
                        yield batch
                        records = []
        except csv.Error as exc:
            raise ValueError(
                f"{path}: malformed CSV near physical line {reader.line_num}: {exc}"
            ) from exc

        if records:
            batch, previous_timestamp = _build_batch(
                records,
                previous_timestamp=previous_timestamp,
            )
            yield batch


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "RAW_TICK_COLUMNS",
    "TICK_BATCH_COLUMNS",
    "discover_dukascopy_tick_files",
    "iter_dukascopy_ticks",
]
