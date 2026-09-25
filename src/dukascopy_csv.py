"""Dukascopy Historical Data Export CSV ingestion and deterministic OHLC aggregation.

Designed for the free Dukascopy CSV exporter path. It does not download data,
alter raw observations, deduplicate ticks, or fabricate missing intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DataQuality:
    files: int
    rows: int
    duplicate_timestamps: int
    non_monotonic_rows: int
    invalid_ohlc_rows: int
    timezone: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in {"etc/utc", "timestamp", "time", "datetime", "date"}:
            mapping[c] = "timestamp"
        elif key == "open":
            mapping[c] = "open"
        elif key == "high":
            mapping[c] = "high"
        elif key == "low":
            mapping[c] = "low"
        elif key == "close":
            mapping[c] = "close"
        elif key == "volume":
            mapping[c] = "volume"

    out = df.rename(columns=mapping).copy()
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return out[list(REQUIRED)]


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read one exported Dukascopy CSV without dropping duplicate timestamps."""
    df = _normalize_columns(pd.read_csv(path))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    for c in REQUIRED[1:]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    df["_source_row"] = range(len(df))
    return df


def read_many(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [read_csv(p) for p in paths]
    if not frames:
        raise ValueError("No CSV files supplied")

    out = pd.concat(frames, ignore_index=True)
    out["_file_order"] = out.index
    out = out.sort_values(
        ["timestamp", "_file_order", "_source_row"],
        kind="stable",
    ).reset_index(drop=True)
    return out.drop(columns=["_file_order", "_source_row"])


def validate(df: pd.DataFrame) -> DataQuality:
    if df.empty:
        raise ValueError("Dataset is empty")
    if str(df["timestamp"].dt.tz) not in {"UTC", "UTC+00:00"}:
        raise ValueError("Timestamps must be timezone-aware UTC")

    invalid_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    non_monotonic = int((df["timestamp"].diff().dropna() < pd.Timedelta(0)).sum())
    duplicates = int(df["timestamp"].duplicated(keep=False).sum())

    return DataQuality(
        files=1,
        rows=len(df),
        duplicate_timestamps=duplicates,
        non_monotonic_rows=non_monotonic,
        invalid_ohlc_rows=int(invalid_ohlc.sum()),
        timezone="UTC",
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
    )


def to_ohlc(df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """Aggregate raw observations into deterministic UTC OHLCV bars.

    Duplicate timestamps are intentionally preserved before resampling.
    Open/close follow chronological source order; high/low span all observations.
    """
    if df.empty:
        raise ValueError("Cannot aggregate an empty dataset")
    d = df.set_index("timestamp").sort_index(kind="stable")
    bars = d.resample(timeframe, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    bars = bars.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return bars


def quality_report(df: pd.DataFrame, files: int = 1) -> dict:
    q = validate(df)
    return {
        "files": files,
        "rows": q.rows,
        "duplicate_timestamps": q.duplicate_timestamps,
        "non_monotonic_rows": q.non_monotonic_rows,
        "invalid_ohlc_rows": q.invalid_ohlc_rows,
        "timezone": q.timezone,
        "start": q.start.isoformat() if q.start is not None else None,
        "end": q.end.isoformat() if q.end is not None else None,
    }
