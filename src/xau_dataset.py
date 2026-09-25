"""Deterministic local XAUUSD dataset manager for free Dukascopy CSV chunks."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .dukascopy_csv import read_many, quality_report, to_ohlc


TIMEFRAMES = ("1min", "3min", "5min", "15min", "1h", "4h", "1D", "1W")


@dataclass(frozen=True)
class DatasetBuildReport:
    source: str
    files: int
    raw_rows: int
    start: str
    end: str
    duplicate_timestamps: int
    non_monotonic_rows: int
    invalid_ohlc_rows: int
    bars: dict[str, int]


def discover_csv(root: str | Path) -> list[Path]:
    """Return CSV files recursively, sorted deterministically."""
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(p)
    return sorted(x for x in p.rglob("*.csv") if x.is_file())


def load_raw(paths: Iterable[str | Path]) -> pd.DataFrame:
    paths = list(paths)
    if not paths:
        raise ValueError("No CSV files supplied")
    return read_many(paths)


def build_timeframes(
    raw: pd.DataFrame,
    out_dir: str | Path,
    *,
    source: str = "dukascopy",
) -> DatasetBuildReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = quality_report(raw, files=1)
    bars = {}
    for tf in TIMEFRAMES:
        frame = to_ohlc(raw, tf)
        name = tf.upper().replace("MIN", "M").replace("H", "H")
        path = out / f"XAUUSD_{name}_BID.csv"
        frame.to_csv(path, index=False)
        bars[tf] = len(frame)

    metadata = {
        "instrument": "XAUUSD",
        "source": source,
        "price_side": "BID",
        "raw_rows": len(raw),
        "raw_start": report["start"],
        "raw_end": report["end"],
        "raw_duplicate_timestamps": report["duplicate_timestamps"],
        "raw_non_monotonic_rows": report["non_monotonic_rows"],
        "raw_invalid_ohlc_rows": report["invalid_ohlc_rows"],
        "timeframes": bars,
        "contract": "Raw observations are preserved; bars are deterministic resamples.",
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return DatasetBuildReport(
        source=source,
        files=1,
        raw_rows=len(raw),
        start=report["start"],
        end=report["end"],
        duplicate_timestamps=report["duplicate_timestamps"],
        non_monotonic_rows=report["non_monotonic_rows"],
        invalid_ohlc_rows=report["invalid_ohlc_rows"],
        bars=bars,
    )
