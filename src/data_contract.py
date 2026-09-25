from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MarketBar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    timeframe: str
    source: str = "UNKNOWN"

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "Open": self.open,
            "High": self.high,
            "Low": self.low,
            "Close": self.close,
            "timeframe": self.timeframe,
            "source": self.source,
        }


def normalize_ohlc(
    df: pd.DataFrame,
    *,
    timeframe: str,
    source: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Normalize external OHLC into the internal UTC contract.

    The source must provide OHLC. The function does not synthesize missing
    prices and does not silently localize naive timestamps.
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    out = df[["Open", "High", "Low", "Close"]].copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("Market data index must be a DatetimeIndex.")

    if out.index.tz is None:
        raise ValueError(
            "Naive timestamps are not accepted. Localize the source explicitly before normalization."
        )

    out.index = out.index.tz_convert("UTC")
    out = out.sort_index()

    if out.index.has_duplicates:
        raise ValueError("Market data contains duplicate timestamps.")

    if not out.index.is_monotonic_increasing:
        raise ValueError("Market data must be sorted ascending.")

    if out[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("Market data contains missing OHLC values.")

    if (out["High"] < out[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("Invalid OHLC: High is below Open or Close.")

    if (out["Low"] > out[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC: Low is above Open or Close.")

    out.attrs["timeframe"] = timeframe
    out.attrs["source"] = source
    out.attrs["timezone"] = "UTC"
    return out


def as_market_bars(df: pd.DataFrame) -> Iterable[MarketBar]:
    timeframe = str(df.attrs.get("timeframe", "UNKNOWN"))
    source = str(df.attrs.get("source", "UNKNOWN"))

    for timestamp, row in df.iterrows():
        yield MarketBar(
            timestamp=timestamp,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            timeframe=timeframe,
            source=source,
        )
