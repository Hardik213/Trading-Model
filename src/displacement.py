from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd

class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"

@dataclass(frozen=True)
class DisplacementEvent:
    timestamp: pd.Timestamp
    direction: Direction
    start_price: float
    end_price: float
    magnitude: float
    body_ratio: float
    close_location: float
    range_expansion: Optional[float]
    follow_through: bool
    structural_consequence: bool = False
    confirmation_timestamp: Optional[pd.Timestamp] = None


def _value(row, *names):
    for name in names:
        if name in row.index:
            return row[name]
    lower = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if str(name).lower() in lower:
            return lower[str(name).lower()]
    raise KeyError(names[0])


def candle_metrics(row):
    high = float(_value(row, "High", "high"))
    low = float(_value(row, "Low", "low"))
    open_ = float(_value(row, "Open", "open"))
    close = float(_value(row, "Close", "close"))
    rng = high - low
    if rng <= 0:
        return 0.0, 0.5, 0.0
    return rng, (close - low) / rng, abs(close - open_) / rng


def detect_displacement(df, position, direction=None, *, baseline_bars=20, min_body_ratio=0.60, min_range_expansion=1.25, follow_through_bars=2):
    if direction is None:
        raise ValueError("direction is required")
    if isinstance(direction, str):
        direction = Direction.BULLISH if direction.upper() in {"LONG", "BULLISH"} else Direction.BEARISH
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("OHLC index must be a DatetimeIndex.")
    if position < 0 or position >= len(df):
        raise IndexError("position outside dataframe.")
    if baseline_bars < 2:
        raise ValueError("baseline_bars must be >= 2.")
    if follow_through_bars < 0:
        raise ValueError("follow_through_bars must be >= 0.")
    row = df.iloc[position]
    rng, close_location, body_ratio = candle_metrics(row)
    if rng <= 0:
        return None
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]
    baseline = (high - low).iloc[max(0, position - baseline_bars):position]
    if len(baseline) < 2:
        return None
    median_range = float(baseline.median())
    if median_range <= 0:
        return None
    expansion = rng / median_range
    open_ = float(_value(row, "Open", "open"))
    close = float(_value(row, "Close", "close"))
    directional = close > open_ if direction is Direction.BULLISH else close < open_
    quality = close_location >= 0.70 if direction is Direction.BULLISH else close_location <= 0.30
    if not directional or body_ratio < min_body_ratio or expansion < min_range_expansion or not quality:
        return None
    follow = False
    confirmation = None
    if follow_through_bars:
        future = df.iloc[position + 1:min(len(df), position + 1 + follow_through_bars)]
        for idx, r in future.iterrows():
            r_high = float(_value(r, "High", "high"))
            r_low = float(_value(r, "Low", "low"))
            ok = r_high > float(_value(row, "High", "high")) if direction is Direction.BULLISH else r_low < float(_value(row, "Low", "low"))
            if ok:
                follow = True
                confirmation = pd.Timestamp(idx)
                break
    else:
        follow = True
        confirmation = pd.Timestamp(df.index[position])
    return DisplacementEvent(pd.Timestamp(df.index[position]), direction, open_, close, abs(close - open_), body_ratio, close_location, expansion, follow, False, confirmation)


def is_confirmed_as_of(event: Optional[DisplacementEvent], as_of: pd.Timestamp) -> bool:
    return event is not None and event.follow_through and event.confirmation_timestamp is not None and pd.Timestamp(event.confirmation_timestamp) <= pd.Timestamp(as_of)
