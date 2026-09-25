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


def candle_metrics(row: pd.Series) -> tuple[float, float, float]:
    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row["Open"])
    close = float(row["Close"])

    rng = high - low
    if rng <= 0:
        return 0.0, 0.5, 0.0

    body = abs(close - open_)
    body_ratio = body / rng
    close_location = (close - low) / rng
    return rng, close_location, body_ratio


def detect_displacement(
    df: pd.DataFrame,
    position: int,
    *,
    direction: Direction,
    baseline_bars: int = 20,
    min_body_ratio: float = 0.60,
    min_range_expansion: float = 1.25,
    follow_through_bars: int = 2,
) -> Optional[DisplacementEvent]:
    """
    Objective first-pass displacement observation.

    This is intentionally NOT a trade signal and does not assign a confidence
    score. Thresholds are explicit research parameters and must be validated
    later. The event is strengthened by follow-through but is never inferred
    from a wick alone.
    """
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

    start = max(0, position - baseline_bars)
    baseline = (df["High"] - df["Low"]).iloc[start:position]
    if len(baseline) < 2:
        return None

    median_range = float(baseline.median())
    if median_range <= 0:
        return None

    expansion = rng / median_range
    open_ = float(row["Open"])
    close = float(row["Close"])

    if direction is Direction.BULLISH:
        directional = close > open_
        close_quality = close_location >= 0.70
    else:
        directional = close < open_
        close_quality = close_location <= 0.30

    if not directional:
        return None
    if body_ratio < min_body_ratio:
        return None
    if expansion < min_range_expansion:
        return None
    if not close_quality:
        return None

    follow_through = False
    if follow_through_bars:
        end = min(len(df), position + 1 + follow_through_bars)
        future = df.iloc[position + 1:end]
        if len(future):
            if direction is Direction.BULLISH:
                follow_through = float(future["High"].max()) > float(row["High"])
            else:
                follow_through = float(future["Low"].min()) < float(row["Low"])

    return DisplacementEvent(
        timestamp=df.index[position],
        direction=direction,
        start_price=open_,
        end_price=close,
        magnitude=abs(close - open_),
        body_ratio=body_ratio,
        close_location=close_location,
        range_expansion=expansion,
        follow_through=follow_through,
    )
