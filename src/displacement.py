from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class DisplacementEvent:
    timestamp: pd.Timestamp
    direction: Direction
    start_price: float
    end_price: float
    magnitude: float
    body_ratio: float
    close_location: float
    range_expansion: float
    follow_through: bool
    confirmation_timestamp: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction(self.direction))


def _normalize_direction(direction: Direction | str) -> Direction:
    if direction is None:
        raise ValueError("direction is required")
    if isinstance(direction, Direction):
        return direction
    text = str(direction).upper().split(".")[-1]
    if text in {"LONG", "BULLISH"}:
        return Direction.BULLISH
    if text in {"SHORT", "BEARISH"}:
        return Direction.BEARISH
    raise ValueError(f"Unsupported direction: {direction!r}")


def _column(frame, *names):
    for name in names:
        if name in frame.columns:
            return frame[name]
    lowered = {str(col).lower(): col for col in frame.columns}
    for name in names:
        if str(name).lower() in lowered:
            return frame[lowered[str(name).lower()]]
    raise KeyError(names[0])


def detect_displacement(
    df,
    position,
    direction,
    *,
    body_ratio: float = 0.6,
    range_multiplier: float = 1.5,
    min_range_expansion: Optional[float] = None,
    lookback: int = 5,
    follow_through_bars: int = 0,
):
    if df is None or len(df) == 0 or position < 0 or position >= len(df):
        return None

    direction = _normalize_direction(direction)
    effective_range_expansion = min_range_expansion if min_range_expansion is not None else range_multiplier

    row = df.iloc[position]
    high = float(_column(row.to_frame().T, "High", "high").iloc[0])
    low = float(_column(row.to_frame().T, "Low", "low").iloc[0])
    op = float(_column(row.to_frame().T, "Open", "open").iloc[0])
    close = float(_column(row.to_frame().T, "Close", "close").iloc[0])
    rng = high - low
    if rng <= 0:
        return None

    impulse_body = abs(close - op) / rng
    if impulse_body < body_ratio:
        return None

    prior = df.iloc[max(0, position - lookback):position]
    if len(prior) == 0:
        return None
    baseline = float((
        _column(prior, "High", "high").astype(float) -
        _column(prior, "Low", "low").astype(float)
    ).mean())
    if baseline <= 0 or rng < baseline * effective_range_expansion:
        return None

    if direction is Direction.BULLISH and close <= op:
        return None
    if direction is Direction.BEARISH and close >= op:
        return None

    if direction is Direction.BULLISH:
        start_price = min(op, close)
        end_price = max(op, close)
        impulse = high
        confirmation_price = high
        event_direction = "LONG"
    else:
        start_price = max(op, close)
        end_price = min(op, close)
        impulse = low
        confirmation_price = low
        event_direction = "SHORT"

    timestamp = pd.Timestamp(df.index[position])
    if follow_through_bars <= 0:
        return DisplacementEvent(
            timestamp=timestamp,
            direction=event_direction,
            start_price=start_price,
            end_price=end_price,
            magnitude=abs(close - op),
            body_ratio=impulse_body,
            close_location=(close - low) / rng if rng else 0.0,
            range_expansion=rng / max(baseline, 1e-9),
            follow_through=True,
            confirmation_timestamp=timestamp,
        )

    future = df.iloc[position + 1 : min(len(df), position + 1 + follow_through_bars)]
    confirmation = None
    for idx, r in future.iterrows():
        value = float(_column(r.to_frame().T, "High", "high").iloc[0]) if direction is Direction.BULLISH else float(_column(r.to_frame().T, "Low", "low").iloc[0])
        if direction is Direction.BULLISH and value > confirmation_price:
            confirmation = pd.Timestamp(idx)
            break
        if direction is Direction.BEARISH and value < confirmation_price:
            confirmation = pd.Timestamp(idx)
            break

    return DisplacementEvent(
        timestamp=timestamp,
        direction=event_direction,
        start_price=start_price,
        end_price=end_price,
        magnitude=abs(close - op),
        body_ratio=impulse_body,
        close_location=(close - low) / rng if rng else 0.0,
        range_expansion=rng / max(baseline, 1e-9),
        follow_through=confirmation is not None,
        confirmation_timestamp=confirmation,
    )


def is_confirmed_as_of(event, as_of):
    return (
        event is not None
        and event.follow_through
        and event.confirmation_timestamp is not None
        and pd.Timestamp(event.confirmation_timestamp) <= pd.Timestamp(as_of)
    )

