from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .displacement import Direction, DisplacementEvent
from .market_structure import (
    BreachOutcome,
    LiquidityEvent,
    SwingPoint,
    SwingType,
)


@dataclass(frozen=True)
class MSSEvent:
    timestamp: pd.Timestamp
    direction: Direction
    structural_point: SwingPoint
    liquidity_event: LiquidityEvent
    displacement: DisplacementEvent
    break_price: float
    follow_through: bool


def _candidate_structure(
    swings: Iterable[SwingPoint],
    *,
    direction: Direction,
    before: pd.Timestamp,
    after_confirmation: pd.Timestamp,
) -> Optional[SwingPoint]:
    candidates = []
    for swing in swings:
        if swing.confirmation_timestamp > after_confirmation:
            continue
        if swing.timestamp >= before:
            continue

        if direction is Direction.BULLISH and swing.kind is SwingType.HIGH:
            candidates.append(swing)
        elif direction is Direction.BEARISH and swing.kind is SwingType.LOW:
            candidates.append(swing)

    if not candidates:
        return None

    # Nearest previously confirmed opposing structural point.
    return max(candidates, key=lambda s: s.timestamp)


def detect_mss(
    df: pd.DataFrame,
    *,
    swings: Iterable[SwingPoint],
    liquidity_event: LiquidityEvent,
    displacement: DisplacementEvent,
    break_position: int,
    follow_through_bars: int = 2,
) -> Optional[MSSEvent]:
    """
    Detect a structural market-structure shift.

    Requirements:
      1. A liquidity event must have resolved as REJECTION.
      2. Displacement must point in the reversal direction.
      3. A meaningful, previously confirmed structural point must be broken.
      4. Follow-through must extend beyond the break candle.

    This deliberately rejects a simple rolling close breach as MSS.
    """
    if liquidity_event.outcome is not BreachOutcome.REJECTION:
        return None

    if displacement.direction is Direction.BULLISH:
        expected_kind = SwingType.HIGH
    else:
        expected_kind = SwingType.LOW

    if break_position < 0 or break_position >= len(df):
        raise IndexError("break_position outside dataframe.")

    break_ts = df.index[break_position]

    if displacement.timestamp > break_ts:
        return None

    structure = _candidate_structure(
        swings,
        direction=displacement.direction,
        before=liquidity_event.breach_timestamp,
        after_confirmation=break_ts,
    )

    if structure is None or structure.kind is not expected_kind:
        return None

    row = df.iloc[break_position]
    break_price = float(row["Close"])

    if displacement.direction is Direction.BULLISH:
        broken = break_price > structure.price
    else:
        broken = break_price < structure.price

    if not broken:
        return None

    end = min(len(df), break_position + 1 + follow_through_bars)
    future = df.iloc[break_position + 1:end]

    if follow_through_bars == 0:
        follow_through = True
    elif len(future) == 0:
        follow_through = False
    elif displacement.direction is Direction.BULLISH:
        follow_through = float(future["High"].max()) > break_price
    else:
        follow_through = float(future["Low"].min()) < break_price

    if not follow_through:
        return None

    return MSSEvent(
        timestamp=break_ts,
        direction=displacement.direction,
        structural_point=structure,
        liquidity_event=liquidity_event,
        displacement=displacement,
        break_price=break_price,
        follow_through=follow_through,
    )
