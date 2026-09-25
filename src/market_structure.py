from __future__ import annotations

"""
Deterministic market-structure primitives for the XAU/USD ICT-2022 engine.

Design principles:
- Raw OHLC is authoritative.
- A swing is only usable after its confirmation time.
- A liquidity breach is NOT automatically a sweep.
- A liquidity event records post-breach behaviour separately.
- No setup/entry decision is made here.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class LiquiditySide(str, Enum):
    BSL = "BSL"
    SSL = "SSL"


class BreachOutcome(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    REJECTION = "REJECTION"
    ACCEPTANCE = "ACCEPTANCE"


@dataclass(frozen=True)
class SwingPoint:
    timestamp: pd.Timestamp
    price: float
    kind: SwingType
    source_index: int
    confirmation_timestamp: pd.Timestamp


@dataclass(frozen=True)
class LiquidityLevel:
    timestamp: pd.Timestamp
    price: float
    side: LiquiditySide
    source: str
    strength: int = 1


@dataclass(frozen=True)
class LiquidityEvent:
    level: LiquidityLevel
    breach_timestamp: pd.Timestamp
    breach_price: float
    breach_depth: float
    outcome: BreachOutcome
    resolution_timestamp: Optional[pd.Timestamp] = None
    resolution_price: Optional[float] = None


def validate_ohlc(df: pd.DataFrame) -> None:
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("OHLC index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLC index must be sorted ascending.")
    if df.index.has_duplicates:
        raise ValueError("OHLC index must not contain duplicate timestamps.")


def detect_confirmed_swings(
    df: pd.DataFrame,
    left_bars: int = 2,
    right_bars: int = 2,
) -> list[SwingPoint]:
    """
    Detect pivot swings without making the pivot available before confirmation.

    A high at i is confirmed at i + right_bars only if its high is strictly
    greater than the highs in the left/right windows. Equal values are not
    promoted to a swing; this avoids manufacturing structure from flat highs.

    In live/replay usage, consume a SwingPoint only when the current timestamp
    >= confirmation_timestamp.
    """
    validate_ohlc(df)
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be >= 1.")

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    idx = df.index
    swings: list[SwingPoint] = []

    start = left_bars
    stop = len(df) - right_bars

    for i in range(start, stop):
        h = highs[i]
        l = lows[i]

        left_h = highs[i - left_bars:i]
        right_h = highs[i + 1:i + 1 + right_bars]
        left_l = lows[i - left_bars:i]
        right_l = lows[i + 1:i + 1 + right_bars]

        is_high = h > np.max(left_h) and h > np.max(right_h)
        is_low = l < np.min(left_l) and l < np.min(right_l)

        confirmation_ts = idx[i + right_bars]

        if is_high:
            swings.append(
                SwingPoint(
                    timestamp=idx[i],
                    price=float(h),
                    kind=SwingType.HIGH,
                    source_index=i,
                    confirmation_timestamp=confirmation_ts,
                )
            )
        if is_low:
            swings.append(
                SwingPoint(
                    timestamp=idx[i],
                    price=float(l),
                    kind=SwingType.LOW,
                    source_index=i,
                    confirmation_timestamp=confirmation_ts,
                )
            )

    return swings


def swings_to_liquidity(
    swings: Iterable[SwingPoint],
    *,
    equal_tolerance: float = 0.0,
) -> list[LiquidityLevel]:
    """
    Convert confirmed swing highs/lows into external liquidity candidates.

    Equal-high/low clustering is intentionally conservative: levels are
    clustered only when prices are within equal_tolerance. With tolerance 0,
    only exact equal prices cluster.
    """
    levels: list[LiquidityLevel] = []
    for s in swings:
        levels.append(
            LiquidityLevel(
                timestamp=s.timestamp,
                price=s.price,
                side=(
                    LiquiditySide.BSL
                    if s.kind is SwingType.HIGH
                    else LiquiditySide.SSL
                ),
                source="SWING",
            )
        )

    if equal_tolerance <= 0 or len(levels) < 2:
        return levels

    # Increase strength for nearby same-side levels without deleting originals.
    out: list[LiquidityLevel] = []
    for level in levels:
        strength = 1
        for other in levels:
            if other is level or other.side is not level.side:
                continue
            if abs(other.price - level.price) <= equal_tolerance:
                strength += 1
        out.append(
            LiquidityLevel(
                timestamp=level.timestamp,
                price=level.price,
                side=level.side,
                source=level.source,
                strength=strength,
            )
        )
    return out


def detect_liquidity_breach(
    bar: pd.Series,
    level: LiquidityLevel,
) -> bool:
    """Return whether the current bar actually trades through the level."""
    if level.side is LiquiditySide.BSL:
        return float(bar["High"]) > level.price
    return float(bar["Low"]) < level.price


def resolve_breach(
    df: pd.DataFrame,
    level: LiquidityLevel,
    breach_position: int,
    *,
    max_resolution_bars: int = 3,
) -> LiquidityEvent:
    """
    Resolve the bars immediately following a breach.

    Rejection:
      BSL: price trades above the level, then a later close returns below it.
      SSL: price trades below the level, then a later close returns above it.

    Acceptance:
      BSL: a later close remains above the level.
      SSL: a later close remains below the level.

    If neither condition is observed inside the resolution window, the event
    remains UNRESOLVED. A wick alone is therefore never labelled a sweep.
    """
    validate_ohlc(df)
    if breach_position < 0 or breach_position >= len(df):
        raise IndexError("breach_position is outside the dataframe.")
    if max_resolution_bars < 1:
        raise ValueError("max_resolution_bars must be >= 1.")

    bar = df.iloc[breach_position]
    breach_price = float(bar["High"] if level.side is LiquiditySide.BSL else bar["Low"])
    depth = abs(breach_price - level.price)

    end = min(len(df), breach_position + 1 + max_resolution_bars)

    for j in range(breach_position + 1, end):
        row = df.iloc[j]
        close = float(row["Close"])

        if level.side is LiquiditySide.BSL and close < level.price:
            return LiquidityEvent(
                level=level,
                breach_timestamp=df.index[breach_position],
                breach_price=breach_price,
                breach_depth=depth,
                outcome=BreachOutcome.REJECTION,
                resolution_timestamp=df.index[j],
                resolution_price=close,
            )

        if level.side is LiquiditySide.SSL and close > level.price:
            return LiquidityEvent(
                level=level,
                breach_timestamp=df.index[breach_position],
                breach_price=breach_price,
                breach_depth=depth,
                outcome=BreachOutcome.REJECTION,
                resolution_timestamp=df.index[j],
                resolution_price=close,
            )

    # Only classify acceptance if a close beyond the level is observed.
    for j in range(breach_position, end):
        close = float(df.iloc[j]["Close"])
        if level.side is LiquiditySide.BSL and close > level.price:
            return LiquidityEvent(
                level=level,
                breach_timestamp=df.index[breach_position],
                breach_price=breach_price,
                breach_depth=depth,
                outcome=BreachOutcome.ACCEPTANCE,
                resolution_timestamp=df.index[j],
                resolution_price=close,
            )
        if level.side is LiquiditySide.SSL and close < level.price:
            return LiquidityEvent(
                level=level,
                breach_timestamp=df.index[breach_position],
                breach_price=breach_price,
                breach_depth=depth,
                outcome=BreachOutcome.ACCEPTANCE,
                resolution_timestamp=df.index[j],
                resolution_price=close,
            )

    return LiquidityEvent(
        level=level,
        breach_timestamp=df.index[breach_position],
        breach_price=breach_price,
        breach_depth=depth,
        outcome=BreachOutcome.UNRESOLVED,
    )


def latest_confirmed_liquidity(
    swings: Iterable[SwingPoint],
    as_of: pd.Timestamp,
) -> list[LiquidityLevel]:
    """
    Return only liquidity derived from swings that were knowable as of as_of.
    """
    confirmed = [
        s for s in swings
        if s.confirmation_timestamp < as_of
    ]
    return swings_to_liquidity(confirmed)
