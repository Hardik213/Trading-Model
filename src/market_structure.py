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
    """A pool observed at ``timestamp``, confirmed later and available at its own event time."""

    timestamp: pd.Timestamp
    price: float
    side: LiquiditySide
    source: str
    strength: int = 1
    confirmation_timestamp: Optional[pd.Timestamp] = None
    availability_timestamp: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class LiquidityEvent:
    """A breach and its first decisive close; availability timestamps are distinct from bar labels."""

    level: LiquidityLevel
    breach_timestamp: pd.Timestamp
    breach_price: float
    breach_depth: float
    outcome: BreachOutcome
    resolution_timestamp: Optional[pd.Timestamp] = None
    resolution_price: Optional[float] = None
    availability_timestamp: Optional[pd.Timestamp] = None
    breach_availability_timestamp: Optional[pd.Timestamp] = None


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
                confirmation_timestamp=s.confirmation_timestamp,
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
                confirmation_timestamp=level.confirmation_timestamp,
                availability_timestamp=level.availability_timestamp,
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


def _bar_is_complete(df: pd.DataFrame, position: int) -> bool:
    row = df.iloc[position]
    for column in ("historical_complete", "is_complete"):
        if column in df.columns:
            if pd.isna(row[column]) or not bool(row[column]):
                return False
    for column in ("available_at", "availability_ts"):
        if column in df.columns and pd.isna(row[column]):
            return False
    return True


def _bar_availability_timestamp(df: pd.DataFrame, position: int) -> pd.Timestamp:
    row = df.iloc[position]
    for column in ("available_at", "availability_ts", "interval_end"):
        if column in df.columns and pd.notna(row[column]):
            return pd.Timestamp(row[column])

    if df.attrs.get("availability_mode") == "event_time":
        raise ValueError("Event-time availability metadata is missing from the OHLC frame.")
    bar_label = str(df.attrs.get("bar_label", "")).lower()
    if bar_label not in {"left", "right"}:
        raise ValueError("Explicit bar_label metadata is required for liquidity timing fallback.")
    if df.index.tz is None:
        raise ValueError("Timezone-aware bar labels are required for liquidity timing fallback.")
    if bar_label == "right":
        return pd.Timestamp(df.index[position])

    timeframe = df.attrs.get("timeframe")
    if timeframe is None:
        raise ValueError("Timeframe metadata is required for left-labeled liquidity timing fallback.")

    from .timeframe_context import _nominal_close_times

    return pd.Timestamp(_nominal_close_times(df.iloc[[position]], str(timeframe))[0])


def resolve_breach(
    df: pd.DataFrame,
    level: LiquidityLevel,
    breach_position: int,
    *,
    max_resolution_bars: int = 3,
) -> LiquidityEvent:
    """
    Resolve the bars immediately following a breach.

        A close on the breach bar beyond the level is immediate acceptance, never
        a sweep. Otherwise the first subsequent complete close beyond either side
        resolves the event: back across the level is rejection; continuation beyond
        it is acceptance. Rejection therefore always requires subsequent evidence.

    If neither condition is observed inside the resolution window, the event
    remains UNRESOLVED. A wick alone is therefore never labelled a sweep.
    """
    validate_ohlc(df)
    availability_mode = df.attrs.get("availability_mode")
    if availability_mode not in {None, "event_time", "nominal_close_fallback"}:
        raise ValueError("Unsupported liquidity availability mode.")
    has_timing_metadata = any(
        column in df.columns
        for column in ("available_at", "availability_ts", "interval_end")
    ) or availability_mode == "event_time"
    if has_timing_metadata:
        if (
            ("available_at" in df.columns or "availability_ts" in df.columns)
            and "interval_end" not in df.columns
            and df.attrs.get("timeframe") is None
        ):
            raise ValueError("Timeframe metadata is required to validate event-time liquidity bars.")
        from .data_contract import normalize_ohlc

        normalize_ohlc(
            df,
            timeframe=str(df.attrs.get("timeframe") or "1min"),
            source=str(df.attrs.get("source", "UNKNOWN")),
        )
    if breach_position < 0 or breach_position >= len(df):
        raise IndexError("breach_position is outside the dataframe.")
    if max_resolution_bars < 1:
        raise ValueError("max_resolution_bars must be >= 1.")

    bar = df.iloc[breach_position]
    breach_price = float(bar["High"] if level.side is LiquiditySide.BSL else bar["Low"])
    depth = abs(breach_price - level.price)
    if not _bar_is_complete(df, breach_position):
        return LiquidityEvent(
            level, df.index[breach_position], breach_price, depth,
            BreachOutcome.UNRESOLVED,
        )
    breach_availability = _bar_availability_timestamp(df, breach_position)

    close = float(bar["Close"])
    if (level.side is LiquiditySide.BSL and close > level.price) or (
        level.side is LiquiditySide.SSL and close < level.price
    ):
        return LiquidityEvent(
            level=level,
            breach_timestamp=df.index[breach_position],
            breach_price=breach_price,
            breach_depth=depth,
            outcome=BreachOutcome.ACCEPTANCE,
            resolution_timestamp=df.index[breach_position],
            resolution_price=close,
            availability_timestamp=breach_availability,
            breach_availability_timestamp=breach_availability,
        )

    resolution_positions = []
    for position in range(breach_position + 1, len(df)):
        if _bar_is_complete(df, position):
            resolution_positions.append(position)
            if len(resolution_positions) == max_resolution_bars:
                break

    for position in resolution_positions:
        close = float(df.iloc[position]["Close"])
        if level.side is LiquiditySide.BSL:
            if close < level.price:
                outcome = BreachOutcome.REJECTION
            elif close > level.price:
                outcome = BreachOutcome.ACCEPTANCE
            else:
                continue
        else:
            if close > level.price:
                outcome = BreachOutcome.REJECTION
            elif close < level.price:
                outcome = BreachOutcome.ACCEPTANCE
            else:
                continue
        return LiquidityEvent(
            level=level,
            breach_timestamp=df.index[breach_position],
            breach_price=breach_price,
            breach_depth=depth,
            outcome=outcome,
            resolution_timestamp=df.index[position],
            resolution_price=close,
            availability_timestamp=_bar_availability_timestamp(df, position),
            breach_availability_timestamp=breach_availability,
        )

    return LiquidityEvent(
        level=level,
        breach_timestamp=df.index[breach_position],
        breach_price=breach_price,
        breach_depth=depth,
        outcome=BreachOutcome.UNRESOLVED,
        breach_availability_timestamp=breach_availability,
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
