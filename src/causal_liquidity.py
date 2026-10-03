from __future__ import annotations

"""Causal liquidity evidence for the XAU/USD ICT-2022 research engine.

This module deliberately separates:
    1. a confirmed liquidity level,
    2. a raw breach,
    3. post-breach rejection/acceptance,
    4. a reversal-relevant liquidity event,
    5. a draw-on-liquidity objective.

Every public query is evaluated *as of* a timestamp.  Future candles may be
present in the dataframe, but they are never allowed to make an event visible
before its confirmation/resolution timestamp.

This is a research detector, not a trade signal generator.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Optional

import pandas as pd

from .market_structure import (
    BreachOutcome,
    LiquidityEvent,
    LiquidityLevel,
    LiquiditySide,
    SwingPoint,
    _bar_availability_timestamp,
    detect_confirmed_swings,
    detect_liquidity_breach,
    latest_confirmed_liquidity,
    resolve_breach,
    swings_to_liquidity,
    validate_ohlc,
)
from .mss import Direction, normalize_direction
from .timeframe_context import _nominal_close_times, bars_available_as_of


class LiquidityEvidenceKind(str, Enum):
    BREACH = "BREACH"
    REJECTION = "REJECTION"
    ACCEPTANCE = "ACCEPTANCE"


def _canonical_liquidity_event(event) -> Optional[LiquidityEvent]:
    if event is None:
        return None
    if isinstance(event, LiquidityEvent):
        return event
    for attr in ("breach", "liquidity_event", "event"):
        nested = getattr(event, attr, None)
        canonical = _canonical_liquidity_event(nested)
        if canonical is not None:
            return canonical
    return None


@dataclass(frozen=True)
class CausalLiquidityEvidence:
    """One liquidity observation whose information set is valid at ``as_of``."""

    as_of: pd.Timestamp
    level: LiquidityLevel
    breach: LiquidityEvent
    kind: LiquidityEvidenceKind

    @property
    def confirmed(self) -> bool:
        event = _canonical_liquidity_event(self.breach)
        if event is None:
            return False
        if self.kind is LiquidityEvidenceKind.BREACH:
            breach_available = (
                event.breach_availability_timestamp or event.breach_timestamp
            )
            return pd.Timestamp(breach_available) <= self.as_of
        if event.resolution_timestamp is None:
            return False
        resolution_available = event.availability_timestamp or event.resolution_timestamp
        return pd.Timestamp(resolution_available) <= self.as_of

    @property
    def is_rejection(self) -> bool:
        return self.kind is LiquidityEvidenceKind.REJECTION

    @property
    def is_acceptance(self) -> bool:
        return self.kind is LiquidityEvidenceKind.ACCEPTANCE


@dataclass(frozen=True)
class LiquidityObjective:
    """A visible liquidity pool that can act as a draw-on-liquidity objective."""

    level: LiquidityLevel
    as_of: pd.Timestamp
    distance: float


def _as_of(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("as_of must be a valid timestamp")
    return ts


def _visible_bars_as_of(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    availability_column = next(
        (column for column in ("available_at", "availability_ts") if column in df.columns),
        None,
    )
    availability_mode = df.attrs.get("availability_mode")
    if (
        availability_mode == "event_time"
        and availability_column is None
        and "interval_end" not in df.columns
    ):
        raise ValueError("Event-time availability metadata is missing from the OHLC frame.")
    if availability_column is not None:
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            raise ValueError("Event-time OHLC labels must be timezone-aware datetimes.")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise ValueError("Event-time OHLC labels must be unique and ordered.")

        available_at = pd.to_datetime(df[availability_column], utc=True, errors="raise")
        if "available_at" in df.columns and "availability_ts" in df.columns:
            availability_alias = pd.to_datetime(df["availability_ts"], utc=True, errors="raise")
            matches = available_at.eq(availability_alias) | (
                available_at.isna() & availability_alias.isna()
            )
            if not matches.all():
                raise ValueError("available_at and availability_ts must match when both are provided.")

        labels = df.index.tz_convert("UTC")
        before_label = available_at.notna() & (available_at < labels)
        if before_label.any():
            raise ValueError("Bar availability cannot precede its timestamp label.")

        timeframe = df.attrs.get("timeframe")
        if "interval_end" in df.columns:
            interval_end = pd.to_datetime(df["interval_end"], utc=True, errors="raise")
            before_end = available_at.notna() & interval_end.notna() & (available_at < interval_end)
            if before_end.any():
                raise ValueError("Bar availability cannot precede its interval end.")
            missing_end = available_at.notna() & interval_end.isna()
        else:
            missing_end = available_at.notna()

        if timeframe is None and missing_end.any() and str(
            df.attrs.get("bar_label", "left")
        ).lower() != "right":
            raise ValueError(
                "Timeframe metadata or right-labeled bars are required to validate availability."
            )
        if timeframe is not None and missing_end.any():
            nominal_close = _nominal_close_times(df, str(timeframe))
            before_close = available_at.notna() & (available_at < nominal_close)
            if before_close.any():
                raise ValueError("Bar availability cannot precede its nominal close.")

        if not available_at.dropna().is_monotonic_increasing:
            raise ValueError("Bar availability timestamps must be non-decreasing.")

    if (
        availability_column is not None
        or "interval_end" in df.columns
        or availability_mode == "nominal_close_fallback"
        or df.attrs.get("timeframe") is not None
        or str(df.attrs.get("bar_label", "left")).lower() == "right"
    ):
        timeframe = df.attrs.get("timeframe")
        if timeframe is None:
            if availability_column is not None and "interval_end" not in df.columns:
                raise ValueError(
                    "Timeframe metadata is required when availability lacks interval_end."
                )
            timeframe = "1min"
        return bars_available_as_of(
            df,
            as_of,
            timeframe=str(timeframe or ""),
        )

    raise ValueError(
        "Timeframe metadata or explicit right-labeled bars are required for causal liquidity queries."
    )


def confirmed_liquidity_map(
    df: pd.DataFrame,
    as_of,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
    equal_tolerance: float = 0.0,
) -> list[LiquidityLevel]:
    """Return only liquidity levels whose source swing is confirmed by ``as_of``."""
    validate_ohlc(df)
    ts = _as_of(as_of)
    visible = _visible_bars_as_of(df, ts)
    if visible.empty:
        return []
    swings = detect_confirmed_swings(
        visible,
        left_bars=left_bars,
        right_bars=right_bars,
    )
    visible_swings = [s for s in swings if pd.Timestamp(s.confirmation_timestamp) < ts]
    levels = swings_to_liquidity(
        visible_swings,
        equal_tolerance=equal_tolerance,
    )
    return [
        replace(
            level,
            confirmation_timestamp=swing.confirmation_timestamp,
            availability_timestamp=_bar_availability_timestamp(
                visible,
                visible.index.get_loc(swing.confirmation_timestamp),
            ),
        )
        for level, swing in zip(levels, visible_swings)
    ]


def _candidate_breach(
    df: pd.DataFrame,
    level: LiquidityLevel,
    as_of: pd.Timestamp,
    *,
    resolution_bars: int,
) -> Optional[LiquidityEvent]:
    """Find the latest fully-resolved breach of ``level`` visible at ``as_of``."""
    visible = _visible_bars_as_of(df, as_of)
    if visible.empty:
        return None

    # A level cannot be used before its source swing was confirmed. The breach
    # scan must start on or after the source swing's confirmation timestamp, not
    # merely the original swing timestamp. This blocks a raw breach from being
    # retroactively turned into reversal evidence before the level was known.
    confirmation_ts = getattr(level, "confirmation_timestamp", None)
    lower_bound = pd.Timestamp(level.timestamp)
    if confirmation_ts is not None:
        lower_bound = max(lower_bound, pd.Timestamp(confirmation_ts))
    eligible_positions = [
        position
        for position, label in enumerate(visible.index)
        if label >= lower_bound
        and (
            level.availability_timestamp is None
            or _bar_availability_timestamp(visible, position)
            >= level.availability_timestamp
        )
    ]
    eligible = visible.iloc[eligible_positions]
    if eligible.empty:
        return None

    for pos in range(len(eligible) - 1, -1, -1):
        row = eligible.iloc[pos]
        if not detect_liquidity_breach(row, level):
            continue
        event = resolve_breach(
            eligible,
            level,
            pos,
            max_resolution_bars=resolution_bars,
        )
        if event.outcome is BreachOutcome.UNRESOLVED:
            continue
        # resolve_breach only sees ``eligible``.  Still explicitly enforce the
        # causal boundary so this function remains safe if the resolver changes.
        event_available = event.availability_timestamp or event.resolution_timestamp
        if event_available is not None and pd.Timestamp(event_available) > as_of:
            continue
        return event
    return None


def liquidity_evidence_as_of(
    df: pd.DataFrame,
    as_of,
    *,
    side: Optional[LiquiditySide] = None,
    left_bars: int = 2,
    right_bars: int = 2,
    resolution_bars: int = 3,
    equal_tolerance: float = 0.0,
) -> list[CausalLiquidityEvidence]:
    """Return fully causal breach/rejection/acceptance evidence visible at ``as_of``.

    A wick that has not yet received post-breach confirmation is deliberately
    absent from the returned rejection/acceptance evidence.
    """
    ts = _as_of(as_of)
    levels = confirmed_liquidity_map(
        df,
        ts,
        left_bars=left_bars,
        right_bars=right_bars,
        equal_tolerance=equal_tolerance,
    )
    if side is not None:
        levels = [level for level in levels if level.side is side]

    out: list[CausalLiquidityEvidence] = []
    for level in levels:
        event = _candidate_breach(
            df,
            level,
            ts,
            resolution_bars=resolution_bars,
        )
        if event is None:
            continue
        kind = {
            BreachOutcome.REJECTION: LiquidityEvidenceKind.REJECTION,
            BreachOutcome.ACCEPTANCE: LiquidityEvidenceKind.ACCEPTANCE,
        }[event.outcome]
        out.append(
            CausalLiquidityEvidence(
                as_of=ts,
                level=level,
                breach=event,
                kind=kind,
            )
        )
    return out


def latest_reversal_liquidity(
    df: pd.DataFrame,
    as_of,
    direction,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
    resolution_bars: int = 3,
    equal_tolerance: float = 0.0,
) -> Optional[CausalLiquidityEvidence]:
    """Return the latest confirmed rejection of opposing liquidity.

    Bullish reversal requires SSL rejection; bearish reversal requires BSL
    rejection.  Acceptance is never treated as a sweep.
    """
    direction = normalize_direction(direction)
    side = LiquiditySide.SSL if direction is Direction.BULLISH else LiquiditySide.BSL
    candidates = [
        item
        for item in liquidity_evidence_as_of(
            df,
            as_of,
            side=side,
            left_bars=left_bars,
            right_bars=right_bars,
            resolution_bars=resolution_bars,
            equal_tolerance=equal_tolerance,
        )
        if item.is_rejection
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: pd.Timestamp(
              (_canonical_liquidity_event(item.breach).availability_timestamp
               or _canonical_liquidity_event(item.breach).resolution_timestamp
               or _canonical_liquidity_event(item.breach).breach_timestamp)
        ),
    )


def draw_on_liquidity(
    df: pd.DataFrame,
    as_of,
    direction,
    *,
    current_price: Optional[float] = None,
    left_bars: int = 2,
    right_bars: int = 2,
    equal_tolerance: float = 0.0,
) -> Optional[LiquidityObjective]:
    """Select the nearest visible opposing liquidity pool beyond price.

    This is an objective selector only.  It does not imply that price will
    reach the level and does not create a trade by itself.
    """
    direction = normalize_direction(direction)
    ts = _as_of(as_of)
    visible = _visible_bars_as_of(df, ts)
    if visible.empty:
        return None
    if current_price is None:
        current_price = float(visible.iloc[-1]["Close"])

    levels = confirmed_liquidity_map(
        visible,
        ts,
        left_bars=left_bars,
        right_bars=right_bars,
        equal_tolerance=equal_tolerance,
    )

    if direction is Direction.BULLISH:
        candidates = [l for l in levels if l.side is LiquiditySide.BSL and l.price > current_price]
    else:
        candidates = [l for l in levels if l.side is LiquiditySide.SSL and l.price < current_price]

    if not candidates:
        return None
    level = min(candidates, key=lambda l: abs(l.price - current_price))
    return LiquidityObjective(
        level=level,
        as_of=ts,
        distance=abs(level.price - current_price),
    )


__all__ = [
    "CausalLiquidityEvidence",
    "LiquidityEvidenceKind",
    "LiquidityObjective",
    "confirmed_liquidity_map",
    "liquidity_evidence_as_of",
    "latest_reversal_liquidity",
    "draw_on_liquidity",
]
