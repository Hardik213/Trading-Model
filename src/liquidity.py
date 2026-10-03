from __future__ import annotations

"""
Compatibility wrapper around the deterministic market-structure/liquidity
engine.

The old implementation treated any breach of a rolling high/low as a
"liquidity sweep". That is intentionally removed. A breach is only a breach;
raid/rejection vs acceptance requires post-breach evidence.
"""

import numpy as np
import pandas as pd
from dataclasses import replace

from .market_structure import (
    BreachOutcome,
    LiquidityEvent,
    LiquidityLevel,
    LiquiditySide,
    SwingPoint,
    SwingType,
    detect_confirmed_swings,
    detect_liquidity_breach,
    _bar_availability_timestamp,
    latest_confirmed_liquidity,
    resolve_breach,
    swings_to_liquidity,
    validate_ohlc,
)
from .data_contract import normalize_ohlc


def add_liquidity_features(
    df: pd.DataFrame,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
    resolution_bars: int = 3,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """
    Return an OHLC frame with auditable liquidity observations.

    IMPORTANT:
    - liquidity_breach_up/down = raw breach observations
    - liquidity_rejection_up/down = post-breach rejection observations
        - rejection and sweep fields are written on the confirming bar, never the
            earlier breach bar; the actual result availability is recorded separately
    """
    data = df.copy()
    validate_ohlc(data)
    if timeframe is not None:
        data.attrs["timeframe"] = timeframe

    has_timing_metadata = any(
        column in data.columns
        for column in ("available_at", "availability_ts", "interval_end")
    ) or data.attrs.get("availability_mode") == "event_time"
    if has_timing_metadata:
        if (
            ("available_at" in data.columns or "availability_ts" in data.columns)
            and "interval_end" not in data.columns
            and timeframe is None
            and data.attrs.get("timeframe") is None
        ):
            raise ValueError("Timeframe metadata is required to validate event-time liquidity bars.")
        timing = normalize_ohlc(
            data,
            timeframe=str(timeframe or data.attrs.get("timeframe") or "1min"),
            source=str(data.attrs.get("source", "UNKNOWN")),
        )
        for column in (
            "available_at",
            "availability_ts",
            "interval_end",
            "historical_complete",
            "is_complete",
        ):
            if column in timing.columns:
                data[column] = timing[column].array
        data.attrs.update(timing.attrs)
    elif (
        timeframe is None
        and data.attrs.get("timeframe") is None
        and str(data.attrs.get("bar_label", "left")).lower() != "right"
    ):
        raise ValueError(
            "Timeframe metadata or explicit right-labeled bars are required for causal liquidity features."
        )

    complete = pd.Series(True, index=data.index)
    for column in ("historical_complete", "is_complete"):
        if column in data.columns:
            complete &= data[column].fillna(False).astype(bool)
    for column in ("available_at", "availability_ts"):
        if column in data.columns:
            complete &= data[column].notna()
    observed = data.loc[complete]

    swings = detect_confirmed_swings(
        observed,
        left_bars=left_bars,
        right_bars=right_bars,
    )

    data["liquidity_breach_up"] = 0
    data["liquidity_breach_down"] = 0
    data["liquidity_rejection_up"] = 0
    data["liquidity_rejection_down"] = 0
    data["liquidity_acceptance_up"] = 0
    data["liquidity_acceptance_down"] = 0
    data["liquidity_sweep"] = 0.0
    data["liquidity_confirmation_available_at"] = pd.Series(
        [pd.NaT] * len(data), index=data.index, dtype=object
    )

    for level in swings_to_liquidity(swings):
        confirmation_position = observed.index.get_loc(level.confirmation_timestamp)
        level = replace(
            level,
            availability_timestamp=_bar_availability_timestamp(
                observed, confirmation_position
            ),
        )

        for observed_position in range(len(observed)):
            if observed.index[observed_position] < level.confirmation_timestamp:
                continue
            if (
                _bar_availability_timestamp(observed, observed_position)
                < level.availability_timestamp
            ):
                continue
            row = observed.iloc[observed_position]
            if not detect_liquidity_breach(row, level):
                continue

            pos = data.index.get_loc(observed.index[observed_position])
            if level.side is LiquiditySide.BSL:
                data.iloc[pos, data.columns.get_loc("liquidity_breach_up")] = 1
            else:
                data.iloc[pos, data.columns.get_loc("liquidity_breach_down")] = 1

            event = resolve_breach(
                observed,
                level,
                observed_position,
                max_resolution_bars=resolution_bars,
            )

            if event.resolution_timestamp is None:
                continue
            result_pos = data.index.get_loc(event.resolution_timestamp)
            data.iloc[
                result_pos,
                data.columns.get_loc("liquidity_confirmation_available_at"),
            ] = event.availability_timestamp
            if event.outcome is BreachOutcome.REJECTION:
                if level.side is LiquiditySide.BSL:
                    data.iloc[result_pos, data.columns.get_loc("liquidity_rejection_up")] = 1
                    data.iloc[result_pos, data.columns.get_loc("liquidity_sweep")] = 1.0
                else:
                    data.iloc[result_pos, data.columns.get_loc("liquidity_rejection_down")] = 1
                    data.iloc[result_pos, data.columns.get_loc("liquidity_sweep")] = -1.0

            elif event.outcome is BreachOutcome.ACCEPTANCE:
                if level.side is LiquiditySide.BSL:
                    data.iloc[result_pos, data.columns.get_loc("liquidity_acceptance_up")] = 1
                else:
                    data.iloc[result_pos, data.columns.get_loc("liquidity_acceptance_down")] = 1

    return data


__all__ = [
    "BreachOutcome",
    "LiquidityEvent",
    "LiquidityLevel",
    "LiquiditySide",
    "SwingPoint",
    "SwingType",
    "detect_confirmed_swings",
    "detect_liquidity_breach",
    "latest_confirmed_liquidity",
    "resolve_breach",
    "swings_to_liquidity",
    "add_liquidity_features",
]
