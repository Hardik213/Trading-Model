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

from .market_structure import (
    BreachOutcome,
    LiquidityEvent,
    LiquidityLevel,
    LiquiditySide,
    SwingPoint,
    SwingType,
    detect_confirmed_swings,
    detect_liquidity_breach,
    latest_confirmed_liquidity,
    resolve_breach,
    swings_to_liquidity,
)


def add_liquidity_features(
    df: pd.DataFrame,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
    resolution_bars: int = 3,
) -> pd.DataFrame:
    """
    Return an OHLC frame with auditable liquidity observations.

    IMPORTANT:
    - liquidity_breach_up/down = raw breach observations
    - liquidity_rejection_up/down = post-breach rejection observations
    - liquidity_sweep = retained only as a compatibility field and is set
      only when a rejection is actually resolved
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("OHLC index must be a DatetimeIndex.")

    data = df.copy()
    swings = detect_confirmed_swings(
        data,
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

    # A level is eligible only after its swing confirmation timestamp.
    for level in swings_to_liquidity(swings):
        eligible = data.index >= next(
            s.confirmation_timestamp
            for s in swings
            if s.timestamp == level.timestamp
            and (
                (s.kind is SwingType.HIGH and level.side is LiquiditySide.BSL)
                or (s.kind is SwingType.LOW and level.side is LiquiditySide.SSL)
            )
        )

        for pos in np.flatnonzero(eligible.to_numpy()):
            row = data.iloc[pos]
            if not detect_liquidity_breach(row, level):
                continue

            if level.side is LiquiditySide.BSL:
                data.iloc[pos, data.columns.get_loc("liquidity_breach_up")] = 1
            else:
                data.iloc[pos, data.columns.get_loc("liquidity_breach_down")] = 1

            event = resolve_breach(
                data,
                level,
                pos,
                max_resolution_bars=resolution_bars,
            )

            if event.outcome is BreachOutcome.REJECTION:
                if level.side is LiquiditySide.BSL:
                    data.iloc[pos, data.columns.get_loc("liquidity_rejection_up")] = 1
                    data.iloc[pos, data.columns.get_loc("liquidity_sweep")] = 1.0
                else:
                    data.iloc[pos, data.columns.get_loc("liquidity_rejection_down")] = 1
                    data.iloc[pos, data.columns.get_loc("liquidity_sweep")] = -1.0

            elif event.outcome is BreachOutcome.ACCEPTANCE:
                if level.side is LiquiditySide.BSL:
                    data.iloc[pos, data.columns.get_loc("liquidity_acceptance_up")] = 1
                else:
                    data.iloc[pos, data.columns.get_loc("liquidity_acceptance_down")] = 1

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
