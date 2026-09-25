from __future__ import annotations

"""
Event-driven trade/backtest primitives.

This module deliberately separates:
1. setup qualification,
2. trade definition,
3. candle-by-candle outcome resolution.

It never converts a future return into a trade result.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeOutcome(str, Enum):
    TARGET = "TARGET"
    STOP = "STOP"
    AMBIGUOUS = "AMBIGUOUS"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class TradePlan:
    trade_id: str
    entry_time: pd.Timestamp
    direction: TradeDirection
    entry_price: float
    stop_price: float
    target_price: float
    planned_r: float
    spread: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        risk = self.risk_distance
        if risk <= 0:
            raise ValueError("TradePlan requires positive structural risk distance.")

        if self.direction is TradeDirection.LONG:
            if not (self.stop_price < self.entry_price < self.target_price):
                raise ValueError("Invalid LONG entry/stop/target geometry.")
        else:
            if not (self.target_price < self.entry_price < self.stop_price):
                raise ValueError("Invalid SHORT entry/stop/target geometry.")

        if self.planned_r <= 0:
            raise ValueError("planned_r must be positive.")

    @property
    def risk_distance(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def gross_target_r(self) -> float:
        return abs(self.target_price - self.entry_price) / self.risk_distance

    def gross_r_for_price(self, exit_price: float) -> float:
        if self.direction is TradeDirection.LONG:
            return (exit_price - self.entry_price) / self.risk_distance
        return (self.entry_price - exit_price) / self.risk_distance

    @property
    def estimated_cost_r(self) -> float:
        # Costs are expressed in price units and converted to R. The caller
        # may provide broker-specific cost assumptions; this module does not
        # invent them.
        price_cost = max(0.0, self.spread) + max(0.0, self.slippage)
        return price_cost / self.risk_distance + max(0.0, self.commission) / self.risk_distance


@dataclass(frozen=True)
class TradeResult:
    trade_id: str
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: TradeDirection
    outcome: TradeOutcome
    entry_price: float
    exit_price: Optional[float]
    gross_r: Optional[float]
    net_r: Optional[float]
    mfe_r: Optional[float]
    mae_r: Optional[float]
    bars_held: int
    reason: str


def _mfe_mae(
    plan: TradePlan,
    path: pd.DataFrame,
    entry_position: int,
    end_position: int,
) -> tuple[float, float]:
    """
    Calculate MFE/MAE from observed OHLC while excluding bars before entry.

    MFE/MAE are path descriptors; they are not used to choose the target after
    the fact.
    """
    risk = plan.risk_distance
    mfe = 0.0
    mae = 0.0

    for i in range(entry_position, end_position + 1):
        row = path.iloc[i]
        high = float(row["High"])
        low = float(row["Low"])

        if plan.direction is TradeDirection.LONG:
            favorable = (high - plan.entry_price) / risk
            adverse = (low - plan.entry_price) / risk
        else:
            favorable = (plan.entry_price - low) / risk
            adverse = (plan.entry_price - high) / risk

        mfe = max(mfe, favorable)
        mae = min(mae, adverse)

    return mfe, mae


def simulate_trade(
    path: pd.DataFrame,
    plan: TradePlan,
    *,
    max_bars: Optional[int] = None,
) -> TradeResult:
    """
    Resolve one already-qualified trade plan against subsequent OHLC bars.

    Conservative path rule:
      - If both stop and target are touched in the same candle, outcome is
        AMBIGUOUS because OHLC alone cannot establish which came first.
      - No optimistic assumption is made.

    Entry semantics:
      - plan.entry_time identifies the first candle from which the trade is
        considered active.
      - The implementation does not fabricate an intrabar fill. The supplied
        entry_price is the research plan's declared price.
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(path.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if not isinstance(path.index, pd.DatetimeIndex):
        raise TypeError("path index must be a DatetimeIndex.")
    if not path.index.is_monotonic_increasing:
        raise ValueError("path index must be sorted ascending.")

    positions = path.index.searchsorted(plan.entry_time, side="left")
    if positions >= len(path):
        return TradeResult(
            plan.trade_id, plan.entry_time, None, plan.direction,
            TradeOutcome.EXPIRED, plan.entry_price, None, None, None,
            None, None, 0, "No candle exists at or after entry_time."
        )

    end = len(path) - 1
    if max_bars is not None:
        if max_bars < 1:
            raise ValueError("max_bars must be >= 1.")
        end = min(end, positions + max_bars - 1)

    for i in range(positions, end + 1):
        row = path.iloc[i]
        high = float(row["High"])
        low = float(row["Low"])

        if plan.direction is TradeDirection.LONG:
            hit_stop = low <= plan.stop_price
            hit_target = high >= plan.target_price
        else:
            hit_stop = high >= plan.stop_price
            hit_target = low <= plan.target_price

        if hit_stop and hit_target:
            mfe, mae = _mfe_mae(plan, path, positions, i)
            return TradeResult(
                plan.trade_id, plan.entry_time, path.index[i], plan.direction,
                TradeOutcome.AMBIGUOUS, plan.entry_price, None, None, None,
                mfe, mae, i - positions + 1,
                "Both target and stop were touched in one OHLC bar; order is unknowable."
            )

        if hit_stop:
            exit_price = plan.stop_price
            gross = plan.gross_r_for_price(exit_price)
            net = gross - plan.estimated_cost_r
            mfe, mae = _mfe_mae(plan, path, positions, i)
            return TradeResult(
                plan.trade_id, plan.entry_time, path.index[i], plan.direction,
                TradeOutcome.STOP, plan.entry_price, exit_price, gross, net,
                mfe, mae, i - positions + 1,
                "Structural invalidation was reached before target."
            )

        if hit_target:
            exit_price = plan.target_price
            gross = plan.gross_r_for_price(exit_price)
            net = gross - plan.estimated_cost_r
            mfe, mae = _mfe_mae(plan, path, positions, i)
            return TradeResult(
                plan.trade_id, plan.entry_time, path.index[i], plan.direction,
                TradeOutcome.TARGET, plan.entry_price, exit_price, gross, net,
                mfe, mae, i - positions + 1,
                "Declared opposing-liquidity target was reached before stop."
            )

    mfe, mae = _mfe_mae(plan, path, positions, end)
    return TradeResult(
        plan.trade_id, plan.entry_time, None, plan.direction,
        TradeOutcome.EXPIRED, plan.entry_price, None, None, None,
        mfe, mae, end - positions + 1,
        "Neither target nor structural invalidation was reached within the test horizon."
    )
