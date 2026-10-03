from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    feature_row_decision_times,
    feature_row_eligibility_mask,
)


def _first_post_decision_position(index: pd.DatetimeIndex, decision_time: pd.Timestamp) -> int:
    return int(index.searchsorted(decision_time, side="right"))


def run_backtest(df: pd.DataFrame, model, initial_capital: float = 10000.0, risk_per_trade: float = 0.01, max_position_pct: float = 0.20, probability_threshold: float = 0.55) -> Dict[str, Any]:
    equity = float(initial_capital)
    equity_curve = [equity]
    peak = equity
    max_drawdown = 0.0
    trades = 0
    wins = 0
    unsupported_signals = 0
    total_return = 0.0
    feature_eligible = feature_row_eligibility_mask(df)
    execution_supported = (
        isinstance(df.index, pd.DatetimeIndex)
        and df.index.is_monotonic_increasing
        and not df.index.has_duplicates
        and str(df.attrs.get("bar_label", "left")).lower() == "left"
        and "Open" in df.columns
    )

    decision_times = feature_row_decision_times(df)
    decision_events: dict[pd.Timestamp, list[int]] = {}
    for position in range(1, len(df) - 1):
        if not feature_eligible.iloc[position]:
            continue
        decision_time = decision_times.iloc[position]
        if pd.notna(decision_time):
            decision_events.setdefault(pd.Timestamp(decision_time), []).append(position)

    bar_positions = {
        pd.Timestamp(timestamp): position
        for position, timestamp in enumerate(df.index)
    } if isinstance(df.index, pd.DatetimeIndex) else {}
    timeline = sorted(set(bar_positions).union(decision_events))
    scheduled_pnl: dict[int, list[tuple[float, float]]] = {}

    for timestamp in timeline:
        bar_position = bar_positions.get(timestamp)
        if bar_position is not None:
            for pnl, trade_return in scheduled_pnl.pop(bar_position, []):
                equity = max(equity + pnl, 0.0)
                trades += 1
                if trade_return > 0.0:
                    wins += 1
            peak = max(peak, equity)
            max_drawdown = max(
                max_drawdown,
                (peak - equity) / peak if peak > 0 else 0.0,
            )

        for position in decision_events.get(timestamp, []):
            row = df.iloc[position]
            feature_values = row[FEATURE_COLUMNS]
            feature_vector = feature_values.to_frame().T
            probability = float(model.predict_proba(feature_vector)[0][1])
            daily_return = float(row["ret_1"])
            trend_alignment = float(row["sma_fast"] - row["sma_slow"])
            session_alignment = float(row["session_bias"])
            regime_bias = float(row.get("regime_bias", 0.0))
            macro_bias = float(row.get("macro_bias", 0.0))
            liquidity_bias = float(row.get("liquidity_sweep", 0.0))
            trend_bias = 1.0 if trend_alignment >= 0 else -1.0

            long_ok = probability >= probability_threshold and (trend_bias >= 0 or regime_bias >= 0) and macro_bias >= 0 and session_alignment >= 0
            short_ok = probability <= (1.0 - probability_threshold) and (trend_bias <= 0 or regime_bias <= 0) and macro_bias <= 0 and session_alignment <= 0
            liquidity_filter = liquidity_bias == 0.0 or liquidity_bias == regime_bias

            if long_ok and liquidity_filter:
                signal = 1.0
            elif short_ok and liquidity_filter:
                signal = -1.0
            else:
                signal = 0.0

            if signal == 0.0:
                continue
            if not execution_supported:
                unsupported_signals += 1
                continue

            try:
                entry_position = _first_post_decision_position(df.index, timestamp)
            except (TypeError, ValueError):
                unsupported_signals += 1
                continue
            exit_position = entry_position + 1
            if exit_position >= len(df):
                unsupported_signals += 1
                continue
            entry_price = float(df["Open"].iloc[entry_position])
            exit_price = float(df["Open"].iloc[exit_position])
            if (
                not np.isfinite(entry_price)
                or not np.isfinite(exit_price)
                or entry_price <= 0.0
                or exit_price <= 0.0
            ):
                unsupported_signals += 1
                continue

            trade_return = signal * (exit_price / entry_price - 1.0)
            position_weight = min(max_position_pct, risk_per_trade / max(abs(daily_return), 0.001))
            position_weight = max(0.0, min(position_weight, max_position_pct))
            pnl = equity * (trade_return * position_weight)
            scheduled_pnl.setdefault(exit_position, []).append((pnl, trade_return))

        if bar_position is not None:
            equity_curve.append(equity)

    total_return = (equity / initial_capital) - 1.0
    return {
        "initial_capital": float(initial_capital),
        "final_capital": float(equity),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "trades": int(trades),
        "wins": int(wins),
        "unsupported_signals": int(unsupported_signals),
        "win_rate": (wins / trades) if trades else 0.0,
        "equity_curve": equity_curve,
    }
