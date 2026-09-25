from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from src.features import FEATURE_COLUMNS


def run_backtest(df: pd.DataFrame, model, initial_capital: float = 10000.0, risk_per_trade: float = 0.01, max_position_pct: float = 0.20, probability_threshold: float = 0.55) -> Dict[str, Any]:
    equity = float(initial_capital)
    equity_curve = [equity]
    peak = equity
    max_drawdown = 0.0
    trades = 0
    wins = 0
    total_return = 0.0

    for idx in range(1, len(df) - 1):
        row = df.iloc[idx]
        feature_vector = row[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0).to_frame().T
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

        if signal != 0.0:
            position_weight = min(max_position_pct, risk_per_trade / max(abs(daily_return), 0.001))
            position_weight = max(0.0, min(position_weight, max_position_pct))
            pnl = equity * (signal * daily_return * position_weight)
            equity = max(equity + pnl, 0.0)
            trades += 1
            if signal * daily_return > 0:
                wins += 1

        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak > 0 else 0.0)
        equity_curve.append(equity)

    total_return = (equity / initial_capital) - 1.0
    return {
        "initial_capital": float(initial_capital),
        "final_capital": float(equity),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "trades": int(trades),
        "wins": int(wins),
        "win_rate": (wins / trades) if trades else 0.0,
        "equity_curve": equity_curve,
    }
