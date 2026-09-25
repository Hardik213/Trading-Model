from __future__ import annotations

import numpy as np
import pandas as pd


def add_liquidity_features(df: pd.DataFrame, lookback: int = 48) -> pd.DataFrame:
    data = df.copy()
    prev_high = data["High"].shift(1).rolling(lookback).max()
    prev_low = data["Low"].shift(1).rolling(lookback).min()

    data["prev_session_high"] = prev_high
    data["prev_session_low"] = prev_low
    data["liquidity_sweep_up"] = (data["High"] > prev_high).astype(int)
    data["liquidity_sweep_down"] = (data["Low"] < prev_low).astype(int)
    data["liquidity_sweep"] = np.where(
        data["liquidity_sweep_up"] == 1, 1.0,
        np.where(data["liquidity_sweep_down"] == 1, -1.0, 0.0)
    )
    data["liquidity_gap"] = ((data["High"] - data["Low"]).rolling(lookback).mean() / data["Close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data
