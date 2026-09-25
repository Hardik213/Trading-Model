from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "sma_fast",
    "sma_slow",
    "ema_fast",
    "ema_slow",
    "macd",
    "signal",
    "rsi",
    "atr",
    "volatility",
    "trend_strength",
    "close_z",
    "session_bias",
    "liquidity_gap",
    "regime_score",
    "macro_score",
    "liquidity_sweep",
    "structure_strength",
    "swing_delta",
    "htf_high_24",
    "htf_low_24",
    "htf_liquidity_obj",
    "liquidity_raid_long",
    "liquidity_raid_short",
    "displacement",
    "mss_bull",
    "mss_bear",
    "pd_array_long",
    "pd_array_short",
    "structure_valid_long",
    "structure_valid_short",
    "rr_ratio",
]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ret_1"] = data["Close"].pct_change().fillna(0.0)
    data["ret_5"] = data["Close"].pct_change(5).fillna(0.0)
    data["sma_fast"] = data["Close"].rolling(window=10).mean()
    data["sma_slow"] = data["Close"].rolling(window=30).mean()
    data["ema_fast"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["ema_slow"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_fast"] - data["ema_slow"]
    data["signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    delta = data["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / down.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    data["rsi"] = 100 - (100 / (1 + rs)).fillna(50.0)

    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close = (data["Low"] - data["Close"].shift(1)).abs()
    data["true_range"] = high_low.combine(high_close, max).combine(low_close, max)
    data["atr"] = data["true_range"].rolling(14).mean().fillna(0.0)
    data["volatility"] = data["ret_1"].rolling(20).std().fillna(0.0)
    data["trend_strength"] = ((data["Close"] - data["Close"].shift(20)) / data["Close"].shift(20)).fillna(0.0)
    rolling_mean = data["Close"].rolling(20).mean()
    rolling_std = data["Close"].rolling(20).std().replace(0, np.nan)
    data["close_z"] = ((data["Close"] - rolling_mean) / rolling_std).fillna(0.0)

    session_high = data["High"].rolling(window=24).max()
    session_low = data["Low"].rolling(window=24).min()
    data["session_bias"] = np.where(
        data["Close"] > session_high.shift(1), 1.0,
        np.where(data["Close"] < session_low.shift(1), -1.0, 0.0)
    )
    data["liquidity_gap"] = ((data["Close"] - data["Close"].shift(24)) / data["Close"].shift(24)).fillna(0.0)
    data["structure_strength"] = ((data["Close"] - data["Close"].shift(12)) / data["Close"].shift(12)).fillna(0.0)
    data["swing_delta"] = ((data["High"].rolling(12).max() - data["Low"].rolling(12).min()) / data["Close"]).fillna(0.0)
    return data


def add_ict_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    htf_high = data["High"].rolling(24).max().shift(1)
    htf_low = data["Low"].rolling(24).min().shift(1)
    data["htf_high_24"] = htf_high
    data["htf_low_24"] = htf_low
    data["htf_mid_24"] = (htf_high + htf_low) / 2.0
    data["htf_liquidity_obj"] = np.where(data["Close"] >= data["htf_mid_24"], htf_high, htf_low)

    data["liquidity_raid_long"] = (data["High"] > htf_high) & (data["Close"] < htf_high)
    data["liquidity_raid_short"] = (data["Low"] < htf_low) & (data["Close"] > htf_low)

    atr = data["atr"].replace(0, np.nan).ffill().fillna(1.0)
    data["displacement"] = (data["Close"] - data["Close"].shift(5)).abs() / atr

    swing_high = data["High"].shift(1).rolling(8).max()
    swing_low = data["Low"].shift(1).rolling(8).min()
    data["mss_bull"] = (data["Close"] > swing_high) & (data["Low"] > data["Low"].shift(1).rolling(8).min())
    data["mss_bear"] = (data["Close"] < swing_low) & (data["High"] < data["High"].shift(1).rolling(8).max())

    risk_buffer = atr * 0.8
    data["pd_array_long"] = (
        (data["Close"] > data["Close"].shift(5) - risk_buffer)
        & (data["Close"] < data["Close"].shift(5) + risk_buffer)
        & (data["Close"] > data["Low"].shift(1).rolling(5).min())
    )
    data["pd_array_short"] = (
        (data["Close"] < data["Close"].shift(5) + risk_buffer)
        & (data["Close"] > data["Close"].shift(5) - risk_buffer)
        & (data["Close"] < data["High"].shift(1).rolling(5).max())
    )

    data["structure_valid_long"] = (data["Close"] > swing_low) & (data["Low"] > swing_low * 0.999)
    data["structure_valid_short"] = (data["Close"] < swing_high) & (data["High"] < swing_high * 1.001)

    long_risk = np.maximum(data["Close"] - htf_low, 1e-6)
    short_risk = np.maximum(htf_high - data["Close"], 1e-6)
    long_target = np.maximum(htf_high - data["Close"], 1e-6)
    short_target = np.maximum(data["Close"] - htf_low, 1e-6)
    data["rr_ratio"] = np.where(
        data["Close"] >= data["htf_mid_24"],
        long_target / long_risk,
        short_target / short_risk,
    )
    return data


def build_training_frame(df: pd.DataFrame, horizon: int = 12, threshold: float = 0.003) -> tuple[pd.DataFrame, pd.Series]:
    frame = df.copy().dropna(subset=FEATURE_COLUMNS)
    future_return = frame["Close"].shift(-horizon) / frame["Close"] - 1
    frame["future_return"] = future_return
    frame["label"] = (frame["future_return"] > threshold).astype(int)
    X = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = frame["label"].fillna(0).astype(int)
    return X, y
