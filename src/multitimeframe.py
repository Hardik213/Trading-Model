from __future__ import annotations

import numpy as np
import pandas as pd


def add_multitimeframe_features(df: pd.DataFrame, base: str = "h") -> pd.DataFrame:
    data = df.copy()

    if len(data) < 4:
        data["mtf_trend_4h"] = 0.0
        data["mtf_bias_4h"] = 0.0
        data["mtf_strength_4h"] = 0.0
        return data

    resampled = data["Close"].resample("4h").last().ffill()
    mtf_4h = resampled.reindex(data.index, method="ffill")

    data["mtf_trend_4h"] = ((mtf_4h - mtf_4h.shift(1)) / mtf_4h.shift(1)).fillna(0.0)
    data["mtf_bias_4h"] = np.where(data["mtf_trend_4h"] > 0, 1.0, np.where(data["mtf_trend_4h"] < 0, -1.0, 0.0))
    data["mtf_strength_4h"] = data["mtf_trend_4h"].rolling(12).mean().fillna(0.0)

    daily = data["Close"].resample("1D").last().ffill()
    data["daily_trend"] = ((daily.reindex(data.index, method="ffill") - data["Close"]) / data["Close"]).fillna(0.0)
    return data
