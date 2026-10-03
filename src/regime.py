from __future__ import annotations

import numpy as np
import pandas as pd


def add_multitimeframe_regime(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "mtf_trend_4h" not in data.columns:
        data["mtf_trend_4h"] = (data["Close"] - data["Close"].shift(4)) / data["Close"].shift(4)
    data["regime_score"] = data["mtf_trend_4h"] + 0.5 * data["trend_strength"]
    data["regime_bias"] = np.sign(data["regime_score"])
    if "mtf_decision_time" in data.columns:
        decision_time = pd.DatetimeIndex(data["mtf_decision_time"])
        trend_available = pd.DatetimeIndex(
            data.get("mtf_trend_4h_available_at", data["mtf_decision_time"])
        )
        regime_available = pd.DatetimeIndex(
            [
                max(trend_available[position], decision_time[position])
                if np.isfinite(data["regime_score"].iloc[position])
                and not pd.isna(trend_available[position])
                and not pd.isna(decision_time[position])
                else pd.NaT
                for position in range(len(data))
            ]
        )
        data["regime_score_available_at"] = pd.Series(regime_available, index=data.index)
        data["regime_bias_available_at"] = pd.Series(regime_available, index=data.index)
    return data
