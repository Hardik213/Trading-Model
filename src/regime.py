from __future__ import annotations

import numpy as np
import pandas as pd


def add_multitimeframe_regime(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["mtf_trend_4h"] = ((data["Close"] - data["Close"].shift(4)) / data["Close"].shift(4)).fillna(0.0)
    data["regime_score"] = data["mtf_trend_4h"] + 0.5 * data["trend_strength"]
    data["regime_bias"] = np.where(
        data["regime_score"] > 0.0, 1.0,
        np.where(data["regime_score"] < 0.0, -1.0, 0.0)
    )
    return data
