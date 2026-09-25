from __future__ import annotations

import numpy as np
import pandas as pd


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["macro_bias"] = np.where(
        data["trend_strength"] > 0.0, 1.0,
        np.where(data["trend_strength"] < 0.0, -1.0, 0.0)
    )
    data["macro_regime"] = np.where(
        data["volatility"] < data["volatility"].rolling(20).median(), 1.0,
        np.where(data["volatility"] > data["volatility"].rolling(20).median(), -1.0, 0.0)
    )
    data["macro_score"] = (data["trend_strength"] + data["ret_5"]) / 2.0
    return data
