from __future__ import annotations

import numpy as np
import pandas as pd


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    hour = data.index.hour

    session_names = np.select(
        [
            (hour >= 0) & (hour < 7),
            (hour >= 7) & (hour < 13),
            (hour >= 13) & (hour < 17),
            (hour >= 17) & (hour < 21),
            (hour >= 21) | (hour < 0),
        ],
        ["Asia", "London", "New York", "Overlap", "Asia"],
        default="Other",
    )
    data["session"] = session_names
    data["session_long"] = np.where(data["session"].isin(["London", "New York", "Overlap"]), 1.0, 0.0)
    data["session_short"] = np.where(data["session"].isin(["Asia", "Other"]), 1.0, 0.0)
    data["session_bias"] = np.where(
        data["Close"] > data["Close"].rolling(12).mean(), 1.0,
        np.where(data["Close"] < data["Close"].rolling(12).mean(), -1.0, 0.0)
    )
    return data
