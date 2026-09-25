from __future__ import annotations

import pandas as pd


DEFAULT_ENTRY_WINDOWS = {
    "Asia": (0, 7),
    "London": (7, 13),
    "New York": (13, 17),
    "Overlap": (13, 15),
}


def get_session_name(hour: int) -> str:
    for name, (start, end) in DEFAULT_ENTRY_WINDOWS.items():
        if start <= hour < end:
            return name
    return "Asia"


def add_session_entry_filter(df: pd.DataFrame, entry_windows=None) -> pd.DataFrame:
    data = df.copy()
    windows = entry_windows or DEFAULT_ENTRY_WINDOWS
    hours = data.index.hour
    data["session_name"] = [get_session_name(h) for h in hours]
    data["is_entry_session"] = False

    for session, (start, end) in windows.items():
        mask = data["session_name"] == session
        if session == "Overlap":
            mask = ((hours >= 13) & (hours < 15))
        data.loc[mask, "is_entry_session"] = True
    return data
