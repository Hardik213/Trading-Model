"""No-lookahead helpers for completed OHLC bars."""
from __future__ import annotations
import pandas as pd

def completed_bar_cutoff(index: pd.DatetimeIndex, as_of: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """Latest left-labelled bar whose close is <= as_of."""
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    offset = pd.tseries.frequencies.to_offset(timeframe)
    labels = pd.DatetimeIndex(index)
    eligible = labels[labels + offset <= as_of]
    return eligible[-1] if len(eligible) else pd.NaT
