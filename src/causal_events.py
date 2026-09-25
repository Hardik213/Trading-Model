from __future__ import annotations
import pandas as pd

def event_is_confirmed_as_of(event, as_of):
    if event is None:
        return False
    ts = getattr(event, "confirmation_timestamp", None)
    return ts is not None and pd.Timestamp(ts) <= pd.Timestamp(as_of)
