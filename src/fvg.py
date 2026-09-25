from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import pandas as pd

class FVGDirection(str, Enum):
    BULLISH="BULLISH"
    BEARISH="BEARISH"

@dataclass(frozen=True)
class FVG:
    direction: FVGDirection
    formation_timestamp: pd.Timestamp
    confirmation_timestamp: pd.Timestamp
    first_candle_timestamp: pd.Timestamp
    third_candle_timestamp: pd.Timestamp
    lower_bound: float
    upper_bound: float
    midpoint: float
    size: float
    def contains(self, price: float) -> bool:
        return self.lower_bound <= price <= self.upper_bound

def detect_fvgs(df: pd.DataFrame) -> list[FVG]:
    required={"Open","High","Low","Close"}
    missing=required.difference(df.columns)
    if missing: raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if not isinstance(df.index,pd.DatetimeIndex): raise TypeError("OHLC index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing: raise ValueError("OHLC index must be sorted ascending.")
    out=[]
    for i in range(2,len(df)):
        first=df.iloc[i-2]; third=df.iloc[i]
        if float(third["Low"]) > float(first["High"]):
            lo=float(first["High"]); hi=float(third["Low"])
            out.append(FVG(FVGDirection.BULLISH,df.index[i],df.index[i],df.index[i-2],df.index[i],lo,hi,(lo+hi)/2,hi-lo))
        if float(third["High"]) < float(first["Low"]):
            lo=float(third["High"]); hi=float(first["Low"])
            out.append(FVG(FVGDirection.BEARISH,df.index[i],df.index[i],df.index[i-2],df.index[i],lo,hi,(lo+hi)/2,hi-lo))
    return out

def active_fvgs(fvgs, *, as_of: pd.Timestamp):
    return [f for f in fvgs if f.confirmation_timestamp <= as_of]
