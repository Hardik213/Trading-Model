from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import pandas as pd

class RangeLocation(str, Enum):
    PREMIUM="PREMIUM"
    EQUILIBRIUM="EQUILIBRIUM"
    DISCOUNT="DISCOUNT"

@dataclass(frozen=True)
class DealingRange:
    anchor_high_timestamp: pd.Timestamp
    anchor_high: float
    anchor_low_timestamp: pd.Timestamp
    anchor_low: float
    equilibrium: float
    @property
    def size(self): return self.anchor_high-self.anchor_low
    def location(self, price: float, *, equilibrium_tolerance: float=0.0):
        if self.size <= 0: raise ValueError("Dealing range must have positive size.")
        if abs(price-self.equilibrium) <= equilibrium_tolerance: return RangeLocation.EQUILIBRIUM
        return RangeLocation.PREMIUM if price > self.equilibrium else RangeLocation.DISCOUNT

def create_dealing_range(*, high_timestamp, high_price, low_timestamp, low_price):
    if high_price <= low_price: raise ValueError("high_price must be greater than low_price.")
    return DealingRange(high_timestamp,float(high_price),low_timestamp,float(low_price),(float(high_price)+float(low_price))/2)
