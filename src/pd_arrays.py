from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
from .dealing_range import DealingRange, RangeLocation
from .fvg import FVG, FVGDirection

class PDArrayType(str, Enum): FVG="FVG"

@dataclass(frozen=True)
class PDArray:
    array_type: PDArrayType
    direction: FVGDirection
    lower_bound: float
    upper_bound: float
    midpoint: float
    source_timestamp: pd.Timestamp
    range_location: Optional[RangeLocation]=None
    def contains(self, price: float): return self.lower_bound <= price <= self.upper_bound

def fvg_to_pd_array(fvg: FVG, *, dealing_range: Optional[DealingRange]=None):
    loc=dealing_range.location(fvg.midpoint) if dealing_range else None
    return PDArray(PDArrayType.FVG,fvg.direction,fvg.lower_bound,fvg.upper_bound,fvg.midpoint,fvg.confirmation_timestamp,loc)

def justified_fvgs(fvgs, *, direction: FVGDirection, as_of: pd.Timestamp, dealing_range=None):
    return [fvg_to_pd_array(f,dealing_range=dealing_range) for f in fvgs if f.confirmation_timestamp <= as_of and f.direction is direction]
