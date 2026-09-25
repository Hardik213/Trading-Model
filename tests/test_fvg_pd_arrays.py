import pandas as pd
from src.fvg import FVGDirection, detect_fvgs, active_fvgs
from src.dealing_range import RangeLocation, create_dealing_range
from src.pd_arrays import justified_fvgs

def frame(rows):
    idx=pd.date_range("2026-01-01",periods=len(rows),freq="5min",tz="UTC")
    return pd.DataFrame(rows,index=idx,columns=["Open","High","Low","Close"])

def test_bullish_fvg():
    df=frame([(100,101,99,100),(100,105,100,104),(104,108,103,107)])
    x=[f for f in detect_fvgs(df) if f.direction is FVGDirection.BULLISH]
    assert len(x)==1 and x[0].lower_bound==101 and x[0].upper_bound==103

def test_bearish_fvg():
    df=frame([(100,101,99,100),(100,100,95,96),(96,97,92,93)])
    x=[f for f in detect_fvgs(df) if f.direction is FVGDirection.BEARISH]
    assert len(x)==1 and x[0].lower_bound==97 and x[0].upper_bound==99

def test_no_fvg_without_gap():
    df=frame([(100,102,99,101),(101,103,100,102),(102,104,101,103)])
    assert detect_fvgs(df)==[]

def test_no_lookahead_for_fvg():
    df=frame([(100,101,99,100),(100,105,100,104),(104,108,103,107)])
    fs=detect_fvgs(df)
    assert len(active_fvgs(fs,as_of=df.index[1]))==0
    assert len(active_fvgs(fs,as_of=df.index[2]))==1

def test_dealing_range_is_explicit():
    dr=create_dealing_range(high_timestamp=pd.Timestamp("2026-01-01T10:00Z"),high_price=120,low_timestamp=pd.Timestamp("2026-01-01T09:00Z"),low_price=100)
    assert dr.equilibrium==110
    assert dr.location(115) is RangeLocation.PREMIUM
    assert dr.location(105) is RangeLocation.DISCOUNT
    assert dr.location(110) is RangeLocation.EQUILIBRIUM

def test_discount_is_only_location():
    dr=create_dealing_range(high_timestamp=pd.Timestamp("2026-01-01T10:00Z"),high_price=120,low_timestamp=pd.Timestamp("2026-01-01T09:00Z"),low_price=100)
    assert dr.location(105) is RangeLocation.DISCOUNT

def test_pd_array_is_context_not_signal():
    df=frame([(100,101,99,100),(100,105,100,104),(104,108,103,107)])
    arr=justified_fvgs(detect_fvgs(df),direction=FVGDirection.BULLISH,as_of=df.index[2])
    assert len(arr)==1 and arr[0].array_type.value=="FVG"
