import pandas as pd
from src.causal_events import event_is_confirmed_as_of
from src.displacement import Direction,detect_displacement,is_confirmed_as_of
from src.market_structure import BreachOutcome,LiquidityEvent,LiquidityLevel,LiquiditySide,SwingPoint,SwingType
from src.mss import detect_mss,is_confirmed_as_of as mss_confirmed

def frame(rows):
    idx=pd.date_range("2026-01-01",periods=len(rows),freq="5min",tz="UTC")
    return pd.DataFrame(rows,index=idx,columns=["Open","High","Low","Close"])

def test_displacement_confirmation_is_later_than_impulse():
    rows=[(100,101,99,100)]*20+[(100,100.5,99.5,100.1),(100.1,104,100,103.8),(103.8,105,103.5,104.7)]
    d=frame(rows); e=detect_displacement(d,21,direction=Direction.BULLISH,min_range_expansion=1.25)
    assert e is not None and e.timestamp==d.index[21] and e.follow_through and e.confirmation_timestamp==d.index[22]
    assert not is_confirmed_as_of(e,d.index[21]); assert is_confirmed_as_of(e,d.index[22])
    assert not event_is_confirmed_as_of(e,d.index[21]); assert event_is_confirmed_as_of(e,d.index[22])

def test_future_candle_does_not_change_earlier_observation():
    rows=[(100,101,99,100)]*20+[(100,100.5,99.5,100.1),(100.1,104,100,103.8),(103.8,105,103.5,104.7)]
    full=frame(rows); early=full.iloc[:22]
    a=detect_displacement(early,21,direction=Direction.BULLISH,min_range_expansion=1.25)
    b=detect_displacement(full,21,direction=Direction.BULLISH,min_range_expansion=1.25)
    assert a is not None and not a.follow_through and a.confirmation_timestamp is None
    assert b is not None and b.follow_through and b.confirmation_timestamp==full.index[22]

def test_mss_confirmation_is_later_than_break():
    rows=[(100,101,99,100)]*20+[(100,101,98,99),(99,98,95,96),(96,97,95,96.5),(96.5,102,96,101.5),(101.5,103,101,102.5)]
    d=frame(rows)
    swing=SwingPoint(d.index[20],101.0,SwingType.HIGH,20,d.index[21])
    level=LiquidityLevel(d.index[19],98.0,LiquiditySide.SSL,"TEST")
    liq=LiquidityEvent(level,d.index[21],95.0,3.0,BreachOutcome.REJECTION,d.index[22],96.5)
    disp=detect_displacement(d,23,direction=Direction.BULLISH,min_range_expansion=1.25)
    assert disp is not None
    e=detect_mss(d,swings=[swing],liquidity_event=liq,displacement=disp,break_position=23)
    assert e is not None and e.timestamp==d.index[23] and e.structural_point.price==101.0 and e.confirmation_timestamp==d.index[24]
    assert not mss_confirmed(e,d.index[23]); assert mss_confirmed(e,d.index[24])

def test_mss_not_available_before_confirmation_candle_is_visible():
    rows=[(100,101,99,100)]*20+[(100,101,98,99),(99,98,95,96),(96,97,95,96.5),(96.5,102,96,101.5),(101.5,103,101,102.5)]
    full=frame(rows); visible=full.iloc[:24]
    swing=SwingPoint(full.index[20],101.0,SwingType.HIGH,20,full.index[21])
    level=LiquidityLevel(full.index[19],98.0,LiquiditySide.SSL,"TEST")
    liq=LiquidityEvent(level,full.index[21],95.0,3.0,BreachOutcome.REJECTION,full.index[22],96.5)
    disp=detect_displacement(full,23,direction=Direction.BULLISH,min_range_expansion=1.25)
    assert disp is not None
    assert detect_mss(visible,swings=[swing],liquidity_event=liq,displacement=disp,break_position=23) is None
