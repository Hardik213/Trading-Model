import pandas as pd
from src.displacement import Direction, DisplacementEvent
from src.ict2022_engine import ICT2022StateMachine, SetupState
from src.market_structure import BreachOutcome, LiquidityEvent, LiquidityLevel, LiquiditySide
from src.mss import MSSEvent

def ts(m):
    return pd.Timestamp(f"2026-01-01T10:{m:02d}:00Z")

def rejection():
    level=LiquidityLevel(ts(0),95,LiquiditySide.SSL,"TEST")
    return LiquidityEvent(level,ts(10),94,1,BreachOutcome.REJECTION,ts(11),96)

def displacement(confirmation):
    return DisplacementEvent(
        timestamp=ts(20),direction=Direction.BULLISH,start_price=100,
        end_price=105,magnitude=5,body_ratio=.8,close_location=.9,
        range_expansion=2.0,follow_through=True,
        confirmation_timestamp=confirmation,
    )

def mss(confirmation, disp_confirmation):
    d=displacement(disp_confirmation)
    return MSSEvent(
        direction="LONG",timestamp=ts(22),broken_level=101,
        displacement=d,follow_through=True,
        confirmation_timestamp=confirmation,
    )

def prepare():
    sm=ICT2022StateMachine()
    sm.identify_context(
        as_of=ts(21),direction=Direction.BULLISH,
        dealing_range=None,draw_on_liquidity=LiquidityLevel(ts(0),110,LiquiditySide.BSL,"DRAW")
    )
    sm.process_liquidity_event(rejection(),direction=Direction.BULLISH)
    return sm

def test_engine_rejects_future_confirmed_mss_at_break_time():
    sm=prepare()
    d=sm.confirm_displacement_and_mss(
        mss=mss(ts(24),ts(24)),direction=Direction.BULLISH,as_of=ts(22)
    )
    assert d.state is SetupState.DEVELOPING

def test_engine_accepts_mss_once_all_required_evidence_is_visible():
    sm=prepare()
    d=sm.confirm_displacement_and_mss(
        mss=mss(ts(24),ts(24)),direction=Direction.BULLISH,as_of=ts(24)
    )
    assert d.state is SetupState.MSS_CONFIRMED

def test_mss_confirmation_includes_later_displacement_confirmation():
    sm=prepare()
    d=sm.confirm_displacement_and_mss(
        mss=mss(ts(23),ts(24)),direction=Direction.BULLISH,as_of=ts(23)
    )
    assert d.state is SetupState.DEVELOPING

def test_mss_can_only_be_confirmed_after_displacement_confirmation():
    sm=prepare()
    d=sm.confirm_displacement_and_mss(
        mss=mss(ts(24),ts(24)),direction=Direction.BULLISH,as_of=ts(24)
    )
    assert d.state is SetupState.MSS_CONFIRMED
