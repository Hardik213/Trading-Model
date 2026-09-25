import pandas as pd

from src.dealing_range import create_dealing_range
from src.fvg import FVG, FVGDirection
from src.ict2022_engine import ICT2022StateMachine, SetupState
from src.market_structure import (
    BreachOutcome,
    LiquidityEvent,
    LiquidityLevel,
    LiquiditySide,
)
from src.mss import Direction


def ts(minute):
    return pd.Timestamp(f"2026-01-01T10:{minute:02d}:00Z")


def level(side, price):
    return LiquidityLevel(ts(0), price, side, "TEST")


def event(side, price, outcome):
    return LiquidityEvent(
        level=level(side, price),
        breach_timestamp=ts(10),
        breach_price=price + (1 if side is LiquiditySide.BSL else -1),
        breach_depth=1,
        outcome=outcome,
        resolution_timestamp=ts(11),
        resolution_price=price,
    )


def fvg(direction):
    return FVG(
        direction=direction,
        formation_timestamp=ts(20),
        confirmation_timestamp=ts(20),
        first_candle_timestamp=ts(18),
        third_candle_timestamp=ts(20),
        lower_bound=100,
        upper_bound=102,
        midpoint=101,
        size=2,
    )


def test_no_draw_no_trade():
    sm = ICT2022StateMachine()
    d = sm.identify_context(
        as_of=ts(5),
        direction=Direction.BULLISH,
        dealing_range=None,
        draw_on_liquidity=None,
    )
    assert d.state is SetupState.DEVELOPING


def test_acceptance_does_not_become_bullish_reversal():
    sm = ICT2022StateMachine()
    sm.identify_context(
        as_of=ts(5),
        direction=Direction.BULLISH,
        dealing_range=None,
        draw_on_liquidity=level(LiquiditySide.BSL, 110),
    )
    d = sm.process_liquidity_event(
        event(LiquiditySide.SSL, 95, BreachOutcome.ACCEPTANCE),
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.NO_TRADE


def test_wrong_liquidity_side_rejected():
    sm = ICT2022StateMachine()
    sm.identify_context(
        as_of=ts(5),
        direction=Direction.BULLISH,
        dealing_range=None,
        draw_on_liquidity=level(LiquiditySide.BSL, 110),
    )
    d = sm.process_liquidity_event(
        event(LiquiditySide.BSL, 110, BreachOutcome.REJECTION),
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.NO_TRADE


def test_pd_array_cannot_qualify_before_mss():
    sm = ICT2022StateMachine()
    d = sm.identify_pd_array(
        pd_array=fvg(FVGDirection.BULLISH),
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.WAITING


def test_retracement_into_fvg_triggers_entry():
    sm = ICT2022StateMachine()
    sm.state = SetupState.MSS_CONFIRMED
    sm.history = [SetupState.WAITING, SetupState.MSS_CONFIRMED]
    d = sm.identify_pd_array(
        pd_array=fvg(FVGDirection.BULLISH),
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.RETRACEMENT_WAIT

    d = sm.evaluate_retracement(
        current_price=101,
        pd_array=fvg(FVGDirection.BULLISH),
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.ENTRY_TRIGGERED


def test_entry_without_structural_invalidation_is_no_trade():
    sm = ICT2022StateMachine()
    sm.state = SetupState.ENTRY_TRIGGERED
    d = sm.activate_trade(
        entry=101,
        invalidation=None,
        target=110,
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.NO_TRADE


def test_invalid_bullish_geometry_is_no_trade():
    sm = ICT2022StateMachine()
    sm.state = SetupState.ENTRY_TRIGGERED
    d = sm.activate_trade(
        entry=101,
        invalidation=102,
        target=110,
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.NO_TRADE


def test_active_trade_target_and_stop_same_bar_is_ambiguous():
    sm = ICT2022StateMachine()
    sm.state = SetupState.ACTIVE_TRADE
    d = sm.resolve_trade(
        current_high=110,
        current_low=95,
        invalidation=98,
        target=108,
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.AMBIGUOUS_DATA


def test_active_trade_resolves_target():
    sm = ICT2022StateMachine()
    sm.state = SetupState.ACTIVE_TRADE
    d = sm.resolve_trade(
        current_high=110,
        current_low=100,
        invalidation=98,
        target=108,
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.TARGET_REACHED


def test_active_trade_resolves_invalidation():
    sm = ICT2022StateMachine()
    sm.state = SetupState.ACTIVE_TRADE
    d = sm.resolve_trade(
        current_high=104,
        current_low=97,
        invalidation=98,
        target=108,
        direction=Direction.BULLISH,
    )
    assert d.state is SetupState.INVALIDATED
