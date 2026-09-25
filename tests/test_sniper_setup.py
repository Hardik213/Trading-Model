import pandas as pd

from src.displacement import Direction, DisplacementEvent
from src.fvg import FVG, FVGDirection
from src.market_structure import BreachOutcome, LiquidityEvent, LiquidityLevel, LiquiditySide
from src.mss import MSSEvent
from src.sniper_setup import GateReason, PrecisionEvidence, PrecisionState, evaluate_precision_setup


def ts(i):
    return pd.Timestamp("2026-01-01 10:00", tz="UTC") + pd.Timedelta(minutes=5*i)


def levels():
    ssl = LiquidityLevel(ts(0), 95.0, LiquiditySide.SSL, "SWING")
    bsl = LiquidityLevel(ts(1), 110.0, LiquiditySide.BSL, "SWING")
    return ssl, bsl


def valid_evidence():
    ssl, bsl = levels()
    event = LiquidityEvent(ssl, ts(4), 94.0, 1.0, BreachOutcome.REJECTION, ts(5), 98.0)
    disp = DisplacementEvent(ts(5), Direction.BULLISH, 98.0, 104.0, 6.0, .8, .9, 2.0, True, True, ts(6))
    mss = MSSEvent("LONG", ts(6), 103.0, disp, True, ts(7), None)
    fvg = FVG(FVGDirection.BULLISH, ts(7), ts(7), ts(5), ts(7), 100.0, 102.0, 101.0, 2.0)
    return PrecisionEvidence(ts(8), Direction.BULLISH, bsl, event, mss, fvg, 101.0, 97.0, 109.0, bsl)


def test_valid_sequence():
    d = evaluate_precision_setup(valid_evidence())
    assert d.state is PrecisionState.VALID
    assert d.reason is GateReason.VALID_SEQUENCE
    assert d.planned_r == 2.0


def test_future_mss_confirmation_is_developing():
    e = valid_evidence()
    e = PrecisionEvidence(ts(6), e.direction, e.draw_on_liquidity, e.liquidity_event, e.mss, e.pd_array,
                          e.entry_price, e.invalidation_price, e.target_price, e.target_liquidity)
    d = evaluate_precision_setup(e)
    assert d.state is PrecisionState.DEVELOPING
    assert d.reason is GateReason.FUTURE_MSS_CONFIRMATION


def test_future_displacement_confirmation_is_developing_even_if_mss_is_confirmed():
    e = valid_evidence()
    mss = MSSEvent(e.mss.direction, e.mss.timestamp, e.mss.broken_level, e.mss.displacement, True, ts(6), None)
    d = evaluate_precision_setup(PrecisionEvidence(ts(6), e.direction, e.draw_on_liquidity, e.liquidity_event, mss, e.pd_array,
                                                   e.entry_price, e.invalidation_price, e.target_price, e.target_liquidity))
    assert d.state is PrecisionState.DEVELOPING
    assert d.reason is GateReason.FUTURE_DISPLACEMENT_CONFIRMATION


def test_acceptance_invalidates_reversal():
    e = valid_evidence()
    event = LiquidityEvent(e.liquidity_event.level, ts(4), 94.0, 1.0, BreachOutcome.ACCEPTANCE, ts(5), 94.5)
    d = evaluate_precision_setup(PrecisionEvidence(e.as_of, e.direction, e.draw_on_liquidity, event, e.mss, e.pd_array,
                                                   e.entry_price, e.invalidation_price, e.target_price, e.target_liquidity))
    assert d.state is PrecisionState.INVALID
    assert d.reason is GateReason.LIQUIDITY_ACCEPTED


def test_wrong_pd_direction_invalid():
    e = valid_evidence()
    fvg = FVG(FVGDirection.BEARISH, e.pd_array.formation_timestamp, e.pd_array.confirmation_timestamp,
              e.pd_array.first_candle_timestamp, e.pd_array.third_candle_timestamp,
              e.pd_array.lower_bound, e.pd_array.upper_bound, e.pd_array.midpoint, e.pd_array.size)
    d = evaluate_precision_setup(PrecisionEvidence(e.as_of, e.direction, e.draw_on_liquidity, e.liquidity_event, e.mss, fvg,
                                                   e.entry_price, e.invalidation_price, e.target_price, e.target_liquidity))
    assert d.state is PrecisionState.INVALID
    assert d.reason is GateReason.PD_DIRECTION_MISMATCH


def test_retracement_outside_pd_array_is_not_trade():
    e = valid_evidence()
    d = evaluate_precision_setup(PrecisionEvidence(e.as_of, e.direction, e.draw_on_liquidity, e.liquidity_event, e.mss, e.pd_array,
                                                   103.0, e.invalidation_price, e.target_price, e.target_liquidity))
    assert d.state is PrecisionState.DEVELOPING
    assert d.reason is GateReason.RETRACEMENT_NOT_IN_PD_ARRAY


def test_target_must_be_opposing_liquidity():
    e = valid_evidence()
    wrong = LiquidityLevel(ts(2), 90.0, LiquiditySide.SSL, "SWING")
    d = evaluate_precision_setup(PrecisionEvidence(e.as_of, e.direction, e.draw_on_liquidity, e.liquidity_event, e.mss, e.pd_array,
                                                   e.entry_price, e.invalidation_price, e.target_price, wrong))
    assert d.state is PrecisionState.INVALID
    assert d.reason is GateReason.TARGET_NOT_DRAW


def test_optional_r_threshold_does_not_change_default_rule():
    e = valid_evidence()
    assert evaluate_precision_setup(e).state is PrecisionState.VALID
    d = evaluate_precision_setup(e, min_planned_r=2.5)
    assert d.state is PrecisionState.NO_TRADE
    assert d.reason is GateReason.R_MULTIPLE_TOO_LOW


def test_pd_array_must_belong_to_post_mss_sequence():
    e = valid_evidence()
    old = FVG(FVGDirection.BULLISH, ts(5), ts(7), ts(3), ts(5), 100.0, 102.0, 101.0, 2.0)
    d = evaluate_precision_setup(PrecisionEvidence(e.as_of, e.direction, e.draw_on_liquidity, e.liquidity_event, e.mss, old,
                                                   e.entry_price, e.invalidation_price, e.target_price, e.target_liquidity))
    assert d.state is PrecisionState.INVALID
    assert d.reason is GateReason.PD_ARRAY_PRECEDES_MSS
