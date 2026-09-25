import pandas as pd

from src.fvg import FVG, FVGDirection
from src.strategy_adapter import ICT2022StrategyAdapter, StrategyEvidence
from src.ict2022_engine import SetupState
from src.mss import Direction
from src.market_structure import BreachOutcome, LiquidityEvent, LiquidityLevel, LiquiditySide


def ts(m):
    return pd.Timestamp(f"2026-01-01T10:{m:02d}:00Z")


def level(side, price):
    return LiquidityLevel(ts(0), price, side, "TEST")


def rejection():
    return LiquidityEvent(
        level=level(LiquiditySide.SSL, 95),
        breach_timestamp=ts(10),
        breach_price=94,
        breach_depth=1,
        outcome=BreachOutcome.REJECTION,
        resolution_timestamp=ts(11),
        resolution_price=96,
    )


def bullish_mss():
    # Importing the real constructor keeps this test tied to the repository's
    # existing MSS contract rather than inventing a fake object.
    from src.displacement import DisplacementEvent
    from src.mss import MSSEvent

    displacement = DisplacementEvent(
        timestamp=ts(12),
        direction=Direction.BULLISH,
        start_price=99,
        end_price=101,
        magnitude=2,
        body_ratio=0.8,
        close_location=0.9,
        range_expansion=1.5,
        follow_through=True,
    )
    return MSSEvent(
        direction=Direction.BULLISH,
        timestamp=ts(13),
        broken_level=100,
        displacement=displacement,
        follow_through=True,
    )


def bullish_fvg():
    return FVG(
        direction=FVGDirection.BULLISH,
        formation_timestamp=ts(14),
        confirmation_timestamp=ts(14),
        first_candle_timestamp=ts(12),
        third_candle_timestamp=ts(14),
        lower_bound=100,
        upper_bound=102,
        midpoint=101,
        size=2,
    )


def evidence(**overrides):
    data = dict(
        timestamp=ts(15),
        direction=Direction.BULLISH,
        dealing_range=None,
        draw_on_liquidity=level(LiquiditySide.BSL, 110),
        liquidity_event=rejection(),
        mss=bullish_mss(),
        pd_array=bullish_fvg(),
        entry_price=101,
        invalidation_price=98,
        target_price=110,
        target_liquidity=level(LiquiditySide.BSL, 110),
    )
    data.update(overrides)
    return StrategyEvidence(**data)


def test_missing_direction_is_developing():
    d=ICT2022StrategyAdapter().evaluate(evidence(direction=None))
    assert d.state is SetupState.NO_TRADE or d.state is SetupState.DEVELOPING


def test_missing_liquidity_event_is_developing():
    d=ICT2022StrategyAdapter().evaluate(evidence(liquidity_event=None))
    assert d.state is SetupState.DEVELOPING


def test_acceptance_is_no_trade():
    from src.market_structure import BreachOutcome
    accepted=LiquidityEvent(
        level=level(LiquiditySide.SSL,95),
        breach_timestamp=ts(10),
        breach_price=94,
        breach_depth=1,
        outcome=BreachOutcome.ACCEPTANCE,
        resolution_timestamp=ts(11),
        resolution_price=97,
    )
    d=ICT2022StrategyAdapter().evaluate(evidence(liquidity_event=accepted))
    assert d.state is SetupState.NO_TRADE


def test_missing_mss_is_developing():
    d=ICT2022StrategyAdapter().evaluate(evidence(mss=None))
    assert d.state is SetupState.DEVELOPING


def test_missing_pd_array_is_no_trade():
    d=ICT2022StrategyAdapter().evaluate(evidence(pd_array=None))
    assert d.state is SetupState.NO_TRADE


def test_missing_structural_invalidation_is_no_trade():
    d=ICT2022StrategyAdapter().evaluate(evidence(invalidation_price=None))
    assert d.state is SetupState.NO_TRADE


def test_missing_target_is_no_trade():
    d=ICT2022StrategyAdapter().evaluate(evidence(target_price=None))
    assert d.state is SetupState.NO_TRADE


def test_complete_sequence_becomes_active_trade():
    d=ICT2022StrategyAdapter().evaluate(evidence())
    assert d.state is SetupState.ACTIVE_TRADE
    assert d.planned_r == 3.0


def test_wrong_pd_direction_is_no_trade():
    d=ICT2022StrategyAdapter().evaluate(
        evidence(pd_array=FVG(
            direction=FVGDirection.BEARISH,
            formation_timestamp=ts(14),
            confirmation_timestamp=ts(14),
            first_candle_timestamp=ts(12),
            third_candle_timestamp=ts(14),
            lower_bound=100,
            upper_bound=102,
            midpoint=101,
            size=2,
        ))
    )
    assert d.state is SetupState.NO_TRADE
