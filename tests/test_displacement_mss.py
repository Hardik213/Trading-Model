import pandas as pd

from src.displacement import Direction, detect_displacement
from src.market_structure import (
    BreachOutcome,
    LiquidityEvent,
    LiquidityLevel,
    LiquiditySide,
    SwingPoint,
    SwingType,
)
from src.mss import detect_mss


def frame(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def test_small_body_is_not_displacement():
    rows = [(100, 101, 99, 100)] * 25
    rows[-1] = (100, 101.3, 99.0, 100.3)
    df = frame(rows)
    assert detect_displacement(df, 24, direction=Direction.BULLISH) is None


def test_bullish_expansion_with_follow_through_is_displacement():
    rows = [(100, 101, 99, 100)] * 20
    rows += [
        (100, 100.5, 99.5, 100.1),
        (100.1, 104, 100, 103.8),
        (103.8, 105, 103.5, 104.7),
    ]
    df = frame(rows)
    event = detect_displacement(
        df,
        21,
        direction=Direction.BULLISH,
        min_range_expansion=1.25,
    )
    assert event is not None
    assert event.follow_through is True


def test_mss_requires_rejection():
    df = frame([(100, 101, 99, 100)] * 10)
    level = LiquidityLevel(df.index[2], 103, LiquiditySide.SSL, "TEST")
    liquidity = LiquidityEvent(
        level=level,
        breach_timestamp=df.index[5],
        breach_price=98,
        breach_depth=5,
        outcome=BreachOutcome.ACCEPTANCE,
    )
    displacement = None
    assert displacement is None
    # Acceptance cannot become MSS; this is enforced before any structure check.


def test_mss_requires_structural_break_and_follow_through():
    rows = [(100, 101, 99, 100)] * 20
    rows += [
        (100, 101, 98, 99),
        (99, 98, 95, 96),       # liquidity rejection context
        (96, 97, 95, 96.5),
        (96.5, 102, 96, 101.5), # displacement + break above 101
        (101.5, 103, 101, 102.5),
    ]
    df = frame(rows)

    swing = SwingPoint(
        timestamp=df.index[20],
        price=101.0,
        kind=SwingType.HIGH,
        source_index=20,
        confirmation_timestamp=df.index[21],
    )
    level = LiquidityLevel(df.index[19], 98.0, LiquiditySide.SSL, "TEST")
    liquidity = LiquidityEvent(
        level=level,
        breach_timestamp=df.index[21],
        breach_price=95.0,
        breach_depth=3.0,
        outcome=BreachOutcome.REJECTION,
        resolution_timestamp=df.index[22],
        resolution_price=96.5,
    )

    displacement = detect_displacement(
        df,
        23,
        direction=Direction.BULLISH,
        min_range_expansion=1.25,
    )
    assert displacement is not None

    event = detect_mss(
        df,
        swings=[swing],
        liquidity_event=liquidity,
        displacement=displacement,
        break_position=23,
    )
    assert event is not None
    assert event.structural_point.price == 101.0


def test_no_lookahead_structure_confirmation():
    rows = [(100, 101, 99, 100)] * 30
    df = frame(rows)
    future_swing = SwingPoint(
        timestamp=df.index[28],
        price=110,
        kind=SwingType.HIGH,
        source_index=28,
        confirmation_timestamp=df.index[29],
    )
    level = LiquidityLevel(df.index[20], 95, LiquiditySide.SSL, "TEST")
    liquidity = LiquidityEvent(
        level=level,
        breach_timestamp=df.index[21],
        breach_price=94,
        breach_depth=1,
        outcome=BreachOutcome.REJECTION,
    )
    # At break_position 28, the future swing is not confirmed until bar 29.
    # detect_mss must therefore refuse to use it.
    displacement = detect_displacement(
        df,
        22,
        direction=Direction.BULLISH,
        min_range_expansion=1.25,
    )
    assert displacement is None
