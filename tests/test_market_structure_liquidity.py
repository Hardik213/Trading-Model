import pandas as pd

from src.market_structure import (
    BreachOutcome,
    LiquidityLevel,
    LiquiditySide,
    SwingType,
    detect_confirmed_swings,
    resolve_breach,
)


def make_ohlc(rows):
    index = pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"])


def test_swing_is_not_available_until_confirmation():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 101, 98, 99),
        (99, 100, 97, 98),
        (98, 99, 96, 97),
    ])
    swings = detect_confirmed_swings(df, left_bars=1, right_bars=1)
    highs = [s for s in swings if s.kind is SwingType.HIGH]
    assert highs
    assert highs[0].timestamp < highs[0].confirmation_timestamp


def test_bsl_wick_is_not_automatically_a_sweep():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 104, 101, 103),  # breach and close above = acceptance
        (103, 105, 102, 104),
    ])
    level = LiquidityLevel(
        timestamp=df.index[1],
        price=103.0,
        side=LiquiditySide.BSL,
        source="TEST",
    )
    event = resolve_breach(df, level, 2, max_resolution_bars=2)
    assert event.outcome is BreachOutcome.ACCEPTANCE


def test_bsl_rejection_requires_close_back_below():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 102),  # breach
        (102, 103, 100, 101),  # close back below = rejection
    ])
    level = LiquidityLevel(
        timestamp=df.index[1],
        price=103.0,
        side=LiquiditySide.BSL,
        source="TEST",
    )
    event = resolve_breach(df, level, 2, max_resolution_bars=2)
    assert event.outcome is BreachOutcome.REJECTION


def test_unresolved_breach_is_not_forced_into_sweep():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 103),  # breach
        (103, 106, 102, 104),  # still above
        (104, 107, 103, 105),  # still above
    ])
    level = LiquidityLevel(
        timestamp=df.index[1],
        price=103.0,
        side=LiquiditySide.BSL,
        source="TEST",
    )
    event = resolve_breach(df, level, 2, max_resolution_bars=2)
    assert event.outcome is BreachOutcome.ACCEPTANCE
