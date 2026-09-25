import pandas as pd

from src.causal_liquidity import (
    LiquidityEvidenceKind,
    confirmed_liquidity_map,
    draw_on_liquidity,
    latest_reversal_liquidity,
    liquidity_evidence_as_of,
)
from src.market_structure import LiquiditySide
from src.mss import Direction


def frame(rows):
    idx = pd.date_range("2026-01-01 10:00", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def test_swing_is_not_visible_before_confirmation():
    df = frame([
        (100, 101, 99, 100),
        (100, 105, 99, 104),  # swing high candidate
        (104, 103, 100, 101),
        (101, 102, 98, 99),   # right-side confirmation for the high
        (99, 100, 97, 98),
    ])
    early = confirmed_liquidity_map(df, df.index[2], left_bars=1, right_bars=1)
    later = confirmed_liquidity_map(df, df.index[3], left_bars=1, right_bars=1)
    assert not any(x.side is LiquiditySide.BSL and x.price == 105 for x in early)
    assert any(x.side is LiquiditySide.BSL and x.price == 105 for x in later)


def test_rejection_is_not_visible_until_resolution():
    # Swing low at index 1, confirmed at index 2. Price breaches SSL at index 3
    # and only rejects it at index 4.
    df = frame([
        (100, 102, 99, 101),
        (101, 102, 95, 100),
        (100, 101, 97, 99),
        (99, 100, 94, 96),
        (96, 101, 95, 100),
    ])
    breach_time = df.index[3]
    resolution_time = df.index[4]
    before = liquidity_evidence_as_of(df, breach_time, side=LiquiditySide.SSL, left_bars=1, right_bars=1)
    after = liquidity_evidence_as_of(df, resolution_time, side=LiquiditySide.SSL, left_bars=1, right_bars=1)
    assert all(item.kind is not LiquidityEvidenceKind.REJECTION for item in before)
    assert any(item.kind is LiquidityEvidenceKind.REJECTION for item in after)


def test_latest_reversal_requires_opposing_rejection():
    df = frame([
        (100, 102, 99, 101),
        (101, 102, 95, 100),
        (100, 101, 97, 99),
        (99, 100, 94, 96),
        (96, 101, 95, 100),
    ])
    event = latest_reversal_liquidity(df, df.index[-1], Direction.BULLISH, left_bars=1, right_bars=1)
    assert event is not None
    assert event.level.side is LiquiditySide.SSL
    assert event.is_rejection


def test_pre_confirmation_breach_never_becomes_reversal_evidence():
    df = frame([
        (100, 101, 99, 100),
        (99, 100, 89, 98),
        (98, 99, 97, 98),
        (95, 96, 93, 94),
        (94, 95, 94, 94.5),
        (94, 101, 93, 100),
    ])
    assert not liquidity_evidence_as_of(
        df,
        df.index[4],
        side=LiquiditySide.SSL,
        left_bars=1,
        right_bars=1,
    )
    assert not liquidity_evidence_as_of(
        df,
        df.index[5],
        side=LiquiditySide.SSL,
        left_bars=1,
        right_bars=1,
    )


def test_draw_on_liquidity_uses_only_visible_confirmed_levels():
    df = frame([
        (100, 102, 99, 101),
        (101, 106, 100, 105),
        (105, 103, 101, 102),
        (102, 104, 99, 100),
        (100, 101, 98, 99),
    ])
    objective = draw_on_liquidity(
        df,
        df.index[-1],
        Direction.BULLISH,
        current_price=100.0,
        left_bars=1,
        right_bars=1,
    )
    assert objective is not None
    assert objective.level.side is LiquiditySide.BSL
    assert objective.level.price == 106
