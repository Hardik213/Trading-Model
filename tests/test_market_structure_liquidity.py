import pandas as pd
import pytest

from src.market_structure import (
    BreachOutcome,
    LiquidityLevel,
    LiquiditySide,
    SwingType,
    detect_confirmed_swings,
    resolve_breach,
)
from src.liquidity import add_liquidity_features


def make_ohlc(rows):
    index = pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC")
    result = pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"])
    result.attrs["bar_label"] = "right"
    return result


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
    assert event.availability_timestamp == df.index[3]


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


def test_first_resolved_outcome_is_not_relabelled_by_later_bars():
    df = make_ohlc([
        (100, 101, 99, 100),
        (105, 106, 104, 106),  # breach close accepts above BSL
        (106, 108, 105, 107),  # subsequent close confirms acceptance
        (107, 108, 103, 104),  # later close returns below the level
    ])
    level = LiquidityLevel(df.index[0], 105.0, LiquiditySide.BSL, "TEST")

    full = resolve_breach(df, level, 1, max_resolution_bars=2)
    prefix = resolve_breach(df.iloc[:3], level, 1, max_resolution_bars=2)

    assert full.outcome is BreachOutcome.ACCEPTANCE
    assert full.resolution_timestamp == df.index[1]
    assert prefix == full


def test_rejection_feature_activates_on_confirmation_not_breach_row():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 105, 100, 104),
        (104, 104, 99, 100),
        (100, 106, 100, 105),  # BSL breach
        (105, 104, 99, 104),  # subsequent rejection confirmation
    ])

    result = add_liquidity_features(
        df,
        left_bars=1,
        right_bars=1,
        resolution_bars=2,
    )

    assert result["liquidity_breach_up"].iloc[3] == 1
    assert result["liquidity_rejection_up"].iloc[3] == 0
    assert result["liquidity_sweep"].iloc[3] == 0.0
    assert result["liquidity_rejection_up"].iloc[4] == 1
    assert result["liquidity_sweep"].iloc[4] == 1.0


def test_liquidity_features_match_incremental_prefixes():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 105, 100, 104),
        (104, 103, 99, 100),
        (100, 106, 100, 105),
        (105, 106, 99, 104),
        (104, 105, 98, 100),
    ])
    columns = [
        "liquidity_breach_up",
        "liquidity_breach_down",
        "liquidity_rejection_up",
        "liquidity_rejection_down",
        "liquidity_acceptance_up",
        "liquidity_acceptance_down",
        "liquidity_sweep",
    ]

    full = add_liquidity_features(df, left_bars=1, right_bars=1)
    for stop in range(1, len(df) + 1):
        prefix = add_liquidity_features(df.iloc[:stop], left_bars=1, right_bars=1)
        pd.testing.assert_frame_equal(full[columns].iloc[:stop], prefix[columns])


def test_delayed_confirmation_is_marked_at_its_available_resolution_row():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 105, 100, 104),
        (104, 104, 99, 100),
        (100, 106, 100, 105),
        (105, 106, 99, 104),
        (104, 105, 98, 100),
    ])
    df["interval_end"] = df.index + pd.Timedelta(minutes=5)
    df["available_at"] = df["interval_end"]
    df.loc[df.index[4], "available_at"] += pd.Timedelta(minutes=10)
    df.loc[df.index[5], "available_at"] = df.loc[df.index[4], "available_at"]

    result = add_liquidity_features(
        df, left_bars=1, right_bars=1, timeframe="5m"
    )

    assert result["liquidity_sweep"].iloc[:4].eq(0.0).all()
    assert result["liquidity_rejection_up"].iloc[4] == 1
    assert result["liquidity_confirmation_available_at"].iloc[4] == df.loc[
        df.index[4], "available_at"
    ]


def test_liquidity_features_reject_availability_before_explicit_interval_end():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 105, 100, 104),
        (104, 103, 99, 100),
        (100, 106, 100, 105),
        (105, 104, 99, 104),
    ])
    df["interval_end"] = df.index + pd.Timedelta(minutes=5)
    df["available_at"] = df["interval_end"]
    df.loc[df.index[4], "available_at"] = df.index[4] + pd.Timedelta(minutes=4)

    with pytest.raises(ValueError, match="interval end"):
        add_liquidity_features(df, left_bars=1, right_bars=1, timeframe="5m")


def test_missing_confirmation_availability_cannot_create_tabular_sweep():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 105, 100, 104),
        (104, 104, 99, 100),
        (100, 106, 100, 105),
        (105, 106, 99, 104),
    ])
    df["interval_end"] = df.index + pd.Timedelta(minutes=5)
    df["available_at"] = df["interval_end"]
    df.loc[df.index[4], "available_at"] = pd.NaT

    result = add_liquidity_features(
        df, left_bars=1, right_bars=1, timeframe="5m"
    )

    assert result["liquidity_rejection_up"].eq(0).all()
    assert result["liquidity_sweep"].eq(0.0).all()


def test_resolve_breach_rejects_availability_before_interval_end():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 103),
        (104, 103, 98, 99),
    ])
    df["interval_end"] = df.index + pd.Timedelta(minutes=5)
    df["available_at"] = df["interval_end"]
    df.loc[df.index[3], "available_at"] = df.index[3] + pd.Timedelta(minutes=4)
    level = LiquidityLevel(df.index[1], 103.0, LiquiditySide.BSL, "TEST")

    with pytest.raises(ValueError, match="interval end"):
        resolve_breach(df, level, 2, max_resolution_bars=1)


def test_direct_resolve_breach_requires_explicit_fallback_convention():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 102),
        (102, 103, 100, 101),
    ])
    df.attrs.pop("bar_label")
    level = LiquidityLevel(df.index[1], 103.0, LiquiditySide.BSL, "TEST")

    with pytest.raises(ValueError, match="bar_label|timeframe|availability"):
        resolve_breach(df, level, 2, max_resolution_bars=1)


def test_direct_left_labeled_resolve_breach_uses_nominal_close():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 102),
        (102, 103, 100, 101),
    ])
    df.attrs["bar_label"] = "left"
    df.attrs["timeframe"] = "5min"
    level = LiquidityLevel(df.index[1], 103.0, LiquiditySide.BSL, "TEST")

    event = resolve_breach(df, level, 2, max_resolution_bars=1)

    assert event.resolution_timestamp == df.index[3]
    assert event.availability_timestamp == df.index[3] + pd.Timedelta(minutes=5)


def test_direct_resolve_breach_rejects_left_label_without_timeframe():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 102),
        (102, 103, 100, 101),
    ])
    df.attrs["bar_label"] = "left"
    level = LiquidityLevel(df.index[1], 103.0, LiquiditySide.BSL, "TEST")

    with pytest.raises(ValueError, match="Timeframe|availability"):
        resolve_breach(df, level, 2, max_resolution_bars=1)


def test_direct_resolve_breach_rejects_naive_fallback_timestamps():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 102),
        (102, 103, 100, 101),
    ])
    df.index = df.index.tz_localize(None)
    level = LiquidityLevel(df.index[1], 103.0, LiquiditySide.BSL, "TEST")

    with pytest.raises(ValueError, match="Timezone-aware"):
        resolve_breach(df, level, 2, max_resolution_bars=1)


def test_direct_resolve_breach_rejects_unknown_availability_mode():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 102),
        (102, 103, 100, 101),
    ])
    df.attrs["availability_mode"] = "unknown"
    level = LiquidityLevel(df.index[1], 103.0, LiquiditySide.BSL, "TEST")

    with pytest.raises(ValueError, match="Unsupported liquidity availability mode"):
        resolve_breach(df, level, 2, max_resolution_bars=1)


def test_missing_post_breach_close_leaves_event_unresolved():
    df = make_ohlc([
        (100, 101, 99, 100),
        (105, 106, 104, 105),
        (105, 107, 104, 105),
        (105, 107, 104, 105),
    ])
    level = LiquidityLevel(df.index[0], 105.0, LiquiditySide.BSL, "TEST")

    event = resolve_breach(df, level, 1, max_resolution_bars=2)

    assert event.outcome is BreachOutcome.UNRESOLVED
    assert event.resolution_timestamp is None
    assert event.breach_availability_timestamp == df.index[1]


def test_incomplete_bar_cannot_confirm_liquidity_rejection():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 103),  # BSL breach without a decisive close
        (104, 104, 99, 104),  # incomplete close below the level
        (104, 103, 98, 99),  # next complete close confirms rejection
    ])
    df["is_complete"] = [True, True, True, False, True]
    level = LiquidityLevel(df.index[1], 103.0, LiquiditySide.BSL, "TEST")

    event = resolve_breach(df, level, 2, max_resolution_bars=2)

    assert event.outcome is BreachOutcome.REJECTION
    assert event.resolution_timestamp == df.index[4]


def test_liquidity_event_records_actual_confirmation_availability():
    df = make_ohlc([
        (100, 101, 99, 100),
        (100, 103, 99, 102),
        (102, 105, 101, 103),  # BSL breach without a decisive close
        (104, 105, 98, 99),  # rejection close
    ])
    df["interval_end"] = df.index + pd.Timedelta(minutes=5)
    df["available_at"] = df["interval_end"]
    df.loc[df.index[3], "available_at"] = df.index[3] + pd.Timedelta(minutes=10)
    level = LiquidityLevel(df.index[1], 103.0, LiquiditySide.BSL, "TEST")

    event = resolve_breach(df, level, 2, max_resolution_bars=1)

    assert event.breach_timestamp == df.index[2]
    assert event.breach_availability_timestamp == df.loc[df.index[2], "available_at"]
    assert event.resolution_timestamp == df.index[3]
    assert event.availability_timestamp == df.loc[df.index[3], "available_at"]
