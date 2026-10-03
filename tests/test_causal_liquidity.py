import pytest
import pandas as pd

from src.causal_liquidity import (
    LiquidityEvidenceKind,
    confirmed_liquidity_map,
    draw_on_liquidity,
    latest_reversal_liquidity,
    liquidity_evidence_as_of,
)
from src.data_contract import normalize_ohlc
from src.market_structure import LiquiditySide
from src.mss import Direction


def frame(rows):
    idx = pd.date_range("2026-01-01 10:00", periods=len(rows), freq="5min", tz="UTC")
    result = pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])
    result.attrs["bar_label"] = "right"
    return result


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


def test_event_time_availability_gates_swing_confirmation_and_shared_rows():
    df=frame([
        (100,101,99,100),
        (100,105,99,104),
        (102,103,100,101),
    ])
    df["interval_end"]=df.index+pd.Timedelta(minutes=5)
    df["available_at"]=pd.to_datetime(
        ["2026-01-01T10:05:00Z","2026-01-01T10:15:00Z","2026-01-01T10:15:00Z"],
        utc=True,
    )
    df["historical_complete"]=[True,True,True]
    df["is_complete"]=[True,True,True]

    before=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:14:59Z"),left_bars=1,right_bars=1
    )
    at_availability=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:15:00Z"),left_bars=1,right_bars=1
    )
    assert not any(level.price == 105 for level in before)
    assert any(level.price == 105 for level in at_availability)
    confirmed_level = next(level for level in at_availability if level.price == 105)
    assert confirmed_level.confirmation_timestamp == df.index[2]
    assert confirmed_level.availability_timestamp == df.loc[df.index[2], "available_at"]


def test_incomplete_event_time_bar_cannot_confirm_a_swing():
    df=frame([
        (100,101,99,100),
        (100,105,99,104),
        (102,103,100,101),
    ])
    df["interval_end"]=df.index+pd.Timedelta(minutes=5)
    df["available_at"]=pd.to_datetime(
        ["2026-01-01T10:05:00Z","2026-01-01T10:10:00Z",None],utc=True
    )
    df["historical_complete"]=[True,True,False]
    df["is_complete"]=[True,True,False]

    levels=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:30:00Z"),left_bars=1,right_bars=1
    )
    assert not any(level.price == 105 for level in levels)


def test_rejection_waits_for_delayed_resolution_bar_availability():
    df=frame([
        (100,102,99,101),
        (101,102,95,100),
        (100,101,97,99),
        (99,100,94,96),
        (96,101,95,100),
    ])
    df["interval_end"]=df.index+pd.Timedelta(minutes=5)
    df["available_at"]=pd.to_datetime(
        [
            "2026-01-01T10:05:00Z",
            "2026-01-01T10:10:00Z",
            "2026-01-01T10:15:00Z",
            "2026-01-01T10:20:00Z",
            "2026-01-01T10:30:00Z",
        ],
        utc=True,
    )
    df["historical_complete"]=[True]*5
    df["is_complete"]=[True]*5

    before=liquidity_evidence_as_of(
        df,pd.Timestamp("2026-01-01T10:29:59Z"),side=LiquiditySide.SSL,
        left_bars=1,right_bars=1,
    )
    at_availability=liquidity_evidence_as_of(
        df,pd.Timestamp("2026-01-01T10:30:00Z"),side=LiquiditySide.SSL,
        left_bars=1,right_bars=1,
    )
    assert not any(item.kind is LiquidityEvidenceKind.REJECTION for item in before)
    assert any(item.kind is LiquidityEvidenceKind.REJECTION for item in at_availability)

    confirmed = next(item for item in at_availability if item.kind is LiquidityEvidenceKind.REJECTION)
    assert confirmed.breach.resolution_timestamp == df.index[4]
    assert confirmed.breach.availability_timestamp == pd.Timestamp("2026-01-01T10:30:00Z")
    assert confirmed.confirmed


def test_endpoint_only_liquidity_levels_wait_for_interval_end():
    df=frame([
        (100,101,99,100),
        (100,105,99,104),
        (102,103,100,101),
    ])
    df["interval_end"]=df.index+pd.Timedelta(minutes=10)

    visible_before_end=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:15:00Z"),left_bars=1,right_bars=1
    )
    visible_at_end=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:25:00Z"),left_bars=1,right_bars=1
    )

    assert not any(level.price==105 for level in visible_before_end)
    assert any(level.price==105 for level in visible_at_end)


def test_normalized_nominal_close_fallback_gates_swing_confirmation():
    raw = frame([
        (100,101,99,100),
        (100,105,99,104),
        (102,103,100,101),
    ])
    raw.attrs["bar_label"] = "left"
    df=normalize_ohlc(
        raw,
        timeframe="5M",
        source="TEST",
    )

    before=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:11:00Z"),left_bars=1,right_bars=1
    )
    at_close=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:15:00Z"),left_bars=1,right_bars=1
    )
    assert not any(level.price == 105 for level in before)
    assert any(level.price == 105 for level in at_close)


def test_left_labeled_liquidity_confirmation_waits_for_nominal_close():
    df = frame([
        (100, 101, 99, 100),
        (100, 105, 99, 104),
        (102, 103, 100, 101),
    ])
    df.attrs["bar_label"] = "left"
    df.attrs["timeframe"] = "5min"
    df.attrs["availability_mode"] = "nominal_close_fallback"

    before_close = confirmed_liquidity_map(
        df, pd.Timestamp("2026-01-01T10:14:59Z"), left_bars=1, right_bars=1
    )
    at_close = confirmed_liquidity_map(
        df, pd.Timestamp("2026-01-01T10:15:00Z"), left_bars=1, right_bars=1
    )

    assert not any(level.price == 105 for level in before_close)
    assert any(level.price == 105 for level in at_close)


def test_liquidity_query_rejects_missing_timeframe_and_availability():
    df = frame([
        (100, 101, 99, 100),
        (100, 105, 99, 104),
        (102, 103, 100, 101),
    ])
    df.attrs.pop("bar_label")

    with pytest.raises(ValueError, match="Timeframe|bar_label|availability"):
        confirmed_liquidity_map(
            df, df.index[2] + pd.Timedelta(minutes=1), left_bars=1, right_bars=1
        )


def test_explicit_event_time_requires_actual_availability_evidence():
    df = frame([(100, 101, 99, 100)])
    df.attrs["availability_mode"] = "event_time"
    df.attrs.pop("bar_label")

    with pytest.raises(ValueError, match="availability"):
        confirmed_liquidity_map(df, df.index[0] + pd.Timedelta(minutes=5))


@pytest.mark.parametrize(
    ("availability", "availability_alias", "drop_availability"),
    [
        ("2026-01-01T10:05:00Z", "2026-01-01T10:06:00Z", False),
        ("2026-01-01T10:04:00Z", "2026-01-01T10:04:00Z", False),
    ],
)
def test_event_time_filter_rejects_inconsistent_or_missing_metadata(
    availability,
    availability_alias,
    drop_availability,
):
    df=frame([(100,101,99,100)])
    df.attrs["availability_mode"]="event_time"
    df.attrs["timeframe"]="5M"
    df["interval_end"]=pd.to_datetime(["2026-01-01T10:05:00Z"],utc=True)
    df["available_at"]=pd.to_datetime([availability],utc=True)
    df["availability_ts"]=pd.to_datetime([availability_alias],utc=True)

    with pytest.raises(ValueError):
        confirmed_liquidity_map(df,pd.Timestamp("2026-01-01T10:10:00Z"))


def test_event_time_mode_accepts_endpoint_only_availability():
    df=frame([
        (100,101,99,100),
        (100,105,99,104),
        (102,103,100,101),
    ])
    df.attrs["availability_mode"]="event_time"
    df.attrs["timeframe"]="5M"
    df["interval_end"]=df.index+pd.Timedelta(minutes=5)

    before=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:14:59Z"),left_bars=1,right_bars=1
    )
    at_end=confirmed_liquidity_map(
        df,pd.Timestamp("2026-01-01T10:15:00Z"),left_bars=1,right_bars=1
    )

    assert not any(level.price==105 for level in before)
    assert any(level.price==105 for level in at_end)


def test_event_time_availability_must_follow_bar_order():
    df=frame([
        (100,101,99,100),
        (100,102,99,101),
    ])
    df["interval_end"]=df.index+pd.Timedelta(minutes=5)
    df["available_at"]=pd.to_datetime(
        ["2026-01-01T10:15:00Z","2026-01-01T10:10:00Z"],utc=True
    )
    df["availability_ts"]=df["available_at"]
    df["historical_complete"]=[True,True]
    df["is_complete"]=[True,True]

    with pytest.raises(ValueError):
        confirmed_liquidity_map(df,pd.Timestamp("2026-01-01T10:20:00Z"))
