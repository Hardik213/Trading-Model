from __future__ import annotations

import pandas as pd
import pytest

from src.tick_aggregation import (
    TickAggregator,
    aggregate_ticks,
    detect_gaps,
    iter_gaps,
    validate_tick_frame,
)


def _tick_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["bid"] = pd.to_numeric(df["bid"], errors="raise")
    df["ask"] = pd.to_numeric(df["ask"], errors="raise")
    return df


def test_ohlc_calculations_for_bid_and_ask():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:00:30Z", "bid": 101.0, "ask": 102.0},
            {"timestamp": "2025-01-01T00:01:00Z", "bid": 102.0, "ask": 103.0},
            {"timestamp": "2025-01-01T00:01:30Z", "bid": 103.0, "ask": 104.0},
        ]
    )
    bars = aggregate_ticks(ticks, "1min", include_incomplete=True)
    assert len(bars) == 2
    first = bars.iloc[0]
    assert first["interval_start"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert first["interval_end"] == pd.Timestamp("2025-01-01T00:01:00Z")
    assert first["bid_open"] == 100.0
    assert first["bid_high"] == 101.0
    assert first["bid_low"] == 100.0
    assert first["bid_close"] == 101.0
    assert first["ask_open"] == 101.0
    assert first["ask_high"] == 102.0
    assert first["ask_low"] == 101.0
    assert first["ask_close"] == 102.0
    assert first["tick_count"] == 2
    assert first["spread_observations"] == 2
    assert first["is_complete"] == True
    assert first["availability_ts"] == pd.Timestamp("2025-01-01T00:01:00Z")


def test_incremental_emission_occurs_only_after_interval_close():
    aggregator = TickAggregator("1min")
    emitted = aggregator.add_tick("2025-01-01T00:00:00Z", 100.0, 101.0)
    assert emitted == []
    emitted = aggregator.add_tick("2025-01-01T00:00:30Z", 101.0, 102.0)
    assert emitted == []
    emitted = aggregator.add_tick("2025-01-01T00:01:00Z", 102.0, 103.0)
    assert len(emitted) == 1
    assert emitted[0]["interval_start"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert emitted[0]["interval_end"] == pd.Timestamp("2025-01-01T00:01:00Z")
    assert emitted[0]["availability_ts"] == pd.Timestamp("2025-01-01T00:01:00Z")
    assert emitted[0]["available_at"] == pd.Timestamp("2025-01-01T00:01:00Z")
    assert emitted[0]["historical_complete"] is True
    assert emitted[0]["is_complete"] is True


def test_delayed_next_tick_after_missing_interval_is_not_synthetic():
    aggregator = TickAggregator("1min")
    assert aggregator.add_tick("2025-01-01T00:00:00Z", 100.0, 101.0) == []
    emitted = aggregator.add_tick("2025-01-01T00:02:30Z", 102.0, 103.0)
    assert len(emitted) == 1
    assert emitted[0]["interval_start"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert emitted[0]["interval_end"] == pd.Timestamp("2025-01-01T00:01:00Z")
    assert emitted[0]["historical_complete"] is True
    assert emitted[0]["available_at"] == pd.Timestamp("2025-01-01T00:02:30Z")
    assert emitted[0]["available_at"] != emitted[0]["interval_end"]

    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:02:30Z", "bid": 102.0, "ask": 103.0},
        ]
    )
    assert detect_gaps(ticks, "1min") == [pd.Timestamp("2025-01-01T00:01:00Z")]
    bars = aggregate_ticks(ticks, "1min", include_incomplete=True)
    assert bars["interval_start"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:02:00Z"),
    ]
    assert bars["is_complete"].tolist() == [True, False]
    assert pd.isna(bars.iloc[1]["availability_ts"])


def test_multiple_consecutive_missing_intervals_are_yielded_lazily():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:05:00Z", "bid": 105.0, "ask": 106.0},
        ]
    )
    expected = [
        pd.Timestamp("2025-01-01T00:01:00Z"),
        pd.Timestamp("2025-01-01T00:02:00Z"),
        pd.Timestamp("2025-01-01T00:03:00Z"),
        pd.Timestamp("2025-01-01T00:04:00Z"),
    ]
    assert list(iter_gaps(ticks, "1min")) == expected
    assert detect_gaps(ticks, "1min") == expected


def test_sparse_gap_iterator_yields_without_materializing_full_gap():
    ticks = _tick_frame(
        [
            {"timestamp": "2020-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 105.0, "ask": 106.0},
        ]
    )
    gaps = iter_gaps(ticks, "1min")
    assert next(gaps) == pd.Timestamp("2020-01-01T00:01:00Z")
    gaps.close()


def test_final_incomplete_interval_after_long_gap_stays_incomplete():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:10:30Z", "bid": 110.0, "ask": 111.0},
        ]
    )
    aggregator = TickAggregator("1min")
    completed = []
    for tick in ticks.itertuples(index=False):
        completed.extend(aggregator.add_tick(tick.timestamp, tick.bid, tick.ask))
    assert len(completed) == 1
    assert completed[0]["interval_start"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert completed[0]["available_at"] == pd.Timestamp("2025-01-01T00:10:30Z")
    incomplete = aggregator.flush(include_incomplete=True)
    assert len(incomplete) == 1
    assert incomplete[0]["interval_start"] == pd.Timestamp("2025-01-01T00:10:00Z")
    assert incomplete[0]["historical_complete"] is False
    assert pd.isna(incomplete[0]["available_at"])
    batch_completed = aggregate_ticks(ticks, "1min")
    assert batch_completed["interval_start"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ]
    assert batch_completed["historical_complete"].tolist() == [True]


def test_interval_boundaries_are_left_closed_right_open():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:01:00Z", "bid": 102.0, "ask": 103.0},
            {"timestamp": "2025-01-01T00:01:30Z", "bid": 101.0, "ask": 102.0},
        ]
    )
    bars = aggregate_ticks(ticks, "1min", include_incomplete=True)
    assert bars["tick_count"].tolist() == [1, 2]
    assert bars["interval_start"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:01:00Z"),
    ]


def test_minute_3_5_and_15_boundary_alignment():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:03:00Z", "bid": 103.0, "ask": 104.0},
            {"timestamp": "2025-01-01T00:05:00Z", "bid": 105.0, "ask": 106.0},
            {"timestamp": "2025-01-01T00:15:00Z", "bid": 115.0, "ask": 116.0},
        ]
    )
    assert aggregate_ticks(ticks, "3min", include_incomplete=True)["interval_start"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:03:00Z"),
        pd.Timestamp("2025-01-01T00:15:00Z"),
    ]
    assert aggregate_ticks(ticks, "5min", include_incomplete=True)["interval_start"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:05:00Z"),
        pd.Timestamp("2025-01-01T00:15:00Z"),
    ]
    assert aggregate_ticks(ticks, "15min", include_incomplete=True)["interval_start"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:15:00Z"),
    ]


def test_duplicate_timestamps_are_preserved_in_order():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 101.0, "ask": 102.0},
            {"timestamp": "2025-01-01T00:00:05Z", "bid": 102.0, "ask": 103.0},
        ]
    )
    bars = aggregate_ticks(ticks, "1min", include_incomplete=True)
    assert bars.iloc[0]["tick_count"] == 3
    assert bars.iloc[0]["bid_open"] == 100.0
    assert bars.iloc[0]["bid_close"] == 102.0
    assert bars.iloc[0]["ask_open"] == 101.0
    assert bars.iloc[0]["ask_close"] == 103.0

    aggregator = TickAggregator("1min")
    for tick in ticks.itertuples(index=False):
        assert aggregator.add_tick(tick.timestamp, tick.bid, tick.ask) == []
    streamed = aggregator.flush(include_incomplete=True)
    assert streamed[0]["bid_open"] == 100.0
    assert streamed[0]["bid_close"] == 102.0
    assert streamed[0]["ask_open"] == 101.0
    assert streamed[0]["ask_close"] == 103.0


def test_incomplete_final_interval_is_not_completed_by_default():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:00:30Z", "bid": 101.0, "ask": 102.0},
        ]
    )
    bars = aggregate_ticks(ticks, "1min")
    assert bars.empty
    incomplete = aggregate_ticks(ticks, "1min", include_incomplete=True)
    assert len(incomplete) == 1
    assert incomplete.iloc[0]["is_complete"] == False
    assert pd.isna(incomplete.iloc[0]["availability_ts"])


def test_invalid_and_crossed_quotes_raise_value_error():
    with pytest.raises(ValueError):
        validate_tick_frame(
            _tick_frame(
                [
                    {"timestamp": "2025-01-01T00:00:00Z", "bid": 101.0, "ask": 100.0},
                ]
            )
        )
    with pytest.raises(ValueError):
        validate_tick_frame(
            _tick_frame(
                [
                    {"timestamp": "2025-01-01T00:00:00Z", "bid": 0.0, "ask": 1.0},
                ]
            )
        )
    with pytest.raises(ValueError):
        validate_tick_frame(
            _tick_frame(
                [
                    {"timestamp": pd.NaT, "bid": 100.0, "ask": 101.0},
                ]
            )
        )
    with pytest.raises(ValueError):
        validate_tick_frame(
            _tick_frame(
                [
                    {"timestamp": "2025-01-01T00:00:00Z", "bid": float("nan"), "ask": 101.0},
                ]
            )
        )
    with pytest.raises(ValueError):
        validate_tick_frame(
            _tick_frame(
                [
                    {"timestamp": "2025-01-01T00:00:00Z", "bid": float("inf"), "ask": 101.0},
                ]
            )
        )
    with pytest.raises(ValueError):
        validate_tick_frame(
            _tick_frame(
                [
                    {"timestamp": "2025-01-01T00:00:00Z", "bid": None, "ask": 101.0},
                ]
            )
        )


@pytest.mark.parametrize(
    ("timestamp", "bid", "ask"),
    [
        (None, 100.0, 101.0),
        (pd.NaT, 100.0, 101.0),
        ("not-a-timestamp", 100.0, 101.0),
        ("2025-01-01T00:00:00Z", None, 101.0),
        ("2025-01-01T00:00:00Z", float("nan"), 101.0),
        ("2025-01-01T00:00:00Z", float("inf"), 101.0),
        ("2025-01-01T00:00:00Z", float("-inf"), 101.0),
        ("2025-01-01T00:00:00Z", 0.0, 1.0),
        ("2025-01-01T00:00:00Z", -1.0, 1.0),
        ("2025-01-01T00:00:00Z", 101.0, 100.0),
    ],
)
def test_add_tick_rejects_invalid_timestamp_and_prices(timestamp, bid, ask):
    with pytest.raises(ValueError):
        TickAggregator("1min").add_tick(timestamp, bid, ask)


def test_empty_input_is_empty():
    empty = pd.DataFrame(columns=["timestamp", "bid", "ask"]).astype({"timestamp": "datetime64[ns, UTC]"})
    assert aggregate_ticks(empty, "1min").empty
    assert detect_gaps(empty, "1min") == []


@pytest.mark.parametrize(
    "columns",
    [[], ["timestamp", "bid"], ["timestamp", "ask"], ["bid", "ask"]],
)
def test_empty_tick_frame_requires_all_schema_columns(columns):
    with pytest.raises(ValueError, match="timestamp, bid, and ask"):
        validate_tick_frame(pd.DataFrame(columns=columns))


def test_empty_input_still_rejects_unsupported_interval():
    empty = pd.DataFrame(columns=["timestamp", "bid", "ask"]).astype({"timestamp": "datetime64[ns, UTC]"})
    with pytest.raises(ValueError, match="Unsupported interval"):
        aggregate_ticks(empty, "2min")


def test_batch_and_incremental_equivalence_for_fully_observed_intervals():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:00:30Z", "bid": 101.0, "ask": 102.0},
            {"timestamp": "2025-01-01T00:01:00Z", "bid": 102.0, "ask": 103.0},
            {"timestamp": "2025-01-01T00:01:30Z", "bid": 103.0, "ask": 104.0},
        ]
    )
    agg = aggregate_ticks(ticks, "1min", include_incomplete=True)
    aggregator = TickAggregator("1min")
    rows: list[dict[str, object]] = []
    for row in ticks.itertuples(index=False):
        rows.extend(aggregator.add_tick(row.timestamp, row.bid, row.ask))
    rows.extend(aggregator.flush(include_incomplete=True))
    stream = pd.DataFrame(rows).sort_values("interval_start", kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(agg, stream)


def test_flush_is_terminal_idempotent_and_never_completes_open_bucket():
    aggregator = TickAggregator("1min")
    aggregator.add_tick("2025-01-01T00:00:00Z", 100.0, 101.0)
    incomplete = aggregator.flush(include_incomplete=True)
    assert len(incomplete) == 1
    assert incomplete[0]["historical_complete"] is False
    assert incomplete[0]["is_complete"] is False
    assert pd.isna(incomplete[0]["available_at"])
    assert aggregator.flush(include_incomplete=True) == []
    assert aggregator.flush() == []
    with pytest.raises(RuntimeError):
        aggregator.add_tick("2025-01-01T00:01:00Z", 101.0, 102.0)


def test_default_flush_discards_incomplete_bucket_from_completed_output():
    aggregator = TickAggregator("1min")
    aggregator.add_tick("2025-01-01T00:00:30Z", 100.0, 101.0)
    assert aggregator.flush() == []
    assert aggregator.flush() == []


def test_flush_during_gap_cannot_put_final_open_interval_in_completed_output():
    aggregator = TickAggregator("1min")
    aggregator.add_tick("2025-01-01T00:00:00Z", 100.0, 101.0)
    completed = aggregator.add_tick("2025-01-01T00:05:30Z", 105.0, 106.0)
    assert len(completed) == 1
    assert completed[0]["historical_complete"] is True
    final_completed = aggregator.flush()
    assert final_completed == []
    incomplete = aggregator.flush(include_incomplete=True)
    assert incomplete == []


def test_non_monotonic_timestamps_raise():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:01:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 101.0, "ask": 102.0},
        ]
    )
    with pytest.raises(ValueError):
        validate_tick_frame(ticks)
    aggregator = TickAggregator("1min")
    aggregator.add_tick("2025-01-01T00:01:00Z", 100.0, 101.0)
    with pytest.raises(ValueError):
        aggregator.add_tick("2025-01-01T00:00:00Z", 101.0, 102.0)


def test_utc_alignment_and_no_lookahead():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:30:00+02:00", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:31:00+02:00", "bid": 101.0, "ask": 102.0},
        ]
    )
    bars = aggregate_ticks(ticks, "1min", include_incomplete=True)
    assert bars.iloc[0]["interval_start"] == pd.Timestamp("2024-12-31T22:30:00Z")
    assert bars.iloc[0]["interval_end"] == pd.Timestamp("2024-12-31T22:31:00Z")
    assert bars.iloc[0]["availability_ts"] == pd.Timestamp("2024-12-31T22:31:00Z")
    assert bars.iloc[0]["is_complete"] == True


def test_reproducibility_and_gap_detection_without_synthetic_bar_creation():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:00Z", "bid": 100.0, "ask": 101.0},
            {"timestamp": "2025-01-01T00:02:00Z", "bid": 102.0, "ask": 103.0},
        ]
    )
    first = aggregate_ticks(ticks, "1min", include_incomplete=True)
    second = aggregate_ticks(ticks, "1min", include_incomplete=True)
    pd.testing.assert_frame_equal(first, second)
    assert detect_gaps(ticks, "1min") == [
        pd.Timestamp("2025-01-01T00:01:00Z"),
    ]


def test_bar_cannot_be_observed_complete_before_end_time():
    ticks = _tick_frame(
        [
            {"timestamp": "2025-01-01T00:00:30Z", "bid": 100.0, "ask": 101.0},
        ]
    )
    bars = aggregate_ticks(ticks, "1min", include_incomplete=True)
    assert len(bars) == 1
    assert bars.iloc[0]["interval_end"] == pd.Timestamp("2025-01-01T00:01:00Z")
    assert bars.iloc[0]["is_complete"] == False
    assert pd.isna(bars.iloc[0]["availability_ts"])
