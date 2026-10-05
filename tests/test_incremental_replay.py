from __future__ import annotations

import pandas as pd
import pytest

from src.replay_engine import (
    HistoricalReplay,
    IncrementalHistoricalReplay,
    IncrementalReplayBar,
    ReplayAvailabilityGroup,
    ReplayConfig,
    ReplayDecision,
    ReplayObservation,
)
from src.replay_subject import ReplaySubject
from src.sniper_setup import PrecisionEvidence
from src.timeframe_context import TimeframeContext


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _bar(
    timestamp: str,
    available_at: str,
    *,
    timeframe: str = "1min",
    interval: str = "1min",
    close: float = 100.0,
    complete: bool | None = True,
) -> IncrementalReplayBar:
    label = _ts(timestamp)
    return IncrementalReplayBar(
        timeframe=timeframe,
        timestamp=label,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        interval_end=label + pd.Timedelta(interval),
        available_at=_ts(available_at),
        availability_ts=_ts(available_at),
        historical_complete=complete,
        is_complete=complete,
    )


def _group(available_at: str, **bars: tuple[IncrementalReplayBar, ...]) -> ReplayAvailabilityGroup:
    return ReplayAvailabilityGroup(_ts(available_at), bars)


def _replay(
    *,
    limits: dict[str, int] | None = None,
    max_groups: int = 4,
    max_bars: int = 32,
) -> IncrementalHistoricalReplay:
    return IncrementalHistoricalReplay(
        config=ReplayConfig(timeframe="1min", source="TEST"),
        history_limits=limits or {"1min": 8, "5min": 4},
        max_pending_groups=max_groups,
        max_pending_bars=max_bars,
    )


def _observation(timestamp, _base, _context):
    return ReplayObservation(timestamp, ReplayDecision.DEVELOPING, "DEVELOPING", "test")


def test_replay_subject_identity_is_utc_hashable_and_round_trips():
    first = ReplaySubject(
        pd.Timestamp("2026-01-01T10:05:17-05:00"),
        _ts("2026-01-01T10:01:00Z"),
    )
    second = ReplaySubject(
        _ts("2026-01-01T15:05:17Z"),
        _ts("2026-01-01T10:02:00Z"),
    )

    assert first.availability_timestamp == second.availability_timestamp
    assert first.availability_timestamp.tz == pd.Timestamp("2026-01-01T00:00:00Z").tz
    assert first != second
    assert len({first, second}) == 2
    assert first.subject_id != second.subject_id
    assert ReplaySubject.from_dict(first.to_dict()) == first
    with pytest.raises(ValueError, match="later than availability"):
        ReplaySubject(
            first.availability_timestamp,
            first.availability_timestamp + pd.Timedelta(minutes=1),
        )


def test_empty_lifecycle_and_finish_rejects_future_input():
    replay = _replay()
    assert replay.process_next(_observation) is None
    replay.finish()
    assert replay.process_next(_observation) is None
    with pytest.raises(RuntimeError, match="finished"):
        replay.feed_group(_group("2026-01-01T00:01:00Z", **{
            "1min": (_bar("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"),)
        }))


def test_finish_requires_pending_groups_to_be_drained():
    replay = _replay()
    replay.feed_group(_group(
        "2026-01-01T00:01:00Z",
        **{"1min": (_bar("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"),)},
    ))
    with pytest.raises(RuntimeError, match="before finish"):
        replay.finish()
    assert replay.process_next(_observation) is not None
    replay.finish()


def test_batch_and_incremental_snapshots_match_within_declared_window():
    labels = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="min")
    availability = labels + pd.Timedelta(minutes=1)
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [101.0, 102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.0, 101.0, 102.0, 103.0],
            "interval_end": availability,
            "available_at": availability,
            "availability_ts": availability,
            "historical_complete": True,
            "is_complete": True,
        },
        index=labels,
    )
    batch = HistoricalReplay(
        frame,
        context=TimeframeContext(frames={"1min": frame.copy()}),
        config=ReplayConfig(timeframe="1min", source="TEST"),
    )
    batch_snapshots = []
    batch.run(lambda timestamp, base, context: (
        batch_snapshots.append((timestamp, base.copy(), context.frames["1min"].copy()))
        or _observation(timestamp, base, context)
    ))

    incremental = _replay(limits={"1min": 8})
    incremental_snapshots = []
    for label, available, close in zip(labels, availability, frame["Close"]):
        group_time = available.isoformat()
        incremental.feed_group(_group(
            group_time,
            **{"1min": (_bar(label.isoformat(), group_time, close=close),)},
        ))
        def capture(timestamp, base, context):
            incremental_snapshots.append(
                (timestamp, base.copy(), context.frames["1min"].copy())
            )
            return _observation(timestamp, base, context)

        incremental.process_next(capture)
    incremental.finish()

    batch_times = [item[0] for item in batch_snapshots]
    incremental_times = [item[0] for item in incremental_snapshots]
    assert batch_times == frame["available_at"].tolist()
    assert incremental_times == batch_times
    for batch_item, incremental_item in zip(batch_snapshots, incremental_snapshots):
        _, batch_base, batch_context = batch_item
        _, incremental_base, incremental_context = incremental_item
        batch_base.index = pd.DatetimeIndex(batch_base.index.to_numpy(), name=None)
        batch_context.index = pd.DatetimeIndex(batch_context.index.to_numpy(), name=None)
        incremental_base.index = pd.DatetimeIndex(incremental_base.index.to_numpy(), name=None)
        incremental_context.index = pd.DatetimeIndex(incremental_context.index.to_numpy(), name=None)
        pd.testing.assert_frame_equal(batch_base, incremental_base, check_dtype=False)
        pd.testing.assert_frame_equal(batch_context, incremental_context, check_dtype=False)

def test_multiple_decisions_for_atomic_group_with_multiple_bars_and_timeframes():
    replay = _replay()
    timestamp = "2026-01-01T00:07:00Z"
    replay.feed_group(_group(
        timestamp,
        **{
            "1min": (
                _bar("2026-01-01T00:00:00Z", timestamp),
                _bar("2026-01-01T00:01:00Z", timestamp, close=101.0),
            ),
            "5min": (
                _bar(
                    "2026-01-01T00:00:00Z",
                    timestamp,
                    timeframe="5min",
                    interval="5min",
                    close=100.0,
                ),
            ),
        },
    ))
    seen = []
    callback = lambda ts, base, context: (
        seen.append((ts, base.index.tolist(), context.frames["5min"].index.tolist()))
        or _observation(ts, base, context)
    )
    first = replay.process_next(callback)
    second = replay.process_next(callback)
    assert first is not None and first.timestamp == _ts(timestamp)
    assert second is not None and second.timestamp == _ts(timestamp)
    assert seen == [
        (_ts(timestamp), [_ts("2026-01-01T00:00:00Z")], [_ts("2026-01-01T00:00:00Z")]),
        (
            _ts(timestamp),
            [_ts("2026-01-01T00:00:00Z"), _ts("2026-01-01T00:01:00Z")],
            [_ts("2026-01-01T00:00:00Z")],
        ),
    ]
    assert replay.process_next(_observation) is None


def test_five_delayed_bars_keep_actual_availability_as_callback_timestamp():
    replay = _replay()
    prior_available = "2026-01-01T10:00:00Z"
    replay.feed_group(_group(
        prior_available,
        **{"1min": (_bar("2026-01-01T09:59:00Z", prior_available),)},
    ))
    available = "2026-01-01T10:05:17Z"
    base_bars = tuple(
        _bar(f"2026-01-01T10:0{minute}:00Z", available)
        for minute in range(0, 5)
    )
    replay.feed_group(_group(
        available,
        **{
            "1min": base_bars,
            "5min": (
                _bar(
                    "2026-01-01T10:00:00Z",
                    available,
                    timeframe="5min",
                    interval="5min",
                ),
            ),
        },
    ))

    seen = []
    evidence_as_of = []
    observations = []

    def callback(ts, base, context):
        seen.append((ts, base.copy(), context.frames["5min"].copy(), context.replay_subject))
        evidence_as_of.append(PrecisionEvidence(
            as_of=ts,
            direction=None,
            draw_on_liquidity=None,
            liquidity_event=None,
            mss=None,
            pd_array=None,
            entry_price=None,
            invalidation_price=None,
            target_price=None,
            target_liquidity=None,
            replay_subject=context.replay_subject,
        ))
        return _observation(ts, base, context)

    prior = replay.process_next(callback)
    assert prior is not None and prior.timestamp == _ts(prior_available)
    observations.append(prior)
    assert prior.replay_subject == seen[0][3]
    assert seen[0][0] == _ts(prior_available)
    assert seen[0][1].index.tolist() == [_ts("2026-01-01T09:59:00Z")]
    assert seen[0][2].empty

    for _ in range(5):
        observation = replay.process_next(callback)
        assert observation is not None and observation.timestamp == _ts(available)
        assert observation.replay_subject is not None
        assert observation.replay_subject.availability_timestamp == observation.timestamp
        observations.append(observation)
    assert replay.process_next(_observation) is None
    assert [item[0] for item in seen[1:]] == [_ts(available)] * 5
    subjects = [item[3] for item in seen[1:]]
    assert [subject.base_bar_timestamp for subject in subjects] == [
        _ts(f"2026-01-01T10:0{minute}:00Z") for minute in range(5)
    ]
    assert all(subject.availability_timestamp == _ts(available) for subject in subjects)
    assert len(set(subjects)) == 5
    assert len({observation.timestamp for observation in observations[1:]}) == 1
    assert len({observation.replay_subject for observation in observations[1:]}) == 5
    assert [len(item[1]) for item in seen[1:]] == [2, 3, 4, 5, 6]
    assert [item[2].index.tolist() for item in seen[1:]] == [
        [_ts("2026-01-01T10:00:00Z")]
    ] * 5
    delayed_subjects = set(subjects)
    assert delayed_subjects.isdisjoint(seen[0][1].index)
    for timestamp, base, higher, _subject in seen:
        assert (base["available_at"] <= timestamp).all()
        if "available_at" in higher:
            assert (higher["available_at"] <= timestamp).all()
    assert [evidence.as_of for evidence in evidence_as_of[1:]] == [_ts(available)] * 5
    assert [evidence.replay_subject for evidence in evidence_as_of[1:]] == subjects


def test_multi_bar_group_with_no_eligible_decisions_is_skipped():
    replay = _replay()
    available = "2026-01-01T00:05:00Z"
    replay.feed_group(_group(
        available,
        **{
            "1min": tuple(
                _bar(
                    f"2026-01-01T00:0{minute}:00Z",
                    available,
                    complete=False,
                )
                for minute in range(3)
            )
        },
    ))

    calls = []
    assert replay.process_next(lambda *args: calls.append(args)) is None
    assert calls == []
    assert replay.retained_bar_counts == {"1min": 0, "5min": 0}


def test_multi_decision_group_transitions_to_next_availability_group():
    replay = _replay()
    first_available = "2026-01-01T00:03:00Z"
    replay.feed_group(_group(
        first_available,
        **{
            "1min": tuple(
                _bar(f"2026-01-01T00:0{minute}:00Z", first_available)
                for minute in range(3)
            )
        },
    ))
    next_available = "2026-01-01T00:04:00Z"
    replay.feed_group(_group(
        next_available,
        **{
            "1min": (
                _bar("2026-01-01T00:03:00Z", next_available),
            )
        },
    ))

    seen = []
    for _ in range(4):
        observation = replay.process_next(lambda ts, base, context: (
            seen.append(ts) or _observation(ts, base, context)
        ))
        assert observation is not None
    assert replay.process_next(_observation) is None
    assert seen == [
        _ts(first_available),
        _ts(first_available),
        _ts(first_available),
        _ts(next_available),
    ]


def test_multiple_groups_may_queue_but_are_processed_in_availability_order():
    replay = _replay(max_groups=2)
    for minute in (1, 2):
        available = f"2026-01-01T00:0{minute}:00Z"
        label = f"2026-01-01T00:0{minute - 1}:00Z"
        replay.feed_group(_group(
            available,
            **{"1min": (_bar(label, available),)},
        ))
    with pytest.raises(ValueError, match="group limit"):
        available = "2026-01-01T00:03:00Z"
        replay.feed_group(_group(
            available,
            **{"1min": (_bar("2026-01-01T00:02:00Z", available),)},
        ))

    seen = []
    callback = lambda ts, base, context: (
        seen.append(ts) or _observation(ts, base, context)
    )
    assert replay.process_next(callback).timestamp == _ts("2026-01-01T00:01:00Z")
    assert replay.process_next(callback).timestamp == _ts("2026-01-01T00:02:00Z")
    assert replay.process_next(callback) is None
    assert seen == [_ts("2026-01-01T00:01:00Z"), _ts("2026-01-01T00:02:00Z")]


def test_availability_is_inclusive_and_delayed_base_bar_waits():
    replay = _replay()
    early = "2026-01-01T00:02:00Z"
    late = "2026-01-01T00:03:00Z"
    replay.feed_group(_group(
        early,
        **{"1min": (_bar("2026-01-01T00:00:00Z", early, interval="1min"),)},
    ))
    seen = []
    result = replay.process_next(lambda ts, base, context: (
        seen.append((ts, len(base))) or _observation(ts, base, context)
    ))
    assert result is not None and result.timestamp == _ts(early)
    replay.feed_group(_group(
        late,
        **{"1min": (_bar("2026-01-01T00:01:00Z", late),)},
    ))
    assert replay.process_next(_observation).timestamp == _ts(late)
    assert seen == [(_ts(early), 1)]


@pytest.mark.parametrize("invalid_time", ["2026-01-01T00:00:00", "not-a-time"])
def test_group_rejects_naive_or_invalid_availability(invalid_time):
    with pytest.raises(ValueError):
        ReplayAvailabilityGroup(pd.Timestamp(invalid_time), {})


def test_out_of_order_and_duplicate_bar_labels_are_rejected_without_mutation():
    replay = _replay()
    first = "2026-01-01T00:01:00Z"
    replay.feed_group(_group(first, **{
        "1min": (_bar("2026-01-01T00:00:00Z", first),)
    }))
    with pytest.raises(ValueError, match="strictly increasing"):
        replay.feed_group(_group("2026-01-01T00:02:00Z", **{
            "1min": (_bar("2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z"),)
        }))
    with pytest.raises(ValueError, match="strictly increasing"):
        replay.feed_group(_group("2026-01-01T00:00:30Z", **{
            "1min": (_bar("2026-01-01T00:00:30Z", "2026-01-01T00:00:30Z"),)
        }))
    assert replay.pending_group_count == 1


def test_late_group_for_an_already_fed_or_processed_time_is_rejected():
    replay = _replay()
    available = "2026-01-01T00:01:00Z"
    replay.feed_group(_group(available, **{
        "1min": (_bar("2026-01-01T00:00:00Z", available),)
    }))
    replay.process_next(_observation)
    with pytest.raises(ValueError, match="strictly increasing"):
        replay.feed_group(_group(available, **{
            "1min": (_bar("2026-01-01T00:00:30Z", available),)
        }))


def test_pending_bar_limit_is_checked_before_group_normalization():
    replay = _replay(max_bars=1)
    available = "2026-01-01T00:02:00Z"
    group = _group(available, **{
        "1min": (
            _bar("2026-01-01T00:00:00Z", available),
            _bar("2026-01-01T00:01:00Z", available),
        )
    })
    with pytest.raises(ValueError, match="bar limit"):
        replay.feed_group(group)
    assert replay.pending_group_count == 0


def test_incomplete_bars_never_decide_or_consume_history_window():
    replay = _replay(limits={"1min": 1})
    first_time = "2026-01-01T00:01:00Z"
    replay.feed_group(_group(first_time, **{
        "1min": (_bar("2026-01-01T00:00:00Z", first_time),)
    }))
    replay.process_next(_observation)
    incomplete_time = "2026-01-01T00:02:00Z"
    replay.feed_group(_group(incomplete_time, **{
        "1min": (_bar("2026-01-01T00:01:00Z", incomplete_time, complete=False),)
    }))
    assert replay.process_next(_observation) is None
    assert replay.retained_bar_counts == {"1min": 1}
    complete_time = "2026-01-01T00:03:00Z"
    replay.feed_group(_group(complete_time, **{
        "1min": (_bar("2026-01-01T00:02:00Z", complete_time),)
    }))
    seen = []
    result = replay.process_next(lambda ts, base, context: (
        seen.append(base.index.tolist()) or _observation(ts, base, context)
    ))
    assert result is not None
    assert seen == [[_ts("2026-01-01T00:02:00Z")]]
    assert replay.retained_bar_counts == {"1min": 1}


def test_missing_intervals_are_not_synthesized():
    replay = _replay(limits={"1min": 4})
    seen = []
    for label, available in (
        ("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"),
        ("2026-01-01T00:03:00Z", "2026-01-01T00:04:00Z"),
    ):
        replay.feed_group(_group(available, **{
            "1min": (_bar(label, available),)
        }))
        replay.process_next(lambda ts, base, context: (
            seen.append(base.index.tolist()) or _observation(ts, base, context)
        ))
    assert seen[-1] == [_ts("2026-01-01T00:00:00Z"), _ts("2026-01-01T00:03:00Z")]


def test_future_higher_timeframe_bar_stays_hidden_until_its_group():
    replay = _replay()
    base_at = "2026-01-01T00:02:00Z"
    higher_at = "2026-01-01T00:07:00Z"
    replay.feed_group(_group(base_at, **{
        "1min": (_bar("2026-01-01T00:01:00Z", base_at),)
    }))
    replay.feed_group(_group(higher_at, **{
        "5min": (_bar(
            "2026-01-01T00:00:00Z", higher_at,
            timeframe="5min", interval="5min",
        ),)
    }))
    observed = []
    first = replay.process_next(lambda ts, base, context: (
        observed.append((ts, len(context.frames["5min"])))
        or _observation(ts, base, context)
    ))
    assert first is not None
    assert observed == [(_ts(base_at), 0)]
    assert replay.process_next(_observation) is None


def test_callback_snapshots_cannot_mutate_internal_history():
    replay = _replay(limits={"1min": 3})
    first = "2026-01-01T00:01:00Z"
    replay.feed_group(_group(first, **{
        "1min": (_bar("2026-01-01T00:00:00Z", first, close=100),)
    }))

    def mutate(timestamp, base, context):
        base.iloc[0, base.columns.get_loc("Close")] = -999
        context.frames["1min"].iloc[0, 0] = -999
        context.frames["1min"] = pd.DataFrame()
        return _observation(timestamp, base, context)

    replay.process_next(mutate)
    second = "2026-01-01T00:02:00Z"
    replay.feed_group(_group(second, **{
        "1min": (_bar("2026-01-01T00:01:00Z", second, close=101),)
    }))
    values = []
    replay.process_next(lambda ts, base, context: (
        values.append(base["Close"].tolist()) or _observation(ts, base, context)
    ))
    assert values == [[100.0, 101.0]]


def test_callback_failure_marks_instance_failed_without_retry():
    replay = _replay()
    available = "2026-01-01T00:01:00Z"
    replay.feed_group(_group(available, **{
        "1min": (_bar("2026-01-01T00:00:00Z", available),)
    }))
    calls = []

    def fail(timestamp, base, context):
        calls.append(timestamp)
        raise LookupError("callback failure")

    with pytest.raises(LookupError, match="callback failure"):
        replay.process_next(fail)
    with pytest.raises(RuntimeError, match="failed"):
        replay.process_next(_observation)
    with pytest.raises(RuntimeError, match="failed"):
        replay.finish()
    assert calls == [_ts(available)]


def test_invalid_group_metadata_is_rejected():
    replay = _replay()
    available = "2026-01-01T00:01:00Z"
    invalid = IncrementalReplayBar(
        timeframe="1min",
        timestamp=_ts("2026-01-01T00:00:00Z"),
        open=100,
        high=99,
        low=98,
        close=100,
        interval_end=_ts(available),
        available_at=_ts(available),
    )
    with pytest.raises(ValueError, match="High"):
        replay.feed_group(_group(available, **{"1min": (invalid,)}))
    assert replay.pending_group_count == 0


def test_long_run_retains_only_configured_bars_and_no_observation_list():
    replay = _replay(limits={"1min": 12}, max_groups=2, max_bars=2)
    for index in range(1200):
        label = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=index)
        available = label + pd.Timedelta(minutes=1)
        group = ReplayAvailabilityGroup(
            available,
            {"1min": (_bar(label.isoformat(), available.isoformat()),)},
        )
        replay.feed_group(group)
        assert replay.process_next(_observation) is not None
    assert replay.retained_bar_counts == {"1min": 12}
    assert replay.pending_group_count == 0
    assert not hasattr(replay, "observations")
    replay.finish()