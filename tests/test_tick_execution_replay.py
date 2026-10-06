from __future__ import annotations

import pandas as pd
import pytest

from src.replay_engine import (
    IncrementalHistoricalReplay,
    ReplayConfig,
    ReplayDecision,
    ReplayObservation,
)
from src.replay_subject import ReplaySubject
from src.tick_aggregation import TickAggregator
from src.tick_execution import (
    DecisionRecord,
    ExecutionDirection,
    QuoteEvent,
)
from src.tick_execution_replay import TickExecutionReplay


class _DelayedEmissionAggregator(TickAggregator):
    def __init__(self, interval: str, release_at: str) -> None:
        super().__init__(interval)
        self.release_at = _ts(release_at)
        self.buffered_rows = []

    def add_tick(self, ts, bid, ask):
        timestamp = pd.Timestamp(ts).tz_convert("UTC")
        self.buffered_rows.extend(super().add_tick(ts, bid, ask))
        if timestamp != self.release_at:
            return []
        emitted = []
        for row in self.buffered_rows:
            row["available_at"] = timestamp
            row["availability_ts"] = timestamp
            emitted.append(row)
        self.buffered_rows.clear()
        return emitted


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _tick(timestamp: str, order: int, bid: float = 100.0, ask: float = 101.0) -> QuoteEvent:
    return QuoteEvent(
        timestamp=_ts(timestamp),
        bid=bid,
        ask=ask,
        source_order=order,
        provenance={"source_file": "ticks.csv", "source_row": order + 1},
    )


def _replay(
    *,
    history_limits: dict[str, int] | None = None,
) -> IncrementalHistoricalReplay:
    limits = history_limits or {"1min": 16, "3min": 8}
    return IncrementalHistoricalReplay(
        config=ReplayConfig(timeframe="1min", source="TEST"),
        history_limits=limits,
        max_pending_groups=1,
        max_pending_bars=32,
    )


def _decision(subject: ReplaySubject, *, quantity: float | None = 1.0) -> DecisionRecord:
    return DecisionRecord(
        subject=subject,
        actionable=True,
        direction=ExecutionDirection.LONG,
        order_type="MARKET",
        stop_loss=90.0,
        take_profit=120.0,
        quantity=quantity,
        strategy_state="ACTIVE_TRADE",
        evidence_reference={"as_of": subject.availability_timestamp.isoformat()},
        reason="qualified",
    )


def _run(
    ticks: list[QuoteEvent],
    *,
    replay: IncrementalHistoricalReplay | None = None,
    aggregators: dict[str, TickAggregator] | None = None,
    strategy_callback=None,
    on_tick=None,
    on_decision_set=None,
):
    replay = replay or _replay()
    aggregators = aggregators or {"1min": TickAggregator("1min")}
    strategy_calls = []
    tick_calls = []
    decision_sets = []

    def strategy(timestamp, visible_base, visible_context):
        subject = visible_context.replay_subject
        strategy_calls.append(
            (timestamp, subject, visible_base.copy(), visible_context)
        )
        if strategy_callback is not None:
            return strategy_callback(timestamp, visible_base, visible_context)
        return _decision(subject)

    def receive_tick(tick):
        tick_calls.append(tick)
        if on_tick is not None:
            on_tick(tick)

    def receive_decisions(decision_set):
        decision_sets.append(decision_set)
        if on_decision_set is not None:
            on_decision_set(decision_set)

    orchestrator = TickExecutionReplay(
        aggregators=aggregators,
        replay=replay,
        strategy_callback=strategy,
        on_tick=receive_tick,
        on_decision_set=receive_decisions,
    )
    summary = orchestrator.run(ticks)
    return summary, strategy_calls, tick_calls, decision_sets


def test_one_emission_tick_forms_one_availability_group():
    summary, _, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:00Z", 1),
        ]
    )

    assert summary.availability_groups == 1
    assert len(decision_sets) == 1
    assert decision_sets[0].available_at == _ts("2026-01-01T00:01:00Z")
    assert len(decision_sets[0].replay_group.bars_by_timeframe["1min"]) == 1


def test_one_tick_groups_completed_bars_across_configured_timeframes():
    _, _, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:05Z", 1),
            _tick("2026-01-01T00:02:05Z", 2),
            _tick("2026-01-01T00:03:00Z", 3),
        ],
        replay=_replay(history_limits={"1min": 16, "3min": 8}),
        aggregators={
            "1min": TickAggregator("1min"),
            "3min": TickAggregator("3min"),
        },
    )

    assert len(decision_sets) == 3
    group = decision_sets[-1].replay_group
    assert group.available_at == _ts("2026-01-01T00:03:00Z")
    assert set(group.bars_by_timeframe) == {"1min", "3min"}
    assert all(
        bar.available_at == group.available_at
        for timeframe_bars in group.bars_by_timeframe.values()
        for bar in timeframe_bars
    )


def test_delayed_tick_groups_multiple_base_bars_at_actual_availability():
    delayed = "2026-01-01T00:03:30Z"
    _, calls, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:05Z", 1),
            _tick("2026-01-01T00:02:05Z", 2),
            _tick(delayed, 3),
        ],
        aggregators={"1min": _DelayedEmissionAggregator("1min", delayed)},
    )

    decision_set = decision_sets[0]
    bars = decision_set.replay_group.bars_by_timeframe["1min"]
    assert [bar.timestamp for bar in bars] == [
        _ts("2026-01-01T00:00:00Z"),
        _ts("2026-01-01T00:01:00Z"),
        _ts("2026-01-01T00:02:00Z"),
    ]
    assert {bar.available_at for bar in bars} == {_ts(delayed)}
    assert len(calls) == 3
    assert {call[0] for call in calls} == {_ts(delayed)}


def test_multiple_subjects_share_availability_but_keep_distinct_identity():
    _, calls, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:05Z", 1),
            _tick("2026-01-01T00:02:05Z", 2),
            _tick("2026-01-01T00:03:30Z", 3),
        ],
        aggregators={
            "1min": _DelayedEmissionAggregator("1min", "2026-01-01T00:03:30Z")
        },
    )

    subjects = [call[1] for call in calls]
    assert len(subjects) == 3
    assert len({subject.subject_id for subject in subjects}) == 3
    assert {subject.availability_timestamp for subject in subjects} == {
        _ts("2026-01-01T00:03:30Z")
    }
    assert len(decision_sets) == 1
    assert [decision.subject.subject_id for decision in decision_sets[0].decisions] == [
        subject.subject_id for subject in subjects
    ]


def test_decision_time_is_availability_not_base_bar_timestamp():
    _, _, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:02:30Z", 1),
        ]
    )

    decision = decision_sets[0].decisions[0]
    assert decision.decision_time == decision.subject.availability_timestamp
    assert decision.decision_time == _ts("2026-01-01T00:02:30Z")
    assert decision.decision_time != decision.subject.base_bar_timestamp


def test_each_subject_receives_its_own_base_visibility_prefix():
    _, calls, _, _ = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:05Z", 1),
            _tick("2026-01-01T00:02:05Z", 2),
            _tick("2026-01-01T00:03:30Z", 3),
        ]
    )

    for _, subject, visible_base, _ in calls:
        assert visible_base.index.max() == subject.base_bar_timestamp
        assert (visible_base.index <= subject.base_bar_timestamp).all()


def test_future_sentinel_is_absent_from_each_subject_evidence_window():
    seen_rows = []

    def strategy(timestamp, visible_base, visible_context):
        seen_rows.append((visible_base.index.copy(), visible_base["Close"].tolist()))
        return _decision(visible_context.replay_subject)

    _, _, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0, 100, 101),
            _tick("2026-01-01T00:01:05Z", 1, 110, 111),
            _tick("2026-01-01T00:02:05Z", 2, 999_999, 1_000_000),
            _tick("2026-01-01T00:03:30Z", 3),
        ],
        aggregators={
            "1min": _DelayedEmissionAggregator("1min", "2026-01-01T00:03:30Z")
        },
        strategy_callback=strategy,
    )

    assert len(decision_sets[0].decisions) == 3
    first_index, first_closes = seen_rows[0]
    assert first_index.tolist() == [_ts("2026-01-01T00:00:00Z")]
    assert 999_999 not in first_closes


def test_full_decision_set_is_submitted_only_after_all_subject_callbacks():
    events = []

    def callback(timestamp, visible_base, visible_context):
        subject = visible_context.replay_subject
        events.append(("strategy", subject.base_bar_timestamp))
        return _decision(subject)

    def on_decision_set(decision_set):
        events.append(("submit", len(decision_set.decisions)))

    _, _, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:05Z", 1),
            _tick("2026-01-01T00:02:05Z", 2),
            _tick("2026-01-01T00:03:30Z", 3),
        ],
        aggregators={
            "1min": _DelayedEmissionAggregator("1min", "2026-01-01T00:03:30Z")
        },
        strategy_callback=callback,
        on_decision_set=on_decision_set,
    )

    assert events == [
        ("strategy", _ts("2026-01-01T00:00:00Z")),
        ("strategy", _ts("2026-01-01T00:01:00Z")),
        ("strategy", _ts("2026-01-01T00:02:00Z")),
        ("submit", 3),
    ]
    assert len(decision_sets) == 1


def test_tick_execution_hook_runs_before_current_tick_decisions():
    events = []

    def on_tick(tick):
        events.append(("execution-tick", tick.timestamp, tick.source_order))

    def strategy(timestamp, visible_base, visible_context):
        events.append(("strategy", timestamp))
        return _decision(visible_context.replay_subject)

    def on_decision_set(decision_set):
        events.append(("submit", decision_set.available_at))

    _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:00Z", 1),
        ],
        on_tick=on_tick,
        strategy_callback=strategy,
        on_decision_set=on_decision_set,
    )

    assert events == [
        ("execution-tick", _ts("2026-01-01T00:00:05Z"), 0),
        ("execution-tick", _ts("2026-01-01T00:01:00Z"), 1),
        ("strategy", _ts("2026-01-01T00:01:00Z")),
        ("submit", _ts("2026-01-01T00:01:00Z")),
    ]


def test_strategy_callback_input_order_does_not_submit_partial_arbitration():
    submitted = []
    callbacks = []

    def strategy(timestamp, visible_base, visible_context):
        subject = visible_context.replay_subject
        callbacks.append(subject)
        # Deliberately alternate directions so arbitration must see all subjects.
        direction = (
            ExecutionDirection.LONG
            if subject.base_bar_timestamp.minute % 2 == 0
            else ExecutionDirection.SHORT
        )
        return DecisionRecord(
            subject=subject,
            actionable=True,
            direction=direction,
            stop_loss=90.0 if direction is ExecutionDirection.LONG else 110.0,
            take_profit=120.0 if direction is ExecutionDirection.LONG else 80.0,
            quantity=1.0,
        )

    _, _, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:05Z", 1),
            _tick("2026-01-01T00:02:05Z", 2),
            _tick("2026-01-01T00:03:30Z", 3),
        ],
        aggregators={
            "1min": _DelayedEmissionAggregator("1min", "2026-01-01T00:03:30Z")
        },
        strategy_callback=strategy,
        on_decision_set=lambda batch: submitted.append(batch),
    )

    assert len(callbacks) == 3
    assert len(submitted) == 1
    assert submitted[0] is decision_sets[0]
    assert len(submitted[0].decisions) == 3
    assert {decision.direction for decision in submitted[0].decisions} == {
        ExecutionDirection.LONG,
        ExecutionDirection.SHORT,
    }


def test_missing_actionable_quantity_is_an_explicit_integration_failure():
    with pytest.raises(ValueError, match="explicit quantity"):
        _run(
            [
                _tick("2026-01-01T00:00:05Z", 0),
                _tick("2026-01-01T00:01:00Z", 1),
            ],
            strategy_callback=lambda _ts, _base, context: _decision(
                context.replay_subject,
                quantity=None,
            ),
        )


def test_planned_strategy_entry_is_not_a_fill_price_or_execution_field():
    planned_entry = 100.5
    received_tick_quotes = []
    received_decisions = []

    def on_tick(tick):
        received_tick_quotes.append((tick.bid, tick.ask))

    def strategy(_timestamp, _visible_base, visible_context):
        decision = _decision(visible_context.replay_subject)
        assert not hasattr(decision, "entry_price")
        assert not hasattr(decision, "fill_price")
        return DecisionRecord(
            subject=decision.subject,
            actionable=decision.actionable,
            direction=decision.direction,
            order_type=decision.order_type,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            quantity=decision.quantity,
            strategy_state=decision.strategy_state,
            evidence_reference={"planned_entry": planned_entry},
            reason=decision.reason,
        )

    _, _, _, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:01:00Z", 1, bid=100.0, ask=101.0),
        ],
        on_tick=on_tick,
        strategy_callback=strategy,
        on_decision_set=lambda batch: received_decisions.extend(batch.decisions),
    )

    assert planned_entry not in [price for quote in received_tick_quotes for price in quote]
    assert received_decisions == list(decision_sets[0].decisions)
    assert received_decisions[0].evidence_reference == {"planned_entry": planned_entry}
    assert not hasattr(decision_sets[0], "fills")


def test_gaps_do_not_create_synthetic_ticks_or_bars():
    summary, _, tick_calls, decision_sets = _run(
        [
            _tick("2026-01-01T00:00:05Z", 0),
            _tick("2026-01-01T00:05:00Z", 1),
        ]
    )

    assert summary.ticks_processed == 2
    assert [tick.timestamp for tick in tick_calls] == [
        _ts("2026-01-01T00:00:05Z"),
        _ts("2026-01-01T00:05:00Z"),
    ]
    assert len(decision_sets) == 1
    assert [
        bar.timestamp
        for bar in decision_sets[0].replay_group.bars_by_timeframe["1min"]
    ] == [_ts("2026-01-01T00:00:00Z")]


def test_tick_with_no_emitted_bars_does_not_submit_decision_set():
    summary, _, _, decision_sets = _run([_tick("2026-01-01T00:00:05Z", 0)])

    assert summary.availability_groups == 0
    assert summary.decision_sets_submitted == 0
    assert decision_sets == []
