from __future__ import annotations

import pandas as pd
import pytest

from src.replay_engine import (
    IncrementalHistoricalReplay,
    IncrementalReplayBar,
    ReplayAvailabilityGroup,
    ReplayConfig,
)
from src.replay_subject import ReplaySubject
from src.tick_aggregation import TickAggregator
from src.tick_execution import (
    DecisionRecord,
    ExecutionConfig,
    ExecutionDirection,
    ExecutionEngine,
    ExecutionStatus,
    QuoteEvent,
)
from src.tick_execution_replay import TickExecutionReplay


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def tick(
    value: str,
    order: int,
    bid: float = 100.0,
    ask: float = 101.0,
) -> QuoteEvent:
    return QuoteEvent(ts(value), bid, ask, order, {"source_row": order + 1})


def subject(available: str, base: str | None = None) -> ReplaySubject:
    available_at = ts(available)
    base_at = ts(base) if base else available_at - pd.Timedelta(minutes=1)
    return ReplaySubject(available_at, base_at)


def decision(
    available: str,
    direction: ExecutionDirection = ExecutionDirection.LONG,
    *,
    base: str | None = None,
    quantity: float | None = 1.0,
    stop: float | None = None,
    target: float | None = None,
    evidence: object = None,
) -> DecisionRecord:
    if stop is None:
        stop = 120.0 if direction is ExecutionDirection.SHORT else 90.0
    if target is None:
        target = 90.0 if direction is ExecutionDirection.SHORT else 120.0
    return DecisionRecord(
        subject(available, base),
        actionable=True,
        direction=direction,
        order_type="MARKET",
        stop_loss=stop,
        take_profit=target,
        quantity=quantity,
        strategy_state="QUALIFIED",
        evidence_reference=evidence or {"fixture": True},
    )


def decision_set(
    available: str,
    decisions: tuple[DecisionRecord, ...],
) -> object:
    from src.tick_execution_replay import TickExecutionDecisionSet

    available_at = ts(available)
    bar = IncrementalReplayBar(
        timeframe="1min",
        timestamp=available_at - pd.Timedelta(minutes=1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        available_at=available_at,
    )
    group = ReplayAvailabilityGroup(available_at, {"1min": (bar,)})
    return TickExecutionDecisionSet(available_at, group, decisions)


def engine() -> ExecutionEngine:
    return ExecutionEngine(
        execution_run_id="stateful-test",
        initial_equity=10_000.0,
        config=ExecutionConfig(
            contract_size=1.0,
            minimum_quantity=0.1,
            quantity_step=0.1,
            account_currency_conversion=1.0,
            commission_per_unit_per_side=0.0,
            slippage_per_unit_per_side=0.0,
        ),
    )


def test_existing_pending_intent_is_processed_before_same_tick_decisions():
    execution = engine()
    first = decision("2026-01-01T00:00:00Z")
    second = decision("2026-01-01T00:00:01Z")
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (first,)))

    execution.process_tick(tick("2026-01-01T00:00:01Z", 1, 102.0, 103.0))
    assert execution.fills[0].subject.subject_id == first.subject.subject_id
    assert execution.fills[0].observed_price == 103.0
    execution.submit_decision_set(decision_set("2026-01-01T00:00:01Z", (second,)))
    results = execution.finish()

    assert results[1].status is ExecutionStatus.NO_TRADE_ACTIVE_POSITION


def test_decision_timestamp_equal_to_observed_tick_cannot_fill_on_that_tick():
    execution = engine()
    proposed = decision("2026-01-01T00:00:00Z")
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0, 98.0, 99.0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (proposed,)))
    execution.process_tick(tick("2026-01-01T00:00:00Z", 1, 98.0, 99.0))

    assert execution.fills == []
    assert execution.intents[0].decision_time == ts("2026-01-01T00:00:00Z")


def test_first_strictly_later_tick_fills_at_its_quote():
    execution = engine()
    proposed = decision("2026-01-01T00:00:00Z")
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (proposed,)))
    execution.process_tick(tick("2026-01-01T00:00:01Z", 1, 102.0, 103.0))

    assert execution.fills[0].timestamp == ts("2026-01-01T00:00:01Z")
    assert execution.fills[0].source_order == 1
    assert execution.fills[0].observed_price == 103.0


def test_submit_requires_processed_tick_and_complete_replay_group():
    execution = engine()
    proposed = decision("2026-01-01T00:00:00Z")
    with pytest.raises(RuntimeError, match="Process the availability tick"):
        execution.submit_decision_set((proposed,))

    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    with pytest.raises(TypeError, match="complete availability decision set"):
        execution.submit_decision_set((proposed,))
    complete_set = decision_set("2026-01-01T00:00:00Z", (proposed,))
    execution.submit_decision_set(complete_set)
    with pytest.raises(ValueError, match="already submitted"):
        execution.submit_decision_set(complete_set)


def test_stateful_tick_stream_rejects_nonincreasing_source_order():
    execution = engine()
    execution.process_tick(tick("2026-01-01T00:00:00Z", 5))
    with pytest.raises(ValueError, match="source_order must be strictly increasing"):
        execution.process_tick(tick("2026-01-01T00:00:01Z", 5))


def test_conflicting_actionable_directions_reject_the_complete_group():
    execution = engine()
    long = decision("2026-01-01T00:01:00Z", base="2026-01-01T00:00:00Z")
    short = decision(
        "2026-01-01T00:01:00Z",
        ExecutionDirection.SHORT,
        base="2026-01-01T00:00:30Z",
    )
    execution.process_tick(tick("2026-01-01T00:01:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:01:00Z", (long, short)))
    results = execution.finish()

    assert execution.intents == []
    assert all(result.status is ExecutionStatus.NO_TRADE_CONFLICTING_DIRECTIONS for result in results)


def test_latest_same_direction_subject_wins_independent_of_callback_order():
    execution = engine()
    older = decision(
        "2026-01-01T00:01:00Z",
        base="2026-01-01T00:00:00Z",
    )
    latest = decision(
        "2026-01-01T00:01:00Z",
        base="2026-01-01T00:00:30Z",
    )
    execution.process_tick(tick("2026-01-01T00:01:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:01:00Z", (latest, older)))
    results = execution.finish()

    assert execution.decision_sets[0].selected_subject_id == latest.subject.subject_id
    assert results[0].intent.subject.subject_id == latest.subject.subject_id
    assert results[1].status is ExecutionStatus.VALID_DECISION_NOT_SELECTED


def test_active_position_blocks_later_actionable_candidate():
    execution = engine()
    first = decision("2026-01-01T00:00:00Z")
    second = decision("2026-01-01T00:00:02Z")
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (first,)))
    execution.process_tick(tick("2026-01-01T00:00:01Z", 1))
    execution.process_tick(tick("2026-01-01T00:00:02Z", 2))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:02Z", (second,)))
    results = execution.finish()

    assert results[1].status is ExecutionStatus.NO_TRADE_ACTIVE_POSITION
    assert len(execution.positions) == 1


@pytest.mark.parametrize(
    ("direction", "expected_side", "expected_price"),
    [
        (ExecutionDirection.LONG, "ASK", 103.0),
        (ExecutionDirection.SHORT, "BID", 102.0),
    ],
)
def test_stateful_entry_uses_directional_executable_quote_side(
    direction,
    expected_side,
    expected_price,
):
    execution = engine()
    proposed = decision("2026-01-01T00:00:00Z", direction)
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (proposed,)))
    execution.process_tick(tick("2026-01-01T00:00:01Z", 1, 102.0, 103.0))

    assert execution.fills[0].side == expected_side
    assert execution.fills[0].observed_price == expected_price


def test_stateful_exit_uses_opposite_executable_side_and_inclusive_trigger():
    execution = engine()
    proposed = decision("2026-01-01T00:00:00Z", target=104.0)
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (proposed,)))
    execution.process_tick(tick("2026-01-01T00:00:01Z", 1, 102.0, 103.0))
    assert len(execution.fills) == 1
    execution.process_tick(tick("2026-01-01T00:00:02Z", 2, 104.0, 105.0))

    results = execution.finish()
    assert len(execution.fills) == 2
    assert execution.fills[1].side == "BID"
    assert execution.fills[1].observed_price == 104.0
    assert results[0].status is ExecutionStatus.CLOSED


@pytest.mark.parametrize(
    ("direction", "stop", "entry_bid", "entry_ask", "exit_bid", "exit_ask", "exit_side"),
    [
        (ExecutionDirection.LONG, 100.0, 100.0, 101.0, 100.0, 101.0, "BID"),
        (ExecutionDirection.SHORT, 101.0, 100.0, 101.0, 100.0, 101.0, "ASK"),
    ],
)
def test_stateful_stop_loss_trigger_is_inclusive(
    direction,
    stop,
    entry_bid,
    entry_ask,
    exit_bid,
    exit_ask,
    exit_side,
):
    execution = engine()
    proposed = decision(
        "2026-01-01T00:00:00Z",
        direction,
        stop=stop,
        target=120.0 if direction is ExecutionDirection.LONG else 90.0,
    )
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (proposed,)))
    execution.process_tick(tick("2026-01-01T00:00:01Z", 1, entry_bid, entry_ask))
    assert len(execution.fills) == 1
    execution.process_tick(tick("2026-01-01T00:00:02Z", 2, exit_bid, exit_ask))

    assert len(execution.fills) == 2
    assert execution.fills[-1].side == exit_side
    assert execution.fills[-1].observed_price == stop


def test_planned_entry_price_cannot_be_used_as_fill_price():
    execution = engine()
    proposed = decision(
        "2026-01-01T00:00:00Z",
        evidence={"planned_entry_price": 1.25},
    )
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (proposed,)))
    execution.process_tick(tick("2026-01-01T00:00:01Z", 1, 102.0, 103.0))

    assert execution.fills[0].observed_price == 103.0
    assert execution.fills[0].observed_price != proposed.evidence_reference["planned_entry_price"]


def test_missing_quantity_is_not_inferred():
    execution = engine()
    proposed = decision("2026-01-01T00:00:00Z", quantity=None)
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (proposed,)))
    result = execution.finish()[0]

    assert execution.intents == []
    assert result.status is ExecutionStatus.UNSPECIFIED_QUANTITY_ROUNDING
    assert "quantity rounding policy is UNSPECIFIED" in result.reason


def test_no_synthetic_tick_or_fill_is_created():
    execution = engine()
    proposed = decision("2026-01-01T00:00:00Z")
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (proposed,)))
    assert execution.fills == []

    result = execution.finish()[0]
    assert execution.fills == []
    assert result.status is ExecutionStatus.UNFILLED_AT_END_OF_INPUT
    assert len(execution.orders) == 1


def test_pending_and_open_position_eof_statuses_remain_unchanged():
    pending_engine = engine()
    pending = decision("2026-01-01T00:00:00Z")
    pending_engine.process_tick(tick("2026-01-01T00:00:00Z", 0))
    pending_engine.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (pending,)))
    pending_result = pending_engine.finish()[0]
    assert pending_result.status is ExecutionStatus.UNFILLED_AT_END_OF_INPUT

    open_engine = engine()
    opened = decision("2026-01-01T00:00:00Z")
    open_engine.process_tick(tick("2026-01-01T00:00:00Z", 0))
    open_engine.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (opened,)))
    open_engine.process_tick(tick("2026-01-01T00:00:01Z", 1))
    open_result = open_engine.finish()[0]
    assert open_result.status is ExecutionStatus.OPEN_AT_END_OF_INPUT_UNSPECIFIED_VALUATION
    assert open_result.position.exit_fill is None


def test_execution_ids_and_subject_identity_remain_distinct_across_subjects():
    execution = engine()
    first = decision("2026-01-01T00:00:00Z", target=102.0)
    second = decision("2026-01-01T00:00:02Z", base="2026-01-01T00:00:01Z")
    execution.process_tick(tick("2026-01-01T00:00:00Z", 0))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:00Z", (first,)))
    execution.process_tick(tick("2026-01-01T00:00:01Z", 1, 101.0, 102.0))
    execution.process_tick(tick("2026-01-01T00:00:02Z", 2))
    execution.submit_decision_set(decision_set("2026-01-01T00:00:02Z", (second,)))
    execution.process_tick(tick("2026-01-01T00:00:03Z", 3))
    results = execution.finish()

    all_ids = [intent.intent_id for intent in execution.intents]
    all_ids += [order.order_id for order in execution.orders]
    all_ids += [fill.fill_id for fill in execution.fills]
    all_ids += [position.position_id for position in execution.positions]
    assert len(all_ids) == len(set(all_ids))
    assert results[0].subject.subject_id == first.subject.subject_id
    assert results[1].subject.subject_id == second.subject.subject_id
    assert execution.fills[-1].subject.subject_id == second.subject.subject_id


def test_orchestration_drives_stateful_execution_in_complete_causal_order():
    execution = engine()
    events: list[str] = []

    def strategy(available_at, visible_base, context):
        events.append("strategy")
        return decision(
            available_at.isoformat(),
            evidence={"planned_entry_price": 1.25},
        )

    def on_tick(current_tick):
        execution.process_tick(current_tick)
        events.append(f"tick:{current_tick.source_order}")

    def on_decision_set(complete_set):
        events.append("complete-decision-set")
        assert complete_set.decisions
        assert execution.fills == []
        execution.submit_decision_set(complete_set)

    replay = IncrementalHistoricalReplay(
        config=ReplayConfig(timeframe="1min", source="TEST"),
        history_limits={"1min": 8},
        max_pending_groups=1,
        max_pending_bars=8,
    )
    orchestrator = TickExecutionReplay(
        aggregators={"1min": TickAggregator("1min")},
        replay=replay,
        strategy_callback=strategy,
        on_tick=on_tick,
        on_decision_set=on_decision_set,
    )
    summary = orchestrator.run(
        [
            tick("2026-01-01T00:00:05Z", 0),
            tick("2026-01-01T00:01:00Z", 1),
            tick("2026-01-01T00:01:01Z", 2, 102.0, 103.0),
        ]
    )
    results = execution.finish()

    assert events == [
        "tick:0",
        "tick:1",
        "strategy",
        "complete-decision-set",
        "tick:2",
    ]
    assert summary.decision_sets_submitted == 1
    assert execution.fills[0].source_order == 2
    assert execution.fills[0].observed_price == 103.0
    assert results[0].entry_fill == execution.fills[0]
