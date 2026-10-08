from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from src.replay_subject import ReplaySubject
from src.tick_execution import (
    DecisionRecord,
    ExecutionConfig,
    ExecutionDirection,
    ExecutionEngine,
    ExecutionStatus,
    PositionState,
    QuoteEvent,
    quote_events_from_tick_batches,
)


def timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def subject(available: str, base: str | None = None) -> ReplaySubject:
    decision_time = timestamp(available)
    base_time = timestamp(base) if base is not None else decision_time - pd.Timedelta(minutes=1)
    return ReplaySubject(decision_time, base_time)


def decision(
    available: str,
    direction: ExecutionDirection | str | None = ExecutionDirection.LONG,
    *,
    base: str | None = None,
    actionable: bool = True,
    stop: float | None = None,
    target: float | None = None,
    quantity: float | None = 1.0,
    order_type: str = "MARKET",
) -> DecisionRecord:
    normalized_direction = direction
    if direction is not None and not isinstance(direction, ExecutionDirection):
        normalized_direction = ExecutionDirection(str(direction).upper())
    if stop is None:
        stop = 120.0 if normalized_direction is ExecutionDirection.SHORT else 90.0
    if target is None:
        target = 90.0 if normalized_direction is ExecutionDirection.SHORT else 120.0
    return DecisionRecord(
        subject=subject(available, base),
        actionable=actionable,
        direction=direction,
        order_type=order_type,
        stop_loss=stop,
        take_profit=target,
        quantity=quantity,
        strategy_state="QUALIFIED" if actionable else "NO_SIGNAL",
        evidence_reference={"evidence": "fixture"},
    )


def quote(
    when: str,
    bid: float = 100.0,
    ask: float = 101.0,
    order: int = 0,
    **provenance: object,
) -> QuoteEvent:
    if not provenance:
        provenance = {"fixture": "test"}
    return QuoteEvent(timestamp(when), bid, ask, order, provenance)


def config(
    *,
    contract_size: float | None = 1.0,
    minimum_quantity: float | None = 0.1,
    quantity_step: float | None = 0.1,
    conversion: float | None = 1.0,
    commission: float | None = 0.0,
    slippage: float = 0.0,
) -> ExecutionConfig:
    return ExecutionConfig(
        contract_size=contract_size,
        minimum_quantity=minimum_quantity,
        quantity_step=quantity_step,
        account_currency_conversion=conversion,
        commission_per_unit_per_side=commission,
        slippage_per_unit_per_side=slippage,
    )


def execute(
    ticks: list[QuoteEvent],
    decisions: list[DecisionRecord],
    *,
    execution_run_id: str = "run-1",
    initial_equity: float = 10_000.0,
    execution_config: ExecutionConfig | None = None,
):
    engine = ExecutionEngine(
        execution_run_id=execution_run_id,
        initial_equity=initial_equity,
        config=execution_config or config(),
    )
    return engine, engine.run(ticks, decisions)


def test_ticks_preserve_order_equal_timestamps_and_provenance():
    batches = [
        pd.DataFrame(
            [
                {"timestamp": timestamp("2026-01-01T00:00:00Z"), "bid": 100, "ask": 101, "source_order": 4, "source_file": "a.csv"},
                {"timestamp": timestamp("2026-01-01T00:00:00Z"), "bid": 99, "ask": 100, "source_order": 5, "source_file": "a.csv"},
            ]
        )
    ]
    ticks = list(quote_events_from_tick_batches(batches))
    assert [tick.source_order for tick in ticks] == [4, 5]
    assert [tick.bid for tick in ticks] == [100.0, 99.0]
    assert [tick.provenance["source_file"] for tick in ticks] == ["a.csv", "a.csv"]


def test_quote_requires_nonempty_source_provenance():
    with pytest.raises(ValueError, match="non-empty mapping"):
        QuoteEvent(timestamp("2026-01-01T00:00:00Z"), 100, 101, 0)
    with pytest.raises(ValueError, match="non-empty mapping"):
        QuoteEvent.from_mapping(
            {
                "timestamp": timestamp("2026-01-01T00:00:00Z"),
                "bid": 100,
                "ask": 101,
                "source_order": 0,
            }
        )
    with pytest.raises(ValueError, match="provenance must be a mapping"):
        QuoteEvent.from_mapping(
            {
                "timestamp": timestamp("2026-01-01T00:00:00Z"),
                "bid": 100,
                "ask": 101,
                "source_order": 0,
                "provenance": "unstructured source reference",
            }
        )


def test_quote_provenance_is_deeply_bounded_and_immutable():
    original = {"source_file": "ticks.csv", "metadata": {"labels": ["canonical", "tick"]}}
    event = QuoteEvent(
        timestamp("2026-01-01T00:00:00Z"),
        100,
        101,
        0,
        original,
    )
    original["metadata"]["labels"].append("mutated")
    original["metadata"]["frame"] = pd.DataFrame({"close": [100.0]})
    original["new"] = "caller mutation"

    assert event.provenance["source_file"] == "ticks.csv"
    assert event.provenance["metadata"]["labels"] == ("canonical", "tick")
    assert "frame" not in event.provenance["metadata"]
    assert "new" not in event.provenance
    with pytest.raises(TypeError):
        event.provenance["source_file"] = "changed"
    with pytest.raises(TypeError):
        event.provenance["metadata"]["source"] = "changed"


@pytest.mark.parametrize(
    "payload",
    [
        pd.DataFrame({"close": [100.0]}),
        pd.Series([100.0]),
        pd.Index([100.0]),
        {"nested": {"frame": pd.DataFrame({"close": [100.0]})}},
        {"nested": [pd.Series([100.0])]},
        {"unsupported": {1, 2}},
    ],
)
def test_quote_provenance_rejects_unbounded_nested_payloads(payload):
    with pytest.raises(TypeError):
        QuoteEvent(
            timestamp("2026-01-01T00:00:00Z"),
            100,
            101,
            0,
            {"metadata": payload},
        )


@pytest.mark.parametrize(
    "provenance",
    [
        {"x" * 4097: "value"},
        {"metadata": "x" * 4097},
    ],
)
def test_quote_provenance_rejects_oversized_keys_and_strings(provenance):
    with pytest.raises(TypeError):
        QuoteEvent(
            timestamp("2026-01-01T00:00:00Z"),
            100,
            101,
            0,
            provenance,
        )


@pytest.mark.parametrize(
    ("bid", "ask"),
    [
        (0.0, 101.0),
        (100.0, float("nan")),
        (float("inf"), 101.0),
        (101.0, 100.0),
    ],
)
def test_malformed_nonfinite_or_crossed_quotes_are_rejected(bid, ask):
    with pytest.raises(ValueError):
        quote("2026-01-01T00:00:00Z", bid, ask, 0)


def test_decreasing_tick_order_is_rejected_without_reordering():
    with pytest.raises(ValueError, match="non-decreasing"):
        execute(
            [
                quote("2026-01-01T00:01:00Z", order=0),
                quote("2026-01-01T00:00:00Z", order=1),
            ],
            [],
        )


def test_nonincreasing_source_order_is_rejected():
    with pytest.raises(ValueError, match="source_order must be strictly increasing"):
        execute(
            [
                quote("2026-01-01T00:00:00Z", order=4),
                quote("2026-01-01T00:00:00Z", order=4),
            ],
            [],
        )


def test_strict_later_fill_ignores_all_same_timestamp_quotes():
    subject_decision = decision("2026-01-01T00:00:00Z")
    engine, results = execute(
        [
            quote("2026-01-01T00:00:00Z", 100, 101, 0),
            quote("2026-01-01T00:00:00Z", 98, 99, 1),
            quote("2026-01-01T00:00:01Z", 102, 103, 2),
        ],
        [subject_decision],
    )
    assert results[0].entry_fill is not None
    assert results[0].entry_fill.timestamp == timestamp("2026-01-01T00:00:01Z")
    assert results[0].entry_fill.source_order == 2
    assert len(engine.fills) == 1


@pytest.mark.parametrize(
    ("direction", "expected_side", "expected_price"),
    [
        (ExecutionDirection.LONG, "ASK", 103.0),
        (ExecutionDirection.SHORT, "BID", 102.0),
    ],
)
def test_market_entry_uses_executable_quote_side(direction, expected_side, expected_price):
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0), quote("2026-01-01T00:00:01Z", 102, 103, 1)],
        [decision("2026-01-01T00:00:00Z", direction)],
    )
    assert results[0].entry_fill.side == expected_side
    assert results[0].entry_fill.observed_price == expected_price


def test_subject_identity_propagates_through_execution_records():
    engine, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 101, 102, 1, source_file="ticks.csv", source_row=7),
        ],
        [decision("2026-01-01T00:00:00Z")],
    )
    result = results[0]
    assert result.intent.subject == result.subject
    assert result.order.subject == result.subject
    assert result.entry_fill.subject == result.subject
    assert result.position.subject == result.subject
    assert result.entry_fill.provenance["source_row"] == 7
    identity = json.loads(result.entry_fill.execution_identity)
    assert identity == ["run-1", result.subject.subject_id, "market-entry"]
    assert engine.positions[0].entry_fill == result.entry_fill


def test_fill_provenance_is_canonical_and_rejects_mutable_payloads():
    entry_original = {"source_file": "ticks.csv", "metadata": {"source_row": 7}}
    exit_original = {"source_file": "ticks.csv", "metadata": {"source_row": 9}}
    entry_tick = QuoteEvent(
        timestamp("2026-01-01T00:00:00Z"),
        100,
        101,
        0,
        {"source_file": "ticks.csv", "source_row": 6},
    )
    fill_tick = QuoteEvent(
        timestamp("2026-01-01T00:00:01Z"),
        101,
        102,
        1,
        entry_original,
    )
    exit_tick = QuoteEvent(
        timestamp("2026-01-01T00:00:02Z"),
        103,
        104,
        2,
        exit_original,
    )
    entry_original["metadata"]["source_row"] = 99
    entry_original["metadata"]["frame"] = pd.DataFrame({"close": [100.0]})
    exit_original["metadata"]["source_row"] = 99
    _, results = execute(
        [entry_tick, fill_tick, exit_tick],
        [decision("2026-01-01T00:00:00Z", target=103)],
    )
    entry_fill = results[0].entry_fill
    exit_fill = results[0].exit_fill

    assert entry_fill.provenance["metadata"]["source_row"] == 7
    assert entry_fill.timestamp == fill_tick.timestamp
    assert entry_fill.source_order == fill_tick.source_order
    assert exit_fill.provenance["metadata"]["source_row"] == 9
    assert exit_fill.timestamp == exit_tick.timestamp
    assert exit_fill.source_order == exit_tick.source_order
    mutable_fill_provenance = {"metadata": {"labels": ["entry"]}}
    independent_fill = replace(
        entry_fill,
        provenance=mutable_fill_provenance,
    )
    mutable_fill_provenance["metadata"]["labels"].append("caller mutation")
    mutable_fill_provenance["metadata"]["frame"] = pd.DataFrame({"close": [100.0]})
    assert independent_fill.provenance["metadata"]["labels"] == ("entry",)
    assert "frame" not in independent_fill.provenance["metadata"]
    with pytest.raises(TypeError):
        replace(
            entry_fill,
            provenance={"nested": {"frame": pd.DataFrame({"close": [100.0]})}},
        )


def test_conflicting_same_availability_directions_create_no_intent():
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0)],
        [
            decision("2026-01-01T00:00:00Z", ExecutionDirection.LONG),
            decision("2026-01-01T00:00:00Z", ExecutionDirection.SHORT, base="2025-12-31T23:58:00Z"),
            decision("2026-01-01T00:00:00Z", None, base="2025-12-31T23:57:00Z"),
        ],
    )
    assert {result.status for result in results} == {
        ExecutionStatus.NO_TRADE_CONFLICTING_DIRECTIONS
    }


def test_same_direction_group_selects_latest_base_subject_independent_of_order():
    earlier = decision("2026-01-01T00:00:00Z", base="2025-12-31T23:57:00Z")
    latest = decision("2026-01-01T00:00:00Z", base="2025-12-31T23:59:00Z")
    engine, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0)],
        [earlier, latest],
    )
    by_subject = {result.subject.subject_id: result for result in results}
    assert by_subject[earlier.subject.subject_id].status is ExecutionStatus.VALID_DECISION_NOT_SELECTED
    assert by_subject[latest.subject.subject_id].status is ExecutionStatus.UNFILLED_AT_END_OF_INPUT
    assert by_subject[latest.subject.subject_id].intent is not None
    assert engine.decision_sets[0].selected_subject_id == latest.subject.subject_id
    assert len(engine.intents) == 1


def test_nonactionable_opposing_direction_does_not_create_conflict():
    long_decision = decision("2026-01-01T00:00:00Z", ExecutionDirection.LONG)
    inactive_short = decision(
        "2026-01-01T00:00:00Z",
        ExecutionDirection.SHORT,
        base="2025-12-31T23:58:00Z",
        actionable=False,
    )
    engine, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0)],
        [inactive_short, long_decision],
    )
    by_subject = {result.subject.subject_id: result for result in results}
    assert by_subject[long_decision.subject.subject_id].status is ExecutionStatus.UNFILLED_AT_END_OF_INPUT
    assert by_subject[inactive_short.subject.subject_id].status is ExecutionStatus.NO_TRADE_NOT_ACTIONABLE
    assert engine.decision_sets[0].arbitration_status == "SELECTED_LATEST_BASE_BAR"
    assert len(engine.intents) == 1


def test_existing_intent_matches_tick_before_same_time_decision_group():
    first = decision("2026-01-01T00:00:00Z")
    next_decision = decision("2026-01-01T00:00:01Z", ExecutionDirection.SHORT)
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 101, 102, 1),
            quote("2026-01-01T00:00:02Z", 101, 102, 2),
        ],
        [first, next_decision],
    )
    by_subject = {result.subject.subject_id: result for result in results}
    assert by_subject[first.subject.subject_id].entry_fill.timestamp == timestamp("2026-01-01T00:00:01Z")
    assert by_subject[next_decision.subject.subject_id].status is ExecutionStatus.NO_TRADE_ACTIVE_POSITION


def test_active_position_rejects_opposing_signal_without_reversal_or_queue():
    first = decision("2026-01-01T00:00:00Z", ExecutionDirection.LONG, stop=90, target=120)
    second = decision("2026-01-01T00:00:02Z", ExecutionDirection.SHORT, base="2026-01-01T00:00:01Z")
    engine, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
            quote("2026-01-01T00:00:02Z", 100, 101, 2),
        ],
        [first, second],
    )
    assert results[1].status is ExecutionStatus.NO_TRADE_ACTIVE_POSITION
    assert len(engine.positions) == 1
    assert engine.positions[0].direction is ExecutionDirection.LONG
    assert len(engine.intents) == 1


def test_entry_quote_cannot_trigger_exit():
    subject_decision = decision(
        "2026-01-01T00:00:00Z", stop=101.5, target=110.0
    )
    engine, results = execute(
        [
            quote("2026-01-01T00:00:00Z", 100, 101, 0),
            quote("2026-01-01T00:00:01Z", 101, 102, 1),
        ],
        [subject_decision],
    )
    assert results[0].entry_fill is not None
    assert results[0].entry_fill.observed_price == 102
    assert results[0].exit_fill is None
    assert engine.positions[0].state is PositionState.OPEN
    assert results[0].status is ExecutionStatus.OPEN_AT_END_OF_INPUT
    assert results[0].net_realized_pnl is None
    assert results[0].mark_to_market.side == "BID"
    assert results[0].mark_to_market.price == 101


def test_later_source_order_quote_at_fill_timestamp_can_trigger_exit():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
            quote("2026-01-01T00:00:01Z", 100.5, 101.5, 2),
        ],
        [decision("2026-01-01T00:00:00Z", stop=100.5, target=110)],
    )
    assert results[0].entry_fill.source_order == 1
    assert results[0].exit_fill.source_order == 2
    assert results[0].exit_fill.timestamp == results[0].entry_fill.timestamp
    assert results[0].reason == "STOP_LOSS"


@pytest.mark.parametrize(
    ("direction", "stop", "target", "trigger_bid", "trigger_ask", "expected_side", "expected_price", "terminal_reason"),
    [
        (ExecutionDirection.LONG, 95.0, 110.0, 95.0, 96.0, "BID", 95.0, "STOP_LOSS"),
        (ExecutionDirection.LONG, 95.0, 110.0, 111.0, 112.0, "BID", 111.0, "TAKE_PROFIT"),
        (ExecutionDirection.SHORT, 110.0, 95.0, 109.0, 110.0, "ASK", 110.0, "STOP_LOSS"),
        (ExecutionDirection.SHORT, 110.0, 95.0, 94.0, 95.0, "ASK", 95.0, "TAKE_PROFIT"),
    ],
)
def test_exit_sides_inclusive_triggers_and_actual_quote(
    direction, stop, target, trigger_bid, trigger_ask, expected_side, expected_price, terminal_reason
):
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", 100, 101, 0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
            quote("2026-01-01T00:00:02Z", trigger_bid, trigger_ask, 2),
        ],
        [decision("2026-01-01T00:00:00Z", direction, stop=stop, target=target)],
    )
    assert results[0].exit_fill.side == expected_side
    assert results[0].exit_fill.observed_price == expected_price
    assert results[0].exit_fill.timestamp == timestamp("2026-01-01T00:00:02Z")
    assert results[0].reason == terminal_reason
    assert results[0].position.state is PositionState.CLOSED
    assert results[0].entry_fill.provenance["fixture"] == "test"
    assert results[0].exit_fill.provenance["fixture"] == "test"


def test_gap_uses_first_observed_later_quote_without_synthetic_fill():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-05T00:00:00Z", 150, 151, 1),
        ],
        [decision("2026-01-01T00:00:00Z", stop=140, target=160)],
    )
    fill = results[0].entry_fill
    assert fill.timestamp == timestamp("2026-01-05T00:00:00Z")
    assert fill.observed_price == 151


def test_eof_without_later_tick_records_unfilled_intent():
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0)],
        [decision("2026-01-01T00:00:00Z")],
    )
    assert results[0].status is ExecutionStatus.UNFILLED_AT_END_OF_INPUT
    assert results[0].entry_fill is None
    assert results[0].position is None
    assert results[0].order.status == ExecutionStatus.UNFILLED_AT_END_OF_INPUT.value


def test_execution_ids_are_unique_across_subjects_and_runs():
    first = decision("2026-01-01T00:00:00Z", base="2025-12-31T23:57:00Z")
    second = decision("2026-01-01T00:00:00Z", base="2025-12-31T23:58:00Z")
    ticks = [quote("2026-01-01T00:00:00Z", order=0), quote("2026-01-01T00:00:01Z", order=1)]
    _, first_results = execute(ticks, [first], execution_run_id="run-a")
    _, second_results = execute(ticks, [second], execution_run_id="run-a")
    _, third_results = execute(ticks, [first], execution_run_id="run-b")
    ids = [
        first_results[0].intent.intent_id,
        second_results[0].intent.intent_id,
        third_results[0].intent.intent_id,
    ]
    assert len(set(ids)) == 3


def test_execution_record_identities_are_unique_within_a_closed_trade():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
            quote("2026-01-01T00:00:02Z", 120, 121, 2),
        ],
        [decision("2026-01-01T00:00:00Z", stop=90, target=120)],
    )
    result = results[0]
    identities = [
        result.intent.intent_id,
        result.order.order_id,
        result.entry_fill.fill_id,
        result.position.position_id,
        result.exit_order.order_id,
        result.exit_fill.fill_id,
    ]
    assert len(set(identities)) == len(identities)


def test_risk_sizing_derives_quantity_below_one_percent_ceiling():
    engine, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0), quote("2026-01-01T00:00:01Z", 100, 101, 1)],
        [decision("2026-01-01T00:00:00Z", stop=90, target=120, quantity=2)],
        initial_equity=10_000,
        execution_config=config(contract_size=10),
    )
    result = results[0]
    assert result.entry_fill.quantity == 0.9
    assert result.intent.requested_quantity == 0.9
    assert result.intent.risk_budget == 100
    assert result.intent.price_risk == 99
    assert result.intent.price_risk <= result.intent.risk_budget
    assert engine.positions[0].quantity == 0.9


def test_position_at_exactly_one_percent_price_risk_is_accepted():
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0), quote("2026-01-01T00:00:01Z", 100, 101, 1)],
        [decision("2026-01-01T00:00:00Z", stop=91, target=120, quantity=1)],
        initial_equity=10_000,
        execution_config=config(contract_size=10),
    )
    assert results[0].intent.risk_budget == 100
    assert results[0].entry_fill is not None
    assert results[0].entry_fill.quantity == 1
    assert results[0].intent.price_risk == 100
    assert results[0].status is ExecutionStatus.OPEN_AT_END_OF_INPUT


def test_risk_quantity_is_rounded_down_to_the_configured_step():
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0), quote("2026-01-01T00:00:01Z", 100, 101, 1)],
        [decision("2026-01-01T00:00:00Z", stop=90, target=120, quantity=0.1)],
        initial_equity=10_000,
        execution_config=config(contract_size=10),
    )
    result = results[0]
    assert result.entry_fill.quantity == 0.9
    assert result.intent.price_risk == 99
    assert result.intent.price_risk <= result.intent.risk_budget


def test_accepted_intent_records_immediate_pre_entry_equity_and_risk_budget():
    engine, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0), quote("2026-01-01T00:00:01Z", 100, 101, 1)],
        [decision("2026-01-01T00:00:00Z", stop=99, target=120, quantity=1)],
        initial_equity=10_000,
    )
    assert results[0].intent.account_equity_before_entry == 10_000
    assert results[0].intent.risk_budget == 100
    assert results[0].diagnostic_only
    assert engine.intents[0] == results[0].intent


def test_missing_contract_metadata_rejects_candidate():
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0)],
        [decision("2026-01-01T00:00:00Z")],
        execution_config=config(contract_size=None),
    )
    assert results[0].status is ExecutionStatus.NO_TRADE_INVALID_EXECUTION_CONTRACT
    assert results[0].intent is None


@pytest.mark.parametrize(
    "contract_override",
    [
        {"minimum_quantity": None},
        {"quantity_step": None},
        {"conversion": None},
    ],
    ids=["minimum-quantity", "quantity-step", "account-conversion"],
)
def test_missing_additional_contract_metadata_rejects_candidate(contract_override):
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0)],
        [decision("2026-01-01T00:00:00Z")],
        execution_config=config(**contract_override),
    )
    assert results[0].status is ExecutionStatus.NO_TRADE_INVALID_EXECUTION_CONTRACT
    assert results[0].intent is None


@pytest.mark.parametrize(
    "cost_override",
    [
        {"commission": None},
        {"commission": -0.1},
        {"slippage": -0.1},
        {"slippage": float("nan")},
    ],
    ids=["missing-commission", "negative-commission", "negative-slippage", "nonfinite-slippage"],
)
def test_invalid_cost_configuration_rejects_candidate(cost_override):
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0)],
        [decision("2026-01-01T00:00:00Z")],
        execution_config=config(**cost_override),
    )
    assert results[0].status is ExecutionStatus.NO_TRADE_INVALID_COST_CONFIGURATION
    assert results[0].intent is None


def test_nonzero_costs_do_not_reject_valid_price_risk_position():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", 100, 101, 0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
        ],
        [decision("2026-01-01T00:00:00Z", stop=90, target=120, quantity=1)],
        execution_config=config(commission=0.5, slippage=0.1),
    )
    result = results[0]
    assert result.status is ExecutionStatus.OPEN_AT_END_OF_INPUT
    assert result.intent.risk_budget == 100
    assert result.entry_fill.quantity == 9
    assert result.entry_fill.commission_cost == 4.5
    assert result.entry_fill.slippage_cost == pytest.approx(0.9)
    assert result.intent.price_risk == 99
    assert not result.diagnostic_only


def test_commission_and_slippage_are_excluded_from_price_risk_ceiling():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
        ],
        [decision("2026-01-01T00:00:00Z", stop=91, target=120, quantity=1)],
        initial_equity=10_000,
        execution_config=config(contract_size=10, commission=60, slippage=50),
    )
    result = results[0]
    assert result.status is ExecutionStatus.OPEN_AT_END_OF_INPUT
    assert result.intent.risk_budget == 100
    assert result.entry_fill.quantity == 1
    assert result.intent.price_risk == 100
    assert result.entry_fill.commission_cost + result.entry_fill.slippage_cost == 110
    assert not result.diagnostic_only


def test_nonzero_costs_are_deducted_from_realized_pnl():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", 100, 101, 0),
            quote("2026-01-01T00:00:01Z", 101, 102, 1),
            quote("2026-01-01T00:00:02Z", 110, 111, 2),
        ],
        [decision("2026-01-01T00:00:00Z", stop=92, target=110, quantity=1)],
        execution_config=config(contract_size=10, commission=0.5, slippage=0.1),
    )
    result = results[0]
    assert result.status is ExecutionStatus.CLOSED
    assert result.entry_fill.observed_price == 102
    assert result.exit_fill.side == "BID"
    assert result.exit_fill.observed_price == 110
    assert result.gross_realized_pnl == 80
    assert result.cost_breakdown == {
        "entry_commission": 0.5,
        "entry_slippage": 0.1,
        "exit_commission": 0.5,
        "exit_slippage": 0.1,
    }
    assert result.net_realized_pnl == pytest.approx(78.8)


def test_no_legal_quantity_meets_minimum_under_one_percent_risk_ceiling():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
        ],
        [decision("2026-01-01T00:00:00Z", stop=0.5, target=120)],
        execution_config=config(minimum_quantity=1.0, quantity_step=0.1),
    )
    assert results[0].status is ExecutionStatus.NO_TRADE_INVALID_QUANTITY
    assert results[0].entry_fill is None
    assert results[0].position is None
    assert results[0].intent.risk_budget == 100


def test_exact_step_quantity_at_risk_ceiling_is_accepted():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
        ],
        [decision("2026-01-01T00:00:00Z", stop=91, target=120)],
        execution_config=config(contract_size=10, quantity_step=0.1),
    )
    assert results[0].entry_fill.quantity == 1
    assert results[0].intent.price_risk == 100
    assert results[0].intent.price_risk <= results[0].intent.risk_budget


@pytest.mark.parametrize(
    ("direction", "stop", "target", "final_bid", "final_ask", "expected_side"),
    [
        (ExecutionDirection.LONG, 91, 120, 105, 106, "BID"),
        (ExecutionDirection.SHORT, 110, 90, 95, 96, "ASK"),
    ],
)
def test_eof_open_position_records_side_correct_unrealized_valuation(
    direction,
    stop,
    target,
    final_bid,
    final_ask,
    expected_side,
):
    engine, results = execute(
        [
            quote("2026-01-01T00:00:00Z", 100, 101, 0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
            quote(
                "2026-01-01T00:00:02Z",
                final_bid,
                final_ask,
                2,
                source_row=3,
            ),
        ],
        [decision("2026-01-01T00:00:00Z", direction, stop=stop, target=target)],
        execution_config=config(contract_size=10),
    )
    result = results[0]
    valuation = result.mark_to_market
    assert result.status is ExecutionStatus.OPEN_AT_END_OF_INPUT
    assert result.position.state is PositionState.OPEN
    assert result.position.exit_fill is None
    assert result.exit_fill is None
    assert result.gross_realized_pnl is None
    assert result.net_realized_pnl is None
    assert valuation.side == expected_side
    assert valuation.price == (final_bid if expected_side == "BID" else final_ask)
    assert valuation.timestamp == timestamp("2026-01-01T00:00:02Z")
    assert valuation.source_order == 2
    assert valuation.provenance["source_row"] == 3
    assert valuation.unrealized_pnl == 40
    assert result.unrealized_pnl == 40
    assert result.gross_realized_pnl is None
    assert result.net_realized_pnl is None
    assert engine.valuations == [valuation]
    assert engine.ending_equity == engine.initial_equity


def test_risk_budget_uses_equity_after_prior_trade():
    first = decision("2026-01-01T00:00:00Z", stop=91, target=111, quantity=1)
    second = decision(
        "2026-01-01T00:00:02Z",
        stop=91.95,
        target=130,
        quantity=1,
        base="2026-01-01T00:00:01Z",
    )
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", 100, 101, 1),
            quote("2026-01-01T00:00:02Z", 111, 112, 2),
            quote("2026-01-01T00:00:03Z", 101, 102, 3),
        ],
        [first, second],
        initial_equity=10_000,
        execution_config=config(contract_size=10),
    )
    by_subject = {result.subject.subject_id: result for result in results}
    assert by_subject[first.subject.subject_id].gross_realized_pnl == 100
    second_result = by_subject[second.subject.subject_id]
    assert second_result.intent.account_equity_before_entry == 10_100
    assert second_result.intent.risk_budget == 101
    assert second_result.status is ExecutionStatus.OPEN_AT_END_OF_INPUT


def test_execution_derives_quantity_when_decision_quantity_is_missing():
    _, results = execute(
        [
            quote("2026-01-01T00:00:00Z", order=0),
            quote("2026-01-01T00:00:01Z", order=1),
        ],
        [decision("2026-01-01T00:00:00Z", quantity=None)],
    )
    assert results[0].entry_fill is not None
    assert results[0].entry_fill.quantity == 9


def test_non_market_order_is_rejected_without_approximation():
    _, results = execute(
        [quote("2026-01-01T00:00:00Z", order=0)],
        [decision("2026-01-01T00:00:00Z", order_type="LIMIT")],
    )
    assert results[0].status is ExecutionStatus.NO_TRADE_UNSUPPORTED_ORDER_TYPE
    assert not results[0].intent


def test_decision_availability_must_match_a_tick_timestamp():
    with pytest.raises(ValueError, match="no tick at its timestamp"):
        execute(
            [quote("2026-01-01T00:00:00Z", order=0)],
            [decision("2026-01-01T00:00:01Z")],
        )