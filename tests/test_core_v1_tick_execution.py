from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Mapping

import pandas as pd
import pytest

from src.core_v1_dealing_range import build_dealing_range_context
from src.core_v1_decision_bridge import CoreV1QuantityInput
from src.core_v1_evidence import SubjectVisibleInputs
from src.core_v1_pipeline import CoreV1PipelineResult
from src.core_v1_tick_execution import CoreV1TickExecution
from src.displacement import Direction
from src.ict2022_engine import SetupState
from src.market_structure import LiquiditySide
from src.replay_engine import IncrementalHistoricalReplay, ReplayConfig
from src.replay_subject import ReplaySubject
from src.sniper_setup import PrecisionState
from src.tick_aggregation import TickAggregator
from src.tick_execution import (
    ExecutionConfig,
    ExecutionEngine,
    ExecutionStatus,
    QuoteEvent,
)
from tests.test_core_v1_pipeline import pipeline_fixture


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def tick(value: str, order: int, bid: float = 100.0, ask: float = 100.5) -> QuoteEvent:
    return QuoteEvent(
        timestamp=ts(value),
        bid=bid,
        ask=ask,
        source_order=order,
        provenance={"source_file": "canonical.csv", "source_row": order + 1},
    )


def replay(*, base: str = "1min", limits=None):
    history_limits = limits or {"1min": 32}
    if base not in history_limits:
        history_limits[base] = 32
    return IncrementalHistoricalReplay(
        config=ReplayConfig(timeframe=base, source="CANONICAL_TEST"),
        history_limits=history_limits,
        max_pending_groups=16,
        max_pending_bars=128,
    )


def engine():
    return ExecutionEngine(
        execution_run_id="core-v1-integration-test",
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


def template_fixture():
    _, kwargs = pipeline_fixture()
    from tests.test_core_v1_pipeline import evaluate

    return kwargs, evaluate(kwargs)


def rebind_subject(value, old_subject, new_subject, memo=None):
    memo = {} if memo is None else memo
    if value is old_subject:
        return new_subject
    if isinstance(value, tuple):
        return tuple(rebind_subject(item, old_subject, new_subject, memo) for item in value)
    if isinstance(value, list):
        return [rebind_subject(item, old_subject, new_subject, memo) for item in value]
    if isinstance(value, Mapping):
        return {
            key: rebind_subject(item, old_subject, new_subject, memo)
            for key, item in value.items()
        }
    if not is_dataclass(value) or isinstance(value, type):
        return value
    if id(value) in memo:
        return memo[id(value)]
    updates = {}
    has_subject = getattr(value, "subject", None) is old_subject
    for item in fields(value):
        current = getattr(value, item.name)
        if item.name == "availability_timestamp" and has_subject:
            updates[item.name] = new_subject.availability_timestamp
        else:
            updates[item.name] = rebind_subject(
                current,
                old_subject,
                new_subject,
                memo,
            )
    rebound = replace(value, **updates)
    memo[id(value)] = rebound
    return rebound


def add_fixture_evidence(inputs, source_inputs):
    base = list(inputs.base_observations)
    context = list(inputs.context_observations)
    cutoff = inputs.subject.base_bar_timestamp
    availability = inputs.subject.availability_timestamp
    existing = [
        source
        for observation in base + context
        for source in observation.timing.source_observations
    ]
    for observation in (
        source_inputs.base_observations + source_inputs.context_observations
    ):
        if any(
            source.availability_timestamp > availability
            or (
                source.is_base_timeframe
                and source.base_bar_timestamp > cutoff
            )
            for source in observation.timing.source_observations
        ):
            continue
        if all(
            source not in existing
            for source in observation.timing.source_observations
        ):
            (base if observation.scope.value == "base" else context).append(observation)
            existing.extend(observation.timing.source_observations)
    return replace(
        inputs,
        base_observations=tuple(base),
        context_observations=tuple(context),
    )


def integration(
    *,
    tick_aggregators=None,
    replay_engine=None,
    execution_engine=None,
    pipeline_builder=None,
    evidence_enricher=None,
    quantity_provider=None,
    order_type="MARKET",
):
    return CoreV1TickExecution(
        aggregators=tick_aggregators or {"1min": TickAggregator("1min")},
        replay=replay_engine or replay(),
        execution=execution_engine or engine(),
        pipeline_builder=pipeline_builder,
        evidence_enricher=evidence_enricher,
        quantity_provider=quantity_provider,
        order_type=order_type,
    )


def delayed_aggregator(interval, release_at):
    class DelayedAggregator(TickAggregator):
        def __init__(self):
            super().__init__(interval)
            self.release_at = release_at
            self.buffered_rows = []

        def add_tick(self, timestamp, bid, ask):
            emitted = super().add_tick(timestamp, bid, ask)
            self.buffered_rows.extend(emitted)
            if pd.Timestamp(timestamp).tz_convert("UTC") != self.release_at:
                return []
            for row in self.buffered_rows:
                row["available_at"] = self.release_at
                row["availability_ts"] = self.release_at
            released = self.buffered_rows
            self.buffered_rows = []
            return released

    return DelayedAggregator()


def incomplete_builder(inputs):
    context = build_dealing_range_context(inputs)
    return CoreV1PipelineResult(
        subject=inputs.subject,
        availability_timestamp=inputs.availability_timestamp,
        state=PrecisionState.DEVELOPING,
        reason="No governed detector output supplied.",
        context=context,
    )


def test_conflicting_governed_directions_are_arbitrated_by_execution_engine():
    source_kwargs, template = template_fixture()
    template_subject = template.subject
    release_at = ts("2026-01-01T10:09:05Z")

    def build(inputs):
        rebound = rebind_subject(template, template_subject, inputs.subject)
        short_displacement = replace(
            rebound.displacement,
            observation=replace(
                rebound.displacement.observation,
                direction=Direction.BEARISH,
            ),
        )
        short_break = replace(
            rebound.mss.structural_break,
            direction=Direction.BEARISH,
        )
        short_confirmation = replace(
            rebound.mss.confirmation,
            structural_break=short_break,
        )
        short_mss = replace(
            rebound.mss,
            displacement=short_displacement,
            structural_break=short_break,
            confirmation=short_confirmation,
        )
        short_entry = replace(
            rebound.retracement,
            displacement=short_displacement,
            mss=short_mss,
        )
        short_level = replace(
            rebound.target.liquidity_level,
            side=LiquiditySide.SSL,
            price=95.0,
        )
        short_target = replace(rebound.target, liquidity_level=short_level)
        short_invalidation = replace(rebound.invalidation, price=105.0)
        short_hypothesis = replace(
            rebound.hypothesis,
            direction=Direction.BEARISH,
            planned_entry=short_entry,
            invalidation=short_invalidation,
            target=short_target,
        )
        return replace(
            rebound,
            displacement=short_displacement,
            mss=short_mss,
            retracement=short_entry,
            invalidation=short_invalidation,
            target=short_target,
            opposing_liquidity_levels=(short_level,),
            hypothesis=short_hypothesis,
        )

    instance = integration(
        tick_aggregators={"1min": delayed_aggregator("1min", release_at)},
        pipeline_builder=lambda inputs: (
            rebind_subject(template, template_subject, inputs.subject)
            if inputs.subject.base_bar_timestamp.minute == 7
            else build(inputs)
        ),
        evidence_enricher=lambda inputs: add_fixture_evidence(
            inputs,
            source_kwargs["inputs"],
        ),
        quantity_provider=lambda _: CoreV1QuantityInput(0.1, "explicit-q"),
    )
    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0),
            tick("2026-01-01T10:08:05Z", 1),
            tick("2026-01-01T10:09:05Z", 2),
        ]
    )

    decisions = report.decision_sets[0].decisions
    assert len(decisions) == 2
    assert {item.direction.value for item in decisions} == {"LONG", "SHORT"}
    assert all(
        item.status is ExecutionStatus.NO_TRADE_CONFLICTING_DIRECTIONS
        for item in report.execution_results
    )
    assert report.execution_results[0].entry_fill is None


def test_same_direction_candidates_delegate_latest_base_bar_selection():
    source_kwargs, template = template_fixture()
    template_subject = template.subject
    release_at = ts("2026-01-01T10:09:05Z")
    instance = integration(
        tick_aggregators={"1min": delayed_aggregator("1min", release_at)},
        pipeline_builder=lambda inputs: rebind_subject(
            template,
            template_subject,
            inputs.subject,
        ),
        evidence_enricher=lambda inputs: add_fixture_evidence(
            inputs,
            source_kwargs["inputs"],
        ),
        quantity_provider=lambda _: CoreV1QuantityInput(0.1, "explicit-q"),
    )

    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0),
            tick("2026-01-01T10:08:05Z", 1),
            tick("2026-01-01T10:09:05Z", 2),
        ]
    )

    decisions = report.decision_sets[0].decisions
    assert len(decisions) == 2
    assert all(item.direction.value == "LONG" for item in decisions)
    assert len(report.execution_results) == 2
    selected = max(decisions, key=lambda item: item.subject.base_bar_timestamp)
    selected_result = next(
        item for item in report.execution_results
        if item.subject is selected.subject
    )
    other_result = next(
        item for item in report.execution_results
        if item.subject is not selected.subject
    )
    assert selected_result.status is ExecutionStatus.UNFILLED_AT_END_OF_INPUT
    assert other_result.status is ExecutionStatus.VALID_DECISION_NOT_SELECTED


def test_canonical_tick_to_replay_subject_to_pipeline_and_decision_record():
    source_kwargs, template = template_fixture()
    template_subject = template.subject

    def enrich(inputs):
        return add_fixture_evidence(inputs, source_kwargs["inputs"])

    def build(inputs):
        return rebind_subject(
            template,
            template_subject,
            inputs.subject,
        )

    instance = integration(
        pipeline_builder=build,
        evidence_enricher=enrich,
        quantity_provider=lambda _: CoreV1QuantityInput(
            0.1,
            "test-approved-quantity",
            {"source": "integration"},
        ),
    )
    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0, 100.0, 100.5),
            tick("2026-01-01T10:08:00Z", 1, 101.0, 101.5),
        ]
    )

    assert report.replay_summary.subjects_processed == 1
    assert len(report.subject_executions) == 1
    execution = report.subject_executions[0]
    assert execution.subject is execution.pipeline_result.subject
    assert execution.subject is execution.bridge_result.decision_record.subject
    assert execution.pipeline_result.state is PrecisionState.VALID
    assert execution.bridge_result.decision_record.actionable
    assert execution.bridge_result.decision_record.decision_time == ts("2026-01-01T10:08:00Z")
    base_observation = execution.visible_inputs.base_observations[0]
    source = base_observation.timing.source_observations[0]
    assert source.provenance["canonical_ticks"][0]["source_order"] == 0
    assert len(source.provenance["canonical_ticks"]) == 1


def test_reconstructed_equal_subject_id_is_accepted_end_to_end():
    source_kwargs, template = template_fixture()

    def build(inputs):
        reconstructed = ReplaySubject.from_dict(inputs.subject.to_dict())
        return rebind_subject(template, template.subject, reconstructed)

    instance = integration(
        pipeline_builder=build,
        evidence_enricher=lambda inputs: add_fixture_evidence(
            inputs,
            source_kwargs["inputs"],
        ),
        quantity_provider=lambda _: CoreV1QuantityInput(0.1, "explicit-q"),
    )
    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0),
            tick("2026-01-01T10:08:00Z", 1),
        ]
    )

    assert report.replay_summary.subjects_processed == 1
    execution = report.subject_executions[0]
    canonical_id = execution.subject.subject_id
    assert execution.pipeline_result.subject is not execution.subject
    assert execution.pipeline_result.subject.subject_id == canonical_id
    assert execution.bridge_result.decision_record.subject.subject_id == canonical_id
    assert report.execution_results[0].subject.subject_id == canonical_id


def test_canonical_tick_provenance_with_nested_frame_fails_closed():
    with pytest.raises(TypeError, match="frames or series"):
        QuoteEvent(
            timestamp=ts("2026-01-01T10:07:05Z"),
            bid=100.0,
            ask=100.5,
            source_order=0,
            provenance={"metadata": {"frame": pd.DataFrame({"close": [100.0]})}},
        )


def test_end_to_end_fill_uses_later_actual_ask_not_planned_entry_and_exits_on_bid():
    source_kwargs, template = template_fixture()

    def build(inputs):
        return rebind_subject(template, template.subject, inputs.subject)

    instance = integration(
        pipeline_builder=build,
        evidence_enricher=lambda inputs: add_fixture_evidence(
            inputs,
            source_kwargs["inputs"],
        ),
        quantity_provider=lambda _: CoreV1QuantityInput(0.1, "explicit-q"),
    )
    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0, 100.0, 100.5),
            tick("2026-01-01T10:08:00Z", 1, 101.0, 101.5),
            tick("2026-01-01T10:08:00Z", 2, 110.0, 111.0),
            tick("2026-01-01T10:08:01Z", 3, 104.0, 104.5),
            tick("2026-01-01T10:08:02Z", 4, 110.25, 110.5),
        ]
    )

    assert len(report.execution_results) == 1
    result = report.execution_results[0]
    assert result.status is ExecutionStatus.CLOSED
    assert result.entry_fill.side == "ASK"
    assert result.entry_fill.timestamp == ts("2026-01-01T10:08:01Z")
    assert result.entry_fill.observed_price == 104.5
    assert result.entry_fill.observed_price != report.subject_executions[0].pipeline_result.hypothesis.planned_entry_price
    assert result.exit_fill.side == "BID"
    assert result.exit_fill.timestamp == ts("2026-01-01T10:08:02Z")
    assert result.exit_fill.observed_price == 110.25


def test_decision_tick_and_same_timestamp_later_source_order_tick_cannot_fill():
    source_kwargs, template = template_fixture()
    instance = integration(
        pipeline_builder=lambda inputs: rebind_subject(
            template,
            template.subject,
            inputs.subject,
        ),
        evidence_enricher=lambda inputs: add_fixture_evidence(
            inputs,
            source_kwargs["inputs"],
        ),
        quantity_provider=lambda _: CoreV1QuantityInput(0.1, "explicit-q"),
    )
    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0),
            tick("2026-01-01T10:08:00Z", 1),
            tick("2026-01-01T10:08:00Z", 2, 103.0, 104.0),
        ]
    )

    assert len(report.execution_results) == 1
    assert report.execution_results[0].status is ExecutionStatus.UNFILLED_AT_END_OF_INPUT
    assert report.execution_results[0].entry_fill is None


def test_missing_quantity_remains_non_actionable_and_never_creates_intent():
    source_kwargs, template = template_fixture()
    instance = integration(
        pipeline_builder=lambda inputs: rebind_subject(
            template,
            template.subject,
            inputs.subject,
        ),
        evidence_enricher=lambda inputs: add_fixture_evidence(
            inputs,
            source_kwargs["inputs"],
        ),
        quantity_provider=None,
    )
    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0),
            tick("2026-01-01T10:08:00Z", 1),
            tick("2026-01-01T10:08:01Z", 2, 104.0, 104.5),
        ]
    )

    assert not report.subject_executions[0].bridge_result.decision_record.actionable
    assert "MISSING_EXPLICIT_QUANTITY" in report.subject_executions[0].bridge_result.decision_record.reason
    assert len(report.execution_results) == 1
    assert report.execution_results[0].status is ExecutionStatus.NO_TRADE_NOT_ACTIONABLE
    assert report.execution_results[0].entry_fill is None


def test_multitimeframe_availability_groups_drain_subjects_before_submit():
    evaluated_subjects = []
    submitted_sets = []
    execution = engine()

    class ObservedExecution(ExecutionEngine):
        def submit_decision_set(self, decision_set):
            already_submitted = sum(
                len(item.decisions) for item in submitted_sets
            )
            assert len(evaluated_subjects) - already_submitted == len(
                decision_set.decisions
            )
            submitted_sets.append(decision_set)
            super().submit_decision_set(decision_set)

    execution = ObservedExecution(
        execution_run_id="delayed-group",
        initial_equity=10_000.0,
        config=execution.config,
    )

    def build(inputs):
        evaluated_subjects.append(inputs.subject)
        return incomplete_builder(inputs)

    instance = integration(
        tick_aggregators={
            "1min": TickAggregator("1min"),
            "3min": TickAggregator("3min"),
        },
        replay_engine=replay(limits={"1min": 16, "3min": 8}),
        execution_engine=execution,
        pipeline_builder=build,
    )
    report = instance.run(
        [
            tick("2026-01-01T10:00:05Z", 0),
            tick("2026-01-01T10:01:05Z", 1),
            tick("2026-01-01T10:02:05Z", 2),
            tick("2026-01-01T10:04:00Z", 3),
        ]
    )

    assert report.replay_summary.subjects_processed == 3
    assert len(report.decision_sets) == len(submitted_sets) == 3
    assert all(len(decision_set.decisions) == 1 for decision_set in report.decision_sets)
    assert len(set(item.subject_id for item in evaluated_subjects)) == 3
    assert len({item.base_bar_timestamp for item in evaluated_subjects}) == 3
    assert all(not decision.actionable for group in report.decision_sets for decision in group.decisions)
    assert report.decision_sets[-1].available_at == ts("2026-01-01T10:04:00Z")
    assert len(report.decision_sets[-1].replay_group.bars_by_timeframe["1min"]) == 1
    assert len(report.decision_sets[-1].replay_group.bars_by_timeframe["3min"]) == 1
    assert len(execution.decision_sets) == 3


def test_active_position_is_processed_before_same_availability_candidate_submission():
    source_kwargs, template = template_fixture()
    build_observations = []

    class ObservedExecution(ExecutionEngine):
        def process_tick(self, current):
            super().process_tick(current)
            if current.timestamp == ts("2026-01-01T10:09:00Z"):
                assert len(self.fills) == 1
                assert self.fills[0].timestamp == current.timestamp

    execution = ObservedExecution(
        execution_run_id="pending-before-decision",
        initial_equity=10_000.0,
        config=engine().config,
    )

    def build(inputs):
        if inputs.subject.availability_timestamp == ts("2026-01-01T10:09:00Z"):
            assert len(execution.fills) == 1
        build_observations.append(inputs.subject)
        return rebind_subject(template, template.subject, inputs.subject)

    instance = integration(
        execution_engine=execution,
        pipeline_builder=build,
        evidence_enricher=lambda inputs: add_fixture_evidence(
            inputs,
            source_kwargs["inputs"],
        ),
        quantity_provider=lambda _: CoreV1QuantityInput(0.1, "explicit-q"),
    )
    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0),
            tick("2026-01-01T10:08:00Z", 1),
            tick("2026-01-01T10:09:00Z", 2, 104.0, 104.5),
        ]
    )

    assert len(build_observations) == 2
    assert len(report.execution_results) == 2
    assert report.execution_results[0].entry_fill.timestamp == ts("2026-01-01T10:09:00Z")
    assert report.execution_results[1].status is ExecutionStatus.NO_TRADE_ACTIVE_POSITION
    assert len(execution.positions) == 1


def test_future_ticks_cannot_alter_earlier_point_in_time_evidence():
    def run(with_future):
        observed = []
        instance = integration(
            pipeline_builder=lambda inputs: (
                observed.append(inputs) or incomplete_builder(inputs)
            ),
        )
        ticks = [
            tick("2026-01-01T10:00:05Z", 0),
            tick("2026-01-01T10:01:05Z", 1),
            tick("2026-01-01T10:02:05Z", 2),
            tick("2026-01-01T10:03:00Z", 3),
        ]
        if with_future:
            ticks.append(tick("2026-01-01T10:03:01Z", 4))
        report = instance.run(ticks)
        return observed[0], report

    earlier, without_future = run(False)
    sentinel_version, with_future = run(True)

    assert earlier.subject == sentinel_version.subject
    assert earlier.base_observations == sentinel_version.base_observations
    assert earlier.context_observations == sentinel_version.context_observations
    assert with_future.replay_summary.subjects_processed == without_future.replay_summary.subjects_processed
    assert all(
        source.availability_timestamp <= earlier.subject.availability_timestamp
        for observation in (
            earlier.base_observations + earlier.context_observations
        )
        for source in observation.timing.source_observations
    )


def test_execution_handoff_preserves_decision_and_provenance_chain():
    source_kwargs, template = template_fixture()
    instance = integration(
        pipeline_builder=lambda inputs: rebind_subject(
            template,
            template.subject,
            inputs.subject,
        ),
        evidence_enricher=lambda inputs: add_fixture_evidence(
            inputs,
            source_kwargs["inputs"],
        ),
        quantity_provider=lambda _: CoreV1QuantityInput(
            0.1,
            "governed-quantity",
            {"decision": "approved"},
        ),
    )
    report = instance.run(
        [
            tick("2026-01-01T10:07:05Z", 0),
            tick("2026-01-01T10:08:00Z", 1),
        ]
    )

    bridge = report.subject_executions[0].bridge_result
    decision = report.decision_sets[0].decisions[0]
    assert decision.evidence_reference is bridge.evidence
    assert bridge.evidence.pipeline_result is report.subject_executions[0].pipeline_result
    assert bridge.evidence.quantity_input.source_id == "governed-quantity"
    assert report.decision_sets[0].replay_group.available_at == decision.decision_time
    assert decision.subject is report.subject_executions[0].subject
