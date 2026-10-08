from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.core_v1_decision_bridge import (
    CoreV1QuantityInput,
    build_core_v1_decision_bridge,
)
from src.ict2022_engine import SetupState
from src.sniper_setup import PrecisionState
from src.tick_execution import DecisionRecord, ExecutionDirection, ExecutionEngine
from src.replay_subject import ReplaySubject
from tests.test_core_v1_pipeline import evaluate, pipeline_fixture


def valid_pipeline():
    _, kwargs = pipeline_fixture()
    return evaluate(kwargs)


def developing_pipeline():
    _, kwargs = pipeline_fixture()
    return evaluate(kwargs, liquidity=None)


def invalid_pipeline():
    fixture, kwargs = pipeline_fixture()
    invalidation, target = kwargs["invalidation"], kwargs["target"]
    wrong_level = replace(
        target.liquidity_level,
        price=101.0,
    )
    wrong_target = replace(target, liquidity_level=wrong_level)
    # Preserve the original timing and source lineage while violating long target geometry.
    kwargs["inputs"] = replace(
        kwargs["inputs"],
        context_observations=kwargs["inputs"].context_observations,
    )
    kwargs["target"] = wrong_target
    kwargs["opposing_liquidity_levels"] = (wrong_level,)
    result = evaluate(kwargs)
    assert result.state is PrecisionState.INVALID
    assert result.hypothesis is None
    return result


def no_trade_pipeline():
    result = developing_pipeline()
    return replace(
        result,
        state=PrecisionState.NO_TRADE,
        reason="Explicit no-trade test pipeline result.",
    )


def quantity(value=0.75):
    return CoreV1QuantityInput(
        quantity=value,
        source_id="risk-service-approved-quantity",
        provenance={"provider": "governed-downstream", "approval_id": "q-17"},
    )


def test_valid_pipeline_and_explicit_quantity_create_actionable_market_decision():
    pipeline = valid_pipeline()
    result = build_core_v1_decision_bridge(pipeline, quantity=quantity())

    assert result.adapter_decision.state is SetupState.ACTIVE_TRADE
    assert isinstance(result.decision_record, DecisionRecord)
    assert result.decision_record.actionable
    assert result.decision_record.order_type == "MARKET"
    assert result.decision_record.direction is ExecutionDirection.LONG
    assert result.decision_record.quantity == 0.75
    assert result.decision_record.stop_loss == pipeline.hypothesis.invalidation_price
    assert result.decision_record.take_profit == pipeline.hypothesis.target_price
    assert result.decision_record.evidence_reference is result.evidence


@pytest.mark.parametrize(
    ("pipeline_factory", "expected_adapter_state"),
    [
        (developing_pipeline, SetupState.DEVELOPING),
        (invalid_pipeline, SetupState.NO_TRADE),
        (no_trade_pipeline, SetupState.NO_TRADE),
    ],
)
def test_non_valid_pipeline_states_never_become_actionable(
    pipeline_factory,
    expected_adapter_state,
):
    result = build_core_v1_decision_bridge(
        pipeline_factory(),
        quantity=quantity(),
    )

    assert not result.decision_record.actionable
    assert result.decision_record.quantity is None
    assert result.adapter_decision.state is expected_adapter_state
    assert result.decision_record.strategy_state != PrecisionState.VALID.value


def test_valid_hypothesis_without_quantity_is_no_trade_and_does_not_infer_quantity():
    pipeline = valid_pipeline()
    result = build_core_v1_decision_bridge(pipeline)

    assert result.adapter_decision.state is SetupState.NO_TRADE
    assert "MISSING_EXPLICIT_QUANTITY" in result.adapter_decision.reason
    assert not result.decision_record.actionable
    assert result.decision_record.quantity is None
    assert result.evidence.quantity_input is None
    assert not hasattr(result.decision_record, "risk_budget")


def test_explicit_quantity_is_preserved_exactly_with_its_source():
    pipeline = valid_pipeline()
    explicit = quantity(0.125)
    result = build_core_v1_decision_bridge(pipeline, quantity=explicit)

    assert result.decision_record.quantity is explicit.quantity
    assert result.evidence.quantity_input is explicit
    assert result.evidence.quantity_input.source_id == explicit.source_id
    assert result.evidence.quantity_input.provenance == explicit.provenance


def test_quantity_provenance_uses_bounded_immutable_snapshot():
    original = {"provider": "governed", "metadata": {"approvals": ["q-17"]}}
    explicit = CoreV1QuantityInput(0.75, "risk-service", original)
    original["metadata"]["approvals"].append("caller-change")
    original["metadata"]["frame"] = pd.DataFrame({"value": [1]})
    original["new"] = "caller-change"

    assert explicit.provenance["metadata"]["approvals"] == ("q-17",)
    assert "frame" not in explicit.provenance["metadata"]
    assert "new" not in explicit.provenance
    with pytest.raises(TypeError):
        explicit.provenance["provider"] = "mutated"


@pytest.mark.parametrize(
    "provenance",
    [
        {"metadata": {"frame": pd.DataFrame({"value": [1]})}},
        {"x" * 4097: "value"},
        {"metadata": "x" * 4097},
        {"metadata": {1, 2}},
    ],
)
def test_quantity_provenance_rejects_unbounded_values(provenance):
    with pytest.raises(TypeError):
        CoreV1QuantityInput(0.75, "risk-service", provenance)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1.0"])
def test_invalid_quantity_input_is_rejected(value):
    with pytest.raises(ValueError, match="Explicit quantity"):
        quantity(value)


def test_untyped_quantity_is_rejected_instead_of_being_inferred_or_cast():
    with pytest.raises(TypeError, match="CoreV1QuantityInput"):
        build_core_v1_decision_bridge(valid_pipeline(), quantity=0.5)


def test_invalid_and_missing_hypothesis_direction_fail_closed():
    pipeline = valid_pipeline()
    hypothesis = pipeline.hypothesis
    object.__setattr__(hypothesis, "direction", None)
    missing = build_core_v1_decision_bridge(pipeline, quantity=quantity())
    assert not missing.decision_record.actionable
    assert missing.decision_record.direction is None
    assert missing.adapter_decision.state is SetupState.NO_TRADE

    object.__setattr__(hypothesis, "direction", "SIDEWAYS")
    invalid = build_core_v1_decision_bridge(pipeline, quantity=quantity())
    assert not invalid.decision_record.actionable
    assert invalid.decision_record.direction is None
    assert invalid.adapter_decision.state is SetupState.NO_TRADE


def test_unsupported_order_type_is_no_trade_without_market_conversion():
    result = build_core_v1_decision_bridge(
        valid_pipeline(),
        quantity=quantity(),
        order_type="LIMIT",
    )

    assert not result.decision_record.actionable
    assert result.adapter_decision.state is SetupState.NO_TRADE
    assert "UNSUPPORTED_ORDER_TYPE" in result.decision_record.reason
    assert result.decision_record.quantity is None
    assert result.decision_record.order_type == "LIMIT"


def test_market_order_is_supported():
    result = build_core_v1_decision_bridge(
        valid_pipeline(),
        quantity=quantity(),
        order_type="market",
    )
    assert result.decision_record.actionable
    assert result.decision_record.order_type == "MARKET"


def test_planned_entry_is_strategy_context_not_execution_or_fill_price():
    pipeline = valid_pipeline()
    result = build_core_v1_decision_bridge(pipeline, quantity=quantity())

    assert result.adapter_decision.entry_price == pipeline.hypothesis.planned_entry_price
    assert result.evidence.planned_entry_price == pipeline.hypothesis.planned_entry_price
    assert result.evidence.execution_price is None
    assert not hasattr(result.decision_record, "entry_price")
    assert not hasattr(result.decision_record, "fill_price")
    assert not hasattr(result.decision_record, "execution_intent")
    assert not hasattr(result.decision_record, "fill")


def test_invalidation_target_and_subject_identity_are_preserved():
    pipeline = valid_pipeline()
    result = build_core_v1_decision_bridge(pipeline, quantity=quantity())

    assert result.decision_record.subject is pipeline.subject
    assert result.adapter_decision.replay_subject is pipeline.subject
    assert result.evidence.subject is pipeline.subject
    assert result.evidence.pipeline_result is pipeline
    assert result.evidence.hypothesis is pipeline.hypothesis
    assert result.decision_record.stop_loss == pipeline.hypothesis.invalidation.price
    assert result.decision_record.take_profit == pipeline.hypothesis.target.price
    assert result.evidence.planned_entry_price == pipeline.retracement.planned_entry_price
    assert result.decision_record.decision_time == pipeline.subject.availability_timestamp
    assert result.decision_record.decision_time != pipeline.subject.base_bar_timestamp


def test_equal_reconstructed_subject_id_traverses_pipeline_bridge_and_decision():
    pipeline = valid_pipeline()
    canonical_subject = pipeline.subject
    reconstructed_subject = ReplaySubject.from_dict(canonical_subject.to_dict())
    assert reconstructed_subject is not canonical_subject
    assert reconstructed_subject.subject_id == canonical_subject.subject_id

    rebound_pipeline = replace(pipeline, subject=reconstructed_subject)
    result = build_core_v1_decision_bridge(
        rebound_pipeline,
        quantity=quantity(),
    )

    assert result.evidence.subject.subject_id == canonical_subject.subject_id
    assert result.decision_record.subject.subject_id == canonical_subject.subject_id
    assert result.adapter_decision.replay_subject.subject_id == canonical_subject.subject_id


def test_different_subject_id_is_rejected_by_pipeline_and_bridge():
    pipeline = valid_pipeline()
    different_subject = ReplaySubject(
        pipeline.subject.availability_timestamp,
        pipeline.subject.base_bar_timestamp - pd.Timedelta(minutes=1),
    )

    with pytest.raises(ValueError, match="ReplaySubject"):
        replace(pipeline, subject=different_subject)


def test_provenance_trace_includes_all_governed_stages_and_quantity_source():
    pipeline = valid_pipeline()
    explicit = quantity()
    result = build_core_v1_decision_bridge(pipeline, quantity=explicit)
    trace = result.evidence.pipeline_result.provenance

    for stage in (
        "liquidity_level",
        "liquidity_event",
        "liquidity_reaction",
        "displacement",
        "structural_point",
        "structural_break",
        "mss_confirmation",
        "reactionary_zone",
        "planned_entry",
        "invalidation",
        "opposing_liquidity_target",
        "target_liquidity_level",
    ):
        assert trace[stage]
    assert result.evidence.quantity_input.source_id == "risk-service-approved-quantity"
    assert result.evidence.pipeline_result is pipeline


def test_bridge_rejects_actionable_hypothesis_with_injected_extra_provenance():
    pipeline = valid_pipeline()
    hypothesis = pipeline.hypothesis
    object.__setattr__(
        hypothesis,
        "provenance",
        {**hypothesis.provenance, "unreferenced": ()},
    )

    with pytest.raises(ValueError, match="hypothesis provenance is incomplete or inconsistent"):
        build_core_v1_decision_bridge(pipeline, quantity=quantity())


def test_bridge_rejects_pipeline_with_injected_extra_provenance():
    pipeline = valid_pipeline()
    object.__setattr__(
        pipeline,
        "provenance",
        {**pipeline.provenance, "unreferenced": ()},
    )

    with pytest.raises(ValueError, match="pipeline provenance is incomplete or inconsistent"):
        build_core_v1_decision_bridge(pipeline, quantity=quantity())


def test_pipeline_timestamp_mismatch_is_rejected():
    pipeline = valid_pipeline()
    object.__setattr__(
        pipeline,
        "availability_timestamp",
        pipeline.subject.base_bar_timestamp,
    )
    with pytest.raises(ValueError, match="Pipeline decision time"):
        build_core_v1_decision_bridge(pipeline, quantity=quantity())


def test_full_frame_is_not_an_accepted_input():
    with pytest.raises(TypeError, match="CoreV1PipelineResult"):
        build_core_v1_decision_bridge(pd.DataFrame({"close": [1.0]}))


def test_bridge_does_not_invoke_execution_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("bridge must not instantiate or call ExecutionEngine")

    monkeypatch.setattr(ExecutionEngine, "__init__", forbidden)
    result = build_core_v1_decision_bridge(valid_pipeline(), quantity=quantity())
    assert result.decision_record.actionable


def test_output_does_not_reuse_subject_as_execution_identity_or_calculate_risk():
    pipeline = valid_pipeline()
    result = build_core_v1_decision_bridge(pipeline, quantity=quantity())

    assert result.decision_record.subject is pipeline.subject
    assert not hasattr(result.decision_record, "order_id")
    assert not hasattr(result.decision_record, "execution_id")
    assert not hasattr(result.decision_record, "risk")
    assert not hasattr(result.decision_record, "risk_amount")
    assert not hasattr(result.decision_record, "risk_budget")


def test_bridge_output_is_deterministic():
    pipeline = valid_pipeline()
    explicit = quantity()

    first = build_core_v1_decision_bridge(pipeline, quantity=explicit)
    second = build_core_v1_decision_bridge(pipeline, quantity=explicit)

    assert first.adapter_decision == second.adapter_decision
    assert first.decision_record == second.decision_record
