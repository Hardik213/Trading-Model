from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

from .core_v1_hypothesis import CoreV1TradeHypothesis, _build_provenance
from .core_v1_pipeline import CoreV1PipelineResult, _derive_pipeline_metadata
from .evidence_timing import validate_bounded_payload
from .ict2022_engine import SetupState
from .mss import Direction as AdapterDirection
from .sniper_setup import PrecisionState
from .tick_execution import DecisionRecord, ExecutionDirection
from .strategy_adapter import AdapterDecision
from .replay_subject import ReplaySubject, same_subject_id


@dataclass(frozen=True)
class CoreV1QuantityInput:
    """Explicit downstream quantity input with caller-supplied provenance."""

    quantity: float
    source_id: str
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, (int, float))
            or not isfinite(float(self.quantity))
            or self.quantity <= 0
        ):
            raise ValueError("Explicit quantity must be finite and greater than zero.")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("Explicit quantity requires a non-empty source_id.")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("Quantity provenance must be a mapping.")
        provenance = validate_bounded_payload(
            self.provenance,
            "Quantity provenance",
        )
        if not isinstance(provenance, Mapping):
            raise TypeError("Quantity provenance must be a mapping.")
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True)
class CoreV1DecisionEvidence:
    """Trace payload distinguishing strategy evidence from execution inputs."""

    subject: ReplaySubject
    pipeline_result: CoreV1PipelineResult
    hypothesis: CoreV1TradeHypothesis | None
    adapter_decision: AdapterDecision
    quantity_input: CoreV1QuantityInput | None
    decision_time: object
    planned_entry_price: float | None
    execution_price: None = None

    def __post_init__(self) -> None:
        if not same_subject_id(self.pipeline_result.subject, self.subject):
            raise ValueError("Evidence must preserve the pipeline ReplaySubject.")
        if (
            not isinstance(self.adapter_decision.replay_subject, ReplaySubject)
            or not same_subject_id(
                self.adapter_decision.replay_subject,
                self.subject,
            )
        ):
            raise ValueError("Adapter decision must preserve the pipeline ReplaySubject.")
        if self.hypothesis is not None and not same_subject_id(
            self.hypothesis.subject,
            self.subject,
        ):
            raise ValueError("Hypothesis evidence must preserve the pipeline ReplaySubject.")
        if self.quantity_input is not None and not isinstance(
            self.quantity_input,
            CoreV1QuantityInput,
        ):
            raise TypeError("quantity_input must be CoreV1QuantityInput or None.")
        decision_time = _utc_timestamp(self.decision_time)
        if decision_time != self.subject.availability_timestamp:
            raise ValueError("Decision time must exactly match subject availability.")
        object.__setattr__(self, "decision_time", decision_time)
        if self.execution_price is not None:
            raise ValueError("Strategy bridge cannot specify an execution price.")


@dataclass(frozen=True)
class CoreV1DecisionBridgeResult:
    """Adapter and execution-interface outputs plus their linked evidence."""

    adapter_decision: AdapterDecision
    decision_record: DecisionRecord
    evidence: CoreV1DecisionEvidence

    def __post_init__(self) -> None:
        if self.decision_record.evidence_reference is not self.evidence:
            raise ValueError("DecisionRecord must reference this bridge evidence.")
        if not same_subject_id(self.decision_record.subject, self.evidence.subject):
            raise ValueError("DecisionRecord must preserve ReplaySubject identity.")
        if self.adapter_decision is not self.evidence.adapter_decision:
            raise ValueError("Bridge result must preserve the adapter decision identity.")


def build_core_v1_decision_bridge(
    pipeline_result: CoreV1PipelineResult,
    *,
    quantity: CoreV1QuantityInput | None = None,
    order_type: str = "MARKET",
) -> CoreV1DecisionBridgeResult:
    """Bridge a Core v1 result to adapter and execution decision schemas.

    No quote, execution engine, quantity calculation, or risk calculation is
    performed here. The planned entry is retained only on AdapterDecision and
    the evidence payload; DecisionRecord has no executable entry-price field.
    """
    if not isinstance(pipeline_result, CoreV1PipelineResult):
        raise TypeError("pipeline_result must be a CoreV1PipelineResult.")
    subject = pipeline_result.subject
    decision_time = _utc_timestamp(pipeline_result.availability_timestamp)
    if decision_time != subject.availability_timestamp:
        raise ValueError("Pipeline decision time must equal subject availability.")
    if pipeline_result.state is PrecisionState.VALID:
        hypothesis = pipeline_result.hypothesis
        if (
            hypothesis is None
            or hypothesis.state is not PrecisionState.VALID
            or not isinstance(hypothesis, CoreV1TradeHypothesis)
        ):
            raise ValueError("VALID pipeline result must contain a complete valid hypothesis.")
        if hypothesis.availability_timestamp != decision_time:
            raise ValueError("Hypothesis decision timestamp must equal subject availability.")
        if not same_subject_id(hypothesis.subject, subject):
            raise ValueError("Hypothesis must preserve exact ReplaySubject identity.")
        _validate_actionable_trace(pipeline_result, hypothesis)
    else:
        hypothesis = pipeline_result.hypothesis
        if hypothesis is not None:
            raise ValueError("Non-VALID pipeline result cannot carry a hypothesis.")

    if quantity is not None and not isinstance(quantity, CoreV1QuantityInput):
        raise TypeError("quantity must be explicit CoreV1QuantityInput or None.")
    normalized_order_type = (
        order_type.upper()
        if isinstance(order_type, str)
        else ""
    )
    decision_order_type = normalized_order_type or str(order_type).upper()
    direction = _adapter_direction(hypothesis) if hypothesis is not None else None
    execution_direction = _execution_direction(hypothesis) if hypothesis is not None else None

    reason: str
    if normalized_order_type != "MARKET":
        state = SetupState.NO_TRADE
        reason = f"NO_TRADE_UNSUPPORTED_ORDER_TYPE: {order_type!r} is not supported."
    elif pipeline_result.state is not PrecisionState.VALID:
        state = (
            SetupState.DEVELOPING
            if pipeline_result.state is PrecisionState.DEVELOPING
            else SetupState.NO_TRADE
        )
        reason = (
            f"NO_TRADE_NON_ACTIONABLE: Core v1 pipeline state is "
            f"{pipeline_result.state.value}."
        )
    elif direction is None or execution_direction is None:
        state = SetupState.NO_TRADE
        reason = "NO_TRADE_MISSING_OR_INVALID_DIRECTION."
    elif quantity is None:
        state = SetupState.NO_TRADE
        reason = "NO_TRADE_MISSING_EXPLICIT_QUANTITY."
    else:
        state = SetupState.ACTIVE_TRADE
        reason = "Governed Core v1 hypothesis and explicit quantity supplied."

    planned_entry = hypothesis.planned_entry_price if hypothesis is not None else None
    adapter_decision = AdapterDecision(
        timestamp=decision_time,
        state=state,
        direction=direction,
        reason=reason,
        entry_price=planned_entry,
        invalidation_price=(
            hypothesis.invalidation_price if hypothesis is not None else None
        ),
        target_price=hypothesis.target_price if hypothesis is not None else None,
        replay_subject=subject,
    )
    evidence = CoreV1DecisionEvidence(
        subject=subject,
        pipeline_result=pipeline_result,
        hypothesis=hypothesis,
        adapter_decision=adapter_decision,
        quantity_input=quantity,
        decision_time=decision_time,
        planned_entry_price=planned_entry,
    )
    actionable = state is SetupState.ACTIVE_TRADE
    decision_record = DecisionRecord(
        subject=subject,
        actionable=actionable,
        direction=execution_direction if actionable else None,
        order_type=decision_order_type,
        stop_loss=hypothesis.invalidation_price if actionable else None,
        take_profit=hypothesis.target_price if actionable else None,
        quantity=quantity.quantity if actionable and quantity is not None else None,
        strategy_state=pipeline_result.state.value,
        evidence_reference=evidence,
        reason=reason,
    )
    return CoreV1DecisionBridgeResult(
        adapter_decision=adapter_decision,
        decision_record=decision_record,
        evidence=evidence,
    )


def _adapter_direction(
    hypothesis: CoreV1TradeHypothesis | None,
) -> AdapterDirection | None:
    if hypothesis is None:
        return None
    direction = hypothesis.direction
    if not isinstance(direction, str):
        return None
    if direction == "BULLISH":
        return AdapterDirection.BULLISH
    if direction == "BEARISH":
        return AdapterDirection.BEARISH
    return None


def _execution_direction(
    hypothesis: CoreV1TradeHypothesis | None,
) -> ExecutionDirection | None:
    if hypothesis is None:
        return None
    direction = hypothesis.direction
    if not isinstance(direction, str):
        return None
    if direction == "BULLISH":
        return ExecutionDirection.LONG
    if direction == "BEARISH":
        return ExecutionDirection.SHORT
    return None


def _validate_actionable_trace(
    pipeline_result: CoreV1PipelineResult,
    hypothesis: CoreV1TradeHypothesis,
) -> None:
    if (
        pipeline_result.liquidity is None
        or pipeline_result.displacement is None
        or pipeline_result.mss is None
        or pipeline_result.zone is None
        or pipeline_result.retracement is None
        or pipeline_result.invalidation is None
        or pipeline_result.target is None
    ):
        raise ValueError("Actionable pipeline result is missing a required governed stage.")
    if (
        hypothesis.planned_entry is not pipeline_result.retracement
        or hypothesis.invalidation is not pipeline_result.invalidation
        or hypothesis.target is not pipeline_result.target
    ):
        raise ValueError("Hypothesis components do not match the pipeline evidence.")
    if (
        pipeline_result.retracement.mss is not pipeline_result.mss
        or pipeline_result.retracement.liquidity is not pipeline_result.liquidity
        or pipeline_result.retracement.displacement is not pipeline_result.displacement
        or pipeline_result.retracement.zone is not pipeline_result.zone
    ):
        raise ValueError("Planned entry does not retain the pipeline prerequisite identities.")
    if (
        pipeline_result.mss.liquidity_event is not pipeline_result.liquidity
        or pipeline_result.mss.displacement is not pipeline_result.displacement
    ):
        raise ValueError("MSS does not retain pipeline liquidity/displacement identity.")

    trace = pipeline_result.provenance
    required = (
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
    )
    if any(not trace.get(name) for name in required):
        raise ValueError("Actionable pipeline result has incomplete component provenance.")
    expected_hypothesis_provenance = _build_provenance(
        hypothesis.planned_entry,
        hypothesis.invalidation,
        hypothesis.target,
    )
    if dict(hypothesis.provenance) != dict(expected_hypothesis_provenance):
        raise ValueError("Actionable hypothesis provenance is incomplete or inconsistent.")
    expected_pipeline_provenance = _derive_pipeline_metadata(pipeline_result)[2]
    if dict(pipeline_result.provenance) != expected_pipeline_provenance:
        raise ValueError("Actionable pipeline provenance is incomplete or inconsistent.")


def _utc_timestamp(value: object):
    import pandas as pd

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("decision_time must be a valid timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("decision_time must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


__all__ = [
    "CoreV1DecisionBridgeResult",
    "CoreV1DecisionEvidence",
    "CoreV1QuantityInput",
    "build_core_v1_decision_bridge",
]
