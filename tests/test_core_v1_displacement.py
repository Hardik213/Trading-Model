from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.core_v1_displacement import (
    CoreV1DisplacementObservation,
    DisplacementStatus,
    evaluate_core_v1_displacement,
)
from src.core_v1_evidence import (
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
)
from src.core_v1_liquidity import (
    CoreV1LiquidityAssessment,
    CoreV1LiquidityLevel,
    evaluate_core_v1_liquidity,
)
from src.displacement import Direction
from src.evidence_timing import EvidenceSourceObservation, EvidenceTiming
from src.market_structure import LiquiditySide
from src.replay_subject import ReplaySubject


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def make_subject() -> ReplaySubject:
    return ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )


def source(
    source_id: str,
    event: str,
    available: str,
    *,
    base_timeframe: bool = True,
) -> EvidenceSourceObservation:
    return EvidenceSourceObservation(
        source_id=source_id,
        source="synthetic",
        event_timestamp=ts(event),
        availability_timestamp=ts(available),
        provenance={"source_id": source_id},
        base_bar_timestamp=ts(event) if base_timeframe else None,
        is_base_timeframe=base_timeframe,
    )


def visible_observation(
    item: EvidenceSourceObservation,
    *,
    scope: ObservationScope = ObservationScope.BASE,
    value: object = None,
) -> SubjectVisibleObservation:
    timing = EvidenceTiming.from_inputs(
        event_timestamp=item.event_timestamp,
        source_observations=(item,),
        base_bar_scoped=scope is ObservationScope.BASE,
    )
    return SubjectVisibleObservation(
        value={"observation": item.source_id} if value is None else value,
        timing=timing,
        scope=scope,
    )


def make_inputs(
    subject: ReplaySubject | None = None,
    observations: tuple[EvidenceSourceObservation, ...] | None = None,
) -> SubjectVisibleInputs:
    subject = subject or make_subject()
    observations = observations or (
        source("impulse-bar", "2026-01-01T10:04:00Z", "2026-01-01T10:04:01Z"),
        source(
            "follow-through",
            "2026-01-01T10:06:00Z",
            "2026-01-01T10:06:17Z",
            base_timeframe=False,
        ),
    )
    base = tuple(
        visible_observation(item)
        for item in observations
        if item.is_base_timeframe
    )
    context = tuple(
        visible_observation(item, scope=ObservationScope.CONTEXT)
        for item in observations
        if not item.is_base_timeframe
    )
    return SubjectVisibleInputs(
        subject=subject,
        availability_timestamp=subject.availability_timestamp,
        base_observations=base,
        context_observations=context,
    )


def governed_observation(
    inputs: SubjectVisibleInputs,
    *,
    direction: Direction | None = Direction.BULLISH,
    source_observations: tuple[EvidenceSourceObservation, ...] | None = None,
    event: str = "2026-01-01T10:04:00Z",
    confirmation: str | None = "2026-01-01T10:06:00Z",
    availability: str = "2026-01-01T10:06:17Z",
) -> CoreV1DisplacementObservation:
    sources = source_observations or tuple(
        item
        for visible in inputs.base_observations + inputs.context_observations
        for item in visible.timing.source_observations
    )
    timing = EvidenceTiming(
        event_timestamp=ts(event),
        confirmation_timestamp=ts(confirmation) if confirmation else None,
        availability_timestamp=ts(availability),
        source_observations=sources,
        provenance={"governance": "synthetic test resolver"},
        base_bar_scoped=False,
    )
    return CoreV1DisplacementObservation(
        subject=inputs.subject,
        timing=timing,
        direction=direction,
        attributes={"interpretation": "caller-classified"},
    )


@dataclass
class FixedResolver:
    value: CoreV1DisplacementObservation | None
    received: SubjectVisibleInputs | None = None

    def classify(
        self,
        inputs: SubjectVisibleInputs,
        liquidity_context: tuple[CoreV1LiquidityAssessment, ...],
    ) -> CoreV1DisplacementObservation | None:
        self.received = inputs
        return self.value


def test_valid_governed_displacement_result_is_preserved():
    inputs = make_inputs()
    candidate = governed_observation(inputs)
    result = evaluate_core_v1_displacement(inputs, observation=candidate)
    assert result.status is DisplacementStatus.DISPLACEMENT
    assert result.observation is candidate
    assert result.direction is Direction.BULLISH


def test_displacement_event_confirmation_and_availability_remain_distinct():
    inputs = make_inputs()
    result = evaluate_core_v1_displacement(
        inputs,
        observation=governed_observation(inputs),
    )
    assert result.event_timestamp == ts("2026-01-01T10:04:00Z")
    assert result.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert result.availability_timestamp == ts("2026-01-01T10:06:17Z")
    assert result.event_timestamp < result.confirmation_timestamp
    assert result.confirmation_timestamp <= result.availability_timestamp
    assert result.confirmation_timestamp > inputs.subject.base_bar_timestamp


def test_availability_cutoff_rejects_late_displacement():
    inputs = make_inputs()
    late = source(
        "late-displacement",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:20Z",
    )
    timing = EvidenceTiming.from_inputs(
        event_timestamp=late.event_timestamp,
        source_observations=(late,),
        base_bar_scoped=False,
    )
    with pytest.raises(ValueError, match="unavailable"):
        CoreV1DisplacementObservation(inputs.subject, timing)


def test_base_bar_cutoff_is_distinct_from_availability_cutoff():
    inputs = make_inputs()
    beyond_base = source(
        "future-base-time",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:00Z",
    )
    timing = EvidenceTiming.from_inputs(
        event_timestamp=beyond_base.event_timestamp,
        source_observations=(beyond_base,),
        base_bar_scoped=False,
    )
    with pytest.raises(ValueError, match="base-bar"):
        CoreV1DisplacementObservation(inputs.subject, timing)


def test_future_observation_is_rejected_at_subject_input_boundary():
    future = source(
        "future-observation",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
        base_timeframe=False,
    )
    with pytest.raises(ValueError, match="unavailable"):
        make_inputs(observations=(future,))


def test_future_confirmation_is_rejected():
    inputs = make_inputs()
    impulse = inputs.base_observations[0].timing.source_observations[0]
    with pytest.raises(ValueError, match="availability cannot precede confirmation"):
        CoreV1DisplacementObservation(
            subject=inputs.subject,
            timing=EvidenceTiming(
                event_timestamp=impulse.event_timestamp,
                confirmation_timestamp=ts("2026-01-01T10:07:00Z"),
                availability_timestamp=inputs.subject.availability_timestamp,
                source_observations=(impulse,),
                base_bar_scoped=False,
            ),
        )


def test_replay_subject_identity_is_preserved():
    inputs = make_inputs()
    result = evaluate_core_v1_displacement(
        inputs,
        observation=governed_observation(inputs),
    )
    assert result.subject is inputs.subject
    assert result.observation.subject is inputs.subject


def test_provenance_survives_the_displacement_result():
    inputs = make_inputs()
    candidate = governed_observation(inputs)
    result = evaluate_core_v1_displacement(inputs, observation=candidate)
    assert result.provenance == candidate.timing.source_observations
    assert result.observation.timing.provenance["governance"] == "synthetic test resolver"


def test_displacement_provenance_must_reference_visible_sources():
    inputs = make_inputs()
    hidden = source(
        "hidden-source",
        "2026-01-01T10:04:00Z",
        "2026-01-01T10:04:01Z",
    )
    confirmation_source = inputs.context_observations[0].timing.source_observations[0]
    candidate = governed_observation(
        inputs,
        source_observations=(hidden, confirmation_source),
    )
    with pytest.raises(ValueError, match="outside visible inputs"):
        evaluate_core_v1_displacement(inputs, observation=candidate)


def test_unavailable_displacement_policy_fails_closed():
    inputs = make_inputs()
    result = evaluate_core_v1_displacement(inputs)
    assert result.status is DisplacementStatus.UNRESOLVED
    assert result.observation is None
    assert result.event_timestamp is None
    assert result.confirmation_timestamp is None
    assert result.direction is None
    assert result.availability_timestamp == inputs.subject.availability_timestamp


def test_no_thresholds_are_applied_when_governance_is_absent():
    inputs = make_inputs()
    weak_move = visible_observation(
        inputs.base_observations[0].timing.source_observations[0],
        value={"body_ratio": 0.01, "range_expansion": 0.1},
    )
    ordinary_inputs = SubjectVisibleInputs(
        subject=inputs.subject,
        availability_timestamp=inputs.subject.availability_timestamp,
        base_observations=(weak_move, *inputs.base_observations[1:]),
        context_observations=(),
    )
    result = evaluate_core_v1_displacement(ordinary_inputs)
    assert result.status is DisplacementStatus.UNRESOLVED
    assert result.observation is None


def test_ordinary_candle_movement_does_not_become_displacement_automatically():
    inputs = make_inputs()
    result = evaluate_core_v1_displacement(inputs)
    assert result.status is DisplacementStatus.UNRESOLVED


def test_displacement_does_not_automatically_become_mss_or_trade_signal():
    inputs = make_inputs()
    result = evaluate_core_v1_displacement(
        inputs,
        observation=governed_observation(inputs),
    )
    assert not hasattr(result, "mss")
    assert not hasattr(result, "entry_price")
    assert not hasattr(result, "target_price")
    assert not hasattr(result, "trade_signal")


def test_future_sentinel_cannot_alter_earlier_subject_result():
    inputs = make_inputs()
    earlier = evaluate_core_v1_displacement(
        inputs,
        observation=governed_observation(inputs),
    )
    future_sentinel = source(
        "future-sentinel",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
        base_timeframe=False,
    )
    future_visible = visible_observation(
        future_sentinel,
        scope=ObservationScope.CONTEXT,
        value={"future_sentinel": True},
    )
    with pytest.raises(ValueError, match="unavailable"):
        SubjectVisibleInputs(
            subject=inputs.subject,
            availability_timestamp=inputs.subject.availability_timestamp,
            base_observations=inputs.base_observations,
            context_observations=(future_visible,),
        )
    repeated = evaluate_core_v1_displacement(
        inputs,
        observation=governed_observation(inputs),
    )
    assert repeated == earlier


def test_unrestricted_historical_frame_is_rejected():
    source_item = source(
        "single-visible-source",
        "2026-01-01T10:04:00Z",
        "2026-01-01T10:04:01Z",
    )
    timing = EvidenceTiming.from_inputs(
        event_timestamp=source_item.event_timestamp,
        source_observations=(source_item,),
        base_bar_scoped=False,
    )
    with pytest.raises(TypeError, match="not frames"):
        SubjectVisibleObservation(
            value=pd.DataFrame({"Open": [100], "High": [101]}),
            timing=timing,
            scope=ObservationScope.BASE,
        )


def test_liquidity_context_is_optional_and_not_a_displacement_trigger():
    inputs = make_inputs()
    level_source = inputs.base_observations[0].timing.source_observations[0]
    level_timing = EvidenceTiming.from_inputs(
        event_timestamp=level_source.event_timestamp,
        source_observations=(level_source,),
    )
    liquidity_level = CoreV1LiquidityLevel(
        subject=inputs.subject,
        level_id="context-only",
        side=LiquiditySide.BSL,
        price=105.0,
        timing=level_timing,
    )
    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(liquidity_level,))
    resolver = FixedResolver(None)
    result = evaluate_core_v1_displacement(
        inputs,
        resolver=resolver,
        liquidity_context=(assessment,),
    )
    assert resolver.received is inputs
    assert result.status is DisplacementStatus.UNRESOLVED
