from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import pytest

from src.core_v1_evidence import (
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
)
from src.core_v1_liquidity import (
    CoreV1LiquidityAssessment,
    CoreV1LiquidityBreach,
    CoreV1LiquidityLevel,
    CoreV1LiquidityReaction,
    ReactionKind,
    SweepKind,
    evaluate_core_v1_liquidity,
)
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
) -> SubjectVisibleObservation:
    timing = EvidenceTiming.from_inputs(
        event_timestamp=item.event_timestamp,
        source_observations=(item,),
        base_bar_scoped=scope is ObservationScope.BASE,
    )
    return SubjectVisibleObservation(
        value={"source_id": item.source_id},
        timing=timing,
        scope=scope,
    )


def make_inputs(
    subject: ReplaySubject | None = None,
    source_observations: tuple[EvidenceSourceObservation, ...] | None = None,
) -> SubjectVisibleInputs:
    subject = subject or make_subject()
    source_observations = source_observations or (
        source("level-source", "2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z"),
        source("breach-source", "2026-01-01T10:04:00Z", "2026-01-01T10:04:01Z"),
        source(
            "confirm-source",
            "2026-01-01T10:06:00Z",
            "2026-01-01T10:06:17Z",
            base_timeframe=False,
        ),
    )
    base = tuple(
        visible_observation(item)
        for item in source_observations
        if item.is_base_timeframe
    )
    context = tuple(
        visible_observation(item, scope=ObservationScope.CONTEXT)
        for item in source_observations
        if not item.is_base_timeframe
    )
    return SubjectVisibleInputs(
        subject=subject,
        availability_timestamp=subject.availability_timestamp,
        base_observations=base,
        context_observations=context,
    )


def level(
    inputs: SubjectVisibleInputs,
    side: LiquiditySide = LiquiditySide.BSL,
    *,
    sources: tuple[EvidenceSourceObservation, ...] | None = None,
    price: float = 105.0,
) -> CoreV1LiquidityLevel:
    sources = sources or (inputs.base_observations[0].timing.source_observations[0],)
    timing = EvidenceTiming.from_inputs(
        event_timestamp=min(item.event_timestamp for item in sources),
        source_observations=sources,
        provenance={"resolver": "test-supplied"},
    )
    return CoreV1LiquidityLevel(
        subject=inputs.subject,
        level_id=f"level-{side.value}",
        side=side,
        price=price,
        timing=timing,
    )


def breach(
    inputs: SubjectVisibleInputs,
    level_value: CoreV1LiquidityLevel,
    *,
    event: str = "2026-01-01T10:04:00Z",
    available: str = "2026-01-01T10:04:01Z",
    sources: tuple[EvidenceSourceObservation, ...] | None = None,
    price: float = 106.0,
) -> CoreV1LiquidityBreach:
    sources = sources or (
        *level_value.timing.source_observations,
        inputs.base_observations[1].timing.source_observations[0],
    )
    timing = EvidenceTiming.from_inputs(
        event_timestamp=ts(event),
        source_observations=sources,
        provenance={"observation": "synthetic breach"},
        base_bar_scoped=True,
    )
    if timing.availability_timestamp != ts(available):
        raise AssertionError("test source availability does not match breach availability")
    return CoreV1LiquidityBreach(
        subject=inputs.subject,
        level=level_value,
        price=price,
        timing=timing,
    )


def reaction(
    inputs: SubjectVisibleInputs,
    breach_value: CoreV1LiquidityBreach,
    kind: ReactionKind,
) -> CoreV1LiquidityReaction:
    confirmation_source = inputs.context_observations[0].timing.source_observations[0]
    sources = (*breach_value.timing.source_observations, confirmation_source)
    timing = EvidenceTiming.from_inputs(
        event_timestamp=breach_value.timing.event_timestamp,
        confirmation_timestamp=confirmation_source.event_timestamp,
        source_observations=sources,
        provenance={"resolver": "explicit test classifier"},
    )
    return CoreV1LiquidityReaction(
        subject=inputs.subject,
        breach=breach_value,
        kind=kind,
        timing=timing,
        price=104.0,
    )


@dataclass
class FixedBreachResolver:
    value: CoreV1LiquidityBreach | None

    def detect(
        self,
        inputs: SubjectVisibleInputs,
        level: CoreV1LiquidityLevel,
    ) -> CoreV1LiquidityBreach | None:
        return self.value


@dataclass
class FixedReactionResolver:
    value: CoreV1LiquidityReaction | None

    def classify(
        self,
        inputs: SubjectVisibleInputs,
        breach: CoreV1LiquidityBreach,
    ) -> CoreV1LiquidityReaction | None:
        return self.value


@dataclass
class FixedSweepResolver:
    value: tuple[SweepKind, EvidenceTiming, Mapping[str, object]] | None

    def classify(
        self,
        inputs: SubjectVisibleInputs,
        reaction: CoreV1LiquidityReaction,
    ) -> tuple[SweepKind, EvidenceTiming, Mapping[str, object]] | None:
        return self.value


def test_valid_bsl_liquidity_level_is_preserved():
    inputs = make_inputs()
    candidate = level(inputs, LiquiditySide.BSL)
    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))
    assert assessment.level.side is LiquiditySide.BSL
    assert assessment.level.price == 105.0
    assert assessment.breach is None


def test_valid_ssl_liquidity_level_is_preserved():
    inputs = make_inputs()
    candidate = level(inputs, LiquiditySide.SSL, price=95.0)
    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))
    assert assessment.level.side is LiquiditySide.SSL
    assert assessment.level.price == 95.0


def test_level_provenance_is_required_and_preserved():
    inputs = make_inputs()
    candidate = level(inputs)
    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))
    assert assessment.level.timing.source_observations == candidate.timing.source_observations
    assert assessment.level.timing.provenance["resolver"] == "test-supplied"


def test_level_availability_after_subject_is_rejected():
    inputs = make_inputs()
    current = inputs.subject
    late_source = source(
        "late-level-source",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:20Z",
        base_timeframe=False,
    )
    with pytest.raises(ValueError, match="unavailable"):
        late_timing = EvidenceTiming.from_inputs(
            event_timestamp=late_source.event_timestamp,
            source_observations=(late_source,),
        )
        CoreV1LiquidityLevel(
            subject=current,
            level_id="too-late",
            side=LiquiditySide.BSL,
            price=105.0,
            timing=late_timing,
        )


def test_breach_is_a_distinct_observation_from_level():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach_value),
    )
    assert assessment.level is candidate
    assert assessment.breach is breach_value
    assert assessment.breach.timing.event_timestamp != assessment.level.timing.event_timestamp
    assert assessment.reaction is None


def test_breach_event_and_availability_are_distinct():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    assert breach_value.timing.event_timestamp == ts("2026-01-01T10:04:00Z")
    assert breach_value.timing.availability_timestamp == ts("2026-01-01T10:04:01Z")
    assert breach_value.timing.confirmation_timestamp is None


def test_future_breach_is_rejected():
    inputs = make_inputs()
    candidate = level(inputs)
    future_source = source(
        "future-breach",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
        base_timeframe=False,
    )
    with pytest.raises(ValueError, match="unavailable"):
        future_timing = EvidenceTiming.from_inputs(
            event_timestamp=future_source.event_timestamp,
            source_observations=(
                *candidate.timing.source_observations,
                future_source,
            ),
            base_bar_scoped=True,
        )
        CoreV1LiquidityBreach(
            subject=inputs.subject,
            level=candidate,
            price=106.0,
            timing=future_timing,
        )


def test_future_confirmation_is_rejected():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    future_confirmation = source(
        "future-confirmation",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
        base_timeframe=False,
    )
    with pytest.raises(ValueError, match="availability cannot precede confirmation"):
        EvidenceTiming(
            event_timestamp=breach_value.timing.event_timestamp,
            confirmation_timestamp=future_confirmation.event_timestamp,
            availability_timestamp=inputs.subject.availability_timestamp,
            source_observations=(
                *breach_value.timing.source_observations,
            ),
        )


def test_confirmation_timestamp_stays_later_than_original_breach():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    reaction_value = reaction(inputs, breach_value, ReactionKind.REJECTION)
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach_value),
        reaction_resolver=FixedReactionResolver(reaction_value),
    )
    assert assessment.breach.timing.event_timestamp == ts("2026-01-01T10:04:00Z")
    assert assessment.reaction.timing.event_timestamp == ts("2026-01-01T10:04:00Z")
    assert assessment.reaction.timing.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert assessment.reaction.timing.availability_timestamp == ts("2026-01-01T10:06:17Z")


def test_breach_does_not_automatically_become_a_sweep():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach_value),
    )
    assert assessment.breach is not None
    assert assessment.sweep is SweepKind.UNRESOLVED


def test_wick_only_observation_does_not_automatically_become_a_sweep():
    inputs = make_inputs()
    candidate = level(inputs)
    wick_only_breach = breach(inputs, candidate, price=106.0)
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(wick_only_breach),
    )
    assert assessment.breach.price == 106.0
    assert assessment.reaction is None
    assert assessment.sweep is SweepKind.UNRESOLVED


@pytest.mark.parametrize("kind", [ReactionKind.REJECTION, ReactionKind.ACCEPTANCE])
def test_rejection_and_acceptance_remain_distinguishable(kind):
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    reaction_value = reaction(inputs, breach_value, kind)
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach_value),
        reaction_resolver=FixedReactionResolver(reaction_value),
    )
    assert assessment.reaction.kind is kind
    assert assessment.sweep is SweepKind.UNRESOLVED


def test_unresolved_sweep_policy_fails_closed():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    reaction_value = reaction(inputs, breach_value, ReactionKind.REJECTION)
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach_value),
        reaction_resolver=FixedReactionResolver(reaction_value),
    )
    assert assessment.sweep is SweepKind.UNRESOLVED
    assert assessment.sweep_timing is None


def test_replay_subject_identity_is_preserved():
    inputs = make_inputs()
    candidate = level(inputs)
    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))
    assert assessment.subject is inputs.subject
    assert assessment.level.subject is inputs.subject


def test_derived_breach_preserves_level_and_observation_provenance():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach_value),
    )
    assert assessment.breach.timing.source_observations == breach_value.timing.source_observations
    assert all(
        source in assessment.breach.timing.source_observations
        for source in candidate.timing.source_observations
    )


def test_breach_requires_a_distinct_breach_time_source_observation():
    inputs = make_inputs()
    candidate = level(inputs)
    invalid_timing = EvidenceTiming(
        event_timestamp=ts("2026-01-01T10:04:00Z"),
        availability_timestamp=inputs.subject.availability_timestamp,
        source_observations=candidate.timing.source_observations,
        base_bar_scoped=True,
    )
    invalid_breach = CoreV1LiquidityBreach(
        subject=inputs.subject,
        level=candidate,
        price=106.0,
        timing=invalid_timing,
    )
    with pytest.raises(ValueError, match="breach-time observation"):
        evaluate_core_v1_liquidity(
            inputs,
            levels=(candidate,),
            breach_resolver=FixedBreachResolver(invalid_breach),
        )


def test_reaction_requires_a_separate_confirmation_source_observation():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    invalid_timing = EvidenceTiming(
        event_timestamp=breach_value.timing.event_timestamp,
        confirmation_timestamp=ts("2026-01-01T10:06:00Z"),
        availability_timestamp=inputs.subject.availability_timestamp,
        source_observations=breach_value.timing.source_observations,
    )
    with pytest.raises(ValueError, match="confirmation observation"):
        CoreV1LiquidityReaction(
            subject=inputs.subject,
            breach=breach_value,
            kind=ReactionKind.REJECTION,
            timing=invalid_timing,
        )


def test_base_bar_cutoff_is_enforced_separately():
    current = make_subject()
    future_base = source(
        "future-base",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:00Z",
    )
    with pytest.raises(ValueError, match="base-bar"):
        make_inputs(current, (future_base,))


def test_availability_cutoff_rejects_late_visible_observation():
    current = make_subject()
    late_context = source(
        "late-context",
        "2026-01-01T10:06:10Z",
        "2026-01-01T10:06:20Z",
        base_timeframe=False,
    )
    with pytest.raises(ValueError, match="unavailable"):
        make_inputs(current, (late_context,))


def test_future_sentinel_cannot_alter_earlier_subject_result():
    current = make_subject()
    inputs = make_inputs(current)
    candidate = level(inputs)
    (earlier_result,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))

    future_sentinel = source(
        "future-sentinel",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
        base_timeframe=False,
    )
    future_observation = visible_observation(
        future_sentinel,
        scope=ObservationScope.CONTEXT,
    )
    with pytest.raises(ValueError, match="unavailable"):
        SubjectVisibleInputs(
            subject=current,
            availability_timestamp=current.availability_timestamp,
            base_observations=inputs.base_observations,
            context_observations=(future_observation,),
        )
    (same_result,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))
    assert same_result == earlier_result


def test_unrestricted_frame_input_is_rejected():
    item = source("frame-ref", "2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z")
    timing = EvidenceTiming.from_inputs(
        event_timestamp=item.event_timestamp,
        source_observations=(item,),
        base_bar_scoped=True,
    )
    with pytest.raises(TypeError, match="not frames"):
        SubjectVisibleObservation(
            value=pd.DataFrame({"High": [106.0]}),
            timing=timing,
            scope=ObservationScope.BASE,
        )


def test_liquidity_assessment_emits_no_trade_signal():
    inputs = make_inputs()
    candidate = level(inputs)
    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))
    assert not hasattr(assessment, "direction")
    assert not hasattr(assessment, "entry_price")
    assert not hasattr(assessment, "target_price")
    assert not hasattr(assessment, "trade_signal")
