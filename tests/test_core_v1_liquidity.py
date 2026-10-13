from __future__ import annotations

import random
from dataclasses import dataclass, replace
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
    _validate_reaction,
    _visible_sources,
    detect_core_v1_liquidity_levels,
    evaluate_core_v1_liquidity,
)
from src.core_v1_tick_execution import CoreV1TickExecution
from src.evidence_timing import EvidenceSourceObservation, EvidenceTiming
from src.market_structure import (
    LiquidityLevel as CanonicalLiquidityLevel,
    LiquiditySide,
    SwingPoint,
    SwingType,
    swings_to_liquidity,
    swings_to_liquidity,
)
from src.replay_subject import ReplaySubject
from src.replay_engine import IncrementalHistoricalReplay, ReplayConfig
from src.tick_aggregation import TickAggregator
from src.tick_execution import ExecutionConfig, ExecutionEngine, QuoteEvent
from src.timeframe_context import TimeframeContext
from src.dukascopy_ticks import discover_dukascopy_tick_files, iter_dukascopy_ticks


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def make_subject() -> ReplaySubject:
    return ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:06:00Z"),
    )


def source(
    source_id: str,
    event: str,
    available: str,
    *,
    base_timeframe: bool = True,
    provenance: Mapping[str, object] | None = None,
) -> EvidenceSourceObservation:
    return EvidenceSourceObservation(
        source_id=source_id,
        source="synthetic",
        event_timestamp=ts(event),
        availability_timestamp=ts(available),
        provenance={"source_id": source_id, **(provenance or {})},
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
        value=dict(item.provenance),
        timing=timing,
        scope=scope,
    )


def canonical_origin(
    side: LiquiditySide,
    price: float,
    *,
    level_id: str,
    timestamp: str = "2026-01-01T10:00:00Z",
    confirmation: str = "2026-01-01T10:01:00Z",
    available: str = "2026-01-01T10:01:01Z",
) -> CanonicalLiquidityLevel:
    swing = SwingPoint(
        timestamp=ts(timestamp),
        price=price,
        kind=SwingType.HIGH if side is LiquiditySide.BSL else SwingType.LOW,
        source_index=0,
        confirmation_timestamp=ts(confirmation),
    )
    (level,) = swings_to_liquidity((swing,))
    return replace(level, availability_timestamp=ts(available))


def make_inputs(
    subject: ReplaySubject | None = None,
    source_observations: tuple[EvidenceSourceObservation, ...] | None = None,
) -> SubjectVisibleInputs:
    subject = subject or make_subject()
    if source_observations is None:
        bar_data = (
            (100.0, 102.0, 98.0, 100.0),
            (100.0, 104.0, 97.0, 101.0),
            (101.0, 105.0, 95.0, 102.0),
            (102.0, 103.0, 98.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 106.0, 96.0, 105.5),
            (105.0, 106.0, 94.0, 100.0),
        )
        source_observations = tuple(
            source(
                f"bar-{index}",
                f"2026-01-01T10:0{index}:00Z",
                f"2026-01-01T10:0{index}:01Z",
                provenance={
                    "Open": values[0],
                    "High": values[1],
                    "Low": values[2],
                    "Close": values[3],
                },
            )
            for index, values in enumerate(bar_data)
        ) + (
            source(
                "confirm-source",
                "2026-01-01T10:06:00Z",
                "2026-01-01T10:06:17Z",
                base_timeframe=False,
            ),
            source(
                "sweep-source",
                "2026-01-01T10:06:10Z",
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
    candidates = tuple(
        candidate
        for candidate in detect_core_v1_liquidity_levels(inputs)
        if candidate.side is side and candidate.price == price
    )
    if len(candidates) != 1:
        raise AssertionError("Expected exactly one detector-confirmed liquidity level.")
    candidate = candidates[0]
    if sources is not None and sources != candidate.timing.source_observations:
        raise ValueError("Liquidity level source override must match swing support bars.")
    return replace(candidate, level_id=f"level-{side.value}")


def priced_inputs(
    rows: tuple[tuple[str, float, float, float, float], ...],
) -> SubjectVisibleInputs:
    observations = []
    for index, (event, open_price, high, low, close) in enumerate(rows):
        event_timestamp = ts(event)
        source_item = source(
            f"market-{index}",
            event,
            (event_timestamp + pd.Timedelta(seconds=1)).isoformat(),
            provenance={
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
            },
        )
        observations.append(visible_observation(source_item))
    last_source = observations[-1].timing.source_observations[0]
    subject = ReplaySubject(
        availability_timestamp=last_source.availability_timestamp,
        base_bar_timestamp=last_source.event_timestamp,
    )
    return SubjectVisibleInputs(
        subject=subject,
        availability_timestamp=subject.availability_timestamp,
        base_observations=tuple(observations),
        context_observations=(),
    )


def priced_level(
    inputs: SubjectVisibleInputs,
    side: LiquiditySide,
    price: float,
    *,
    level_id: str,
    pivot_timestamp: str | None = None,
) -> CoreV1LiquidityLevel:
    candidates = [
        candidate
        for candidate in detect_core_v1_liquidity_levels(inputs)
        if candidate.side is side
        and candidate.price == price
        and candidate.origin is not None
        and candidate.origin.side is side
    ]
    if pivot_timestamp is not None:
        candidates = [
            candidate for candidate in candidates
            if candidate.origin is not None
            and candidate.origin.timestamp == ts(pivot_timestamp)
        ]
    if len(candidates) != 1:
        raise AssertionError("Expected exactly one detector-confirmed liquidity level.")
    return replace(candidates[0], level_id=level_id)


def breach(
    inputs: SubjectVisibleInputs,
    level_value: CoreV1LiquidityLevel,
    *,
    event: str = "2026-01-01T10:05:00Z",
    available: str = "2026-01-01T10:05:01Z",
    sources: tuple[EvidenceSourceObservation, ...] | None = None,
    price: float = 106.0,
) -> CoreV1LiquidityBreach:
    sources = sources or (
        *level_value.timing.source_observations,
        next(
            item
            for observation in inputs.base_observations
            for item in observation.timing.source_observations
            if item.event_timestamp == ts(event)
        ),
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
    confirmation_source = next(
        item
        for observation in inputs.base_observations
        for item in observation.timing.source_observations
        if item.event_timestamp > breach_value.timing.event_timestamp
    )
    sources = (*breach_value.timing.source_observations, confirmation_source)
    timing = EvidenceTiming.from_inputs(
        event_timestamp=breach_value.timing.event_timestamp,
        confirmation_timestamp=confirmation_source.event_timestamp,
        source_observations=sources,
        provenance={"resolver": "explicit test classifier"},
    )
    confirmation_observation = next(
        observation
        for observation in inputs.base_observations
        if confirmation_source in observation.timing.source_observations
    )
    prices = {
        str(key).lower(): value
        for key, value in confirmation_observation.value.items()
    }
    price_key = (
        "low"
        if breach_value.level.side is LiquiditySide.BSL
        else "high"
    )
    return CoreV1LiquidityReaction(
        subject=inputs.subject,
        breach=breach_value,
        kind=kind,
        timing=timing,
        price=float(prices[price_key]),
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
    assert assessment.level.origin is candidate.origin
    assert assessment.level.origin_source_id == "bar-4"
    assert assessment.breach is not None


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
    assert (
        assessment.level.timing.provenance["resolver"]
        == "core-v1-visible-swing-detector"
    )
    assert candidate.timing.source_observations == tuple(
        item.timing.source_observations[0]
        for item in inputs.base_observations[:5]
    )


def test_level_price_must_match_its_canonical_origin():
    inputs = make_inputs()
    candidate = level(inputs)
    with pytest.raises(ValueError, match="price does not match"):
        replace(
            candidate,
            origin=replace(candidate.origin, price=104.0),
        )


def test_level_side_must_match_its_canonical_origin():
    inputs = make_inputs()
    candidate = level(inputs)
    with pytest.raises(ValueError, match="side does not match"):
        replace(
            candidate,
            origin=replace(candidate.origin, side=LiquiditySide.SSL),
        )


def test_level_provenance_must_identify_its_originating_observation():
    inputs = make_inputs()
    candidate = level(inputs)
    with pytest.raises(ValueError, match="exactly one level observation"):
        replace(candidate, origin_source_id="unrelated-visible-source")


def test_arbitrary_price_with_visible_origin_provenance_is_rejected():
    inputs = make_inputs()
    candidate = level(inputs)
    with pytest.raises(ValueError, match="price does not match"):
        replace(candidate, price=106.0)


def test_level_without_origin_is_not_allowed_to_resolve_a_breach():
    inputs = make_inputs()
    candidate = replace(level(inputs), origin=None, origin_source_id=None)
    breach_value = breach(inputs, candidate)
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach_value),
    )
    assert assessment.level is candidate
    assert assessment.breach is None
    assert assessment.sweep is SweepKind.UNRESOLVED


def test_level_origin_must_be_visible_in_subject_inputs():
    inputs = make_inputs()
    candidate = level(inputs)
    inputs_without_origin = SubjectVisibleInputs(
        subject=inputs.subject,
        availability_timestamp=inputs.availability_timestamp,
        base_observations=tuple(
            observation
            for observation in inputs.base_observations
            if observation.value.get("source_id") != candidate.origin_source_id
        ),
        context_observations=inputs.context_observations,
    )
    with pytest.raises(ValueError, match="outside the visible inputs"):
        evaluate_core_v1_liquidity(inputs_without_origin, levels=(candidate,))


def test_visible_observation_must_establish_the_exact_canonical_level():
    inputs = make_inputs()
    candidate = level(inputs)
    altered_observations = tuple(
        replace(
            observation,
            value={
                **dict(observation.value),
                "High": 104.0,
            },
        )
        if observation.timing.source_observations[0].source_id == "bar-2"
        else observation
        for observation in inputs.base_observations
    )
    altered_inputs = SubjectVisibleInputs(
        subject=inputs.subject,
        availability_timestamp=inputs.availability_timestamp,
        base_observations=altered_observations,
        context_observations=inputs.context_observations,
    )
    (assessment,) = evaluate_core_v1_liquidity(
        altered_inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach(inputs, candidate)),
    )
    assert assessment.breach is None
    assert assessment.sweep is SweepKind.UNRESOLVED


def test_level_confirmation_after_subject_cutoff_is_rejected():
    inputs = make_inputs()
    level_id = "future-level"
    origin = canonical_origin(
        LiquiditySide.BSL,
        105.0,
        level_id=level_id,
        timestamp="2026-01-01T10:06:00Z",
        confirmation="2026-01-01T10:07:00Z",
        available="2026-01-01T10:07:01Z",
    )
    origin_source = source(
        level_id,
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:01Z",
    )
    with pytest.raises(ValueError, match="unavailable"):
        timing = EvidenceTiming.from_inputs(
            event_timestamp=origin.timestamp,
            confirmation_timestamp=origin.confirmation_timestamp,
            source_observations=(origin_source,),
        )
        CoreV1LiquidityLevel(
            subject=inputs.subject,
            level_id=level_id,
            side=LiquiditySide.BSL,
            price=105.0,
            timing=timing,
            origin=origin,
            origin_source_id=level_id,
        )


def test_confirmed_canonical_level_can_produce_a_later_confirmed_sweep():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    reaction_value = reaction(inputs, breach_value, ReactionKind.REJECTION)
    sweep_source = next(
        item
        for observation in inputs.context_observations
        for item in observation.timing.source_observations
        if item.source_id == "sweep-source"
    )
    sweep_timing = EvidenceTiming.from_inputs(
        event_timestamp=breach_value.timing.event_timestamp,
        confirmation_timestamp=sweep_source.event_timestamp,
        source_observations=(
            *reaction_value.timing.source_observations,
            sweep_source,
        ),
    )
    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(candidate,),
        breach_resolver=FixedBreachResolver(breach_value),
        reaction_resolver=FixedReactionResolver(reaction_value),
        sweep_resolver=FixedSweepResolver(
            (SweepKind.SWEEP, sweep_timing, {"resolver": "test"})
        ),
    )
    assert assessment.sweep is SweepKind.SWEEP
    assert assessment.level.origin is candidate.origin
    assert assessment.sweep_timing.confirmation_timestamp == sweep_source.event_timestamp


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


def test_breach_at_level_availability_is_rejected():
    inputs = make_inputs()
    candidate = level(inputs)
    equal_timestamp = candidate.timing.availability_timestamp
    equal_source = source(
        "breach-at-level-availability",
        equal_timestamp.isoformat(),
        (equal_timestamp + pd.Timedelta(seconds=1)).isoformat(),
    )
    timing = EvidenceTiming.from_inputs(
        event_timestamp=equal_timestamp,
        source_observations=(
            *candidate.timing.source_observations,
            equal_source,
        ),
        base_bar_scoped=True,
    )

    with pytest.raises(ValueError, match="strictly after level availability"):
        CoreV1LiquidityBreach(
            subject=inputs.subject,
            level=candidate,
            price=candidate.price + 1.0,
            timing=timing,
        )


def test_rejection_requires_an_observed_price():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    reaction_value = reaction(inputs, breach_value, ReactionKind.REJECTION)

    with pytest.raises(ValueError, match="requires an observed reaction price"):
        CoreV1LiquidityReaction(
            subject=inputs.subject,
            breach=breach_value,
            kind=ReactionKind.REJECTION,
            timing=reaction_value.timing,
            price=None,
        )


def test_rejection_price_must_match_the_visible_confirmation_bar():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    fabricated_reaction = replace(
        reaction(inputs, breach_value, ReactionKind.REJECTION),
        price=93.0,
    )

    with pytest.raises(ValueError, match="match the observed reaction bar price"):
        evaluate_core_v1_liquidity(
            inputs,
            levels=(candidate,),
            breach_resolver=FixedBreachResolver(breach_value),
            reaction_resolver=FixedReactionResolver(fabricated_reaction),
        )


def test_rejection_confirmation_with_duplicate_base_timestamps_fails_closed():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    reaction_value = reaction(inputs, breach_value, ReactionKind.REJECTION)
    confirmation_source = reaction_value.timing.source_observations[-1]
    duplicate_source = replace(
        confirmation_source,
        source_id="duplicate-reaction-confirmation",
        provenance={
            **confirmation_source.provenance,
            "source_id": "duplicate-reaction-confirmation",
        },
    )
    confirmation_observation = next(
        observation
        for observation in inputs.base_observations
        if confirmation_source in observation.timing.source_observations
    )
    duplicate_observation = replace(
        confirmation_observation,
        timing=EvidenceTiming.from_inputs(
            event_timestamp=duplicate_source.event_timestamp,
            source_observations=(duplicate_source,),
            base_bar_scoped=True,
        ),
    )
    duplicate_inputs = replace(
        inputs,
        base_observations=(*inputs.base_observations, duplicate_observation),
    )
    ambiguous_reaction = replace(
        reaction_value,
        timing=replace(
            reaction_value.timing,
            source_observations=(
                *reaction_value.timing.source_observations,
                duplicate_source,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="exactly one visible base observation",
    ):
        _validate_reaction(
            duplicate_inputs,
            breach_value,
            ambiguous_reaction,
            _visible_sources(duplicate_inputs),
        )


def test_rejection_confirmation_with_duplicate_timestamp_uses_source_provenance():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    reaction_value = reaction(inputs, breach_value, ReactionKind.REJECTION)
    confirmation_source = reaction_value.timing.source_observations[-1]
    duplicate_source = replace(
        confirmation_source,
        source_id="duplicate-reaction-confirmation",
        provenance={
            **confirmation_source.provenance,
            "source_id": "duplicate-reaction-confirmation",
        },
    )
    confirmation_observation = next(
        observation
        for observation in inputs.base_observations
        if confirmation_source in observation.timing.source_observations
    )
    duplicate_observation = replace(
        confirmation_observation,
        value={
            **confirmation_observation.value,
            "Low": float(confirmation_observation.value["Low"]) + 1.0,
        },
        timing=EvidenceTiming.from_inputs(
            event_timestamp=duplicate_source.event_timestamp,
            source_observations=(duplicate_source,),
            base_bar_scoped=True,
        ),
    )
    duplicate_inputs = replace(
        inputs,
        base_observations=(*inputs.base_observations, duplicate_observation),
    )

    _validate_reaction(
        duplicate_inputs,
        breach_value,
        reaction_value,
        _visible_sources(duplicate_inputs),
    )


def test_breach_event_and_availability_are_distinct():
    inputs = make_inputs()
    candidate = level(inputs)
    breach_value = breach(inputs, candidate)
    assert breach_value.timing.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert breach_value.timing.availability_timestamp == ts("2026-01-01T10:05:01Z")
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
    assert assessment.breach.timing.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert assessment.reaction.timing.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert assessment.reaction.timing.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert assessment.reaction.timing.availability_timestamp == ts("2026-01-01T10:06:01Z")


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
        event_timestamp=ts("2026-01-01T10:05:00Z"),
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
        "2026-01-01T10:06:10Z",
        "2026-01-01T10:06:17Z",
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
    assert assessment.event_direction.value == "BEARISH"
    assert not hasattr(assessment, "entry_price")
    assert not hasattr(assessment, "target_price")
    assert not hasattr(assessment, "trade_signal")


@pytest.mark.parametrize(
    ("side", "pivot_price", "breach_high", "breach_low", "reaction_high", "reaction_low", "direction"),
    [
        (LiquiditySide.BSL, 105.0, 106.0, 101.0, 105.0, 104.0, "BEARISH"),
        (LiquiditySide.SSL, 95.0, 99.0, 94.0, 96.0, 95.0, "BULLISH"),
    ],
)
def test_visible_breach_and_later_reaction_confirm_directional_sweep(
    side,
    pivot_price,
    breach_high,
    breach_low,
    reaction_high,
    reaction_low,
    direction,
):
    if side is LiquiditySide.BSL:
        support = (
            ("10:00", 100.0, 102.0, 98.0, 100.0),
            ("10:01", 100.0, 104.0, 97.0, 101.0),
            ("10:02", 101.0, 105.0, 96.0, 102.0),
            ("10:03", 102.0, 103.0, 97.0, 100.0),
            ("10:04", 100.0, 101.0, 98.0, 100.0),
        )
    else:
        support = (
            ("10:00", 100.0, 103.0, 98.0, 101.0),
            ("10:01", 101.0, 104.0, 97.0, 102.0),
            ("10:02", 102.0, 105.0, 95.0, 103.0),
            ("10:03", 103.0, 104.0, 96.0, 102.0),
            ("10:04", 102.0, 103.0, 98.0, 101.0),
        )
    rows = tuple(
        (f"2026-01-01T{time}:00Z", open_price, high, low, close)
        for time, open_price, high, low, close in support
    ) + (
        (
            "2026-01-01T10:05:00Z",
            pivot_price,
            breach_high,
            breach_low,
            breach_high if side is LiquiditySide.BSL else breach_low,
        ),
        (
            "2026-01-01T10:06:00Z",
            pivot_price,
            reaction_high,
            reaction_low,
            pivot_price,
        ),
    )
    inputs = priced_inputs(rows)
    candidate = priced_level(
        inputs,
        side,
        pivot_price,
        level_id=f"confirmed-{side.value}",
    )

    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))

    assert assessment.level is candidate
    assert assessment.breach.level is candidate
    assert assessment.breach.timing.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert assessment.reaction.timing.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert assessment.sweep is SweepKind.SWEEP
    assert assessment.event_direction.value == direction


def test_touch_without_strict_breach_does_not_create_liquidity_event():
    rows = (
        ("2026-01-01T10:00:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 97.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 105.0, 96.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 103.0, 97.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 101.0, 98.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 105.0, 99.0, 104.0),
        ("2026-01-01T10:06:00Z", 104.0, 104.5, 100.0, 101.0),
    )
    inputs = priced_inputs(rows)
    candidate = priced_level(inputs, LiquiditySide.BSL, 105.0, level_id="touch")

    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))

    assert assessment.breach is None
    assert assessment.sweep is SweepKind.UNRESOLVED


def test_breach_without_later_return_remains_unconfirmed():
    rows = (
        ("2026-01-01T10:00:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 97.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 105.0, 96.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 103.0, 97.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 101.0, 98.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 106.0, 99.0, 105.5),
        ("2026-01-01T10:06:00Z", 105.5, 107.0, 105.1, 106.5),
    )
    inputs = priced_inputs(rows)
    candidate = priced_level(inputs, LiquiditySide.BSL, 105.0, level_id="no-return")

    (assessment,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))

    assert assessment.breach is not None
    assert assessment.reaction is None
    assert assessment.sweep is SweepKind.UNRESOLVED


def test_pre_breach_or_same_observation_reaction_cannot_confirm_sweep():
    before_breach = (
        ("2026-01-01T10:00:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 97.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 105.0, 96.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 103.0, 97.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 101.0, 98.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 104.5, 99.0, 104.0),
        ("2026-01-01T10:06:00Z", 104.0, 106.0, 104.0, 105.5),
    )
    inputs = priced_inputs(before_breach)
    candidate = priced_level(inputs, LiquiditySide.BSL, 105.0, level_id="pre-return")
    (earlier_return,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))
    assert earlier_return.breach is not None
    assert earlier_return.reaction is None
    assert earlier_return.sweep is SweepKind.UNRESOLVED

    same_observation = (
        ("2026-01-01T10:00:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 97.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 105.0, 96.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 103.0, 97.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 101.0, 98.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 106.0, 99.0, 105.0),
    )
    inputs = priced_inputs(same_observation)
    candidate = priced_level(inputs, LiquiditySide.BSL, 105.0, level_id="same-bar")
    (same_bar_result,) = evaluate_core_v1_liquidity(inputs, levels=(candidate,))
    assert same_bar_result.breach is not None
    assert same_bar_result.reaction is None
    assert same_bar_result.sweep is SweepKind.UNRESOLVED


def test_future_reaction_is_invisible_until_its_availability_subject():
    rows = (
        ("2026-01-01T10:00:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 97.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 105.0, 96.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 103.0, 97.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 101.0, 98.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 106.0, 99.0, 105.5),
        ("2026-01-01T10:06:00Z", 105.5, 105.8, 104.0, 104.5),
    )
    before_reaction = priced_inputs(rows[:6])
    candidate = priced_level(
        before_reaction,
        LiquiditySide.BSL,
        105.0,
        level_id="cutoff",
    )
    (earlier,) = evaluate_core_v1_liquidity(
        before_reaction,
        levels=(candidate,),
    )
    assert earlier.breach is not None
    assert earlier.reaction is None
    assert earlier.sweep is SweepKind.UNRESOLVED

    after_reaction = priced_inputs(rows)
    later_candidate = priced_level(
        after_reaction,
        LiquiditySide.BSL,
        105.0,
        level_id="cutoff",
    )
    (later,) = evaluate_core_v1_liquidity(after_reaction, levels=(later_candidate,))
    assert later.sweep is SweepKind.SWEEP
    assert later.reaction.timing.confirmation_timestamp == ts("2026-01-01T10:06:00Z")


def test_ambiguous_opposite_side_breaches_fail_closed():
    rows = (
        ("2026-01-01T10:00:00Z", 100.0, 103.0, 97.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 96.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 106.0, 94.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 104.0, 96.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 103.0, 97.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 107.0, 93.0, 100.0),
    )
    inputs = priced_inputs(rows)
    high = priced_level(inputs, LiquiditySide.BSL, 106.0, level_id="high")
    low = priced_level(inputs, LiquiditySide.SSL, 94.0, level_id="low")

    assessments = evaluate_core_v1_liquidity(inputs, levels=(high, low))

    assert len(assessments) == 2
    assert all(item.breach is None for item in assessments)
    assert all(item.sweep is SweepKind.UNRESOLVED for item in assessments)
    assert all(
        item.sweep_provenance["reason"] == "AMBIGUOUS_LEVEL_SELECTION"
        for item in assessments
    )


def test_nearest_breached_level_is_selected_and_level_identity_preserved():
    rows = (
        ("2026-01-01T10:00:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 97.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 107.0, 96.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 104.0, 97.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 103.0, 98.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 104.0, 98.0, 102.0),
        ("2026-01-01T10:06:00Z", 102.0, 105.0, 97.0, 103.0),
        ("2026-01-01T10:07:00Z", 103.0, 104.0, 98.0, 102.0),
        ("2026-01-01T10:08:00Z", 102.0, 103.0, 99.0, 102.0),
        ("2026-01-01T10:09:00Z", 102.0, 108.0, 101.0, 107.0),
        ("2026-01-01T10:10:00Z", 107.0, 107.5, 104.0, 104.5),
    )
    inputs = priced_inputs(rows)
    nearer = priced_level(inputs, LiquiditySide.BSL, 105.0, level_id="nearer")
    farther = priced_level(
        inputs,
        LiquiditySide.BSL,
        107.0,
        level_id="farther",
        pivot_timestamp="2026-01-01T10:02:00Z",
    )

    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(farther, nearer),
    )

    assert assessment.level is nearer
    assert assessment.level.level_id == "nearer"
    assert assessment.breach.level is nearer
    assert assessment.breach.timing.event_timestamp == ts("2026-01-01T10:09:00Z")
    assert assessment.reaction.timing.confirmation_timestamp == ts("2026-01-01T10:10:00Z")


def test_equal_nearest_levels_are_ambiguous_not_arbitrarily_selected():
    rows = (
        ("2026-01-01T10:00:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 97.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 105.0, 96.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 103.0, 97.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 101.0, 98.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:06:00Z", 100.0, 105.0, 97.0, 102.0),
        ("2026-01-01T10:07:00Z", 102.0, 103.0, 98.0, 100.0),
        ("2026-01-01T10:08:00Z", 100.0, 101.0, 98.0, 100.0),
        ("2026-01-01T10:09:00Z", 100.0, 106.0, 99.0, 105.5),
    )
    inputs = priced_inputs(rows)
    first = priced_level(
        inputs,
        LiquiditySide.BSL,
        105.0,
        level_id="first",
        pivot_timestamp="2026-01-01T10:02:00Z",
    )
    second = priced_level(
        inputs,
        LiquiditySide.BSL,
        105.0,
        level_id="second",
        pivot_timestamp="2026-01-01T10:06:00Z",
    )

    assessments = evaluate_core_v1_liquidity(inputs, levels=(first, second))

    assert all(item.breach is None for item in assessments)
    assert all(
        item.sweep_provenance["reason"] == "AMBIGUOUS_LEVEL_SELECTION"
        for item in assessments
    )


def test_fabricated_swing_and_matching_provenance_are_rejected_against_visible_bars():
    inputs = make_inputs()
    actual = level(inputs)
    fake_origin = canonical_origin(
        LiquiditySide.BSL,
        104.0,
        level_id="fabricated",
        timestamp="2026-01-01T10:02:00Z",
        confirmation="2026-01-01T10:04:00Z",
        available="2026-01-01T10:04:01Z",
    )
    fake_level = CoreV1LiquidityLevel(
        subject=inputs.subject,
        level_id="fabricated-level",
        side=LiquiditySide.BSL,
        price=104.0,
        timing=actual.timing,
        origin=fake_origin,
        origin_source_id=actual.origin_source_id,
    )

    (assessment,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(fake_level,),
        breach_resolver=FixedBreachResolver(breach(inputs, fake_level, price=106.0)),
    )

    assert assessment.level is fake_level
    assert assessment.breach is None
    assert assessment.sweep is SweepKind.UNRESOLVED


def test_visible_swing_detector_preserves_distinct_pivot_confirmation_and_availability():
    inputs = make_inputs()

    candidate = next(
        item
        for item in detect_core_v1_liquidity_levels(inputs)
        if item.side is LiquiditySide.BSL
    )

    assert candidate.origin.timestamp == ts("2026-01-01T10:02:00Z")
    assert candidate.origin.confirmation_timestamp == ts("2026-01-01T10:04:00Z")
    assert candidate.origin.availability_timestamp == ts("2026-01-01T10:04:01Z")
    assert candidate.timing.event_timestamp == candidate.origin.timestamp
    assert candidate.timing.confirmation_timestamp == candidate.origin.confirmation_timestamp
    assert candidate.timing.availability_timestamp == candidate.origin.availability_timestamp
    assert candidate.origin_source_id == candidate.timing.source_observations[-1].source_id


def test_equal_highs_and_lows_do_not_form_strict_swings():
    start = pd.Timestamp("2026-01-01T10:00:00")
    rows = tuple(
        (
            (start + pd.Timedelta(minutes=index)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            100.0,
            105.0,
            95.0,
            100.0,
        )
        for index in range(7)
    )
    inputs = priced_inputs(rows)

    assert detect_core_v1_liquidity_levels(inputs) == ()


def test_confirmation_and_availability_equal_to_subject_cutoff_are_visible():
    all_inputs = make_inputs()
    support = tuple(
        observation
        for observation in all_inputs.base_observations
        if observation.timing.event_timestamp <= ts("2026-01-01T10:04:00Z")
    )
    cutoff = ts("2026-01-01T10:04:01Z")
    subject = ReplaySubject(
        availability_timestamp=cutoff,
        base_bar_timestamp=ts("2026-01-01T10:04:00Z"),
    )
    inputs = SubjectVisibleInputs(
        subject=subject,
        availability_timestamp=cutoff,
        base_observations=support,
        context_observations=(),
    )

    levels = detect_core_v1_liquidity_levels(inputs)

    assert any(
        item.origin is not None
        and item.origin.confirmation_timestamp == subject.base_bar_timestamp
        and item.origin.availability_timestamp == subject.availability_timestamp
        for item in levels
    )


def test_equal_timestamp_base_observations_are_not_tie_broken_into_a_swing():
    original = make_inputs()
    duplicate = original.base_observations[3]
    equal_timestamp_inputs = SubjectVisibleInputs(
        subject=original.subject,
        availability_timestamp=original.availability_timestamp,
        base_observations=(
            *original.base_observations[:4],
            replace(
                duplicate,
                timing=EvidenceTiming.from_inputs(
                    event_timestamp=duplicate.timing.event_timestamp,
                    source_observations=(
                        replace(
                            duplicate.timing.source_observations[0],
                            source_id="duplicate-time-different-observation",
                        ),
                    ),
                    base_bar_scoped=True,
                ),
            ),
            *original.base_observations[4:],
        ),
        context_observations=(),
    )

    assert detect_core_v1_liquidity_levels(equal_timestamp_inputs) == ()


def test_detector_level_rejects_source_lineage_from_another_observation_set():
    inputs = make_inputs()
    candidate = level(inputs)
    replaced_observation = inputs.base_observations[0]
    original_source = replaced_observation.timing.source_observations[0]
    mismatched_source = replace(original_source, source_id="unrelated-replay-source")
    mismatch_inputs = SubjectVisibleInputs(
        subject=inputs.subject,
        availability_timestamp=inputs.availability_timestamp,
        base_observations=(
            visible_observation(mismatched_source),
            *inputs.base_observations[1:],
        ),
        context_observations=inputs.context_observations,
    )

    with pytest.raises(ValueError, match="outside the visible inputs"):
        evaluate_core_v1_liquidity(mismatch_inputs, levels=(candidate,))


def test_detector_level_rejects_another_replay_subject():
    inputs = make_inputs()
    candidate = level(inputs)
    other_subject = ReplaySubject(
        availability_timestamp=inputs.subject.availability_timestamp,
        base_bar_timestamp=inputs.subject.base_bar_timestamp - pd.Timedelta(seconds=1),
    )
    other_candidate = replace(candidate, subject=other_subject)

    with pytest.raises(ValueError, match="different ReplaySubject"):
        evaluate_core_v1_liquidity(inputs, levels=(other_candidate,))


def test_delayed_gap_bar_availability_is_the_first_later_finalizing_tick():
    aggregator = TickAggregator("5min")
    aggregator.add_tick("2026-01-01T10:20:05Z", 100.0, 100.5)
    delayed_tick = ts("2026-01-01T10:32:30Z")
    emitted = aggregator.add_tick(delayed_tick, 101.0, 101.5)

    assert len(emitted) == 1
    assert emitted[0]["interval_start"] == ts("2026-01-01T10:20:00Z")
    assert emitted[0]["interval_end"] == ts("2026-01-01T10:25:00Z")
    assert emitted[0]["available_at"] == delayed_tick
    assert emitted[0]["available_at"] != emitted[0]["interval_end"]

    rows = tuple(
        (
            f"2026-01-01T10:{minute:02d}:00Z",
            100.0,
            high,
            low,
            close,
        )
        for minute, high, low, close in (
            (0, 101.0, 99.0, 100.0),
            (5, 102.0, 98.0, 101.0),
            (10, 105.0, 97.0, 102.0),
            (15, 103.0, 98.0, 100.0),
            (20, 101.0, 99.0, 100.0),
        )
    )
    ordinary_inputs = priced_inputs(rows)
    last_observation = ordinary_inputs.base_observations[-1]
    last_source = last_observation.timing.source_observations[0]
    delayed_source = replace(
        last_source,
        availability_timestamp=emitted[0]["available_at"],
    )
    delayed_timing = EvidenceTiming.from_inputs(
        event_timestamp=last_source.event_timestamp,
        source_observations=(delayed_source,),
        base_bar_scoped=True,
    )
    delayed_observation = replace(last_observation, timing=delayed_timing)
    subject = ReplaySubject(
        availability_timestamp=delayed_tick,
        base_bar_timestamp=last_source.event_timestamp,
    )
    delayed_inputs = SubjectVisibleInputs(
        subject=subject,
        availability_timestamp=delayed_tick,
        base_observations=(
            *ordinary_inputs.base_observations[:-1],
            delayed_observation,
        ),
        context_observations=(),
    )
    candidate = next(
        item
        for item in detect_core_v1_liquidity_levels(delayed_inputs)
        if item.side is LiquiditySide.BSL
    )

    assert candidate.origin.availability_timestamp == emitted[0]["available_at"]
    assert candidate.timing.availability_timestamp == emitted[0]["available_at"]


def test_detector_rejects_future_observation_when_input_validation_is_bypassed():
    inputs = make_inputs()
    future_source = source(
        "future-base-observation",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:01Z",
    )
    future_observation = visible_observation(future_source)
    object.__setattr__(
        inputs,
        "base_observations",
        (*inputs.base_observations, future_observation),
    )

    with pytest.raises(ValueError, match="exceeds the ReplaySubject cutoff"):
        detect_core_v1_liquidity_levels(inputs)


def test_detector_is_invariant_to_post_cutoff_development_bars(require_canonical_dataset):
    raw_root = require_canonical_dataset("dukascopy")
    january_2022 = tuple(
        path
        for path in discover_dukascopy_tick_files(raw_root / "2022")
        if "2022-01-01-2022-02-01" in path.name
    )
    assert january_2022, "Expected a January 2022 development tick file."

    aggregator = TickAggregator("5min")
    bars = []
    for batch in iter_dukascopy_ticks(january_2022, chunk_size=65_536):
        for tick in batch.itertuples(index=False):
            bars.extend(aggregator.add_tick(tick.timestamp, tick.bid, tick.ask))
        if len(bars) >= 48:
            break
    assert len(bars) >= 48, "Development slice did not yield enough complete bars."

    observations = []
    for index, bar in enumerate(bars):
        item = source(
            f"dev-2022-bar-{index}",
            bar["interval_start"].isoformat(),
            bar["available_at"].isoformat(),
            provenance={
                "Open": bar["bid_open"],
                "High": bar["bid_high"],
                "Low": bar["bid_low"],
                "Close": bar["bid_close"],
                "availability_source": "first later observed finalizing tick",
            },
        )
        observations.append(visible_observation(item))

    rng = random.Random(20261010)
    candidate_indices = list(range(5, len(observations)))
    cutoffs = sorted(rng.sample(candidate_indices, min(16, len(candidate_indices))))

    for index in cutoffs:
        as_of = observations[index].timing.source_observations[0].availability_timestamp
        base_cutoff = max(
            observation.timing.event_timestamp
            for observation in observations
            if observation.timing.availability_timestamp <= as_of
        )

        def visible_inputs(candidates):
            causal = tuple(
                observation
                for observation in candidates
                if observation.timing.event_timestamp <= base_cutoff
                and observation.timing.availability_timestamp <= as_of
            )
            subject = ReplaySubject(as_of, base_cutoff)
            return SubjectVisibleInputs(
                subject=subject,
                availability_timestamp=as_of,
                base_observations=causal,
                context_observations=(),
            )

        with_future = visible_inputs(observations)
        without_future = visible_inputs(
            tuple(
                observation
                for observation in observations
                if observation.timing.event_timestamp <= base_cutoff
            )
        )
        assert tuple(
            item.timing.source_observations[0].source_id
            for item in with_future.base_observations
        ) == tuple(
            item.timing.source_observations[0].source_id
            for item in without_future.base_observations
        )

        def signature(inputs):
            return tuple(
                (
                    item.side,
                    item.price,
                    item.origin.timestamp,
                    item.origin.confirmation_timestamp,
                    item.origin.availability_timestamp,
                    item.origin_source_id,
                    item.timing.source_observations,
                )
                for item in detect_core_v1_liquidity_levels(inputs)
            )

        assert signature(with_future) == signature(without_future)


def test_production_visible_inputs_are_truncation_invariant_on_development_ticks(
    require_canonical_dataset,
):
    raw_root = require_canonical_dataset("dukascopy")
    january_2022 = tuple(
        path
        for path in discover_dukascopy_tick_files(raw_root / "2022")
        if "2022-01-01-2022-02-01" in path.name
    )
    assert january_2022, "Expected a January 2022 development tick file."

    aggregator = TickAggregator("1min")
    bars = []
    tick_events_by_interval = {}
    source_order = 0
    for batch in iter_dukascopy_ticks(january_2022, chunk_size=65_536):
        for tick in batch.itertuples(index=False):
            timestamp = pd.Timestamp(tick.timestamp).tz_convert("UTC")
            tick_events_by_interval.setdefault(timestamp.floor("1min"), []).append(
                QuoteEvent(
                    timestamp=timestamp,
                    bid=float(tick.bid),
                    ask=float(tick.ask),
                    source_order=source_order,
                    provenance={"development_slice": "2022-01"},
                )
            )
            source_order += 1
            bars.extend(aggregator.add_tick(timestamp, tick.bid, tick.ask))
            if len(bars) >= 64:
                break
        if len(bars) >= 64:
            break
    assert len(bars) >= 32, "Development slice did not yield enough complete bars."

    frame = pd.DataFrame(
        {
            "Open": [bar["bid_open"] for bar in bars],
            "High": [bar["bid_high"] for bar in bars],
            "Low": [bar["bid_low"] for bar in bars],
            "Close": [bar["bid_close"] for bar in bars],
            "available_at": [bar["available_at"] for bar in bars],
        },
        index=[bar["interval_start"] for bar in bars],
    )
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
    timeframe = "1min"
    replay = IncrementalHistoricalReplay(
        config=ReplayConfig(timeframe=timeframe, source="DUKASCOPY_2022"),
        history_limits={timeframe: 128},
        max_pending_groups=8,
        max_pending_bars=128,
    )
    execution = ExecutionEngine(
        execution_run_id="liquidity-truncation-test",
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
    integration = CoreV1TickExecution(
        aggregators={timeframe: TickAggregator(timeframe)},
        replay=replay,
        execution=execution,
    )
    for interval_start, events in tick_events_by_interval.items():
        integration._ticks_by_interval[timeframe][interval_start].extend(events)

    full_as_of = pd.Timestamp(frame["available_at"].max()).tz_convert("UTC")
    full_base_cutoff = pd.Timestamp(frame.index.max()).tz_convert("UTC")
    full_subject = ReplaySubject(full_as_of, full_base_cutoff)
    full_inputs = integration._visible_inputs(
        full_subject,
        frame,
        TimeframeContext(frames={timeframe: frame}, replay_subject=full_subject),
    )
    candidates = detect_core_v1_liquidity_levels(full_inputs)
    assert candidates, "Development slice must include a confirmed swing."
    confirming_candidate = min(
        candidates,
        key=lambda item: item.origin.availability_timestamp,
    )
    confirming_bar = confirming_candidate.origin.confirmation_timestamp
    confirming_row = frame.loc[confirming_bar]
    confirming_availability = pd.Timestamp(
        confirming_row["available_at"]
    ).tz_convert("UTC")
    special_cutoffs = (
        confirming_availability - pd.Timedelta(nanoseconds=1),
        confirming_availability,
        confirming_availability + pd.Timedelta(nanoseconds=1),
    )
    rng = random.Random(20261010)
    available_cutoffs = sorted(
        {
            pd.Timestamp(value).tz_convert("UTC")
            for value in frame["available_at"]
            if pd.Timestamp(value).tz_convert("UTC") > confirming_availability
        }
    )
    cutoffs = tuple(
        sorted(
            set(special_cutoffs)
            | set(rng.sample(available_cutoffs, min(16, len(available_cutoffs))))
        )
    )
    assert len(cutoffs) >= 16

    def visible_inputs(candidates_frame, as_of):
        visible_rows = candidates_frame.loc[
            (candidates_frame.index <= as_of)
            & (candidates_frame["available_at"] <= as_of)
        ]
        if visible_rows.empty:
            return None
        base_cutoff = pd.Timestamp(visible_rows.index.max()).tz_convert("UTC")
        subject = ReplaySubject(as_of, base_cutoff)
        return integration._visible_inputs(
            subject,
            visible_rows,
            TimeframeContext(
                frames={timeframe: visible_rows},
                replay_subject=subject,
            ),
        )

    def signature(inputs):
        return tuple(
            (
                item.side,
                item.price,
                item.origin.timestamp,
                item.origin.confirmation_timestamp,
                item.origin.availability_timestamp,
                item.origin_source_id,
                item.timing.source_observations,
            )
            for item in detect_core_v1_liquidity_levels(inputs)
        )

    assert confirming_candidate.origin.confirmation_timestamp == confirming_bar
    assert confirming_candidate.origin.availability_timestamp >= confirming_availability
    for as_of in cutoffs:
        with_post_cutoff_data = visible_inputs(frame, as_of)
        if with_post_cutoff_data is None:
            assert visible_inputs(frame.loc[frame.index <= as_of], as_of) is None
            continue
        raw_without_post_cutoff = frame.loc[
            frame.index <= with_post_cutoff_data.subject.base_bar_timestamp
        ]
        without_post_cutoff_data = visible_inputs(raw_without_post_cutoff, as_of)
        if with_post_cutoff_data is None or without_post_cutoff_data is None:
            assert with_post_cutoff_data is without_post_cutoff_data
            continue
        assert tuple(
            item.timing.source_observations[0].source_id
            for item in with_post_cutoff_data.base_observations
        ) == tuple(
            item.timing.source_observations[0].source_id
            for item in without_post_cutoff_data.base_observations
        )
        assert signature(with_post_cutoff_data) == signature(
            without_post_cutoff_data
        )
