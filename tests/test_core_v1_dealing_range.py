from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.core_v1_dealing_range import (
    DealingRangeContext,
    DealingRangeStatus,
    UnavailableDealingRangeResolver,
    build_dealing_range_context,
)
from src.core_v1_evidence import (
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
)
from src.dealing_range import DealingRange, RangeLocation
from src.evidence_timing import EvidenceSourceObservation, EvidenceTiming, SubjectEvidence
from src.replay_subject import ReplaySubject


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def make_subject() -> ReplaySubject:
    return ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )


def source(source_id: str, event: str, available: str) -> EvidenceSourceObservation:
    base_timestamp = ts(event)
    return EvidenceSourceObservation(
        source_id=source_id,
        source="synthetic",
        event_timestamp=base_timestamp,
        availability_timestamp=ts(available),
        provenance={"row_id": source_id},
        base_bar_timestamp=base_timestamp,
        is_base_timeframe=True,
    )


def observation(item: EvidenceSourceObservation) -> SubjectVisibleObservation:
    timing = EvidenceTiming.from_inputs(
        event_timestamp=item.event_timestamp,
        source_observations=(item,),
        base_bar_scoped=True,
    )
    return SubjectVisibleObservation(
        value={"source_id": item.source_id},
        timing=timing,
        scope=ObservationScope.BASE,
    )


def visible_inputs(
    subject: ReplaySubject | None = None,
    observations: tuple[SubjectVisibleObservation, ...] | None = None,
) -> SubjectVisibleInputs:
    subject = subject or make_subject()
    return SubjectVisibleInputs(
        subject=subject,
        availability_timestamp=subject.availability_timestamp,
        base_observations=observations or (),
        context_observations=(),
    )


def range_evidence(
    subject: ReplaySubject,
    provenance: tuple[EvidenceSourceObservation, ...],
    *,
    high_time: str = "2026-01-01T10:00:00Z",
    low_time: str = "2026-01-01T10:05:00Z",
    high: float = 120.0,
    low: float = 100.0,
    equilibrium: float = 110.0,
    event: str = "2026-01-01T10:05:00Z",
    confirmation: str | None = "2026-01-01T10:06:00Z",
    available: str = "2026-01-01T10:06:17Z",
) -> SubjectEvidence[DealingRange]:
    timing = EvidenceTiming(
        event_timestamp=ts(event),
        confirmation_timestamp=ts(confirmation) if confirmation else None,
        availability_timestamp=ts(available),
        source_observations=provenance,
        provenance={"resolver": "synthetic"},
        base_bar_scoped=False,
    )
    return SubjectEvidence(
        subject=subject,
        timing=timing,
        value=DealingRange(
            anchor_high_timestamp=ts(high_time),
            anchor_high=high,
            anchor_low_timestamp=ts(low_time),
            anchor_low=low,
            equilibrium=equilibrium,
        ),
    )


@dataclass
class FixedResolver:
    value: SubjectEvidence[DealingRange] | None
    received: SubjectVisibleInputs | None = None

    def resolve(
        self,
        inputs: SubjectVisibleInputs,
    ) -> SubjectEvidence[DealingRange] | None:
        self.received = inputs
        return self.value


def test_no_canonical_resolver_yields_explicit_no_range_without_anchors():
    inputs = visible_inputs()
    result = build_dealing_range_context(inputs, UnavailableDealingRangeResolver())
    assert result.status is DealingRangeStatus.NO_RANGE_UNRESOLVED
    assert result.dealing_range is None
    assert result.event_timestamp is None
    assert result.confirmation_timestamp is None
    assert result.availability_timestamp == inputs.availability_timestamp
    assert result.location is None
    assert "anchor policy remains unresolved" in result.reason


def test_resolved_context_preserves_subject_timing_and_provenance():
    current = make_subject()
    sources = (
        source("high-anchor", "2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z"),
        source("low-anchor", "2026-01-01T10:05:00Z", "2026-01-01T10:06:17Z"),
    )
    observations = tuple(observation(item) for item in sources)
    inputs = visible_inputs(current, observations)
    candidate = range_evidence(current, sources)
    result = build_dealing_range_context(inputs, FixedResolver(candidate))

    assert result.status is DealingRangeStatus.RESOLVED
    assert result.subject is current
    assert result.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert result.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert result.availability_timestamp == ts("2026-01-01T10:06:17Z")
    assert result.provenance == sources
    assert result.dealing_range == candidate.value


def test_unavailable_range_candidate_is_rejected_before_range_resolution():
    current = make_subject()
    too_late = source(
        "late-anchor",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:20Z",
    )
    with pytest.raises(ValueError, match="unavailable"):
        range_evidence(
            current,
            (too_late,),
            low_time="2026-01-01T10:05:00Z",
            available="2026-01-01T10:06:20Z",
        )


def test_range_anchor_beyond_base_bar_cutoff_is_rejected():
    current = make_subject()
    too_late = source(
        "future-base-anchor",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:00Z",
    )
    with pytest.raises(ValueError, match="base-bar"):
        visible_inputs(current, (observation(too_late),))


def test_unavailable_or_future_resolver_sources_never_reach_resolver():
    current = make_subject()
    future = source(
        "future-sentinel",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
    )
    resolver = FixedResolver(None)
    future_observation = SubjectVisibleObservation(
        value={"sentinel": True},
        timing=EvidenceTiming.from_inputs(
            event_timestamp=future.event_timestamp,
            source_observations=(future,),
            base_bar_scoped=False,
        ),
        scope=ObservationScope.CONTEXT,
    )
    with pytest.raises(ValueError, match="unavailable"):
        inputs = visible_inputs(current)
        SubjectVisibleInputs(
            subject=inputs.subject,
            availability_timestamp=inputs.availability_timestamp,
            base_observations=(),
            context_observations=(future_observation,),
        )
    assert resolver.received is None


def test_future_sentinel_cannot_influence_resolved_range():
    current = make_subject()
    high_source = source(
        "high-anchor",
        "2026-01-01T10:00:00Z",
        "2026-01-01T10:00:01Z",
    )
    low_source = source(
        "low-anchor",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:17Z",
    )
    allowed = (observation(high_source), observation(low_source))
    resolver = FixedResolver(range_evidence(current, (high_source, low_source)))
    result = build_dealing_range_context(visible_inputs(current, allowed), resolver)
    assert result.status is DealingRangeStatus.RESOLVED
    assert resolver.received is not None
    future_sentinel = source(
        "future-sentinel",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
    )
    assert all(
        item.value["source_id"] != future_sentinel.source_id
        for item in resolver.received.base_observations
    )
    future_observation = SubjectVisibleObservation(
        value={"source_id": future_sentinel.source_id, "sentinel": 999999.0},
        timing=EvidenceTiming.from_inputs(
            event_timestamp=future_sentinel.event_timestamp,
            source_observations=(future_sentinel,),
            base_bar_scoped=False,
        ),
        scope=ObservationScope.CONTEXT,
    )
    blocked_resolver = FixedResolver(None)
    with pytest.raises(ValueError, match="unavailable"):
        SubjectVisibleInputs(
            subject=current,
            availability_timestamp=current.availability_timestamp,
            base_observations=allowed,
            context_observations=(future_observation,),
        )
    assert blocked_resolver.received is None


@pytest.mark.parametrize(
    ("high", "low", "equilibrium"),
    [
        (100.0, 100.0, 100.0),
        (99.0, 100.0, 99.5),
        (120.0, 100.0, 111.0),
    ],
)
def test_invalid_range_geometry_returns_explicit_no_range(high, low, equilibrium):
    current = make_subject()
    sources = (
        source("high-anchor", "2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z"),
        source("low-anchor", "2026-01-01T10:05:00Z", "2026-01-01T10:06:17Z"),
    )
    candidate = range_evidence(
        current,
        sources,
        high=high,
        low=low,
        equilibrium=equilibrium,
    )
    result = build_dealing_range_context(
        visible_inputs(current, tuple(observation(item) for item in sources)),
        FixedResolver(candidate),
    )
    assert result.status is DealingRangeStatus.NO_RANGE_UNRESOLVED
    assert result.dealing_range is None
    assert result.location is None


@pytest.mark.parametrize(
    ("price", "expected_location"),
    [
        (115.0, RangeLocation.PREMIUM),
        (110.0, RangeLocation.EQUILIBRIUM),
        (105.0, RangeLocation.DISCOUNT),
    ],
)
def test_premium_equilibrium_discount_are_only_descriptive_context(
    price,
    expected_location,
):
    current = make_subject()
    high_source = source("high", "2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z")
    low_source = source("low", "2026-01-01T10:05:00Z", "2026-01-01T10:06:17Z")
    sources = (high_source, low_source)
    inputs = visible_inputs(current, tuple(observation(item) for item in sources))
    candidate = range_evidence(current, sources)
    mid_source = source("reference", "2026-01-01T10:05:00Z", "2026-01-01T10:06:17Z")
    reference_timing = EvidenceTiming.from_inputs(
        event_timestamp=mid_source.event_timestamp,
        source_observations=(mid_source,),
        base_bar_scoped=True,
    )
    # Only observations supplied as visible inputs may support the descriptive price.
    inputs = visible_inputs(
        current,
        tuple(observation(item) for item in (*sources, mid_source)),
    )
    reference = SubjectEvidence(current, reference_timing, price)

    result = build_dealing_range_context(
        inputs,
        FixedResolver(candidate),
        reference_price=reference,
    )
    assert result.location is expected_location
    assert result.reference_price == price
    assert not hasattr(result, "direction")
    assert not hasattr(result, "entry_price")


def test_future_confirmation_is_rejected_even_when_event_is_earlier():
    current = make_subject()
    sources = (
        source("high", "2026-01-01T10:00:00Z", "2026-01-01T10:00:01Z"),
        source("low", "2026-01-01T10:05:00Z", "2026-01-01T10:05:00Z"),
    )
    with pytest.raises(ValueError, match="availability cannot precede confirmation"):
        range_evidence(
            current,
            sources,
            confirmation="2026-01-01T10:07:00Z",
            available="2026-01-01T10:06:17Z",
        )


def test_unresolved_context_rejects_fabricated_anchors():
    current = make_subject()
    with pytest.raises(ValueError, match="cannot contain fabricated"):
        DealingRangeContext(
            subject=current,
            status=DealingRangeStatus.NO_RANGE_UNRESOLVED,
            availability_timestamp=current.availability_timestamp,
            dealing_range=DealingRange(
                anchor_high_timestamp=ts("2026-01-01T10:00:00Z"),
                anchor_high=120,
                anchor_low_timestamp=ts("2026-01-01T10:05:00Z"),
                anchor_low=100,
                equilibrium=110,
            ),
        )
