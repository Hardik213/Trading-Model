from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.core_v1_dealing_range import DealingRangeContext, DealingRangeStatus
from src.core_v1_evidence import (
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
)
from src.core_v1_reactionary_zones import (
    CoreV1FVGObservation,
    CoreV1OrderBlockObservation,
    CoreV1PDArrayObservation,
    UnavailableFVGResolver,
    UnavailableOrderBlockResolver,
    ZoneStatus,
    evaluate_core_v1_fvg,
    evaluate_core_v1_order_block,
    evaluate_core_v1_pd_array,
)
from src.dealing_range import DealingRange, RangeLocation
from src.evidence_timing import EvidenceSourceObservation, EvidenceTiming
from src.fvg import FVG, FVGDirection
from src.pd_arrays import PDArray, PDArrayType
from src.replay_subject import ReplaySubject


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def subject() -> ReplaySubject:
    return ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )


def source(
    source_id: str,
    event: str,
    available: str | None = None,
) -> EvidenceSourceObservation:
    event_time = ts(event)
    return EvidenceSourceObservation(
        source_id=source_id,
        source="synthetic",
        event_timestamp=event_time,
        availability_timestamp=ts(available or event),
        provenance={"row_id": source_id},
        base_bar_timestamp=event_time,
        is_base_timeframe=True,
    )


def visible_observation(item: EvidenceSourceObservation) -> SubjectVisibleObservation:
    return SubjectVisibleObservation(
        value={"source_id": item.source_id},
        timing=EvidenceTiming.from_inputs(
            event_timestamp=item.event_timestamp,
            source_observations=(item,),
            base_bar_scoped=True,
        ),
        scope=ObservationScope.BASE,
    )


def setup(
    current: ReplaySubject | None = None,
    sources: tuple[EvidenceSourceObservation, ...] | None = None,
) -> tuple[ReplaySubject, SubjectVisibleInputs, CoreV1FVGObservation]:
    current = current or subject()
    sources = sources or (
        source("first", "2026-01-01T10:00:00Z"),
        source("third", "2026-01-01T10:05:00Z", "2026-01-01T10:06:10Z"),
    )
    inputs = SubjectVisibleInputs(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=tuple(visible_observation(item) for item in sources),
        context_observations=(),
    )
    fvg = FVG(
        direction=FVGDirection.BULLISH,
        formation_timestamp=ts("2026-01-01T10:05:00Z"),
        confirmation_timestamp=ts("2026-01-01T10:05:00Z"),
        first_candle_timestamp=ts("2026-01-01T10:00:00Z"),
        third_candle_timestamp=ts("2026-01-01T10:05:00Z"),
        lower_bound=101.0,
        upper_bound=103.0,
        midpoint=102.0,
        size=2.0,
    )
    timing = EvidenceTiming(
        event_timestamp=fvg.formation_timestamp,
        confirmation_timestamp=fvg.confirmation_timestamp,
        availability_timestamp=max(item.availability_timestamp for item in sources),
        source_observations=sources,
        provenance={"resolver": "caller-governed", "reference": "fvg-observation"},
        base_bar_scoped=True,
    )
    return current, inputs, CoreV1FVGObservation(current, fvg, timing)


def test_governed_fvg_preserves_subject_timing_and_provenance():
    current, inputs, candidate = setup()
    result = evaluate_core_v1_fvg(inputs, observation=candidate)

    assert result.status is ZoneStatus.RESOLVED
    assert result.observation is candidate
    assert result.subject is current
    assert result.observation.subject is current
    assert result.observation.timing.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert result.observation.timing.confirmation_timestamp == ts("2026-01-01T10:05:00Z")
    assert result.observation.timing.availability_timestamp == ts("2026-01-01T10:06:10Z")
    assert result.observation.timing.source_observations == candidate.timing.source_observations
    assert result.observation.timing.provenance["reference"] == "fvg-observation"


@pytest.mark.parametrize(
    ("lower", "upper", "midpoint", "size"),
    [
        (float("nan"), 103.0, 102.0, 2.0),
        (103.0, 101.0, 102.0, 2.0),
        (101.0, 103.0, float("inf"), 2.0),
        (101.0, 103.0, 101.5, 2.0),
        (101.0, 103.0, 102.0, 3.0),
    ],
)
def test_fvg_bounds_must_be_finite_and_consistent(lower, upper, midpoint, size):
    current, _, candidate = setup()
    invalid_fvg = replace(
        candidate.fvg,
        lower_bound=lower,
        upper_bound=upper,
        midpoint=midpoint,
        size=size,
    )
    with pytest.raises(ValueError, match="Zone|FVG"):
        CoreV1FVGObservation(current, invalid_fvg, candidate.timing)


def test_later_fvg_confirmation_is_preserved_and_must_be_visible():
    current = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:06:00Z"),
    )
    confirmation_source = source(
        "confirmation",
        "2026-01-01T10:05:30Z",
        "2026-01-01T10:06:10Z",
    )
    inputs_sources = (
        source("first", "2026-01-01T10:00:00Z"),
        source("third", "2026-01-01T10:05:00Z"),
        confirmation_source,
    )
    inputs = SubjectVisibleInputs(
        current,
        current.availability_timestamp,
        tuple(visible_observation(item) for item in inputs_sources),
        (),
    )
    initial = FVG(
        FVGDirection.BULLISH,
        ts("2026-01-01T10:05:00Z"),
        ts("2026-01-01T10:05:30Z"),
        ts("2026-01-01T10:00:00Z"),
        ts("2026-01-01T10:05:00Z"),
        101.0,
        103.0,
        102.0,
        2.0,
    )
    timing = EvidenceTiming(
        event_timestamp=initial.formation_timestamp,
        confirmation_timestamp=initial.confirmation_timestamp,
        availability_timestamp=ts("2026-01-01T10:06:10Z"),
        source_observations=inputs_sources,
        base_bar_scoped=True,
    )
    candidate = CoreV1FVGObservation(current, initial, timing)
    result = evaluate_core_v1_fvg(inputs, observation=candidate)
    assert result.observation.timing.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert result.observation.timing.confirmation_timestamp == ts("2026-01-01T10:05:30Z")
    assert result.observation.timing.availability_timestamp == ts("2026-01-01T10:06:10Z")


def test_unavailable_fvg_is_rejected_at_subject_availability_cutoff():
    current, _, candidate = setup()
    late_source = source("late", "2026-01-01T10:05:00Z", "2026-01-01T10:06:20Z")
    late_timing = EvidenceTiming(
        event_timestamp=candidate.timing.event_timestamp,
        confirmation_timestamp=candidate.timing.confirmation_timestamp,
        availability_timestamp=late_source.availability_timestamp,
        source_observations=(late_source,),
        base_bar_scoped=True,
    )
    with pytest.raises(ValueError, match="unavailable"):
        CoreV1FVGObservation(current, candidate.fvg, late_timing)


def test_fvg_event_or_confirmation_after_base_bar_cutoff_is_rejected():
    current, _, candidate = setup()
    late = ts("2026-01-01T10:06:00Z")
    future_fvg = replace(
        candidate.fvg,
        formation_timestamp=late,
        confirmation_timestamp=late,
        third_candle_timestamp=late,
    )
    late_source = source("future", late.isoformat(), late.isoformat())
    future_timing = EvidenceTiming(
        event_timestamp=late,
        confirmation_timestamp=late,
        availability_timestamp=late,
        source_observations=(late_source,),
        base_bar_scoped=False,
    )
    with pytest.raises(ValueError, match="base-bar"):
        CoreV1FVGObservation(current, future_fvg, future_timing)


def test_fvg_resolver_requires_visible_source_lineage_and_subject_identity():
    current, inputs, candidate = setup()
    foreign_subject = ReplaySubject(
        current.availability_timestamp + pd.Timedelta(minutes=1),
        current.base_bar_timestamp,
    )
    other, _, other_candidate = setup(foreign_subject)
    with pytest.raises(ValueError, match="another ReplaySubject"):
        evaluate_core_v1_fvg(inputs, observation=other_candidate)

    hidden = source("hidden", "2026-01-01T10:04:00Z")
    hidden_timing = replace(
        candidate.timing,
        source_observations=candidate.timing.source_observations + (hidden,),
    )
    hidden_candidate = CoreV1FVGObservation(current, candidate.fvg, hidden_timing)
    with pytest.raises(ValueError, match="outside visible inputs"):
        evaluate_core_v1_fvg(inputs, observation=hidden_candidate)


def test_missing_canonical_fvg_detector_fails_closed_without_invented_threshold():
    current, inputs, _ = setup()
    result = evaluate_core_v1_fvg(inputs, resolver=UnavailableFVGResolver())
    assert result.status is ZoneStatus.UNRESOLVED
    assert result.observation is None
    assert result.subject is current
    assert not hasattr(result, "threshold")


def test_only_repository_supported_fvg_pd_array_is_resolved():
    current, inputs, candidate = setup()
    fvg_result = evaluate_core_v1_fvg(inputs, observation=candidate)
    result = evaluate_core_v1_pd_array(inputs, fvg_result)
    assert result.status is ZoneStatus.RESOLVED
    assert result.observation.array.array_type is PDArrayType.FVG
    assert result.observation.subject is current
    assert result.observation.fvg_observation is candidate

    unsupported = PDArray(
        array_type="ORDER_BLOCK",
        direction=FVGDirection.BULLISH,
        lower_bound=101.0,
        upper_bound=103.0,
        midpoint=102.0,
        source_timestamp=candidate.fvg.confirmation_timestamp,
    )
    with pytest.raises(ValueError, match="Unsupported PD-array"):
        CoreV1PDArrayObservation(current, unsupported, candidate.timing, candidate)


def test_fvg_only_pd_array_location_is_descriptive_not_entry_qualification():
    current, inputs, candidate = setup()
    fvg_result = evaluate_core_v1_fvg(inputs, observation=candidate)
    timing = candidate.timing
    context = DealingRangeContext(
        subject=current,
        status=DealingRangeStatus.RESOLVED,
        availability_timestamp=timing.availability_timestamp,
        event_timestamp=timing.event_timestamp,
        dealing_range=DealingRange(
            anchor_high_timestamp=ts("2026-01-01T10:00:00Z"),
            anchor_high=120.0,
            anchor_low_timestamp=ts("2026-01-01T10:05:00Z"),
            anchor_low=100.0,
            equilibrium=110.0,
        ),
        provenance=timing.source_observations,
    )
    result = evaluate_core_v1_pd_array(inputs, fvg_result, dealing_range=context)

    assert result.observation.array.range_location is RangeLocation.DISCOUNT
    assert not hasattr(result, "entry")
    assert not hasattr(result, "trade")


def test_touching_fvg_does_not_create_entry_mss_or_trade():
    current, inputs, candidate = setup()
    result = evaluate_core_v1_fvg(inputs, observation=candidate)

    assert candidate.fvg.contains(102.0)
    assert result.status is ZoneStatus.RESOLVED
    assert not hasattr(result, "entry")
    assert not hasattr(result, "mss")
    assert not hasattr(result, "trade")


def test_no_fvg_means_pd_array_remains_unresolved():
    current, inputs, _ = setup()
    no_fvg = evaluate_core_v1_fvg(inputs)
    array = evaluate_core_v1_pd_array(inputs, no_fvg)
    assert array.status is ZoneStatus.UNRESOLVED
    assert array.observation is None


def test_order_block_is_unresolved_without_governed_observation_or_resolver():
    current, inputs, _ = setup()
    result = evaluate_core_v1_order_block(inputs, resolver=UnavailableOrderBlockResolver())
    assert result.status is ZoneStatus.UNRESOLVED
    assert result.observation is None
    assert result.subject is current
    assert not hasattr(result, "heuristic")


def test_governed_order_block_is_only_context_and_preserves_provenance():
    current, inputs, fvg_candidate = setup()
    candidate = CoreV1OrderBlockObservation(
        subject=current,
        block_id="caller-supplied-ob-1",
        lower_bound=99.0,
        upper_bound=101.0,
        timing=fvg_candidate.timing,
    )
    result = evaluate_core_v1_order_block(inputs, observation=candidate)
    assert result.status is ZoneStatus.RESOLVED
    assert result.observation is candidate
    assert result.observation.subject is current
    assert result.observation.timing.source_observations == fvg_candidate.timing.source_observations
    assert not hasattr(result, "entry")
    assert not hasattr(result, "mss")
    assert not hasattr(result, "trade")


def test_touching_order_block_does_not_create_entry_or_trade():
    current, inputs, fvg_candidate = setup()
    candidate = CoreV1OrderBlockObservation(
        current,
        "caller-ob",
        99.0,
        101.0,
        fvg_candidate.timing,
    )
    result = evaluate_core_v1_order_block(inputs, observation=candidate)
    assert candidate.lower_bound <= 100.0 <= candidate.upper_bound
    assert result.status is ZoneStatus.RESOLVED
    assert not hasattr(result, "entry")
    assert not hasattr(result, "trade")


def test_premium_order_block_remains_context_without_direction_or_entry():
    current, inputs, fvg_candidate = setup()
    candidate = CoreV1OrderBlockObservation(
        current,
        "premium-ob",
        115.0,
        117.0,
        fvg_candidate.timing,
    )
    result = evaluate_core_v1_order_block(inputs, observation=candidate)

    assert candidate.lower_bound > 110.0
    assert result.status is ZoneStatus.RESOLVED
    assert result.observation is candidate
    assert not hasattr(result, "direction")
    assert not hasattr(result, "entry")
    assert not hasattr(result, "trade")


def test_future_order_block_observation_is_rejected_by_base_bar_cutoff():
    current, inputs, fvg_candidate = setup()
    future_source = source("future-ob", "2026-01-01T10:06:00Z")
    timing = EvidenceTiming(
        event_timestamp=future_source.event_timestamp,
        availability_timestamp=future_source.availability_timestamp,
        source_observations=(future_source,),
        base_bar_scoped=False,
    )
    with pytest.raises(ValueError, match="base-bar"):
        candidate = CoreV1OrderBlockObservation(
            current,
            "future-ob",
            99.0,
            101.0,
            timing,
        )
        evaluate_core_v1_order_block(inputs, observation=candidate)


def test_future_sentinel_cannot_be_added_to_an_earlier_subject():
    current, inputs, candidate = setup()
    baseline = evaluate_core_v1_fvg(inputs, observation=candidate)
    sentinel = source(
        "future-sentinel",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
    )
    sentinel_observation = SubjectVisibleObservation(
        value={"source_id": sentinel.source_id, "price": 1_000_000.0},
        timing=EvidenceTiming.from_inputs(
            event_timestamp=sentinel.event_timestamp,
            source_observations=(sentinel,),
            base_bar_scoped=False,
        ),
        scope=ObservationScope.CONTEXT,
    )
    with pytest.raises(ValueError, match="unavailable"):
        SubjectVisibleInputs(
            current,
            current.availability_timestamp,
            inputs.base_observations,
            (sentinel_observation,),
        )
    assert baseline.observation is candidate
    assert all(
        source_item.source_id != sentinel.source_id
        for source_item in baseline.observation.timing.source_observations
    )


def test_public_contract_rejects_unrestricted_full_frame_inputs():
    frame = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.0]})
    with pytest.raises(TypeError, match="SubjectVisibleInputs"):
        evaluate_core_v1_fvg(frame)  # type: ignore[arg-type]
    current, _, candidate = setup()
    with pytest.raises(TypeError, match="individual observations"):
        SubjectVisibleObservation(
            value=frame,
            timing=candidate.timing,
            scope=ObservationScope.BASE,
        )
    with pytest.raises(TypeError, match="SubjectVisibleInputs"):
        evaluate_core_v1_order_block(frame)  # type: ignore[arg-type]
    _, inputs, candidate = setup()
    fvg_result = evaluate_core_v1_fvg(inputs, observation=candidate)
    with pytest.raises(TypeError, match="SubjectVisibleInputs"):
        evaluate_core_v1_pd_array(frame, fvg_result)  # type: ignore[arg-type]


def test_future_confirmation_is_rejected_even_if_event_is_visible():
    current, _, candidate = setup()
    future_confirmation = ts("2026-01-01T10:07:00Z")
    future_fvg = replace(candidate.fvg, confirmation_timestamp=future_confirmation)
    future_timing = EvidenceTiming(
        event_timestamp=candidate.timing.event_timestamp,
        confirmation_timestamp=future_confirmation,
        availability_timestamp=future_confirmation,
        source_observations=candidate.timing.source_observations
        + (source("future-confirmation", "2026-01-01T10:07:00Z"),),
        base_bar_scoped=False,
    )
    with pytest.raises(ValueError):
        CoreV1FVGObservation(current, future_fvg, future_timing)


def test_fvg_confirmation_timestamp_cannot_be_omitted_when_delayed():
    current, _, candidate = setup()
    later = replace(
        candidate.fvg,
        confirmation_timestamp=ts("2026-01-01T10:05:30Z"),
    )
    timing_without_confirmation = replace(candidate.timing, confirmation_timestamp=None)
    with pytest.raises(ValueError, match="preserve its later confirmation"):
        CoreV1FVGObservation(current, later, timing_without_confirmation)


def test_observation_results_do_not_emit_signal_fields():
    current, inputs, candidate = setup()
    fvg = evaluate_core_v1_fvg(inputs, observation=candidate)
    pd_array = evaluate_core_v1_pd_array(inputs, fvg)
    ob = evaluate_core_v1_order_block(inputs)

    for result in (fvg, pd_array, ob):
        assert not hasattr(result, "direction")
        assert not hasattr(result, "entry")
        assert not hasattr(result, "trade")
        assert not hasattr(result, "mss")


def test_resolvers_only_receive_timed_visible_inputs():
    current, inputs, _ = setup()

    class Resolver:
        received = None

        def resolve(self, supplied):
            self.received = supplied
            return None

    resolver = Resolver()
    result = evaluate_core_v1_fvg(inputs, resolver=resolver)
    assert result.status is ZoneStatus.UNRESOLVED
    assert resolver.received is inputs
    assert resolver.received.subject is current
    assert all(
        not isinstance(item.value, (pd.DataFrame, pd.Series))
        for item in resolver.received.base_observations
    )
