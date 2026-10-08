from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from src.core_v1_evidence import (
    CoreV1DetectorResult,
    CoreV1EvidenceEnvelope,
    CoreV1EvidenceProvider,
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
    UnavailableCoreV1DetectorBundle,
)
from src.evidence_timing import (
    EvidenceSourceObservation,
    EvidenceTiming,
    SubjectEvidence,
)
from src.ict2022_engine import SetupState
from src.replay_subject import ReplaySubject
from src.strategy_adapter import StrategyEvidence


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
    available: str,
    *,
    base: str | None = None,
    is_base: bool = False,
) -> EvidenceSourceObservation:
    return EvidenceSourceObservation(
        source_id=source_id,
        source="synthetic",
        event_timestamp=ts(event),
        availability_timestamp=ts(available),
        provenance={"row_id": source_id},
        base_bar_timestamp=ts(base) if base is not None else None,
        is_base_timeframe=is_base,
    )


def visible(
    source_observations: tuple[EvidenceSourceObservation, ...],
    *,
    event: str,
    confirmation: str | None = None,
    available: str,
    scope: ObservationScope = ObservationScope.BASE,
) -> SubjectVisibleObservation:
    timing = EvidenceTiming(
        event_timestamp=ts(event),
        confirmation_timestamp=ts(confirmation) if confirmation else None,
        availability_timestamp=ts(available),
        source_observations=source_observations,
        base_bar_scoped=scope is ObservationScope.BASE,
    )
    return SubjectVisibleObservation(
        value={"observation": "synthetic"},
        timing=timing,
        scope=scope,
    )


def inputs(
    *,
    base: tuple[SubjectVisibleObservation, ...] = (),
    context: tuple[SubjectVisibleObservation, ...] = (),
    replay_subject: ReplaySubject | None = None,
) -> SubjectVisibleInputs:
    replay_subject = replay_subject or subject()
    return SubjectVisibleInputs(
        subject=replay_subject,
        availability_timestamp=replay_subject.availability_timestamp,
        base_observations=base,
        context_observations=context,
    )


def test_exact_subject_availability_is_required():
    current = subject()
    with pytest.raises(ValueError, match="exactly match"):
        SubjectVisibleInputs(
            subject=current,
            availability_timestamp=ts("2026-01-01T10:06:00Z"),
            base_observations=(),
            context_observations=(),
        )


def test_subject_availability_cutoff_rejects_late_input():
    late = visible(
        (source(
            "late",
            "2026-01-01T10:06:20Z",
            "2026-01-01T10:06:20Z",
        ),),
        event="2026-01-01T10:06:20Z",
        available="2026-01-01T10:06:20Z",
        scope=ObservationScope.CONTEXT,
    )
    with pytest.raises(ValueError, match="unavailable"):
        inputs(context=(late,))


def test_base_bar_cutoff_is_separate_from_availability_cutoff():
    later_base_event = visible(
        (source(
            "later-base",
            "2026-01-01T10:06:00Z",
            "2026-01-01T10:06:00Z",
            base="2026-01-01T10:06:00Z",
            is_base=True,
        ),),
        event="2026-01-01T10:06:00Z",
        available="2026-01-01T10:06:00Z",
    )
    with pytest.raises(ValueError, match="base-bar"):
        inputs(base=(later_base_event,))


def test_future_observation_is_rejected_even_when_derived_event_is_earlier():
    future_source = source(
        "future",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:00Z",
    )
    item = visible(
        (future_source,),
        event="2026-01-01T10:05:00Z",
        available="2026-01-01T10:07:00Z",
        scope=ObservationScope.CONTEXT,
    )
    with pytest.raises(ValueError, match="unavailable"):
        inputs(context=(item,))


def test_future_confirmation_requires_timed_input_and_is_rejected():
    early_input = source(
        "event",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:05:00Z",
    )
    with pytest.raises(ValueError, match="availability cannot precede confirmation"):
        visible(
            (early_input,),
            event="2026-01-01T10:05:00Z",
            confirmation="2026-01-01T10:07:00Z",
            available="2026-01-01T10:05:00Z",
        )

    future_confirmation = source(
        "future-confirmation",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
    )
    confirmed = visible(
        (early_input, future_confirmation),
        event="2026-01-01T10:05:00Z",
        confirmation="2026-01-01T10:07:00Z",
        available="2026-01-01T10:07:17Z",
        scope=ObservationScope.CONTEXT,
    )
    with pytest.raises(ValueError, match="unavailable"):
        inputs(context=(confirmed,))


def test_multi_input_availability_is_maximum_required_input_time():
    first = source(
        "first",
        "2026-01-01T10:04:00Z",
        "2026-01-01T10:04:01Z",
    )
    second = source(
        "second",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:10Z",
    )
    derived = EvidenceTiming.from_inputs(
        event_timestamp=ts("2026-01-01T10:05:00Z"),
        source_observations=(first, second),
        provenance={"derivation": "synthetic"},
    )
    assert derived.availability_timestamp == ts("2026-01-01T10:06:10Z")


def test_derived_timing_preserves_event_confirmation_and_availability():
    first = source(
        "event",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:05:00Z",
    )
    confirmation = source(
        "confirmation",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:17Z",
    )
    derived = EvidenceTiming.from_inputs(
        event_timestamp=first.event_timestamp,
        confirmation_timestamp=confirmation.event_timestamp,
        source_observations=(first, confirmation),
        provenance={"derivation": "test"},
    )
    assert derived.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert derived.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert derived.availability_timestamp == ts("2026-01-01T10:06:17Z")


def test_provenance_survives_through_subject_evidence_and_result():
    current = subject()
    original = source(
        "row-1",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:05:00Z",
    )
    derived = EvidenceTiming.from_inputs(
        event_timestamp=original.event_timestamp,
        source_observations=(original,),
        provenance={"detector": "placeholder-test"},
        base_bar_scoped=True,
    )
    result = CoreV1DetectorResult(
        subject=current,
        fields={
            "direction": SubjectEvidence(
                subject=current,
                timing=derived,
                value="not interpreted by this test",
            )
        },
    )
    assert result.fields["direction"].timing.source_observations == (original,)
    assert result.fields["direction"].timing.provenance["detector"] == "placeholder-test"


def test_unrestricted_frames_and_series_are_rejected_at_the_payload_boundary():
    stamp = ts("2026-01-01T10:05:00Z")
    timing = EvidenceTiming(
        event_timestamp=stamp,
        availability_timestamp=stamp,
        source_observations=(source("row", stamp.isoformat(), stamp.isoformat()),),
    )
    with pytest.raises(TypeError, match="frames or series"):
        SubjectVisibleObservation(
            value=pd.DataFrame({"close": [1.0]}),
            timing=timing,
            scope=ObservationScope.CONTEXT,
        )
    with pytest.raises(TypeError, match="frames or series"):
        SubjectVisibleObservation(
            value=pd.Series([1.0]),
            timing=timing,
            scope=ObservationScope.CONTEXT,
        )


def test_nested_frames_and_series_are_rejected_at_the_payload_boundary():
    stamp = ts("2026-01-01T10:05:00Z")
    timing = EvidenceTiming(
        event_timestamp=stamp,
        availability_timestamp=stamp,
        source_observations=(source("row", stamp.isoformat(), stamp.isoformat()),),
    )
    for payload in (
        {"nested": pd.DataFrame({"close": [1.0]})},
        {"nested": {"values": pd.Series([1.0])}},
    ):
        with pytest.raises(TypeError, match="frames or series"):
            SubjectVisibleObservation(
                value=payload,
                timing=timing,
                scope=ObservationScope.CONTEXT,
            )


def test_nested_frame_is_rejected_from_detector_result_evidence_too():
    current = subject()
    stamp = current.base_bar_timestamp
    timing = EvidenceTiming(
        event_timestamp=stamp,
        availability_timestamp=stamp,
        source_observations=(
            source("row", stamp.isoformat(), stamp.isoformat()),
        ),
        base_bar_scoped=True,
    )

    with pytest.raises(TypeError, match="frames or series"):
        CoreV1DetectorResult(
            subject=current,
            fields={
                "direction": SubjectEvidence(
                    subject=current,
                    timing=timing,
                    value={"nested": {"frame": pd.DataFrame({"x": [1]})}},
                )
            },
        )


def test_scalar_and_structured_observation_payload_remains_valid():
    stamp = ts("2026-01-01T10:05:00Z")
    observation = SubjectVisibleObservation(
        value={
            "close": 101.25,
            "timestamp": stamp,
            "lineage": {"source_id": "bar-1", "labels": ("canonical", "base")},
        },
        timing=EvidenceTiming(
            event_timestamp=stamp,
            availability_timestamp=stamp,
            source_observations=(source("row", stamp.isoformat(), stamp.isoformat()),),
        ),
        scope=ObservationScope.CONTEXT,
    )
    assert observation.value["lineage"]["labels"] == ("canonical", "base")


@pytest.mark.parametrize(
    "provenance",
    [
        {"frame": pd.DataFrame({"close": [1.0]})},
        {"series": pd.Series([1.0])},
        {"nested": {"frame": pd.DataFrame({"close": [1.0]})}},
        {"nested": ({"frame": pd.DataFrame({"close": [1.0]})},)},
        {"nested": [{"frame": pd.DataFrame({"close": [1.0]})}]},
    ],
)
def test_timing_provenance_rejects_nested_frames_and_series(provenance):
    stamp = ts("2026-01-01T10:05:00Z")
    with pytest.raises(TypeError, match="frames or series"):
        EvidenceSourceObservation(
            source_id="bounded-source",
            source="synthetic",
            event_timestamp=stamp,
            availability_timestamp=stamp,
            provenance=provenance,
        )


def test_timing_provenance_rejects_unsupported_containers():
    stamp = ts("2026-01-01T10:05:00Z")
    with pytest.raises(TypeError, match="Unsupported provenance payload type"):
        EvidenceSourceObservation(
            source_id="bounded-source",
            source="synthetic",
            event_timestamp=stamp,
            availability_timestamp=stamp,
            provenance={"opaque": SimpleNamespace(frame=pd.DataFrame({"x": [1]}))},
        )


def test_timing_provenance_is_deeply_frozen_and_keeps_valid_tick_metadata():
    stamp = ts("2026-01-01T10:05:00Z")
    provenance = {
        "source_file": "canonical.csv",
        "source_row": 2,
        "canonical_ticks": (
            {
                "timestamp": stamp,
                "source_order": 1,
                "provenance": {"source_file": "canonical.csv"},
            },
        ),
    }
    observation = EvidenceSourceObservation(
        source_id="canonical-bar",
        source="canonical-quote-ticks",
        event_timestamp=stamp,
        availability_timestamp=stamp,
        provenance=provenance,
    )

    assert observation.provenance["canonical_ticks"][0]["source_order"] == 1
    with pytest.raises(TypeError):
        observation.provenance["canonical_ticks"][0]["provenance"]["new"] = True


def test_unsupported_container_payload_is_rejected():
    stamp = ts("2026-01-01T10:05:00Z")
    with pytest.raises(TypeError, match="Unsupported observation payload type"):
        SubjectVisibleObservation(
            value={"history": {100.0}},
            timing=EvidenceTiming(
                event_timestamp=stamp,
                availability_timestamp=stamp,
                source_observations=(
                    source("row", stamp.isoformat(), stamp.isoformat()),
                ),
            ),
            scope=ObservationScope.CONTEXT,
        )


def test_mutable_observation_payload_is_deeply_snapshotted():
    stamp = ts("2026-01-01T10:05:00Z")
    payload = {
        "nested": {"rows": [{"value": 1}]},
        "labels": ("canonical", {"name": "base"}),
    }
    observation = SubjectVisibleObservation(
        value=payload,
        timing=EvidenceTiming(
            event_timestamp=stamp,
            availability_timestamp=stamp,
            source_observations=(source("row", stamp.isoformat(), stamp.isoformat()),),
        ),
        scope=ObservationScope.CONTEXT,
    )

    payload["nested"]["rows"][0]["value"] = 2
    payload["nested"]["rows"].append(pd.DataFrame({"close": [1.0]}))
    payload["labels"][1]["name"] = "mutated"
    payload["new"] = pd.Series([1.0])

    assert observation.value["nested"]["rows"] == ({"value": 1},)
    assert observation.value["labels"][1]["name"] == "base"
    assert "new" not in observation.value
    with pytest.raises(TypeError):
        observation.value["nested"]["rows"][0]["value"] = 3


def test_post_construction_mutation_cannot_reach_detector():
    current = subject()
    stamp = current.base_bar_timestamp
    source_observation = source(
        "base-row",
        stamp.isoformat(),
        current.availability_timestamp.isoformat(),
        base=stamp.isoformat(),
        is_base=True,
    )
    timing = EvidenceTiming(
        event_timestamp=stamp,
        availability_timestamp=current.availability_timestamp,
        source_observations=(source_observation,),
        base_bar_scoped=True,
    )
    payload = {"items": [{"value": "original"}]}
    observation = SubjectVisibleObservation(
        value=payload,
        timing=timing,
        scope=ObservationScope.BASE,
    )
    payload["items"][0]["value"] = "mutated"
    payload["items"].append(pd.DataFrame({"close": [1.0]}))
    detector = EchoDetector()

    CoreV1EvidenceProvider(detector).build(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=(observation,),
        context_observations=(),
    )

    assert detector.received is not None
    assert detector.received.base_observations[0].value["items"] == (
        {"value": "original"},
    )


def test_detector_result_and_envelope_store_canonical_field_values():
    current = subject()
    stamp = current.base_bar_timestamp
    source_observation = source(
        "visible-row",
        stamp.isoformat(),
        current.availability_timestamp.isoformat(),
        base=stamp.isoformat(),
        is_base=True,
    )
    timing = EvidenceTiming(
        event_timestamp=stamp,
        availability_timestamp=current.availability_timestamp,
        source_observations=(source_observation,),
        base_bar_scoped=True,
    )
    payload = {"details": [{"value": "before"}]}
    evidence = SubjectEvidence(current, timing, payload)
    result = CoreV1DetectorResult(
        subject=current,
        fields={"entry_price": evidence},
    )
    payload["details"][0]["value"] = "after-result"

    class StaticDetector:
        def detect(self, inputs: SubjectVisibleInputs) -> CoreV1DetectorResult:
            return result

    input_observation = SubjectVisibleObservation(
        value={"source_id": source_observation.source_id},
        timing=timing,
        scope=ObservationScope.BASE,
    )
    envelope = CoreV1EvidenceProvider(StaticDetector()).build(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=(input_observation,),
        context_observations=(),
    )

    assert result.fields["entry_price"].value["details"][0]["value"] == "before"
    assert envelope.field_evidence["entry_price"].value["details"][0]["value"] == "before"
    assert envelope.strategy_evidence.entry_price["details"][0]["value"] == "before"


def test_envelope_snapshots_mutable_strategy_evidence_fields():
    current = subject()
    payload = {"details": [{"value": "before"}]}
    strategy_evidence = StrategyEvidence(
        timestamp=current.availability_timestamp,
        direction=None,
        dealing_range=None,
        draw_on_liquidity=None,
        liquidity_event=None,
        mss=None,
        pd_array=None,
        entry_price=payload,
        invalidation_price=None,
        target_price=None,
        target_liquidity=None,
        replay_subject=current,
    )
    envelope = CoreV1EvidenceEnvelope(
        subject=current,
        strategy_evidence=strategy_evidence,
        field_evidence={},
    )

    payload["details"][0]["value"] = "after"
    assert envelope.strategy_evidence.entry_price["details"][0]["value"] == "before"
    assert strategy_evidence.entry_price["details"][0]["value"] == "after"


def test_observation_metadata_rejects_oversized_keys():
    stamp = ts("2026-01-01T10:05:00Z")
    timing = EvidenceTiming(
        event_timestamp=stamp,
        availability_timestamp=stamp,
        source_observations=(source("row", stamp.isoformat(), stamp.isoformat()),),
    )
    with pytest.raises(TypeError, match="mapping keys"):
        SubjectVisibleObservation(
            value={"k" * 4097: "value"},
            timing=timing,
            scope=ObservationScope.CONTEXT,
        )


def test_replay_subject_identity_is_preserved_through_provider_and_adapter():
    current = subject()
    provider = CoreV1EvidenceProvider()
    envelope = provider.build(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=(),
        context_observations=(),
    )
    decision = provider.adapter.evaluate(envelope.strategy_evidence)
    assert envelope.subject is current
    assert envelope.strategy_evidence.replay_subject is current
    assert decision.replay_subject is current
    assert envelope.strategy_evidence.timestamp == current.availability_timestamp


def test_provider_preserves_replay_subject_value_identity_contract():
    current = subject()
    reconstructed = ReplaySubject.from_dict(current.to_dict())

    class ReconstructedSubjectDetector:
        def detect(self, inputs: SubjectVisibleInputs) -> CoreV1DetectorResult:
            return CoreV1DetectorResult(subject=reconstructed, fields={})

    envelope = CoreV1EvidenceProvider(ReconstructedSubjectDetector()).build(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=(),
        context_observations=(),
    )

    assert reconstructed is not current
    assert reconstructed == current
    assert envelope.subject is current


def test_unavailable_detector_fails_closed_without_fabricating_output():
    provider = CoreV1EvidenceProvider(UnavailableCoreV1DetectorBundle())
    decision = provider.evaluate(
        subject=subject(),
        availability_timestamp=subject().availability_timestamp,
        base_observations=(),
        context_observations=(),
    )
    assert decision.state is SetupState.DEVELOPING
    assert decision.direction is None
    assert decision.entry_price is None
    assert decision.invalidation_price is None
    assert decision.target_price is None


@dataclass
class EchoDetector:
    received: SubjectVisibleInputs | None = None

    def detect(self, inputs: SubjectVisibleInputs) -> CoreV1DetectorResult:
        self.received = inputs
        return CoreV1DetectorResult(subject=inputs.subject, fields={})


def test_provider_passes_only_validated_subject_observation_contract():
    current = subject()
    item = visible(
        (source(
            "base-row",
            "2026-01-01T10:05:00Z",
            "2026-01-01T10:06:17Z",
            base="2026-01-01T10:05:00Z",
            is_base=True,
        ),),
        event="2026-01-01T10:05:00Z",
        available="2026-01-01T10:06:17Z",
    )
    detector = EchoDetector()
    CoreV1EvidenceProvider(detector).build(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=(item,),
        context_observations=(),
    )
    assert detector.received is not None
    assert detector.received.subject is current
    assert detector.received.base_observations == (item,)


def test_provider_envelope_preserves_field_timing_and_provenance():
    current = subject()
    input_source = source(
        "derived-from",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:17Z",
        base="2026-01-01T10:05:00Z",
        is_base=True,
    )
    field_timing = EvidenceTiming.from_inputs(
        event_timestamp=input_source.event_timestamp,
        source_observations=(input_source,),
        provenance={"detector": "synthetic"},
        base_bar_scoped=True,
    )
    input_observation = visible(
        (input_source,),
        event="2026-01-01T10:05:00Z",
        available="2026-01-01T10:06:17Z",
    )

    class TimedValueDetector:
        def detect(self, visible_inputs: SubjectVisibleInputs) -> CoreV1DetectorResult:
            return CoreV1DetectorResult(
                subject=visible_inputs.subject,
                fields={
                    "entry_price": SubjectEvidence(
                        subject=visible_inputs.subject,
                        timing=field_timing,
                        value=101.0,
                    )
                },
            )

    envelope = CoreV1EvidenceProvider(TimedValueDetector()).build(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=(input_observation,),
        context_observations=(),
    )
    assert envelope.strategy_evidence.entry_price == 101.0
    preserved = envelope.field_evidence["entry_price"].timing
    assert preserved.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert preserved.availability_timestamp == current.availability_timestamp
    assert preserved.source_observations == (input_source,)
    assert preserved.provenance["detector"] == "synthetic"


def test_detector_cannot_return_provenance_outside_visible_inputs():
    current = subject()
    unprovided = source(
        "unprovided",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:05:00Z",
        base="2026-01-01T10:05:00Z",
        is_base=True,
    )
    result_timing = EvidenceTiming.from_inputs(
        event_timestamp=unprovided.event_timestamp,
        source_observations=(unprovided,),
        base_bar_scoped=True,
    )

    class UnprovidedLineageDetector:
        def detect(self, visible_inputs: SubjectVisibleInputs) -> CoreV1DetectorResult:
            return CoreV1DetectorResult(
                subject=visible_inputs.subject,
                fields={
                    "entry_price": SubjectEvidence(
                        subject=visible_inputs.subject,
                        timing=result_timing,
                        value=101.0,
                    )
                },
            )

    with pytest.raises(ValueError, match="outside the visible inputs"):
        CoreV1EvidenceProvider(UnprovidedLineageDetector()).build(
            subject=current,
            availability_timestamp=current.availability_timestamp,
            base_observations=(),
            context_observations=(),
        )
