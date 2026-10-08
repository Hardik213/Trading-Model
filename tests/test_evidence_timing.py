from __future__ import annotations

import pandas as pd
import pytest

from src.evidence_timing import (
    EvidenceSourceObservation,
    EvidenceTiming,
    MAX_BOUNDED_STRING_LENGTH,
    SubjectEvidence,
)
from src.replay_subject import ReplaySubject


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def source(
    source_id: str,
    event: str,
    available: str,
    *,
    base: str | None = None,
    base_timeframe: bool = False,
    provenance: dict[str, object] | None = None,
) -> EvidenceSourceObservation:
    return EvidenceSourceObservation(
        source_id=source_id,
        source="synthetic",
        event_timestamp=ts(event),
        availability_timestamp=ts(available),
        provenance=provenance or {"row_id": source_id},
        base_bar_timestamp=ts(base) if base else None,
        is_base_timeframe=base_timeframe,
    )


def timing(
    *,
    event: str = "2026-01-01T10:05:00Z",
    confirmation: str | None = None,
    available: str = "2026-01-01T10:05:00Z",
    observations: tuple[EvidenceSourceObservation, ...] | None = None,
    base_bar_scoped: bool = False,
) -> EvidenceTiming:
    inputs = observations or (
        source("input-1", event, available),
    )
    return EvidenceTiming(
        event_timestamp=ts(event),
        confirmation_timestamp=ts(confirmation) if confirmation else None,
        availability_timestamp=ts(available),
        source_observations=inputs,
        provenance={"detector": "synthetic-test"},
        base_bar_scoped=base_bar_scoped,
    )


def test_event_timestamp_can_precede_delayed_availability():
    item = source(
        "bar-1",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:17Z",
        base="2026-01-01T10:05:00Z",
        base_timeframe=True,
    )
    evidence_timing = EvidenceTiming.from_inputs(
        event_timestamp=item.event_timestamp,
        source_observations=(item,),
        base_bar_scoped=True,
    )

    assert evidence_timing.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert evidence_timing.availability_timestamp == ts("2026-01-01T10:06:17Z")


def test_confirmation_timestamp_can_be_later_than_event_timestamp():
    item = source(
        "impulse",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:05:00Z",
    )
    confirmation = source(
        "follow-through",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:00Z",
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=item.event_timestamp,
        confirmation_timestamp=confirmation.event_timestamp,
        source_observations=(item, confirmation),
    )

    assert result.event_timestamp == ts("2026-01-01T10:05:00Z")
    assert result.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert result.availability_timestamp == ts("2026-01-01T10:06:00Z")


def test_availability_can_be_later_than_confirmation():
    item = source(
        "delayed-confirmation-bar",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:17Z",
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=ts("2026-01-01T10:04:00Z"),
        confirmation_timestamp=ts("2026-01-01T10:06:00Z"),
        source_observations=(item,),
    )

    assert result.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert result.availability_timestamp == ts("2026-01-01T10:06:17Z")


def test_multi_input_availability_is_latest_input_availability():
    first = source(
        "base",
        "2026-01-01T10:01:00Z",
        "2026-01-01T10:01:05Z",
    )
    second = source(
        "higher-timeframe",
        "2026-01-01T10:00:00Z",
        "2026-01-01T10:03:17Z",
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=ts("2026-01-01T10:03:00Z"),
        source_observations=(first, second),
    )

    assert result.availability_timestamp == ts("2026-01-01T10:03:17Z")


def test_delayed_finalized_bar_uses_finalizing_availability_not_label():
    bar = source(
        "bar-10:05",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:17Z",
        base="2026-01-01T10:05:00Z",
        base_timeframe=True,
        provenance={"finalizing_tick_order": 42},
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=ts("2026-01-01T10:05:00Z"),
        source_observations=(bar,),
        base_bar_scoped=True,
    )

    assert result.availability_timestamp == ts("2026-01-01T10:06:17Z")
    assert result.source_observations[0].provenance["finalizing_tick_order"] == 42


def test_subject_cutoff_rejects_unavailable_evidence():
    future_input = source(
        "late-input",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:17Z",
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=future_input.event_timestamp,
        source_observations=(future_input,),
    )
    subject = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:00Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )

    with pytest.raises(ValueError, match="unavailable"):
        SubjectEvidence(subject, result, value="must not enter strategy evidence")


def test_base_bar_visibility_is_separate_from_availability_cutoff():
    base_input = source(
        "later-base-bar",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:00Z",
        base="2026-01-01T10:06:00Z",
        base_timeframe=True,
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=base_input.event_timestamp,
        source_observations=(base_input,),
        base_bar_scoped=True,
    )
    subject = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:00Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )

    with pytest.raises(ValueError, match="base-bar"):
        SubjectEvidence(subject, result, value="out-of-prefix evidence")


@pytest.mark.parametrize(
    "event",
    ["2026-01-01T10:04:00Z", "2026-01-01T10:05:00Z"],
)
def test_base_timeframe_source_event_at_or_before_cutoff_is_accepted(event):
    current = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )
    observation = source(
        "base-source",
        event,
        "2026-01-01T10:06:17Z",
        base="2026-01-01T10:05:00Z",
        base_timeframe=True,
    )
    result = timing(
        event=event,
        available="2026-01-01T10:06:17Z",
        observations=(observation,),
        base_bar_scoped=False,
    )

    result.validate_for_subject(current)


def test_base_timeframe_source_event_after_cutoff_is_rejected_when_unscoped():
    current = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )
    observation = source(
        "future-base-source",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:17Z",
        base="2026-01-01T10:05:00Z",
        base_timeframe=True,
    )
    result = timing(
        event="2026-01-01T10:06:00Z",
        available="2026-01-01T10:06:17Z",
        observations=(observation,),
        base_bar_scoped=False,
    )

    with pytest.raises(ValueError, match="source event exceeds"):
        result.validate_for_subject(current)


def test_later_confirmation_for_valid_base_event_uses_availability_rules():
    current = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )
    event_source = source(
        "base-event",
        "2026-01-01T10:04:00Z",
        "2026-01-01T10:04:00Z",
        base="2026-01-01T10:04:00Z",
        base_timeframe=True,
    )
    confirmation_source = source(
        "confirmation",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:00Z",
    )
    result = timing(
        event="2026-01-01T10:04:00Z",
        confirmation="2026-01-01T10:06:00Z",
        available="2026-01-01T10:06:00Z",
        observations=(event_source, confirmation_source),
        base_bar_scoped=False,
    )

    result.validate_for_subject(current)


def test_context_source_event_after_base_cutoff_remains_available():
    current = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )
    context_source = source(
        "context-source",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:00Z",
    )
    result = timing(
        event="2026-01-01T10:06:00Z",
        available="2026-01-01T10:06:00Z",
        observations=(context_source,),
        base_bar_scoped=False,
    )

    result.validate_for_subject(current)


def test_future_observation_cannot_satisfy_an_earlier_subject():
    observation = source(
        "future-confirmation",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=ts("2026-01-01T10:05:00Z"),
        confirmation_timestamp=observation.event_timestamp,
        source_observations=(observation,),
    )
    earlier_subject = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:00Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )

    with pytest.raises(ValueError, match="unavailable"):
        result.validate_for_subject(earlier_subject)


def test_provenance_survives_derived_evidence_construction():
    inputs = (
        source(
            "tick-row-1",
            "2026-01-01T10:05:00Z",
            "2026-01-01T10:05:00Z",
            provenance={"file": "ticks.csv", "row": 10},
        ),
        source(
            "tick-row-2",
            "2026-01-01T10:05:01Z",
            "2026-01-01T10:05:01Z",
            provenance={"file": "ticks.csv", "row": 11},
        ),
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=ts("2026-01-01T10:05:01Z"),
        source_observations=inputs,
        provenance={"rule": "synthetic-derived-observation"},
    )

    assert result.source_observations == inputs
    assert result.source_observations[0].provenance["row"] == 10
    assert result.provenance["rule"] == "synthetic-derived-observation"
    with pytest.raises(TypeError):
        result.provenance["rule"] = "mutated"


def test_provenance_mapping_key_length_uses_bounded_string_contract():
    stamp = ts("2026-01-01T10:05:00Z")
    accepted_key = "k" * MAX_BOUNDED_STRING_LENGTH
    accepted = source(
        "accepted-key",
        stamp.isoformat(),
        stamp.isoformat(),
        provenance={accepted_key: "value"},
    )
    assert accepted.provenance[accepted_key] == "value"

    with pytest.raises(TypeError, match="mapping keys"):
        source(
            "oversized-key",
            stamp.isoformat(),
            stamp.isoformat(),
            provenance={"k" * (MAX_BOUNDED_STRING_LENGTH + 1): "value"},
        )


def test_nested_and_timing_provenance_keys_use_the_same_bound():
    stamp = ts("2026-01-01T10:05:00Z")
    with pytest.raises(TypeError, match="mapping keys"):
        source(
            "nested-oversized-key",
            stamp.isoformat(),
            stamp.isoformat(),
            provenance={"nested": {"k" * (MAX_BOUNDED_STRING_LENGTH + 1): True}},
        )

    input_source = source("input", stamp.isoformat(), stamp.isoformat())
    with pytest.raises(TypeError, match="mapping keys"):
        EvidenceTiming(
            event_timestamp=stamp,
            availability_timestamp=stamp,
            source_observations=(input_source,),
            provenance={"k" * (MAX_BOUNDED_STRING_LENGTH + 1): "value"},
        )


def test_confirmation_does_not_backdate_or_mutate_event_timestamp():
    original_event = ts("2026-01-01T10:05:00Z")
    confirm = ts("2026-01-01T10:06:00Z")
    observations = (
        source("event", original_event.isoformat(), original_event.isoformat()),
        source("confirmation", confirm.isoformat(), confirm.isoformat()),
    )
    result = EvidenceTiming.from_inputs(
        event_timestamp=original_event,
        confirmation_timestamp=confirm,
        source_observations=observations,
    )

    assert result.event_timestamp == original_event
    assert result.confirmation_timestamp == confirm
    assert result.availability_timestamp == confirm
    assert observations[0].event_timestamp == original_event


def test_confirmation_label_does_not_substitute_for_input_availability():
    input_item = source(
        "event",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:05:00Z",
    )
    with pytest.raises(ValueError, match="availability cannot precede confirmation"):
        EvidenceTiming.from_inputs(
            event_timestamp=input_item.event_timestamp,
            confirmation_timestamp=ts("2026-01-01T10:06:00Z"),
            source_observations=(input_item,),
        )


def test_explicit_availability_cannot_precede_required_inputs_or_confirmation():
    input_item = source(
        "input",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:06:00Z",
    )
    with pytest.raises(ValueError, match="input availability"):
        EvidenceTiming(
            event_timestamp=ts("2026-01-01T10:05:00Z"),
            availability_timestamp=ts("2026-01-01T10:05:30Z"),
            source_observations=(input_item,),
        )
    with pytest.raises(ValueError, match="confirmation"):
        EvidenceTiming(
            event_timestamp=ts("2026-01-01T10:05:00Z"),
            confirmation_timestamp=ts("2026-01-01T10:07:00Z"),
            availability_timestamp=ts("2026-01-01T10:06:00Z"),
            source_observations=(
                source(
                    "early-input",
                    "2026-01-01T10:05:00Z",
                    "2026-01-01T10:05:00Z",
                ),
            ),
        )
