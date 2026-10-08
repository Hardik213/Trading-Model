from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol

import pandas as pd

from .evidence_timing import (
    EvidenceTiming,
    SubjectEvidence,
    validate_bounded_payload,
)
from .replay_subject import ReplaySubject, same_subject_id
from .strategy_adapter import AdapterDecision, ICT2022StrategyAdapter, StrategyEvidence


class ObservationScope(Enum):
    BASE = "base"
    CONTEXT = "context"


def _validate_observation_value(value: object) -> object:
    """Accept only bounded scalar/structured snapshots, never arbitrary payloads."""
    return validate_bounded_payload(value, "Observation payload")


@dataclass(frozen=True)
class SubjectVisibleObservation:
    """One source observation with explicit timing and lineage, never a frame."""

    value: object
    timing: EvidenceTiming
    scope: ObservationScope

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_observation_value(self.value))
        if not isinstance(self.scope, ObservationScope):
            raise TypeError("scope must be an ObservationScope.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Visible inputs require EvidenceTiming provenance.")


@dataclass(frozen=True)
class SubjectVisibleInputs:
    """Base and context observations visible for one exact replay subject."""

    subject: ReplaySubject
    availability_timestamp: pd.Timestamp
    base_observations: tuple[SubjectVisibleObservation, ...]
    context_observations: tuple[SubjectVisibleObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        availability = _utc_timestamp(
            self.availability_timestamp,
            "availability_timestamp",
        )
        if availability != self.subject.availability_timestamp:
            raise ValueError(
                "availability_timestamp must exactly match the ReplaySubject."
            )
        object.__setattr__(self, "availability_timestamp", availability)

        base_observations = tuple(self.base_observations)
        context_observations = tuple(self.context_observations)
        object.__setattr__(self, "base_observations", base_observations)
        object.__setattr__(self, "context_observations", context_observations)

        self._validate_observations(base_observations, ObservationScope.BASE)
        self._validate_observations(context_observations, ObservationScope.CONTEXT)

    def _validate_observations(
        self,
        observations: tuple[SubjectVisibleObservation, ...],
        expected_scope: ObservationScope,
    ) -> None:
        for observation in observations:
            if not isinstance(observation, SubjectVisibleObservation):
                raise TypeError(
                    "Visible inputs must contain SubjectVisibleObservation values."
                )
            if observation.scope is not expected_scope:
                raise ValueError(
                    f"{expected_scope.value} inputs contain an observation with "
                    f"{observation.scope.value} scope."
                )
            observation.timing.validate_for_subject(self.subject)
            if expected_scope is ObservationScope.BASE:
                if not observation.timing.base_bar_scoped:
                    raise ValueError("Base observations must be base-bar scoped.")
                base_sources = tuple(
                    source
                    for source in observation.timing.source_observations
                    if source.is_base_timeframe
                )
                if not base_sources:
                    raise ValueError(
                        "Base observations require base-timeframe source provenance."
                    )
                if any(
                    source.base_bar_timestamp > self.subject.base_bar_timestamp
                    for source in base_sources
                ):
                    raise ValueError(
                        "Base observation source exceeds the ReplaySubject base-bar cutoff."
                    )


_STRATEGY_FIELDS = (
    "direction",
    "dealing_range",
    "draw_on_liquidity",
    "liquidity_event",
    "mss",
    "pd_array",
    "entry_price",
    "invalidation_price",
    "target_price",
    "target_liquidity",
)


@dataclass(frozen=True)
class CoreV1DetectorResult:
    """Only timed, subject-bound values may populate strategy evidence."""

    subject: ReplaySubject
    fields: Mapping[str, SubjectEvidence[object]]

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.fields, Mapping):
            raise TypeError("fields must be a mapping.")
        unknown = set(self.fields) - set(_STRATEGY_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported StrategyEvidence fields: {sorted(unknown)}")
        copied = dict(self.fields)
        for name, evidence in copied.items():
            if name not in _STRATEGY_FIELDS:
                raise ValueError(f"Unsupported StrategyEvidence field: {name}")
            if not isinstance(evidence, SubjectEvidence):
                raise TypeError(f"{name} must be wrapped in SubjectEvidence.")
            if not same_subject_id(evidence.subject, self.subject):
                raise ValueError(f"{name} belongs to a different ReplaySubject.")
            copied[name] = SubjectEvidence(
                subject=evidence.subject,
                timing=evidence.timing,
                value=_validate_observation_value(evidence.value),
            )
            evidence.timing.validate_for_subject(self.subject)
        object.__setattr__(self, "fields", MappingProxyType(copied))


@dataclass(frozen=True)
class CoreV1EvidenceEnvelope:
    """StrategyEvidence plus the per-field timing/provenance it cannot store."""

    subject: ReplaySubject
    strategy_evidence: StrategyEvidence
    field_evidence: Mapping[str, SubjectEvidence[object]]

    def __post_init__(self) -> None:
        if not same_subject_id(self.strategy_evidence.replay_subject, self.subject):
            raise ValueError("StrategyEvidence belongs to a different ReplaySubject.")
        if self.strategy_evidence.timestamp != self.subject.availability_timestamp:
            raise ValueError("StrategyEvidence timestamp must equal subject availability.")
        if not isinstance(self.field_evidence, Mapping):
            raise TypeError("field_evidence must be a mapping.")
        copied = dict(self.field_evidence)
        for name, evidence in copied.items():
            if name not in _STRATEGY_FIELDS:
                raise ValueError(f"Unsupported StrategyEvidence field: {name}")
            if not isinstance(evidence, SubjectEvidence):
                raise TypeError(f"{name} must be wrapped in SubjectEvidence.")
            if not same_subject_id(evidence.subject, self.subject):
                raise ValueError(f"{name} belongs to a different ReplaySubject.")
            copied[name] = SubjectEvidence(
                subject=evidence.subject,
                timing=evidence.timing,
                value=_validate_observation_value(evidence.value),
            )
            evidence.timing.validate_for_subject(self.subject)
        object.__setattr__(self, "field_evidence", MappingProxyType(copied))
        strategy_values = {
            name: (
                None
                if getattr(self.strategy_evidence, name) is None
                else _validate_observation_value(
                    getattr(self.strategy_evidence, name)
                )
            )
            for name in _STRATEGY_FIELDS
        }
        object.__setattr__(
            self,
            "strategy_evidence",
            replace(self.strategy_evidence, **strategy_values),
        )


class CoreV1DetectorBundle(Protocol):
    """Governed detector seam; implementations must not receive source frames."""

    def detect(self, inputs: SubjectVisibleInputs) -> CoreV1DetectorResult: ...


class UnavailableCoreV1DetectorBundle:
    """Explicit fail-closed placeholder until governed detectors are implemented."""

    def detect(self, inputs: SubjectVisibleInputs) -> CoreV1DetectorResult:
        if not isinstance(inputs, SubjectVisibleInputs):
            raise TypeError("inputs must be SubjectVisibleInputs.")
        return CoreV1DetectorResult(subject=inputs.subject, fields={})


class CoreV1EvidenceProvider:
    """Validates visible inputs and adapts timed detector results to StrategyEvidence."""

    def __init__(
        self,
        detectors: CoreV1DetectorBundle | None = None,
        adapter: ICT2022StrategyAdapter | None = None,
    ) -> None:
        self.detectors = detectors or UnavailableCoreV1DetectorBundle()
        self.adapter = adapter or ICT2022StrategyAdapter()

    def build(
        self,
        *,
        subject: ReplaySubject,
        availability_timestamp: object,
        base_observations: tuple[SubjectVisibleObservation, ...],
        context_observations: tuple[SubjectVisibleObservation, ...],
    ) -> CoreV1EvidenceEnvelope:
        inputs = SubjectVisibleInputs(
            subject=subject,
            availability_timestamp=_utc_timestamp(
                availability_timestamp,
                "availability_timestamp",
            ),
            base_observations=base_observations,
            context_observations=context_observations,
        )
        result = self.detectors.detect(inputs)
        if not isinstance(result, CoreV1DetectorResult):
            raise TypeError("Detector bundle must return CoreV1DetectorResult.")
        if not same_subject_id(result.subject, subject):
            raise ValueError("Detector bundle returned a result for another subject.")

        visible_sources = tuple(
            source
            for observation in (
                inputs.base_observations + inputs.context_observations
            )
            for source in observation.timing.source_observations
        )
        values = {name: None for name in _STRATEGY_FIELDS}
        for name, timed_value in result.fields.items():
            timed_value.timing.validate_for_subject(subject)
            if any(
                source not in visible_sources
                for source in timed_value.timing.source_observations
            ):
                raise ValueError(
                    f"{name} provenance references observations outside the visible inputs."
                )
            values[name] = timed_value.value
        strategy_evidence = StrategyEvidence(
            timestamp=subject.availability_timestamp,
            replay_subject=subject,
            **values,
        )
        return CoreV1EvidenceEnvelope(
            subject=subject,
            strategy_evidence=strategy_evidence,
            field_evidence=result.fields,
        )

    def evaluate(
        self,
        *,
        subject: ReplaySubject,
        availability_timestamp: object,
        base_observations: tuple[SubjectVisibleObservation, ...],
        context_observations: tuple[SubjectVisibleObservation, ...],
    ) -> AdapterDecision:
        envelope = self.build(
            subject=subject,
            availability_timestamp=availability_timestamp,
            base_observations=base_observations,
            context_observations=context_observations,
        )
        return self.adapter.evaluate(envelope.strategy_evidence)


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


__all__ = [
    "CoreV1DetectorBundle",
    "CoreV1DetectorResult",
    "CoreV1EvidenceEnvelope",
    "CoreV1EvidenceProvider",
    "ObservationScope",
    "SubjectVisibleInputs",
    "SubjectVisibleObservation",
    "UnavailableCoreV1DetectorBundle",
]
