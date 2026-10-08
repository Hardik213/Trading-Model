from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol

import pandas as pd

from .core_v1_evidence import SubjectVisibleInputs
from .core_v1_liquidity import CoreV1LiquidityAssessment
from .displacement import Direction
from .evidence_timing import EvidenceSourceObservation, EvidenceTiming
from .replay_subject import ReplaySubject, same_subject_id


class DisplacementStatus(str, Enum):
    DISPLACEMENT = "DISPLACEMENT"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class CoreV1DisplacementObservation:
    """Caller-governed displacement classification with causal timing."""

    subject: ReplaySubject
    timing: EvidenceTiming
    direction: Direction | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Displacement requires EvidenceTiming provenance.")
        self.timing.validate_for_subject(self.subject)
        if self.timing.event_timestamp > self.subject.base_bar_timestamp:
            raise ValueError("Displacement event exceeds the ReplaySubject base-bar cutoff.")
        base_sources = tuple(
            source
            for source in self.timing.source_observations
            if source.is_base_timeframe
        )
        if not base_sources:
            raise ValueError("Displacement requires base-timeframe source provenance.")
        if any(
            source.base_bar_timestamp > self.subject.base_bar_timestamp
            for source in base_sources
        ):
            raise ValueError(
                "Displacement source exceeds the ReplaySubject base-bar cutoff."
            )
        if not isinstance(self.attributes, Mapping):
            raise TypeError("attributes must be a mapping.")
        if self.direction is not None and not isinstance(self.direction, Direction):
            raise TypeError("direction must be a governed displacement Direction.")
        if self.timing.confirmation_timestamp is not None and not any(
            source.event_timestamp == self.timing.confirmation_timestamp
            and source.availability_timestamp >= self.timing.confirmation_timestamp
            for source in self.timing.source_observations
        ):
            raise ValueError(
                "Displacement provenance must identify its confirmation observation."
            )
        attributes = dict(self.attributes)
        for name, value in attributes.items():
            if isinstance(value, (float, int)) and not isfinite(float(value)):
                raise ValueError(f"Displacement attribute {name!r} must be finite.")
        object.__setattr__(self, "attributes", MappingProxyType(attributes))


@dataclass(frozen=True)
class CoreV1DisplacementAssessment:
    """Point-in-time displacement-stage result, never a trade signal."""

    subject: ReplaySubject
    status: DisplacementStatus
    availability_timestamp: object
    observation: CoreV1DisplacementObservation | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.status, DisplacementStatus):
            raise TypeError("status must be a DisplacementStatus.")
        availability = _utc_timestamp(self.availability_timestamp)
        if availability != self.subject.availability_timestamp:
            raise ValueError("Displacement result availability must match its subject.")
        object.__setattr__(self, "availability_timestamp", availability)
        if self.status is DisplacementStatus.DISPLACEMENT:
            if self.observation is None:
                raise ValueError("DISPLACEMENT status requires an observation.")
            if not same_subject_id(self.observation.subject, self.subject):
                raise ValueError("Displacement observation belongs to another subject.")
        elif self.observation is not None:
            raise ValueError("UNRESOLVED result cannot contain a displacement observation.")

    @property
    def event_timestamp(self):
        return self.observation.timing.event_timestamp if self.observation else None

    @property
    def confirmation_timestamp(self):
        return self.observation.timing.confirmation_timestamp if self.observation else None

    @property
    def provenance(self) -> tuple[EvidenceSourceObservation, ...]:
        return (
            self.observation.timing.source_observations
            if self.observation is not None
            else ()
        )

    @property
    def direction(self) -> Direction | None:
        return self.observation.direction if self.observation is not None else None


class DisplacementResolver(Protocol):
    """Governed displacement policy supplied externally; no Core v1 thresholds."""

    def classify(
        self,
        inputs: SubjectVisibleInputs,
        liquidity_context: tuple[CoreV1LiquidityAssessment, ...],
    ) -> CoreV1DisplacementObservation | None: ...


def evaluate_core_v1_displacement(
    inputs: SubjectVisibleInputs,
    *,
    observation: CoreV1DisplacementObservation | None = None,
    resolver: DisplacementResolver | None = None,
    liquidity_context: tuple[CoreV1LiquidityAssessment, ...] = (),
) -> CoreV1DisplacementAssessment:
    """Validate governed displacement output or fail closed as UNRESOLVED."""
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    if observation is not None and resolver is not None:
        raise ValueError("Supply a displacement observation or resolver, not both.")
    context = tuple(liquidity_context)
    for item in context:
        if not isinstance(item, CoreV1LiquidityAssessment):
            raise TypeError("liquidity_context must contain liquidity assessments.")
        if not same_subject_id(item.subject, inputs.subject):
            raise ValueError("Liquidity context belongs to another ReplaySubject.")

    candidate = observation
    if candidate is None and resolver is not None:
        candidate = resolver.classify(inputs, context)
    if candidate is None:
        return CoreV1DisplacementAssessment(
            subject=inputs.subject,
            status=DisplacementStatus.UNRESOLVED,
            availability_timestamp=inputs.subject.availability_timestamp,
            reason="No governed displacement result is available.",
        )
    if not isinstance(candidate, CoreV1DisplacementObservation):
        raise TypeError(
            "Displacement resolver must return CoreV1DisplacementObservation or None."
        )
    if not same_subject_id(candidate.subject, inputs.subject):
        raise ValueError("Displacement result belongs to another ReplaySubject.")

    visible_sources = tuple(
        source
        for visible in inputs.base_observations + inputs.context_observations
        for source in visible.timing.source_observations
    )
    candidate.timing.validate_for_subject(inputs.subject)
    if any(
        source not in visible_sources
        for source in candidate.timing.source_observations
    ):
        raise ValueError(
            "Displacement provenance references observations outside visible inputs."
        )
    return CoreV1DisplacementAssessment(
        subject=inputs.subject,
        status=DisplacementStatus.DISPLACEMENT,
        availability_timestamp=inputs.subject.availability_timestamp,
        observation=candidate,
        reason="Displacement supplied by the governed resolver.",
    )


def _utc_timestamp(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("availability_timestamp must be a valid timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("availability_timestamp must be timezone-aware.")
    return timestamp.tz_convert("UTC")


__all__ = [
    "CoreV1DisplacementAssessment",
    "CoreV1DisplacementObservation",
    "DisplacementResolver",
    "DisplacementStatus",
    "evaluate_core_v1_displacement",
]
