from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol

import pandas as pd

from .core_v1_evidence import SubjectVisibleInputs
from .dealing_range import DealingRange, RangeLocation
from .evidence_timing import EvidenceSourceObservation, SubjectEvidence
from .replay_subject import ReplaySubject, same_subject_id


class DealingRangeStatus(str, Enum):
    RESOLVED = "RESOLVED"
    NO_RANGE_UNRESOLVED = "NO_RANGE_UNRESOLVED"


@dataclass(frozen=True)
class DealingRangeContext:
    """Point-in-time range description; never an actionable trade signal."""

    subject: ReplaySubject
    status: DealingRangeStatus
    availability_timestamp: object
    event_timestamp: object | None = None
    confirmation_timestamp: object | None = None
    dealing_range: DealingRange | None = None
    provenance: tuple[EvidenceSourceObservation, ...] = ()
    reference_price: float | None = None
    location: RangeLocation | None = None
    reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.status, DealingRangeStatus):
            raise TypeError("status must be a DealingRangeStatus.")
        availability = _utc_timestamp(
            self.availability_timestamp,
            "availability_timestamp",
        )
        if availability > self.subject.availability_timestamp:
            raise ValueError("Range context is unavailable at its ReplaySubject.")
        object.__setattr__(self, "availability_timestamp", availability)

        if self.event_timestamp is not None:
            event_timestamp = _utc_timestamp(self.event_timestamp, "event_timestamp")
            if event_timestamp > availability:
                raise ValueError("Range event cannot occur after its availability.")
            object.__setattr__(self, "event_timestamp", event_timestamp)
        if self.confirmation_timestamp is not None:
            confirmation = _utc_timestamp(
                self.confirmation_timestamp,
                "confirmation_timestamp",
            )
            if self.event_timestamp is not None and confirmation < self.event_timestamp:
                raise ValueError("Range confirmation cannot precede its event.")
            if confirmation > availability:
                raise ValueError("Range confirmation cannot occur after availability.")
            object.__setattr__(self, "confirmation_timestamp", confirmation)

        provenance = tuple(self.provenance)
        if any(not isinstance(item, EvidenceSourceObservation) for item in provenance):
            raise TypeError("provenance must contain source observations.")
        if any(
            item.availability_timestamp > self.subject.availability_timestamp
            for item in provenance
        ):
            raise ValueError("Range provenance is unavailable at its ReplaySubject.")
        if any(
            item.is_base_timeframe
            and item.base_bar_timestamp > self.subject.base_bar_timestamp
            for item in provenance
        ):
            raise ValueError("Range provenance exceeds the ReplaySubject base-bar cutoff.")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

        if self.status is DealingRangeStatus.RESOLVED:
            if self.dealing_range is None or self.event_timestamp is None:
                raise ValueError("Resolved range context requires a range and event time.")
            if not self.provenance:
                raise ValueError("Resolved range context requires provenance.")
        elif (
            self.dealing_range is not None
            or self.event_timestamp is not None
            or self.confirmation_timestamp is not None
            or self.reference_price is not None
            or self.location is not None
        ):
            raise ValueError("Unresolved range context cannot contain fabricated range data.")

        if self.reference_price is None:
            if self.location is not None:
                raise ValueError("Range location requires a visible reference price.")
        elif self.status is not DealingRangeStatus.RESOLVED:
            raise ValueError("Unresolved range cannot classify a reference price.")
        elif not isfinite(float(self.reference_price)):
            raise ValueError("reference_price must be finite.")


class DealingRangeResolver(Protocol):
    """Unspecified anchor-selection policy supplied by a future governed resolver."""

    def resolve(
        self,
        inputs: SubjectVisibleInputs,
    ) -> SubjectEvidence[DealingRange] | None: ...


class UnavailableDealingRangeResolver:
    """Fail-closed placeholder; no anchor policy is defined by this resolver."""

    def resolve(
        self,
        inputs: SubjectVisibleInputs,
    ) -> SubjectEvidence[DealingRange] | None:
        if not isinstance(inputs, SubjectVisibleInputs):
            raise TypeError("inputs must be SubjectVisibleInputs.")
        return None


def build_dealing_range_context(
    inputs: SubjectVisibleInputs,
    resolver: DealingRangeResolver | None = None,
    *,
    reference_price: SubjectEvidence[float] | None = None,
) -> DealingRangeContext:
    """Resolve only from validated visible observations; unresolved means NO RANGE."""
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    resolver = resolver or UnavailableDealingRangeResolver()
    candidate = resolver.resolve(inputs)
    all_visible_sources = tuple(
        source
        for observation in inputs.base_observations + inputs.context_observations
        for source in observation.timing.source_observations
    )

    if candidate is None:
        return _unresolved(
            inputs,
            all_visible_sources,
            "No canonical range was resolved; anchor policy remains unresolved.",
        )
    if not isinstance(candidate, SubjectEvidence):
        raise TypeError("Range resolver must return SubjectEvidence[DealingRange] or None.")
    if not same_subject_id(candidate.subject, inputs.subject):
        raise ValueError("Range resolver returned a different ReplaySubject.")
    timing = candidate.timing
    timing.validate_for_subject(inputs.subject)
    if any(source not in all_visible_sources for source in timing.source_observations):
        raise ValueError("Range provenance references observations outside visible inputs.")
    if not isinstance(candidate.value, DealingRange):
        return _unresolved(inputs, all_visible_sources, "Resolver returned no valid range.")
    if not _valid_range(candidate.value):
        return _unresolved(inputs, all_visible_sources, "Range anchors are invalid.")

    anchor_timestamps = {
        _utc_timestamp(
            candidate.value.anchor_high_timestamp,
            "anchor_high_timestamp",
        ),
        _utc_timestamp(
            candidate.value.anchor_low_timestamp,
            "anchor_low_timestamp",
        ),
    }
    source_event_timestamps = {
        source.event_timestamp for source in timing.source_observations
    }
    if not anchor_timestamps.issubset(source_event_timestamps):
        return _unresolved(
            inputs,
            all_visible_sources,
            "Range anchors are not supported by the declared source observations.",
        )

    if reference_price is not None:
        if not isinstance(reference_price, SubjectEvidence):
            raise TypeError("reference_price must be wrapped in SubjectEvidence.")
        if not same_subject_id(reference_price.subject, inputs.subject):
            raise ValueError("Reference price belongs to a different ReplaySubject.")
        reference_price.timing.validate_for_subject(inputs.subject)
        if any(
            source not in all_visible_sources
            for source in reference_price.timing.source_observations
        ):
            raise ValueError(
                "Reference price provenance references observations outside visible inputs."
            )
        if not isfinite(float(reference_price.value)):
            raise ValueError("Reference price must be finite.")
        price = float(reference_price.value)
        location = candidate.value.location(price)
    else:
        price = None
        location = None

    return DealingRangeContext(
        subject=inputs.subject,
        status=DealingRangeStatus.RESOLVED,
        availability_timestamp=timing.availability_timestamp,
        event_timestamp=timing.event_timestamp,
        confirmation_timestamp=timing.confirmation_timestamp,
        dealing_range=candidate.value,
        provenance=timing.source_observations,
        reference_price=price,
        location=location,
        reason="Range supplied by the configured resolver.",
        metadata=timing.provenance,
    )


def _valid_range(value: DealingRange) -> bool:
    try:
        _utc_timestamp(value.anchor_high_timestamp, "anchor_high_timestamp")
        _utc_timestamp(value.anchor_low_timestamp, "anchor_low_timestamp")
        high = float(value.anchor_high)
        low = float(value.anchor_low)
        equilibrium = float(value.equilibrium)
        return (
            isfinite(high)
            and isfinite(low)
            and isfinite(equilibrium)
            and high > low
            and low <= equilibrium <= high
            and equilibrium == (high + low) / 2
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


def _unresolved(
    inputs: SubjectVisibleInputs,
    provenance: tuple[EvidenceSourceObservation, ...],
    reason: str,
) -> DealingRangeContext:
    return DealingRangeContext(
        subject=inputs.subject,
        status=DealingRangeStatus.NO_RANGE_UNRESOLVED,
        availability_timestamp=inputs.availability_timestamp,
        provenance=provenance,
        reason=reason,
    )


__all__ = [
    "DealingRangeContext",
    "DealingRangeResolver",
    "DealingRangeStatus",
    "UnavailableDealingRangeResolver",
    "build_dealing_range_context",
]
