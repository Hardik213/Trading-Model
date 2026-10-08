from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Generic, Mapping, TypeVar

import numpy as np
import pandas as pd

from .replay_subject import ReplaySubject

MAX_BOUNDED_STRING_LENGTH = 4096
MAX_BOUNDED_MAPPING_ENTRIES = 256


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


def validate_bounded_mapping_keys(value: Mapping[str, object], name: str) -> None:
    """Enforce the shared entry-count and key-length bounds for metadata maps."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if len(value) > MAX_BOUNDED_MAPPING_ENTRIES:
        raise TypeError(f"{name} mappings are limited to 256 entries.")
    if any(
        type(key) is not str or len(key) > MAX_BOUNDED_STRING_LENGTH
        for key in value
    ):
        raise TypeError(
            f"{name} mapping keys must be strings no longer than "
            f"{MAX_BOUNDED_STRING_LENGTH} characters."
        )


def validate_bounded_payload(
    value: object,
    name: str,
    *,
    allow_lists: bool = True,
) -> object:
    """Validate and freeze bounded metadata using the Core v1 payload contract."""
    remaining = 4096

    def normalize(item: object, depth: int) -> object:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 8:
            raise TypeError(f"{name} exceeds the bounded payload contract.")
        if isinstance(item, (pd.DataFrame, pd.Series, pd.Index)):
            raise TypeError(
                f"{name} must contain individual observations, not frames or series."
            )
        if isinstance(item, np.generic):
            return normalize(item.item(), depth)
        if item is None or type(item) in (bool, int):
            return item
        if type(item) is float:
            if not isfinite(item):
                raise TypeError(f"{name} numbers must be finite.")
            return item
        if type(item) is str:
            if len(item) > MAX_BOUNDED_STRING_LENGTH:
                raise TypeError(f"{name} strings exceed the bounded payload contract.")
            return item
        if type(item) is bytes:
            if len(item) > MAX_BOUNDED_STRING_LENGTH:
                raise TypeError(f"{name} byte strings exceed the bounded payload contract.")
            return item
        if isinstance(item, pd.Timestamp):
            if pd.isna(item):
                raise TypeError(f"{name} timestamps must not be NaT.")
            return item
        if isinstance(item, pd.Timedelta):
            if pd.isna(item):
                raise TypeError(f"{name} timedeltas must not be NaT.")
            return item
        if isinstance(item, (datetime, date)):
            return item
        if isinstance(item, Enum):
            enum_value = normalize(item.value, depth + 1)
            if isinstance(enum_value, (Mapping, tuple)):
                raise TypeError(f"{name} enums must have immutable scalar values.")
            return item
        if type(item) is dict or isinstance(item, MappingProxyType):
            validate_bounded_mapping_keys(item, name)
            return MappingProxyType(
                {
                    key: normalize(child, depth + 1)
                    for key, child in item.items()
                }
            )
        if type(item) in (tuple, list):
            if type(item) is list and not allow_lists:
                label = name.lower()
                suffix = "type" if label.endswith("payload") else "payload type"
                raise TypeError(f"Unsupported {label} {suffix}: list.")
            if len(item) > MAX_BOUNDED_MAPPING_ENTRIES:
                raise TypeError(f"{name} sequences are limited to 256 values.")
            return tuple(normalize(child, depth + 1) for child in item)
        label = name.lower()
        if label.endswith("payload"):
            label = label[: -len("payload")].strip()
        raise TypeError(f"Unsupported {label} payload type: {type(item).__name__}.")

    return normalize(value, 0)


def _immutable_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    normalized = validate_bounded_payload(value, name)
    if not isinstance(normalized, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return normalized


@dataclass(frozen=True)
class EvidenceSourceObservation:
    """A provenance reference to one source observation, not its full source frame."""

    source_id: str
    source: str
    event_timestamp: pd.Timestamp
    availability_timestamp: pd.Timestamp
    provenance: Mapping[str, object] = field(default_factory=dict)
    base_bar_timestamp: pd.Timestamp | None = None
    is_base_timeframe: bool = False

    def __post_init__(self) -> None:
        for name, value in (("source_id", self.source_id), ("source", self.source)):
            if type(value) is not str or not value or len(value) > 4096:
                raise ValueError(f"{name} must be a non-empty bounded string.")
        if type(self.is_base_timeframe) is not bool:
            raise TypeError("is_base_timeframe must be a boolean.")

        event_timestamp = _utc_timestamp(self.event_timestamp, "event_timestamp")
        availability_timestamp = _utc_timestamp(
            self.availability_timestamp,
            "availability_timestamp",
        )
        if availability_timestamp < event_timestamp:
            raise ValueError("Source availability cannot precede its event timestamp.")

        object.__setattr__(self, "event_timestamp", event_timestamp)
        object.__setattr__(self, "availability_timestamp", availability_timestamp)
        object.__setattr__(
            self,
            "provenance",
            _immutable_mapping(self.provenance, "provenance"),
        )
        if self.base_bar_timestamp is not None:
            base_bar_timestamp = _utc_timestamp(
                self.base_bar_timestamp,
                "base_bar_timestamp",
            )
            if availability_timestamp < base_bar_timestamp:
                raise ValueError("Source availability cannot precede its base-bar timestamp.")
            object.__setattr__(self, "base_bar_timestamp", base_bar_timestamp)
        if self.is_base_timeframe and self.base_bar_timestamp is None:
            raise ValueError(
                "Base-timeframe source observations require base_bar_timestamp."
            )


@dataclass(frozen=True)
class EvidenceTiming:
    """Causal timing and source lineage for one observed or derived evidence item."""

    event_timestamp: pd.Timestamp
    availability_timestamp: pd.Timestamp
    source_observations: tuple[EvidenceSourceObservation, ...]
    confirmation_timestamp: pd.Timestamp | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    base_bar_scoped: bool = False

    def __post_init__(self) -> None:
        event_timestamp = _utc_timestamp(self.event_timestamp, "event_timestamp")
        availability_timestamp = _utc_timestamp(
            self.availability_timestamp,
            "availability_timestamp",
        )
        if type(self.source_observations) not in (tuple, list):
            raise TypeError("source_observations must be a bounded tuple or list.")
        if len(self.source_observations) > 256:
            raise ValueError("source_observations are limited to 256 references.")
        source_observations = tuple(self.source_observations)
        if not source_observations:
            raise ValueError("Evidence timing requires source observations.")
        if type(self.base_bar_scoped) is not bool:
            raise TypeError("base_bar_scoped must be a boolean.")
        if any(not isinstance(item, EvidenceSourceObservation) for item in source_observations):
            raise TypeError(
                "source_observations must contain EvidenceSourceObservation values."
            )

        confirmation_timestamp = self.confirmation_timestamp
        if confirmation_timestamp is not None:
            confirmation_timestamp = _utc_timestamp(
                confirmation_timestamp,
                "confirmation_timestamp",
            )
            if confirmation_timestamp < event_timestamp:
                raise ValueError("Confirmation cannot precede the evidence event.")
            object.__setattr__(self, "confirmation_timestamp", confirmation_timestamp)

        latest_input_availability = max(
            item.availability_timestamp for item in source_observations
        )
        if availability_timestamp < latest_input_availability:
            raise ValueError(
                "Evidence availability cannot precede any required input availability."
            )
        if availability_timestamp < event_timestamp:
            raise ValueError("Evidence availability cannot precede its event timestamp.")
        if (
            confirmation_timestamp is not None
            and availability_timestamp < confirmation_timestamp
        ):
            raise ValueError("Evidence availability cannot precede confirmation.")

        object.__setattr__(self, "event_timestamp", event_timestamp)
        object.__setattr__(self, "availability_timestamp", availability_timestamp)
        object.__setattr__(self, "source_observations", source_observations)
        object.__setattr__(
            self,
            "provenance",
            _immutable_mapping(self.provenance, "provenance"),
        )

    @classmethod
    def from_inputs(
        cls,
        *,
        event_timestamp: object,
        source_observations: tuple[EvidenceSourceObservation, ...],
        confirmation_timestamp: object | None = None,
        provenance: Mapping[str, object] | None = None,
        base_bar_scoped: bool = False,
    ) -> EvidenceTiming:
        """Set availability to the latest input; confirmation needs timed support."""
        inputs = tuple(source_observations)
        if not inputs:
            raise ValueError("Derived evidence requires at least one source observation.")
        availability = max(item.availability_timestamp for item in inputs)
        if confirmation_timestamp is not None:
            confirmation = _utc_timestamp(
                confirmation_timestamp,
                "confirmation_timestamp",
            )
        else:
            confirmation = None
        return cls(
            event_timestamp=event_timestamp,
            confirmation_timestamp=confirmation,
            availability_timestamp=availability,
            source_observations=inputs,
            provenance=provenance or {},
            base_bar_scoped=base_bar_scoped,
        )

    def validate_for_subject(self, subject: ReplaySubject) -> None:
        """Reject evidence unavailable at T or outside a base subject's prefix."""
        if not isinstance(subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if self.availability_timestamp > subject.availability_timestamp:
            raise ValueError(
                "Evidence is unavailable at the ReplaySubject availability timestamp."
            )
        if any(
            item.availability_timestamp > subject.availability_timestamp
            for item in self.source_observations
        ):
            raise ValueError(
                "A source observation is unavailable at the ReplaySubject availability timestamp."
            )

        if self.base_bar_scoped:
            if self.event_timestamp > subject.base_bar_timestamp:
                raise ValueError(
                    "Base-scoped evidence event exceeds the ReplaySubject base-bar timestamp."
                )
            if (
                self.confirmation_timestamp is not None
                and self.confirmation_timestamp > subject.base_bar_timestamp
            ):
                raise ValueError(
                    "Base-scoped evidence confirmation exceeds the ReplaySubject base-bar timestamp."
                )
        for item in self.source_observations:
            if item.is_base_timeframe:
                if item.event_timestamp > subject.base_bar_timestamp:
                    raise ValueError(
                        "A base-timeframe source event exceeds the ReplaySubject base-bar cutoff."
                    )
                if item.base_bar_timestamp > subject.base_bar_timestamp:
                    raise ValueError(
                        "A base-timeframe source exceeds the ReplaySubject base-bar prefix."
                    )


EvidenceValue = TypeVar("EvidenceValue")


@dataclass(frozen=True)
class SubjectEvidence(Generic[EvidenceValue]):
    """Evidence that has passed timing checks for one ReplaySubject."""

    subject: ReplaySubject
    timing: EvidenceTiming
    value: EvidenceValue

    def __post_init__(self) -> None:
        self.timing.validate_for_subject(self.subject)


__all__ = [
    "EvidenceSourceObservation",
    "EvidenceTiming",
    "MAX_BOUNDED_MAPPING_ENTRIES",
    "MAX_BOUNDED_STRING_LENGTH",
    "SubjectEvidence",
    "validate_bounded_payload",
    "validate_bounded_mapping_keys",
]
