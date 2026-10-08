from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class ReplaySubject:
    """One replay decision subject, distinct from its causal as-of time."""

    availability_timestamp: pd.Timestamp
    base_bar_timestamp: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "availability_timestamp",
            _utc_timestamp(self.availability_timestamp, "availability_timestamp"),
        )
        object.__setattr__(
            self,
            "base_bar_timestamp",
            _utc_timestamp(self.base_bar_timestamp, "base_bar_timestamp"),
        )
        if self.base_bar_timestamp > self.availability_timestamp:
            raise ValueError("base_bar_timestamp cannot be later than availability_timestamp.")

    def __hash__(self) -> int:
        return hash((self.availability_timestamp.value, self.base_bar_timestamp.value))

    @property
    def subject_id(self) -> str:
        return (
            "replay-subject-v1|"
            f"{self.availability_timestamp.isoformat()}|"
            f"{self.base_bar_timestamp.isoformat()}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "availability_timestamp": self.availability_timestamp.isoformat(),
            "base_bar_timestamp": self.base_bar_timestamp.isoformat(),
            "subject_id": self.subject_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ReplaySubject:
        subject = cls(
            availability_timestamp=value["availability_timestamp"],
            base_bar_timestamp=value["base_bar_timestamp"],
        )
        supplied_id = value.get("subject_id")
        if supplied_id is not None and supplied_id != subject.subject_id:
            raise ValueError("subject_id does not match the replay subject timestamps.")
        return subject


def same_subject_id(left: ReplaySubject, right: ReplaySubject) -> bool:
    """Return whether two ReplaySubject values identify the same replay subject."""
    return left.subject_id == right.subject_id