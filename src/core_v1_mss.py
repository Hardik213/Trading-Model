from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol

from .core_v1_displacement import (
    CoreV1DisplacementAssessment,
    DisplacementStatus,
)
from .core_v1_evidence import SubjectVisibleInputs
from .core_v1_liquidity import (
    CoreV1LiquidityAssessment,
    ReactionKind,
)
from .displacement import Direction
from .evidence_timing import EvidenceSourceObservation, EvidenceTiming
from .market_structure import SwingType
from .replay_subject import ReplaySubject, same_subject_id


class MSSStatus(str, Enum):
    MSS = "MSS"
    NOT_MSS_UNRESOLVED = "NOT_MSS_UNRESOLVED"


@dataclass(frozen=True)
class CoreV1StructuralPoint:
    """Caller-governed meaningful structural point; no swing rule is implied."""

    subject: ReplaySubject
    point_id: str
    kind: SwingType
    price: float
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not self.point_id:
            raise ValueError("point_id must not be empty.")
        if not isinstance(self.kind, SwingType):
            raise TypeError("kind must be SwingType.HIGH or SwingType.LOW.")
        if not isfinite(float(self.price)):
            raise ValueError("Structural point price must be finite.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Structural points require EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        _validate_base_visibility(self.timing, self.subject, "Structural point")
        if not any(
            source.event_timestamp == self.timing.event_timestamp
            for source in self.timing.source_observations
        ):
            raise ValueError(
                "Structural point provenance must identify its event observation."
            )


@dataclass(frozen=True)
class CoreV1StructuralBreak:
    """Explicit caller-supplied break observation referencing its structural point."""

    subject: ReplaySubject
    structural_point: CoreV1StructuralPoint
    break_price: float
    timing: EvidenceTiming
    direction: Direction | None = None

    def __post_init__(self) -> None:
        if not same_subject_id(self.subject, self.structural_point.subject):
            raise ValueError("Structural break and point must share the ReplaySubject.")
        if not isfinite(float(self.break_price)):
            raise ValueError("break_price must be finite.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Structural break requires EvidenceTiming.")
        if not self.timing.base_bar_scoped:
            raise ValueError("Structural break timing must be base-bar scoped.")
        self.timing.validate_for_subject(self.subject)
        if self.timing.event_timestamp < self.structural_point.timing.event_timestamp:
            raise ValueError("Structural break cannot precede its structural point.")
        if any(
            source not in self.timing.source_observations
            for source in self.structural_point.timing.source_observations
        ):
            raise ValueError(
                "Structural break provenance must retain structural-point sources."
            )
        if not any(
            source not in self.structural_point.timing.source_observations
            and source.event_timestamp == self.timing.event_timestamp
            for source in self.timing.source_observations
        ):
            raise ValueError("Structural break provenance must identify its event observation.")
        if self.direction is not None and not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction or None.")


@dataclass(frozen=True)
class CoreV1MSSConfirmation:
    """Governed follow-through evidence, with confirmation separate from event time."""

    subject: ReplaySubject
    structural_break: CoreV1StructuralBreak
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not same_subject_id(self.subject, self.structural_break.subject):
            raise ValueError("Confirmation and structural break must share the subject.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("MSS confirmation requires EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        confirmation = self.timing.confirmation_timestamp
        if confirmation is None:
            raise ValueError("MSS follow-through requires a confirmation timestamp.")
        if self.timing.event_timestamp != self.structural_break.timing.event_timestamp:
            raise ValueError("MSS confirmation must retain the structural break event time.")
        if confirmation < self.structural_break.timing.event_timestamp:
            raise ValueError("MSS confirmation cannot precede the structural break.")
        _validate_confirmation_provenance(
            self.timing,
            self.structural_break.timing.source_observations,
        )


@dataclass(frozen=True)
class CoreV1MSSResult:
    """MSS evidence result only; contains no entry, trade, or execution action."""

    subject: ReplaySubject
    status: MSSStatus
    availability_timestamp: object
    structural_point: CoreV1StructuralPoint | None = None
    liquidity_event: CoreV1LiquidityAssessment | None = None
    displacement: CoreV1DisplacementAssessment | None = None
    structural_break: CoreV1StructuralBreak | None = None
    confirmation: CoreV1MSSConfirmation | None = None
    reason: str = ""
    provenance: Mapping[str, tuple[EvidenceSourceObservation, ...]] | None = None

    def __post_init__(self) -> None:
        import pandas as pd

        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.status, MSSStatus):
            raise TypeError("status must be an MSSStatus.")
        try:
            available = pd.Timestamp(self.availability_timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("availability_timestamp must be valid.") from exc
        if pd.isna(available) or available.tzinfo is None:
            raise ValueError("availability_timestamp must be timezone-aware.")
        available = available.tz_convert("UTC")
        if available > self.subject.availability_timestamp:
            raise ValueError("MSS result is unavailable at its ReplaySubject.")
        object.__setattr__(self, "availability_timestamp", available)
        provenance = {
            key: tuple(value)
            for key, value in (self.provenance or {}).items()
        }
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

        if self.status is MSSStatus.MSS:
            required = (
                self.structural_point,
                self.liquidity_event,
                self.displacement,
                self.structural_break,
                self.confirmation,
            )
            if any(item is None for item in required):
                raise ValueError("MSS status requires every governed component.")
            if any(not same_subject_id(item.subject, self.subject) for item in required):
                raise ValueError("MSS components must preserve the exact ReplaySubject.")
        elif any(
            item is not None
            for item in (
                self.structural_point,
                self.liquidity_event,
                self.displacement,
                self.structural_break,
                self.confirmation,
            )
        ):
            raise ValueError("Unresolved MSS cannot expose a partial MSS result.")

    @property
    def structural_point_timestamp(self):
        return (
            self.structural_point.timing.event_timestamp
            if self.structural_point is not None
            else None
        )

    @property
    def liquidity_event_timestamp(self):
        return (
            self.liquidity_event.breach.timing.event_timestamp
            if self.liquidity_event is not None
            and self.liquidity_event.breach is not None
            else None
        )

    @property
    def displacement_event_timestamp(self):
        if (
            self.displacement is None
            or self.displacement.observation is None
        ):
            return None
        return self.displacement.observation.timing.event_timestamp

    @property
    def confirmation_timestamp(self):
        return (
            self.confirmation.timing.confirmation_timestamp
            if self.confirmation is not None
            else None
        )


class StructuralPointResolver(Protocol):
    """Resolver for meaningful structure; policy is not defined in Core v1 yet."""

    def resolve(
        self,
        inputs: SubjectVisibleInputs,
    ) -> CoreV1StructuralPoint | None: ...


class MSSFollowThroughResolver(Protocol):
    """Resolver for follow-through; no bar count or close rule is implied."""

    def confirm(
        self,
        inputs: SubjectVisibleInputs,
        structural_break: CoreV1StructuralBreak,
    ) -> CoreV1MSSConfirmation | None: ...


def evaluate_core_v1_mss(
    inputs: SubjectVisibleInputs,
    *,
    structural_point: CoreV1StructuralPoint | None = None,
    structural_point_resolver: StructuralPointResolver | None = None,
    liquidity_event: CoreV1LiquidityAssessment | None = None,
    displacement: CoreV1DisplacementAssessment | None = None,
    structural_break: CoreV1StructuralBreak | None = None,
    confirmation: CoreV1MSSConfirmation | None = None,
    follow_through_resolver: MSSFollowThroughResolver | None = None,
) -> CoreV1MSSResult:
    """Gate caller-governed MSS components; unresolved prerequisites fail closed."""
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    if structural_point is not None and structural_point_resolver is not None:
        raise ValueError("Supply a structural point or resolver, not both.")

    point = structural_point
    if point is None and structural_point_resolver is not None:
        point = structural_point_resolver.resolve(inputs)
    if point is None:
        return _unresolved(inputs, "Meaningful structural point is unresolved.")
    _validate_component_subject(point.subject, inputs.subject, "Structural point")
    point.timing.validate_for_subject(inputs.subject)
    _validate_base_visibility(point.timing, inputs.subject, "Structural point")
    _validate_visible_lineage(point.timing, inputs, "Structural point")

    if liquidity_event is None or not _liquidity_is_resolved(liquidity_event):
        return _unresolved(inputs, "Liquidity event is unresolved.")
    _validate_liquidity(liquidity_event, inputs)

    if displacement is None or displacement.status is not DisplacementStatus.DISPLACEMENT:
        return _unresolved(inputs, "Displacement is unresolved.")
    if (
        not same_subject_id(displacement.subject, inputs.subject)
        or displacement.observation is None
    ):
        raise ValueError("Displacement result belongs to another subject or is incomplete.")
    displacement_timing = displacement.observation.timing
    displacement_timing.validate_for_subject(inputs.subject)
    _validate_base_visibility(displacement_timing, inputs.subject, "Displacement")
    _validate_visible_lineage(displacement_timing, inputs, "Displacement")

    if structural_break is None:
        return _unresolved(inputs, "Structural break is unresolved.")
    _validate_component_subject(structural_break.subject, inputs.subject, "Structural break")
    if structural_break.structural_point is not point:
        raise ValueError("Structural break must reference the supplied structural point.")
    structural_break.timing.validate_for_subject(inputs.subject)
    _validate_base_visibility(structural_break.timing, inputs.subject, "Structural break")
    _validate_visible_lineage(structural_break.timing, inputs, "Structural break")

    follow = confirmation
    if follow is None and follow_through_resolver is not None:
        follow = follow_through_resolver.confirm(inputs, structural_break)
    if follow is None:
        return _unresolved(inputs, "MSS follow-through is unresolved.")
    _validate_component_subject(follow.subject, inputs.subject, "MSS confirmation")
    if follow.structural_break is not structural_break:
        raise ValueError("MSS confirmation must reference the supplied structural break.")
    follow.timing.validate_for_subject(inputs.subject)
    _validate_base_visibility(follow.timing, inputs.subject, "MSS confirmation")
    _validate_visible_lineage(follow.timing, inputs, "MSS confirmation")

    liquidity_timing = liquidity_event.reaction.timing
    ordered = (
        (
            "structural point",
            point.timing.event_timestamp,
            "liquidity event",
            liquidity_event.breach.timing.event_timestamp,
        ),
        (
            "liquidity event",
            liquidity_event.breach.timing.event_timestamp,
            "displacement",
            displacement_timing.event_timestamp,
        ),
        (
            "displacement",
            displacement_timing.event_timestamp,
            "structural break",
            structural_break.timing.event_timestamp,
        ),
        (
            "structural break",
            structural_break.timing.event_timestamp,
            "confirmation",
            follow.timing.confirmation_timestamp,
        ),
    )
    for earlier_name, earlier, later_name, later in ordered:
        if earlier > later:
            raise ValueError(
                f"Causal MSS ordering violation: {earlier_name} follows {later_name}."
            )
    availability = max(
        point.timing.availability_timestamp,
        liquidity_timing.availability_timestamp,
        displacement_timing.availability_timestamp,
        structural_break.timing.availability_timestamp,
        follow.timing.availability_timestamp,
    )
    if availability > inputs.subject.availability_timestamp:
        raise ValueError("MSS components exceed ReplaySubject availability.")
    if (
        follow.timing.confirmation_timestamp is None
        or follow.timing.confirmation_timestamp > inputs.subject.availability_timestamp
    ):
        raise ValueError("MSS confirmation exceeds ReplaySubject availability.")

    provenance = {
        "structural_point": point.timing.source_observations,
        "liquidity_event": liquidity_event.reaction.timing.source_observations,
        "displacement": displacement_timing.source_observations,
        "structural_break": structural_break.timing.source_observations,
        "confirmation": follow.timing.source_observations,
    }
    return CoreV1MSSResult(
        subject=inputs.subject,
        status=MSSStatus.MSS,
        availability_timestamp=availability,
        structural_point=point,
        liquidity_event=liquidity_event,
        displacement=displacement,
        structural_break=structural_break,
        confirmation=follow,
        reason="All caller-governed MSS components are present and causally ordered.",
        provenance=provenance,
    )


def _liquidity_is_resolved(value: CoreV1LiquidityAssessment) -> bool:
    return (
        value.breach is not None
        and value.reaction is not None
        and value.reaction.kind in {ReactionKind.REJECTION, ReactionKind.ACCEPTANCE}
    )


def _validate_liquidity(
    value: CoreV1LiquidityAssessment,
    inputs: SubjectVisibleInputs,
) -> None:
    _validate_component_subject(value.subject, inputs.subject, "Liquidity event")
    if not _liquidity_is_resolved(value):
        raise ValueError("Liquidity event is unresolved.")
    assert value.breach is not None
    assert value.reaction is not None
    value.breach.timing.validate_for_subject(inputs.subject)
    value.reaction.timing.validate_for_subject(inputs.subject)
    _validate_visible_lineage(value.level.timing, inputs, "Liquidity level")
    _validate_visible_lineage(value.breach.timing, inputs, "Liquidity breach")
    _validate_visible_lineage(value.reaction.timing, inputs, "Liquidity reaction")


def _validate_component_subject(
    actual: ReplaySubject,
    expected: ReplaySubject,
    name: str,
) -> None:
    if not same_subject_id(actual, expected):
        raise ValueError(f"{name} does not preserve the exact ReplaySubject.")


def _validate_base_visibility(
    timing: EvidenceTiming,
    subject: ReplaySubject,
    name: str,
) -> None:
    if timing.event_timestamp > subject.base_bar_timestamp:
        raise ValueError(f"{name} event exceeds the ReplaySubject base-bar cutoff.")
    base_sources = tuple(
        source for source in timing.source_observations if source.is_base_timeframe
    )
    if not base_sources:
        raise ValueError(f"{name} requires base-timeframe source provenance.")
    if any(source.base_bar_timestamp > subject.base_bar_timestamp for source in base_sources):
        raise ValueError(f"{name} provenance exceeds the ReplaySubject base-bar cutoff.")


def _validate_visible_lineage(
    timing: EvidenceTiming,
    inputs: SubjectVisibleInputs,
    name: str,
) -> None:
    visible = tuple(
        source
        for observation in inputs.base_observations + inputs.context_observations
        for source in observation.timing.source_observations
    )
    if any(source not in visible for source in timing.source_observations):
        raise ValueError(f"{name} provenance references observations outside visible inputs.")


def _validate_confirmation_provenance(
    timing: EvidenceTiming,
    inherited: tuple[EvidenceSourceObservation, ...],
) -> None:
    confirmation = timing.confirmation_timestamp
    if confirmation is None:
        raise ValueError("MSS follow-through requires confirmation timestamp.")
    if not any(
        source not in inherited
        and source.event_timestamp == confirmation
        and source.availability_timestamp >= confirmation
        for source in timing.source_observations
    ):
        raise ValueError("MSS confirmation provenance must identify follow-through.")


def _unresolved(inputs: SubjectVisibleInputs, reason: str) -> CoreV1MSSResult:
    return CoreV1MSSResult(
        subject=inputs.subject,
        status=MSSStatus.NOT_MSS_UNRESOLVED,
        availability_timestamp=inputs.subject.availability_timestamp,
        reason=reason,
    )


__all__ = [
    "CoreV1MSSConfirmation",
    "CoreV1MSSResult",
    "CoreV1StructuralBreak",
    "CoreV1StructuralPoint",
    "MSSFollowThroughResolver",
    "MSSStatus",
    "StructuralPointResolver",
    "evaluate_core_v1_mss",
]
