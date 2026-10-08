from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol

from .core_v1_evidence import SubjectVisibleInputs
from .evidence_timing import EvidenceSourceObservation, EvidenceTiming
from .market_structure import LiquiditySide
from .replay_subject import ReplaySubject, same_subject_id


class ReactionKind(str, Enum):
    REJECTION = "REJECTION"
    ACCEPTANCE = "ACCEPTANCE"


class SweepKind(str, Enum):
    SWEEP = "SWEEP"
    NOT_SWEEP = "NOT_SWEEP"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class CoreV1LiquidityLevel:
    """A supplied BSL/SSL level with subject timing and source lineage."""

    subject: ReplaySubject
    level_id: str
    side: LiquiditySide
    price: float
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not self.level_id:
            raise ValueError("level_id must not be empty.")
        if not isinstance(self.side, LiquiditySide):
            raise TypeError("side must be LiquiditySide.BSL or LiquiditySide.SSL.")
        if not isfinite(float(self.price)):
            raise ValueError("Liquidity level price must be finite.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Liquidity levels require EvidenceTiming provenance.")
        self.timing.validate_for_subject(self.subject)


@dataclass(frozen=True)
class CoreV1LiquidityBreach:
    """A distinct through-level observation; it is not a sweep classification."""

    subject: ReplaySubject
    level: CoreV1LiquidityLevel
    price: float | None
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not same_subject_id(self.subject, self.level.subject):
            raise ValueError("Breach and level must preserve the same ReplaySubject.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Liquidity breaches require EvidenceTiming provenance.")
        if not self.timing.base_bar_scoped:
            raise ValueError("Liquidity breach timing must be base-bar scoped.")
        self.timing.validate_for_subject(self.subject)
        if self.timing.event_timestamp < self.level.timing.availability_timestamp:
            raise ValueError("Breach event cannot precede level availability.")
        if self.price is not None and not isfinite(float(self.price)):
            raise ValueError("Breach price must be finite.")


@dataclass(frozen=True)
class CoreV1LiquidityReaction:
    """A caller-classified reaction retaining the original breach event time."""

    subject: ReplaySubject
    breach: CoreV1LiquidityBreach
    kind: ReactionKind
    timing: EvidenceTiming
    price: float | None = None

    def __post_init__(self) -> None:
        if not same_subject_id(self.subject, self.breach.subject):
            raise ValueError("Reaction and breach must preserve the same ReplaySubject.")
        if not isinstance(self.kind, ReactionKind):
            raise TypeError("kind must be REJECTION or ACCEPTANCE.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Liquidity reactions require EvidenceTiming provenance.")
        self.timing.validate_for_subject(self.subject)
        if self.timing.event_timestamp != self.breach.timing.event_timestamp:
            raise ValueError("Reaction must preserve the original breach event timestamp.")
        if self.timing.confirmation_timestamp is None:
            raise ValueError("A classified reaction requires a confirmation timestamp.")
        if (
            self.timing.availability_timestamp
            < self.breach.timing.availability_timestamp
        ):
            raise ValueError("Reaction availability cannot precede breach availability.")
        _validate_confirmation_provenance(
            self.timing,
            self.breach.timing.source_observations,
            "reaction",
        )
        if self.price is not None and not isfinite(float(self.price)):
            raise ValueError("Reaction price must be finite.")


@dataclass(frozen=True)
class CoreV1LiquidityAssessment:
    """Liquidity-stage state only; it contains no direction or trade signal."""

    subject: ReplaySubject
    level: CoreV1LiquidityLevel
    breach: CoreV1LiquidityBreach | None = None
    reaction: CoreV1LiquidityReaction | None = None
    sweep: SweepKind = SweepKind.UNRESOLVED
    sweep_timing: EvidenceTiming | None = None
    sweep_provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not same_subject_id(self.subject, self.level.subject):
            raise ValueError("Assessment and level must preserve the same ReplaySubject.")
        if self.breach is None and (self.reaction is not None or self.sweep is not SweepKind.UNRESOLVED):
            raise ValueError("Reaction or sweep classification requires a breach.")
        if self.breach is not None and self.breach.level is not self.level:
            raise ValueError("Assessment breach must reference its level.")
        if self.breach is not None and any(
            source not in self.breach.timing.source_observations
            for source in self.level.timing.source_observations
        ):
            raise ValueError("Breach provenance must retain level source observations.")
        if self.reaction is not None:
            if self.breach is None or self.reaction.breach is not self.breach:
                raise ValueError("Assessment reaction must reference its breach.")
            if any(
                source not in self.reaction.timing.source_observations
                for source in self.breach.timing.source_observations
            ):
                raise ValueError("Reaction provenance must retain breach observations.")
        if self.sweep is not SweepKind.UNRESOLVED:
            if self.reaction is None or self.sweep_timing is None:
                raise ValueError("Resolved sweep status requires reaction and timing.")
            if (
                self.sweep is SweepKind.SWEEP
                and self.reaction.kind is not ReactionKind.REJECTION
            ):
                raise ValueError("Acceptance cannot be classified as a sweep.")
            self.sweep_timing.validate_for_subject(self.subject)
            if self.sweep_timing.confirmation_timestamp is None:
                raise ValueError("Resolved sweep status requires confirmation time.")
            if (
                self.breach is None
                or self.sweep_timing.event_timestamp
                != self.breach.timing.event_timestamp
            ):
                raise ValueError("Sweep status must preserve the original breach event time.")
            if any(
                source not in self.sweep_timing.source_observations
                for source in self.reaction.timing.source_observations
            ):
                raise ValueError("Sweep provenance must retain reaction observations.")
            _validate_confirmation_provenance(
                self.sweep_timing,
                self.reaction.timing.source_observations,
                "sweep",
            )
        elif self.sweep_timing is not None:
            raise ValueError("Unresolved sweep status cannot have confirmation timing.")
        provenance = {} if self.sweep_provenance is None else dict(self.sweep_provenance)
        object.__setattr__(self, "sweep_provenance", MappingProxyType(provenance))


class LiquidityLevelResolver(Protocol):
    """Optional governed level selector. No canonical selector currently exists."""

    def resolve(
        self,
        inputs: SubjectVisibleInputs,
    ) -> tuple[CoreV1LiquidityLevel, ...]: ...


class LiquidityBreachResolver(Protocol):
    """Optional governed breach detector over visible observations only."""

    def detect(
        self,
        inputs: SubjectVisibleInputs,
        level: CoreV1LiquidityLevel,
    ) -> CoreV1LiquidityBreach | None: ...


class LiquidityReactionResolver(Protocol):
    """Optional governed rejection/acceptance resolver."""

    def classify(
        self,
        inputs: SubjectVisibleInputs,
        breach: CoreV1LiquidityBreach,
    ) -> CoreV1LiquidityReaction | None: ...


class LiquiditySweepResolver(Protocol):
    """Optional governed sweep classifier; a breach/rejection is not auto-promoted."""

    def classify(
        self,
        inputs: SubjectVisibleInputs,
        reaction: CoreV1LiquidityReaction,
    ) -> tuple[SweepKind, EvidenceTiming, Mapping[str, object]] | None: ...


def evaluate_core_v1_liquidity(
    inputs: SubjectVisibleInputs,
    *,
    levels: tuple[CoreV1LiquidityLevel, ...] | None = None,
    level_resolver: LiquidityLevelResolver | None = None,
    breach_resolver: LiquidityBreachResolver | None = None,
    reaction_resolver: LiquidityReactionResolver | None = None,
    sweep_resolver: LiquiditySweepResolver | None = None,
) -> tuple[CoreV1LiquidityAssessment, ...]:
    """Evaluate supplied governance outputs without inventing missing policies."""
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    if levels is not None and level_resolver is not None:
        raise ValueError("Supply levels or a level_resolver, not both.")
    visible_sources = _visible_sources(inputs)

    if levels is None:
        selected = (
            tuple(level_resolver.resolve(inputs))
            if level_resolver is not None
            else ()
        )
    else:
        selected = tuple(levels)

    assessments: list[CoreV1LiquidityAssessment] = []
    seen_ids: set[str] = set()
    for level in selected:
        if not isinstance(level, CoreV1LiquidityLevel):
            raise TypeError("levels must contain CoreV1LiquidityLevel values.")
        if not same_subject_id(level.subject, inputs.subject):
            raise ValueError("Liquidity level belongs to a different ReplaySubject.")
        if level.level_id in seen_ids:
            raise ValueError(f"Duplicate liquidity level_id: {level.level_id}")
        seen_ids.add(level.level_id)
        _validate_lineage(level.timing, visible_sources, "level")

        breach = None
        reaction = None
        sweep = SweepKind.UNRESOLVED
        sweep_timing = None
        sweep_provenance: Mapping[str, object] = {}

        if breach_resolver is not None:
            breach = breach_resolver.detect(inputs, level)
        if breach is not None:
            _validate_breach(inputs.subject, level, breach, visible_sources)
            if reaction_resolver is not None:
                reaction = reaction_resolver.classify(inputs, breach)
            if reaction is not None:
                _validate_reaction(
                    inputs.subject,
                    breach,
                    reaction,
                    visible_sources,
                )
                if sweep_resolver is not None:
                    classification = sweep_resolver.classify(inputs, reaction)
                    if classification is not None:
                        sweep, sweep_timing, sweep_provenance = classification
                        if not isinstance(sweep, SweepKind):
                            raise TypeError("Sweep resolver must return a SweepKind.")
                        if sweep is not SweepKind.UNRESOLVED:
                            if not isinstance(sweep_timing, EvidenceTiming):
                                raise TypeError(
                                    "Resolved sweep classification requires EvidenceTiming."
                                )
                            if (
                                sweep is SweepKind.SWEEP
                                and reaction.kind is not ReactionKind.REJECTION
                            ):
                                raise ValueError(
                                    "Acceptance cannot be classified as a sweep."
                                )
                            sweep_timing.validate_for_subject(inputs.subject)
                            if (
                                sweep_timing.event_timestamp
                                != breach.timing.event_timestamp
                            ):
                                raise ValueError(
                                    "Sweep classification must preserve breach event time."
                                )
                            if sweep_timing.confirmation_timestamp is None:
                                raise ValueError(
                                    "Resolved sweep classification requires confirmation time."
                                )
                            _validate_lineage(
                                sweep_timing,
                                visible_sources,
                                "sweep",
                            )
                            if any(
                                source not in sweep_timing.source_observations
                                for source in reaction.timing.source_observations
                            ):
                                raise ValueError(
                                    "Sweep provenance must retain reaction source observations."
                                )
                            _validate_confirmation_provenance(
                                sweep_timing,
                                reaction.timing.source_observations,
                                "sweep",
                            )
                        else:
                            sweep_timing = None
                            sweep_provenance = {}

        assessments.append(
            CoreV1LiquidityAssessment(
                subject=inputs.subject,
                level=level,
                breach=breach,
                reaction=reaction,
                sweep=sweep,
                sweep_timing=sweep_timing,
                sweep_provenance=sweep_provenance,
            )
        )
    return tuple(assessments)


def _visible_sources(
    inputs: SubjectVisibleInputs,
) -> tuple[EvidenceSourceObservation, ...]:
    return tuple(
        source
        for observation in inputs.base_observations + inputs.context_observations
        for source in observation.timing.source_observations
    )


def _validate_lineage(
    timing: EvidenceTiming,
    visible_sources: tuple[EvidenceSourceObservation, ...],
    label: str,
) -> None:
    if not isinstance(timing, EvidenceTiming):
        raise TypeError(f"{label} requires EvidenceTiming.")
    if any(source not in visible_sources for source in timing.source_observations):
        raise ValueError(
            f"{label} provenance references observations outside the visible inputs."
        )


def _validate_breach(
    subject: ReplaySubject,
    level: CoreV1LiquidityLevel,
    breach: CoreV1LiquidityBreach,
    visible_sources: tuple[EvidenceSourceObservation, ...],
) -> None:
    if not isinstance(breach, CoreV1LiquidityBreach):
        raise TypeError("Breach resolver must return CoreV1LiquidityBreach or None.")
    if not same_subject_id(breach.subject, subject) or breach.level is not level:
        raise ValueError("Breach resolver returned a breach for another subject or level.")
    breach.timing.validate_for_subject(subject)
    _validate_lineage(breach.timing, visible_sources, "breach")
    if any(
        source not in breach.timing.source_observations
        for source in level.timing.source_observations
    ):
        raise ValueError("Breach provenance must retain its level source observations.")
    if not any(
        source not in level.timing.source_observations
        and source.event_timestamp == breach.timing.event_timestamp
        for source in breach.timing.source_observations
    ):
        raise ValueError("Breach provenance must identify its breach-time observation.")


def _validate_reaction(
    subject: ReplaySubject,
    breach: CoreV1LiquidityBreach,
    reaction: CoreV1LiquidityReaction,
    visible_sources: tuple[EvidenceSourceObservation, ...],
) -> None:
    if not isinstance(reaction, CoreV1LiquidityReaction):
        raise TypeError(
            "Reaction resolver must return CoreV1LiquidityReaction or None."
        )
    if not same_subject_id(reaction.subject, subject) or reaction.breach is not breach:
        raise ValueError("Reaction resolver returned a reaction for another breach.")
    reaction.timing.validate_for_subject(subject)
    _validate_lineage(reaction.timing, visible_sources, "reaction")
    if any(
        source not in reaction.timing.source_observations
        for source in breach.timing.source_observations
    ):
        raise ValueError("Reaction provenance must retain breach source observations.")
    _validate_confirmation_provenance(
        reaction.timing,
        breach.timing.source_observations,
        "reaction",
    )


def _validate_confirmation_provenance(
    timing: EvidenceTiming,
    inherited_sources: tuple[EvidenceSourceObservation, ...],
    label: str,
) -> None:
    confirmation = timing.confirmation_timestamp
    if confirmation is None:
        raise ValueError(f"{label.capitalize()} requires a confirmation timestamp.")
    if not any(
        source not in inherited_sources
        and source.event_timestamp == confirmation
        and source.availability_timestamp >= confirmation
        for source in timing.source_observations
    ):
        raise ValueError(
            f"{label.capitalize()} provenance must identify its confirmation observation."
        )


__all__ = [
    "CoreV1LiquidityAssessment",
    "CoreV1LiquidityBreach",
    "CoreV1LiquidityLevel",
    "CoreV1LiquidityReaction",
    "LiquidityBreachResolver",
    "LiquidityLevelResolver",
    "LiquidityReactionResolver",
    "LiquiditySweepResolver",
    "ReactionKind",
    "SweepKind",
    "evaluate_core_v1_liquidity",
]
