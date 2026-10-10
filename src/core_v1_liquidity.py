from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol

import pandas as pd

from .core_v1_evidence import SubjectVisibleInputs
from .evidence_timing import EvidenceSourceObservation, EvidenceTiming
from .market_structure import LiquidityLevel as CanonicalLiquidityLevel
from .market_structure import (
    LiquiditySide,
    SwingType,
    detect_confirmed_swings,
    swings_to_liquidity,
)
from .replay_subject import ReplaySubject, same_subject_id


class ReactionKind(str, Enum):
    REJECTION = "REJECTION"
    ACCEPTANCE = "ACCEPTANCE"


class SweepKind(str, Enum):
    SWEEP = "SWEEP"
    NOT_SWEEP = "NOT_SWEEP"
    UNRESOLVED = "UNRESOLVED"


class LiquidityEventDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass(frozen=True)
class CoreV1LiquidityLevel:
    """Detector output for a BSL/SSL level, with subject timing and lineage.

    Instances are not trusted merely because they have this type. The liquidity
    resolver recomputes their swing from the current subject's visible bars.
    """

    subject: ReplaySubject
    level_id: str
    side: LiquiditySide
    price: float
    timing: EvidenceTiming
    origin: CanonicalLiquidityLevel | None = None
    origin_source_id: str | None = None

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
        if (self.origin is None) != (self.origin_source_id is None):
            raise ValueError(
                "Liquidity level origin and origin_source_id must be supplied together."
            )
        if self.origin is not None:
            _validate_level_origin_contract(self)


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
        if self.timing.event_timestamp <= self.level.timing.availability_timestamp:
            raise ValueError("Breach event must occur strictly after level availability.")
        if self.price is None or not isfinite(float(self.price)):
            raise ValueError("Breach price must be finite and observed.")
        if (
            self.level.side is LiquiditySide.BSL
            and float(self.price) <= self.level.price
        ):
            raise ValueError("A BSL breach price must be strictly above its level.")
        if (
            self.level.side is LiquiditySide.SSL
            and float(self.price) >= self.level.price
        ):
            raise ValueError("An SSL breach price must be strictly below its level.")


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
            self.timing.confirmation_timestamp
            <= self.breach.timing.event_timestamp
        ):
            raise ValueError("Reaction must be observed strictly after the breach.")
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
        if self.kind is ReactionKind.REJECTION:
            if self.price is None:
                raise ValueError("A rejection requires an observed reaction price.")
            side = self.breach.level.side
            if (
                side is LiquiditySide.BSL
                and float(self.price) > self.breach.level.price
            ):
                raise ValueError("A BSL rejection must return through its level.")
            if (
                side is LiquiditySide.SSL
                and float(self.price) < self.breach.level.price
            ):
                raise ValueError("An SSL rejection must return through its level.")


@dataclass(frozen=True)
class CoreV1LiquidityAssessment:
    """Liquidity-stage state; event direction is distinct from trade direction."""

    subject: ReplaySubject
    level: CoreV1LiquidityLevel
    breach: CoreV1LiquidityBreach | None = None
    reaction: CoreV1LiquidityReaction | None = None
    sweep: SweepKind = SweepKind.UNRESOLVED
    sweep_timing: EvidenceTiming | None = None
    sweep_provenance: Mapping[str, object] | None = None

    @property
    def event_direction(self) -> LiquidityEventDirection | None:
        if self.sweep is not SweepKind.SWEEP:
            return None
        return (
            LiquidityEventDirection.BEARISH
            if self.level.side is LiquiditySide.BSL
            else LiquidityEventDirection.BULLISH
        )

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
            if self.sweep is SweepKind.SWEEP and self.level.origin is None:
                raise ValueError("A confirmed sweep requires a canonical liquidity origin.")
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
                allow_inherited=True,
            )
        elif self.sweep_timing is not None:
            raise ValueError("Unresolved sweep status cannot have confirmation timing.")
        provenance = {} if self.sweep_provenance is None else dict(self.sweep_provenance)
        object.__setattr__(self, "sweep_provenance", MappingProxyType(provenance))


class LiquidityLevelResolver(Protocol):
    """Optional selector whose proposed levels are independently recomputed."""

    def resolve(
        self,
        inputs: SubjectVisibleInputs,
    ) -> tuple[CoreV1LiquidityLevel, ...]: ...


def detect_core_v1_liquidity_levels(
    inputs: SubjectVisibleInputs,
) -> tuple[CoreV1LiquidityLevel, ...]:
    """Derive confirmed swing liquidity only from this subject's visible base bars."""
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")

    bars: list[dict[str, float]] = []
    sources: list[EvidenceSourceObservation] = []
    for observation in inputs.base_observations:
        if (
            observation.timing.event_timestamp
            > inputs.subject.base_bar_timestamp
            or observation.timing.availability_timestamp
            > inputs.subject.availability_timestamp
        ):
            raise ValueError(
                "Visible base observation exceeds the ReplaySubject cutoff."
            )
        if observation.timing.confirmation_timestamp is not None and (
            observation.timing.confirmation_timestamp
            > inputs.subject.base_bar_timestamp
        ):
            raise ValueError(
                "Visible base observation confirmation exceeds the base-bar cutoff."
            )
        if not isinstance(observation.value, Mapping):
            return ()
        values = {str(key).lower(): value for key, value in observation.value.items()}
        if not {"open", "high", "low", "close"}.issubset(values):
            return ()
        if len(observation.timing.source_observations) != 1:
            return ()
        source = observation.timing.source_observations[0]
        if not source.is_base_timeframe or not source.provenance:
            return ()
        if (
            source.event_timestamp > inputs.subject.base_bar_timestamp
            or source.availability_timestamp
            > inputs.subject.availability_timestamp
            or (
                source.base_bar_timestamp is not None
                and source.base_bar_timestamp
                > inputs.subject.base_bar_timestamp
            )
        ):
            raise ValueError(
                "Visible base source exceeds the ReplaySubject cutoff."
            )
        try:
            bar = {
                name: float(values[name])
                for name in ("open", "high", "low", "close")
            }
        except (TypeError, ValueError, OverflowError):
            return ()
        if any(not isfinite(value) for value in bar.values()):
            return ()
        if (
            bar["high"] < max(bar["open"], bar["close"], bar["low"])
            or bar["low"] > min(bar["open"], bar["close"], bar["high"])
        ):
            return ()
        bars.append(bar)
        sources.append(source)

    timestamps = [source.event_timestamp for source in sources]
    if len(timestamps) < 5 or any(
        right <= left for left, right in zip(timestamps, timestamps[1:])
    ):
        return ()

    frame = pd.DataFrame(
        bars,
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )
    frame.columns = [column.title() for column in frame.columns]
    output: list[CoreV1LiquidityLevel] = []
    for swing in detect_confirmed_swings(frame):
        if swing.confirmation_timestamp > inputs.subject.base_bar_timestamp:
            continue
        expected_side = (
            LiquiditySide.BSL
            if swing.kind is SwingType.HIGH
            else LiquiditySide.SSL
        )
        support_sources = tuple(
            sources[swing.source_index - 2 : swing.source_index + 3]
        )
        if len(support_sources) != 5:
            continue
        availability = max(
            source.availability_timestamp for source in support_sources
        )
        if availability > inputs.subject.availability_timestamp:
            continue

        (origin,) = swings_to_liquidity((swing,))
        origin = replace(origin, availability_timestamp=availability)
        timing = EvidenceTiming.from_inputs(
            event_timestamp=swing.timestamp,
            confirmation_timestamp=swing.confirmation_timestamp,
            source_observations=support_sources,
            provenance={"resolver": "core-v1-visible-swing-detector"},
            base_bar_scoped=True,
        )
        output.append(
            CoreV1LiquidityLevel(
                subject=inputs.subject,
                level_id=(
                    f"swing:{swing.kind.value}:{support_sources[0].source_id}:"
                    f"{support_sources[-1].source_id}"
                ),
                side=expected_side,
                price=swing.price,
                timing=timing,
                origin=origin,
                origin_source_id=support_sources[-1].source_id,
            )
        )
    return tuple(output)


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
    if (
        breach_resolver is None
        and reaction_resolver is None
        and sweep_resolver is None
    ):
        return _resolve_visible_liquidity(inputs, selected)

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
        if not _validate_level_origin_visibility(level, inputs):
            assessments.append(
                CoreV1LiquidityAssessment(subject=inputs.subject, level=level)
            )
            continue

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
                    inputs,
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
                                allow_inherited=True,
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


def _resolve_visible_liquidity(
    inputs: SubjectVisibleInputs,
    levels: tuple[CoreV1LiquidityLevel, ...],
) -> tuple[CoreV1LiquidityAssessment, ...]:
    visible_sources = _visible_sources(inputs)
    seen_ids: set[str] = set()
    eligible: list[CoreV1LiquidityLevel] = []
    unresolved: list[CoreV1LiquidityAssessment] = []
    for level in levels:
        if not isinstance(level, CoreV1LiquidityLevel):
            raise TypeError("levels must contain CoreV1LiquidityLevel values.")
        if not same_subject_id(level.subject, inputs.subject):
            raise ValueError("Liquidity level belongs to a different ReplaySubject.")
        if level.level_id in seen_ids:
            raise ValueError(f"Duplicate liquidity level_id: {level.level_id}")
        seen_ids.add(level.level_id)
        _validate_lineage(level.timing, visible_sources, "level")
        if _validate_level_origin_visibility(level, inputs):
            eligible.append(level)
        else:
            unresolved.append(
                CoreV1LiquidityAssessment(subject=inputs.subject, level=level)
            )

    observations = _visible_price_observations(inputs)
    if not observations or not eligible:
        return tuple(unresolved) + tuple(
            CoreV1LiquidityAssessment(subject=inputs.subject, level=level)
            for level in eligible
        )

    candidates: list[
        tuple[CoreV1LiquidityLevel, CoreV1LiquidityBreach, float]
    ] = []
    for level in eligible:
        baseline = next(
            (
                item
                for item in reversed(observations)
                if item[0].event_timestamp <= level.timing.availability_timestamp
                and item[0].availability_timestamp
                <= level.timing.availability_timestamp
            ),
            None,
        )
        if baseline is None:
            unresolved.append(
                CoreV1LiquidityAssessment(subject=inputs.subject, level=level)
            )
            continue

        previous_close = baseline[1]["close"]
        for source, prices in observations:
            if source.event_timestamp <= level.timing.availability_timestamp:
                continue
            if source.availability_timestamp < level.timing.availability_timestamp:
                continue
            if level.side is LiquiditySide.BSL:
                breached = previous_close <= level.price and prices["high"] > level.price
                breach_price = prices["high"]
            else:
                breached = previous_close >= level.price and prices["low"] < level.price
                breach_price = prices["low"]
            if breached:
                timing = EvidenceTiming.from_inputs(
                    event_timestamp=source.event_timestamp,
                    source_observations=(
                        *level.timing.source_observations,
                        source,
                    ),
                    provenance={
                        "resolver": "point-in-time-core-v1-liquidity",
                        "stage": "breach",
                        "level_id": level.level_id,
                    },
                    base_bar_scoped=True,
                )
                breach = CoreV1LiquidityBreach(
                    subject=inputs.subject,
                    level=level,
                    price=breach_price,
                    timing=timing,
                )
                candidates.append(
                    (level, breach, abs(level.price - previous_close))
                )
                break
            previous_close = prices["close"]

    if not candidates:
        return tuple(unresolved) + tuple(
            CoreV1LiquidityAssessment(subject=inputs.subject, level=level)
            for level in eligible
        )

    earliest = min(
        breach.timing.event_timestamp for _, breach, _ in candidates
    )
    same_time = [
        item for item in candidates if item[1].timing.event_timestamp == earliest
    ]
    nearest_distance = min(distance for _, _, distance in same_time)
    nearest = [
        item for item in same_time if item[2] == nearest_distance
    ]
    if len({level.side for level, _, _ in same_time}) > 1 or len(nearest) != 1:
        unresolved.extend(_ambiguous_assessments(inputs.subject, tuple(eligible)))
        return tuple(unresolved)

    selected, breach, _ = nearest[0]
    reaction = _visible_reaction(inputs, selected, breach, observations)
    if reaction is None:
        unresolved.append(
            CoreV1LiquidityAssessment(
                subject=inputs.subject,
                level=selected,
                breach=breach,
            )
        )
        return tuple(unresolved)
    return tuple(unresolved) + (
        CoreV1LiquidityAssessment(
            subject=inputs.subject,
            level=selected,
            breach=breach,
            reaction=reaction,
            sweep=SweepKind.SWEEP,
            sweep_timing=reaction.timing,
            sweep_provenance={
                "level_id": selected.level_id,
                "level_side": selected.side.value,
                "breach_price": breach.price,
                "reaction_price": reaction.price,
            },
        ),
    )


def _visible_price_observations(
    inputs: SubjectVisibleInputs,
) -> tuple[tuple[EvidenceSourceObservation, dict[str, float]], ...]:
    observations: list[tuple[EvidenceSourceObservation, dict[str, float]]] = []
    for visible in inputs.base_observations:
        if not isinstance(visible.value, Mapping):
            return ()
        values = {str(key).lower(): value for key, value in visible.value.items()}
        if not {"high", "low", "close"}.issubset(values):
            return ()
        if len(visible.timing.source_observations) != 1:
            return ()
        source = visible.timing.source_observations[0]
        try:
            prices = {
                name: float(values[name])
                for name in ("high", "low", "close")
            }
        except (TypeError, ValueError, OverflowError):
            return ()
        if any(not isfinite(value) for value in prices.values()):
            return ()
        if prices["high"] < prices["low"]:
            return ()
        observations.append((source, prices))

    timestamps = [source.event_timestamp for source, _ in observations]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        return ()
    return tuple(observations)


def _visible_reaction(
    inputs: SubjectVisibleInputs,
    level: CoreV1LiquidityLevel,
    breach: CoreV1LiquidityBreach,
    observations: tuple[tuple[EvidenceSourceObservation, dict[str, float]], ...],
) -> CoreV1LiquidityReaction | None:
    for source, prices in observations:
        if source.event_timestamp <= breach.timing.event_timestamp:
            continue
        if source.availability_timestamp < breach.timing.availability_timestamp:
            continue
        if level.side is LiquiditySide.BSL and prices["low"] <= level.price:
            reaction_price = prices["low"]
        elif level.side is LiquiditySide.SSL and prices["high"] >= level.price:
            reaction_price = prices["high"]
        else:
            continue

        timing = EvidenceTiming.from_inputs(
            event_timestamp=breach.timing.event_timestamp,
            confirmation_timestamp=source.event_timestamp,
            source_observations=(
                *breach.timing.source_observations,
                source,
            ),
            provenance={
                "resolver": "point-in-time-core-v1-liquidity",
                "stage": "reaction",
                "level_id": level.level_id,
            },
            base_bar_scoped=True,
        )
        return CoreV1LiquidityReaction(
            subject=inputs.subject,
            breach=breach,
            kind=ReactionKind.REJECTION,
            timing=timing,
            price=reaction_price,
        )
    return None


def _ambiguous_assessments(
    subject: ReplaySubject,
    levels: tuple[CoreV1LiquidityLevel, ...],
) -> tuple[CoreV1LiquidityAssessment, ...]:
    return tuple(
        CoreV1LiquidityAssessment(
            subject=subject,
            level=level,
            sweep_provenance={
                "resolution": "UNRESOLVED",
                "reason": "AMBIGUOUS_LEVEL_SELECTION",
            },
        )
        for level in levels
    )


def _validate_level_origin_contract(level: CoreV1LiquidityLevel) -> None:
    origin = level.origin
    if not isinstance(origin, CanonicalLiquidityLevel):
        raise TypeError("origin must be a canonical market_structure.LiquidityLevel.")
    if type(level.origin_source_id) is not str or not level.origin_source_id:
        raise ValueError("Canonical liquidity origin requires a source observation ID.")
    if origin.side is not level.side:
        raise ValueError("Liquidity level side does not match its canonical origin.")
    if float(origin.price) != float(level.price):
        raise ValueError("Liquidity level price does not match its canonical origin.")
    if origin.source != "SWING":
        raise ValueError(
            "Canonical liquidity origin must use the existing SWING source."
        )
    if origin.confirmation_timestamp is None:
        raise ValueError("Canonical liquidity origin requires confirmation time.")
    if origin.availability_timestamp is None:
        raise ValueError("Canonical liquidity origin requires availability time.")
    source_matches = tuple(
        source
        for source in level.timing.source_observations
        if source.source_id == level.origin_source_id
    )
    if len(source_matches) != 1:
        raise ValueError(
            "Liquidity origin source ID must identify exactly one level observation."
        )
    source = source_matches[0]
    if (
        source.event_timestamp != origin.confirmation_timestamp
        or source.availability_timestamp != origin.availability_timestamp
    ):
        raise ValueError("Liquidity origin timing does not match its source observation.")
    if (
        level.timing.event_timestamp != origin.timestamp
        or level.timing.confirmation_timestamp != origin.confirmation_timestamp
        or level.timing.availability_timestamp != origin.availability_timestamp
    ):
        raise ValueError("Liquidity level timing does not match its canonical origin.")


def _validate_level_origin_visibility(
    level: CoreV1LiquidityLevel,
    inputs: SubjectVisibleInputs,
) -> bool:
    if level.origin is None or level.origin_source_id is None:
        return False

    for expected in detect_core_v1_liquidity_levels(inputs):
        if (
            level.side is expected.side
            and float(level.price) == float(expected.price)
            and level.origin == expected.origin
            and level.origin_source_id == expected.origin_source_id
            and level.timing.event_timestamp == expected.timing.event_timestamp
            and level.timing.confirmation_timestamp
            == expected.timing.confirmation_timestamp
            and level.timing.availability_timestamp
            == expected.timing.availability_timestamp
            and level.timing.base_bar_scoped
            == expected.timing.base_bar_scoped
            and level.timing.source_observations
            == expected.timing.source_observations
            and level.timing.provenance == expected.timing.provenance
        ):
            return True
    return False


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
    inputs: SubjectVisibleInputs,
    breach: CoreV1LiquidityBreach,
    reaction: CoreV1LiquidityReaction,
    visible_sources: tuple[EvidenceSourceObservation, ...],
) -> None:
    if not isinstance(reaction, CoreV1LiquidityReaction):
        raise TypeError(
            "Reaction resolver must return CoreV1LiquidityReaction or None."
        )
    if (
        not same_subject_id(reaction.subject, inputs.subject)
        or reaction.breach is not breach
    ):
        raise ValueError("Reaction resolver returned a reaction for another breach.")
    reaction.timing.validate_for_subject(inputs.subject)
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
    if reaction.kind is ReactionKind.REJECTION:
        confirmation = reaction.timing.confirmation_timestamp
        observed_price_key = (
            "low" if breach.level.side is LiquiditySide.BSL else "high"
        )
        confirmation_observations = tuple(
            observation
            for observation in inputs.base_observations
            if observation.timing.event_timestamp == confirmation
        )
        if len(confirmation_observations) != 1:
            raise ValueError(
                "Rejection confirmation must identify exactly one visible base observation."
            )
        observed_price = None
        observation = confirmation_observations[0]
        if isinstance(observation.value, Mapping):
            values = {
                str(key).lower(): value
                for key, value in observation.value.items()
            }
            if observed_price_key in values:
                try:
                    observed_price = float(values[observed_price_key])
                except (TypeError, ValueError, OverflowError):
                    pass
        if observed_price is None or reaction.price != observed_price:
            raise ValueError(
                "Rejection price must match the observed reaction bar price."
            )


def _validate_confirmation_provenance(
    timing: EvidenceTiming,
    inherited_sources: tuple[EvidenceSourceObservation, ...],
    label: str,
    *,
    allow_inherited: bool = False,
) -> None:
    confirmation = timing.confirmation_timestamp
    if confirmation is None:
        raise ValueError(f"{label.capitalize()} requires a confirmation timestamp.")
    if not any(
        (allow_inherited or source not in inherited_sources)
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
    "detect_core_v1_liquidity_levels",
    "LiquidityBreachResolver",
    "LiquidityLevelResolver",
    "LiquidityReactionResolver",
    "LiquiditySweepResolver",
    "ReactionKind",
    "SweepKind",
    "evaluate_core_v1_liquidity",
]
