from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol

from .core_v1_liquidity import CoreV1LiquidityLevel
from .core_v1_retracement import CoreV1PlannedEntry, CoreV1RetracementResult
from .displacement import Direction
from .evidence_timing import (
    EvidenceSourceObservation,
    EvidenceTiming,
    validate_bounded_mapping_keys,
)
from .market_structure import LiquiditySide
from .replay_subject import ReplaySubject, same_subject_id
from .sniper_setup import PrecisionState


@dataclass(frozen=True)
class CoreV1StructuralInvalidation:
    """Caller-governed structural invalidation; not an executed stop order."""

    subject: ReplaySubject
    invalidation_id: str
    price: float
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not self.invalidation_id:
            raise ValueError("invalidation_id must not be empty.")
        _validate_price(self.price, "invalidation")
        object.__setattr__(self, "price", float(self.price))
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Structural invalidation requires EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        _require_event_source(self.timing, "Structural invalidation")
        if (
            self.timing.confirmation_timestamp is not None
            and self.timing.confirmation_timestamp > self.timing.event_timestamp
            and not any(
                source.event_timestamp == self.timing.confirmation_timestamp
                for source in self.timing.source_observations
            )
        ):
            raise ValueError("Invalidation confirmation requires a timed source observation.")


@dataclass(frozen=True)
class CoreV1OpposingLiquidityTarget:
    """Explicit governed target selection tied to one supplied liquidity level."""

    subject: ReplaySubject
    target_id: str
    liquidity_level: CoreV1LiquidityLevel
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not self.target_id:
            raise ValueError("target_id must not be empty.")
        if not isinstance(self.liquidity_level, CoreV1LiquidityLevel):
            raise TypeError("target must reference a governed CoreV1LiquidityLevel.")
        if not same_subject_id(self.liquidity_level.subject, self.subject):
            raise ValueError("Target and liquidity level must preserve ReplaySubject identity.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Opposing-liquidity target requires EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        level_sources = self.liquidity_level.timing.source_observations
        if any(source not in self.timing.source_observations for source in level_sources):
            raise ValueError("Target provenance must retain its liquidity-level sources.")
        _require_event_source(self.timing, "Opposing-liquidity target")
        if self.timing.event_timestamp != self.liquidity_level.timing.event_timestamp:
            raise ValueError("Target must preserve the liquidity-level event timestamp.")
        if (
            self.timing.confirmation_timestamp is not None
            and self.timing.confirmation_timestamp > self.timing.event_timestamp
            and not any(
                source.event_timestamp == self.timing.confirmation_timestamp
                for source in self.timing.source_observations
            )
        ):
            raise ValueError("Target confirmation requires a timed source observation.")
        if self.timing.availability_timestamp < self.liquidity_level.timing.availability_timestamp:
            raise ValueError("Target availability cannot precede its liquidity level.")

    @property
    def price(self) -> float:
        return float(self.liquidity_level.price)


@dataclass(frozen=True)
class CoreV1HypothesisInputs:
    """Already-qualified entry plus explicit candidate opposing-liquidity levels."""

    planned_entry: CoreV1RetracementResult
    opposing_liquidity_levels: tuple[CoreV1LiquidityLevel, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.planned_entry, CoreV1RetracementResult):
            raise TypeError("planned_entry must be a CoreV1RetracementResult.")
        levels = tuple(self.opposing_liquidity_levels)
        if any(not isinstance(level, CoreV1LiquidityLevel) for level in levels):
            raise TypeError(
                "opposing_liquidity_levels must contain governed liquidity levels."
            )
        if any(
            not same_subject_id(level.subject, self.planned_entry.subject)
            for level in levels
        ):
            raise ValueError("Opposing liquidity levels belong to another ReplaySubject.")
        for level in levels:
            level.timing.validate_for_subject(self.planned_entry.subject)
        object.__setattr__(self, "opposing_liquidity_levels", levels)


@dataclass(frozen=True)
class CoreV1TradeHypothesis:
    """Completed strategy hypothesis, never an order, intent, fill, or position."""

    subject: ReplaySubject
    state: PrecisionState
    availability_timestamp: object
    reason: str
    direction: Direction | None = None
    planned_entry: CoreV1RetracementResult | None = None
    invalidation: CoreV1StructuralInvalidation | None = None
    target: CoreV1OpposingLiquidityTarget | None = None
    provenance: Mapping[str, tuple[EvidenceSourceObservation, ...]] | None = None

    def __post_init__(self) -> None:
        import pandas as pd

        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.state, PrecisionState):
            raise TypeError("state must use the existing VALID/DEVELOPING/INVALID/NO_TRADE states.")
        available = _utc_timestamp(self.availability_timestamp, "availability_timestamp")
        if available != self.subject.availability_timestamp:
            raise ValueError("Hypothesis availability must match its ReplaySubject.")
        object.__setattr__(self, "availability_timestamp", available)

        for name, component in (
            ("planned entry", self.planned_entry),
            ("invalidation", self.invalidation),
            ("target", self.target),
        ):
            if component is not None and not same_subject_id(component.subject, self.subject):
                raise ValueError(f"{name} must preserve the exact ReplaySubject.")

        if self.provenance is not None and not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping.")
        validate_bounded_mapping_keys(self.provenance or {}, "Hypothesis provenance")
        provenance = {
            name: tuple(sources)
            for name, sources in (self.provenance or {}).items()
        }
        if any(
            not isinstance(source, EvidenceSourceObservation)
            for sources in provenance.values()
            for source in sources
        ):
            raise TypeError("Hypothesis provenance must contain source observations.")

        if self.state is PrecisionState.VALID:
            if (
                self.planned_entry is None
                or self.invalidation is None
                or self.target is None
                or self.direction is None
            ):
                raise ValueError(
                    "VALID hypothesis requires direction, planned entry, invalidation, and target."
                )
            if self.planned_entry.state is not PrecisionState.VALID:
                raise ValueError("VALID hypothesis requires a VALID planned-entry result.")
            if not isinstance(self.planned_entry.planned_entry, CoreV1PlannedEntry):
                raise ValueError("VALID hypothesis requires governed planned-entry evidence.")
            if not isinstance(self.direction, Direction):
                raise TypeError("VALID hypothesis direction must be governed Direction.")
            invalid_reason = _validate_hypothesis_geometry(
                self.planned_entry,
                self.invalidation,
                self.target,
                self.direction,
            )
            if invalid_reason is not None:
                raise ValueError(invalid_reason)
            _validate_component_timing(
                self.planned_entry,
                self.invalidation,
                self.target,
            )
            expected_provenance = _build_provenance(
                self.planned_entry,
                self.invalidation,
                self.target,
            )
            if provenance != dict(expected_provenance):
                raise ValueError("VALID hypothesis provenance is incomplete or inconsistent.")

        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @property
    def planned_entry_price(self) -> float | None:
        return (
            self.planned_entry.planned_entry_price
            if self.planned_entry is not None
            else None
        )

    @property
    def invalidation_price(self) -> float | None:
        return self.invalidation.price if self.invalidation is not None else None

    @property
    def target_price(self) -> float | None:
        return self.target.price if self.target is not None else None

    @property
    def planned_entry_timestamp(self):
        if self.planned_entry is None or self.planned_entry.planned_entry is None:
            return None
        return self.planned_entry.planned_entry.timing.event_timestamp


class StructuralInvalidationResolver(Protocol):
    """Governed invalidation selection; no offset or stop algorithm is implied."""

    def resolve(
        self,
        inputs: CoreV1HypothesisInputs,
    ) -> CoreV1StructuralInvalidation | None: ...


class OpposingLiquidityTargetResolver(Protocol):
    """Governed target selection; no nearest-level or RR policy is implied."""

    def resolve(
        self,
        inputs: CoreV1HypothesisInputs,
    ) -> CoreV1OpposingLiquidityTarget | None: ...


def evaluate_core_v1_hypothesis(
    planned_entry: CoreV1RetracementResult,
    *,
    opposing_liquidity_levels: tuple[CoreV1LiquidityLevel, ...] = (),
    invalidation: CoreV1StructuralInvalidation | None = None,
    invalidation_resolver: StructuralInvalidationResolver | None = None,
    target: CoreV1OpposingLiquidityTarget | None = None,
    target_resolver: OpposingLiquidityTargetResolver | None = None,
) -> CoreV1TradeHypothesis:
    """Complete a valid planned-entry hypothesis only from governed components."""
    if not isinstance(planned_entry, CoreV1RetracementResult):
        raise TypeError("planned_entry must be a CoreV1RetracementResult.")
    if invalidation is not None and invalidation_resolver is not None:
        raise ValueError("Supply an invalidation or resolver, not both.")
    if target is not None and target_resolver is not None:
        raise ValueError("Supply a target or resolver, not both.")

    subject = planned_entry.subject
    if planned_entry.state is not PrecisionState.VALID:
        return _incomplete(
            planned_entry,
            "DEVELOPING: a valid governed planned entry is required.",
        )
    if not isinstance(planned_entry.planned_entry, CoreV1PlannedEntry):
        raise ValueError("VALID retracement result does not contain a planned entry.")
    if planned_entry.mss is None or planned_entry.liquidity is None or planned_entry.displacement is None:
        raise ValueError("VALID planned entry lacks its required governed sequence.")

    resolver_inputs = CoreV1HypothesisInputs(
        planned_entry=planned_entry,
        opposing_liquidity_levels=opposing_liquidity_levels,
    )
    invalidation_candidate = invalidation
    if invalidation_candidate is None and invalidation_resolver is not None:
        invalidation_candidate = invalidation_resolver.resolve(resolver_inputs)
    target_candidate = target
    if target_candidate is None and target_resolver is not None:
        target_candidate = target_resolver.resolve(resolver_inputs)

    if invalidation_candidate is not None and not isinstance(
        invalidation_candidate,
        CoreV1StructuralInvalidation,
    ):
        raise TypeError(
            "Invalidation resolver must return CoreV1StructuralInvalidation or None."
        )
    if target_candidate is not None and not isinstance(
        target_candidate,
        CoreV1OpposingLiquidityTarget,
    ):
        raise TypeError(
            "Target resolver must return CoreV1OpposingLiquidityTarget or None."
        )
    if invalidation_candidate is None:
        return _incomplete(
            planned_entry,
            "DEVELOPING: governed structural invalidation is unresolved.",
            target=target_candidate,
        )
    if target_candidate is None:
        return _incomplete(
            planned_entry,
            "DEVELOPING: opposing-liquidity target is unresolved.",
            invalidation=invalidation_candidate,
        )
    if not any(
        target_candidate.liquidity_level is level
        for level in resolver_inputs.opposing_liquidity_levels
    ):
        return _incomplete(
            planned_entry,
            "DEVELOPING: target does not reference a supplied governed opposing-liquidity level.",
            invalidation=invalidation_candidate,
            target=target_candidate,
        )
    _validate_component_timing(planned_entry, invalidation_candidate, target_candidate)

    direction = _entry_direction(planned_entry)
    invalid_reason = _validate_hypothesis_geometry(
        planned_entry,
        invalidation_candidate,
        target_candidate,
        direction,
    )
    if invalid_reason is not None:
        return _invalid(
            planned_entry,
            invalidation_candidate,
            target_candidate,
            direction,
            invalid_reason,
        )

    provenance = _build_provenance(
        planned_entry,
        invalidation_candidate,
        target_candidate,
    )
    return CoreV1TradeHypothesis(
        subject=subject,
        state=PrecisionState.VALID,
        availability_timestamp=subject.availability_timestamp,
        reason="Governed planned entry, structural invalidation, and opposing-liquidity target are coherent.",
        direction=direction,
        planned_entry=planned_entry,
        invalidation=invalidation_candidate,
        target=target_candidate,
        provenance=provenance,
    )


def _entry_direction(planned_entry: CoreV1RetracementResult):
    displacement = planned_entry.displacement
    if displacement is None or displacement.observation is None:
        raise ValueError("VALID planned entry has no governed displacement direction.")
    direction = displacement.observation.direction
    if direction is None:
        raise ValueError("VALID planned entry has unresolved direction.")
    if not isinstance(direction, Direction):
        raise ValueError("Planned-entry direction must be governed Direction.")
    structural_direction = (
        planned_entry.mss.structural_break.direction
        if planned_entry.mss is not None
        and planned_entry.mss.structural_break is not None
        else None
    )
    if structural_direction is not None and structural_direction is not direction:
        raise ValueError("MSS structural-break direction conflicts with displacement.")
    return direction


def _validate_hypothesis_geometry(
    planned_entry: CoreV1RetracementResult,
    invalidation: CoreV1StructuralInvalidation,
    target: CoreV1OpposingLiquidityTarget,
    direction,
) -> str | None:
    planned_price = planned_entry.planned_entry_price
    if planned_price is None:
        return "INVALID: planned entry is missing."
    _validate_price(planned_price, "planned entry")
    entry_price = float(planned_price)
    _validate_price(invalidation.price, "invalidation")
    _validate_price(target.price, "target")

    if not same_subject_id(invalidation.subject, planned_entry.subject):
        raise ValueError("Invalidation belongs to another ReplaySubject.")
    if not same_subject_id(target.subject, planned_entry.subject):
        raise ValueError("Target belongs to another ReplaySubject.")

    if direction.value == "BULLISH":
        expected_target_side = LiquiditySide.BSL
        if invalidation.price >= entry_price:
            return "INVALID: LONG invalidation must be below planned entry."
        if target.price <= entry_price:
            return "INVALID: LONG opposing-liquidity target must be above planned entry."
    else:
        expected_target_side = LiquiditySide.SSL
        if invalidation.price <= entry_price:
            return "INVALID: SHORT invalidation must be above planned entry."
        if target.price >= entry_price:
            return "INVALID: SHORT opposing-liquidity target must be below planned entry."

    if target.liquidity_level.side is not expected_target_side:
        return "INVALID: target liquidity is not on the opposing side for the planned direction."
    if target.price != float(target.liquidity_level.price):
        return "INVALID: target price does not match its referenced liquidity level."
    return None


def _validate_component_timing(
    planned_entry: CoreV1RetracementResult,
    invalidation: CoreV1StructuralInvalidation,
    target: CoreV1OpposingLiquidityTarget,
) -> None:
    subject = planned_entry.subject
    entry = planned_entry.planned_entry
    assert entry is not None
    entry.timing.validate_for_subject(subject)
    invalidation.timing.validate_for_subject(subject)
    target.timing.validate_for_subject(subject)
    target.liquidity_level.timing.validate_for_subject(subject)

    for label, timing in (
        ("planned entry", entry.timing),
        ("invalidation", invalidation.timing),
        ("target", target.timing),
        ("target liquidity level", target.liquidity_level.timing),
    ):
        if timing.availability_timestamp > subject.availability_timestamp:
            raise ValueError(f"{label} is unavailable at the ReplaySubject.")
        if timing.base_bar_scoped:
            if timing.event_timestamp > subject.base_bar_timestamp:
                raise ValueError(f"{label} event exceeds the base-bar cutoff.")
            if (
                timing.confirmation_timestamp is not None
                and timing.confirmation_timestamp > subject.base_bar_timestamp
            ):
                raise ValueError(f"{label} confirmation exceeds the base-bar cutoff.")
        for source in timing.source_observations:
            if source.is_base_timeframe and (
                source.event_timestamp > subject.base_bar_timestamp
                or source.base_bar_timestamp > subject.base_bar_timestamp
            ):
                raise ValueError(f"{label} source exceeds the base-bar cutoff.")

    if (
        target.timing.confirmation_timestamp is not None
        and target.timing.confirmation_timestamp > subject.availability_timestamp
    ):
        raise ValueError("Target confirmation exceeds the ReplaySubject availability cutoff.")


def _require_event_source(timing: EvidenceTiming, label: str) -> None:
    if not any(
        source.event_timestamp == timing.event_timestamp
        for source in timing.source_observations
    ):
        raise ValueError(f"{label} provenance must identify its event observation.")


def _build_provenance(
    planned_entry: CoreV1RetracementResult,
    invalidation: CoreV1StructuralInvalidation,
    target: CoreV1OpposingLiquidityTarget,
) -> Mapping[str, tuple[EvidenceSourceObservation, ...]]:
    result = {
        name: tuple(sources)
        for name, sources in planned_entry.provenance.items()
    }
    if planned_entry.planned_entry is not None:
        result["planned_entry"] = planned_entry.planned_entry.timing.source_observations
    result["invalidation"] = invalidation.timing.source_observations
    result["opposing_liquidity_target"] = target.timing.source_observations
    result["target_liquidity_level"] = target.liquidity_level.timing.source_observations
    return result


def _incomplete(
    planned_entry: CoreV1RetracementResult,
    reason: str,
    *,
    invalidation: CoreV1StructuralInvalidation | None = None,
    target: CoreV1OpposingLiquidityTarget | None = None,
) -> CoreV1TradeHypothesis:
    provenance = {
        name: tuple(sources)
        for name, sources in planned_entry.provenance.items()
    }
    if planned_entry.planned_entry is not None:
        provenance["planned_entry"] = planned_entry.planned_entry.timing.source_observations
    if invalidation is not None:
        provenance["invalidation"] = invalidation.timing.source_observations
    if target is not None:
        provenance["opposing_liquidity_target"] = target.timing.source_observations
        provenance["target_liquidity_level"] = target.liquidity_level.timing.source_observations
    return CoreV1TradeHypothesis(
        subject=planned_entry.subject,
        state=PrecisionState.DEVELOPING,
        availability_timestamp=planned_entry.subject.availability_timestamp,
        reason=reason,
        planned_entry=planned_entry,
        invalidation=invalidation,
        target=target,
        provenance=provenance,
    )


def _invalid(
    planned_entry: CoreV1RetracementResult,
    invalidation: CoreV1StructuralInvalidation,
    target: CoreV1OpposingLiquidityTarget,
    direction,
    reason: str,
) -> CoreV1TradeHypothesis:
    return CoreV1TradeHypothesis(
        subject=planned_entry.subject,
        state=PrecisionState.INVALID,
        availability_timestamp=planned_entry.subject.availability_timestamp,
        reason=reason,
        direction=direction,
        planned_entry=planned_entry,
        invalidation=invalidation,
        target=target,
        provenance=_build_provenance(planned_entry, invalidation, target),
    )


def _validate_price(value: float, name: str) -> None:
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} price must be finite.") from exc
    if not isfinite(price):
        raise ValueError(f"{name} price must be finite.")


def _utc_timestamp(value: object, name: str):
    import pandas as pd

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


__all__ = [
    "CoreV1HypothesisInputs",
    "CoreV1OpposingLiquidityTarget",
    "CoreV1StructuralInvalidation",
    "CoreV1TradeHypothesis",
    "OpposingLiquidityTargetResolver",
    "StructuralInvalidationResolver",
    "evaluate_core_v1_hypothesis",
]
