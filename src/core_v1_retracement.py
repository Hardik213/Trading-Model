from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol, TypeAlias

from .core_v1_dealing_range import DealingRangeContext
from .core_v1_displacement import (
    CoreV1DisplacementAssessment,
    DisplacementStatus,
)
from .core_v1_evidence import SubjectVisibleInputs
from .core_v1_liquidity import CoreV1LiquidityAssessment, CoreV1LiquidityLevel
from .core_v1_mss import CoreV1MSSResult, MSSStatus
from .core_v1_reactionary_zones import (
    CoreV1FVGObservation,
    CoreV1OrderBlockObservation,
    CoreV1PDArrayObservation,
)
from .evidence_timing import EvidenceSourceObservation, EvidenceTiming
from .replay_subject import ReplaySubject, same_subject_id
# Reuse these four state labels only; no precision-gate rules are imported.
from .sniper_setup import PrecisionState


ReactionaryZone: TypeAlias = (
    CoreV1FVGObservation
    | CoreV1PDArrayObservation
    | CoreV1OrderBlockObservation
)


@dataclass(frozen=True)
class CoreV1ZoneInteraction:
    """A timed zone interaction observation; interaction alone is not an entry."""

    subject: ReplaySubject
    zone: ReactionaryZone
    timing: EvidenceTiming
    price: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(
            self.zone,
            (
                CoreV1FVGObservation,
                CoreV1PDArrayObservation,
                CoreV1OrderBlockObservation,
            ),
        ):
            raise TypeError("zone must be a governed FVG, PD-array, or OB observation.")
        if not same_subject_id(self.zone.subject, self.subject):
            raise ValueError("Zone interaction must preserve the zone ReplaySubject.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Zone interaction requires EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        _validate_base_bar_scope(self.timing, self.subject, "Zone interaction")
        if self.timing.event_timestamp < _zone_timing(self.zone).availability_timestamp:
            raise ValueError("Zone interaction cannot precede zone availability.")
        if self.price is not None and not isfinite(float(self.price)):
            raise ValueError("Zone interaction price must be finite.")
        zone_sources = _zone_timing(self.zone).source_observations
        if any(source not in self.timing.source_observations for source in zone_sources):
            raise ValueError("Zone interaction provenance must retain zone provenance.")
        if not any(
            source.event_timestamp == self.timing.event_timestamp
            and source not in zone_sources
            for source in self.timing.source_observations
        ):
            raise ValueError("Zone interaction provenance must identify its event.")

    @property
    def interaction_timestamp(self):
        return self.timing.event_timestamp


@dataclass(frozen=True)
class CoreV1PlannedEntry:
    """A strategy-level planned entry, never an execution price or fill."""

    subject: ReplaySubject
    planned_entry_price: float
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isfinite(float(self.planned_entry_price)):
            raise ValueError("Planned entry price must be finite.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Planned entry requires EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        if not any(
            source.event_timestamp == self.timing.event_timestamp
            for source in self.timing.source_observations
        ):
            raise ValueError("Planned entry provenance must identify its event.")


@dataclass(frozen=True)
class CoreV1RetracementInputs:
    """Governed prerequisites supplied to an entry-policy resolver."""

    visible_inputs: SubjectVisibleInputs
    mss: CoreV1MSSResult | None
    liquidity: CoreV1LiquidityAssessment | None
    displacement: CoreV1DisplacementAssessment | None
    context: DealingRangeContext | None = None
    draw_on_liquidity: CoreV1LiquidityLevel | None = None
    zone: ReactionaryZone | None = None
    interaction: CoreV1ZoneInteraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.visible_inputs, SubjectVisibleInputs):
            raise TypeError("visible_inputs must be SubjectVisibleInputs.")
        subject = self.visible_inputs.subject
        for label, component in (
            ("MSS", self.mss),
            ("liquidity", self.liquidity),
            ("displacement", self.displacement),
        ):
            if component is not None and not same_subject_id(component.subject, subject):
                raise ValueError(f"{label} belongs to another ReplaySubject.")
        if self.context is not None and not same_subject_id(self.context.subject, subject):
            raise ValueError("Context belongs to another ReplaySubject.")
        if self.draw_on_liquidity is not None:
            if not isinstance(self.draw_on_liquidity, CoreV1LiquidityLevel):
                raise TypeError("draw_on_liquidity must be a governed liquidity level.")
            if not same_subject_id(self.draw_on_liquidity.subject, subject):
                raise ValueError("Draw-on-liquidity belongs to another ReplaySubject.")
        if self.zone is not None and not same_subject_id(self.zone.subject, subject):
            raise ValueError("Reactionary zone belongs to another ReplaySubject.")
        if self.interaction is not None:
            if not same_subject_id(self.interaction.subject, subject):
                raise ValueError("Zone interaction belongs to another ReplaySubject.")
            if self.zone is None or self.interaction.zone is not self.zone:
                raise ValueError("Zone interaction must reference the supplied zone identity.")


@dataclass(frozen=True)
class CoreV1RetracementResult:
    """Entry-qualification evidence; VALID does not mean submitted or filled."""

    subject: ReplaySubject
    state: PrecisionState
    availability_timestamp: object
    reason: str
    planned_entry: CoreV1PlannedEntry | None = None
    mss: CoreV1MSSResult | None = None
    liquidity: CoreV1LiquidityAssessment | None = None
    displacement: CoreV1DisplacementAssessment | None = None
    context: DealingRangeContext | None = None
    draw_on_liquidity: CoreV1LiquidityLevel | None = None
    zone: ReactionaryZone | None = None
    interaction: CoreV1ZoneInteraction | None = None
    event_timestamp: object | None = None
    confirmation_timestamp: object | None = None
    provenance: Mapping[str, tuple[EvidenceSourceObservation, ...]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.state, PrecisionState):
            raise TypeError("state must use the existing four-state precision vocabulary.")
        availability = _utc_timestamp(self.availability_timestamp, "availability_timestamp")
        if availability != self.subject.availability_timestamp:
            raise ValueError("Result availability must match the ReplaySubject.")
        object.__setattr__(self, "availability_timestamp", availability)

        event = self.event_timestamp
        if event is not None:
            event = _utc_timestamp(event, "event_timestamp")
            if event > availability:
                raise ValueError("Entry qualification event exceeds availability.")
            object.__setattr__(self, "event_timestamp", event)
        confirmation = self.confirmation_timestamp
        if confirmation is not None:
            confirmation = _utc_timestamp(confirmation, "confirmation_timestamp")
            if event is not None and confirmation < event:
                raise ValueError("Entry confirmation cannot precede its event.")
            if confirmation > availability:
                raise ValueError("Entry confirmation exceeds availability.")
            object.__setattr__(self, "confirmation_timestamp", confirmation)

        for label, component in (
            ("MSS", self.mss),
            ("liquidity", self.liquidity),
            ("displacement", self.displacement),
            ("context", self.context),
            ("draw-on-liquidity", self.draw_on_liquidity),
            ("zone", self.zone),
            ("interaction", self.interaction),
        ):
            if component is not None and not same_subject_id(component.subject, self.subject):
                raise ValueError(f"{label} must preserve the exact ReplaySubject.")
        if self.provenance is not None and not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping.")
        provenance = {
            key: tuple(sources)
            for key, sources in (self.provenance or {}).items()
        }
        if any(
            not isinstance(source, EvidenceSourceObservation)
            for sources in provenance.values()
            for source in sources
        ):
            raise TypeError("provenance values must contain source observations.")
        if self.interaction is not None and (
            self.zone is None or self.interaction.zone is not self.zone
        ):
            raise ValueError("Result interaction must preserve the exact zone identity.")
        if self.state is PrecisionState.VALID:
            if self.planned_entry is None:
                raise ValueError("VALID entry qualification requires a planned entry.")
            if not same_subject_id(self.planned_entry.subject, self.subject):
                raise ValueError("Planned entry must preserve the exact ReplaySubject.")
            if (
                self.mss is None
                or self.mss.status is not MSSStatus.MSS
                or self.liquidity is None
                or self.displacement is None
                or self.displacement.status is not DisplacementStatus.DISPLACEMENT
            ):
                raise ValueError(
                    "VALID entry qualification requires resolved MSS, liquidity, and displacement."
                )
            if (
                self.mss.liquidity_event is not self.liquidity
                or self.mss.displacement is not self.displacement
            ):
                raise ValueError(
                    "VALID entry qualification must preserve MSS prerequisite identities."
                )
            required_sources = _unique_sources(
                (
                    *_mss_sources(self.mss),
                    *_liquidity_sources(self.liquidity),
                    *_displacement_sources(self.displacement),
                    *(
                        _zone_timing(self.zone).source_observations
                        if self.zone is not None
                        else ()
                    ),
                    *(
                        self.interaction.timing.source_observations
                        if self.interaction is not None
                        else ()
                    ),
                    *(
                        self.context.provenance
                        if self.context is not None
                        else ()
                    ),
                    *(
                        self.draw_on_liquidity.timing.source_observations
                        if self.draw_on_liquidity is not None
                        else ()
                    ),
                )
            )
            if any(
                source not in self.planned_entry.timing.source_observations
                for source in required_sources
            ):
                raise ValueError("VALID planned entry must trace every prerequisite source.")
            if provenance.get("mss") != _mss_sources(self.mss):
                raise ValueError("VALID result must preserve MSS provenance.")
            if provenance.get("liquidity") != _liquidity_sources(self.liquidity):
                raise ValueError("VALID result must preserve liquidity provenance.")
            if provenance.get("displacement") != _displacement_sources(self.displacement):
                raise ValueError("VALID result must preserve displacement provenance.")
            for name, expected in (
                ("entry_resolver", self.planned_entry.timing.source_observations),
                ("planned_entry", self.planned_entry.timing.source_observations),
            ):
                if provenance.get(name) != expected:
                    raise ValueError(f"VALID result must preserve {name} provenance.")
            if self.zone is not None and (
                provenance.get("reactionary_zone")
                != _zone_timing(self.zone).source_observations
            ):
                raise ValueError("VALID result must preserve reactionary-zone provenance.")
            if self.interaction is not None and (
                provenance.get("zone_interaction")
                != self.interaction.timing.source_observations
            ):
                raise ValueError("VALID result must preserve zone-interaction provenance.")
            if self.context is not None and (
                provenance.get("context") != self.context.provenance
            ):
                raise ValueError("VALID result must preserve context provenance.")
            if self.draw_on_liquidity is not None and (
                provenance.get("draw_on_liquidity")
                != self.draw_on_liquidity.timing.source_observations
            ):
                raise ValueError("VALID result must preserve draw-on-liquidity provenance.")
            if self.event_timestamp != self.planned_entry.timing.event_timestamp:
                raise ValueError("Result event must preserve planned-entry event time.")
            if (
                self.confirmation_timestamp
                != self.planned_entry.timing.confirmation_timestamp
            ):
                raise ValueError("Result confirmation must preserve planned-entry confirmation.")
        elif self.planned_entry is not None:
            raise ValueError("Non-VALID entry qualification cannot expose a planned entry.")

        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @property
    def planned_entry_price(self) -> float | None:
        return (
            self.planned_entry.planned_entry_price
            if self.planned_entry is not None
            else None
        )

    @property
    def interaction_timestamp(self):
        return self.interaction.interaction_timestamp if self.interaction else None


class RetracementEntryResolver(Protocol):
    """Optional governed entry policy; no retracement or entry rule is implied."""

    def qualify(
        self,
        inputs: CoreV1RetracementInputs,
    ) -> CoreV1PlannedEntry | None: ...


def evaluate_core_v1_retracement(
    inputs: SubjectVisibleInputs,
    *,
    mss: CoreV1MSSResult | None,
    liquidity: CoreV1LiquidityAssessment | None,
    displacement: CoreV1DisplacementAssessment | None,
    context: DealingRangeContext | None = None,
    draw_on_liquidity: CoreV1LiquidityLevel | None = None,
    zone: ReactionaryZone | None = None,
    interaction: CoreV1ZoneInteraction | None = None,
    resolver: RetracementEntryResolver | None = None,
) -> CoreV1RetracementResult:
    """Validate the established sequence and accept only governed planned entries."""
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    request = CoreV1RetracementInputs(
        visible_inputs=inputs,
        mss=mss,
        liquidity=liquidity,
        displacement=displacement,
        context=context,
        draw_on_liquidity=draw_on_liquidity,
        zone=zone,
        interaction=interaction,
    )
    subject = inputs.subject
    _validate_supplied_evidence(request)
    if liquidity is None:
        return _unresolved(request, "Required liquidity evidence is missing.")
    if displacement is None or displacement.status is not DisplacementStatus.DISPLACEMENT:
        return _unresolved(request, "Required displacement evidence is unresolved.")
    if mss is None or mss.status is not MSSStatus.MSS:
        return _unresolved(request, "Required MSS evidence is missing or unresolved.")

    _validate_mss_sequence(request)
    if resolver is None:
        return _unresolved(
            request,
            "NO_ENTRY_UNRESOLVED: no governed retracement/entry resolver is available.",
        )
    candidate = resolver.qualify(request)
    if candidate is None:
        return _unresolved(
            request,
            "NO_ENTRY_UNRESOLVED: the governed entry resolver returned no planned entry.",
        )
    if not isinstance(candidate, CoreV1PlannedEntry):
        raise TypeError("Entry resolver must return CoreV1PlannedEntry or None.")
    if not same_subject_id(candidate.subject, subject):
        raise ValueError("Planned entry belongs to another ReplaySubject.")
    candidate.timing.validate_for_subject(subject)
    _validate_visible_sources(candidate.timing.source_observations, inputs, "Planned entry")

    required_sources = _prerequisite_sources(request)
    if any(source not in candidate.timing.source_observations for source in required_sources):
        raise ValueError(
            "Planned entry provenance must retain every supplied prerequisite source."
        )
    latest_required_time = _latest_prerequisite_availability(request)
    if candidate.timing.event_timestamp < latest_required_time:
        raise ValueError("Planned entry event precedes prerequisite availability.")
    if interaction is not None and (
        candidate.timing.event_timestamp < interaction.interaction_timestamp
    ):
        raise ValueError("Planned entry event precedes the supplied zone interaction.")

    provenance = _provenance(request, candidate)
    return CoreV1RetracementResult(
        subject=subject,
        state=PrecisionState.VALID,
        availability_timestamp=subject.availability_timestamp,
        reason="Governed resolver supplied a timed planned-entry hypothesis.",
        planned_entry=candidate,
        mss=mss,
        liquidity=liquidity,
        displacement=displacement,
        context=context,
        draw_on_liquidity=draw_on_liquidity,
        zone=zone,
        interaction=interaction,
        event_timestamp=candidate.timing.event_timestamp,
        confirmation_timestamp=candidate.timing.confirmation_timestamp,
        provenance=provenance,
    )


def _validate_mss_sequence(request: CoreV1RetracementInputs) -> None:
    assert request.mss is not None
    assert request.liquidity is not None
    assert request.displacement is not None
    if not same_subject_id(request.mss.subject, request.visible_inputs.subject):
        raise ValueError("MSS belongs to another ReplaySubject.")
    if request.mss.liquidity_event is not request.liquidity:
        raise ValueError("MSS must reference the supplied governed liquidity assessment.")
    if request.mss.displacement is not request.displacement:
        raise ValueError("MSS must reference the supplied governed displacement assessment.")
    if request.displacement.status is not DisplacementStatus.DISPLACEMENT:
        raise ValueError("MSS prerequisite displacement is unresolved.")
    if (
        request.liquidity.breach is None
        or request.liquidity.reaction is None
        or request.displacement.observation is None
        or request.mss.confirmation is None
    ):
        raise ValueError("Resolved MSS must retain resolved liquidity, displacement, and confirmation.")
    confirmation_timing = request.mss.confirmation.timing
    confirmation_timing.validate_for_subject(request.visible_inputs.subject)
    for component in (
        request.mss.structural_point,
        request.mss.liquidity_event,
        request.mss.displacement,
        request.mss.structural_break,
        request.mss.confirmation,
    ):
        if component is None or not same_subject_id(
            component.subject,
            request.visible_inputs.subject,
        ):
            raise ValueError("Resolved MSS must preserve all governed components and subject.")
    assert request.mss.structural_point is not None
    assert request.mss.structural_break is not None
    timestamps = (
        request.mss.structural_point.timing.event_timestamp,
        request.liquidity.breach.timing.event_timestamp,
        request.displacement.observation.timing.event_timestamp,
        request.mss.structural_break.timing.event_timestamp,
        confirmation_timing.confirmation_timestamp,
    )
    if any(
        earlier is None or later is None or earlier > later
        for earlier, later in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("MSS prerequisite events are not causally ordered.")
    expected_mss_provenance = {
        "structural_point": request.mss.structural_point.timing.source_observations,
        "liquidity_event": _liquidity_sources(request.liquidity),
        "displacement": _displacement_sources(request.displacement),
        "structural_break": request.mss.structural_break.timing.source_observations,
        "confirmation": confirmation_timing.source_observations,
    }
    if any(
        request.mss.provenance.get(name) != sources
        for name, sources in expected_mss_provenance.items()
    ):
        raise ValueError("MSS provenance does not retain every governed component.")


def _validate_prerequisite_lineage(request: CoreV1RetracementInputs) -> None:
    inputs = request.visible_inputs
    if request.mss is not None:
        _validate_visible_sources(_mss_sources(request.mss), inputs, "MSS")
    if request.liquidity is not None:
        _validate_visible_sources(
            _liquidity_sources(request.liquidity),
            inputs,
            "Liquidity",
        )
    if request.displacement is not None:
        _validate_visible_sources(
            _displacement_sources(request.displacement),
            inputs,
            "Displacement",
        )


def _validate_supplied_evidence(request: CoreV1RetracementInputs) -> None:
    _validate_prerequisite_lineage(request)
    if (
        request.mss is not None
        and request.mss.status is MSSStatus.MSS
        and request.liquidity is not None
        and request.displacement is not None
        and request.displacement.status is DisplacementStatus.DISPLACEMENT
    ):
        _validate_mss_sequence(request)
    if request.zone is not None:
        _validate_zone(request.zone, request.visible_inputs)
    if request.interaction is not None:
        _validate_interaction(request.interaction, request.visible_inputs)
    if request.context is not None:
        _validate_context(request.context, request.visible_inputs)
    if request.draw_on_liquidity is not None:
        _validate_draw(request.draw_on_liquidity, request.visible_inputs)


def _validate_zone(zone: ReactionaryZone, inputs: SubjectVisibleInputs) -> None:
    if not same_subject_id(zone.subject, inputs.subject):
        raise ValueError("Reactionary zone belongs to another ReplaySubject.")
    _validate_visible_sources(_zone_timing(zone).source_observations, inputs, "Reactionary zone")


def _validate_interaction(
    interaction: CoreV1ZoneInteraction,
    inputs: SubjectVisibleInputs,
) -> None:
    if not same_subject_id(interaction.subject, inputs.subject):
        raise ValueError("Zone interaction belongs to another ReplaySubject.")
    _validate_visible_sources(
        interaction.timing.source_observations,
        inputs,
        "Zone interaction",
    )
    if interaction.timing.event_timestamp < _zone_timing(interaction.zone).availability_timestamp:
        raise ValueError("Zone interaction precedes zone availability.")
    _validate_base_bar_scope(interaction.timing, inputs.subject, "Zone interaction")


def _validate_context(
    context: DealingRangeContext,
    inputs: SubjectVisibleInputs,
) -> None:
    if not same_subject_id(context.subject, inputs.subject):
        raise ValueError("Context belongs to another ReplaySubject.")
    if context.availability_timestamp > inputs.subject.availability_timestamp:
        raise ValueError("Context is unavailable at the ReplaySubject.")
    if (
        context.event_timestamp is not None
        and context.event_timestamp > inputs.subject.base_bar_timestamp
    ):
        raise ValueError("Context event exceeds the ReplaySubject base-bar cutoff.")
    if (
        context.confirmation_timestamp is not None
        and context.confirmation_timestamp > inputs.subject.base_bar_timestamp
    ):
        raise ValueError("Context confirmation exceeds the ReplaySubject base-bar cutoff.")
    _validate_visible_sources(context.provenance, inputs, "Context")


def _validate_draw(draw: CoreV1LiquidityLevel, inputs: SubjectVisibleInputs) -> None:
    timing = draw.timing
    if not same_subject_id(draw.subject, inputs.subject):
        raise ValueError("Draw-on-liquidity belongs to another ReplaySubject.")
    _validate_visible_sources(timing.source_observations, inputs, "Draw-on-liquidity")
    _validate_base_bar_scope(timing, inputs.subject, "Draw-on-liquidity")


def _validate_visible_sources(
    sources: tuple[EvidenceSourceObservation, ...],
    inputs: SubjectVisibleInputs,
    label: str,
) -> None:
    visible = tuple(
        source
        for observation in inputs.base_observations + inputs.context_observations
        for source in observation.timing.source_observations
    )
    if any(source not in visible for source in sources):
        raise ValueError(f"{label} provenance references observations outside visible inputs.")
    if any(
        source.availability_timestamp > inputs.subject.availability_timestamp
        for source in sources
    ):
        raise ValueError(f"{label} contains evidence unavailable at the ReplaySubject.")


def _validate_base_bar_scope(
    timing: EvidenceTiming,
    subject: ReplaySubject,
    label: str,
) -> None:
    if not timing.base_bar_scoped:
        raise ValueError(f"{label} timing must be base-bar scoped.")
    timing.validate_for_subject(subject)
    if timing.event_timestamp > subject.base_bar_timestamp:
        raise ValueError(f"{label} event exceeds the ReplaySubject base-bar cutoff.")
    if (
        timing.confirmation_timestamp is not None
        and timing.confirmation_timestamp > subject.base_bar_timestamp
    ):
        raise ValueError(f"{label} confirmation exceeds the ReplaySubject base-bar cutoff.")
    if any(
        source.is_base_timeframe
        and (
            source.event_timestamp > subject.base_bar_timestamp
            or source.base_bar_timestamp is None
            or source.base_bar_timestamp > subject.base_bar_timestamp
        )
        for source in timing.source_observations
    ):
        raise ValueError(f"{label} source exceeds the ReplaySubject base-bar cutoff.")


def _prerequisite_sources(
    request: CoreV1RetracementInputs,
) -> tuple[EvidenceSourceObservation, ...]:
    sources: list[EvidenceSourceObservation] = []
    if request.mss is not None:
        sources.extend(_mss_sources(request.mss))
    if request.liquidity is not None:
        sources.extend(_liquidity_sources(request.liquidity))
    if request.displacement is not None:
        sources.extend(_displacement_sources(request.displacement))
    if request.zone is not None:
        sources.extend(_zone_timing(request.zone).source_observations)
    if request.interaction is not None:
        sources.extend(request.interaction.timing.source_observations)
    if request.context is not None:
        sources.extend(request.context.provenance)
    if request.draw_on_liquidity is not None:
        sources.extend(request.draw_on_liquidity.timing.source_observations)
    return _unique_sources(sources)


def _latest_prerequisite_availability(
    request: CoreV1RetracementInputs,
):
    sources = _prerequisite_sources(request)
    timestamps = [source.availability_timestamp for source in sources]
    if request.mss is not None:
        timestamps.append(request.mss.availability_timestamp)
        if request.mss.confirmation_timestamp is not None:
            timestamps.append(request.mss.confirmation_timestamp)
    if request.liquidity is not None:
        for timing in _liquidity_timings(request.liquidity):
            timestamps.append(timing.availability_timestamp)
    if request.displacement is not None:
        timestamps.append(request.displacement.availability_timestamp)
    if request.zone is not None:
        timestamps.append(_zone_timing(request.zone).availability_timestamp)
    if request.interaction is not None:
        timestamps.append(request.interaction.timing.availability_timestamp)
    if request.context is not None:
        timestamps.append(request.context.availability_timestamp)
    if request.draw_on_liquidity is not None:
        timestamps.append(request.draw_on_liquidity.timing.availability_timestamp)
    return max(timestamps)


def _mss_sources(mss: CoreV1MSSResult) -> tuple[EvidenceSourceObservation, ...]:
    components = (
        mss.structural_point,
        mss.structural_break,
        mss.confirmation,
    )
    sources = [
        source
        for component in components
        if component is not None
        for source in _component_timing_sources(component)
    ]
    if mss.liquidity_event is not None:
        sources.extend(_liquidity_sources(mss.liquidity_event))
    if mss.displacement is not None:
        sources.extend(_displacement_sources(mss.displacement))
    return _unique_sources(sources)


def _liquidity_sources(
    assessment: CoreV1LiquidityAssessment,
) -> tuple[EvidenceSourceObservation, ...]:
    items = _liquidity_timings(assessment)
    return _unique_sources(
        source for timing in items for source in timing.source_observations
    )


def _liquidity_timings(
    assessment: CoreV1LiquidityAssessment,
) -> tuple[EvidenceTiming, ...]:
    items = [assessment.level.timing]
    if assessment.breach is not None:
        items.append(assessment.breach.timing)
    if assessment.reaction is not None:
        items.append(assessment.reaction.timing)
    if assessment.sweep_timing is not None:
        items.append(assessment.sweep_timing)
    return tuple(items)


def _displacement_sources(
    assessment: CoreV1DisplacementAssessment,
) -> tuple[EvidenceSourceObservation, ...]:
    if assessment.observation is None:
        return ()
    return assessment.observation.timing.source_observations


def _component_timing_sources(component: object) -> tuple[EvidenceSourceObservation, ...]:
    timing = getattr(component, "timing", None)
    if not isinstance(timing, EvidenceTiming):
        return ()
    return timing.source_observations


def _unique_sources(
    sources,
) -> tuple[EvidenceSourceObservation, ...]:
    unique: list[EvidenceSourceObservation] = []
    seen: set[int] = set()
    for source in sources:
        identity = id(source)
        if identity not in seen:
            seen.add(identity)
            unique.append(source)
    return tuple(unique)


def _zone_timing(zone: ReactionaryZone) -> EvidenceTiming:
    if isinstance(zone, CoreV1PDArrayObservation):
        return zone.timing
    return zone.timing


def _provenance(
    request: CoreV1RetracementInputs,
    planned_entry: CoreV1PlannedEntry,
) -> Mapping[str, tuple[EvidenceSourceObservation, ...]]:
    assert request.mss is not None
    assert request.liquidity is not None
    assert request.displacement is not None
    result: dict[str, tuple[EvidenceSourceObservation, ...]] = {
        "mss": _mss_sources(request.mss),
        "liquidity": _liquidity_sources(request.liquidity),
        "displacement": _displacement_sources(request.displacement),
        "entry_resolver": planned_entry.timing.source_observations,
        "planned_entry": planned_entry.timing.source_observations,
    }
    if request.zone is not None:
        result["reactionary_zone"] = _zone_timing(request.zone).source_observations
    if request.interaction is not None:
        result["zone_interaction"] = request.interaction.timing.source_observations
    if request.context is not None:
        result["context"] = request.context.provenance
    if request.draw_on_liquidity is not None:
        result["draw_on_liquidity"] = request.draw_on_liquidity.timing.source_observations
    return result


def _unresolved(
    request: CoreV1RetracementInputs,
    reason: str,
) -> CoreV1RetracementResult:
    subject = request.visible_inputs.subject
    return CoreV1RetracementResult(
        subject=subject,
        state=PrecisionState.DEVELOPING,
        availability_timestamp=subject.availability_timestamp,
        reason=reason,
        mss=request.mss,
        liquidity=request.liquidity,
        displacement=request.displacement,
        context=request.context,
        draw_on_liquidity=request.draw_on_liquidity,
        zone=request.zone,
        interaction=request.interaction,
        provenance=_unresolved_provenance(request),
    )


def _unresolved_provenance(
    request: CoreV1RetracementInputs,
) -> Mapping[str, tuple[EvidenceSourceObservation, ...]]:
    result: dict[str, tuple[EvidenceSourceObservation, ...]] = {}
    if request.mss is not None:
        result["mss"] = _mss_sources(request.mss)
    if request.liquidity is not None:
        result["liquidity"] = _liquidity_sources(request.liquidity)
    if request.displacement is not None:
        result["displacement"] = _displacement_sources(request.displacement)
    if request.zone is not None:
        result["reactionary_zone"] = _zone_timing(request.zone).source_observations
    if request.interaction is not None:
        result["zone_interaction"] = request.interaction.timing.source_observations
    return result


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
    "CoreV1PlannedEntry",
    "CoreV1RetracementInputs",
    "CoreV1RetracementResult",
    "CoreV1ZoneInteraction",
    "ReactionaryZone",
    "RetracementEntryResolver",
    "evaluate_core_v1_retracement",
]
