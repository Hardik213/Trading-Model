from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from .core_v1_dealing_range import (
    DealingRangeContext,
    DealingRangeResolver,
    build_dealing_range_context,
)
from .core_v1_displacement import (
    CoreV1DisplacementAssessment,
    DisplacementStatus,
)
from .core_v1_evidence import SubjectVisibleInputs
from .core_v1_hypothesis import (
    CoreV1OpposingLiquidityTarget,
    CoreV1StructuralInvalidation,
    CoreV1TradeHypothesis,
    OpposingLiquidityTargetResolver,
    StructuralInvalidationResolver,
    evaluate_core_v1_hypothesis,
)
from .core_v1_liquidity import CoreV1LiquidityAssessment, CoreV1LiquidityLevel
from .core_v1_mss import CoreV1MSSResult, MSSStatus, evaluate_core_v1_mss
from .core_v1_reactionary_zones import (
    CoreV1FVGObservation,
    CoreV1OrderBlockObservation,
    CoreV1PDArrayObservation,
)
from .core_v1_retracement import (
    CoreV1RetracementResult,
    CoreV1ZoneInteraction,
    ReactionaryZone,
    RetracementEntryResolver,
    evaluate_core_v1_retracement,
)
from .evidence_timing import (
    EvidenceSourceObservation,
    SubjectEvidence,
    validate_bounded_mapping_keys,
)
from .replay_subject import ReplaySubject, same_subject_id
from .sniper_setup import PrecisionState


@dataclass(frozen=True)
class CoreV1PipelineResult:
    """Ordered Core v1 evidence composition; a valid hypothesis is not execution."""

    subject: ReplaySubject
    availability_timestamp: object
    state: PrecisionState
    reason: str
    context: DealingRangeContext
    draw_on_liquidity: CoreV1LiquidityLevel | None = None
    opposing_liquidity_levels: tuple[CoreV1LiquidityLevel, ...] = ()
    liquidity: CoreV1LiquidityAssessment | None = None
    displacement: CoreV1DisplacementAssessment | None = None
    mss: CoreV1MSSResult | None = None
    zone: ReactionaryZone | None = None
    interaction: CoreV1ZoneInteraction | None = None
    retracement: CoreV1RetracementResult | None = None
    invalidation: CoreV1StructuralInvalidation | None = None
    target: CoreV1OpposingLiquidityTarget | None = None
    hypothesis: CoreV1TradeHypothesis | None = None
    event_timestamps: Mapping[str, object] | None = None
    confirmation_timestamps: Mapping[str, object] | None = None
    provenance: Mapping[str, tuple[EvidenceSourceObservation, ...]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.state, PrecisionState):
            raise TypeError("state must use the existing Core v1 state vocabulary.")
        availability = _utc_timestamp(self.availability_timestamp)
        if availability != self.subject.availability_timestamp:
            raise ValueError("Pipeline availability must exactly match its ReplaySubject.")
        object.__setattr__(self, "availability_timestamp", availability)
        if not isinstance(self.context, DealingRangeContext):
            raise TypeError("context must be a DealingRangeContext.")
        if not same_subject_id(self.context.subject, self.subject):
            raise ValueError("Pipeline context must preserve the ReplaySubject.")
        levels = tuple(self.opposing_liquidity_levels)
        if any(not isinstance(level, CoreV1LiquidityLevel) for level in levels):
            raise TypeError("Pipeline target candidates must be governed liquidity levels.")
        object.__setattr__(self, "opposing_liquidity_levels", levels)

        for label, component in (
            ("draw-on-liquidity", self.draw_on_liquidity),
            ("liquidity", self.liquidity),
            ("displacement", self.displacement),
            ("MSS", self.mss),
            ("zone", self.zone),
            ("interaction", self.interaction),
            ("retracement", self.retracement),
            ("invalidation", self.invalidation),
            ("target", self.target),
            ("hypothesis", self.hypothesis),
        ):
            if component is not None and not same_subject_id(component.subject, self.subject):
                raise ValueError(f"Pipeline {label} must preserve the ReplaySubject.")
        if any(not same_subject_id(level.subject, self.subject) for level in levels):
            raise ValueError("Pipeline target candidates must preserve the ReplaySubject.")

        if self.state is PrecisionState.VALID:
            if (
                self.hypothesis is None
                or self.hypothesis.state is not PrecisionState.VALID
            ):
                raise ValueError("VALID pipeline results require a VALID hypothesis.")
            if (
                self.liquidity is None
                or not _liquidity_is_resolved(self.liquidity)
                or self.displacement is None
                or self.displacement.status is not DisplacementStatus.DISPLACEMENT
                or self.mss is None
                or self.mss.status is not MSSStatus.MSS
                or self.zone is None
                or self.retracement is None
                or self.retracement.state is not PrecisionState.VALID
                or self.hypothesis.planned_entry is not self.retracement
                or self.invalidation is None
                or self.target is None
                or self.hypothesis.invalidation is not self.invalidation
                or self.hypothesis.target is not self.target
            ):
                raise ValueError(
                    "VALID pipeline results require every governed stage and their VALID planned entry."
                )
        elif self.hypothesis is not None:
            raise ValueError("Non-VALID pipeline results cannot contain a hypothesis.")

        expected_events, expected_confirmations, expected_provenance = (
            _derive_pipeline_metadata(self)
        )
        for label, values in (
            ("event", expected_events),
            ("confirmation", expected_confirmations),
        ):
            for name, value in values.items():
                timestamp = _utc_timestamp(value)
                if timestamp > availability:
                    raise ValueError(
                        f"Referenced stage {label} timestamp for {name} exceeds availability."
                    )
                values[name] = timestamp
        for sources in expected_provenance.values():
            for source in sources:
                if source.availability_timestamp > availability:
                    raise ValueError(
                        "Referenced stage provenance exceeds subject availability."
                    )
                if source.is_base_timeframe and (
                    source.event_timestamp > self.subject.base_bar_timestamp
                    or source.base_bar_timestamp > self.subject.base_bar_timestamp
                ):
                    raise ValueError(
                        "Referenced stage provenance exceeds the base-bar cutoff."
                    )
        event_timestamps = (
            None
            if self.event_timestamps is None
            else dict(self.event_timestamps)
        )
        confirmation_timestamps = (
            None
            if self.confirmation_timestamps is None
            else dict(self.confirmation_timestamps)
        )
        provenance = (
            None
            if self.provenance is None
            else {
                name: tuple(sources)
                for name, sources in self.provenance.items()
            }
        )
        validate_bounded_mapping_keys(
            provenance if provenance is not None else expected_provenance,
            "Pipeline provenance",
        )
        if any(
            not isinstance(source, EvidenceSourceObservation)
            for sources in (provenance or {}).values()
            for source in sources
        ):
            raise TypeError("Pipeline provenance must contain source observations.")
        for label, values in (
            ("event", event_timestamps),
            ("confirmation", confirmation_timestamps),
        ):
            if values is None:
                continue
            for name, value in values.items():
                timestamp = _utc_timestamp(value)
                if timestamp > availability:
                    raise ValueError(f"Pipeline {label} timestamp for {name} exceeds availability.")
                values[name] = timestamp
        for sources in (provenance or {}).values():
            for source in sources:
                if source.availability_timestamp > availability:
                    raise ValueError("Pipeline provenance exceeds subject availability.")
                if source.is_base_timeframe and (
                    source.event_timestamp > self.subject.base_bar_timestamp
                    or source.base_bar_timestamp > self.subject.base_bar_timestamp
                ):
                    raise ValueError("Pipeline provenance exceeds the base-bar cutoff.")

        if event_timestamps is not None and event_timestamps != expected_events:
            raise ValueError(
                "Pipeline event timestamps must match the referenced stage evidence."
            )
        if (
            confirmation_timestamps is not None
            and confirmation_timestamps != expected_confirmations
        ):
            raise ValueError(
                "Pipeline confirmation timestamps must match the referenced stage evidence."
            )
        if provenance is not None and provenance != expected_provenance:
            raise ValueError(
                "Pipeline provenance must match the referenced stage evidence."
            )
        object.__setattr__(
            self,
            "event_timestamps",
            MappingProxyType(expected_events),
        )
        object.__setattr__(
            self,
            "confirmation_timestamps",
            MappingProxyType(expected_confirmations),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(expected_provenance),
        )


def _derive_pipeline_metadata(
    result: CoreV1PipelineResult,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, tuple[EvidenceSourceObservation, ...]],
]:
    events: dict[str, object] = {}
    confirmations: dict[str, object] = {}
    provenance: dict[str, tuple[EvidenceSourceObservation, ...]] = {}

    def add(name: str, timing: object, sources=None) -> None:
        event = getattr(timing, "event_timestamp")
        confirmation = getattr(timing, "confirmation_timestamp")
        actual_sources = (
            getattr(timing, "source_observations")
            if sources is None
            else sources
        )
        if event is not None:
            events[name] = event
        if confirmation is not None:
            confirmations[name] = confirmation
        if actual_sources:
            provenance[name] = tuple(actual_sources)

    add(
        "context",
        result.context,
        result.context.provenance,
    )
    if result.draw_on_liquidity is not None:
        add("draw_on_liquidity", result.draw_on_liquidity.timing)
    for level in result.opposing_liquidity_levels:
        add(
            f"opposing_liquidity_candidate:{level.level_id}",
            level.timing,
        )
    if result.liquidity is not None:
        add("liquidity_level", result.liquidity.level.timing)
        if result.liquidity.breach is not None:
            add("liquidity_event", result.liquidity.breach.timing)
        if result.liquidity.reaction is not None:
            add("liquidity_reaction", result.liquidity.reaction.timing)
        if result.liquidity.sweep_timing is not None:
            add("liquidity_sweep_classification", result.liquidity.sweep_timing)
    if result.displacement is not None and result.displacement.observation is not None:
        add("displacement", result.displacement.observation.timing)
    if result.mss is not None:
        if result.mss.structural_point is not None:
            add("structural_point", result.mss.structural_point.timing)
        if result.mss.structural_break is not None:
            add("structural_break", result.mss.structural_break.timing)
        if result.mss.confirmation is not None:
            add("mss_confirmation", result.mss.confirmation.timing)
    if result.zone is not None:
        add("reactionary_zone", _zone_timing(result.zone))
    if result.interaction is not None:
        add("zone_interaction", result.interaction.timing)
    if result.retracement is not None:
        if result.retracement.planned_entry is not None:
            add("planned_entry", result.retracement.planned_entry.timing)
        provenance.update(
            {
                f"retracement_{name}": tuple(sources)
                for name, sources in result.retracement.provenance.items()
            }
        )
    if result.invalidation is not None:
        add("invalidation", result.invalidation.timing)
    if result.target is not None:
        add("opposing_liquidity_target", result.target.timing)
        add("target_liquidity_level", result.target.liquidity_level.timing)
    if result.hypothesis is not None:
        provenance.update(result.hypothesis.provenance)
    return events, confirmations, provenance


def evaluate_core_v1_pipeline(
    inputs: SubjectVisibleInputs,
    *,
    context: DealingRangeContext | None = None,
    context_resolver: DealingRangeResolver | None = None,
    reference_price: SubjectEvidence[float] | None = None,
    liquidity: CoreV1LiquidityAssessment | None = None,
    displacement: CoreV1DisplacementAssessment | None = None,
    mss: CoreV1MSSResult | None = None,
    zone: ReactionaryZone | None = None,
    interaction: CoreV1ZoneInteraction | None = None,
    draw_on_liquidity: CoreV1LiquidityLevel | None = None,
    entry_resolver: RetracementEntryResolver | None = None,
    invalidation: CoreV1StructuralInvalidation | None = None,
    invalidation_resolver: StructuralInvalidationResolver | None = None,
    opposing_liquidity_levels: tuple[CoreV1LiquidityLevel, ...] = (),
    target: CoreV1OpposingLiquidityTarget | None = None,
    target_resolver: OpposingLiquidityTargetResolver | None = None,
) -> CoreV1PipelineResult:
    """Compose existing governed stage outputs without supplying detector policy."""
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    subject = inputs.subject
    if context is not None and context_resolver is not None:
        raise ValueError("Supply a dealing-range context or resolver, not both.")
    if context is None:
        context = build_dealing_range_context(
            inputs,
            context_resolver,
            reference_price=reference_price,
        )
    elif not same_subject_id(context.subject, subject):
        raise ValueError("Dealing-range context belongs to another ReplaySubject.")
    _validate_visible_component(inputs, context.provenance, "Dealing-range context")
    if draw_on_liquidity is not None:
        if not isinstance(draw_on_liquidity, CoreV1LiquidityLevel):
            raise TypeError("draw_on_liquidity must be a governed liquidity level.")
        _validate_subject(draw_on_liquidity, subject, "Draw-on-liquidity")
        draw_on_liquidity.timing.validate_for_subject(subject)
        _validate_visible_component(
            inputs,
            draw_on_liquidity.timing.source_observations,
            "Draw-on-liquidity",
        )
    opposing_liquidity_levels = tuple(opposing_liquidity_levels)
    for level in opposing_liquidity_levels:
        if not isinstance(level, CoreV1LiquidityLevel):
            raise TypeError("opposing_liquidity_levels must contain governed liquidity levels.")
        _validate_subject(level, subject, "Opposing-liquidity candidate")
        level.timing.validate_for_subject(subject)
        _validate_visible_component(
            inputs,
            level.timing.source_observations,
            "Opposing-liquidity candidate",
        )

    if liquidity is None:
        return _incomplete(
            inputs,
            context,
            "DEVELOPING: required governed liquidity evidence is missing.",
            draw_on_liquidity=draw_on_liquidity,
            opposing_liquidity_levels=opposing_liquidity_levels,
        )
    _validate_subject(liquidity, subject, "Liquidity")
    if not _liquidity_is_resolved(liquidity):
        return _incomplete(
            inputs,
            context,
            "DEVELOPING: governed liquidity event is unresolved.",
            draw_on_liquidity=draw_on_liquidity,
            opposing_liquidity_levels=opposing_liquidity_levels,
            liquidity=liquidity,
        )
    _validate_visible_component(inputs, _liquidity_sources(liquidity), "Liquidity")

    if displacement is None or displacement.status is not DisplacementStatus.DISPLACEMENT:
        return _incomplete(
            inputs,
            context,
            "DEVELOPING: governed displacement evidence is missing or unresolved.",
            draw_on_liquidity=draw_on_liquidity,
            opposing_liquidity_levels=opposing_liquidity_levels,
            liquidity=liquidity,
            displacement=displacement,
        )
    _validate_subject(displacement, subject, "Displacement")
    if displacement.observation is None:
        raise ValueError("Resolved displacement result has no observation.")
    _validate_visible_component(
        inputs,
        displacement.observation.timing.source_observations,
        "Displacement",
    )

    if mss is None or mss.status is not MSSStatus.MSS:
        return _incomplete(
            inputs,
            context,
            "DEVELOPING: governed MSS evidence is missing or unresolved.",
            draw_on_liquidity=draw_on_liquidity,
            opposing_liquidity_levels=opposing_liquidity_levels,
            liquidity=liquidity,
            displacement=displacement,
            mss=mss,
        )
    _validate_subject(mss, subject, "MSS")
    if mss.liquidity_event is not liquidity or mss.displacement is not displacement:
        raise ValueError("MSS must reference the supplied liquidity and displacement stages.")
    mss = evaluate_core_v1_mss(
        inputs,
        structural_point=mss.structural_point,
        liquidity_event=liquidity,
        displacement=displacement,
        structural_break=mss.structural_break,
        confirmation=mss.confirmation,
    )
    if mss.status is not MSSStatus.MSS:
        return _incomplete(
            inputs,
            context,
            f"DEVELOPING: {mss.reason}",
            draw_on_liquidity=draw_on_liquidity,
            opposing_liquidity_levels=opposing_liquidity_levels,
            liquidity=liquidity,
            displacement=displacement,
            mss=mss,
        )

    if zone is None:
        entry = evaluate_core_v1_retracement(
            inputs,
            mss=mss,
            liquidity=liquidity,
            displacement=displacement,
            context=context,
            draw_on_liquidity=draw_on_liquidity,
            interaction=None,
            resolver=None,
        )
        return _incomplete(
            inputs,
            context,
            "DEVELOPING: no governed reactionary-zone evidence is available; entry is blocked.",
            draw_on_liquidity=draw_on_liquidity,
            opposing_liquidity_levels=opposing_liquidity_levels,
            liquidity=liquidity,
            displacement=displacement,
            mss=mss,
            retracement=entry,
        )

    if not isinstance(
        zone,
        (
            CoreV1FVGObservation,
            CoreV1PDArrayObservation,
            CoreV1OrderBlockObservation,
        ),
    ):
        raise TypeError("zone must be governed FVG, PD-array, or order-block evidence.")
    _validate_subject(zone, subject, "Reactionary zone")
    zone_timing = _zone_timing(zone)
    _validate_visible_component(inputs, zone_timing.source_observations, "Reactionary zone")
    if interaction is not None:
        _validate_subject(interaction, subject, "Zone interaction")
        if interaction.zone is not zone:
            raise ValueError("Zone interaction must reference the supplied zone identity.")
        _validate_visible_component(
            inputs,
            interaction.timing.source_observations,
            "Zone interaction",
        )

    entry = evaluate_core_v1_retracement(
        inputs,
        mss=mss,
        liquidity=liquidity,
        displacement=displacement,
        context=context,
        draw_on_liquidity=draw_on_liquidity,
        zone=zone,
        interaction=interaction,
        resolver=entry_resolver,
    )
    if entry.state is not PrecisionState.VALID:
        return _incomplete(
            inputs,
            context,
            entry.reason,
            draw_on_liquidity=draw_on_liquidity,
            opposing_liquidity_levels=opposing_liquidity_levels,
            liquidity=liquidity,
            displacement=displacement,
            mss=mss,
            zone=zone,
            interaction=interaction,
            retracement=entry,
        )

    hypothesis_result = evaluate_core_v1_hypothesis(
        entry,
        opposing_liquidity_levels=opposing_liquidity_levels,
        invalidation=invalidation,
        invalidation_resolver=invalidation_resolver,
        target=target,
        target_resolver=target_resolver,
    )
    if hypothesis_result.state is not PrecisionState.VALID:
        if hypothesis_result.invalidation is not None:
            _validate_visible_component(
                inputs,
                hypothesis_result.invalidation.timing.source_observations,
                "Structural invalidation",
            )
        if hypothesis_result.target is not None:
            _validate_visible_component(
                inputs,
                hypothesis_result.target.timing.source_observations,
                "Opposing-liquidity target",
            )
            _validate_visible_component(
                inputs,
                hypothesis_result.target.liquidity_level.timing.source_observations,
                "Target liquidity level",
            )
        return _incomplete(
            inputs,
            context,
            hypothesis_result.reason,
            draw_on_liquidity=draw_on_liquidity,
            opposing_liquidity_levels=opposing_liquidity_levels,
            liquidity=liquidity,
            displacement=displacement,
            mss=mss,
            zone=zone,
            interaction=interaction,
            retracement=entry,
            invalidation=hypothesis_result.invalidation,
            target=hypothesis_result.target,
            state=hypothesis_result.state,
        )

    assert hypothesis_result.invalidation is not None
    assert hypothesis_result.target is not None
    _validate_visible_component(
        inputs,
        hypothesis_result.invalidation.timing.source_observations,
        "Structural invalidation",
    )
    _validate_visible_component(
        inputs,
        hypothesis_result.target.timing.source_observations,
        "Opposing-liquidity target",
    )
    _validate_visible_component(
        inputs,
        hypothesis_result.target.liquidity_level.timing.source_observations,
        "Target liquidity level",
    )

    return _result(
        inputs,
        context,
        hypothesis_result.reason,
        PrecisionState.VALID,
        draw_on_liquidity=draw_on_liquidity,
        opposing_liquidity_levels=opposing_liquidity_levels,
        liquidity=liquidity,
        displacement=displacement,
        mss=mss,
        zone=zone,
        interaction=interaction,
        retracement=entry,
        invalidation=hypothesis_result.invalidation,
        target=hypothesis_result.target,
        hypothesis=hypothesis_result,
    )


def _incomplete(
    inputs: SubjectVisibleInputs,
    context: DealingRangeContext,
    reason: str,
    *,
    draw_on_liquidity: CoreV1LiquidityLevel | None = None,
    opposing_liquidity_levels: tuple[CoreV1LiquidityLevel, ...] = (),
    liquidity: CoreV1LiquidityAssessment | None = None,
    displacement: CoreV1DisplacementAssessment | None = None,
    mss: CoreV1MSSResult | None = None,
    zone: ReactionaryZone | None = None,
    interaction: CoreV1ZoneInteraction | None = None,
    retracement: CoreV1RetracementResult | None = None,
    invalidation: CoreV1StructuralInvalidation | None = None,
    target: CoreV1OpposingLiquidityTarget | None = None,
    state: PrecisionState = PrecisionState.DEVELOPING,
) -> CoreV1PipelineResult:
    return _result(
        inputs,
        context,
        reason,
        state,
        draw_on_liquidity=draw_on_liquidity,
        opposing_liquidity_levels=opposing_liquidity_levels,
        liquidity=liquidity,
        displacement=displacement,
        mss=mss,
        zone=zone,
        interaction=interaction,
        retracement=retracement,
        invalidation=invalidation,
        target=target,
    )


def _result(
    inputs: SubjectVisibleInputs,
    context: DealingRangeContext,
    reason: str,
    state: PrecisionState,
    *,
    draw_on_liquidity: CoreV1LiquidityLevel | None = None,
    opposing_liquidity_levels: tuple[CoreV1LiquidityLevel, ...] = (),
    liquidity: CoreV1LiquidityAssessment | None = None,
    displacement: CoreV1DisplacementAssessment | None = None,
    mss: CoreV1MSSResult | None = None,
    zone: ReactionaryZone | None = None,
    interaction: CoreV1ZoneInteraction | None = None,
    retracement: CoreV1RetracementResult | None = None,
    invalidation: CoreV1StructuralInvalidation | None = None,
    target: CoreV1OpposingLiquidityTarget | None = None,
    hypothesis: CoreV1TradeHypothesis | None = None,
) -> CoreV1PipelineResult:
    event_times: dict[str, object] = {}
    confirmation_times: dict[str, object] = {}
    provenance: dict[str, tuple[EvidenceSourceObservation, ...]] = {}
    _add_timing(
        "context",
        context.event_timestamp,
        context.confirmation_timestamp,
        context.provenance,
        event_times,
        confirmation_times,
        provenance,
    )
    if draw_on_liquidity is not None:
        timing = draw_on_liquidity.timing
        _add_timing(
            "draw_on_liquidity",
            timing.event_timestamp,
            timing.confirmation_timestamp,
            timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
    for level in opposing_liquidity_levels:
        timing = level.timing
        _add_timing(
            f"opposing_liquidity_candidate:{level.level_id}",
            timing.event_timestamp,
            timing.confirmation_timestamp,
            timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
    if liquidity is not None:
        level = liquidity.level
        _add_timing(
            "liquidity_level",
            level.timing.event_timestamp,
            level.timing.confirmation_timestamp,
            level.timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
        if liquidity.breach is not None:
            _add_timing(
                "liquidity_event",
                liquidity.breach.timing.event_timestamp,
                liquidity.breach.timing.confirmation_timestamp,
                liquidity.breach.timing.source_observations,
                event_times,
                confirmation_times,
                provenance,
            )
        if liquidity.reaction is not None:
            _add_timing(
                "liquidity_reaction",
                liquidity.reaction.timing.event_timestamp,
                liquidity.reaction.timing.confirmation_timestamp,
                liquidity.reaction.timing.source_observations,
                event_times,
                confirmation_times,
                provenance,
            )
        if liquidity.sweep_timing is not None:
            timing = liquidity.sweep_timing
            _add_timing(
                "liquidity_sweep_classification",
                timing.event_timestamp,
                timing.confirmation_timestamp,
                timing.source_observations,
                event_times,
                confirmation_times,
                provenance,
            )
    if displacement is not None and displacement.observation is not None:
        timing = displacement.observation.timing
        _add_timing(
            "displacement",
            timing.event_timestamp,
            timing.confirmation_timestamp,
            timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
    if mss is not None:
        if mss.structural_point is not None:
            timing = mss.structural_point.timing
            _add_timing(
                "structural_point",
                timing.event_timestamp,
                timing.confirmation_timestamp,
                timing.source_observations,
                event_times,
                confirmation_times,
                provenance,
            )
        if mss.structural_break is not None:
            timing = mss.structural_break.timing
            _add_timing(
                "structural_break",
                timing.event_timestamp,
                timing.confirmation_timestamp,
                timing.source_observations,
                event_times,
                confirmation_times,
                provenance,
            )
        if mss.confirmation is not None:
            timing = mss.confirmation.timing
            _add_timing(
                "mss_confirmation",
                timing.event_timestamp,
                timing.confirmation_timestamp,
                timing.source_observations,
                event_times,
                confirmation_times,
                provenance,
            )
    if zone is not None:
        timing = _zone_timing(zone)
        _add_timing(
            "reactionary_zone",
            timing.event_timestamp,
            timing.confirmation_timestamp,
            timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
    if interaction is not None:
        timing = interaction.timing
        _add_timing(
            "zone_interaction",
            timing.event_timestamp,
            timing.confirmation_timestamp,
            timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
    if retracement is not None:
        if retracement.planned_entry is not None:
            timing = retracement.planned_entry.timing
            _add_timing(
                "planned_entry",
                timing.event_timestamp,
                timing.confirmation_timestamp,
                timing.source_observations,
                event_times,
                confirmation_times,
                provenance,
            )
        provenance.update(
            {
                f"retracement_{name}": sources
                for name, sources in retracement.provenance.items()
            }
        )
    if invalidation is not None:
        timing = invalidation.timing
        _add_timing(
            "invalidation",
            timing.event_timestamp,
            timing.confirmation_timestamp,
            timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
    if target is not None:
        timing = target.timing
        _add_timing(
            "opposing_liquidity_target",
            timing.event_timestamp,
            timing.confirmation_timestamp,
            timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
        timing = target.liquidity_level.timing
        _add_timing(
            "target_liquidity_level",
            timing.event_timestamp,
            timing.confirmation_timestamp,
            timing.source_observations,
            event_times,
            confirmation_times,
            provenance,
        )
    if hypothesis is not None:
        provenance.update(hypothesis.provenance)
    return CoreV1PipelineResult(
        subject=inputs.subject,
        availability_timestamp=inputs.availability_timestamp,
        state=state,
        reason=reason,
        context=context,
        draw_on_liquidity=draw_on_liquidity,
        opposing_liquidity_levels=opposing_liquidity_levels,
        liquidity=liquidity,
        displacement=displacement,
        mss=mss,
        zone=zone,
        interaction=interaction,
        retracement=retracement,
        invalidation=invalidation,
        target=target,
        hypothesis=hypothesis,
        event_timestamps=event_times,
        confirmation_timestamps=confirmation_times,
        provenance=provenance,
    )


def _add_timing(
    name: str,
    event: object | None,
    confirmation: object | None,
    sources: tuple[EvidenceSourceObservation, ...],
    events: dict[str, object],
    confirmations: dict[str, object],
    provenance: dict[str, tuple[EvidenceSourceObservation, ...]],
) -> None:
    if event is not None:
        events[name] = event
    if confirmation is not None:
        confirmations[name] = confirmation
    if sources:
        provenance[name] = tuple(sources)


def _validate_subject(component: object, subject: ReplaySubject, label: str) -> None:
    component_subject = getattr(component, "subject", None)
    if component_subject is None or not same_subject_id(component_subject, subject):
        raise ValueError(f"{label} must preserve the exact ReplaySubject.")


def _validate_visible_component(
    inputs: SubjectVisibleInputs,
    sources: tuple[EvidenceSourceObservation, ...],
    label: str,
) -> None:
    visible = tuple(
        source
        for observation in inputs.base_observations + inputs.context_observations
        for source in observation.timing.source_observations
    )
    if any(source not in visible for source in sources):
        raise ValueError(f"{label} provenance references evidence outside visible inputs.")


def _liquidity_is_resolved(value: CoreV1LiquidityAssessment) -> bool:
    return value.breach is not None and value.reaction is not None


def _liquidity_sources(
    value: CoreV1LiquidityAssessment,
) -> tuple[EvidenceSourceObservation, ...]:
    sources = list(value.level.timing.source_observations)
    if value.breach is not None:
        sources.extend(value.breach.timing.source_observations)
    if value.reaction is not None:
        sources.extend(value.reaction.timing.source_observations)
    if value.sweep_timing is not None:
        sources.extend(value.sweep_timing.source_observations)
    unique: list[EvidenceSourceObservation] = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    return tuple(unique)


def _zone_timing(zone: ReactionaryZone):
    return zone.timing


def _utc_timestamp(value: object):
    import pandas as pd

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("availability_timestamp must be timezone-aware.")
    return timestamp.tz_convert("UTC")


__all__ = [
    "CoreV1PipelineResult",
    "evaluate_core_v1_pipeline",
]
