from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
import pytest

from src.core_v1_dealing_range import DealingRangeContext, DealingRangeStatus
from src.core_v1_displacement import (
    CoreV1DisplacementAssessment,
    CoreV1DisplacementObservation,
    DisplacementStatus,
)
from src.core_v1_evidence import (
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
)
from src.core_v1_liquidity import (
    CoreV1LiquidityAssessment,
    CoreV1LiquidityBreach,
    CoreV1LiquidityLevel,
    CoreV1LiquidityReaction,
    ReactionKind,
    SweepKind,
)
from src.core_v1_mss import (
    CoreV1MSSConfirmation,
    CoreV1MSSResult,
    CoreV1StructuralBreak,
    CoreV1StructuralPoint,
    MSSStatus,
    evaluate_core_v1_mss,
)
from src.core_v1_reactionary_zones import (
    CoreV1FVGObservation,
    CoreV1OrderBlockObservation,
    evaluate_core_v1_fvg,
)
from src.core_v1_retracement import (
    CoreV1PlannedEntry,
    CoreV1ZoneInteraction,
    evaluate_core_v1_retracement,
)
from src.dealing_range import DealingRange, RangeLocation
from src.displacement import Direction
from src.evidence_timing import EvidenceSourceObservation, EvidenceTiming
from src.fvg import FVG, FVGDirection
from src.market_structure import LiquiditySide, SwingType
from src.replay_subject import ReplaySubject
from src.sniper_setup import PrecisionState


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def make_source(
    source_id: str,
    event: str,
    available: str | None = None,
    *,
    base: bool = True,
) -> EvidenceSourceObservation:
    event_time = ts(event)
    return EvidenceSourceObservation(
        source_id=source_id,
        source="synthetic",
        event_timestamp=event_time,
        availability_timestamp=ts(available or event),
        provenance={"source_id": source_id},
        base_bar_timestamp=event_time if base else None,
        is_base_timeframe=base,
    )


def make_timing(
    event: str,
    sources: tuple[EvidenceSourceObservation, ...],
    *,
    confirmation: str | None = None,
    availability: str | None = None,
    base_scoped: bool = True,
) -> EvidenceTiming:
    return EvidenceTiming(
        event_timestamp=ts(event),
        confirmation_timestamp=ts(confirmation) if confirmation else None,
        availability_timestamp=ts(availability) if availability else max(
            item.availability_timestamp for item in sources
        ),
        source_observations=sources,
        provenance={"governance": "synthetic-test"},
        base_bar_scoped=base_scoped,
    )


@dataclass
class Fixture:
    subject: ReplaySubject
    inputs: SubjectVisibleInputs
    sources: dict[str, EvidenceSourceObservation]
    liquidity: CoreV1LiquidityAssessment
    displacement: CoreV1DisplacementAssessment
    mss: CoreV1MSSResult
    zone: CoreV1FVGObservation
    interaction: CoreV1ZoneInteraction
    planned_entry: CoreV1PlannedEntry


def make_fixture() -> Fixture:
    current = ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:06:10Z"),
    )
    sources = {
        "point": make_source("point", "2026-01-01T10:00:00Z"),
        "liquidity-level": make_source("liquidity-level", "2026-01-01T10:00:30Z"),
        "liquidity-breach": make_source("liquidity-breach", "2026-01-01T10:01:00Z"),
        "liquidity-confirmation": make_source(
            "liquidity-confirmation", "2026-01-01T10:01:30Z"
        ),
        "displacement": make_source("displacement", "2026-01-01T10:02:00Z"),
        "displacement-confirmation": make_source(
            "displacement-confirmation", "2026-01-01T10:02:30Z"
        ),
        "break": make_source("break", "2026-01-01T10:03:00Z"),
        "follow-through": make_source(
            "follow-through",
            "2026-01-01T10:05:00Z",
            "2026-01-01T10:06:17Z",
        ),
        "fvg-first": make_source("fvg-first", "2026-01-01T09:55:00Z"),
        "fvg-third": make_source("fvg-third", "2026-01-01T10:05:00Z"),
        "interaction": make_source("interaction", "2026-01-01T10:06:10Z"),
        "entry-resolver": make_source(
            "entry-resolver",
            "2026-01-01T10:06:17Z",
            "2026-01-01T10:06:17Z",
            base=False,
        ),
    }
    visible = tuple(
        SubjectVisibleObservation(
            value={"source_id": source.source_id},
            timing=make_timing(
                source.event_timestamp.isoformat(),
                (source,),
                availability=source.availability_timestamp.isoformat(),
                base_scoped=source.is_base_timeframe,
            ),
            scope=(
                ObservationScope.BASE
                if source.is_base_timeframe
                else ObservationScope.CONTEXT
            ),
        )
        for source in sources.values()
    )
    inputs = SubjectVisibleInputs(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=tuple(
            item for item in visible if item.scope is ObservationScope.BASE
        ),
        context_observations=tuple(
            item for item in visible if item.scope is ObservationScope.CONTEXT
        ),
    )
    point = CoreV1StructuralPoint(
        current,
        "meaningful-high",
        SwingType.HIGH,
        105.0,
        make_timing(
            "2026-01-01T10:00:00Z",
            (sources["point"],),
        ),
    )
    level = CoreV1LiquidityLevel(
        current,
        "liquidity-1",
        LiquiditySide.BSL,
        106.0,
        make_timing("2026-01-01T10:00:30Z", (sources["liquidity-level"],)),
    )
    breach = CoreV1LiquidityBreach(
        current,
        level,
        107.0,
        make_timing(
            "2026-01-01T10:01:00Z",
            (sources["liquidity-level"], sources["liquidity-breach"]),
        ),
    )
    reaction = CoreV1LiquidityReaction(
        current,
        breach,
        ReactionKind.REJECTION,
        make_timing(
            "2026-01-01T10:01:00Z",
            (
                sources["liquidity-level"],
                sources["liquidity-breach"],
                sources["liquidity-confirmation"],
            ),
            confirmation="2026-01-01T10:01:30Z",
        ),
    )
    liquidity = CoreV1LiquidityAssessment(
        current,
        level,
        breach=breach,
        reaction=reaction,
        sweep=SweepKind.UNRESOLVED,
    )
    displacement_source = sources["displacement"]
    displacement_confirmation = sources["displacement-confirmation"]
    displacement_observation = CoreV1DisplacementObservation(
        current,
        make_timing(
            "2026-01-01T10:02:00Z",
            (displacement_source, displacement_confirmation),
            confirmation="2026-01-01T10:02:30Z",
        ),
        direction=Direction.BULLISH,
    )
    displacement = CoreV1DisplacementAssessment(
        current,
        DisplacementStatus.DISPLACEMENT,
        current.availability_timestamp,
        observation=displacement_observation,
    )
    structural_break = CoreV1StructuralBreak(
        current,
        point,
        108.0,
        make_timing(
            "2026-01-01T10:03:00Z",
            (sources["point"], sources["break"]),
        ),
        direction=Direction.BULLISH,
    )
    confirmation = CoreV1MSSConfirmation(
        current,
        structural_break,
        make_timing(
            "2026-01-01T10:03:00Z",
            (
                sources["point"],
                sources["break"],
                sources["follow-through"],
            ),
            confirmation="2026-01-01T10:05:00Z",
            availability=current.availability_timestamp.isoformat(),
        ),
    )
    mss = evaluate_core_v1_mss(
        inputs,
        structural_point=point,
        liquidity_event=liquidity,
        displacement=displacement,
        structural_break=structural_break,
        confirmation=confirmation,
    )
    fvg = FVG(
        FVGDirection.BULLISH,
        ts("2026-01-01T10:05:00Z"),
        ts("2026-01-01T10:05:00Z"),
        ts("2026-01-01T09:55:00Z"),
        ts("2026-01-01T10:05:00Z"),
        101.0,
        103.0,
        102.0,
        2.0,
    )
    zone_timing = make_timing(
        "2026-01-01T10:05:00Z",
        (sources["fvg-first"], sources["fvg-third"]),
    )
    zone = CoreV1FVGObservation(current, fvg, zone_timing)
    interaction_sources = (
        *zone_timing.source_observations,
        sources["interaction"],
    )
    interaction = CoreV1ZoneInteraction(
        current,
        zone,
        make_timing(
            "2026-01-01T10:06:10Z",
            interaction_sources,
        ),
        price=102.0,
    )
    entry_sources = tuple(sources.values())
    planned_entry = CoreV1PlannedEntry(
        current,
        102.25,
        make_timing(
            "2026-01-01T10:06:17Z",
            entry_sources,
            availability=current.availability_timestamp.isoformat(),
            base_scoped=False,
        ),
    )
    return Fixture(
        current,
        inputs,
        sources,
        liquidity,
        displacement,
        mss,
        zone,
        interaction,
        planned_entry,
    )


@dataclass
class FixedEntryResolver:
    result: CoreV1PlannedEntry | None
    received: object | None = None

    def qualify(self, inputs):
        self.received = inputs
        return self.result


def evaluate(fixture: Fixture, **overrides):
    args = {
        "mss": fixture.mss,
        "liquidity": fixture.liquidity,
        "displacement": fixture.displacement,
        "zone": fixture.zone,
        "interaction": fixture.interaction,
    }
    args.update(overrides)
    return evaluate_core_v1_retracement(fixture.inputs, **args)


def test_valid_governed_entry_is_planned_only_and_preserves_evidence():
    fixture = make_fixture()
    resolver = FixedEntryResolver(fixture.planned_entry)
    result = evaluate(fixture, resolver=resolver)

    assert result.state is PrecisionState.VALID
    assert result.subject is fixture.subject
    assert result.planned_entry is fixture.planned_entry
    assert result.planned_entry_price == 102.25
    assert result.event_timestamp == ts("2026-01-01T10:06:17Z")
    assert result.interaction_timestamp == ts("2026-01-01T10:06:10Z")
    assert result.confirmation_timestamp is None
    assert result.availability_timestamp == fixture.subject.availability_timestamp
    assert result.zone is fixture.zone
    assert result.mss is fixture.mss
    assert result.liquidity is fixture.liquidity
    assert result.displacement is fixture.displacement
    assert result.provenance["mss"]
    assert result.provenance["liquidity"]
    assert result.provenance["displacement"]
    assert result.provenance["reactionary_zone"] == fixture.zone.timing.source_observations
    assert result.provenance["zone_interaction"] == fixture.interaction.timing.source_observations
    assert result.provenance["entry_resolver"] == fixture.planned_entry.timing.source_observations
    assert resolver.received.visible_inputs is fixture.inputs


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"liquidity": None}, "liquidity"),
        ({"displacement": None}, "displacement"),
        ({"mss": None}, "MSS"),
    ],
)
def test_missing_mandatory_sequence_component_fails_closed(overrides, reason):
    fixture = make_fixture()
    result = evaluate(fixture, resolver=FixedEntryResolver(fixture.planned_entry), **overrides)
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None
    assert reason.lower() in result.reason.lower()


def test_unresolved_mss_fails_closed_even_with_other_sequence_evidence():
    fixture = make_fixture()
    unresolved = CoreV1MSSResult(
        fixture.subject,
        MSSStatus.NOT_MSS_UNRESOLVED,
        fixture.subject.availability_timestamp,
        reason="missing governed follow-through",
    )
    result = evaluate(
        fixture,
        mss=unresolved,
        resolver=FixedEntryResolver(fixture.planned_entry),
    )
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None
    assert "MSS" in result.reason


def test_mss_cannot_be_combined_with_different_liquidity_or_displacement_results():
    fixture = make_fixture()
    with pytest.raises(ValueError, match="supplied governed liquidity"):
        evaluate(
            fixture,
            liquidity=replace(fixture.liquidity),
            resolver=FixedEntryResolver(fixture.planned_entry),
        )
    with pytest.raises(ValueError, match="supplied governed displacement"):
        evaluate(
            fixture,
            displacement=replace(fixture.displacement),
            resolver=FixedEntryResolver(fixture.planned_entry),
        )


def test_unresolved_displacement_is_not_qualified():
    fixture = make_fixture()
    unresolved = CoreV1DisplacementAssessment(
        fixture.subject,
        DisplacementStatus.UNRESOLVED,
        fixture.subject.availability_timestamp,
    )
    result = evaluate(
        fixture,
        displacement=unresolved,
        resolver=FixedEntryResolver(fixture.planned_entry),
    )
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None


@pytest.mark.parametrize("zone_kind", ["FVG", "OB"])
def test_touch_without_governed_entry_resolver_never_creates_entry(zone_kind):
    fixture = make_fixture()
    if zone_kind == "OB":
        zone = CoreV1OrderBlockObservation(
            fixture.subject,
            "governed-ob-1",
            100.0,
            103.0,
            fixture.zone.timing,
        )
        interaction_sources = (*zone.timing.source_observations, fixture.sources["interaction"])
        interaction = CoreV1ZoneInteraction(
            fixture.subject,
            zone,
            make_timing("2026-01-01T10:06:10Z", interaction_sources),
            price=102.0,
        )
    else:
        zone = fixture.zone
        interaction = fixture.interaction
    result = evaluate(fixture, zone=zone, interaction=interaction, resolver=None)
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None
    assert "NO_ENTRY_UNRESOLVED" in result.reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"mss": None, "liquidity": None, "displacement": None},
        {"mss": None, "liquidity": None},
        {"mss": None, "displacement": None},
    ],
)
def test_zone_touch_liquidity_or_mss_without_full_sequence_never_qualifies(overrides):
    fixture = make_fixture()
    result = evaluate(
        fixture,
        resolver=FixedEntryResolver(fixture.planned_entry),
        **overrides,
    )
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None


def test_missing_zone_and_interaction_can_only_qualify_through_explicit_policy():
    fixture = make_fixture()
    result = evaluate(
        fixture,
        zone=None,
        interaction=None,
        resolver=FixedEntryResolver(fixture.planned_entry),
    )
    assert result.state is PrecisionState.VALID
    assert result.zone is None
    assert result.interaction is None
    assert result.planned_entry_price == fixture.planned_entry.planned_entry_price


def test_zone_required_by_caller_policy_is_missing_no_entry():
    fixture = make_fixture()

    class ZoneRequiredResolver:
        def qualify(self, inputs):
            if inputs.zone is None or inputs.interaction is None:
                return None
            return fixture.planned_entry

    result = evaluate(
        fixture,
        zone=None,
        interaction=None,
        resolver=ZoneRequiredResolver(),
    )
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None
    assert "NO_ENTRY_UNRESOLVED" in result.reason


def test_zone_interaction_must_reference_exact_zone_identity():
    fixture = make_fixture()
    different_zone = replace(fixture.zone)
    with pytest.raises(ValueError, match="supplied zone identity"):
        evaluate(fixture, zone=different_zone)


def test_resolver_is_given_context_only_after_governed_prerequisites():
    fixture = make_fixture()
    resolver = FixedEntryResolver(None)
    result = evaluate(fixture, resolver=resolver)
    assert resolver.received is not None
    assert resolver.received.mss is fixture.mss
    assert resolver.received.liquidity is fixture.liquidity
    assert resolver.received.displacement is fixture.displacement
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None
    assert "NO_ENTRY_UNRESOLVED" in result.reason


def test_fvg_touch_and_plausible_complete_story_without_policy_is_not_a_trade():
    fixture = make_fixture()
    assert fixture.zone.fvg.contains(fixture.interaction.price)
    result = evaluate(fixture, resolver=None)
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None
    assert "NO_ENTRY_UNRESOLVED" in result.reason


def test_planned_entry_timestamp_and_confirmation_are_preserved():
    fixture = make_fixture()
    confirmed_source = make_source(
        "entry-confirmation",
        "2026-01-01T10:06:17Z",
        "2026-01-01T10:06:17Z",
        base=False,
    )
    confirmation_observation = SubjectVisibleObservation(
        value={"source_id": confirmed_source.source_id},
        timing=make_timing(
            confirmed_source.event_timestamp.isoformat(),
            (confirmed_source,),
            availability=confirmed_source.availability_timestamp.isoformat(),
            base_scoped=False,
        ),
        scope=ObservationScope.CONTEXT,
    )
    fixture = replace(
        fixture,
        inputs=replace(
            fixture.inputs,
            context_observations=(
                *fixture.inputs.context_observations,
                confirmation_observation,
            ),
        ),
    )
    entry_sources = (
        *fixture.planned_entry.timing.source_observations,
        confirmed_source,
    )
    planned = CoreV1PlannedEntry(
        fixture.subject,
        fixture.planned_entry.planned_entry_price,
        make_timing(
            "2026-01-01T10:06:17Z",
            entry_sources,
            confirmation="2026-01-01T10:06:17Z",
            availability="2026-01-01T10:06:17Z",
            base_scoped=False,
        ),
    )
    result = evaluate(fixture, resolver=FixedEntryResolver(planned))
    assert result.event_timestamp == planned.timing.event_timestamp
    assert result.confirmation_timestamp == planned.timing.confirmation_timestamp


def test_future_entry_confirmation_is_rejected():
    fixture = make_fixture()
    future = make_source(
        "future-entry-confirmation",
        "2026-01-01T10:06:20Z",
        "2026-01-01T10:06:20Z",
        base=False,
    )
    with pytest.raises(ValueError, match="unavailable"):
        CoreV1PlannedEntry(
            fixture.subject,
            102.25,
            make_timing(
                "2026-01-01T10:06:17Z",
                (*fixture.planned_entry.timing.source_observations, future),
                confirmation="2026-01-01T10:06:20Z",
                availability="2026-01-01T10:06:20Z",
                base_scoped=False,
            ),
        )


def test_future_planned_entry_source_is_rejected_by_subject_availability():
    fixture = make_fixture()
    future = make_source(
        "future-entry",
        "2026-01-01T10:06:20Z",
        "2026-01-01T10:06:20Z",
        base=False,
    )
    with pytest.raises(ValueError, match="unavailable"):
        CoreV1PlannedEntry(
            fixture.subject,
            102.25,
            make_timing(
                "2026-01-01T10:06:20Z",
                (*fixture.planned_entry.timing.source_observations, future),
                availability="2026-01-01T10:06:20Z",
                base_scoped=False,
            ),
        )


def test_planned_entry_cannot_precede_prerequisite_availability_or_zone_interaction():
    fixture = make_fixture()
    too_early = CoreV1PlannedEntry(
        fixture.subject,
        102.25,
        make_timing(
            "2026-01-01T10:06:10Z",
            fixture.planned_entry.timing.source_observations,
            availability="2026-01-01T10:06:17Z",
            base_scoped=False,
        ),
    )
    with pytest.raises(ValueError, match="precedes prerequisite availability"):
        evaluate(fixture, resolver=FixedEntryResolver(too_early))


def test_planned_entry_provenance_must_trace_every_prerequisite_and_be_visible():
    fixture = make_fixture()
    only_entry_source = fixture.sources["entry-resolver"]
    untraceable = CoreV1PlannedEntry(
        fixture.subject,
        102.25,
        make_timing(
            "2026-01-01T10:06:17Z",
            (only_entry_source,),
            availability="2026-01-01T10:06:17Z",
            base_scoped=False,
        ),
    )
    with pytest.raises(ValueError, match="retain every supplied prerequisite"):
        evaluate(fixture, resolver=FixedEntryResolver(untraceable))

    hidden = make_source(
        "hidden-entry",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:00Z",
        base=False,
    )
    hidden_entry = CoreV1PlannedEntry(
        fixture.subject,
        102.25,
        make_timing(
            "2026-01-01T10:06:17Z",
            (*fixture.planned_entry.timing.source_observations, hidden),
            availability="2026-01-01T10:06:17Z",
            base_scoped=False,
        ),
    )
    with pytest.raises(ValueError, match="outside visible inputs"):
        evaluate(fixture, resolver=FixedEntryResolver(hidden_entry))


def test_entry_event_can_follow_base_bar_cutoff_without_widening_bar_evidence():
    fixture = make_fixture()
    assert fixture.planned_entry.timing.event_timestamp > fixture.subject.base_bar_timestamp
    result = evaluate(fixture, resolver=FixedEntryResolver(fixture.planned_entry))
    assert result.state is PrecisionState.VALID
    assert result.availability_timestamp == fixture.subject.availability_timestamp


def test_future_base_bar_interaction_is_rejected_even_before_availability_cutoff():
    fixture = make_fixture()
    future_touch = make_source(
        "future-touch",
        "2026-01-01T10:06:12Z",
        "2026-01-01T10:06:12Z",
    )
    future_observation = SubjectVisibleObservation(
        value={"source_id": future_touch.source_id},
        timing=make_timing(
            future_touch.event_timestamp.isoformat(),
            (future_touch,),
            base_scoped=True,
        ),
        scope=ObservationScope.BASE,
    )
    with pytest.raises(ValueError, match="base-bar"):
        SubjectVisibleInputs(
            fixture.subject,
            fixture.subject.availability_timestamp,
            (*fixture.inputs.base_observations, future_observation),
            fixture.inputs.context_observations,
        )


def test_replay_subject_identity_is_preserved_and_mismatches_are_rejected():
    fixture = make_fixture()
    result = evaluate(fixture, resolver=FixedEntryResolver(fixture.planned_entry))
    assert result.subject is fixture.subject

    other = ReplaySubject(
        fixture.subject.availability_timestamp + pd.Timedelta(minutes=1),
        fixture.subject.base_bar_timestamp,
    )
    foreign_entry = replace(fixture.planned_entry, subject=other)
    with pytest.raises(ValueError, match="another ReplaySubject"):
        evaluate(fixture, resolver=FixedEntryResolver(foreign_entry))


def test_full_frame_inputs_are_rejected():
    fixture = make_fixture()
    frame = pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0]})
    with pytest.raises(TypeError, match="SubjectVisibleInputs"):
        evaluate_core_v1_retracement(
            frame,  # type: ignore[arg-type]
            mss=fixture.mss,
            liquidity=fixture.liquidity,
            displacement=fixture.displacement,
        )


def test_future_sentinel_cannot_reach_resolver_or_change_earlier_result():
    fixture = make_fixture()
    baseline = evaluate(fixture, resolver=FixedEntryResolver(fixture.planned_entry))
    future_source = make_source(
        "future-sentinel",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
        base=False,
    )
    future_observation = SubjectVisibleObservation(
        value={"sentinel": 999999.0},
        timing=make_timing(
            future_source.event_timestamp.isoformat(),
            (future_source,),
            availability=future_source.availability_timestamp.isoformat(),
            base_scoped=False,
        ),
        scope=ObservationScope.CONTEXT,
    )
    with pytest.raises(ValueError, match="unavailable"):
        SubjectVisibleInputs(
            fixture.subject,
            fixture.subject.availability_timestamp,
            fixture.inputs.base_observations,
            (*fixture.inputs.context_observations, future_observation),
        )
    assert baseline.planned_entry_price == fixture.planned_entry.planned_entry_price


def test_price_is_only_a_planned_entry_not_an_execution_fill():
    fixture = make_fixture()
    result = evaluate(fixture, resolver=FixedEntryResolver(fixture.planned_entry))
    assert result.planned_entry_price == 102.25
    assert result.planned_entry_price is not None
    assert not hasattr(result, "execution_price")
    assert not hasattr(result, "fill_price")
    assert not hasattr(result, "fill")


def test_contract_never_infers_quantity_or_invokes_execution():
    fixture = make_fixture()
    result = evaluate(fixture, resolver=FixedEntryResolver(fixture.planned_entry))
    assert not hasattr(result, "quantity")
    assert not hasattr(result, "executable_quantity")
    assert not hasattr(result, "order")
    assert not hasattr(result, "fill")


def test_contract_does_not_import_or_invoke_tick_execution():
    import src.core_v1_retracement as retracement_module

    assert "tick_execution" not in retracement_module.__dict__
    assert not hasattr(retracement_module, "execute_tick")
    assert not hasattr(retracement_module, "simulate_fill")


def test_premium_discount_or_wick_only_context_does_not_create_entry():
    fixture = make_fixture()
    context_source = fixture.sources["point"]
    context = DealingRangeContext(
        subject=fixture.subject,
        status=DealingRangeStatus.RESOLVED,
        availability_timestamp=fixture.subject.availability_timestamp,
        event_timestamp=ts("2026-01-01T10:00:00Z"),
        dealing_range=DealingRange(
            ts("2026-01-01T10:00:00Z"),
            120.0,
            ts("2026-01-01T09:55:00Z"),
            100.0,
            110.0,
        ),
        provenance=(context_source,),
        reference_price=105.0,
        location=RangeLocation.DISCOUNT,
    )
    result = evaluate_core_v1_retracement(
        fixture.inputs,
        mss=None,
        liquidity=None,
        displacement=None,
        context=context,
        resolver=FixedEntryResolver(fixture.planned_entry),
    )
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None


def test_wick_only_observation_cannot_qualify_without_the_sequence():
    fixture = make_fixture()
    result = evaluate_core_v1_retracement(
        fixture.inputs,
        mss=None,
        liquidity=None,
        displacement=None,
        resolver=FixedEntryResolver(fixture.planned_entry),
    )
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry is None


def test_entry_state_uses_only_existing_four_state_vocabulary():
    assert {item.value for item in PrecisionState} == {
        "VALID",
        "DEVELOPING",
        "INVALID",
        "NO_TRADE",
    }
    fixture = make_fixture()
    unresolved = evaluate(fixture, resolver=None)
    resolved = evaluate(
        fixture,
        resolver=FixedEntryResolver(fixture.planned_entry),
    )
    assert unresolved.state in PrecisionState
    assert resolved.state in PrecisionState
    assert not hasattr(unresolved, "execution_state")
