from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.core_v1_evidence import ObservationScope, SubjectVisibleObservation
from src.core_v1_dealing_range import DealingRangeContext, DealingRangeStatus
from src.core_v1_mss import (
    CoreV1MSSResult,
    MSSStatus,
    evaluate_core_v1_mss,
)
from src.core_v1_pipeline import evaluate_core_v1_pipeline
from src.core_v1_reactionary_zones import CoreV1OrderBlockObservation
from src.evidence_timing import EvidenceTiming
from src.replay_subject import ReplaySubject
from src.sniper_setup import PrecisionState
from src.dealing_range import DealingRange, RangeLocation
import src.core_v1_pipeline as pipeline_module
from tests.test_core_v1_hypothesis import (
    FixedInvalidationResolver,
    FixedTargetResolver,
    components,
)
from tests.test_core_v1_retracement import (
    FixedEntryResolver,
    make_fixture,
    make_timing,
    ts,
)
from src.core_v1_displacement import DisplacementStatus
from src.core_v1_liquidity import (
    CoreV1LiquidityAssessment,
    CoreV1LiquidityLevel,
    ReactionKind,
    SweepKind,
    detect_core_v1_liquidity_levels,
    evaluate_core_v1_liquidity,
)


def with_extra_visible(inputs, source_observations):
    base = list(inputs.base_observations)
    context = list(inputs.context_observations)
    known = [
        source
        for observation in base + context
        for source in observation.timing.source_observations
    ]
    for source in source_observations:
        if source in known:
            continue
        timing = EvidenceTiming(
            event_timestamp=source.event_timestamp,
            availability_timestamp=source.availability_timestamp,
            source_observations=(source,),
            provenance={"pipeline_test": True},
            base_bar_scoped=source.is_base_timeframe,
        )
        value = (
            {
                "open": 107.25,
                "high": 107.5,
                "low": 94.5,
                "close": 107.0,
                "source_id": source.source_id,
            }
            if source.is_base_timeframe
            else {"source_id": source.source_id}
        )
        observation = SubjectVisibleObservation(
            value=value,
            timing=timing,
            scope=(
                ObservationScope.BASE
                if source.is_base_timeframe
                else ObservationScope.CONTEXT
            ),
        )
        (base if source.is_base_timeframe else context).append(observation)
        known.append(source)
    return replace(
        inputs,
        base_observations=tuple(
            sorted(base, key=lambda item: item.timing.event_timestamp)
        ),
        context_observations=tuple(context),
    )


def target_for_visible_swing(inputs, target):
    (level,) = tuple(
        candidate
        for candidate in detect_core_v1_liquidity_levels(inputs)
        if candidate.side is target.liquidity_level.side
    )
    selection_sources = tuple(
        source
        for source in target.timing.source_observations
        if source not in target.liquidity_level.timing.source_observations
    )
    timing = replace(
        target.timing,
        event_timestamp=level.timing.event_timestamp,
        source_observations=(
            *level.timing.source_observations,
            *selection_sources,
        ),
    )
    return replace(target, liquidity_level=level, timing=timing)


def pipeline_fixture():
    fixture = make_fixture()
    invalidation, target = components(fixture, fixture.planned_entry)
    invalidation_source = replace(
        invalidation.timing.source_observations[0],
        event_timestamp=ts("2026-01-01T10:01:15Z"),
        availability_timestamp=ts("2026-01-01T10:01:15Z"),
        base_bar_timestamp=ts("2026-01-01T10:01:15Z"),
    )
    invalidation_timing = replace(
        invalidation.timing,
        event_timestamp=invalidation_source.event_timestamp,
        availability_timestamp=invalidation_source.availability_timestamp,
        source_observations=(invalidation_source,),
    )
    invalidation = replace(invalidation, timing=invalidation_timing)
    target = target_for_visible_swing(fixture.inputs, target)
    added_sources = (
        *invalidation.timing.source_observations,
        *target.timing.source_observations,
        *target.liquidity_level.timing.source_observations,
    )
    inputs = with_extra_visible(fixture.inputs, added_sources)
    (liquidity,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(fixture.liquidity.level,),
    )
    mss = evaluate_core_v1_mss(
        inputs,
        structural_point=fixture.mss.structural_point,
        liquidity_event=liquidity,
        displacement=fixture.displacement,
        structural_break=fixture.mss.structural_break,
        confirmation=fixture.mss.confirmation,
    )
    fixture.liquidity = liquidity
    fixture.mss = mss
    entry_sources = list(fixture.planned_entry.timing.source_observations)
    for source in added_sources:
        if source not in entry_sources:
            entry_sources.append(source)
    entry_timing = replace(
        fixture.planned_entry.timing,
        source_observations=tuple(entry_sources),
    )
    planned_entry = replace(fixture.planned_entry, timing=entry_timing)
    kwargs = {
        "inputs": inputs,
        "liquidity": fixture.liquidity,
        "displacement": fixture.displacement,
        "mss": fixture.mss,
        "zone": fixture.zone,
        "interaction": fixture.interaction,
        "entry_resolver": FixedEntryResolver(planned_entry),
        "invalidation": invalidation,
        "target": target,
        "opposing_liquidity_levels": (target.liquidity_level,),
    }
    return fixture, kwargs


def evaluate(kwargs, **overrides):
    arguments = dict(kwargs)
    arguments.update(overrides)
    return evaluate_core_v1_pipeline(**arguments)


def _visible_price_at(inputs, timestamp, field):
    (observation,) = tuple(
        observation
        for observation in inputs.base_observations
        if observation.timing.event_timestamp == timestamp
    )
    return float(observation.value[field])


def test_complete_governed_pipeline_preserves_all_stages_and_timing():
    fixture, kwargs = pipeline_fixture()
    result = evaluate(kwargs)

    assert result.state is PrecisionState.VALID
    assert result.subject is fixture.subject
    assert result.availability_timestamp == fixture.subject.availability_timestamp
    assert result.hypothesis is not None
    assert result.hypothesis.state is PrecisionState.VALID
    assert result.hypothesis.planned_entry is result.retracement
    assert result.liquidity is fixture.liquidity
    assert result.displacement is fixture.displacement
    assert result.mss.subject is fixture.subject
    assert result.zone is fixture.zone
    assert result.opposing_liquidity_levels == (kwargs["target"].liquidity_level,)
    assert result.retracement.planned_entry is kwargs["entry_resolver"].result
    assert result.event_timestamps["liquidity_event"] == fixture.liquidity.breach.timing.event_timestamp
    assert result.event_timestamps["displacement"] == fixture.displacement.event_timestamp
    assert result.confirmation_timestamps["mss_confirmation"] == fixture.mss.confirmation_timestamp
    assert result.event_timestamps["target_liquidity_level"] == kwargs["target"].liquidity_level.timing.event_timestamp
    assert result.provenance["reactionary_zone"] == fixture.zone.timing.source_observations
    assert result.provenance[
        f"opposing_liquidity_candidate:{kwargs['target'].liquidity_level.level_id}"
    ] == kwargs["target"].liquidity_level.timing.source_observations
    assert result.provenance["invalidation"] == kwargs["invalidation"].timing.source_observations
    assert result.provenance["opposing_liquidity_target"] == kwargs["target"].timing.source_observations
    for name, sources in result.hypothesis.provenance.items():
        assert result.provenance[name] == sources
    assert set(result.hypothesis.provenance).issubset(result.provenance)


def test_pipeline_provenance_rejects_oversized_mapping_keys():
    _, kwargs = pipeline_fixture()
    result = evaluate(kwargs)
    with pytest.raises(TypeError, match="mapping keys"):
        replace(result, provenance={"k" * 4097: ()})


def test_pipeline_rejects_liquidity_level_without_visible_swing_recomputation():
    fixture, kwargs = pipeline_fixture()
    accepted = fixture.liquidity
    fabricated_level = CoreV1LiquidityLevel(
        subject=fixture.subject,
        level_id="fabricated-pipeline-level",
        side=accepted.level.side,
        price=105.0,
        timing=accepted.level.timing,
    )
    fabricated = CoreV1LiquidityAssessment(
        subject=fixture.subject,
        level=fabricated_level,
    )

    with pytest.raises(ValueError, match="recomputed from visible base observations"):
        evaluate(kwargs, liquidity=fabricated)


def test_pipeline_rejects_forged_breach_on_genuine_visible_swing():
    fixture, kwargs = pipeline_fixture()
    canonical = fixture.liquidity
    assert canonical.level in detect_core_v1_liquidity_levels(kwargs["inputs"])
    assert canonical.breach.price == _visible_price_at(
        kwargs["inputs"],
        canonical.breach.timing.event_timestamp,
        "high",
    )
    other_visible_high = next(
        float(observation.value["high"])
        for observation in kwargs["inputs"].base_observations
        if observation.timing.event_timestamp
        != canonical.breach.timing.event_timestamp
        and float(observation.value["high"]) > canonical.level.price
    )
    forged_breach = replace(canonical.breach, price=other_visible_high)
    forged = replace(
        canonical,
        breach=forged_breach,
        reaction=replace(canonical.reaction, breach=forged_breach),
    )

    with pytest.raises(ValueError, match="full assessment recomputed"):
        evaluate(kwargs, liquidity=forged, displacement=None)


def test_pipeline_rejects_forged_reaction_on_genuine_visible_swing():
    fixture, kwargs = pipeline_fixture()
    canonical = fixture.liquidity
    confirmation_price = _visible_price_at(
        kwargs["inputs"],
        canonical.reaction.timing.confirmation_timestamp,
        "low",
    )
    assert canonical.reaction.price == confirmation_price
    other_visible_low = next(
        float(observation.value["low"])
        for observation in kwargs["inputs"].base_observations
        if observation.timing.event_timestamp
        != canonical.reaction.timing.confirmation_timestamp
        and float(observation.value["low"]) != canonical.reaction.price
        and float(observation.value["low"]) <= canonical.level.price
    )
    forged = replace(
        canonical,
        reaction=replace(canonical.reaction, price=other_visible_low),
    )

    with pytest.raises(ValueError, match="full assessment recomputed"):
        evaluate(kwargs, liquidity=forged, displacement=None)


def test_pipeline_rejects_acceptance_reaction(monkeypatch):
    fixture, kwargs = pipeline_fixture()
    acceptance = replace(
        fixture.liquidity,
        reaction=replace(
            fixture.liquidity.reaction,
            kind=ReactionKind.ACCEPTANCE,
        ),
        sweep=SweepKind.UNRESOLVED,
        sweep_timing=None,
        sweep_provenance={},
    )
    monkeypatch.setattr(
        pipeline_module,
        "evaluate_core_v1_liquidity",
        lambda inputs, *, levels: (acceptance,),
    )

    with pytest.raises(ValueError, match="ACCEPTANCE reactions"):
        evaluate(kwargs, liquidity=acceptance, displacement=None)


def test_pipeline_rejects_opposing_candidate_without_visible_swing_recomputation():
    _, kwargs = pipeline_fixture()
    original_target = kwargs["target"]
    fabricated_level = CoreV1LiquidityLevel(
        subject=original_target.subject,
        level_id="fabricated-target-level",
        side=original_target.liquidity_level.side,
        price=105.0,
        timing=original_target.liquidity_level.timing,
    )
    fabricated_target = replace(
        original_target,
        liquidity_level=fabricated_level,
    )

    with pytest.raises(ValueError, match="recomputed from visible base observations"):
        evaluate(
            kwargs,
            target=fabricated_target,
            opposing_liquidity_levels=(fabricated_level,),
        )


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"liquidity": None}, "liquidity evidence is missing"),
        (
            {
                "liquidity": lambda fixture: replace(
                    fixture.liquidity,
                    breach=None,
                    reaction=None,
                    sweep=SweepKind.UNRESOLVED,
                    sweep_timing=None,
                    sweep_provenance={},
                )
            },
            "liquidity event is unresolved",
        ),
        ({"displacement": None}, "displacement evidence is missing or unresolved"),
        (
            {
                "displacement": lambda fixture: replace(
                    fixture.displacement,
                    status=DisplacementStatus.UNRESOLVED,
                    observation=None,
                )
            },
            "displacement evidence is missing or unresolved",
        ),
        ({"mss": None}, "MSS evidence is missing or unresolved"),
        (
            {
                "mss": lambda fixture: CoreV1MSSResult(
                    subject=fixture.subject,
                    status=MSSStatus.NOT_MSS_UNRESOLVED,
                    availability_timestamp=fixture.subject.availability_timestamp,
                    reason="unresolved",
                )
            },
            "MSS evidence is missing or unresolved",
        ),
    ],
)
def test_missing_or_unresolved_prerequisite_blocks_every_later_stage(
    overrides,
    expected_reason,
):
    fixture, kwargs = pipeline_fixture()
    resolved = {
        name: value(fixture) if callable(value) else value
        for name, value in overrides.items()
    }
    if expected_reason == "liquidity event is unresolved":
        breach_timestamp = fixture.liquidity.breach.timing.event_timestamp
        unresolved_inputs = replace(
            kwargs["inputs"],
            base_observations=tuple(
                observation
                for observation in kwargs["inputs"].base_observations
                if observation.timing.event_timestamp < breach_timestamp
            ),
        )
        (resolved["liquidity"],) = evaluate_core_v1_liquidity(
            unresolved_inputs,
            levels=(fixture.liquidity.level,),
        )
        kwargs["inputs"] = unresolved_inputs
    entry_resolver = kwargs["entry_resolver"]
    result = evaluate(kwargs, **resolved)

    assert result.state is PrecisionState.DEVELOPING
    assert result.hypothesis is None
    assert expected_reason in result.reason
    assert entry_resolver.received is None


def test_missing_zone_blocks_entry_even_when_every_other_stage_is_available():
    fixture, kwargs = pipeline_fixture()
    resolver = kwargs["entry_resolver"]
    result = evaluate(kwargs, zone=None)

    assert result.state is PrecisionState.DEVELOPING
    assert result.zone is None
    assert result.retracement.state is PrecisionState.DEVELOPING
    assert result.hypothesis is None
    assert resolver.received is None


def test_unresolved_entry_resolver_cannot_create_a_planned_entry():
    fixture, kwargs = pipeline_fixture()
    kwargs["entry_resolver"] = FixedEntryResolver(None)
    result = evaluate(kwargs)

    assert result.state is PrecisionState.DEVELOPING
    assert result.retracement.state is PrecisionState.DEVELOPING
    assert result.hypothesis is None


def test_unresolved_entry_never_invokes_invalidation_or_target_resolvers():
    fixture, kwargs = pipeline_fixture()
    invalidation_resolver = FixedInvalidationResolver(kwargs["invalidation"])
    target_resolver = FixedTargetResolver(kwargs["target"])
    kwargs["entry_resolver"] = FixedEntryResolver(None)
    result = evaluate(
        kwargs,
        invalidation=None,
        invalidation_resolver=invalidation_resolver,
        target=None,
        target_resolver=target_resolver,
    )

    assert result.state is PrecisionState.DEVELOPING
    assert result.hypothesis is None
    assert invalidation_resolver.received is None
    assert target_resolver.received is None


@pytest.mark.parametrize(
    ("invalidation", "target"),
    [(None, "keep"), ("keep", None)],
)
def test_missing_invalidation_or_target_blocks_complete_hypothesis(invalidation, target):
    _, kwargs = pipeline_fixture()
    overrides = {}
    if invalidation is None:
        overrides["invalidation"] = None
    if target is None:
        overrides["target"] = None
        overrides["opposing_liquidity_levels"] = ()
    result = evaluate(kwargs, **overrides)

    assert result.state is PrecisionState.DEVELOPING
    assert result.hypothesis is None
    assert "unresolved" in result.reason


@pytest.mark.parametrize(
    ("bypass", "overrides"),
    [
        ("FVG", {"liquidity": None, "displacement": None, "mss": None}),
        ("OB", {"liquidity": None, "displacement": None, "mss": None}),
        ("MSS", {"liquidity": None, "displacement": None}),
        ("liquidity", {"displacement": None, "mss": None}),
        ("displacement", {"liquidity": None, "mss": None}),
        ("range", {"liquidity": None, "displacement": None, "mss": None}),
        ("planned-entry", {"liquidity": None, "displacement": None, "mss": None}),
    ],
)
def test_single_stage_bypasses_never_produce_entry_or_hypothesis(bypass, overrides):
    fixture, kwargs = pipeline_fixture()
    resolver = kwargs["entry_resolver"]
    supplied = {
        "FVG": {"zone": fixture.zone},
        "OB": {
            "zone": CoreV1OrderBlockObservation(
                fixture.subject,
                "governed-ob",
                100.0,
                101.0,
                fixture.zone.timing,
            )
        },
        "MSS": {"mss": fixture.mss},
        "liquidity": {"liquidity": fixture.liquidity},
        "displacement": {"displacement": fixture.displacement},
        "range": {"context": None},
        "planned-entry": {"entry_resolver": resolver},
    }[bypass]
    supplied.update(overrides)
    result = evaluate(kwargs, **supplied)

    assert result.state is PrecisionState.DEVELOPING
    assert result.retracement is None
    assert result.hypothesis is None
    assert resolver.received is None


def test_premium_context_alone_does_not_qualify_an_entry():
    fixture, kwargs = pipeline_fixture()
    source_provenance = (
        fixture.sources["point"],
        fixture.sources["follow-through"],
    )
    context = DealingRangeContext(
        subject=fixture.subject,
        status=DealingRangeStatus.RESOLVED,
        availability_timestamp=fixture.subject.availability_timestamp,
        event_timestamp=ts("2026-01-01T10:05:00Z"),
        confirmation_timestamp=ts("2026-01-01T10:06:00Z"),
        dealing_range=DealingRange(
            anchor_high_timestamp=ts("2026-01-01T10:00:00Z"),
            anchor_high=120.0,
            anchor_low_timestamp=ts("2026-01-01T10:05:00Z"),
            anchor_low=100.0,
            equilibrium=110.0,
        ),
        provenance=source_provenance,
        reference_price=115.0,
        location=RangeLocation.PREMIUM,
    )
    resolver = kwargs["entry_resolver"]
    result = evaluate(
        kwargs,
        context=context,
        liquidity=None,
        displacement=None,
        mss=None,
    )

    assert result.context.location is RangeLocation.PREMIUM
    assert result.state is PrecisionState.DEVELOPING
    assert result.retracement is None
    assert result.hypothesis is None
    assert resolver.received is None


def test_contradictory_target_geometry_returns_invalid_without_hypothesis():
    fixture, kwargs = pipeline_fixture()
    invalidation = kwargs["invalidation"]
    target = kwargs["target"]
    inputs = with_extra_visible(
        kwargs["inputs"],
        (*invalidation.timing.source_observations, *target.timing.source_observations),
    )
    planned_entry = replace(
        kwargs["entry_resolver"].result,
        planned_entry_price=107.0,
    )
    result = evaluate(
        kwargs,
        inputs=inputs,
        entry_resolver=FixedEntryResolver(planned_entry),
        invalidation=invalidation,
        target=target,
        opposing_liquidity_levels=(target.liquidity_level,),
    )

    assert result.state is PrecisionState.INVALID
    assert result.hypothesis is None
    assert "INVALID" in result.reason


def test_causal_ordering_violation_in_mss_is_rejected():
    fixture, kwargs = pipeline_fixture()
    altered_point = replace(
        fixture.mss.structural_point,
        timing=make_timing(
            "2026-01-01T10:02:00Z",
            (fixture.sources["displacement"],),
        ),
    )
    altered_break = replace(
        fixture.mss.structural_break,
        structural_point=altered_point,
        timing=make_timing(
            "2026-01-01T10:03:00Z",
            (fixture.sources["displacement"], fixture.sources["break"]),
        ),
    )
    altered_confirmation = replace(
        fixture.mss.confirmation,
        structural_break=altered_break,
        timing=make_timing(
            "2026-01-01T10:03:00Z",
            (
                fixture.sources["displacement"],
                fixture.sources["break"],
                fixture.sources["follow-through"],
            ),
            confirmation="2026-01-01T10:05:00Z",
            availability=fixture.subject.availability_timestamp.isoformat(),
        ),
    )
    with pytest.raises(ValueError, match="Causal MSS ordering violation"):
        evaluate(
            kwargs,
            mss=replace(
                fixture.mss,
                structural_point=altered_point,
                structural_break=altered_break,
                confirmation=altered_confirmation,
            ),
        )


def test_future_or_foreign_component_is_rejected():
    fixture, kwargs = pipeline_fixture()
    other = make_fixture()
    foreign_subject = ReplaySubject(
        other.subject.availability_timestamp,
        other.subject.base_bar_timestamp - pd.Timedelta(minutes=1),
    )
    object.__setattr__(other.mss, "subject", foreign_subject)
    with pytest.raises(ValueError):
        evaluate(kwargs, mss=other.mss)


def test_pipeline_result_rejects_future_component_timestamps():
    _, kwargs = pipeline_fixture()
    result = evaluate(kwargs)
    with pytest.raises(ValueError, match="exceeds availability"):
        replace(
            result,
            event_timestamps={
                **result.event_timestamps,
                "future_component": ts("2026-01-01T10:07:00Z"),
            },
        )


def test_pipeline_result_accepts_metadata_matching_referenced_stages():
    _, kwargs = pipeline_fixture()
    result = evaluate(kwargs)

    rebuilt = replace(
        result,
        event_timestamps=dict(result.event_timestamps),
        confirmation_timestamps=dict(result.confirmation_timestamps),
        provenance=dict(result.provenance),
    )

    assert rebuilt.event_timestamps == result.event_timestamps
    assert rebuilt.confirmation_timestamps == result.confirmation_timestamps
    assert rebuilt.provenance == result.provenance


def test_pipeline_provenance_rejects_unreferenced_entries():
    _, kwargs = pipeline_fixture()
    result = evaluate(kwargs)

    with pytest.raises(ValueError, match="provenance must match the referenced stage evidence"):
        replace(
            result,
            provenance={**result.provenance, "unreferenced": ()},
        )


def test_pipeline_result_rejects_bounded_but_wrong_stage_timestamp():
    _, kwargs = pipeline_fixture()
    result = evaluate(kwargs)
    altered = dict(result.event_timestamps)
    altered["displacement"] = result.subject.base_bar_timestamp

    with pytest.raises(ValueError, match="match the referenced stage evidence"):
        replace(result, event_timestamps=altered)


def test_pipeline_result_rejects_stage_timestamp_mismatch():
    _, kwargs = pipeline_fixture()
    result = evaluate(kwargs)
    altered = dict(result.event_timestamps)
    altered["displacement"] = altered["displacement"] + pd.Timedelta(seconds=1)

    with pytest.raises(ValueError, match="match the referenced stage evidence"):
        replace(result, event_timestamps=altered)


def test_pipeline_result_rejects_bounded_but_wrong_stage_provenance():
    _, kwargs = pipeline_fixture()
    result = evaluate(kwargs)
    altered = dict(result.provenance)
    altered["displacement"] = result.provenance["mss_confirmation"]

    with pytest.raises(ValueError, match="match the referenced stage evidence"):
        replace(result, provenance=altered)


def test_pipeline_result_derives_metadata_when_maps_are_omitted():
    _, kwargs = pipeline_fixture()
    result = evaluate(kwargs)

    rebuilt = replace(
        result,
        event_timestamps=None,
        confirmation_timestamps=None,
        provenance=None,
    )

    assert rebuilt.event_timestamps == result.event_timestamps
    assert rebuilt.confirmation_timestamps == result.confirmation_timestamps
    assert rebuilt.provenance == result.provenance


def test_future_sentinel_cannot_be_used_when_mandatory_stage_is_missing():
    _, kwargs = pipeline_fixture()
    resolver = kwargs["entry_resolver"]
    earlier = evaluate(kwargs, mss=None)
    assert earlier.state is PrecisionState.DEVELOPING
    assert resolver.received is None
    assert earlier.hypothesis is None


def test_pipeline_rejects_unrestricted_frames():
    with pytest.raises(TypeError, match="SubjectVisibleInputs"):
        evaluate_core_v1_pipeline(pd.DataFrame({"close": [1.0]}))


def test_result_is_deterministic_and_separated_from_execution():
    fixture, kwargs = pipeline_fixture()
    first = evaluate(kwargs)
    second = evaluate(kwargs)

    assert first == second
    assert first.state is PrecisionState.VALID
    assert not hasattr(first, "quantity")
    assert not hasattr(first, "risk")
    assert not hasattr(first, "execution_intent")
    assert not hasattr(first, "order")
    assert not hasattr(first, "fill")
    assert not hasattr(first, "position")


def test_pipeline_module_has_no_execution_engine_surface():
    assert not hasattr(pipeline_module, "ExecutionEngine")
    assert not hasattr(pipeline_module, "ExecutionIntent")
    assert not hasattr(pipeline_module, "OrderRecord")
    assert not hasattr(pipeline_module, "FillRecord")
    assert not hasattr(pipeline_module, "PositionRecord")


def test_supplied_foreign_liquidity_reference_is_not_accepted_by_mss():
    fixture, kwargs = pipeline_fixture()
    foreign_subject = ReplaySubject(
        fixture.subject.availability_timestamp,
        fixture.subject.base_bar_timestamp - pd.Timedelta(minutes=1),
    )
    with pytest.raises(ValueError):
        object.__setattr__(fixture.liquidity, "subject", foreign_subject)
        evaluate(kwargs, liquidity=fixture.liquidity)
