from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
from src.core_v1_sessions import core_v1_entry_session
from src.core_v1_reactionary_zones import (
    CoreV1OrderBlockObservation,
    FVGResult,
    ZoneStatus,
)
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
    governed_directional_evidence,
    make_fixture,
    make_timing,
    ts,
)
from tests.test_core_v1_liquidity import make_inputs as make_liquidity_inputs
from tests.test_core_v1_liquidity import priced_inputs
from src.core_v1_displacement import DisplacementStatus
from tests.test_core_v1_displacement import swept_displacement_inputs
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
    candidates = tuple(
        candidate
        for candidate in detect_core_v1_liquidity_levels(inputs)
        if candidate.side is target.liquidity_level.side
    )
    beyond_seed = tuple(
        candidate
        for candidate in candidates
        if (
            candidate.price > target.price
            if candidate.side.value == "BSL"
            else candidate.price < target.price
        )
    )
    if beyond_seed:
        nearest = (
            min(candidate.price for candidate in beyond_seed)
            if beyond_seed[0].side.value == "BSL"
            else max(candidate.price for candidate in beyond_seed)
        )
        (level,) = tuple(candidate for candidate in beyond_seed if candidate.price == nearest)
    else:
        (level,) = candidates
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
    swing_bars = tuple(
        replace(item, value={**item.value, "high": 116.0})
        if item.timing.event_timestamp == ts("2026-01-01T10:04:59Z")
        else item
        for item in fixture.inputs.base_observations
    )
    fixture.inputs = replace(fixture.inputs, base_observations=swing_bars)
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
    context, draw_on_liquidity = governed_directional_evidence(
        inputs,
        fixture.displacement.observation.direction,
        before=fixture.liquidity.breach.timing.event_timestamp,
    )
    (liquidity,) = evaluate_core_v1_liquidity(
        inputs,
        levels=(fixture.liquidity.level,),
    )
    confirmation = fixture.mss.confirmation
    confirmation_sources = list(confirmation.timing.source_observations)
    for observation in inputs.base_observations:
        if (
            fixture.mss.structural_break.timing.event_timestamp
            < observation.timing.event_timestamp
            <= confirmation.timing.confirmation_timestamp
        ):
            for source in observation.timing.source_observations:
                if source not in confirmation_sources:
                    confirmation_sources.append(source)
    confirmation = replace(
        confirmation,
        timing=replace(
            confirmation.timing,
            source_observations=tuple(confirmation_sources),
        ),
    )
    mss = evaluate_core_v1_mss(
        inputs,
        structural_point=fixture.mss.structural_point,
        liquidity_event=liquidity,
        displacement=fixture.displacement,
        structural_break=fixture.mss.structural_break,
        confirmation=confirmation,
    )
    fixture.liquidity = liquidity
    fixture.mss = mss
    entry_sources = list(fixture.planned_entry.timing.source_observations)
    required_directional_sources = (
        *context.provenance,
        *draw_on_liquidity.timing.source_observations,
    )
    for source in (*added_sources, *required_directional_sources):
        if source not in entry_sources:
            entry_sources.append(source)
    entry_timing = replace(
        fixture.planned_entry.timing,
        source_observations=tuple(entry_sources),
    )
    planned_entry = replace(fixture.planned_entry, timing=entry_timing)
    kwargs = {
        "inputs": inputs,
        "context": context,
        "draw_on_liquidity": draw_on_liquidity,
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


@pytest.fixture(autouse=True)
def allow_legacy_synthetic_fixture_session(monkeypatch):
    """Keep pre-session synthetic fixtures focused on their existing contracts."""
    monkeypatch.setattr(pipeline_module, "core_v1_entry_session", lambda _: "LONDON")


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


def test_valid_planned_entry_inside_owner_session_window_reaches_hypothesis(monkeypatch):
    # The shared fixture is intentionally synthetic and predates the session
    # policy; model an otherwise-valid entry inside the selected window here.
    monkeypatch.setattr(pipeline_module, "core_v1_entry_session", lambda _: "NEW_YORK")
    _, kwargs = pipeline_fixture()

    result = evaluate(kwargs)

    assert result.retracement is not None
    assert result.retracement.planned_entry is not None
    assert result.hypothesis is not None
    assert result.hypothesis.state is PrecisionState.VALID


def test_planned_entry_at_session_end_is_blocked_before_hypothesis(monkeypatch):
    monkeypatch.setattr(pipeline_module, "core_v1_entry_session", core_v1_entry_session)
    fixture, kwargs = pipeline_fixture()
    source = replace(
        fixture.sources["entry-resolver"],
        event_timestamp=ts("2026-01-01T10:06:10Z"),
        availability_timestamp=ts("2026-01-01T10:06:10Z"),
        base_bar_timestamp=None,
        is_base_timeframe=False,
    )
    entry_timing = replace(
        kwargs["entry_resolver"].result.timing,
        event_timestamp=source.event_timestamp,
        source_observations=tuple(
            source if item.source_id == fixture.sources["entry-resolver"].source_id else item
            for item in kwargs["entry_resolver"].result.timing.source_observations
        ),
    )
    inputs = replace(
        kwargs["inputs"],
        context_observations=tuple(kwargs["inputs"].context_observations)
        + (
            SubjectVisibleObservation(
                value={"source_id": source.source_id},
                timing=EvidenceTiming.from_inputs(
                    event_timestamp=source.event_timestamp,
                    source_observations=(source,),
                    base_bar_scoped=False,
                ),
                scope=ObservationScope.CONTEXT,
            ),
        ),
    )
    planned = replace(kwargs["entry_resolver"].result, timing=entry_timing)

    result = evaluate(
        {**kwargs, "inputs": inputs, "entry_resolver": FixedEntryResolver(planned)},
    )

    assert result.state is PrecisionState.DEVELOPING
    assert result.hypothesis is None
    assert result.retracement is not None
    assert result.retracement.planned_entry is None
    assert "outside the configured ICT session" in result.reason


def test_target_breach_eligibility_uses_only_observations_available_at_planned_entry():
    fixture, kwargs = pipeline_fixture()
    retracement = evaluate(kwargs).retracement
    assert retracement is not None
    early_level = next(
        level
        for level in detect_core_v1_liquidity_levels(kwargs["inputs"])
        if level.side.value == "BSL" and level.price == 106.0
    )
    entry_source = fixture.sources["liquidity-level-right-2"]
    early_planned_entry = replace(
        retracement.planned_entry,
        planned_entry_price=100.0,
        timing=make_timing("2026-01-01T10:00:50Z", (entry_source,)),
    )
    early_entry = SimpleNamespace(
        displacement=retracement.displacement,
        planned_entry=early_planned_entry,
    )

    visible_at_entry = pipeline_module._visible_unbreached_opposing_levels(
        kwargs["inputs"], (early_level,), early_entry
    )
    visible_at_subject = pipeline_module._visible_unbreached_opposing_levels(
        kwargs["inputs"], (early_level,), retracement
    )

    assert visible_at_entry == (early_level,)
    assert visible_at_subject == ()


def test_pipeline_derives_visible_liquidity_and_keeps_later_stages_fail_closed():
    inputs = make_liquidity_inputs()

    result = evaluate_core_v1_pipeline(inputs=inputs)

    assert result.state is PrecisionState.DEVELOPING
    assert result.liquidity is not None
    assert result.liquidity.sweep is SweepKind.SWEEP
    assert result.liquidity.reaction is not None
    assert result.liquidity.reaction.kind is ReactionKind.REJECTION
    assert result.displacement is not None
    assert result.displacement.status is DisplacementStatus.UNRESOLVED
    assert result.hypothesis is None
    assert "displacement evidence is missing" in result.reason


def test_pipeline_derives_displacement_from_visible_price_action_by_default():
    inputs = swept_displacement_inputs()

    result = evaluate_core_v1_pipeline(inputs=inputs)

    assert result.liquidity is not None
    assert result.liquidity.sweep is SweepKind.SWEEP
    assert result.displacement is not None
    assert result.displacement.status is DisplacementStatus.DISPLACEMENT
    assert result.displacement.direction.value == "BEARISH"
    assert result.displacement.event_timestamp == ts("2026-01-01T10:28:00Z")
    assert result.state is PrecisionState.DEVELOPING
    assert result.mss is not None
    assert result.mss.status is MSSStatus.NOT_MSS_UNRESOLVED
    assert result.hypothesis is None


def test_automatic_liquidity_respects_the_subject_visibility_cutoff():
    rows = (
        ("2026-01-01T10:00:00Z", 100.0, 102.0, 98.0, 100.0),
        ("2026-01-01T10:01:00Z", 100.0, 104.0, 97.0, 101.0),
        ("2026-01-01T10:02:00Z", 101.0, 105.0, 95.0, 102.0),
        ("2026-01-01T10:03:00Z", 102.0, 103.0, 98.0, 100.0),
        ("2026-01-01T10:04:00Z", 100.0, 101.0, 99.0, 100.0),
        ("2026-01-01T10:05:00Z", 100.0, 106.0, 96.0, 105.5),
        ("2026-01-01T10:06:00Z", 105.0, 106.0, 94.0, 100.0),
    )
    before_reaction = evaluate_core_v1_pipeline(
        inputs=priced_inputs(rows[:6]),
    )
    after_reaction = evaluate_core_v1_pipeline(
        inputs=priced_inputs(rows),
    )

    assert before_reaction.liquidity is not None
    assert before_reaction.liquidity.breach is not None
    assert before_reaction.liquidity.reaction is None
    assert before_reaction.liquidity.sweep is SweepKind.UNRESOLVED
    assert after_reaction.liquidity is not None
    assert after_reaction.liquidity.sweep is SweepKind.SWEEP
    assert (
        before_reaction.liquidity.breach.timing.event_timestamp
        == after_reaction.liquidity.breach.timing.event_timestamp
    )


def test_automatic_liquidity_conflicts_fail_closed():
    inputs = priced_inputs(
        (
            ("2026-01-01T10:00:00Z", 100.0, 103.0, 97.0, 100.0),
            ("2026-01-01T10:01:00Z", 100.0, 104.0, 96.0, 101.0),
            ("2026-01-01T10:02:00Z", 101.0, 106.0, 94.0, 102.0),
            ("2026-01-01T10:03:00Z", 102.0, 104.0, 96.0, 100.0),
            ("2026-01-01T10:04:00Z", 100.0, 103.0, 97.0, 100.0),
            ("2026-01-01T10:05:00Z", 100.0, 107.0, 93.0, 100.0),
        )
    )

    result = evaluate_core_v1_pipeline(inputs=inputs)

    assert result.state is PrecisionState.DEVELOPING
    assert result.liquidity is None
    assert result.hypothesis is None
    assert "unavailable or ambiguous" in result.reason


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
        "low" if canonical.level.side.value == "SSL" else "high",
    )
    other_visible_breach_price = next(
        float(
            observation.value["low"]
            if canonical.level.side.value == "SSL"
            else observation.value["high"]
        )
        for observation in kwargs["inputs"].base_observations
        if observation.timing.event_timestamp
        != canonical.breach.timing.event_timestamp
        and float(
            observation.value["low"]
            if canonical.level.side.value == "SSL"
            else observation.value["high"]
        ) != canonical.breach.price
        and (
            float(observation.value["low"]) < canonical.level.price
            if canonical.level.side.value == "SSL"
            else float(observation.value["high"]) > canonical.level.price
        )
    )
    forged_breach = replace(canonical.breach, price=other_visible_breach_price)
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
        "high" if canonical.level.side.value == "SSL" else "low",
    )
    assert canonical.reaction.price == confirmation_price
    other_visible_reaction_price = next(
        float(
            observation.value["high"]
            if canonical.level.side.value == "SSL"
            else observation.value["low"]
        )
        for observation in kwargs["inputs"].base_observations
        if observation.timing.event_timestamp
        != canonical.reaction.timing.confirmation_timestamp
        and float(
            observation.value["high"]
            if canonical.level.side.value == "SSL"
            else observation.value["low"]
        ) != canonical.reaction.price
        and (
            float(observation.value["high"]) >= canonical.level.price
            if canonical.level.side.value == "SSL"
            else float(observation.value["low"]) <= canonical.level.price
        )
    )
    forged = replace(
        canonical,
        reaction=replace(canonical.reaction, price=other_visible_reaction_price),
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
        kwargs["context"], kwargs["draw_on_liquidity"] = governed_directional_evidence(
            unresolved_inputs,
            fixture.displacement.observation.direction,
            before=fixture.liquidity.breach.timing.event_timestamp,
        )
        kwargs["target"] = None
        kwargs["opposing_liquidity_levels"] = ()
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


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"context": None}, "resolved dealing-range context is required"),
        ({"draw_on_liquidity": None}, "explicit draw-on-liquidity evidence is required"),
    ],
)
def test_missing_directional_context_or_draw_blocks_canonical_entry(override, reason):
    fixture, kwargs = pipeline_fixture()
    resolver = kwargs["entry_resolver"]

    result = evaluate(kwargs, **override)

    assert result.state is PrecisionState.DEVELOPING
    assert result.retracement is not None
    assert result.retracement.state is PrecisionState.DEVELOPING
    assert reason in result.reason
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
        overrides["invalidation_resolver"] = FixedInvalidationResolver(None)
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
    if bypass == "displacement":
        unresolved_level = next(
            level
            for level in detect_core_v1_liquidity_levels(kwargs["inputs"])
            if level.side.value == "BSL" and level.price == 116.0
        )
        (supplied["liquidity"],) = evaluate_core_v1_liquidity(
            kwargs["inputs"],
            levels=(unresolved_level,),
        )
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


def test_target_beyond_entry_geometry_is_unresolved_without_eligible_candidate():
    fixture, kwargs = pipeline_fixture()
    invalidation = kwargs["invalidation"]
    target = kwargs["target"]
    inputs = with_extra_visible(
        kwargs["inputs"],
        (*invalidation.timing.source_observations, *target.timing.source_observations),
    )
    planned_entry = replace(
        kwargs["entry_resolver"].result,
        planned_entry_price=117.0,
    )
    result = evaluate(
        kwargs,
        inputs=inputs,
        entry_resolver=FixedEntryResolver(planned_entry),
        invalidation=invalidation,
        target=target,
        opposing_liquidity_levels=(target.liquidity_level,),
    )

    assert result.state is PrecisionState.DEVELOPING
    assert result.hypothesis is None
    assert "target" in result.reason


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
    with pytest.raises(
        ValueError,
        match="Structural point must be confirmed|Causal MSS ordering violation",
    ):
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
    fixture, kwargs = pipeline_fixture()
    resolver = kwargs["entry_resolver"]
    unresolved_mss = CoreV1MSSResult(
        subject=fixture.subject,
        status=MSSStatus.NOT_MSS_UNRESOLVED,
        availability_timestamp=fixture.subject.availability_timestamp,
        reason="unresolved",
    )
    earlier = evaluate(kwargs, mss=unresolved_mss)
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


def test_pipeline_runs_experimental_entry_through_structural_risk_gates(monkeypatch):
    fixture, kwargs = pipeline_fixture()
    calls = []

    def derive(inputs, *, displacement, mss, derive_visible):
        calls.append((inputs, displacement, mss, derive_visible))
        return FVGResult(inputs.subject, ZoneStatus.RESOLVED, fixture.zone)

    monkeypatch.setattr(pipeline_module, "evaluate_core_v1_fvg", derive)
    result = evaluate(
        kwargs,
        zone=None,
        interaction=None,
        entry_resolver=None,
    )

    assert calls == [(kwargs["inputs"], fixture.displacement, fixture.mss, True)]
    assert result.zone is fixture.zone
    assert result.interaction is not None
    assert result.interaction.price is None
    assert result.retracement is not None
    assert result.retracement.state is PrecisionState.VALID
    assert result.retracement.planned_entry is not None
    assert result.retracement.planned_entry_price == 113.0
    assert (
        result.retracement.planned_entry.timing.provenance["resolver"]
        == "core-v1-e009-fvg-quarter-rejection"
    )
    assert result.state is PrecisionState.VALID
    assert result.hypothesis is not None
    assert result.hypothesis.state is PrecisionState.VALID
