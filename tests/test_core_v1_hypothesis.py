from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.core_v1_hypothesis import (
    CoreV1OpposingLiquidityTarget,
    CoreV1StructuralInvalidation,
    CoreV1TradeHypothesis,
    evaluate_core_v1_hypothesis,
)
from src.core_v1_liquidity import CoreV1LiquidityLevel
from src.displacement import Direction
from src.evidence_timing import EvidenceSourceObservation, EvidenceTiming
from src.market_structure import LiquiditySide
from src.replay_subject import ReplaySubject
from src.sniper_setup import PrecisionState
from tests.test_core_v1_retracement import FixedEntryResolver, make_fixture
from src.core_v1_retracement import evaluate_core_v1_retracement


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def source(
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


def timing(
    event: str,
    sources: tuple[EvidenceSourceObservation, ...],
    *,
    confirmation: str | None = None,
    available: str | None = None,
    base_scoped: bool = False,
) -> EvidenceTiming:
    return EvidenceTiming(
        event_timestamp=ts(event),
        confirmation_timestamp=ts(confirmation) if confirmation else None,
        availability_timestamp=ts(available) if available else max(
            item.availability_timestamp for item in sources
        ),
        source_observations=sources,
        provenance={"resolver": "caller-governed-test"},
        base_bar_scoped=base_scoped,
    )


def valid_entry(fixture=None):
    fixture = fixture or make_fixture()
    result = evaluate_core_v1_retracement(
        fixture.inputs,
        mss=fixture.mss,
        liquidity=fixture.liquidity,
        displacement=fixture.displacement,
        zone=fixture.zone,
        interaction=fixture.interaction,
        resolver=FixedEntryResolver(fixture.planned_entry),
    )
    assert result.state is PrecisionState.VALID
    return fixture, result


def complete(entry, *, invalidation=None, target=None, **kwargs):
    levels = kwargs.pop("opposing_liquidity_levels", ())
    if target is not None and not levels:
        levels = (target.liquidity_level,)
    return evaluate_core_v1_hypothesis(
        entry,
        opposing_liquidity_levels=levels,
        invalidation=invalidation,
        target=target,
        **kwargs,
    )


def components(fixture, entry, *, side=LiquiditySide.BSL, invalidation_price=99, target_price=110):
    invalidation_source = source(
        "structural-invalidation",
        "2026-01-01T10:01:00Z",
    )
    invalidation = CoreV1StructuralInvalidation(
        subject=fixture.subject,
        invalidation_id="governed-structural-anchor",
        price=invalidation_price,
        timing=timing(
            "2026-01-01T10:01:00Z",
            (invalidation_source,),
            base_scoped=True,
        ),
    )
    level_source = source("opposing-liquidity-level", "2026-01-01T10:04:00Z")
    level = CoreV1LiquidityLevel(
        fixture.subject,
        "opposing-level",
        side,
        target_price,
        timing("2026-01-01T10:04:00Z", (level_source,), base_scoped=True),
    )
    selection_source = source(
        "target-selection",
        "2026-01-01T10:06:00Z",
        "2026-01-01T10:06:17Z",
        base=False,
    )
    target = CoreV1OpposingLiquidityTarget(
        subject=fixture.subject,
        target_id="governed-target",
        liquidity_level=level,
        timing=timing(
            "2026-01-01T10:04:00Z",
            (level_source, selection_source),
            confirmation="2026-01-01T10:06:00Z",
            available="2026-01-01T10:06:17Z",
        ),
    )
    return invalidation, target


class FixedInvalidationResolver:
    def __init__(self, result):
        self.result = result
        self.received = None

    def resolve(self, inputs):
        self.received = inputs
        return self.result


class FixedTargetResolver:
    def __init__(self, result):
        self.result = result
        self.received = None

    def resolve(self, inputs):
        self.received = inputs
        return self.result


def test_valid_long_complete_hypothesis_preserves_linkage_and_is_not_an_order():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    result = complete(
        entry,
        invalidation=invalidation,
        target=target,
    )

    assert result.state is PrecisionState.VALID
    assert result.subject is fixture.subject
    assert result.direction is Direction.BULLISH
    assert result.planned_entry is entry
    assert result.planned_entry_price == fixture.planned_entry.planned_entry_price
    assert result.invalidation is invalidation
    assert result.target is target
    assert invalidation.price < result.planned_entry_price < target.price
    assert result.planned_entry_timestamp == fixture.planned_entry.timing.event_timestamp
    assert not hasattr(result, "order")
    assert not hasattr(result, "execution_intent")
    assert not hasattr(result, "fill")
    assert not hasattr(result, "position")


def test_hypothesis_provenance_rejects_oversized_mapping_keys():
    fixture = make_fixture()
    with pytest.raises(TypeError, match="mapping keys"):
        CoreV1TradeHypothesis(
            subject=fixture.subject,
            state=PrecisionState.DEVELOPING,
            availability_timestamp=fixture.subject.availability_timestamp,
            reason="incomplete",
            provenance={"k" * 4097: ()},
        )


def test_valid_short_complete_hypothesis_uses_governed_bearish_direction():
    fixture, entry = valid_entry()
    short_displacement = replace(
        fixture.displacement,
        observation=replace(
            fixture.displacement.observation,
            direction=Direction.BEARISH,
        ),
    )
    short_break = replace(
        fixture.mss.structural_break,
        direction=Direction.BEARISH,
    )
    short_confirmation = replace(
        fixture.mss.confirmation,
        structural_break=short_break,
    )
    short_mss = replace(
        fixture.mss,
        displacement=short_displacement,
        structural_break=short_break,
        confirmation=short_confirmation,
    )
    short_entry = replace(
        entry,
        displacement=short_displacement,
        mss=short_mss,
    )
    invalidation, target = components(
        fixture,
        short_entry,
        side=LiquiditySide.SSL,
        invalidation_price=105.0,
        target_price=95.0,
    )

    result = complete(
        short_entry,
        invalidation=invalidation,
        target=target,
    )
    assert result.state is PrecisionState.VALID
    assert result.direction is Direction.BEARISH
    assert target.price < result.planned_entry_price < invalidation.price


def test_missing_or_unresolved_planned_entry_stays_developing():
    fixture = make_fixture()
    developing_entry = evaluate_core_v1_retracement(
        fixture.inputs,
        mss=None,
        liquidity=None,
        displacement=None,
    )
    result = evaluate_core_v1_hypothesis(developing_entry)
    assert result.state is PrecisionState.DEVELOPING
    assert result.planned_entry_price is None
    assert result.invalidation is None
    assert result.target is None


def test_missing_invalidation_fails_closed_without_fabricating_stop():
    fixture, entry = valid_entry()
    _, target = components(fixture, entry)
    result = complete(entry, target=target)
    assert result.state is PrecisionState.DEVELOPING
    assert result.invalidation is None
    assert entry.zone is fixture.zone
    assert "invalidation is unresolved" in result.reason


def test_missing_opposing_liquidity_fails_closed_without_fabricating_target():
    fixture, entry = valid_entry()
    invalidation, _ = components(fixture, entry)
    result = complete(entry, invalidation=invalidation)
    assert result.state is PrecisionState.DEVELOPING
    assert result.target is None
    assert "target is unresolved" in result.reason


def test_target_resolver_must_select_from_explicit_governed_liquidity_candidates():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    invalidation_resolver = FixedInvalidationResolver(invalidation)
    target_resolver = FixedTargetResolver(target)
    result = evaluate_core_v1_hypothesis(
        entry,
        opposing_liquidity_levels=(target.liquidity_level,),
        invalidation_resolver=invalidation_resolver,
        target_resolver=target_resolver,
    )

    assert result.state is PrecisionState.VALID
    assert target_resolver.received.opposing_liquidity_levels == (
        target.liquidity_level,
    )


def test_target_not_selected_from_supplied_liquidity_candidates_fails_closed():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    unrelated_source = source("unrelated-target-level", "2026-01-01T10:04:30Z")
    unrelated_level = CoreV1LiquidityLevel(
        fixture.subject,
        "unrelated-level",
        LiquiditySide.BSL,
        115.0,
        timing("2026-01-01T10:04:30Z", (unrelated_source,), base_scoped=True),
    )
    result = evaluate_core_v1_hypothesis(
        entry,
        opposing_liquidity_levels=(unrelated_level,),
        invalidation=invalidation,
        target=target,
    )
    assert result.state is PrecisionState.DEVELOPING
    assert "does not reference a supplied" in result.reason


def test_long_target_on_non_opposing_liquidity_side_is_invalid():
    fixture, entry = valid_entry()
    invalidation, wrong_side_target = components(
        fixture,
        entry,
        side=LiquiditySide.SSL,
        target_price=110.0,
    )
    result = complete(
        entry,
        invalidation=invalidation,
        target=wrong_side_target,
    )
    assert result.state is PrecisionState.INVALID
    assert "not on the opposing side" in result.reason


def test_unresolved_invalidation_and_target_resolvers_fail_closed():
    fixture, entry = valid_entry()
    invalidation_resolver = FixedInvalidationResolver(None)
    target_resolver = FixedTargetResolver(None)
    result = complete(
        entry,
        invalidation_resolver=invalidation_resolver,
        target_resolver=target_resolver,
        opposing_liquidity_levels=(),
    )
    assert result.state is PrecisionState.DEVELOPING
    assert invalidation_resolver.received.planned_entry is entry
    assert target_resolver.received.planned_entry is entry
    assert result.invalidation is None
    assert result.target is None


@pytest.mark.parametrize(
    ("direction", "invalidation_price", "side", "target_price", "reason"),
    [
        ("LONG", 103.0, LiquiditySide.BSL, 110.0, "LONG invalidation"),
        ("SHORT", 101.0, LiquiditySide.SSL, 95.0, "SHORT invalidation"),
        ("LONG", 99.0, LiquiditySide.BSL, 102.25, "LONG opposing-liquidity target"),
        ("LONG", 99.0, LiquiditySide.BSL, 100.0, "LONG opposing-liquidity target"),
        ("SHORT", 105.0, LiquiditySide.SSL, 102.25, "SHORT opposing-liquidity target"),
        ("SHORT", 105.0, LiquiditySide.SSL, 103.0, "SHORT opposing-liquidity target"),
    ],
)
def test_incoherent_directional_geometry_returns_invalid(
    direction,
    invalidation_price,
    side,
    target_price,
    reason,
):
    fixture, entry = valid_entry()
    if direction == "SHORT":
        short_displacement = replace(
            fixture.displacement,
            observation=replace(
                fixture.displacement.observation,
                direction=Direction.BEARISH,
            ),
        )
        short_break = replace(fixture.mss.structural_break, direction=Direction.BEARISH)
        short_mss = replace(
            fixture.mss,
            displacement=short_displacement,
            structural_break=short_break,
            confirmation=replace(
                fixture.mss.confirmation,
                structural_break=short_break,
            ),
        )
        entry = replace(entry, displacement=short_displacement, mss=short_mss)
    invalidation, target = components(
        fixture,
        entry,
        side=side,
        invalidation_price=invalidation_price,
        target_price=target_price,
    )
    result = complete(entry, invalidation=invalidation, target=target)
    assert result.state is PrecisionState.INVALID
    assert reason in result.reason


@pytest.mark.parametrize("price", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_invalidation_price_is_rejected(price):
    fixture, _ = valid_entry()
    invalidation_source = source("invalid-nonfinite", "2026-01-01T10:01:00Z")
    with pytest.raises(ValueError, match="finite"):
        CoreV1StructuralInvalidation(
            fixture.subject,
            "invalid",
            price,
            timing(
                "2026-01-01T10:01:00Z",
                (invalidation_source,),
                base_scoped=True,
            ),
        )


def test_invalid_target_level_price_is_rejected_as_nonfinite():
    fixture, _ = valid_entry()
    level_source = source("bad-target", "2026-01-01T10:04:00Z")
    with pytest.raises(ValueError, match="finite"):
        CoreV1LiquidityLevel(
            fixture.subject,
            "bad-level",
            LiquiditySide.BSL,
            float("nan"),
            timing("2026-01-01T10:04:00Z", (level_source,), base_scoped=True),
        )


def test_invalidation_target_and_prior_evidence_provenance_is_preserved():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    result = complete(entry, invalidation=invalidation, target=target)

    assert result.provenance["planned_entry"] == fixture.planned_entry.timing.source_observations
    assert result.provenance["invalidation"] == invalidation.timing.source_observations
    assert result.provenance["opposing_liquidity_target"] == target.timing.source_observations
    assert result.provenance["target_liquidity_level"] == target.liquidity_level.timing.source_observations
    assert result.provenance["mss"] == entry.provenance["mss"]
    assert result.provenance["liquidity"] == entry.provenance["liquidity"]
    assert result.provenance["displacement"] == entry.provenance["displacement"]


def test_valid_hypothesis_requires_exact_component_provenance():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    hypothesis = complete(entry, invalidation=invalidation, target=target)
    expected = {
        **entry.provenance,
        "planned_entry": entry.planned_entry.timing.source_observations,
        "invalidation": invalidation.timing.source_observations,
        "opposing_liquidity_target": target.timing.source_observations,
        "target_liquidity_level": target.liquidity_level.timing.source_observations,
    }

    assert hypothesis.provenance == expected


@pytest.mark.parametrize(
    "alter",
    [
        lambda provenance: {
            key: value for key, value in provenance.items() if key != "planned_entry"
        },
        lambda provenance: {**provenance, "unreferenced": ()},
        lambda provenance: {
            **provenance,
            "unreferenced": (source("plausible", "2026-01-01T10:05:00Z"),),
        },
        lambda provenance: {
            **provenance,
            "invalidation": (source("wrong", "2026-01-01T10:01:00Z"),),
        },
    ],
)
def test_valid_hypothesis_rejects_missing_extra_or_wrong_provenance(alter):
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    hypothesis = complete(entry, invalidation=invalidation, target=target)

    with pytest.raises(ValueError, match="provenance is incomplete or inconsistent"):
        replace(hypothesis, provenance=alter(dict(hypothesis.provenance)))


def test_event_confirmation_and_availability_timestamps_remain_distinct():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    result = complete(entry, invalidation=invalidation, target=target)

    assert result.planned_entry_timestamp == ts("2026-01-01T10:06:17Z")
    assert result.invalidation.timing.event_timestamp == ts("2026-01-01T10:01:00Z")
    assert result.target.timing.event_timestamp == ts("2026-01-01T10:04:00Z")
    assert result.target.timing.confirmation_timestamp == ts("2026-01-01T10:06:00Z")
    assert result.target.timing.availability_timestamp == ts("2026-01-01T10:06:17Z")
    assert result.availability_timestamp == fixture.subject.availability_timestamp


def test_future_invalidation_beyond_base_bar_cutoff_is_rejected():
    fixture, entry = valid_entry()
    future_source = source("future-stop", "2026-01-01T10:06:12Z")
    with pytest.raises(ValueError, match="base-bar"):
        invalidation = CoreV1StructuralInvalidation(
            fixture.subject,
            "future-stop",
            99.0,
            timing(
                "2026-01-01T10:06:12Z",
                (future_source,),
                base_scoped=False,
            ),
        )
        _, target = components(fixture, entry)
        evaluate_core_v1_hypothesis(entry, invalidation=invalidation, target=target)


def test_future_target_evidence_beyond_base_bar_cutoff_is_rejected():
    fixture, entry = valid_entry()
    future_level_source = source("future-level", "2026-01-01T10:06:12Z")
    with pytest.raises(ValueError, match="base-bar"):
        CoreV1LiquidityLevel(
            fixture.subject,
            "future-level",
            LiquiditySide.BSL,
            110.0,
            timing(
                "2026-01-01T10:06:12Z",
                (future_level_source,),
                base_scoped=False,
            ),
        )


def test_target_confirmation_past_subject_availability_is_rejected():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    late_confirmation = source(
        "late-target-confirmation",
        "2026-01-01T10:06:20Z",
        "2026-01-01T10:06:20Z",
        base=False,
    )
    with pytest.raises(ValueError, match="unavailable"):
        late_target = CoreV1OpposingLiquidityTarget(
            fixture.subject,
            "late-target",
            target.liquidity_level,
            timing(
                target.timing.event_timestamp.isoformat(),
                (*target.timing.source_observations, late_confirmation),
                confirmation="2026-01-01T10:06:20Z",
                available="2026-01-01T10:06:20Z",
            ),
        )
        evaluate_core_v1_hypothesis(entry, invalidation=invalidation, target=late_target)


def test_base_bar_cutoff_does_not_collapse_into_availability_cutoff():
    fixture, entry = valid_entry()
    assert entry.availability_timestamp > fixture.subject.base_bar_timestamp
    invalidation, target = components(fixture, entry)
    completed = complete(entry, invalidation=invalidation, target=target)
    assert completed.state is PrecisionState.VALID
    assert completed.availability_timestamp == fixture.subject.availability_timestamp


def test_exact_replay_subject_is_preserved_and_foreign_components_rejected():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    result = complete(entry, invalidation=invalidation, target=target)
    assert result.subject is fixture.subject

    other = ReplaySubject(
        fixture.subject.availability_timestamp,
        fixture.subject.base_bar_timestamp - pd.Timedelta(minutes=1),
    )
    foreign = replace(invalidation, subject=other)
    with pytest.raises(ValueError, match="another ReplaySubject"):
        complete(entry, invalidation=foreign, target=target)


def test_future_sentinel_cannot_become_a_target_retroactively():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    baseline = complete(entry, invalidation=invalidation, target=target)

    future_price_source = source(
        "future-excursion-sentinel",
        "2026-01-01T10:07:00Z",
        "2026-01-01T10:07:17Z",
        base=False,
    )
    future_timing = timing(
        "2026-01-01T10:04:00Z",
        (target.liquidity_level.timing.source_observations[0], future_price_source),
        confirmation="2026-01-01T10:07:00Z",
        available="2026-01-01T10:07:17Z",
    )
    with pytest.raises(ValueError, match="unavailable"):
        future_target = CoreV1OpposingLiquidityTarget(
            fixture.subject,
            "retrospective-target",
            target.liquidity_level,
            future_timing,
        )
        complete(
            entry,
            invalidation=invalidation,
            target=future_target,
        )
    assert baseline.target is target
    assert baseline.target.timing.event_timestamp == ts("2026-01-01T10:04:00Z")


def test_full_frame_is_rejected():
    fixture, _ = valid_entry()
    frame = pd.DataFrame({"Open": [100], "High": [101], "Low": [99], "Close": [100]})
    with pytest.raises(TypeError, match="CoreV1RetracementResult"):
        evaluate_core_v1_hypothesis(frame)  # type: ignore[arg-type]


def test_plain_rr_target_or_fixed_distance_stop_are_not_governed_inputs():
    fixture, entry = valid_entry()
    with pytest.raises(TypeError, match="CoreV1StructuralInvalidation"):
        evaluate_core_v1_hypothesis(
            entry,
            invalidation=99.0,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="CoreV1OpposingLiquidityTarget"):
        evaluate_core_v1_hypothesis(
            entry,
            target=110.0,  # type: ignore[arg-type]
        )


def test_hypothesis_has_no_risk_quantity_or_execution_fields():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    result = complete(entry, invalidation=invalidation, target=target)

    for name in (
        "quantity",
        "risk",
        "risk_budget",
        "price_risk",
        "execution_price",
        "order",
        "execution_intent",
        "fill",
        "position",
    ):
        assert not hasattr(result, name)


def test_hypothesis_module_does_not_import_execution_engine():
    import src.core_v1_hypothesis as hypothesis_module

    assert "tick_execution" not in hypothesis_module.__dict__
    assert not hasattr(hypothesis_module, "ExecutionEngine")
    assert not hasattr(hypothesis_module, "ExecutionIntent")


def test_complete_hypothesis_is_not_an_execution_order():
    fixture, entry = valid_entry()
    invalidation, target = components(fixture, entry)
    result = complete(entry, invalidation=invalidation, target=target)

    assert result.state is PrecisionState.VALID
    assert result.planned_entry_price != getattr(result, "execution_price", None)
    assert result.invalidation_price != getattr(result, "stop_order_price", None)
    assert result.target_price != getattr(result, "take_profit_order_price", None)
    assert not hasattr(result, "submitted")


def test_state_semantics_remain_with_existing_four_values():
    assert {state.value for state in PrecisionState} == {
        "VALID",
        "DEVELOPING",
        "INVALID",
        "NO_TRADE",
    }
    fixture, entry = valid_entry()
    unresolved = evaluate_core_v1_hypothesis(entry)
    invalidation, target = components(fixture, entry, invalidation_price=103.0)
    invalid = complete(entry, invalidation=invalidation, target=target)
    assert unresolved.state is PrecisionState.DEVELOPING
    assert invalid.state is PrecisionState.INVALID
    assert not hasattr(unresolved, "execution_state")
