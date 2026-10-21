from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

from src.core_v1_pipeline import evaluate_core_v1_pipeline
from src.core_v1_liquidity import (
    detect_core_v1_liquidity_levels,
    evaluate_core_v1_liquidity,
)
from src.core_v1_mss import evaluate_core_v1_mss
from src.core_v1_retracement import FVGQuarterRejectionEntryResolver
from src.core_v1_evidence import (
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
)
from src.core_v1_dealing_range import DealingRangeContext
from src.replay_subject import ReplaySubject
from src.sniper_setup import PrecisionState
from tests.test_core_v1_pipeline import pipeline_fixture
from tests.test_core_v1_retracement import (
    make_source,
    make_timing,
    ts,
)


def _rebind_preserving_availability(value, subject, memo):
    if isinstance(value, ReplaySubject):
        return subject
    if isinstance(value, tuple):
        return tuple(
            _rebind_preserving_availability(item, subject, memo) for item in value
        )
    if isinstance(value, list):
        return [
            _rebind_preserving_availability(item, subject, memo) for item in value
        ]
    if is_dataclass(value):
        if id(value) in memo:
            return memo[id(value)]
        updates = {}
        for field in fields(value):
            current = getattr(value, field.name)
            rebound = _rebind_preserving_availability(current, subject, memo)
            if rebound is not current:
                updates[field.name] = rebound
        if isinstance(value, SubjectVisibleInputs) or (
            not isinstance(value, DealingRangeContext)
            and any(field.name == "subject" for field in fields(value))
            and any(field.name == "availability_timestamp" for field in fields(value))
        ):
            updates["availability_timestamp"] = subject.availability_timestamp
        result = replace(value, **updates)
        memo[id(value)] = result
        return result
    return value


def _cross_session_pipeline_case(kwargs, cutoff: str, later_window: bool):
    cutoff_timestamp = ts(cutoff)
    subject = ReplaySubject(
        availability_timestamp=cutoff_timestamp,
        base_bar_timestamp=cutoff_timestamp,
    )
    memo = {}
    inputs = _rebind_preserving_availability(kwargs["inputs"], subject, memo)
    inputs = replace(inputs, availability_timestamp=cutoff_timestamp)

    if later_window:
        bars = (
            ("inside-close", "2026-01-01T10:06:20Z", 102.5, 103.0, 102.0, 102.75),
            ("first-outside-window", "2026-01-01T10:06:30Z", 102.75, 103.25, 102.5, 103.1),
            ("later-in-window", "2026-01-01T11:06:30Z", 103.0, 103.3, 102.8, 103.2),
        )
    else:
        bars = (
            ("inside-close", "2026-01-01T10:06:20Z", 102.5, 103.0, 102.0, 102.75),
            ("first-outside-window", "2026-01-01T10:06:30Z", 102.75, 103.25, 102.5, 103.1),
        )

    extra_observations = []
    for source_id, timestamp, open_, high, low, close in bars:
        source = make_source(source_id, timestamp)
        extra_observations.append(
            SubjectVisibleObservation(
                value={"open": open_, "high": high, "low": low, "close": close},
                timing=make_timing(timestamp, (source,)),
                scope=ObservationScope.BASE,
            )
        )
    inputs = replace(
        inputs,
        base_observations=tuple(
            sorted(
                tuple(
                    replace(
                        item,
                        value={**item.value, "close": 102.5, "low": 100.5},
                    )
                    if item.timing.event_timestamp == ts("2026-01-01T10:06:10Z")
                    else item
                    for item in inputs.base_observations
                )
                + tuple(extra_observations),
                key=lambda item: item.timing.event_timestamp,
            )
        ),
    )

    rebound = {
        name: _rebind_preserving_availability(value, subject, memo)
        for name, value in (
            ("context", kwargs["context"]),
            ("draw_on_liquidity", kwargs["draw_on_liquidity"]),
            ("liquidity", kwargs["liquidity"]),
            ("displacement", kwargs["displacement"]),
            ("mss", kwargs["mss"]),
            ("zone", kwargs["zone"]),
            ("interaction", kwargs["interaction"]),
            ("invalidation", kwargs["invalidation"]),
            ("target", kwargs["target"]),
        )
    }
    # The pipeline fixture supplies synthetic, already-governed invalidation and
    # target evidence so this regression isolates cross-session E009 continuity;
    # no market or execution outcome is part of the fixture.
    visible_levels = detect_core_v1_liquidity_levels(inputs)
    liquidity_level = next(
        level
        for level in visible_levels
        if level.level_id == rebound["liquidity"].level.level_id
    )
    rebound["liquidity"] = evaluate_core_v1_liquidity(
        inputs,
        levels=(liquidity_level,),
    )[0]
    rebound["draw_on_liquidity"] = next(
        level
        for level in visible_levels
        if level.level_id == rebound["draw_on_liquidity"].level_id
    )
    rebound["mss"] = evaluate_core_v1_mss(
        inputs,
        structural_point=rebound["mss"].structural_point,
        liquidity_event=rebound["liquidity"],
        displacement=rebound["displacement"],
        structural_break=rebound["mss"].structural_break,
        confirmation=rebound["mss"].confirmation,
    )
    rebound["opposing_liquidity_levels"] = (rebound["target"].liquidity_level,)
    return {
        "inputs": inputs,
        **rebound,
        "entry_resolver": FVGQuarterRejectionEntryResolver(),
    }


def test_e009_cross_session_rejection_continuity_through_canonical_pipeline():
    _, kwargs = pipeline_fixture()

    early = evaluate_core_v1_pipeline(
        **_cross_session_pipeline_case(
            kwargs,
            "2026-01-01T10:06:30Z",
            later_window=False,
        )
    )

    assert early.state is PrecisionState.DEVELOPING
    assert early.retracement is not None
    assert early.retracement.planned_entry is None
    assert early.hypothesis is None
    assert early.availability_timestamp == ts("2026-01-01T10:06:30Z")

    later = evaluate_core_v1_pipeline(
        **_cross_session_pipeline_case(
            kwargs,
            "2026-01-01T11:06:30Z",
            later_window=True,
        )
    )

    assert later.state is PrecisionState.VALID
    assert later.hypothesis is not None
    assert later.retracement is not None
    assert later.retracement.planned_entry is not None
    assert (
        later.retracement.planned_entry.timing.event_timestamp
        == ts("2026-01-01T11:06:30Z")
    )
    assert (
        later.retracement.planned_entry.timing.availability_timestamp
        <= later.availability_timestamp
    )
    planned_sources = later.retracement.planned_entry.timing.source_observations
    assert any(source.source_id == "first-outside-window" for source in planned_sources)
    assert any(source.source_id == "later-in-window" for source in planned_sources)
    assert later.provenance["planned_entry"] == planned_sources
    assert later.event_timestamps["planned_entry"] == ts("2026-01-01T11:06:30Z")
