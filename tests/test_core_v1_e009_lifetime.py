from __future__ import annotations

from dataclasses import replace

from src.core_v1_retracement import (
    FVGQuarterRejectionEntryResolver,
    derive_visible_fvg_interaction,
    evaluate_core_v1_retracement,
)
from src.core_v1_sessions import core_v1_candidate_expiry
from src.core_v1_evidence import ObservationScope, SubjectVisibleObservation
from src.replay_subject import ReplaySubject
from tests.test_core_v1_cross_session_pipeline import _rebind_preserving_availability
from tests.test_core_v1_retracement import make_source, make_timing, make_fixture, ts
from tests.test_tick_execution import decision, execute, quote


def _e009_case(
    interaction_time: str,
    bars: tuple[tuple[str, str, float], ...],
    cutoff: str | None = None,
):
    fixture = make_fixture()
    last_time = max(
        (ts(event) for _, event, _ in bars),
        default=ts(interaction_time),
    )
    cutoff_timestamp = ts(cutoff) if cutoff is not None else last_time
    subject = ReplaySubject(
        availability_timestamp=cutoff_timestamp,
        base_bar_timestamp=cutoff_timestamp,
    )
    memo = {}
    inputs = _rebind_preserving_availability(fixture.inputs, subject, memo)
    zone = _rebind_preserving_availability(fixture.zone, subject, memo)
    mss = _rebind_preserving_availability(fixture.mss, subject, memo)
    liquidity = _rebind_preserving_availability(fixture.liquidity, subject, memo)
    displacement = _rebind_preserving_availability(fixture.displacement, subject, memo)

    base = tuple(
        item
        for item in inputs.base_observations
        if item.timing.event_timestamp <= mss.confirmation_timestamp
    )
    additions = []
    for source_id, event, close in bars:
        event_timestamp = ts(event)
        if event_timestamp > cutoff_timestamp:
            continue
        source = make_source(source_id, event)
        additions.append(
            SubjectVisibleObservation(
                value={
                    "open": close,
                    "high": max(close, 104.0),
                    "low": min(close, 100.0),
                    "close": close,
                },
                timing=make_timing(event, (source,)),
                scope=ObservationScope.BASE,
            )
        )
    inputs = replace(
        inputs,
        base_observations=tuple(
            sorted(base + tuple(additions), key=lambda item: item.timing.event_timestamp)
        ),
        availability_timestamp=cutoff_timestamp,
    )
    interaction = derive_visible_fvg_interaction(inputs, zone, mss)
    assert interaction is not None
    result = evaluate_core_v1_retracement(
        inputs,
        mss=mss,
        liquidity=liquidity,
        displacement=displacement,
        zone=zone,
        interaction=interaction,
        resolver=FVGQuarterRejectionEntryResolver(),
    )
    return result, interaction, core_v1_candidate_expiry(interaction.interaction_timestamp)


def test_interaction_inside_window_counts_window_one_and_third_window_can_qualify():
    result, interaction, expiry = _e009_case(
        "2026-01-01T11:06:10Z",
        (
            ("interaction", "2026-01-01T11:06:10Z", 102.0),
            ("third-window-rejection", "2026-01-02T14:59:59Z", 103.5),
        ),
    )
    assert interaction.interaction_timestamp == ts("2026-01-01T11:06:10Z")
    assert expiry == ts("2026-01-02T15:00:00Z")
    assert result.planned_entry is not None
    assert result.planned_entry.timing.event_timestamp == ts("2026-01-02T14:59:59Z")


def test_interaction_outside_windows_starts_with_next_window():
    result, interaction, expiry = _e009_case(
        "2026-01-01T10:06:10Z",
        (
            ("interaction", "2026-01-01T10:06:10Z", 102.0),
            ("first-window-rejection", "2026-01-01T11:06:10Z", 103.5),
        ),
    )
    assert interaction.interaction_timestamp == ts("2026-01-01T10:06:10Z")
    assert expiry == ts("2026-01-02T15:00:00Z")
    assert result.planned_entry is not None


def test_out_of_window_rejection_does_not_consume_candidate_or_window():
    result, _, _ = _e009_case(
        "2026-01-01T10:06:10Z",
        (
            ("interaction", "2026-01-01T10:06:10Z", 102.0),
            ("outside-rejection", "2026-01-01T10:30:00Z", 103.5),
            ("later-in-window-rejection", "2026-01-01T11:06:10Z", 103.5),
        ),
    )
    assert result.planned_entry is not None
    assert result.planned_entry.timing.event_timestamp == ts("2026-01-01T11:06:10Z")
    assert any(
        source.source_id == "outside-rejection"
        for source in result.planned_entry.timing.source_observations
    )


def test_candidate_expires_at_exclusive_third_window_end():
    result, _, expiry = _e009_case(
        "2026-01-01T11:06:10Z",
        (
            ("interaction", "2026-01-01T11:06:10Z", 102.0),
            ("boundary-rejection", "2026-01-02T15:00:00Z", 103.5),
        ),
        cutoff="2026-01-02T15:00:00Z",
    )
    assert expiry == ts("2026-01-02T15:00:00Z")
    assert result.planned_entry is None


def test_adverse_quarter_invalidation_ends_candidate_before_expiry():
    result, _, expiry = _e009_case(
        "2026-01-01T11:06:10Z",
        (
            ("interaction", "2026-01-01T11:06:10Z", 102.0),
            ("adverse-quarter", "2026-01-01T11:30:00Z", 101.4),
            ("later-rejection", "2026-01-01T12:06:10Z", 103.5),
        ),
    )
    assert expiry > ts("2026-01-01T11:30:00Z")
    assert result.planned_entry is None


def test_earlier_cutoff_cannot_use_later_bar_or_later_window_count():
    result, _, _ = _e009_case(
        "2026-01-01T10:06:10Z",
        (
            ("interaction", "2026-01-01T10:06:10Z", 102.0),
            ("later-window-rejection", "2026-01-02T11:06:10Z", 103.5),
        ),
        cutoff="2026-01-01T10:06:17Z",
    )
    assert result.planned_entry is None


def test_expiry_does_not_change_an_already_open_position_tp_sl_lifecycle():
    _, results = execute(
        [
            quote("2026-01-01T11:00:00Z", 100.0, 101.0, 0),
            quote("2026-01-01T11:00:01Z", 100.0, 101.0, 1),
            quote("2026-01-02T15:00:01Z", 110.0, 111.0, 2),
        ],
        [
            decision(
                "2026-01-01T11:00:00Z",
                stop=90.0,
                target=110.0,
            )
        ],
    )
    assert results[0].exit_fill is not None
    assert results[0].reason == "TAKE_PROFIT"
