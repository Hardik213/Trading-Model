from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import src.core_v1_sessions as sessions
from src.core_v1_sessions import (
    core_v1_candidate_expiry,
    core_v1_entry_session,
    core_v1_entry_windows_after,
)


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2026-01-15T06:00:00Z", "LONDON"),  # 01:00 EST
        ("2026-01-15T09:59:59Z", "LONDON"),  # 04:59:59 EST
        ("2026-01-15T11:00:00Z", "NEW_YORK"),  # 06:00 EST
        ("2026-01-15T14:59:59Z", "NEW_YORK"),  # 09:59:59 EST
        ("2026-01-15T10:00:00Z", None),  # 05:00 EST
        ("2026-01-15T15:00:00Z", None),  # 10:00 EST
        ("2026-01-15T05:59:59Z", None),
        ("2026-01-15T15:00:01Z", None),
    ],
)
def test_entry_windows_are_half_open_and_fail_closed_outside(timestamp, expected):
    assert core_v1_entry_session(ts(timestamp)) == expected


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2026-07-15T05:00:00Z", "LONDON"),  # 01:00 EDT
        ("2026-07-15T08:59:59Z", "LONDON"),  # 04:59:59 EDT
        ("2026-07-15T10:00:00Z", "NEW_YORK"),  # 06:00 EDT
        ("2026-07-15T13:59:59Z", "NEW_YORK"),  # 09:59:59 EDT
        ("2026-07-15T09:00:00Z", None),  # 05:00 EDT
        ("2026-07-15T14:00:00Z", None),  # 10:00 EDT
    ],
)
def test_entry_windows_are_dst_aware(timestamp, expected):
    assert core_v1_entry_session(ts(timestamp)) == expected


def test_fall_back_repeated_new_york_local_time_is_deterministic_and_eligible():
    # Both UTC instants map to the repeated 01:30 America/New_York local time.
    assert core_v1_entry_session(ts("2026-11-01T05:30:00Z")) == "LONDON"
    assert core_v1_entry_session(ts("2026-11-01T06:30:00Z")) == "LONDON"


def test_spring_forward_skipped_local_time_is_not_fabricated():
    # 02:30 local does not exist on the DST transition; UTC conversion jumps
    # from 01:59:59 EST to 03:00:00 EDT. The first valid post-jump instant is
    # still inside the London window.
    assert core_v1_entry_session(ts("2026-03-08T06:59:59Z")) == "LONDON"
    assert core_v1_entry_session(ts("2026-03-08T07:00:00Z")) == "LONDON"
    assert core_v1_entry_session(datetime(2026, 3, 8, 2, 30)) is None


@pytest.mark.parametrize("timestamp", [None, "not-a-timestamp", "2026-01-15T06:00:00"])
def test_naive_or_invalid_timestamps_fail_closed(timestamp):
    assert core_v1_entry_session(timestamp) is None


def test_missing_timezone_database_fails_closed(monkeypatch):
    def unavailable(_):
        raise sessions.ZoneInfoNotFoundError("America/New_York")

    monkeypatch.setattr(sessions, "ZoneInfo", unavailable)

    assert core_v1_entry_session(ts("2026-01-15T06:00:00Z")) is None


def test_three_window_lifetime_counts_containing_window_then_two_later_windows():
    windows = core_v1_entry_windows_after(ts("2026-01-01T06:30:00Z"))
    assert [window.session for window in windows] == ["LONDON", "NEW_YORK", "LONDON"]
    assert core_v1_candidate_expiry(ts("2026-01-01T06:30:00Z")) == ts(
        "2026-01-02T10:00:00Z"
    )


def test_three_window_lifetime_starts_with_next_window_when_interaction_is_outside():
    windows = core_v1_entry_windows_after(ts("2026-01-01T10:30:00Z"))
    assert [window.session for window in windows] == ["NEW_YORK", "LONDON", "NEW_YORK"]
    assert core_v1_candidate_expiry(ts("2026-01-01T10:30:00Z")) == ts(
        "2026-01-02T15:00:00Z"
    )


def test_three_window_expiry_uses_dst_aware_local_boundaries():
    assert core_v1_candidate_expiry(ts("2026-03-08T06:30:00Z")) == ts(
        "2026-03-09T09:00:00Z"
    )
    assert core_v1_candidate_expiry(ts("2026-07-05T09:30:00Z")) == ts(
        "2026-07-06T14:00:00Z"
    )
