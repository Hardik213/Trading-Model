"""DST-aware Core v1 entry-window policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


SESSION_TIMEZONE = "America/New_York"
ENTRY_WINDOW_COUNT = 3


@dataclass(frozen=True)
class CoreV1EntryWindow:
    """One DST-aware half-open entry window represented in UTC."""

    session: str
    start: pd.Timestamp
    end: pd.Timestamp


def _as_utc_timestamp(timestamp: object) -> pd.Timestamp | None:
    try:
        point = pd.Timestamp(timestamp)
        if pd.isna(point) or point.tzinfo is None:
            return None
        return point.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        return None


def _local_window(local_date: date, session: str, start: time, end: time) -> CoreV1EntryWindow:
    zone = ZoneInfo(SESSION_TIMEZONE)
    start_local = datetime.combine(local_date, start, tzinfo=zone)
    end_local = datetime.combine(local_date, end, tzinfo=zone)
    return CoreV1EntryWindow(
        session=session,
        start=pd.Timestamp(start_local.astimezone(timezone.utc)),
        end=pd.Timestamp(end_local.astimezone(timezone.utc)),
    )


def core_v1_entry_windows_after(
    timestamp: object,
    *,
    count: int = ENTRY_WINDOW_COUNT,
) -> tuple[CoreV1EntryWindow, ...]:
    """Return the next eligible windows, including the containing window.

    A timestamp inside a window includes that window. A timestamp outside all
    windows starts with the first later window. The result is derived only from
    the timestamp and the local DST-aware session policy.
    """
    point = _as_utc_timestamp(timestamp)
    if point is None or type(count) is not int or count <= 0:
        return ()
    try:
        local_date = point.tz_convert(SESSION_TIMEZONE).date()
        candidates: list[CoreV1EntryWindow] = []
        for offset in range(count + 2):
            current = local_date + timedelta(days=offset)
            candidates.extend(
                (
                    _local_window(current, "LONDON", time(1, 0), time(5, 0)),
                    _local_window(current, "NEW_YORK", time(6, 0), time(10, 0)),
                )
            )
        return tuple(window for window in candidates if window.end > point)[:count]
    except ZoneInfoNotFoundError:
        return ()


def core_v1_candidate_expiry(
    interaction_timestamp: object,
    *,
    count: int = ENTRY_WINDOW_COUNT,
) -> pd.Timestamp | None:
    """Return the exclusive expiry boundary for an unfilled candidate."""
    windows = core_v1_entry_windows_after(interaction_timestamp, count=count)
    return windows[-1].end if len(windows) == count else None


def core_v1_entry_session(timestamp: object) -> str | None:
    """Return the eligible ICT session for an aware planned-entry event time.

    Windows are half-open New York local times. The user-selected one-hour
    pre-killzone extensions produce London 01:00-05:00 and New York 06:00-10:00.
    Invalid, naive, or unresolvable timestamps fail closed as ineligible.
    """
    point = _as_utc_timestamp(timestamp)
    if point is None:
        return None
    try:
        local = point.to_pydatetime().astimezone(ZoneInfo(SESSION_TIMEZONE))
    except ZoneInfoNotFoundError:
        return None

    local_time = local.timetz().replace(tzinfo=None)
    if time(1, 0) <= local_time < time(5, 0):
        return "LONDON"
    if time(6, 0) <= local_time < time(10, 0):
        return "NEW_YORK"
    return None
