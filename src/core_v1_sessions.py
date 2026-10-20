"""DST-aware Core v1 entry-window policy."""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


SESSION_TIMEZONE = "America/New_York"


def core_v1_entry_session(timestamp: object) -> str | None:
    """Return the eligible ICT session for an aware planned-entry event time.

    Windows are half-open New York local times. The user-selected one-hour
    pre-killzone extensions produce London 01:00-05:00 and New York 06:00-10:00.
    Invalid, naive, or unresolvable timestamps fail closed as ineligible.
    """
    try:
        point = pd.Timestamp(timestamp)
        if pd.isna(point) or point.tzinfo is None:
            return None
        local = point.tz_convert("UTC").to_pydatetime().astimezone(
            ZoneInfo(SESSION_TIMEZONE)
        )
    except (TypeError, ValueError, OverflowError, ZoneInfoNotFoundError):
        return None

    local_time = local.timetz().replace(tzinfo=None)
    if time(1, 0) <= local_time < time(5, 0):
        return "LONDON"
    if time(6, 0) <= local_time < time(10, 0):
        return "NEW_YORK"
    return None
