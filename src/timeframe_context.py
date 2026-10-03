from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .data_contract import normalize_ohlc


_TIMEFRAME_OFFSETS = {
    "1M": "1min",
    "3M": "3min",
    "5M": "5min",
    "15M": "15min",
    "M1": "1min",
    "M3": "3min",
    "M5": "5min",
    "M15": "15min",
    "1H": "1h",
    "4H": "4h",
    "H1": "1h",
    "H4": "4h",
    "1D": "1D",
    "D1": "1D",
    "1W": "1W",
    "W1": "1W",
}


def _frame_availability_column(frame: pd.DataFrame) -> str | None:
    return next(
        (column for column in ("available_at", "availability_ts") if column in frame.columns),
        None,
    )


def _event_time_visibility_mask(
    frame: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Select complete event-time rows visible at an inclusive UTC boundary."""
    availability_column = _frame_availability_column(frame)
    if availability_column is None:
        raise ValueError("Event-time availability metadata is required.")
    available_at = pd.to_datetime(frame[availability_column], utc=True, errors="raise")
    return _complete_mask(frame) & available_at.notna() & (available_at <= as_of)


def _complete_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in ("historical_complete", "is_complete"):
        if column in frame.columns:
            mask &= frame[column].fillna(False).astype(bool)
    return mask


def _nominal_close_times(
    frame: pd.DataFrame,
    timeframe: str,
) -> pd.DatetimeIndex:
    if str(frame.attrs.get("bar_label", "left")).lower() == "right":
        return pd.DatetimeIndex(frame.index)
    rule = _TIMEFRAME_OFFSETS.get(str(timeframe).upper(), timeframe)
    offset = pd.tseries.frequencies.to_offset(rule)
    return pd.DatetimeIndex(frame.index + offset)


def eligible_decision_times(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Return unique decision times under the replay's established eligibility rules."""
    complete = _complete_mask(frame)
    if "available_at" in frame.columns:
        eligible = complete & frame["available_at"].notna()
        timestamps = pd.DatetimeIndex(
            frame.loc[eligible, "available_at"].drop_duplicates()
        ).sort_values()
    elif "availability_ts" in frame.columns:
        eligible = complete & frame["availability_ts"].notna()
        timestamps = pd.DatetimeIndex(
            frame.loc[eligible, "availability_ts"].drop_duplicates()
        ).sort_values()
    else:
        timestamps = _nominal_close_times(frame, timeframe)[complete.to_numpy()]

    if start is not None:
        timestamps = timestamps[timestamps >= start]
    if end is not None:
        timestamps = timestamps[timestamps <= end]
    return timestamps


def bars_available_as_of(
    frame: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    timeframe: str,
) -> pd.DataFrame:
    """Filter bars by event availability, or explicit nominal-close fallback."""
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.tz_convert("UTC")

    has_timing_metadata = any(
        column in frame.columns
        for column in ("available_at", "availability_ts", "interval_end")
    ) or frame.attrs.get("availability_mode") == "event_time"
    if has_timing_metadata:
        if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("Event-time OHLC labels must be unique and ordered datetimes.")
        normalized = normalize_ohlc(
            frame,
            timeframe=timeframe,
            source=str(frame.attrs.get("source", "UNKNOWN")),
        )
        frame = frame.copy()
        frame.index = normalized.index
        for column in (
            "available_at",
            "availability_ts",
            "interval_end",
            "historical_complete",
            "is_complete",
        ):
            if column in normalized.columns:
                frame[column] = normalized[column].array
        frame.attrs.update(normalized.attrs)

    availability_column = _frame_availability_column(frame)
    availability_mode = frame.attrs.get("availability_mode")
    if availability_mode not in {None, "event_time", "nominal_close_fallback"}:
        raise ValueError("Unsupported availability_mode.")
    if availability_mode == "event_time" and availability_column is None:
        raise ValueError("Event-time availability metadata is required.")
    if availability_mode == "nominal_close_fallback" and availability_column is not None:
        raise ValueError("Nominal-close fallback cannot include event-time availability metadata.")

    event_time_input = availability_column is not None
    if event_time_input:
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("Event-time OHLC labels must be timezone-aware datetimes.")
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("Event-time OHLC labels must be unique and ordered.")

    if event_time_input or "interval_end" in frame.columns:
        for column in ("available_at", "availability_ts", "interval_end"):
            if column not in frame.columns:
                continue
            for value in frame[column].dropna():
                timestamp = pd.Timestamp(value)
                if timestamp.tzinfo is None:
                    raise ValueError(f"{column} timestamps must be timezone-aware.")

    if "available_at" in frame.columns and "availability_ts" in frame.columns:
        available_at_alias = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
        availability_ts_alias = pd.to_datetime(frame["availability_ts"], utc=True, errors="raise")
        aliases_match = available_at_alias.eq(availability_ts_alias) | (
            available_at_alias.isna() & availability_ts_alias.isna()
        )
        if not aliases_match.all():
            raise ValueError("available_at and availability_ts must match when both are provided.")

    complete = _complete_mask(frame)
    if availability_column is not None:
        available_at = pd.to_datetime(frame[availability_column], utc=True, errors="raise")
        labels = frame.index.tz_convert("UTC")
        if (available_at.notna() & (available_at < labels)).any():
            raise ValueError("Bar availability cannot precede its timestamp label.")

        if "interval_end" in frame.columns:
            interval_end = pd.to_datetime(frame["interval_end"], utc=True, errors="raise")
            if not interval_end.dropna().is_monotonic_increasing:
                raise ValueError("Bar interval ends must be non-decreasing.")
            if (interval_end.notna() & (interval_end < labels)).any():
                raise ValueError("Bar interval end cannot precede its timestamp label.")
            if (available_at.notna() & interval_end.notna() & (available_at < interval_end)).any():
                raise ValueError("Bar availability cannot precede its interval end.")
            missing_interval_end = interval_end.isna() & available_at.notna()
        else:
            missing_interval_end = available_at.notna()

        if missing_interval_end.any():
            nominal_close = _nominal_close_times(frame, timeframe)
            if (missing_interval_end & (available_at < nominal_close)).any():
                raise ValueError("Bar availability cannot precede its nominal close.")
        if not available_at.dropna().is_monotonic_increasing:
            raise ValueError("Bar availability timestamps must be non-decreasing.")
        mask = _event_time_visibility_mask(frame, as_of)
    else:
        close_times = _nominal_close_times(frame, timeframe)
        mask = complete & (close_times <= as_of)
    return frame.loc[mask].copy()


@dataclass(frozen=True)
class TimeframeContext:
    """
    Timeframe views used by the replay engine.

    Each dataframe is independently normalized to UTC. Resampling is only
    performed from a lower-timeframe source and is anchored to UTC. The replay
    engine only exposes completed higher-timeframe bars as of a decision time.
    """

    frames: Mapping[str, pd.DataFrame]

    def available_as_of(self, timeframe: str, as_of: pd.Timestamp) -> pd.DataFrame:
        if timeframe not in self.frames:
            raise KeyError(f"Unknown timeframe: {timeframe}")

        frame = self.frames[timeframe]
        frame_timeframe = str(frame.attrs.get("timeframe", timeframe))
        return bars_available_as_of(frame, as_of, timeframe=frame_timeframe)


def build_context(
    base: pd.DataFrame,
    *,
    base_timeframe: str = "5M",
    source: str = "UNKNOWN",
) -> TimeframeContext:
    """
    Build a deterministic context set from one normalized base frame.

    This intentionally keeps resampling conservative and explicit. The caller
    can replace these generated frames with feed-native 1H/15M/3M/1M data when
    available. No synthetic tick/order information is created.
    """
    base = normalize_ohlc(
        base,
        timeframe=base_timeframe,
        source=source,
    )

    rules = {
        "15M": "15min",
        "1H": "1h",
        "4H": "4h",
        "1D": "1D",
    }

    frames: dict[str, pd.DataFrame] = {base_timeframe: base}

    event_time_input = "available_at" in base.columns
    context_source = base
    if event_time_input:
        eligible = _complete_mask(base) & base["available_at"].notna()
        context_source = base.loc[eligible]
    else:
        context_source = base.loc[_complete_mask(base)]

    resample_source = context_source
    if event_time_input and "interval_end" in context_source.columns:
        resample_source = context_source.set_index("interval_end", drop=False)
    elif str(base.attrs.get("bar_label", "left")).lower() != "right":
        base_rule = _TIMEFRAME_OFFSETS.get(str(base_timeframe).upper(), base_timeframe)
        base_offset = pd.tseries.frequencies.to_offset(base_rule)
        resample_source = context_source.copy()
        resample_source.index = resample_source.index + base_offset

    for label, rule in rules.items():
        if label == base_timeframe:
            continue

        resampled = (
            resample_source[["Open", "High", "Low", "Close"]]
            .resample(rule, label="right", closed="right")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                }
            )
            .dropna()
        )
        if event_time_input:
            base_rule = _TIMEFRAME_OFFSETS.get(str(base_timeframe).upper(), base_timeframe)
            base_offset = pd.tseries.frequencies.to_offset(base_rule)
            target_offset = pd.tseries.frequencies.to_offset(rule)
            try:
                base_nanos = base_offset.nanos
                target_nanos = target_offset.nanos
            except ValueError:
                expected_count = None
            else:
                expected_count = (
                    target_nanos // base_nanos
                    if target_nanos >= base_nanos and target_nanos % base_nanos == 0
                    else None
                )

            if expected_count is None:
                resampled = resampled.iloc[0:0].copy()
            else:
                source_labels = pd.DatetimeIndex(resample_source.index)
                complete_labels = []
                for interval_end in resampled.index:
                    interval_start = interval_end - target_offset
                    left = source_labels.searchsorted(interval_start, side="right")
                    right = source_labels.searchsorted(interval_end, side="right")
                    actual_labels = source_labels[left:right]
                    expected_labels = pd.date_range(
                        end=interval_end,
                        periods=expected_count,
                        freq=base_offset,
                    )
                    if actual_labels.tolist() == expected_labels.tolist():
                        complete_labels.append(interval_end)
                resampled = resampled.loc[complete_labels].copy()

            availability = resample_source["available_at"].resample(
                rule, label="right", closed="right"
            ).max()
            availability = availability.reindex(resampled.index)
            availability = availability.where(availability >= resampled.index)
            resampled["available_at"] = availability
            resampled["availability_ts"] = availability
            resampled["historical_complete"] = availability.notna()
            resampled["is_complete"] = availability.notna()
        else:
            base_rule = _TIMEFRAME_OFFSETS.get(str(base_timeframe).upper(), base_timeframe)
            base_offset = pd.tseries.frequencies.to_offset(base_rule)
            target_offset = pd.tseries.frequencies.to_offset(rule)
            try:
                base_nanos = base_offset.nanos
                target_nanos = target_offset.nanos
            except ValueError:
                expected_count = None
            else:
                expected_count = (
                    target_nanos // base_nanos
                    if target_nanos >= base_nanos and target_nanos % base_nanos == 0
                    else None
                )

            if expected_count is None:
                resampled = resampled.iloc[0:0].copy()
            else:
                source_labels = pd.DatetimeIndex(resample_source.index)
                complete_labels = []
                for interval_end in resampled.index:
                    interval_start = interval_end - target_offset
                    left = source_labels.searchsorted(interval_start, side="right")
                    right = source_labels.searchsorted(interval_end, side="right")
                    actual_labels = source_labels[left:right]
                    expected_labels = pd.date_range(
                        end=interval_end,
                        periods=expected_count,
                        freq=base_offset,
                    )
                    if actual_labels.tolist() == expected_labels.tolist():
                        complete_labels.append(interval_end)
                resampled = resampled.loc[complete_labels].copy()
        resampled.attrs["timeframe"] = label
        resampled.attrs["source"] = source
        resampled.attrs["timezone"] = "UTC"
        resampled.attrs["availability_mode"] = (
            "event_time" if event_time_input else "nominal_close_fallback"
        )
        resampled.attrs["bar_label"] = "right"
        frames[label] = resampled

    return TimeframeContext(frames=frames)
