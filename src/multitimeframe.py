from __future__ import annotations

import numpy as np
import pandas as pd

from .data_contract import normalize_ohlc


_BASE_INTERVALS = {
    "m": "1min",
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "h": "1h",
    "1h": "1h",
    "4h": "4h",
    "d": "1D",
    "1d": "1D",
    "w": "1W",
    "1w": "1W",
}


def _asof_group_values(
    query_times: pd.DatetimeIndex,
    group_ends: pd.DatetimeIndex,
    group_available: pd.DatetimeIndex,
    group_valid: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    output = np.full(len(query_times), np.nan, dtype=float)
    output_available = [pd.NaT] * len(query_times)
    for row_position, query_time in enumerate(query_times):
        if pd.isna(query_time):
            continue
        latest_period = group_ends.searchsorted(query_time, side="right") - 1
        if latest_period < 0 or not group_valid[latest_period]:
            continue

        available = (
            group_valid[:latest_period + 1]
            & ~group_available[:latest_period + 1].isna()
            & (group_available[:latest_period + 1] <= query_time)
        )
        candidates = np.flatnonzero(available)
        if not len(candidates):
            continue
        selected = int(candidates[-1])
        if not group_valid[selected + 1:latest_period + 1].all():
            continue
        if not np.isfinite(values[selected]):
            continue
        output[row_position] = values[selected]
        output_available[row_position] = group_available[selected]
    return output, pd.DatetimeIndex(output_available)


def _max_availability(
    first: pd.DatetimeIndex,
    second: pd.DatetimeIndex,
    valid: np.ndarray,
) -> pd.DatetimeIndex:
    values = [
        max(first[position], second[position])
        if valid[position] and not pd.isna(first[position]) and not pd.isna(second[position])
        else pd.NaT
        for position in range(len(valid))
    ]
    return pd.DatetimeIndex(values)


def _rolling_availability(
    values: pd.Series,
    availability: pd.DatetimeIndex,
    window: int,
) -> pd.DatetimeIndex:
    output = [pd.NaT] * len(values)
    for position, value in enumerate(values.to_numpy(dtype=float)):
        start = position - window + 1
        if start < 0 or not np.isfinite(value):
            continue
        window_availability = availability[start:position + 1]
        if window_availability.isna().any():
            continue
        output[position] = window_availability.max()
    return pd.DatetimeIndex(output)


def _completed_higher_close(
    data: pd.DataFrame,
    *,
    base_interval: pd.tseries.offsets.BaseOffset,
    higher_interval: str,
    close_labels: pd.DatetimeIndex,
    row_available: pd.DatetimeIndex,
    row_complete: np.ndarray,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    higher_offset = pd.tseries.frequencies.to_offset(higher_interval)
    try:
        base_nanos = base_interval.nanos
        higher_nanos = higher_offset.nanos
    except ValueError:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([]), np.array([], dtype=bool), np.array([], dtype=float)

    if higher_nanos < base_nanos or higher_nanos % base_nanos:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([]), np.array([], dtype=bool), np.array([], dtype=float)

    expected_count = higher_nanos // base_nanos
    if not len(close_labels):
        return pd.DatetimeIndex([]), pd.DatetimeIndex([]), np.array([], dtype=bool), np.array([], dtype=float)

    first_end = close_labels.min().ceil(higher_offset)
    last_end = close_labels.max().ceil(higher_offset)
    group_ends = pd.date_range(first_end, last_end, freq=higher_offset)
    eligible_positions = np.flatnonzero(row_complete & ~row_available.isna())
    eligible_labels = close_labels[eligible_positions]
    if eligible_labels.has_duplicates or not eligible_labels.is_monotonic_increasing:
        raise ValueError("Base bar close timestamps must be unique and ordered.")

    closes = data["Close"].to_numpy(dtype=float)
    group_valid = np.zeros(len(group_ends), dtype=bool)
    group_available_values = [pd.NaT] * len(group_ends)
    group_close_values = np.full(len(group_ends), np.nan, dtype=float)

    for group_position, group_end in enumerate(group_ends):
        expected_labels = pd.date_range(
            end=group_end,
            periods=expected_count,
            freq=base_interval,
        )
        positions = eligible_labels.get_indexer(expected_labels)
        if (positions < 0).any():
            continue
        source_positions = eligible_positions[positions]
        group_valid[group_position] = True
        group_available_values[group_position] = row_available[source_positions].max()
        group_close_values[group_position] = closes[source_positions[-1]]

    group_available = pd.DatetimeIndex(group_available_values)
    return group_ends, group_available, group_valid, group_close_values


def add_multitimeframe_features(df: pd.DataFrame, base: str = "h") -> pd.DataFrame:
    data = df.copy()
    if "Close" not in data.columns:
        raise ValueError("OHLC data must include a Close column.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("OHLC index must be a DatetimeIndex.")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("OHLC index must be unique and ordered.")

    bar_label = str(data.attrs.get("bar_label", "left")).lower()
    if bar_label not in {"left", "right"}:
        raise ValueError("bar_label must be either 'left' or 'right'.")
    base_rule = _BASE_INTERVALS.get(str(base).lower(), base)
    base_offset = pd.tseries.frequencies.to_offset(base_rule)
    data.attrs["timeframe"] = str(base_rule)
    data.attrs["bar_label"] = bar_label

    has_event_metadata = any(
        column in data.columns
        for column in ("available_at", "availability_ts", "interval_end")
    ) or data.attrs.get("availability_mode") == "event_time"
    data.attrs["mtf_availability_mode"] = (
        "event_time" if has_event_metadata else "nominal_close_fallback"
    )
    data.attrs["availability_mode"] = (
        "event_time" if has_event_metadata else "nominal_close_fallback"
    )
    if has_event_metadata:
        timing = normalize_ohlc(
            data,
            timeframe=str(base),
            source=str(data.attrs.get("source", "UNKNOWN")),
        )
        nominal_close = pd.DatetimeIndex(
            timing.index if bar_label == "right" else timing.index + base_offset
        )
        if "interval_end" in timing.columns:
            close_labels = pd.DatetimeIndex(
                timing["interval_end"].where(
                    timing["interval_end"].notna(), nominal_close
                )
            )
        else:
            close_labels = nominal_close
        if "available_at" in timing.columns:
            row_available = pd.DatetimeIndex(timing["available_at"])
        elif "interval_end" in timing.columns:
            row_available = pd.DatetimeIndex(
                timing["interval_end"].where(
                    timing["interval_end"].notna(), nominal_close
                )
            )
        else:
            row_available = nominal_close
        row_complete = np.ones(len(data), dtype=bool)
        for column in ("historical_complete", "is_complete"):
            if column in timing.columns:
                row_complete &= timing[column].fillna(False).to_numpy(dtype=bool)
    else:
        close_labels = pd.DatetimeIndex(
            data.index if bar_label == "right" else data.index + base_offset
        )
        row_available = close_labels
        row_complete = np.ones(len(data), dtype=bool)
        for column in ("historical_complete", "is_complete"):
            if column in data.columns:
                row_complete &= data[column].fillna(False).to_numpy(dtype=bool)

    if len(data) and data.attrs.get("market_source") == "live_mt5":
        has_completeness = any(
            column in data.columns
            for column in ("historical_complete", "is_complete")
        )
        has_timing = any(
            column in data.columns
            for column in ("available_at", "availability_ts", "interval_end")
        )
        if not has_completeness or not has_timing:
            row_complete[-1] = False

    if close_labels.has_duplicates or not close_labels.is_monotonic_increasing:
        raise ValueError("Base bar close timestamps must be unique and ordered.")

    base_query_times = pd.DatetimeIndex(
        [row_available[position] if row_complete[position] else pd.NaT for position in range(len(data))]
    )
    data["mtf_decision_time"] = pd.Series(base_query_times, index=data.index)

    h4_ends, h4_available, h4_valid, h4_close = _completed_higher_close(
        data,
        base_interval=base_offset,
        higher_interval="4h",
        close_labels=close_labels,
        row_available=row_available,
        row_complete=row_complete,
    )
    h4_previous = np.roll(h4_close, 1)
    h4_trend = np.full(len(h4_close), np.nan, dtype=float)
    if len(h4_close) > 1:
        valid_trends = h4_valid[1:] & h4_valid[:-1] & (h4_previous[1:] != 0)
        h4_trend[1:][valid_trends] = (
            h4_close[1:][valid_trends] / h4_previous[1:][valid_trends] - 1.0
        )
    mtf_trend, mtf_trend_available = _asof_group_values(
        base_query_times,
        h4_ends,
        h4_available,
        h4_valid,
        h4_trend,
    )
    data["mtf_trend_4h"] = mtf_trend
    data["mtf_trend_4h_available_at"] = pd.Series(mtf_trend_available, index=data.index)
    data["mtf_bias_4h"] = np.sign(mtf_trend)
    data["mtf_bias_4h_available_at"] = pd.Series(mtf_trend_available, index=data.index)
    mtf_strength = data["mtf_trend_4h"].rolling(12).mean()
    mtf_strength_available = _rolling_availability(
        mtf_strength, mtf_trend_available, window=12
    )
    data["mtf_strength_4h"] = mtf_strength
    data["mtf_strength_4h_available_at"] = pd.Series(
        mtf_strength_available, index=data.index
    )

    daily_ends, daily_available, daily_valid, daily_close = _completed_higher_close(
        data,
        base_interval=base_offset,
        higher_interval="1D",
        close_labels=close_labels,
        row_available=row_available,
        row_complete=row_complete,
    )
    daily_close_asof, daily_close_available = _asof_group_values(
        base_query_times,
        daily_ends,
        daily_available,
        daily_valid,
        daily_close,
    )
    denominator = data["Close"].replace(0, np.nan).to_numpy(dtype=float)
    daily_trend = (daily_close_asof - denominator) / denominator
    daily_trend_available = _max_availability(
        daily_close_available,
        base_query_times,
        np.isfinite(daily_trend),
    )
    data["daily_trend"] = daily_trend
    data["daily_trend_available_at"] = pd.Series(
        daily_trend_available, index=data.index
    )
    return data