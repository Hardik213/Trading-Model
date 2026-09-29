from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Iterator

import numpy as np
import pandas as pd

SUPPORTED_INTERVALS = ("1min", "3min", "5min", "15min")


def _coerce_utc_timestamp(value: Any) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Tick timestamp must be valid") from exc
    if pd.isna(ts):
        raise ValueError("Tick timestamp must not be null or NaT")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _validate_numeric_price(value: Any, name: str) -> float:
    if value is None or pd.isna(value):
        raise ValueError(f"{name} must not be null")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def validate_tick_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"timestamp", "bid", "ask"}.issubset(frame.columns):
        raise ValueError("Tick data must include timestamp, bid, and ask columns")
    if frame.empty:
        return frame.copy()

    df = frame.copy()
    if df["timestamp"].isna().any():
        raise ValueError("Tick timestamps must not be null")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df["bid"] = pd.to_numeric(df["bid"], errors="raise")
    df["ask"] = pd.to_numeric(df["ask"], errors="raise")

    if df["timestamp"].isna().any():
        raise ValueError("Tick timestamps must not be NaT")
    if df["bid"].isna().any() or df["ask"].isna().any():
        raise ValueError("Bid and ask prices must not be null")
    if not np.isfinite(df["bid"].to_numpy(dtype=float)).all() or not np.isfinite(
        df["ask"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Bid and ask prices must be finite")
    if (df["bid"] <= 0).any() or (df["ask"] <= 0).any():
        raise ValueError("Bid and ask prices must be positive")
    if (df["ask"] < df["bid"]).any():
        raise ValueError("Ask price cannot be less than bid price")
    if (df["timestamp"].diff().dropna() < pd.Timedelta(0)).any():
        raise ValueError("Tick timestamps must be non-decreasing")

    return df


def _interval_bounds(ts: pd.Timestamp, interval: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    ts = _coerce_utc_timestamp(ts)
    base = ts.floor(interval)
    end = base + pd.Timedelta(interval)
    return base, end


def _empty_result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "interval_start",
            "interval_end",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
            "tick_count",
            "spread_observations",
            "historical_complete",
            "available_at",
            "availability_ts",
            "is_complete",
        ],
        dtype=object,
    )


class TickAggregator:
    """Aggregate ordered ticks and emit bars on simulated event-time close.

    ``available_at`` records the timestamp of the replay tick that causes an
    interval to be emitted. It is not measured feed-arrival time; the input has
    no arrival-time field and this API assumes no separate wall-clock latency.

    ``historical_complete`` and ``is_complete`` indicate only that replay has
    advanced to or beyond an interval's end. They do not establish that the
    source feed contained every tick or that the underlying data is complete.
    """

    def __init__(self, interval: str):
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")
        self.interval = interval
        self._buckets: OrderedDict[pd.Timestamp, dict[str, Any]] = OrderedDict()
        self._last_ts: pd.Timestamp | None = None
        self._flushed = False

    def add_tick(self, ts: Any, bid: Any, ask: Any) -> list[dict[str, Any]]:
        if self._flushed:
            raise RuntimeError("Cannot add ticks after the aggregator has been flushed")
        ts = _coerce_utc_timestamp(ts)

        bid = _validate_numeric_price(bid, "bid")
        ask = _validate_numeric_price(ask, "ask")
        if ask < bid:
            raise ValueError("Ask price cannot be less than bid price")
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError("Tick timestamps must be non-decreasing")
        self._last_ts = ts

        interval_start, interval_end = _interval_bounds(ts, self.interval)
        bucket = self._buckets.get(interval_start)
        if bucket is None:
            bucket = {
                "interval_start": interval_start,
                "interval_end": interval_end,
                "bid_open": bid,
                "bid_high": bid,
                "bid_low": bid,
                "bid_close": bid,
                "ask_open": ask,
                "ask_high": ask,
                "ask_low": ask,
                "ask_close": ask,
                "tick_count": 0,
                "spread_observations": 0,
            }
            self._buckets[interval_start] = bucket
        bucket["bid_high"] = max(bucket["bid_high"], bid)
        bucket["bid_low"] = min(bucket["bid_low"], bid)
        bucket["bid_close"] = bid
        bucket["ask_high"] = max(bucket["ask_high"], ask)
        bucket["ask_low"] = min(bucket["ask_low"], ask)
        bucket["ask_close"] = ask
        bucket["tick_count"] += 1
        bucket["spread_observations"] += 1

        emitted: list[dict[str, Any]] = []
        while self._buckets:
            start = next(iter(self._buckets))
            if start + pd.Timedelta(self.interval) > ts:
                break
            _, completed_bucket = self._buckets.popitem(last=False)
            emitted.append(
                _bucket_to_row(
                    completed_bucket,
                    available_at=ts,
                    historical_complete=True,
                )
            )
        return emitted

    def flush(self, include_incomplete: bool = False) -> list[dict[str, Any]]:
        """Seal the stream and optionally return remaining incomplete buckets.

        Flush is terminal: repeated calls return no rows, and adding ticks after
        flushing raises RuntimeError. Incomplete buckets never enter completed
        output and have no event-time availability timestamp.
        """
        if self._flushed:
            return []
        self._flushed = True
        rows: list[dict[str, Any]] = []
        for start in list(self._buckets):
            bucket = self._buckets.pop(start)
            rows.append(_bucket_to_row(bucket, available_at=pd.NaT, historical_complete=False))
        if not include_incomplete:
            return [row for row in rows if row["historical_complete"]]
        return rows


def _bucket_to_row(bucket: dict[str, Any], *, available_at: pd.Timestamp | pd.NaT, historical_complete: bool) -> dict[str, Any]:
    start = bucket["interval_start"]
    end = bucket["interval_end"]
    row = {
        "interval_start": pd.Timestamp(start),
        "interval_end": pd.Timestamp(end),
        "bid_open": float(bucket["bid_open"]),
        "bid_high": float(bucket["bid_high"]),
        "bid_low": float(bucket["bid_low"]),
        "bid_close": float(bucket["bid_close"]),
        "ask_open": float(bucket["ask_open"]),
        "ask_high": float(bucket["ask_high"]),
        "ask_low": float(bucket["ask_low"]),
        "ask_close": float(bucket["ask_close"]),
        "tick_count": int(bucket["tick_count"]),
        "spread_observations": int(bucket["spread_observations"]),
        # Completeness means event-time finalization, not source-feed completeness.
        "historical_complete": bool(historical_complete),
        # This is simulated event-time replay availability, not measured feed arrival.
        "available_at": pd.NaT if pd.isna(available_at) else pd.Timestamp(available_at),
        "availability_ts": pd.NaT if pd.isna(available_at) else pd.Timestamp(available_at),
        # Kept as a compatibility alias for event-time finalization status.
        "is_complete": bool(historical_complete),
    }
    return row


def iter_gaps(frame: pd.DataFrame, interval: str) -> Iterator[pd.Timestamp]:
    """Yield absent interval starts lazily, without allocating a date range.

    This is a forensic signal only: the presence of a gap does not by itself prove
    a feed outage, a market holiday, or a data-quality bug. It simply records that
    the stream did not emit any ticks for one or more interval starts. Memory use
    is proportional to observed input ticks, not the duration of the gaps.
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    if frame.empty:
        return
    interval_delta = pd.Timedelta(interval)
    required_columns = {"timestamp", "bid", "ask"}
    if not required_columns.issubset(frame.columns):
        raise ValueError("Tick data must include timestamp, bid, and ask columns")

    previous_timestamp: pd.Timestamp | None = None
    previous_start: pd.Timestamp | None = None
    for tick in frame.itertuples(index=False):
        timestamp = _coerce_utc_timestamp(tick.timestamp)
        bid = _validate_numeric_price(tick.bid, "bid")
        ask = _validate_numeric_price(tick.ask, "ask")
        if ask < bid:
            raise ValueError("Ask price cannot be less than bid price")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("Tick timestamps must be non-decreasing")
        previous_timestamp = timestamp

        current_start, _ = _interval_bounds(timestamp, interval)
        if current_start == previous_start:
            continue
        if previous_start is not None:
            missing_start = previous_start + interval_delta
            while missing_start < current_start:
                yield pd.Timestamp(missing_start)
                missing_start += interval_delta
        previous_start = current_start


def detect_gaps(frame: pd.DataFrame, interval: str) -> list[pd.Timestamp]:
    """Materialize :func:`iter_gaps` for callers that need a list.

    For potentially long sparse histories, consume ``iter_gaps`` directly to
    avoid retaining every missing interval in memory.
    """
    return list(iter_gaps(frame, interval))


def aggregate_ticks_incremental(
    frame: pd.DataFrame,
    interval: str,
    *,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """Replay an ordered frame through :class:`TickAggregator`.

    Availability values are simulated event-time replay timestamps, not measured
    market-data arrival times. The final open interval is omitted unless
    ``include_incomplete`` is true.
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    if frame.empty:
        return _empty_result_frame()
    df = validate_tick_frame(frame)
    aggregator = TickAggregator(interval)
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        rows.extend(aggregator.add_tick(row.timestamp, row.bid, row.ask))
    rows.extend(aggregator.flush(include_incomplete=include_incomplete))
    if not rows:
        return _empty_result_frame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["interval_start", "interval_end"], kind="stable").reset_index(drop=True)
    return out


def aggregate_ticks(
    frame: pd.DataFrame,
    interval: str,
    *,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    return aggregate_ticks_incremental(frame, interval, include_incomplete=include_incomplete)
