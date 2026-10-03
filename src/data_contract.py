from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MarketBar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    timeframe: str
    source: str = "UNKNOWN"

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "Open": self.open,
            "High": self.high,
            "Low": self.low,
            "Close": self.close,
            "timeframe": self.timeframe,
            "source": self.source,
        }


def normalize_ohlc(
    df: pd.DataFrame,
    *,
    timeframe: str,
    source: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Normalize external OHLC into the internal UTC contract.

    The source must provide OHLC. The function does not synthesize missing
    prices and does not silently localize naive timestamps.
    """
    availability_mode = df.attrs.get("availability_mode")
    if availability_mode not in {None, "event_time", "nominal_close_fallback"}:
        raise ValueError("Unsupported availability_mode.")
    has_availability = "available_at" in df.columns or "availability_ts" in df.columns
    has_interval_end = "interval_end" in df.columns
    if availability_mode == "event_time" and not (has_availability or has_interval_end):
        raise ValueError("Event-time availability metadata is required.")
    if availability_mode == "nominal_close_fallback" and (has_availability or has_interval_end):
        raise ValueError("Nominal-close fallback cannot include event-time availability metadata.")

    bar_label = str(df.attrs.get("bar_label", "left")).lower()
    if bar_label not in {"left", "right"}:
        raise ValueError("bar_label must be either 'left' or 'right'.")

    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    metadata_columns = [
        column
        for column in (
            "available_at",
            "availability_ts",
            "interval_end",
            "historical_complete",
            "is_complete",
        )
        if column in df.columns
    ]
    out = df[["Open", "High", "Low", "Close", *metadata_columns]].copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("Market data index must be a DatetimeIndex.")

    if out.index.tz is None:
        raise ValueError(
            "Naive timestamps are not accepted. Localize the source explicitly before normalization."
        )

    out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    out.attrs["bar_label"] = bar_label

    for column in ("available_at", "availability_ts", "interval_end"):
        if column in out.columns:
            for value in out[column].dropna():
                try:
                    timestamp = pd.Timestamp(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{column} must contain valid timestamps.") from exc
                if timestamp.tzinfo is None:
                    raise ValueError(f"{column} timestamps must be timezone-aware.")
            out[column] = pd.to_datetime(out[column], utc=True, errors="raise")

    if "interval_end" in out.columns:
        if out["interval_end"].isna().any():
            raise ValueError("Bar interval end cannot be missing.")
        if not out["interval_end"].is_monotonic_increasing:
            raise ValueError("Bar interval ends must be non-decreasing.")
        if (out["interval_end"] < out.index).any():
            raise ValueError("Bar interval end cannot precede its timestamp label.")

        if "available_at" not in out.columns and "availability_ts" not in out.columns:
            out["available_at"] = out["interval_end"]
            has_availability = True

    if "available_at" not in out.columns and "availability_ts" in out.columns:
        out["available_at"] = out["availability_ts"]
    if "available_at" in out.columns:
        if "availability_ts" in out.columns:
            same_availability = out["available_at"].eq(out["availability_ts"]) | (
                out["available_at"].isna() & out["availability_ts"].isna()
            )
            if not same_availability.all():
                raise ValueError("available_at and availability_ts must match when both are provided.")
        before_bar = out["available_at"].notna() & (out["available_at"] < out.index)
        if before_bar.any():
            raise ValueError("Bar availability cannot precede its timestamp label.")
        if "interval_end" in out.columns:
            before_close = out["available_at"].notna() & (
                out["available_at"] < out["interval_end"]
            )
            if before_close.any():
                raise ValueError("Bar availability cannot precede its interval end.")
            missing_end = out["interval_end"].isna() & out["available_at"].notna()
            if missing_end.any():
                from .timeframe_context import _nominal_close_times

                nominal_close = _nominal_close_times(out, timeframe)
                before_nominal_close = missing_end & (
                    out["available_at"] < nominal_close
                )
                if before_nominal_close.any():
                    raise ValueError("Bar availability cannot precede its nominal close.")
        elif out["available_at"].notna().any():
            out.attrs["bar_label"] = bar_label
            from .timeframe_context import _nominal_close_times

            nominal_close = _nominal_close_times(out, timeframe)
            before_close = out["available_at"].notna() & (
                out["available_at"] < nominal_close
            )
            if before_close.any():
                raise ValueError("Bar availability cannot precede its nominal close.")

        if not out["available_at"].dropna().is_monotonic_increasing:
            raise ValueError("Bar availability timestamps must be non-decreasing.")

    for column in ("historical_complete", "is_complete"):
        if column in out.columns:
            out[column] = out[column].fillna(False).astype(bool)

    if out.index.has_duplicates:
        raise ValueError("Market data contains duplicate timestamps.")

    if not out.index.is_monotonic_increasing:
        raise ValueError("Market data must be sorted ascending.")

    if out[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("Market data contains missing OHLC values.")

    if (out["High"] < out[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("Invalid OHLC: High is below Open or Close.")

    if (out["Low"] > out[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC: Low is above Open or Close.")

    out.attrs["timeframe"] = timeframe
    out.attrs["source"] = source
    out.attrs["timezone"] = "UTC"
    out.attrs["availability_mode"] = (
        "event_time"
        if "available_at" in out.columns or "availability_ts" in out.columns or "interval_end" in out.columns
        else "nominal_close_fallback"
    )
    out.attrs["bar_label"] = bar_label
    return out


def as_market_bars(df: pd.DataFrame) -> Iterable[MarketBar]:
    timeframe = str(df.attrs.get("timeframe", "UNKNOWN"))
    source = str(df.attrs.get("source", "UNKNOWN"))

    for timestamp, row in df.iterrows():
        yield MarketBar(
            timestamp=timestamp,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            timeframe=timeframe,
            source=source,
        )
