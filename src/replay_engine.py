from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence

import pandas as pd

from .data_contract import normalize_ohlc
from .replay_subject import ReplaySubject
from .timeframe_context import (
    TimeframeContext,
    _complete_mask,
    _event_time_visibility_mask,
    bars_available_as_of,
    eligible_decision_times,
)


class ReplayDecision(str, Enum):
    NO_TRADE = "NO_TRADE"
    DEVELOPING = "DEVELOPING"
    VALID = "VALID"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class ReplayObservation:
    timestamp: pd.Timestamp
    decision: ReplayDecision
    state: str
    reason: str
    replay_subject: ReplaySubject | None = None


@dataclass(frozen=True)
class ReplayConfig:
    timeframe: str = "5M"
    source: str = "UNKNOWN"
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None


DecisionCallback = Callable[
    [pd.Timestamp, pd.DataFrame, TimeframeContext],
    ReplayObservation,
]


class HistoricalReplay:
    """
    Single chronological driver for the strategy stack.

    At decision time t it provides only:
      - base bars <= t
      - higher-timeframe bars <= t
      - no future rows

    The callback is where the already-built ICT-2022 state machine is connected.
    The replay engine itself does not invent a setup.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        context: TimeframeContext,
        config: ReplayConfig,
    ) -> None:
        self.data = normalize_ohlc(
            data,
            timeframe=config.timeframe,
            source=config.source,
        )
        self.context = context
        self.config = config

    def decision_times(self) -> pd.DatetimeIndex:
        return eligible_decision_times(
            self.data,
            timeframe=self.config.timeframe,
            start=self.config.start,
            end=self.config.end,
        )

    def run(self, callback: DecisionCallback) -> list[ReplayObservation]:
        observations: list[ReplayObservation] = []

        for timestamp in self.decision_times():
            # This is the no-lookahead boundary.
            visible_base = bars_available_as_of(
                self.data,
                timestamp,
                timeframe=self.config.timeframe,
            )

            visible_frames = {
                name: self.context.available_as_of(name, timestamp)
                for name, frame in self.context.frames.items()
            }
            visible_context = TimeframeContext(frames=visible_frames)

            observation = callback(
                timestamp,
                visible_base,
                visible_context,
            )

            if observation.timestamp != timestamp:
                raise ValueError(
                    "Replay callback returned an observation with the wrong timestamp."
                )

            observations.append(observation)

        return observations


@dataclass(frozen=True)
class IncrementalReplayBar:
    """One finalized OHLC bar for an incremental replay availability group."""

    timeframe: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    available_at: pd.Timestamp
    interval_end: pd.Timestamp | None = None
    availability_ts: pd.Timestamp | None = None
    historical_complete: bool | None = None
    is_complete: bool | None = None
    bar_label: str = "left"


@dataclass(frozen=True)
class ReplayAvailabilityGroup:
    """Atomic set of bars sharing one explicit UTC availability timestamp."""

    available_at: pd.Timestamp
    bars_by_timeframe: Mapping[str, Sequence[IncrementalReplayBar]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_at", _require_utc(self.available_at, "available_at"))
        frozen_bars = {
            timeframe: tuple(bars)
            for timeframe, bars in self.bars_by_timeframe.items()
        }
        object.__setattr__(self, "bars_by_timeframe", MappingProxyType(frozen_bars))


def _require_utc(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


class IncrementalHistoricalReplay:
    """Bounded rolling-window replay for explicitly grouped event-time bars.

    Each accepted group is atomic: all its timeframes are incorporated before
    its base decision can run. The callback receives isolated copies containing
    at most the configured number of visible bars per timeframe. It must not
    rely on history outside those windows unless it maintains separately
    bounded state itself.
    """

    def __init__(
        self,
        *,
        config: ReplayConfig,
        history_limits: Mapping[str, int],
        max_pending_groups: int,
        max_pending_bars: int,
    ) -> None:
        if not history_limits or config.timeframe not in history_limits:
            raise ValueError("history_limits must include the replay base timeframe.")
        if any(
            not timeframe
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            for timeframe, limit in history_limits.items()
        ):
            raise ValueError("Each timeframe history limit must be a positive integer.")
        for name, limit in (
            ("max_pending_groups", max_pending_groups),
            ("max_pending_bars", max_pending_bars),
        ):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"{name} must be a positive integer.")

        self.config = config
        self.history_limits = dict(history_limits)
        self._history: dict[str, deque[IncrementalReplayBar]] = {
            timeframe: deque(maxlen=limit)
            for timeframe, limit in sorted(history_limits.items())
        }
        self._bar_labels: dict[str, str] = {}
        self._interval_end_schema: dict[str, bool] = {}
        self._last_fed_label: dict[str, pd.Timestamp] = {}
        self._last_fed_interval_end: dict[str, pd.Timestamp] = {}
        self._last_fed_availability: pd.Timestamp | None = None
        self._last_processed_decision: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self._pending: deque[tuple[ReplayAvailabilityGroup, dict[str, pd.DataFrame]]] = deque()
        self._current_decision_subjects: deque[ReplaySubject] = deque()
        self._pending_bar_count = 0
        self._max_pending_groups = max_pending_groups
        self._max_pending_bars = max_pending_bars
        self._state = "open"

    @property
    def pending_group_count(self) -> int:
        return len(self._pending)

    @property
    def retained_bar_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {timeframe: len(bars) for timeframe, bars in self._history.items()}
        )

    def _ensure_open(self) -> None:
        if self._state != "open":
            raise RuntimeError(f"Incremental replay is {self._state}.")

    def _validate_group(
        self,
        group: ReplayAvailabilityGroup,
    ) -> dict[str, pd.DataFrame]:
        availability = _require_utc(group.available_at, "group availability")
        if self._last_fed_availability is not None and availability <= self._last_fed_availability:
            raise ValueError("Availability groups must have strictly increasing timestamps.")
        if not group.bars_by_timeframe:
            raise ValueError("An availability group must contain at least one bar.")

        validated: dict[str, pd.DataFrame] = {}
        for timeframe, bars in group.bars_by_timeframe.items():
            if timeframe not in self._history:
                raise ValueError(f"Unconfigured timeframe: {timeframe}")
            if not bars:
                raise ValueError(f"Availability group contains no bars for {timeframe}.")
            if any(not isinstance(bar, IncrementalReplayBar) for bar in bars):
                raise TypeError("Availability groups must contain IncrementalReplayBar values.")
            if any(bar.timeframe != timeframe for bar in bars):
                raise ValueError("Bar timeframe does not match its availability-group key.")
            if any(_require_utc(bar.available_at, "bar availability") != availability for bar in bars):
                raise ValueError("Every bar in a group must share its group availability timestamp.")

            labels = [_require_utc(bar.timestamp, "bar timestamp") for bar in bars]
            if any(right <= left for left, right in zip(labels, labels[1:])):
                raise ValueError("Bar labels within a timeframe must be strictly increasing.")
            previous_label = self._last_fed_label.get(timeframe)
            if previous_label is not None and labels[0] <= previous_label:
                raise ValueError("Bar labels must be strictly increasing across groups.")

            label_style = {bar.bar_label.lower() for bar in bars}
            if len(label_style) != 1 or not label_style.issubset({"left", "right"}):
                raise ValueError("Bars in a timeframe group must share a valid bar_label.")
            established_style = self._bar_labels.get(timeframe)
            if established_style is not None and established_style not in label_style:
                raise ValueError("bar_label cannot change within a timeframe stream.")
            has_interval_end = [bar.interval_end is not None for bar in bars]
            if any(has_interval_end) and not all(has_interval_end):
                raise ValueError("interval_end must be present consistently within a group.")
            established_interval_end = self._interval_end_schema.get(timeframe)
            if (
                established_interval_end is not None
                and established_interval_end != all(has_interval_end)
            ):
                raise ValueError("interval_end presence cannot change within a timeframe stream.")

            rows = []
            for bar in bars:
                row: dict[str, object] = {
                    "Open": bar.open,
                    "High": bar.high,
                    "Low": bar.low,
                    "Close": bar.close,
                    "available_at": availability,
                    "availability_ts": (
                        availability
                        if bar.availability_ts is None
                        else _require_utc(bar.availability_ts, "availability_ts")
                    ),
                }
                if bar.interval_end is not None:
                    row["interval_end"] = _require_utc(bar.interval_end, "interval_end")
                if bar.historical_complete is not None:
                    row["historical_complete"] = bar.historical_complete
                if bar.is_complete is not None:
                    row["is_complete"] = bar.is_complete
                rows.append(row)

            frame = pd.DataFrame(rows, index=pd.DatetimeIndex(labels, name="timestamp"))
            frame.attrs.update(
                {
                    "availability_mode": "event_time",
                    "bar_label": next(iter(label_style)),
                    "timeframe": timeframe,
                    "source": self.config.source,
                }
            )
            # Check ordering before normalization, which sorts batch input by design.
            frame = normalize_ohlc(
                frame,
                timeframe=timeframe,
                source=self.config.source,
            )
            if frame["available_at"].nunique(dropna=False) != 1:
                raise ValueError("An availability group must have one event time per timeframe.")
            interval_ends = frame.get("interval_end")
            if interval_ends is not None:
                previous_end = self._last_fed_interval_end.get(timeframe)
                if previous_end is not None and interval_ends.iloc[0] < previous_end:
                    raise ValueError("Interval ends must be non-decreasing across groups.")
            validated[timeframe] = frame
        return validated

    def feed_group(self, group: ReplayAvailabilityGroup) -> None:
        """Validate and enqueue one complete, bounded availability group."""
        self._ensure_open()
        if not isinstance(group, ReplayAvailabilityGroup):
            raise TypeError("group must be a ReplayAvailabilityGroup.")
        input_bar_count = sum(len(bars) for bars in group.bars_by_timeframe.values())
        if len(self._pending) + 1 > self._max_pending_groups:
            raise ValueError("Pending availability-group limit exceeded.")
        if self._pending_bar_count + input_bar_count > self._max_pending_bars:
            raise ValueError("Pending bar limit exceeded.")
        validated = self._validate_group(group)
        bar_count = sum(len(frame) for frame in validated.values())

        for timeframe, frame in validated.items():
            self._last_fed_label[timeframe] = frame.index[-1]
            self._bar_labels[timeframe] = str(frame.attrs["bar_label"])
            self._interval_end_schema[timeframe] = "interval_end" in frame.columns
            if "interval_end" in frame.columns:
                self._last_fed_interval_end[timeframe] = frame["interval_end"].iloc[-1]
        self._last_fed_availability = _require_utc(group.available_at, "group availability")
        self._pending.append((group, validated))
        self._pending_bar_count += bar_count

    def _history_frame(self, timeframe: str) -> pd.DataFrame:
        bars = tuple(self._history[timeframe])
        rows: list[dict[str, object]] = []
        labels: list[pd.Timestamp] = []
        for bar in bars:
            row: dict[str, object] = {
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "available_at": bar.available_at,
            }
            row["availability_ts"] = (
                bar.available_at
                if bar.availability_ts is None
                else bar.availability_ts
            )
            if bar.interval_end is not None:
                row["interval_end"] = bar.interval_end
            if bar.historical_complete is not None:
                row["historical_complete"] = bar.historical_complete
            if bar.is_complete is not None:
                row["is_complete"] = bar.is_complete
            rows.append(row)
            labels.append(bar.timestamp)

        columns = ["Open", "High", "Low", "Close", "available_at"]
        for optional in (
            "availability_ts",
            "interval_end",
            "historical_complete",
            "is_complete",
        ):
            if any(optional in row for row in rows):
                columns.append(optional)
        index = pd.DatetimeIndex(labels, name="timestamp")
        if not labels:
            index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        frame = pd.DataFrame(
            rows,
            columns=columns,
            index=index,
        )
        frame.index.name = None
        frame.attrs.update(
            {
                "availability_mode": "event_time",
                "bar_label": self._bar_labels.get(timeframe, "left"),
                "timeframe": timeframe,
                "source": self.config.source,
            }
        )
        return frame

    def process_next(self, callback: DecisionCallback) -> ReplayObservation | None:
        """Process the next eligible decision, returning no retained history."""
        if self._state == "failed":
            raise RuntimeError("Incremental replay is failed and cannot be resumed.")
        if self._state == "finished":
            return None
        try:
            while not self._current_decision_subjects:
                if not self._pending:
                    return None
                _group, frames = self._pending.popleft()
                self._pending_bar_count -= sum(len(frame) for frame in frames.values())
                for timeframe, frame in frames.items():
                    complete = _complete_mask(frame)
                    for timestamp, row in frame.loc[complete].iterrows():
                        self._history[timeframe].append(
                            IncrementalReplayBar(
                                timeframe=timeframe,
                                timestamp=timestamp,
                                open=float(row["Open"]),
                                high=float(row["High"]),
                                low=float(row["Low"]),
                                close=float(row["Close"]),
                                available_at=row["available_at"],
                                interval_end=row.get("interval_end"),
                                availability_ts=row.get("availability_ts"),
                                historical_complete=row.get("historical_complete"),
                                is_complete=row.get("is_complete"),
                                bar_label=str(frame.attrs["bar_label"]),
                            )
                        )

                base_frame = frames.get(self.config.timeframe)
                if base_frame is None:
                    continue
                eligible_availability = set(eligible_decision_times(
                    base_frame,
                    timeframe=self.config.timeframe,
                    start=self.config.start,
                    end=self.config.end,
                ))
                decision_subjects = []
                for base_timestamp, row in base_frame.loc[_complete_mask(base_frame)].iterrows():
                    availability = _require_utc(row["available_at"], "decision timestamp")
                    if availability in eligible_availability:
                        decision_subjects.append(
                            ReplaySubject(
                                availability_timestamp=availability,
                                base_bar_timestamp=_require_utc(
                                    base_timestamp,
                                    "base bar timestamp",
                                ),
                            )
                        )
                self._current_decision_subjects.extend(
                    sorted(
                        decision_subjects,
                        key=lambda subject: (
                            subject.availability_timestamp,
                            subject.base_bar_timestamp,
                        ),
                    )
                )

            subject = self._current_decision_subjects.popleft()
            timestamp = subject.availability_timestamp
            base_bar_timestamp = subject.base_bar_timestamp
            decision_identity = (timestamp, base_bar_timestamp)
            if (
                self._last_processed_decision is not None
                and decision_identity <= self._last_processed_decision
            ):
                raise ValueError("Decision identities must be strictly increasing.")

            snapshots = {
                timeframe: self._history_frame(timeframe)
                for timeframe in self._history
            }
            visible_frames = {
                timeframe: (
                    frame.loc[_event_time_visibility_mask(frame, timestamp)].copy()
                    if len(frame)
                    else frame.copy()
                )
                for timeframe, frame in snapshots.items()
            }
            visible_frames[self.config.timeframe] = visible_frames[
                self.config.timeframe
            ].loc[lambda frame: frame.index <= base_bar_timestamp].copy()
            visible_base = visible_frames[self.config.timeframe].reindex(
                columns=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "available_at",
                    "availability_ts",
                    "interval_end",
                    "historical_complete",
                    "is_complete",
                ]
            )
            visible_context = TimeframeContext(
                frames={
                    timeframe: frame.reindex(
                        columns=[
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            *[
                                column
                                for column in (
                                    "interval_end",
                                    "available_at",
                                    "availability_ts",
                                    "historical_complete",
                                    "is_complete",
                                )
                                if column in frame.columns
                            ],
                        ]
                    )
                    for timeframe, frame in visible_frames.items()
                },
                replay_subject=subject,
            )

            # Mark before callback invocation: callback side effects are not retry-safe.
            self._last_processed_decision = decision_identity
            observation = callback(timestamp, visible_base, visible_context)
            if not isinstance(observation, ReplayObservation):
                raise TypeError("Replay callback must return a ReplayObservation.")
            if observation.timestamp != timestamp:
                raise ValueError("Replay callback returned an observation with the wrong timestamp.")
            if observation.replay_subject not in (None, subject):
                raise ValueError("Replay callback returned an observation for the wrong subject.")
            if observation.replay_subject is None:
                observation = replace(observation, replay_subject=subject)
            return observation
        except Exception:
            self._state = "failed"
            raise

    def finish(self) -> None:
        """Close the stream after all accepted availability groups are drained."""
        if self._state == "finished":
            return
        self._ensure_open()
        if self._pending or self._current_decision_subjects:
            raise RuntimeError("Pending availability groups must be processed before finish().")
        self._state = "finished"


def assert_no_future_data(
    visible: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> None:
    future = visible.index[visible.index > as_of]
    if len(future):
        raise AssertionError(
            f"Replay leaked {len(future)} future rows at {as_of}."
        )
    availability_column = next(
        (column for column in ("available_at", "availability_ts") if column in visible.columns),
        None,
    )
    if availability_column is not None:
        available_at = pd.to_datetime(visible[availability_column], utc=True, errors="raise")
        unavailable = available_at.isna() | (available_at > as_of)
        for column in ("historical_complete", "is_complete"):
            if column in visible.columns:
                unavailable |= ~visible[column].fillna(False).astype(bool)
        if unavailable.any():
            raise AssertionError(
                f"Replay exposed {int(unavailable.sum())} bars before event-time availability at {as_of}."
            )
    elif visible.attrs.get("availability_mode") == "nominal_close_fallback":
        timeframe = str(visible.attrs.get("timeframe", ""))
        from .timeframe_context import _nominal_close_times
        if (_nominal_close_times(visible, timeframe) > as_of).any():
            raise AssertionError(
                f"Replay exposed bars before nominal close at {as_of}; feed arrival is not verified."
            )
