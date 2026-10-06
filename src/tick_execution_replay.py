from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import pandas as pd

from .replay_engine import (
    IncrementalHistoricalReplay,
    IncrementalReplayBar,
    ReplayAvailabilityGroup,
    ReplayDecision,
    ReplayObservation,
    assert_no_future_data,
)
from .replay_subject import ReplaySubject
from .tick_aggregation import TickAggregator
from .tick_execution import DecisionRecord, QuoteEvent
from .timeframe_context import TimeframeContext


StrategyCallback = Callable[
    [pd.Timestamp, pd.DataFrame, TimeframeContext],
    DecisionRecord,
]
ExecutionTickCallback = Callable[[QuoteEvent], None]
DecisionSetCallback = Callable[["TickExecutionDecisionSet"], None]


def _as_utc(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class TickExecutionDecisionSet:
    """All strategy decisions generated from one complete replay availability group."""

    available_at: pd.Timestamp
    replay_group: ReplayAvailabilityGroup
    decisions: tuple[DecisionRecord, ...]

    def __post_init__(self) -> None:
        available_at = _as_utc(self.available_at, "available_at")
        object.__setattr__(self, "available_at", available_at)
        if self.replay_group.available_at != available_at:
            raise ValueError("Decision set and replay group availability must match.")
        seen_subjects: set[str] = set()
        for decision in self.decisions:
            if decision.subject.availability_timestamp != available_at:
                raise ValueError("Every decision subject must share the decision-set availability.")
            if decision.decision_time != available_at:
                raise ValueError("Decision time must equal subject availability.")
            if decision.subject.subject_id in seen_subjects:
                raise ValueError("A decision set cannot contain duplicate subjects.")
            seen_subjects.add(decision.subject.subject_id)


@dataclass(frozen=True)
class TickExecutionReplaySummary:
    ticks_processed: int
    availability_groups: int
    subjects_processed: int
    decision_sets_submitted: int


class TickExecutionReplay:
    """Causal orchestration seam from canonical ticks to complete decision sets.

    The execution-tick callback runs before aggregation and strategy evaluation
    for each quote. The decision-set callback receives all decisions produced
    by one availability group only after every subject in that group is drained.
    """

    def __init__(
        self,
        *,
        aggregators: Mapping[str, TickAggregator],
        replay: IncrementalHistoricalReplay,
        strategy_callback: StrategyCallback,
        on_tick: ExecutionTickCallback,
        on_decision_set: DecisionSetCallback,
    ) -> None:
        if not aggregators:
            raise ValueError("At least one TickAggregator is required.")
        if replay.config.timeframe not in aggregators:
            raise ValueError("Aggregators must include the replay base timeframe.")
        for timeframe, aggregator in aggregators.items():
            if not isinstance(aggregator, TickAggregator):
                raise TypeError("Each aggregator must be a TickAggregator.")
            if timeframe != aggregator.interval:
                raise ValueError("Aggregator mapping keys must match their intervals.")
            if timeframe not in replay.history_limits:
                raise ValueError(f"Replay history is not configured for {timeframe}.")
        if not callable(strategy_callback) or not callable(on_tick) or not callable(on_decision_set):
            raise TypeError("Strategy, execution-tick, and decision-set callbacks must be callable.")

        self.aggregators = dict(aggregators)
        self.replay = replay
        self.strategy_callback = strategy_callback
        self.on_tick = on_tick
        self.on_decision_set = on_decision_set
        self._has_run = False
        self._last_decision: DecisionRecord | None = None

    def run(self, ticks: Iterable[QuoteEvent]) -> TickExecutionReplaySummary:
        if self._has_run:
            raise RuntimeError("A TickExecutionReplay instance can run only once.")
        self._has_run = True

        ticks_processed = 0
        groups_processed = 0
        subjects_processed = 0
        decision_sets_submitted = 0
        previous_timestamp: pd.Timestamp | None = None
        previous_source_order: int | None = None

        for tick in ticks:
            if not isinstance(tick, QuoteEvent):
                raise TypeError("ticks must contain QuoteEvent values.")
            if previous_timestamp is not None and tick.timestamp < previous_timestamp:
                raise ValueError("Tick timestamps must be non-decreasing.")
            if previous_source_order is not None and tick.source_order <= previous_source_order:
                raise ValueError("Tick source_order must be strictly increasing.")
            previous_timestamp = tick.timestamp
            previous_source_order = tick.source_order
            ticks_processed += 1

            self.on_tick(tick)

            emitted_by_timeframe: dict[str, tuple[IncrementalReplayBar, ...]] = {}
            for timeframe, aggregator in self.aggregators.items():
                rows = aggregator.add_tick(tick.timestamp, tick.bid, tick.ask)
                bars = tuple(
                    self._replay_bar(timeframe, row, tick.timestamp)
                    for row in rows
                )
                if bars:
                    emitted_by_timeframe[timeframe] = bars

            if not emitted_by_timeframe:
                continue

            group = ReplayAvailabilityGroup(
                available_at=tick.timestamp,
                bars_by_timeframe=emitted_by_timeframe,
            )
            self.replay.feed_group(group)
            groups_processed += 1
            decisions: list[DecisionRecord] = []

            while True:
                self._last_decision = None
                observation = self.replay.process_next(self._evaluate_subject)
                if observation is None:
                    break
                subjects_processed += 1
                decision = self._last_decision
                if decision is None:
                    raise RuntimeError("Strategy callback completed without producing a decision.")
                decisions.append(decision)
                self._last_decision = None

            decision_set = TickExecutionDecisionSet(
                available_at=tick.timestamp,
                replay_group=group,
                decisions=tuple(decisions),
            )
            self.on_decision_set(decision_set)
            decision_sets_submitted += 1

        for aggregator in self.aggregators.values():
            aggregator.flush()
        self.replay.finish()
        return TickExecutionReplaySummary(
            ticks_processed=ticks_processed,
            availability_groups=groups_processed,
            subjects_processed=subjects_processed,
            decision_sets_submitted=decision_sets_submitted,
        )

    def _evaluate_subject(
        self,
        timestamp: pd.Timestamp,
        visible_base: pd.DataFrame,
        visible_context: TimeframeContext,
    ) -> ReplayObservation:
        subject = visible_context.replay_subject
        if not isinstance(subject, ReplaySubject):
            raise ValueError("Incremental replay must provide a ReplaySubject for each decision.")
        if subject.availability_timestamp != timestamp:
            raise ValueError("Decision time must equal ReplaySubject availability_timestamp.")
        if not visible_base.empty and (visible_base.index > subject.base_bar_timestamp).any():
            raise ValueError("Visible base bars exceed the ReplaySubject base-bar timestamp.")
        assert_no_future_data(visible_base, as_of=timestamp)
        for frame in visible_context.frames.values():
            assert_no_future_data(frame, as_of=timestamp)

        decision = self.strategy_callback(timestamp, visible_base, visible_context)
        if not isinstance(decision, DecisionRecord):
            raise TypeError("Strategy callback must return a DecisionRecord.")
        if decision.subject != subject:
            raise ValueError("Strategy callback returned a decision for the wrong ReplaySubject.")
        if decision.decision_time != subject.availability_timestamp:
            raise ValueError("Decision time must equal ReplaySubject availability_timestamp.")
        if decision.actionable and decision.quantity is None:
            raise ValueError(
                "An actionable execution decision requires an explicit quantity; "
                "quantity is not inferred by the replay/execution orchestrator."
            )

        self._last_decision = decision
        replay_decision = ReplayDecision.VALID if decision.actionable else ReplayDecision.NO_TRADE
        return ReplayObservation(
            timestamp=timestamp,
            decision=replay_decision,
            state=decision.strategy_state or replay_decision.value,
            reason=decision.reason or "",
            replay_subject=subject,
        )

    @staticmethod
    def _replay_bar(
        timeframe: str,
        row: Mapping[str, object],
        available_at: pd.Timestamp,
    ) -> IncrementalReplayBar:
        row_availability = _as_utc(row["available_at"], "bar available_at")
        if row_availability != available_at:
            raise ValueError("Bars emitted by one tick must share that tick's availability.")
        return IncrementalReplayBar(
            timeframe=timeframe,
            timestamp=_as_utc(row["interval_start"], "bar timestamp"),
            open=float(row["bid_open"]),
            high=float(row["bid_high"]),
            low=float(row["bid_low"]),
            close=float(row["bid_close"]),
            available_at=row_availability,
            interval_end=_as_utc(row["interval_end"], "bar interval_end"),
            availability_ts=row_availability,
            historical_complete=bool(row["historical_complete"]),
            is_complete=bool(row["is_complete"]),
            bar_label="left",
        )


__all__ = [
    "TickExecutionDecisionSet",
    "TickExecutionReplay",
    "TickExecutionReplaySummary",
]
