from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import pandas as pd

from .core_v1_decision_bridge import (
    CoreV1DecisionBridgeResult,
    CoreV1QuantityInput,
    build_core_v1_decision_bridge,
)
from .core_v1_evidence import (
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
)
from .core_v1_pipeline import CoreV1PipelineResult
from .evidence_timing import EvidenceSourceObservation, EvidenceTiming
from .replay_engine import IncrementalHistoricalReplay
from .replay_subject import ReplaySubject, same_subject_id
from .tick_aggregation import TickAggregator
from .tick_execution import (
    ExecutionEngine,
    ExecutionResult,
    QuoteEvent,
    quote_events_from_tick_batches,
)
from .tick_execution_replay import (
    TickExecutionDecisionSet,
    TickExecutionReplay,
    TickExecutionReplaySummary,
)
from .timeframe_context import TimeframeContext


PipelineBuilder = Callable[[SubjectVisibleInputs], CoreV1PipelineResult]
QuantityProvider = Callable[[CoreV1PipelineResult], CoreV1QuantityInput | None]
EvidenceEnricher = Callable[[SubjectVisibleInputs], SubjectVisibleInputs]


@dataclass(frozen=True)
class CoreV1SubjectExecution:
    subject: ReplaySubject
    visible_inputs: SubjectVisibleInputs
    pipeline_result: CoreV1PipelineResult
    bridge_result: CoreV1DecisionBridgeResult

    def __post_init__(self) -> None:
        if not same_subject_id(self.visible_inputs.subject, self.subject):
            raise ValueError("Visible inputs must preserve the exact ReplaySubject.")
        if not same_subject_id(self.pipeline_result.subject, self.subject):
            raise ValueError("Pipeline result must preserve the exact ReplaySubject.")
        if not same_subject_id(self.bridge_result.decision_record.subject, self.subject):
            raise ValueError("DecisionRecord must preserve ReplaySubject identity.")
        if self.bridge_result.evidence.pipeline_result is not self.pipeline_result:
            raise ValueError("Bridge evidence must retain the exact pipeline result.")


@dataclass(frozen=True)
class CoreV1TickExecutionReport:
    replay_summary: TickExecutionReplaySummary
    execution_results: tuple[ExecutionResult, ...]
    subject_executions: tuple[CoreV1SubjectExecution, ...]
    decision_sets: tuple[TickExecutionDecisionSet, ...]


class CoreV1TickExecution:
    """Additive Core v1 integration over existing replay and execution contracts."""

    def __init__(
        self,
        *,
        aggregators: Mapping[str, TickAggregator],
        replay: IncrementalHistoricalReplay,
        execution: ExecutionEngine,
        pipeline_builder: PipelineBuilder,
        evidence_enricher: EvidenceEnricher | None = None,
        quantity_provider: QuantityProvider | None = None,
        order_type: str = "MARKET",
    ) -> None:
        if not isinstance(execution, ExecutionEngine):
            raise TypeError("execution must be the existing stateful ExecutionEngine.")
        if not callable(pipeline_builder):
            raise TypeError("pipeline_builder must be callable.")
        if evidence_enricher is not None and not callable(evidence_enricher):
            raise TypeError("evidence_enricher must be callable or None.")
        if quantity_provider is not None and not callable(quantity_provider):
            raise TypeError("quantity_provider must be callable or None.")
        self.execution = execution
        self.pipeline_builder = pipeline_builder
        self.evidence_enricher = evidence_enricher
        self.quantity_provider = quantity_provider
        self.order_type = order_type
        self._ticks_by_interval: dict[str, dict[pd.Timestamp, list[QuoteEvent]]] = {
            timeframe: defaultdict(list) for timeframe in aggregators
        }
        self._subject_executions: list[CoreV1SubjectExecution] = []
        self._decision_sets: list[TickExecutionDecisionSet] = []
        self._has_run = False
        self._last_tick: QuoteEvent | None = None
        self._runner = TickExecutionReplay(
            aggregators=aggregators,
            replay=replay,
            strategy_callback=self._evaluate_subject,
            on_tick=self._process_tick,
            on_decision_set=self._submit_decision_set,
        )
        self.replay = replay

    def run(self, ticks: Iterable[QuoteEvent]) -> CoreV1TickExecutionReport:
        if self._has_run:
            raise RuntimeError("CoreV1TickExecution instances may run only once.")
        self._has_run = True
        summary = self._runner.run(ticks)
        execution_results = self.execution.finish()
        return CoreV1TickExecutionReport(
            replay_summary=summary,
            execution_results=execution_results,
            subject_executions=tuple(self._subject_executions),
            decision_sets=tuple(self._decision_sets),
        )

    def run_tick_batches(
        self,
        batches: Iterable[pd.DataFrame],
    ) -> CoreV1TickExecutionReport:
        """Consume canonical tick batches using the existing QuoteEvent converter."""
        return self.run(quote_events_from_tick_batches(batches))

    def _process_tick(self, tick: QuoteEvent) -> None:
        # Existing pending intents and positions must observe the quote first.
        self.execution.process_tick(tick)
        self._last_tick = tick
        for timeframe, buckets in self._ticks_by_interval.items():
            start = tick.timestamp.floor(timeframe)
            buckets[start].append(tick)

    def _evaluate_subject(
        self,
        timestamp: pd.Timestamp,
        visible_base: pd.DataFrame,
        visible_context: TimeframeContext,
    ):
        subject = visible_context.replay_subject
        if not isinstance(subject, ReplaySubject):
            raise ValueError("Replay must provide a ReplaySubject for every decision.")
        if timestamp != subject.availability_timestamp:
            raise ValueError("Decision time must equal subject availability.")

        visible_inputs = self._visible_inputs(subject, visible_base, visible_context)
        if self.evidence_enricher is not None:
            visible_inputs = self.evidence_enricher(visible_inputs)
            if not isinstance(visible_inputs, SubjectVisibleInputs):
                raise TypeError("evidence_enricher must return SubjectVisibleInputs.")
            if not same_subject_id(visible_inputs.subject, subject):
                raise ValueError("Evidence enrichment must preserve ReplaySubject identity.")
            if visible_inputs.availability_timestamp != timestamp:
                raise ValueError("Evidence enrichment must preserve subject availability.")
        pipeline_result = self.pipeline_builder(visible_inputs)
        if not isinstance(pipeline_result, CoreV1PipelineResult):
            raise TypeError("pipeline_builder must return CoreV1PipelineResult.")
        if not same_subject_id(pipeline_result.subject, subject):
            raise ValueError("Pipeline builder must preserve exact ReplaySubject identity.")
        if pipeline_result.availability_timestamp != timestamp:
            raise ValueError("Pipeline result availability must equal decision time.")
        visible_sources = tuple(
            source
            for item in (
                visible_inputs.base_observations
                + visible_inputs.context_observations
            )
            for source in item.timing.source_observations
        )
        for name, sources in pipeline_result.provenance.items():
            if any(source not in visible_sources for source in sources):
                raise ValueError(
                    f"Pipeline {name} provenance references evidence outside visible inputs."
                )

        quantity = None
        if pipeline_result.state.value == "VALID" and self.quantity_provider is not None:
            quantity = self.quantity_provider(pipeline_result)
            if quantity is not None and not isinstance(quantity, CoreV1QuantityInput):
                raise TypeError("quantity_provider must return CoreV1QuantityInput or None.")
        bridge = build_core_v1_decision_bridge(
            pipeline_result,
            quantity=quantity,
            order_type=self.order_type,
        )
        self._subject_executions.append(
            CoreV1SubjectExecution(
                subject=subject,
                visible_inputs=visible_inputs,
                pipeline_result=pipeline_result,
                bridge_result=bridge,
            )
        )
        return bridge.decision_record

    def _visible_inputs(
        self,
        subject: ReplaySubject,
        visible_base: pd.DataFrame,
        visible_context: TimeframeContext,
    ) -> SubjectVisibleInputs:
        base_observations = self._observations_for_frame(
            self.replay.config.timeframe,
            visible_base,
            subject,
            is_base=True,
        )
        context_observations: list[SubjectVisibleObservation] = []
        for timeframe, frame in visible_context.frames.items():
            if timeframe == self.replay.config.timeframe:
                continue
            context_observations.extend(
                self._observations_for_frame(
                    timeframe,
                    frame,
                    subject,
                    is_base=False,
                )
            )
        return SubjectVisibleInputs(
            subject=subject,
            availability_timestamp=subject.availability_timestamp,
            base_observations=tuple(base_observations),
            context_observations=tuple(context_observations),
        )

    def _observations_for_frame(
        self,
        timeframe: str,
        frame: pd.DataFrame,
        subject: ReplaySubject,
        *,
        is_base: bool,
    ) -> list[SubjectVisibleObservation]:
        observations: list[SubjectVisibleObservation] = []
        if frame.empty:
            return observations
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("Visible replay frame labels must be timezone-aware.")
        for label, row in frame.iterrows():
            event_timestamp = _as_utc(label, "bar event timestamp")
            availability = _row_availability(row)
            if availability > subject.availability_timestamp:
                raise ValueError("Replay exposed a bar after subject availability.")
            if is_base and event_timestamp > subject.base_bar_timestamp:
                raise ValueError("Base evidence exceeds the subject base-bar cutoff.")
            interval_start = event_timestamp
            source_ticks = tuple(
                self._ticks_by_interval.get(timeframe, {}).get(interval_start, ())
            )
            if not source_ticks:
                raise ValueError(
                    f"No canonical tick provenance exists for visible {timeframe} bar "
                    f"{event_timestamp}."
                )
            provenance = {
                "timeframe": timeframe,
                "bar_timestamp": event_timestamp.isoformat(),
                "available_at": availability.isoformat(),
                "canonical_ticks": tuple(
                    {
                        "timestamp": tick.timestamp.isoformat(),
                        "source_order": tick.source_order,
                        "provenance": dict(tick.provenance),
                    }
                    for tick in source_ticks
                ),
            }
            source = EvidenceSourceObservation(
                source_id=(
                    f"canonical-bar:{timeframe}:"
                    f"{event_timestamp.isoformat()}:{availability.isoformat()}"
                ),
                source="canonical-quote-ticks",
                event_timestamp=event_timestamp,
                availability_timestamp=availability,
                provenance=provenance,
                base_bar_timestamp=event_timestamp if is_base else None,
                is_base_timeframe=is_base,
            )
            timing = EvidenceTiming(
                event_timestamp=event_timestamp,
                availability_timestamp=availability,
                source_observations=(source,),
                provenance={
                    "integration": "core-v1-tick-execution",
                    "timeframe": timeframe,
                },
                base_bar_scoped=is_base,
            )
            values = {
                str(column): _plain_value(value)
                for column, value in row.items()
            }
            values["timestamp"] = event_timestamp
            observations.append(
                SubjectVisibleObservation(
                    value=values,
                    timing=timing,
                    scope=(
                        ObservationScope.BASE
                        if is_base
                        else ObservationScope.CONTEXT
                    ),
                )
            )
        return observations

    def _submit_decision_set(self, decision_set: TickExecutionDecisionSet) -> None:
        # The existing engine arbitrates only after every subject was drained.
        self.execution.submit_decision_set(decision_set)
        self._decision_sets.append(decision_set)
        visible_labels: dict[str, set[pd.Timestamp]] = {
            timeframe: set() for timeframe in self._ticks_by_interval
        }
        group_subjects = {
            decision.subject.subject_id for decision in decision_set.decisions
        }
        for item in self._subject_executions:
            if item.subject.subject_id not in group_subjects:
                continue
            for observation in (
                item.visible_inputs.base_observations
                + item.visible_inputs.context_observations
            ):
                for source in observation.timing.source_observations:
                    timeframe = source.provenance.get("timeframe")
                    bar_timestamp = source.provenance.get("bar_timestamp")
                    if timeframe in visible_labels and bar_timestamp is not None:
                        visible_labels[timeframe].add(
                            _as_utc(bar_timestamp, "provenance bar timestamp")
                        )
        for timeframe, buckets in self._ticks_by_interval.items():
            keep = visible_labels[timeframe]
            if self._last_tick is not None:
                keep.add(self._last_tick.timestamp.floor(timeframe))
            for interval_start in tuple(buckets):
                if interval_start not in keep:
                    del buckets[interval_start]


def _row_availability(row: pd.Series) -> pd.Timestamp:
    column = next(
        (name for name in ("available_at", "availability_ts") if name in row.index),
        None,
    )
    if column is None or pd.isna(row[column]):
        raise ValueError("Visible replay bars require explicit availability metadata.")
    return _as_utc(row[column], "bar availability")


def _as_utc(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return timestamp.tz_convert("UTC")


def _plain_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


__all__ = [
    "CoreV1SubjectExecution",
    "CoreV1TickExecution",
    "CoreV1TickExecutionReport",
    "EvidenceEnricher",
    "PipelineBuilder",
    "QuantityProvider",
]
