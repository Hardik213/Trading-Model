from __future__ import annotations

"""Chronological XAU/USD historical replay for the causal ICT-2022 gate.

This module is deliberately an orchestration layer. It never predicts direction,
manufactures a setup, or uses future candles. At each decision timestamp it gives
the supplied evidence builder only the rows that were visible at that timestamp,
then evaluates the already-built causal precision gate.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .data_contract import normalize_ohlc
from .replay_engine import HistoricalReplay, ReplayConfig
from .sniper_setup import PrecisionDecision, PrecisionEvidence, evaluate_precision_setup
from .timeframe_context import TimeframeContext, build_context


EvidenceBuilder = Callable[
    [pd.Timestamp, pd.DataFrame, TimeframeContext], PrecisionEvidence
]


@dataclass(frozen=True)
class PrecisionReplayObservation:
    timestamp: pd.Timestamp
    state: str
    reason: str
    detail: str
    planned_r: Optional[float]


@dataclass(frozen=True)
class PrecisionReplayResult:
    observations: tuple[PrecisionReplayObservation, ...]

    @property
    def valid(self) -> tuple[PrecisionReplayObservation, ...]:
        return tuple(x for x in self.observations if x.state == "VALID")

    @property
    def developing(self) -> tuple[PrecisionReplayObservation, ...]:
        return tuple(x for x in self.observations if x.state == "DEVELOPING")

    @property
    def invalid(self) -> tuple[PrecisionReplayObservation, ...]:
        return tuple(x for x in self.observations if x.state == "INVALID")

    @property
    def no_trade(self) -> tuple[PrecisionReplayObservation, ...]:
        return tuple(x for x in self.observations if x.state == "NO_TRADE")

    def state_counts(self) -> dict[str, int]:
        counts = {"VALID": 0, "DEVELOPING": 0, "INVALID": 0, "NO_TRADE": 0}
        for item in self.observations:
            counts[item.state] = counts.get(item.state, 0) + 1
        return counts


class HistoricalPrecisionReplay:
    """Run the Phase 10.3 precision gate chronologically over historical data.

    The evidence builder is the only component allowed to construct strategy
    evidence. The replay engine controls visibility, so this class cannot widen
    the information set for a timestamp.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        evidence_builder: EvidenceBuilder,
        context: Optional[TimeframeContext] = None,
        config: Optional[ReplayConfig] = None,
        min_planned_r: Optional[float] = None,
    ) -> None:
        self.config = config or ReplayConfig()
        normalized = normalize_ohlc(
            data,
            timeframe=self.config.timeframe,
            source=self.config.source,
        )
        self.replay = HistoricalReplay(
            normalized,
            context=context or build_context(
                normalized,
                base_timeframe=self.config.timeframe,
                source=self.config.source,
            ),
            config=self.config,
        )
        self.evidence_builder = evidence_builder
        self.min_planned_r = min_planned_r

    def run(self) -> PrecisionReplayResult:
        observations: list[PrecisionReplayObservation] = []

        def callback(timestamp, visible_base, visible_context):
            evidence = self.evidence_builder(
                timestamp,
                visible_base,
                visible_context,
            )
            if pd.Timestamp(evidence.as_of) != pd.Timestamp(timestamp):
                raise ValueError(
                    "Evidence builder returned evidence for a different timestamp."
                )
            decision: PrecisionDecision = evaluate_precision_setup(
                evidence,
                min_planned_r=self.min_planned_r,
            )
            observations.append(
                PrecisionReplayObservation(
                    timestamp=timestamp,
                    state=decision.state.value,
                    reason=decision.reason.value,
                    detail=decision.detail,
                    planned_r=decision.planned_r,
                )
            )
            # The replay engine only needs a timestamp-validated observation.
            from .replay_engine import ReplayObservation, ReplayDecision
            return ReplayObservation(
                timestamp=timestamp,
                decision=ReplayDecision(decision.state.value),
                state=decision.state.value,
                reason=decision.reason.value,
            )

        self.replay.run(callback)
        return PrecisionReplayResult(tuple(observations))
