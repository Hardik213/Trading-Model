from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .replay_engine import HistoricalReplay, ReplayObservation, ReplayDecision
from .strategy_adapter import EvidenceProvider, ICT2022StrategyAdapter, AdapterDecision


@dataclass(frozen=True)
class StrategyReplayResult:
    observations: tuple[AdapterDecision, ...]

    @property
    def valid(self) -> tuple[AdapterDecision, ...]:
        return tuple(x for x in self.observations if x.state.value == "ACTIVE_TRADE")

    @property
    def no_trade(self) -> tuple[AdapterDecision, ...]:
        return tuple(x for x in self.observations if x.state.value == "NO_TRADE")

    @property
    def developing(self) -> tuple[AdapterDecision, ...]:
        return tuple(x for x in self.observations if x.state.value == "DEVELOPING")


class ICT2022ReplayRunner:
    """
    Single-unit replay orchestration:

        historical replay -> evidence provider -> ICT-2022 adapter

    This layer deliberately stops at setup qualification. Trade execution and
    event-driven outcome simulation remain the responsibility of Phase 5.
    """

    def __init__(
        self,
        replay: HistoricalReplay,
        *,
        provider: EvidenceProvider,
        adapter: ICT2022StrategyAdapter | None = None,
    ) -> None:
        self.replay = replay
        self.provider = provider
        self.adapter = adapter or ICT2022StrategyAdapter()

    def run(self) -> StrategyReplayResult:
        decisions: list[AdapterDecision] = []

        def callback(timestamp, visible_base, visible_context):
            evidence = self.provider.build(
                timestamp,
                visible_base,
                visible_context,
            )
            if evidence.timestamp != timestamp:
                raise ValueError(
                    "Evidence provider returned a timestamp different from replay time."
                )

            decision = self.adapter.evaluate(evidence)
            decisions.append(decision)

            return ReplayObservation(
                timestamp=timestamp,
                decision=(
                    ReplayDecision.VALID
                    if decision.state.value == "ACTIVE_TRADE"
                    else (
                        ReplayDecision.NO_TRADE
                        if decision.state.value == "NO_TRADE"
                        else ReplayDecision.DEVELOPING
                    )
                ),
                state=decision.state.value,
                reason=decision.reason,
            )

        self.replay.run(callback)
        return StrategyReplayResult(tuple(decisions))
