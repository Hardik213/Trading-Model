from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .ict2022_engine import ICT2022StateMachine, SetupState
from .replay_engine import HistoricalReplay, ReplayConfig, ReplayObservation, ReplayDecision
from .timeframe_context import TimeframeContext, build_context


@dataclass
class ICT2022Pipeline:
    """
    Integration boundary for the full deterministic strategy stack.

    This first integration pass deliberately makes the state-machine handoff
    explicit while leaving setup-context selection to a strategy adapter.
    """

    replay: HistoricalReplay

    @classmethod
    def from_base_data(
        cls,
        data: pd.DataFrame,
        *,
        timeframe: str = "5M",
        source: str = "UNKNOWN",
    ) -> "ICT2022Pipeline":
        context = build_context(
            data,
            base_timeframe=timeframe,
            source=source,
        )
        replay = HistoricalReplay(
            data,
            context=context,
            config=ReplayConfig(
                timeframe=timeframe,
                source=source,
            ),
        )
        return cls(replay=replay)

    def run_observation_pass(self) -> list[ReplayObservation]:
        """
        Run the connected chronological pipeline without fabricating setup
        evidence.

        Until the concrete strategy adapter supplies context/liquidity/MSS/PD
        objects, every timestamp is explicitly classified as DEVELOPING.
        This is intentional: absence of evidence is not evidence of a trade.
        """
        def callback(
            timestamp: pd.Timestamp,
            visible_base: pd.DataFrame,
            visible_context: TimeframeContext,
        ) -> ReplayObservation:
            # This first replay pass intentionally treats every timestamp as
            # a developing observation until a concrete directional setup is
            # supplied by a higher-level strategy adapter. Absence of evidence
            # is not a trade decision here.
            return ReplayObservation(
                timestamp=timestamp,
                decision=ReplayDecision.DEVELOPING,
                state=SetupState.DEVELOPING.value,
                reason="No setup evidence is available yet; the replay remains in a developing state.",
            )

        return self.replay.run(callback)
