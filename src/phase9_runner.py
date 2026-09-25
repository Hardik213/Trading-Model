from __future__ import annotations

from dataclasses import dataclass

from .historical_evidence import HistoricalEvidenceProvider
from .replay_engine import HistoricalReplay
from .replay_strategy import ICT2022ReplayRunner, StrategyReplayResult
from .strategy_adapter import ICT2022StrategyAdapter


@dataclass(frozen=True)
class Phase9Run:
    result: StrategyReplayResult
    provider: HistoricalEvidenceProvider


def run_historical_strategy(
    replay: HistoricalReplay,
    *,
    provider: HistoricalEvidenceProvider,
) -> Phase9Run:
    """
    Execute the complete chronology -> evidence -> state-machine path.

    The function deliberately stops before execution/backtest outcome
    simulation. That remains Phase 5's responsibility.
    """
    result = ICT2022ReplayRunner(
        replay,
        provider=provider,
        adapter=ICT2022StrategyAdapter(),
    ).run()

    return Phase9Run(result=result, provider=provider)
