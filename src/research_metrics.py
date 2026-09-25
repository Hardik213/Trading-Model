from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .event_backtester import TradeOutcome, TradeResult


@dataclass(frozen=True)
class PerformanceSummary:
    n_resolved: int
    n_target: int
    n_stop: int
    n_ambiguous: int
    n_expired: int
    win_rate: float | None
    expectancy_r: float | None
    median_r: float | None
    profit_factor: float | None
    max_drawdown_r: float | None


def summarize_results(results: Iterable[TradeResult]) -> PerformanceSummary:
    results = list(results)

    resolved = [
        r for r in results
        if r.outcome in {TradeOutcome.TARGET, TradeOutcome.STOP}
        and r.net_r is not None
    ]
    wins = [r.net_r for r in resolved if r.outcome is TradeOutcome.TARGET]
    losses = [r.net_r for r in resolved if r.outcome is TradeOutcome.STOP]

    values = np.array([r.net_r for r in resolved], dtype=float)
    expectancy = float(values.mean()) if len(values) else None
    median = float(np.median(values)) if len(values) else None
    win_rate = len(wins) / len(resolved) if resolved else None

    gross_profit = sum(x for x in wins if x > 0)
    gross_loss = abs(sum(x for x in losses if x < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else None
    )

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    return PerformanceSummary(
        n_resolved=len(resolved),
        n_target=len(wins),
        n_stop=len(losses),
        n_ambiguous=sum(r.outcome is TradeOutcome.AMBIGUOUS for r in results),
        n_expired=sum(r.outcome is TradeOutcome.EXPIRED for r in results),
        win_rate=win_rate,
        expectancy_r=expectancy,
        median_r=median,
        profit_factor=profit_factor,
        max_drawdown_r=abs(max_dd),
    )
