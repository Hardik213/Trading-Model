from __future__ import annotations

"""Chronological liquidity-event scanner for historical XAU/USD research.

This scanner is intentionally narrower than the full setup gate. It answers one
empirical question: what liquidity evidence was actually visible at each replay
timestamp? It does not call a breach a sweep until the existing causal resolver
has classified the post-breach behaviour.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .causal_liquidity import (
    CausalLiquidityEvidence,
    LiquidityEvidenceKind,
    confirmed_liquidity_map,
    draw_on_liquidity,
    liquidity_evidence_as_of,
)
from .data_contract import normalize_ohlc
from .replay_engine import ReplayConfig


@dataclass(frozen=True)
class LiquidityReplayObservation:
    timestamp: pd.Timestamp
    visible_levels: int
    resolved_rejections: int
    resolved_acceptances: int
    draw_bullish_price: Optional[float]
    draw_bearish_price: Optional[float]
    latest_rejection_side: Optional[str]
    latest_rejection_timestamp: Optional[pd.Timestamp]


@dataclass(frozen=True)
class LiquidityReplayResult:
    observations: tuple[LiquidityReplayObservation, ...]

    @property
    def rejection_observations(self) -> tuple[LiquidityReplayObservation, ...]:
        return tuple(x for x in self.observations if x.resolved_rejections)

    @property
    def acceptance_observations(self) -> tuple[LiquidityReplayObservation, ...]:
        return tuple(x for x in self.observations if x.resolved_acceptances)


class HistoricalLiquidityReplay:
    """Scan a normalized OHLC series in strict chronological order."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        config: Optional[ReplayConfig] = None,
        left_bars: int = 2,
        right_bars: int = 2,
        resolution_bars: int = 3,
        equal_tolerance: float = 0.0,
    ) -> None:
        self.config = config or ReplayConfig()
        self.data = normalize_ohlc(
            data,
            timeframe=self.config.timeframe,
            source=self.config.source,
        )
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.resolution_bars = resolution_bars
        self.equal_tolerance = equal_tolerance

    def decision_times(self) -> pd.DatetimeIndex:
        idx = self.data.index
        if self.config.start is not None:
            idx = idx[idx >= self.config.start]
        if self.config.end is not None:
            idx = idx[idx <= self.config.end]
        return idx

    def run(self) -> LiquidityReplayResult:
        records: list[LiquidityReplayObservation] = []
        for timestamp in self.decision_times():
            visible = self.data.loc[self.data.index <= timestamp].copy()
            levels = confirmed_liquidity_map(
                self.data,
                timestamp,
                left_bars=self.left_bars,
                right_bars=self.right_bars,
                equal_tolerance=self.equal_tolerance,
            )
            evidence = liquidity_evidence_as_of(
                self.data,
                timestamp,
                left_bars=self.left_bars,
                right_bars=self.right_bars,
                resolution_bars=self.resolution_bars,
                equal_tolerance=self.equal_tolerance,
            )
            rejections = [x for x in evidence if x.kind is LiquidityEvidenceKind.REJECTION]
            acceptances = [x for x in evidence if x.kind is LiquidityEvidenceKind.ACCEPTANCE]
            current_price = float(visible.iloc[-1]["Close"])
            bullish_draw = draw_on_liquidity(
                self.data,
                timestamp,
                "BULLISH",
                current_price=current_price,
                left_bars=self.left_bars,
                right_bars=self.right_bars,
                equal_tolerance=self.equal_tolerance,
            )
            bearish_draw = draw_on_liquidity(
                self.data,
                timestamp,
                "BEARISH",
                current_price=current_price,
                left_bars=self.left_bars,
                right_bars=self.right_bars,
                equal_tolerance=self.equal_tolerance,
            )
            latest = max(
                rejections,
                key=lambda x: pd.Timestamp(
                    x.breach.resolution_timestamp or x.breach.breach_timestamp
                ),
                default=None,
            )
            records.append(
                LiquidityReplayObservation(
                    timestamp=timestamp,
                    visible_levels=len(levels),
                    resolved_rejections=len(rejections),
                    resolved_acceptances=len(acceptances),
                    draw_bullish_price=(bullish_draw.level.price if bullish_draw else None),
                    draw_bearish_price=(bearish_draw.level.price if bearish_draw else None),
                    latest_rejection_side=(latest.level.side.value if latest else None),
                    latest_rejection_timestamp=(
                        latest.breach.resolution_timestamp if latest else None
                    ),
                )
            )
        return LiquidityReplayResult(tuple(records))

    @staticmethod
    def event_rows(result: LiquidityReplayResult) -> list[dict]:
        """Return flat rows suitable for CSV/JSON research export."""
        return [
            {
                "timestamp": x.timestamp.isoformat(),
                "visible_levels": x.visible_levels,
                "resolved_rejections": x.resolved_rejections,
                "resolved_acceptances": x.resolved_acceptances,
                "draw_bullish_price": x.draw_bullish_price,
                "draw_bearish_price": x.draw_bearish_price,
                "latest_rejection_side": x.latest_rejection_side,
                "latest_rejection_timestamp": (
                    x.latest_rejection_timestamp.isoformat()
                    if x.latest_rejection_timestamp is not None
                    else None
                ),
            }
            for x in result.observations
        ]
