from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import pandas as pd

from .dealing_range import DealingRange, create_dealing_range
from .fvg import FVG, FVGDirection, detect_fvgs
from .mss import Direction, MSSEvent
from .market_structure import LiquidityEvent, LiquidityLevel, LiquiditySide
from .strategy_adapter import EvidenceProvider, StrategyEvidence


@dataclass(frozen=True)
class DetectorBundle:
    """
    Integration contract for the existing Phase 1-3 detector implementations.

    The historical provider owns chronology and evidence assembly. Existing
    detectors remain the authority for their own observations.

    This bundle is intentionally explicit so a detector cannot quietly obtain
    future data from global state or a full unbounded dataframe.
    """

    direction: callable
    draw_on_liquidity: callable
    liquidity_event: callable
    mss: callable
    pd_array: callable
    dealing_range: callable
    entry_price: callable
    invalidation_price: callable
    target_price: callable
    target_liquidity: callable


class HistoricalEvidenceProvider(EvidenceProvider):
    """
    Builds StrategyEvidence from the visible replay window.

    Every detector receives only the as-of visible data supplied by replay.
    The provider does not call a detector with the full historical dataset.

    The provider is deliberately strict: if a detector cannot establish a
    required observation, the corresponding field remains None.
    """

    def __init__(self, detectors: DetectorBundle):
        self.detectors = detectors

    def build(
        self,
        timestamp: pd.Timestamp,
        visible_base: pd.DataFrame,
        visible_context,
    ) -> StrategyEvidence:
        direction = self.detectors.direction(
            timestamp, visible_base, visible_context
        )

        draw = self.detectors.draw_on_liquidity(
            timestamp, visible_base, visible_context, direction
        )

        liquidity_event = self.detectors.liquidity_event(
            timestamp, visible_base, visible_context, direction
        )

        mss = self.detectors.mss(
            timestamp, visible_base, visible_context, direction, liquidity_event
        )

        pd_array = self.detectors.pd_array(
            timestamp, visible_base, visible_context, direction, mss
        )

        dealing_range = self.detectors.dealing_range(
            timestamp, visible_base, visible_context, direction
        )

        entry = self.detectors.entry_price(
            timestamp, visible_base, visible_context, direction, pd_array, mss
        )

        invalidation = self.detectors.invalidation_price(
            timestamp,
            visible_base,
            visible_context,
            direction,
            liquidity_event,
            mss,
            pd_array,
        )

        target = self.detectors.target_price(
            timestamp,
            visible_base,
            visible_context,
            direction,
            draw,
            liquidity_event,
        )

        target_liquidity = self.detectors.target_liquidity(
            timestamp,
            visible_base,
            visible_context,
            direction,
            target,
        )

        return StrategyEvidence(
            timestamp=timestamp,
            direction=direction,
            dealing_range=dealing_range,
            draw_on_liquidity=draw,
            liquidity_event=liquidity_event,
            mss=mss,
            pd_array=pd_array,
            entry_price=entry,
            invalidation_price=invalidation,
            target_price=target,
            target_liquidity=target_liquidity,
        )
