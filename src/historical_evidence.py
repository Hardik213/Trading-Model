from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .dealing_range import DealingRange, create_dealing_range
from .fvg import FVG, FVGDirection, detect_fvgs
from .mss import Direction, MSSEvent
from .market_structure import LiquidityEvent, LiquidityLevel, LiquiditySide
from .strategy_adapter import EvidenceProvider, StrategyEvidence
from .causal_events import event_is_confirmed_as_of


@dataclass(frozen=True)
class DetectorBundle:
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
    """Build evidence from the visible replay window with a causal event gate."""

    def __init__(self, detectors: DetectorBundle):
        self.detectors = detectors

    def build(self, timestamp, visible_base, visible_context):
        direction = self.detectors.direction(timestamp, visible_base, visible_context)

        draw = self.detectors.draw_on_liquidity(
            timestamp, visible_base, visible_context, direction
        )
        liquidity_event = self.detectors.liquidity_event(
            timestamp, visible_base, visible_context, direction
        )

        mss = self.detectors.mss(
            timestamp, visible_base, visible_context, direction, liquidity_event
        )
        if mss is not None and not event_is_confirmed_as_of(mss, timestamp):
            mss = None

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
            timestamp, visible_base, visible_context, direction,
            liquidity_event, mss, pd_array
        )
        target = self.detectors.target_price(
            timestamp, visible_base, visible_context, direction,
            draw, liquidity_event
        )
        target_liquidity = self.detectors.target_liquidity(
            timestamp, visible_base, visible_context, direction, target
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
