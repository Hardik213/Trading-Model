from __future__ import annotations

"""Causal ICT-2022 precision setup gate.

This module is intentionally a *gate*, not a predictor. It accepts only evidence
that has already been established as visible at ``as_of`` and verifies the
canonical sequence:

context/draw -> opposing liquidity rejection -> displacement/MSS confirmation
-> directional FVG/PD array -> retracement -> structural invalidation ->
opposing-liquidity target -> valid geometry.

It does not invent directional bias, institutional intent, or probabilities.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from .fvg import FVG, FVGDirection
from .market_structure import BreachOutcome, LiquidityEvent, LiquidityLevel, LiquiditySide
from .mss import Direction, MSSEvent, normalize_direction


class PrecisionState(str, Enum):
    DEVELOPING = "DEVELOPING"
    INVALID = "INVALID"
    VALID = "VALID"
    NO_TRADE = "NO_TRADE"


class GateReason(str, Enum):
    MISSING_DIRECTION = "MISSING_DIRECTION"
    MISSING_DRAW = "MISSING_DRAW"
    DRAW_SIDE_MISMATCH = "DRAW_SIDE_MISMATCH"
    MISSING_LIQUIDITY_REJECTION = "MISSING_LIQUIDITY_REJECTION"
    LIQUIDITY_ACCEPTED = "LIQUIDITY_ACCEPTED"
    LIQUIDITY_SIDE_MISMATCH = "LIQUIDITY_SIDE_MISMATCH"
    FUTURE_LIQUIDITY_EVENT = "FUTURE_LIQUIDITY_EVENT"
    MISSING_MSS = "MISSING_MSS"
    FUTURE_MSS_CONFIRMATION = "FUTURE_MSS_CONFIRMATION"
    FUTURE_DISPLACEMENT_CONFIRMATION = "FUTURE_DISPLACEMENT_CONFIRMATION"
    MSS_DIRECTION_MISMATCH = "MSS_DIRECTION_MISMATCH"
    MISSING_DISPLACEMENT_FOLLOW_THROUGH = "MISSING_DISPLACEMENT_FOLLOW_THROUGH"
    MISSING_MSS_FOLLOW_THROUGH = "MISSING_MSS_FOLLOW_THROUGH"
    MISSING_PD_ARRAY = "MISSING_PD_ARRAY"
    PD_DIRECTION_MISMATCH = "PD_DIRECTION_MISMATCH"
    FUTURE_PD_ARRAY = "FUTURE_PD_ARRAY"
    PD_ARRAY_PRECEDES_MSS = "PD_ARRAY_PRECEDES_MSS"
    RETRACEMENT_NOT_IN_PD_ARRAY = "RETRACEMENT_NOT_IN_PD_ARRAY"
    MISSING_INVALIDATION = "MISSING_INVALIDATION"
    MISSING_TARGET = "MISSING_TARGET"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    TARGET_NOT_DRAW = "TARGET_NOT_DRAW"
    R_MULTIPLE_TOO_LOW = "R_MULTIPLE_TOO_LOW"
    VALID_SEQUENCE = "VALID_SEQUENCE"


@dataclass(frozen=True)
class PrecisionEvidence:
    as_of: pd.Timestamp
    direction: Optional[Direction]
    draw_on_liquidity: Optional[LiquidityLevel]
    liquidity_event: Optional[LiquidityEvent]
    mss: Optional[MSSEvent]
    pd_array: Optional[FVG]
    entry_price: Optional[float]
    invalidation_price: Optional[float]
    target_price: Optional[float]
    target_liquidity: Optional[LiquidityLevel]


@dataclass(frozen=True)
class PrecisionDecision:
    state: PrecisionState
    reason: GateReason
    detail: str
    planned_r: Optional[float] = None

    @property
    def valid(self) -> bool:
        return self.state is PrecisionState.VALID


def _ts(value) -> pd.Timestamp:
    return pd.Timestamp(value)


def _visible(value, as_of: pd.Timestamp) -> bool:
    return value is not None and _ts(value) <= as_of


def _strictly_visible(value, as_of: pd.Timestamp) -> bool:
    return value is not None and _ts(value) < as_of


def _expected_sides(direction: Direction):
    d = normalize_direction(direction)
    if d is Direction.BULLISH:
        return LiquiditySide.SSL, LiquiditySide.BSL, FVGDirection.BULLISH
    return LiquiditySide.BSL, LiquiditySide.SSL, FVGDirection.BEARISH


def _planned_r(direction: Direction, entry: float, invalidation: float, target: float) -> Optional[float]:
    risk = abs(entry - invalidation)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    d = normalize_direction(direction)
    if d is Direction.BULLISH and not invalidation < entry < target:
        return None
    if d is Direction.BEARISH and not target < entry < invalidation:
        return None
    return reward / risk


def evaluate_precision_setup(
    evidence: PrecisionEvidence,
    *,
    min_planned_r: Optional[float] = None,
) -> PrecisionDecision:
    """Evaluate one timestamp without consulting future candles.

    ``min_planned_r`` is deliberately optional because the strategy constitution
    requires experimental approval before introducing a new R:R threshold.
    """
    as_of = _ts(evidence.as_of)
    direction = evidence.direction
    if direction is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_DIRECTION,
                                 "No directional hypothesis has been established.")
    direction = normalize_direction(direction)
    reversal_side, target_side, fvg_direction = _expected_sides(direction)

    draw = evidence.draw_on_liquidity
    if draw is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_DRAW,
                                 "No visible opposing-liquidity objective exists.")
    if draw.side is not target_side:
        return PrecisionDecision(PrecisionState.INVALID, GateReason.DRAW_SIDE_MISMATCH,
                                 "Draw-on-liquidity is not on the opposing side of the setup.")
    if not _visible(getattr(draw, "timestamp", None), as_of):
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.FUTURE_LIQUIDITY_EVENT,
                                 "Draw-on-liquidity is not yet visible at the decision timestamp.")

    event = evidence.liquidity_event
    if event is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_LIQUIDITY_REJECTION,
                                 "The required opposing-liquidity event has not been established.")
    if event.outcome is BreachOutcome.ACCEPTANCE:
        return PrecisionDecision(PrecisionState.INVALID, GateReason.LIQUIDITY_ACCEPTED,
                                 "Liquidity was accepted rather than rejected; reversal sequence is invalid.")
    if event.outcome is not BreachOutcome.REJECTION:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_LIQUIDITY_REJECTION,
                                 "Liquidity breach remains unresolved.")
    if event.level.side is not reversal_side:
        return PrecisionDecision(PrecisionState.INVALID, GateReason.LIQUIDITY_SIDE_MISMATCH,
                                 "Rejected liquidity does not oppose the proposed reversal.")
    if not _visible(event.resolution_timestamp, as_of):
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.FUTURE_LIQUIDITY_EVENT,
                                 "Liquidity rejection confirmation is not yet visible.")

    mss = evidence.mss
    if mss is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_MSS,
                                 "Meaningful structural shift has not been confirmed.")
    if normalize_direction(mss.direction) is not direction:
        return PrecisionDecision(PrecisionState.INVALID, GateReason.MSS_DIRECTION_MISMATCH,
                                 "MSS direction conflicts with the setup direction.")
    if not mss.follow_through:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_MSS_FOLLOW_THROUGH,
                                 "MSS lacks required follow-through.")
    if mss.confirmation_timestamp is None or not _visible(mss.confirmation_timestamp, as_of):
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.FUTURE_MSS_CONFIRMATION,
                                 "MSS confirmation is not yet visible at this timestamp.")
    displacement = mss.displacement
    if displacement is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_MSS,
                                 "MSS has no associated displacement evidence.")
    if not displacement.follow_through:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_DISPLACEMENT_FOLLOW_THROUGH,
                                 "Displacement lacks required follow-through.")
    if displacement.confirmation_timestamp is None or not _strictly_visible(displacement.confirmation_timestamp, as_of):
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.FUTURE_DISPLACEMENT_CONFIRMATION,
                                 "Displacement confirmation is not yet visible at this timestamp.")

    fvg = evidence.pd_array
    if fvg is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_PD_ARRAY,
                                 "No directional PD array is available after MSS.")
    if fvg.direction is not fvg_direction:
        return PrecisionDecision(PrecisionState.INVALID, GateReason.PD_DIRECTION_MISMATCH,
                                 "PD array direction conflicts with the confirmed MSS.")
    if not _visible(fvg.confirmation_timestamp, as_of):
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.FUTURE_PD_ARRAY,
                                 "PD array confirmation is not yet visible.")
    if _ts(fvg.formation_timestamp) < _ts(mss.timestamp):
        return PrecisionDecision(PrecisionState.INVALID, GateReason.PD_ARRAY_PRECEDES_MSS,
                                 "The supplied PD array predates the confirmed structural shift and is not the post-MSS array for this setup.")
    if evidence.entry_price is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.RETRACEMENT_NOT_IN_PD_ARRAY,
                                 "No entry/retracement price is available.")
    if not fvg.contains(float(evidence.entry_price)):
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.RETRACEMENT_NOT_IN_PD_ARRAY,
                                 "Price has not retraced into the justified PD array.")

    if evidence.invalidation_price is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_INVALIDATION,
                                 "Structural invalidation is not defined.")
    if evidence.target_price is None:
        return PrecisionDecision(PrecisionState.DEVELOPING, GateReason.MISSING_TARGET,
                                 "Opposing-liquidity target is not defined.")
    if evidence.target_liquidity is None or evidence.target_liquidity.side is not target_side:
        return PrecisionDecision(PrecisionState.INVALID, GateReason.TARGET_NOT_DRAW,
                                 "Target is not tied to visible opposing liquidity.")

    planned_r = _planned_r(direction, float(evidence.entry_price),
                           float(evidence.invalidation_price), float(evidence.target_price))
    if planned_r is None:
        return PrecisionDecision(PrecisionState.INVALID, GateReason.INVALID_GEOMETRY,
                                 "Entry, structural invalidation and target do not form valid geometry.")
    if min_planned_r is not None and planned_r < min_planned_r:
        return PrecisionDecision(PrecisionState.NO_TRADE, GateReason.R_MULTIPLE_TOO_LOW,
                                 f"Planned R multiple {planned_r:.4f} is below the configured threshold {min_planned_r:.4f}.",
                                 planned_r=planned_r)

    return PrecisionDecision(PrecisionState.VALID, GateReason.VALID_SEQUENCE,
                             "All causal ICT-2022 precision gates are satisfied.", planned_r=planned_r)


__all__ = [
    "GateReason",
    "PrecisionDecision",
    "PrecisionEvidence",
    "PrecisionState",
    "evaluate_precision_setup",
]
