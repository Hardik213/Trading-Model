from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import pandas as pd

from .dealing_range import DealingRange
from .fvg import FVG, FVGDirection
from .mss import Direction, MSSEvent, normalize_direction
from .market_structure import (
    BreachOutcome,
    LiquidityEvent,
    LiquidityLevel,
    LiquiditySide,
)


class SetupState(str, Enum):
    WAITING = "WAITING"
    CONTEXT_IDENTIFIED = "CONTEXT_IDENTIFIED"
    LIQUIDITY_TARGET_IDENTIFIED = "LIQUIDITY_TARGET_IDENTIFIED"
    LIQUIDITY_BREACHED = "LIQUIDITY_BREACHED"
    RAID_OR_ACCEPTANCE_RESOLUTION = "RAID_OR_ACCEPTANCE_RESOLUTION"
    DISPLACEMENT_CONFIRMED = "DISPLACEMENT_CONFIRMED"
    MSS_CONFIRMED = "MSS_CONFIRMED"
    PD_ARRAY_IDENTIFIED = "PD_ARRAY_IDENTIFIED"
    RETRACEMENT_WAIT = "RETRACEMENT_WAIT"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    ACTIVE_TRADE = "ACTIVE_TRADE"
    TARGET_REACHED = "TARGET_REACHED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    NO_TRADE = "NO_TRADE"
    DEVELOPING = "DEVELOPING"
    AMBIGUOUS_DATA = "AMBIGUOUS_DATA"


@dataclass(frozen=True)
class SetupContext:
    as_of: pd.Timestamp
    direction: Direction
    dealing_range: Optional[DealingRange]
    draw_on_liquidity: Optional[LiquidityLevel]
    liquidity_event: Optional[LiquidityEvent]
    mss: Optional[MSSEvent]
    pd_array: Optional[FVG]
    invalidation_price: Optional[float]
    target_price: Optional[float]
    target_liquidity: Optional[LiquidityLevel]


@dataclass(frozen=True)
class SetupDecision:
    state: SetupState
    direction: Optional[Direction]
    reason: str
    context: Optional[SetupContext] = None


@dataclass
class ICT2022StateMachine:
    """
    Conservative deterministic gate for the ICT-2022 sequence.

    This object deliberately does not invent missing context. Each transition
    must be supported by an observation supplied by upstream engines.
    """

    state: SetupState = SetupState.WAITING
    history: list[SetupState] = field(default_factory=lambda: [SetupState.WAITING])

    def _set(self, state: SetupState) -> SetupState:
        self.state = state
        self.history.append(state)
        return state

    def reset(self) -> None:
        self.state = SetupState.WAITING
        self.history = [SetupState.WAITING]

    def identify_context(
        self,
        *,
        as_of: pd.Timestamp,
        direction: Optional[Direction],
        dealing_range: Optional[DealingRange],
        draw_on_liquidity: Optional[LiquidityLevel],
    ) -> SetupDecision:
        if direction is None:
            self._set(SetupState.NO_TRADE)
            return SetupDecision(
                self.state, None,
                "No directional hypothesis is established."
            )

        direction = normalize_direction(direction)

        self._set(SetupState.CONTEXT_IDENTIFIED)
        if draw_on_liquidity is None:
            self._set(SetupState.DEVELOPING)
            return SetupDecision(
                self.state, direction,
                "Context exists, but no draw-on-liquidity objective is identified."
            )

        self._set(SetupState.LIQUIDITY_TARGET_IDENTIFIED)
        return SetupDecision(
            self.state, direction,
            "Directional context and draw-on-liquidity objective identified."
        )

    def process_liquidity_event(
        self,
        event: Optional[LiquidityEvent],
        *,
        direction: Direction,
    ) -> SetupDecision:
        if self.state not in {
            SetupState.LIQUIDITY_TARGET_IDENTIFIED,
            SetupState.CONTEXT_IDENTIFIED,
            SetupState.DEVELOPING,
        }:
            return SetupDecision(
                self.state, direction,
                "Liquidity event arrived in an invalid state-machine phase."
            )

        if event is None:
            self._set(SetupState.DEVELOPING)
            return SetupDecision(
                self.state, direction,
                "Relevant liquidity has not been breached."
            )

        self._set(SetupState.LIQUIDITY_BREACHED)

        if event.outcome is BreachOutcome.UNRESOLVED:
            self._set(SetupState.DEVELOPING)
            return SetupDecision(
                self.state, direction,
                "Liquidity was breached but post-breach behaviour is unresolved."
            )

        self._set(SetupState.RAID_OR_ACCEPTANCE_RESOLUTION)

        expected_rejection_side = (
            LiquiditySide.SSL if normalize_direction(direction) is Direction.BULLISH
            else LiquiditySide.BSL
        )

        if event.outcome is not BreachOutcome.REJECTION:
            self._set(SetupState.NO_TRADE)
            return SetupDecision(
                self.state, direction,
                "Liquidity was accepted rather than rejected; the reversal sequence is not confirmed."
            )

        if event.level.side is not expected_rejection_side:
            self._set(SetupState.NO_TRADE)
            return SetupDecision(
                self.state, direction,
                "The rejected liquidity side does not oppose the proposed reversal."
            )

        return SetupDecision(
            self.state, direction,
            "Relevant opposing liquidity was breached and rejected."
        )

    def confirm_displacement_and_mss(
        self,
        *,
        mss: Optional[MSSEvent],
        direction: Direction,
    ) -> SetupDecision:
        if self.state is not SetupState.RAID_OR_ACCEPTANCE_RESOLUTION:
            return SetupDecision(
                self.state, direction,
                "MSS cannot be accepted before a resolved liquidity rejection."
            )

        if mss is None:
            self._set(SetupState.DEVELOPING)
            return SetupDecision(
                self.state, direction,
                "Displacement/MSS confirmation is incomplete."
            )

        if normalize_direction(mss.direction) is not normalize_direction(direction):
            self._set(SetupState.NO_TRADE)
            return SetupDecision(
                self.state, direction,
                "Structural shift direction conflicts with the proposed setup."
            )

        if mss.displacement is None or not mss.displacement.follow_through or not mss.follow_through:
            self._set(SetupState.DEVELOPING)
            return SetupDecision(
                self.state, direction,
                "Structural break lacks required follow-through."
            )

        self._set(SetupState.DISPLACEMENT_CONFIRMED)
        self._set(SetupState.MSS_CONFIRMED)
        return SetupDecision(
            self.state, direction,
            "Displacement and meaningful structural shift are confirmed."
        )

    def identify_pd_array(
        self,
        *,
        pd_array: Optional[FVG],
        direction: Direction,
    ) -> SetupDecision:
        if self.state is not SetupState.MSS_CONFIRMED:
            return SetupDecision(
                self.state, direction,
                "PD-array search is only valid after MSS confirmation."
            )

        if pd_array is None:
            self._set(SetupState.NO_TRADE)
            return SetupDecision(
                self.state, direction,
                "No justified directional PD array is available."
            )

        expected = (
            FVGDirection.BULLISH if normalize_direction(direction) is Direction.BULLISH
            else FVGDirection.BEARISH
        )
        if pd_array.direction is not expected:
            self._set(SetupState.NO_TRADE)
            return SetupDecision(
                self.state, direction,
                "PD-array direction conflicts with the confirmed MSS."
            )

        self._set(SetupState.PD_ARRAY_IDENTIFIED)
        self._set(SetupState.RETRACEMENT_WAIT)
        return SetupDecision(
            self.state, direction,
            "Directional FVG/PD array identified; awaiting retracement."
        )

    def evaluate_retracement(
        self,
        *,
        current_price: float,
        pd_array: FVG,
        direction: Direction,
    ) -> SetupDecision:
        if self.state is not SetupState.RETRACEMENT_WAIT:
            return SetupDecision(
                self.state, direction,
                "Retracement cannot be evaluated before PD-array identification."
            )

        if not pd_array.contains(current_price):
            return SetupDecision(
                self.state, direction,
                "Price has not retraced into the identified PD array."
            )

        self._set(SetupState.ENTRY_TRIGGERED)
        return SetupDecision(
            self.state, direction,
            "Price has retraced into the identified PD array."
        )

    def activate_trade(
        self,
        *,
        entry: float,
        invalidation: Optional[float],
        target: Optional[float],
        direction: Direction,
    ) -> SetupDecision:
        if self.state is not SetupState.ENTRY_TRIGGERED:
            return SetupDecision(
                self.state, direction,
                "Trade cannot activate before an entry trigger."
            )

        if invalidation is None or target is None:
            self._set(SetupState.NO_TRADE)
            return SetupDecision(
                self.state, direction,
                "No structural invalidation and/or opposing-liquidity target is defined."
            )

        if normalize_direction(direction) is Direction.BULLISH:
            valid_geometry = invalidation < entry < target
        else:
            valid_geometry = target < entry < invalidation

        if not valid_geometry:
            self._set(SetupState.NO_TRADE)
            return SetupDecision(
                self.state, direction,
                "Entry, invalidation and target geometry is invalid."
            )

        self._set(SetupState.ACTIVE_TRADE)
        return SetupDecision(
            self.state, direction,
            "Trade is active with explicit structural invalidation and target."
        )

    def resolve_trade(
        self,
        *,
        current_high: float,
        current_low: float,
        invalidation: float,
        target: float,
        direction: Direction,
    ) -> SetupDecision:
        if self.state is not SetupState.ACTIVE_TRADE:
            return SetupDecision(
                self.state, direction,
                "No active trade exists."
            )

        if normalize_direction(direction) is Direction.BULLISH:
            hit_stop = current_low <= invalidation
            hit_target = current_high >= target
        else:
            hit_stop = current_high >= invalidation
            hit_target = current_low <= target

        if hit_stop and hit_target:
            self._set(SetupState.AMBIGUOUS_DATA)
            return SetupDecision(
                self.state, direction,
                "Both invalidation and target were touched in the same unresolved bar."
            )

        if hit_stop:
            self._set(SetupState.INVALIDATED)
            return SetupDecision(
                self.state, direction,
                "Structural invalidation was reached."
            )

        if hit_target:
            self._set(SetupState.TARGET_REACHED)
            return SetupDecision(
                self.state, direction,
                "Opposing-liquidity target was reached."
            )

        return SetupDecision(
            self.state, direction,
            "Trade remains active."
        )
