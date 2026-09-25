from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from .dealing_range import DealingRange
from .fvg import FVG
from .ict2022_engine import ICT2022StateMachine, SetupContext, SetupDecision, SetupState
from .mss import Direction, MSSEvent
from .market_structure import LiquidityEvent, LiquidityLevel


@dataclass(frozen=True)
class StrategyEvidence:
    """
    Evidence assembled for one replay timestamp.

    Every field represents information that the caller has already established
    from data available at `timestamp`. The adapter never searches future bars
    or manufactures missing evidence.
    """
    timestamp: pd.Timestamp
    direction: Optional[Direction]
    dealing_range: Optional[DealingRange]
    draw_on_liquidity: Optional[LiquidityLevel]
    liquidity_event: Optional[LiquidityEvent]
    mss: Optional[MSSEvent]
    pd_array: Optional[FVG]
    entry_price: Optional[float]
    invalidation_price: Optional[float]
    target_price: Optional[float]
    target_liquidity: Optional[LiquidityLevel]


@dataclass(frozen=True)
class AdapterDecision:
    timestamp: pd.Timestamp
    state: SetupState
    direction: Optional[Direction]
    reason: str
    entry_price: Optional[float] = None
    invalidation_price: Optional[float] = None
    target_price: Optional[float] = None
    planned_r: Optional[float] = None
    context: Optional[SetupContext] = None


def _planned_r(
    *,
    direction: Direction,
    entry: float,
    invalidation: float,
    target: float,
) -> Optional[float]:
    risk = abs(entry - invalidation)
    reward = abs(target - entry)
    if risk <= 0:
        return None

    if direction is Direction.BULLISH:
        if not invalidation < entry < target:
            return None
    else:
        if not target < entry < invalidation:
            return None

    return reward / risk


class ICT2022StrategyAdapter:
    """
    Connects the evidence layer to the ICT-2022 state machine.

    The adapter is intentionally an orchestration layer, not a second strategy
    engine. Structure/liquidity/displacement/MSS/FVG detectors produce evidence;
    this adapter orders that evidence through the state machine.

    One timestamp is evaluated once. If a mandatory stage is missing, the
    result remains DEVELOPING or NO_TRADE rather than being upgraded by a
    later observation.
    """

    def evaluate(self, evidence: StrategyEvidence) -> AdapterDecision:
        # Absence of directional evidence means the setup is still developing.
        # It is not a rejected setup.
        if evidence.direction is None:
            return AdapterDecision(
                timestamp=evidence.timestamp,
                state=SetupState.DEVELOPING,
                direction=None,
                reason="Directional evidence has not yet been established.",
            )

        machine = ICT2022StateMachine()

        context_decision = machine.identify_context(
            as_of=evidence.timestamp,
            direction=evidence.direction,
            dealing_range=evidence.dealing_range,
            draw_on_liquidity=evidence.draw_on_liquidity,
        )

        if context_decision.state in {SetupState.NO_TRADE, SetupState.DEVELOPING}:
            return self._decision(evidence, context_decision)

        liquidity_decision = machine.process_liquidity_event(
            evidence.liquidity_event,
            direction=evidence.direction,  # context decision guarantees non-None
        )

        if liquidity_decision.state in {SetupState.NO_TRADE, SetupState.DEVELOPING}:
            return self._decision(evidence, liquidity_decision)

        mss_decision = machine.confirm_displacement_and_mss(
            mss=evidence.mss,
            direction=evidence.direction,
        )

        if mss_decision.state in {SetupState.NO_TRADE, SetupState.DEVELOPING}:
            return self._decision(evidence, mss_decision)

        pd_decision = machine.identify_pd_array(
            pd_array=evidence.pd_array,
            direction=evidence.direction,
        )

        if pd_decision.state in {SetupState.NO_TRADE, SetupState.DEVELOPING}:
            return self._decision(evidence, pd_decision)

        if evidence.entry_price is None:
            return AdapterDecision(
                evidence.timestamp,
                SetupState.DEVELOPING,
                evidence.direction,
                "All structural prerequisites are present; entry price is not yet defined.",
            )

        retracement = machine.evaluate_retracement(
            current_price=evidence.entry_price,
            pd_array=evidence.pd_array,
            direction=evidence.direction,
        )

        if retracement.state is not SetupState.ENTRY_TRIGGERED:
            return self._decision(evidence, retracement)

        activation = machine.activate_trade(
            entry=evidence.entry_price,
            invalidation=evidence.invalidation_price,
            target=evidence.target_price,
            direction=evidence.direction,
        )

        if activation.state is not SetupState.ACTIVE_TRADE:
            return self._decision(evidence, activation)

        planned_r = _planned_r(
            direction=evidence.direction,
            entry=evidence.entry_price,
            invalidation=evidence.invalidation_price,
            target=evidence.target_price,
        )

        if planned_r is None:
            return AdapterDecision(
                evidence.timestamp,
                SetupState.NO_TRADE,
                evidence.direction,
                "Entry, structural invalidation and target do not form valid trade geometry.",
            )

        context = SetupContext(
            as_of=evidence.timestamp,
            direction=evidence.direction,
            dealing_range=evidence.dealing_range,
            draw_on_liquidity=evidence.draw_on_liquidity,
            liquidity_event=evidence.liquidity_event,
            mss=evidence.mss,
            pd_array=evidence.pd_array,
            invalidation_price=evidence.invalidation_price,
            target_price=evidence.target_price,
            target_liquidity=evidence.target_liquidity,
        )

        return AdapterDecision(
            timestamp=evidence.timestamp,
            state=SetupState.ACTIVE_TRADE,
            direction=evidence.direction,
            reason="All mandatory ICT-2022 setup conditions are satisfied.",
            entry_price=evidence.entry_price,
            invalidation_price=evidence.invalidation_price,
            target_price=evidence.target_price,
            planned_r=planned_r,
            context=context,
        )

    @staticmethod
    def _decision(
        evidence: StrategyEvidence,
        decision: SetupDecision,
    ) -> AdapterDecision:
        return AdapterDecision(
            timestamp=evidence.timestamp,
            state=decision.state,
            direction=decision.direction,
            reason=decision.reason,
            entry_price=evidence.entry_price,
            invalidation_price=evidence.invalidation_price,
            target_price=evidence.target_price,
        )


class EvidenceProvider:
    """
    Small integration protocol.

    A concrete provider should use ONLY the visible replay frames and return
    StrategyEvidence for that timestamp.

    Keeping this interface separate prevents the replay engine from becoming
    coupled to one detector implementation or silently gaining look-ahead.
    """

    def build(
        self,
        timestamp: pd.Timestamp,
        visible_base: pd.DataFrame,
        visible_context,
    ) -> StrategyEvidence:
        raise NotImplementedError
