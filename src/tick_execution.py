from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

import pandas as pd

from .replay_subject import ReplaySubject


class ExecutionDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExecutionStatus(str, Enum):
    NO_TRADE_NOT_ACTIONABLE = "NO_TRADE_NOT_ACTIONABLE"
    NO_TRADE_MISSING_DIRECTION = "NO_TRADE_MISSING_DIRECTION"
    NO_TRADE_CONFLICTING_DIRECTIONS = "NO_TRADE_CONFLICTING_DIRECTIONS"
    VALID_DECISION_NOT_SELECTED = "VALID_DECISION_NOT_SELECTED"
    NO_TRADE_ACTIVE_POSITION = "NO_TRADE_ACTIVE_POSITION"
    NO_TRADE_UNSUPPORTED_ORDER_TYPE = "NO_TRADE_UNSUPPORTED_ORDER_TYPE"
    NO_TRADE_INVALID_EXECUTION_CONTRACT = "NO_TRADE_INVALID_EXECUTION_CONTRACT"
    NO_TRADE_INVALID_COST_CONFIGURATION = "NO_TRADE_INVALID_COST_CONFIGURATION"
    NO_TRADE_INVALID_GEOMETRY = "NO_TRADE_INVALID_GEOMETRY"
    NO_TRADE_INVALID_QUANTITY = "NO_TRADE_INVALID_QUANTITY"
    NO_TRADE_RISK_LIMIT = "NO_TRADE_RISK_LIMIT"
    UNSPECIFIED_QUANTITY_ROUNDING = "UNSPECIFIED_QUANTITY_ROUNDING"
    INTENT_CREATED = "INTENT_CREATED"
    FILLED_OPEN = "FILLED_OPEN"
    CLOSED = "CLOSED"
    UNFILLED_AT_END_OF_INPUT = "UNFILLED_AT_END_OF_INPUT"
    OPEN_AT_END_OF_INPUT_UNSPECIFIED_VALUATION = (
        "OPEN_AT_END_OF_INPUT_UNSPECIFIED_VALUATION"
    )


class PositionState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp.")
    return timestamp.tz_convert("UTC")


def _finite_positive(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _quantity_matches_step(quantity: float, step: float) -> bool:
    quantity_numerator, quantity_denominator = Decimal(str(quantity)).as_integer_ratio()
    step_numerator, step_denominator = Decimal(str(step)).as_integer_ratio()
    return (
        quantity_numerator * step_denominator
    ) % (
        quantity_denominator * step_numerator
    ) == 0


def _execution_identity(
    execution_run_id: str,
    subject_id: str,
    order_discriminator: str,
) -> str:
    return json.dumps(
        [execution_run_id, subject_id, order_discriminator],
        ensure_ascii=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class QuoteEvent:
    timestamp: pd.Timestamp
    bid: float
    ask: float
    source_order: int
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc_timestamp(self.timestamp, "timestamp"))
        if not _finite_positive(self.bid) or not _finite_positive(self.ask):
            raise ValueError("Quote bid and ask must be finite and positive.")
        if self.ask < self.bid:
            raise ValueError("Quote ask cannot be less than bid.")
        if isinstance(self.source_order, bool) or not isinstance(self.source_order, int):
            raise ValueError("source_order must be an integer.")
        if self.source_order < 0:
            raise ValueError("source_order must be non-negative.")
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError("Quote provenance must be a non-empty mapping.")
        object.__setattr__(self, "bid", float(self.bid))
        object.__setattr__(self, "ask", float(self.ask))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> QuoteEvent:
        required = {"timestamp", "bid", "ask", "source_order"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Canonical tick is missing fields: {sorted(missing)}")
        explicit_provenance = row.get("provenance", {})
        if not isinstance(explicit_provenance, Mapping):
            raise ValueError("Canonical tick provenance must be a mapping.")
        provenance = dict(explicit_provenance)
        provenance.update({
            key: value
            for key, value in row.items()
            if key not in required and key != "provenance"
        })
        return cls(
            timestamp=row["timestamp"],
            bid=row["bid"],
            ask=row["ask"],
            source_order=row["source_order"],
            provenance=provenance,
        )


def quote_events_from_tick_batches(
    batches: Iterable[pd.DataFrame],
) -> Iterable[QuoteEvent]:
    """Convert canonical Dukascopy batches without changing their row order."""
    for batch in batches:
        for row in batch.to_dict(orient="records"):
            yield QuoteEvent.from_mapping(row)


@dataclass(frozen=True)
class DecisionRecord:
    subject: ReplaySubject
    actionable: bool
    direction: ExecutionDirection | str | None = None
    order_type: str = "MARKET"
    stop_loss: float | None = None
    take_profit: float | None = None
    quantity: float | None = None
    strategy_state: str | None = None
    evidence_reference: object | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.direction is not None and not isinstance(self.direction, ExecutionDirection):
            try:
                object.__setattr__(self, "direction", ExecutionDirection(str(self.direction).upper()))
            except ValueError:
                object.__setattr__(self, "direction", None)
        object.__setattr__(self, "order_type", str(self.order_type).upper())

    @property
    def decision_time(self) -> pd.Timestamp:
        return self.subject.availability_timestamp


@dataclass(frozen=True)
class DecisionSet:
    decision_time: pd.Timestamp
    decisions: tuple[DecisionRecord, ...]
    selected_subject_id: str | None
    arbitration_status: str
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time", _utc_timestamp(self.decision_time, "decision_time"))
        if any(decision.decision_time != self.decision_time for decision in self.decisions):
            raise ValueError("All decisions in a set must share its decision_time.")


@dataclass(frozen=True)
class ExecutionConfig:
    contract_size: float | None
    minimum_quantity: float | None
    quantity_step: float | None
    account_currency_conversion: float | None
    commission_per_unit_per_side: float | None
    slippage_per_unit_per_side: float = 0.0

    @property
    def contract_is_valid(self) -> bool:
        return all(
            _finite_positive(value)
            for value in (
                self.contract_size,
                self.minimum_quantity,
                self.quantity_step,
                self.account_currency_conversion,
            )
        )

    @property
    def costs_are_valid(self) -> bool:
        return _finite_nonnegative(self.commission_per_unit_per_side) and _finite_nonnegative(
            self.slippage_per_unit_per_side
        )


@dataclass(frozen=True)
class ExecutionIntent:
    intent_id: str
    execution_run_id: str
    subject: ReplaySubject
    order_id: str
    decision_time: pd.Timestamp
    direction: ExecutionDirection
    order_type: str
    stop_loss: float
    take_profit: float
    requested_quantity: float
    account_equity_before_entry: float | None
    risk_budget: float | None
    cost_configuration: ExecutionConfig


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    execution_run_id: str
    subject: ReplaySubject
    order_discriminator: str
    order_type: str
    status: str
    eligible_after: pd.Timestamp
    fill_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class FillRecord:
    fill_id: str
    execution_identity: str
    order_id: str
    subject: ReplaySubject
    timestamp: pd.Timestamp
    source_order: int
    provenance: Mapping[str, object]
    side: str
    observed_price: float
    quantity: float
    commission_cost: float
    slippage_cost: float

    @property
    def total_cost(self) -> float:
        return self.commission_cost + self.slippage_cost


@dataclass(frozen=True)
class PositionRecord:
    position_id: str
    execution_run_id: str
    subject: ReplaySubject
    order_id: str
    direction: ExecutionDirection
    entry_fill: FillRecord
    quantity: float
    stop_loss: float
    take_profit: float
    state: PositionState
    exit_fill: FillRecord | None = None
    gross_realized_pnl: float | None = None
    net_realized_pnl: float | None = None


@dataclass(frozen=True)
class ExecutionResult:
    subject: ReplaySubject
    decision: DecisionRecord
    decision_set: DecisionSet
    status: ExecutionStatus
    reason: str | None = None
    intent: ExecutionIntent | None = None
    order: OrderRecord | None = None
    exit_order: OrderRecord | None = None
    position: PositionRecord | None = None
    entry_fill: FillRecord | None = None
    exit_fill: FillRecord | None = None
    gross_realized_pnl: float | None = None
    net_realized_pnl: float | None = None
    cost_breakdown: Mapping[str, float] = field(default_factory=dict)
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost_breakdown", MappingProxyType(dict(self.cost_breakdown)))


@dataclass
class _ResultState:
    decision: DecisionRecord
    decision_set: DecisionSet
    status: ExecutionStatus
    reason: str | None = None
    intent: ExecutionIntent | None = None
    order: OrderRecord | None = None
    exit_order: OrderRecord | None = None
    position: PositionRecord | None = None
    entry_fill: FillRecord | None = None
    exit_fill: FillRecord | None = None
    gross_realized_pnl: float | None = None
    net_realized_pnl: float | None = None
    cost_breakdown: dict[str, float] = field(default_factory=dict)


class ExecutionEngine:
    """Standalone single-position historical execution over ordered quote events."""

    def __init__(
        self,
        *,
        execution_run_id: str,
        initial_equity: float,
        config: ExecutionConfig,
    ) -> None:
        if not execution_run_id:
            raise ValueError("execution_run_id must not be empty.")
        if not _finite_positive(initial_equity):
            raise ValueError("initial_equity must be finite and positive.")
        self.execution_run_id = execution_run_id
        self.initial_equity = float(initial_equity)
        self.config = config
        self._current_equity = float(initial_equity)
        self.ending_equity = float(initial_equity)
        self.decision_sets: list[DecisionSet] = []
        self.intents: list[ExecutionIntent] = []
        self.orders: list[OrderRecord] = []
        self.fills: list[FillRecord] = []
        self.positions: list[PositionRecord] = []
        self.results: tuple[ExecutionResult, ...] = ()
        self._has_run = False

    def run(
        self,
        ticks: Iterable[QuoteEvent],
        decisions: Iterable[DecisionRecord],
    ) -> tuple[ExecutionResult, ...]:
        if self._has_run:
            raise RuntimeError("An ExecutionEngine instance can run only once.")
        self._has_run = True

        decision_list = list(decisions)
        grouped: dict[pd.Timestamp, list[DecisionRecord]] = {}
        seen_subjects: set[str] = set()
        for decision in decision_list:
            if not isinstance(decision, DecisionRecord):
                raise TypeError("decisions must contain DecisionRecord values.")
            subject_id = decision.subject.subject_id
            if subject_id in seen_subjects:
                raise ValueError(f"Duplicate decision subject: {subject_id}")
            seen_subjects.add(subject_id)
            grouped.setdefault(decision.decision_time, []).append(decision)

        group_times = sorted(grouped)
        group_index = 0
        states: dict[str, _ResultState] = {}
        pending: tuple[ExecutionIntent, _ResultState] | None = None
        active: tuple[PositionRecord, _ResultState] | None = None
        previous_timestamp: pd.Timestamp | None = None
        previous_source_order: int | None = None

        for tick in ticks:
            if not isinstance(tick, QuoteEvent):
                raise TypeError("ticks must contain QuoteEvent values.")
            if previous_timestamp is not None and tick.timestamp < previous_timestamp:
                raise ValueError("Tick timestamps must be non-decreasing.")
            if previous_source_order is not None and tick.source_order <= previous_source_order:
                raise ValueError("Tick source_order must be strictly increasing.")
            previous_timestamp = tick.timestamp
            previous_source_order = tick.source_order

            if group_index < len(group_times) and group_times[group_index] < tick.timestamp:
                raise ValueError("A decision availability group has no tick at its timestamp.")

            if active is not None:
                active = self._monitor_position(active, tick)

            if pending is not None and tick.timestamp > pending[0].decision_time:
                pending, active = self._match_pending(pending, tick, active)

            if group_index < len(group_times) and group_times[group_index] == tick.timestamp:
                group, group_states = self._arbitrate_group(
                    tick.timestamp,
                    grouped[tick.timestamp],
                    active,
                )
                self.decision_sets.append(group)
                states.update(group_states)
                group_index += 1
                candidate_id = group.selected_subject_id
                if candidate_id is not None:
                    state = states[candidate_id]
                    pending, active = self._create_intent(state, pending, active)

        if group_index != len(group_times):
            raise ValueError("A decision availability group has no tick at its timestamp.")
        if pending is not None:
            intent, state = pending
            self._replace_order(intent.order_id, status=ExecutionStatus.UNFILLED_AT_END_OF_INPUT.value)
            state.order = self._order_by_id(intent.order_id)
            state.status = ExecutionStatus.UNFILLED_AT_END_OF_INPUT
            state.reason = ExecutionStatus.UNFILLED_AT_END_OF_INPUT.value
        if active is not None:
            position, state = active
            state.position = position
            state.status = ExecutionStatus.OPEN_AT_END_OF_INPUT_UNSPECIFIED_VALUATION
            state.reason = "Open-position end-of-input valuation is UNSPECIFIED; no mark was created."

        self.ending_equity = self._equity
        self.results = tuple(
            ExecutionResult(
                subject=decision.subject,
                decision=decision,
                decision_set=states[decision.subject.subject_id].decision_set,
                status=states[decision.subject.subject_id].status,
                reason=states[decision.subject.subject_id].reason,
                intent=states[decision.subject.subject_id].intent,
                order=states[decision.subject.subject_id].order,
                exit_order=states[decision.subject.subject_id].exit_order,
                position=states[decision.subject.subject_id].position,
                entry_fill=states[decision.subject.subject_id].entry_fill,
                exit_fill=states[decision.subject.subject_id].exit_fill,
                gross_realized_pnl=states[decision.subject.subject_id].gross_realized_pnl,
                net_realized_pnl=states[decision.subject.subject_id].net_realized_pnl,
                cost_breakdown=states[decision.subject.subject_id].cost_breakdown,
                diagnostic_only=(
                    self.config.costs_are_valid
                    and self.config.commission_per_unit_per_side == 0
                    and self.config.slippage_per_unit_per_side == 0
                ),
            )
            for decision in decision_list
        )
        return self.results

    @property
    def _equity(self) -> float:
        return getattr(self, "_current_equity", self.initial_equity)

    @_equity.setter
    def _equity(self, value: float) -> None:
        self._current_equity = value

    def _arbitrate_group(
        self,
        timestamp: pd.Timestamp,
        decisions: list[DecisionRecord],
        active: tuple[PositionRecord, _ResultState] | None,
    ) -> tuple[DecisionSet, dict[str, _ResultState]]:
        ordered = tuple(
            sorted(decisions, key=lambda item: (item.subject.base_bar_timestamp, item.subject.subject_id))
        )
        actionable = [item for item in ordered if item.actionable]
        actionable_directions = {item.direction for item in actionable if item.direction is not None}
        selected: DecisionRecord | None = None
        reason = None
        if not actionable:
            arbitration_status = "NO_ACTIONABLE_DECISIONS"
        elif len(actionable_directions) > 1:
            arbitration_status = "CONFLICTING_DIRECTIONS"
            reason = ExecutionStatus.NO_TRADE_CONFLICTING_DIRECTIONS.value
        elif any(item.direction is None for item in actionable):
            arbitration_status = "MISSING_ACTIONABLE_DIRECTION"
            reason = ExecutionStatus.NO_TRADE_MISSING_DIRECTION.value
        else:
            selected = max(
                actionable,
                key=lambda item: (item.subject.base_bar_timestamp, item.subject.subject_id),
            )
            arbitration_status = "SELECTED_LATEST_BASE_BAR"

        group = DecisionSet(
            decision_time=timestamp,
            decisions=ordered,
            selected_subject_id=selected.subject.subject_id if selected else None,
            arbitration_status=arbitration_status,
            reason=reason,
        )
        state_map: dict[str, _ResultState] = {}
        for decision in ordered:
            if not decision.actionable:
                status = ExecutionStatus.NO_TRADE_NOT_ACTIONABLE
                decision_reason = decision.reason or status.value
            elif arbitration_status == "CONFLICTING_DIRECTIONS":
                status = ExecutionStatus.NO_TRADE_CONFLICTING_DIRECTIONS
                decision_reason = status.value
            elif arbitration_status == "MISSING_ACTIONABLE_DIRECTION":
                status = ExecutionStatus.NO_TRADE_MISSING_DIRECTION
                decision_reason = status.value
            elif selected is not None and decision.subject.subject_id != selected.subject.subject_id:
                status = ExecutionStatus.VALID_DECISION_NOT_SELECTED
                decision_reason = f"Latest base-bar subject selected: {selected.subject.subject_id}"
            elif active is not None:
                status = ExecutionStatus.NO_TRADE_ACTIVE_POSITION
                decision_reason = status.value
            else:
                status = ExecutionStatus.INTENT_CREATED
                decision_reason = None
            state = _ResultState(decision, group, status, decision_reason)
            state_map[decision.subject.subject_id] = state
        return group, state_map

    def _create_intent(
        self,
        state: _ResultState,
        pending: tuple[ExecutionIntent, _ResultState] | None,
        active: tuple[PositionRecord, _ResultState] | None,
    ) -> tuple[tuple[ExecutionIntent, _ResultState] | None, tuple[PositionRecord, _ResultState] | None]:
        decision = state.decision
        if active is not None:
            state.status = ExecutionStatus.NO_TRADE_ACTIVE_POSITION
            state.reason = state.status.value
            return pending, active
        if pending is not None:
            state.status = ExecutionStatus.NO_TRADE_ACTIVE_POSITION
            state.reason = "An earlier market intent is awaiting its first later tick."
            return pending, active
        if decision.order_type != "MARKET":
            state.status = ExecutionStatus.NO_TRADE_UNSUPPORTED_ORDER_TYPE
            state.reason = state.status.value
            return pending, active
        if not self.config.contract_is_valid:
            state.status = ExecutionStatus.NO_TRADE_INVALID_EXECUTION_CONTRACT
            state.reason = state.status.value
            return pending, active
        if not self.config.costs_are_valid:
            state.status = ExecutionStatus.NO_TRADE_INVALID_COST_CONFIGURATION
            state.reason = state.status.value
            return pending, active
        if decision.direction is None:
            state.status = ExecutionStatus.NO_TRADE_MISSING_DIRECTION
            state.reason = state.status.value
            return pending, active
        if not _finite_positive(decision.stop_loss) or not _finite_positive(decision.take_profit):
            state.status = ExecutionStatus.NO_TRADE_INVALID_GEOMETRY
            state.reason = "Finite positive stop-loss and take-profit triggers are required."
            return pending, active
        if decision.quantity is None:
            state.status = ExecutionStatus.UNSPECIFIED_QUANTITY_ROUNDING
            state.reason = "No quantity supplied; quantity rounding policy is UNSPECIFIED."
            return pending, active
        if not _finite_positive(decision.quantity):
            state.status = ExecutionStatus.NO_TRADE_INVALID_QUANTITY
            state.reason = state.status.value
            return pending, active

        subject = decision.subject
        intent_id = _execution_identity(self.execution_run_id, subject.subject_id, "market-entry/intent")
        order_id = _execution_identity(self.execution_run_id, subject.subject_id, "market-entry/order")
        intent = ExecutionIntent(
            intent_id=intent_id,
            execution_run_id=self.execution_run_id,
            subject=subject,
            order_id=order_id,
            decision_time=decision.decision_time,
            direction=decision.direction,
            order_type=decision.order_type,
            stop_loss=float(decision.stop_loss),
            take_profit=float(decision.take_profit),
            requested_quantity=float(decision.quantity),
            account_equity_before_entry=None,
            risk_budget=None,
            cost_configuration=self.config,
        )
        order = OrderRecord(
            order_id=order_id,
            execution_run_id=self.execution_run_id,
            subject=subject,
            order_discriminator="market-entry",
            order_type="MARKET",
            status="WAITING_FOR_LATER_TICK",
            eligible_after=decision.decision_time,
        )
        self.intents.append(intent)
        self.orders.append(order)
        state.intent = intent
        state.order = order
        state.status = ExecutionStatus.INTENT_CREATED
        return (intent, state), active

    def _match_pending(
        self,
        pending: tuple[ExecutionIntent, _ResultState],
        tick: QuoteEvent,
        active: tuple[PositionRecord, _ResultState] | None,
    ) -> tuple[tuple[ExecutionIntent, _ResultState] | None, tuple[PositionRecord, _ResultState] | None]:
        intent, state = pending
        if active is not None:
            raise RuntimeError("A pending entry intent cannot coexist with an active position.")
        if tick.timestamp <= intent.decision_time:
            return pending, active

        config = intent.cost_configuration
        quantity = intent.requested_quantity
        if quantity < float(config.minimum_quantity):
            state.status = ExecutionStatus.NO_TRADE_INVALID_QUANTITY
            state.reason = "Quantity is below the configured minimum."
            self._replace_order(intent.order_id, status=state.status.value, reason=state.reason)
            state.order = self._order_by_id(intent.order_id)
            return None, active
        if not _quantity_matches_step(quantity, float(config.quantity_step)):
            state.status = ExecutionStatus.NO_TRADE_INVALID_QUANTITY
            state.reason = "Quantity does not conform to the configured quantity step."
            self._replace_order(intent.order_id, status=state.status.value, reason=state.reason)
            state.order = self._order_by_id(intent.order_id)
            return None, active

        price = tick.ask if intent.direction is ExecutionDirection.LONG else tick.bid
        geometry_valid = (
            intent.stop_loss < price < intent.take_profit
            if intent.direction is ExecutionDirection.LONG
            else intent.take_profit < price < intent.stop_loss
        )
        if not geometry_valid:
            state.status = ExecutionStatus.NO_TRADE_INVALID_GEOMETRY
            state.reason = "Stop-loss and take-profit do not bracket the executable entry quote."
            self._replace_order(intent.order_id, status=state.status.value, reason=state.reason)
            state.order = self._order_by_id(intent.order_id)
            return None, active

        equity_before_entry = self._equity
        risk_budget = equity_before_entry * 0.01
        risk_amount = (
            abs(price - intent.stop_loss)
            * float(config.contract_size)
            * float(config.account_currency_conversion)
            * quantity
        )
        if risk_amount > risk_budget:
            state.status = ExecutionStatus.NO_TRADE_RISK_LIMIT
            state.reason = f"Configured quantity risks {risk_amount:g}, above 1% budget {risk_budget:g}."
            self._replace_order(intent.order_id, status=state.status.value, reason=state.reason)
            state.order = self._order_by_id(intent.order_id)
            return None, active

        intent = replace(
            intent,
            account_equity_before_entry=equity_before_entry,
            risk_budget=risk_budget,
        )
        intent_index = next(
            index for index, existing in enumerate(self.intents)
            if existing.intent_id == intent.intent_id
        )
        self.intents[intent_index] = intent
        commission = float(config.commission_per_unit_per_side) * quantity
        slippage = float(config.slippage_per_unit_per_side) * quantity
        fill_id = _execution_identity(self.execution_run_id, intent.subject.subject_id, "market-entry/fill")
        fill = FillRecord(
            fill_id=fill_id,
            execution_identity=_execution_identity(self.execution_run_id, intent.subject.subject_id, "market-entry"),
            order_id=intent.order_id,
            subject=intent.subject,
            timestamp=tick.timestamp,
            source_order=tick.source_order,
            provenance=tick.provenance,
            side="ASK" if intent.direction is ExecutionDirection.LONG else "BID",
            observed_price=price,
            quantity=quantity,
            commission_cost=commission,
            slippage_cost=slippage,
        )
        position_id = _execution_identity(self.execution_run_id, intent.subject.subject_id, "position")
        position = PositionRecord(
            position_id=position_id,
            execution_run_id=self.execution_run_id,
            subject=intent.subject,
            order_id=intent.order_id,
            direction=intent.direction,
            entry_fill=fill,
            quantity=quantity,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            state=PositionState.OPEN,
        )
        self.fills.append(fill)
        self.positions.append(position)
        self._equity = equity_before_entry - fill.total_cost
        self._replace_order(intent.order_id, status="FILLED", fill_id=fill.fill_id)
        state.intent = intent
        state.order = self._order_by_id(intent.order_id)
        state.entry_fill = fill
        state.position = position
        state.status = ExecutionStatus.FILLED_OPEN
        state.reason = None
        state.cost_breakdown = {"entry_commission": commission, "entry_slippage": slippage}
        return None, (position, state)

    def _monitor_position(
        self,
        active: tuple[PositionRecord, _ResultState],
        tick: QuoteEvent,
    ) -> tuple[PositionRecord, _ResultState] | None:
        position, state = active
        if position.direction is ExecutionDirection.LONG:
            trigger = (
                (tick.bid <= position.stop_loss, "STOP_LOSS")
                if tick.bid <= position.stop_loss
                else (tick.bid >= position.take_profit, "TAKE_PROFIT")
            )
            exit_side = "BID"
            exit_price = tick.bid
        else:
            trigger = (
                (tick.ask >= position.stop_loss, "STOP_LOSS")
                if tick.ask >= position.stop_loss
                else (tick.ask <= position.take_profit, "TAKE_PROFIT")
            )
            exit_side = "ASK"
            exit_price = tick.ask
        triggered, reason = trigger
        if not triggered:
            return active

        subject = position.subject
        order_id = _execution_identity(self.execution_run_id, subject.subject_id, "market-exit/order")
        fill_id = _execution_identity(self.execution_run_id, subject.subject_id, "market-exit/fill")
        order = OrderRecord(
            order_id=order_id,
            execution_run_id=self.execution_run_id,
            subject=subject,
            order_discriminator="market-exit",
            order_type="MARKET",
            status="FILLED",
            eligible_after=position.entry_fill.timestamp,
            fill_id=fill_id,
            reason=reason,
        )
        config = self.config
        commission = float(config.commission_per_unit_per_side) * position.quantity
        slippage = float(config.slippage_per_unit_per_side) * position.quantity
        fill = FillRecord(
            fill_id=fill_id,
            execution_identity=_execution_identity(self.execution_run_id, subject.subject_id, "market-exit"),
            order_id=order_id,
            subject=subject,
            timestamp=tick.timestamp,
            source_order=tick.source_order,
            provenance=tick.provenance,
            side=exit_side,
            observed_price=exit_price,
            quantity=position.quantity,
            commission_cost=commission,
            slippage_cost=slippage,
        )
        sign = 1.0 if position.direction is ExecutionDirection.LONG else -1.0
        gross = (
            sign
            * (fill.observed_price - position.entry_fill.observed_price)
            * position.quantity
            * float(config.contract_size)
            * float(config.account_currency_conversion)
        )
        total_cost = position.entry_fill.total_cost + fill.total_cost
        net = gross - total_cost
        closed = replace(
            position,
            state=PositionState.CLOSED,
            exit_fill=fill,
            gross_realized_pnl=gross,
            net_realized_pnl=net,
        )
        position_index = self.positions.index(position)
        self.positions[position_index] = closed
        self.fills.append(fill)
        self.orders.append(order)
        self._equity += gross - fill.total_cost
        state.exit_order = order
        state.exit_fill = fill
        state.position = closed
        state.gross_realized_pnl = gross
        state.net_realized_pnl = net
        state.status = ExecutionStatus.CLOSED
        state.reason = reason
        state.cost_breakdown.update({"exit_commission": commission, "exit_slippage": slippage})
        return None

    def _order_by_id(self, order_id: str) -> OrderRecord:
        return next(order for order in self.orders if order.order_id == order_id)

    def _replace_order(
        self,
        order_id: str,
        *,
        status: str,
        fill_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        index = next(index for index, order in enumerate(self.orders) if order.order_id == order_id)
        self.orders[index] = replace(
            self.orders[index],
            status=status,
            fill_id=fill_id if fill_id is not None else self.orders[index].fill_id,
            reason=reason,
        )


__all__ = [
    "DecisionRecord",
    "DecisionSet",
    "ExecutionConfig",
    "ExecutionDirection",
    "ExecutionEngine",
    "ExecutionIntent",
    "ExecutionResult",
    "ExecutionStatus",
    "FillRecord",
    "OrderRecord",
    "PositionRecord",
    "PositionState",
    "QuoteEvent",
    "quote_events_from_tick_batches",
]