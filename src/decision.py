from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionMode(str, Enum):
    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    LIVE = "LIVE"

    @classmethod
    def from_value(cls, value: Any) -> "ExecutionMode":
        if value is None:
            return cls.RESEARCH
        text = str(value).strip().upper()
        try:
            return cls(text)
        except ValueError as exc:  # pragma: no cover - defensive validation
            valid = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Unsupported execution mode '{value}'. Expected one of: {valid}") from exc


class SetupResearchModel:
    def score(self, qualified_setup: Any, context: Optional[Dict[str, Any]] = None) -> float:
        return 0.0


@dataclass
class Invalidation:
    price: float | None = None
    structural_reason: str = ""
    invalidates_state: str = ""


@dataclass
class Scenario:
    direction: str = "NONE"
    required_trigger: str = ""
    invalidation: str = ""
    target: str = ""
    status: str = "DEVELOPING"


@dataclass
class LiquidityLevel:
    id: str | None = None
    type: str = ""
    classification: str = ""
    price: float | None = None
    timeframe: str = ""
    source: str = ""
    created_at: str | None = None
    significance: float = 0.0
    status: str = "ACTIVE"


@dataclass
class LiquidityEvent:
    liquidity_level_id: str | None = None
    breached_at: str | None = None
    breach_price: float | None = None
    direction: str = ""
    post_breach_behavior: str = ""
    classification: str = "UNRESOLVED"
    confirmed_at: str | None = None


@dataclass
class DealingRange:
    timeframe: str = ""
    high: float | None = None
    low: float | None = None
    anchor_type: str = ""
    anchor_reference: str = ""
    equilibrium: float | None = None


@dataclass
class SwingPoint:
    id: str | None = None
    type: str = ""
    price: float | None = None
    timestamp: str | None = None
    timeframe: str = ""
    strength: float = 0.0
    confirmed_at: str | None = None


@dataclass
class DisplacementEvent:
    direction: str = ""
    start_time: str | None = None
    end_time: str | None = None
    candles: int = 0
    range_expansion: float = 0.0
    efficiency: float = 0.0
    structural_consequence: str = ""
    event_context: str = ""
    quality: str = ""


@dataclass
class MSS:
    direction: str = ""
    broken_structure_id: str | None = None
    broken_price: float | None = None
    timeframe: str = ""
    liquidity_event_id: str | None = None
    displacement_event_id: str | None = None
    confirmed_at: str | None = None


@dataclass
class FVG:
    direction: str = ""
    lower_bound: float | None = None
    upper_bound: float | None = None
    created_at: str | None = None
    timeframe: str = ""
    source_candles: List[Any] = field(default_factory=list)
    displacement_event_id: str | None = None
    liquidity_event_id: str | None = None
    qualified: bool = False
    invalidated: bool = False


@dataclass
class Decision:
    status: str = "DEVELOPING"
    direction: str = "NONE"
    model: str = "NONE"
    state: str = "WAITING"
    evidence: Dict[str, Any] = field(default_factory=dict)
    entry: Any | None = None
    invalidation: Any | None = None
    target: Any | None = None
    planned_R: float = 0.0
    macro_state: str = "UNKNOWN"
    session: str = "UNKNOWN"
    contradictions: List[str] = field(default_factory=list)
    missing_conditions: List[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "direction": self.direction,
            "model": self.model,
            "state": self.state,
            "evidence": self.evidence,
            "entry": self.entry,
            "invalidation": self.invalidation,
            "target": self.target,
            "planned_R": self.planned_R,
            "macro_state": self.macro_state,
            "session": self.session,
            "contradictions": self.contradictions,
            "missing_conditions": self.missing_conditions,
            "reason": self.reason,
        }
