from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

import json
import pandas as pd


class DecisionClass(str, Enum):
    VALID = "VALID"
    DEVELOPING = "DEVELOPING"
    INVALID = "INVALID"
    NO_TRADE = "NO_TRADE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class TradeJournalRecord:
    trade_id: str
    decision_time: pd.Timestamp
    session: Optional[str]
    regime: Optional[str]
    htf_bias: Optional[str]
    model: str
    liquidity_event: Optional[str]
    trigger: Optional[str]
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    planned_r: Optional[float]
    realized_r: Optional[float]
    mfe_r: Optional[float]
    mae_r: Optional[float]
    result: Optional[str]
    execution_deviation: Optional[float]
    error_class: Optional[str]
    news_state: Optional[str]
    decision_class: DecisionClass
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_class",
            DecisionClass(self.decision_class),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision_time"] = self.decision_time.isoformat()
        data["decision_class"] = self.decision_class.value
        return data


@dataclass(frozen=True)
class NoTradeRecord:
    observation_id: str
    decision_time: pd.Timestamp
    market_state: Optional[str]
    htf_bias: Optional[str]
    model: str
    tempting_setup: Optional[str]
    rejection_reason: str
    observable_subsequent_outcome: Optional[str] = None
    session: Optional[str] = None
    regime: Optional[str] = None
    news_state: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision_time"] = self.decision_time.isoformat()
        return data


class ResearchJournal:
    """
    Append-only in-memory research journal.

    The journal records what was knowable at decision time. It does not
    rewrite an original decision after observing the outcome.
    """

    def __init__(self) -> None:
        self._trades: list[TradeJournalRecord] = []
        self._no_trades: list[NoTradeRecord] = []

    @property
    def trades(self) -> tuple[TradeJournalRecord, ...]:
        return tuple(self._trades)

    @property
    def no_trades(self) -> tuple[NoTradeRecord, ...]:
        return tuple(self._no_trades)

    def record_trade(self, record: TradeJournalRecord) -> None:
        if any(x.trade_id == record.trade_id for x in self._trades):
            raise ValueError(f"Duplicate trade_id: {record.trade_id}")
        self._trades.append(record)

    def record_no_trade(self, record: NoTradeRecord) -> None:
        if any(x.observation_id == record.observation_id for x in self._no_trades):
            raise ValueError(f"Duplicate observation_id: {record.observation_id}")
        self._no_trades.append(record)

    def export_jsonl(self, path: str) -> None:
        """
        Export both datasets as JSONL with an explicit record_type.

        Existing records are serialized exactly as stored; no retrospective
        classification or optimization occurs during export.
        """
        with open(path, "w", encoding="utf-8") as handle:
            for record in self._trades:
                handle.write(json.dumps(
                    {"record_type": "TRADE", **record.to_dict()},
                    separators=(",", ":"),
                ) + "\n")
            for record in self._no_trades:
                handle.write(json.dumps(
                    {"record_type": "NO_TRADE", **record.to_dict()},
                    separators=(",", ":"),
                ) + "\n")
