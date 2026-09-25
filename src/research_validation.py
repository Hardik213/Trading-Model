from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .research_journal import (
    DecisionClass,
    NoTradeRecord,
    ResearchJournal,
    TradeJournalRecord,
)


@dataclass(frozen=True)
class DatasetCounts:
    total_trade_records: int
    valid_trade_records: int
    total_no_trade_records: int
    developing_records: int
    invalid_records: int
    ambiguous_records: int


def dataset_counts(journal: ResearchJournal) -> DatasetCounts:
    return DatasetCounts(
        total_trade_records=len(journal.trades),
        valid_trade_records=sum(
            x.decision_class == DecisionClass.VALID for x in journal.trades
        ),
        total_no_trade_records=len(journal.no_trades),
        developing_records=sum(
            x.decision_class == DecisionClass.DEVELOPING for x in journal.trades
        ),
        invalid_records=sum(
            x.decision_class == DecisionClass.INVALID for x in journal.trades
        ),
        ambiguous_records=sum(
            x.decision_class == DecisionClass.AMBIGUOUS for x in journal.trades
        ),
    )


def require_no_trade_reason(record: NoTradeRecord) -> None:
    if not record.rejection_reason.strip():
        raise ValueError("A no-trade record requires a rejection reason.")


def validate_trade_record(record: TradeJournalRecord) -> None:
    if record.decision_class == DecisionClass.VALID:
        required = {
            "entry_price": record.entry_price,
            "stop_price": record.stop_price,
            "target_price": record.target_price,
            "planned_r": record.planned_r,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "VALID trade record is missing: " + ", ".join(missing)
            )

    if record.realized_r is not None and record.planned_r is None:
        raise ValueError("realized_r cannot exist without planned_r.")


def validate_journal(journal: ResearchJournal) -> None:
    for record in journal.trades:
        validate_trade_record(record)
    for record in journal.no_trades:
        require_no_trade_reason(record)
