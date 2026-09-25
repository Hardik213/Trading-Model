import json
import pandas as pd
import pytest

from src.research_journal import (
    DecisionClass,
    NoTradeRecord,
    ResearchJournal,
    TradeJournalRecord,
)
from src.research_validation import dataset_counts, validate_journal


def ts():
    return pd.Timestamp("2026-01-01T10:00:00Z")


def trade_record(trade_id="T1", decision_class=DecisionClass.VALID):
    return TradeJournalRecord(
        trade_id=trade_id,
        decision_time=ts(),
        session="NY",
        regime="TREND",
        htf_bias="BULLISH",
        model="XAU-2022 Core v1.0",
        liquidity_event="SSL_REJECTION",
        trigger="MSS_FVG_RETRACEMENT",
        entry_price=100,
        stop_price=98,
        target_price=104,
        planned_r=2,
        realized_r=2,
        mfe_r=2.5,
        mae_r=-0.5,
        result="TARGET",
        execution_deviation=0,
        error_class=None,
        news_state="NORMAL",
        decision_class=decision_class,
    )


def test_trade_is_recorded_without_rewriting_it():
    j=ResearchJournal()
    r=trade_record()
    j.record_trade(r)
    assert j.trades[0] == r


def test_duplicate_trade_is_rejected():
    j=ResearchJournal()
    j.record_trade(trade_record())
    with pytest.raises(ValueError):
        j.record_trade(trade_record())


def test_no_trade_requires_reason():
    r=NoTradeRecord(
        observation_id="N1",
        decision_time=ts(),
        market_state="RANGE",
        htf_bias="BULLISH",
        model="XAU-2022 Core v1.0",
        tempting_setup="SSL sweep",
        rejection_reason="",
    )
    j=ResearchJournal()
    j.record_no_trade(r)
    with pytest.raises(ValueError):
        validate_journal(j)


def test_valid_trade_requires_trade_geometry():
    r=trade_record()
    r=TradeJournalRecord(**{**r.to_dict(), "decision_time": ts(), "entry_price": None})
    j=ResearchJournal()
    j.record_trade(r)
    with pytest.raises(ValueError):
        validate_journal(j)


def test_counts_keep_no_trade_separate():
    j=ResearchJournal()
    j.record_trade(trade_record())
    j.record_trade(trade_record("D1", DecisionClass.DEVELOPING))
    j.record_no_trade(NoTradeRecord(
        observation_id="N1",
        decision_time=ts(),
        market_state="RANGE",
        htf_bias="BULLISH",
        model="XAU-2022 Core v1.0",
        tempting_setup="FVG",
        rejection_reason="No displacement",
    ))
    counts=dataset_counts(j)
    assert counts.total_trade_records == 2
    assert counts.valid_trade_records == 1
    assert counts.total_no_trade_records == 1
    assert counts.developing_records == 1


def test_jsonl_export_preserves_record_type(tmp_path):
    j=ResearchJournal()
    j.record_trade(trade_record())
    j.record_no_trade(NoTradeRecord(
        observation_id="N1",
        decision_time=ts(),
        market_state="RANGE",
        htf_bias="BULLISH",
        model="XAU-2022 Core v1.0",
        tempting_setup="FVG",
        rejection_reason="No MSS",
    ))
    path=tmp_path/"journal.jsonl"
    j.export_jsonl(str(path))
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert rows[0]["record_type"] == "TRADE"
    assert rows[1]["record_type"] == "NO_TRADE"
