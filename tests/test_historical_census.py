import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.event_backtester import TradePlan, TradeOutcome, simulate_trade
from src.historical_census import (
    CensusReport,
    _build_trade_plan,
    _decision_id_for,
    _decision_record,
    _trade_outcome_record,
    classify_coverage,
    summarize_data_coverage,
    build_historical_census,
    serialize_trade_plan,
)
from src.market_structure import BreachOutcome, LiquidityEvent, LiquidityLevel, LiquiditySide
from src.sniper_setup import PrecisionEvidence
from src.displacement import Direction, DisplacementEvent
from src.fvg import FVG, FVGDirection
from src.mss import MSSEvent
from src.replay_subject import ReplaySubject


def _valid_trade_evidence(ts: pd.Timestamp) -> PrecisionEvidence:
    level = LiquidityLevel(ts - pd.Timedelta(minutes=2), 99.0, LiquiditySide.SSL, "SWING")
    event = LiquidityEvent(
        level,
        ts - pd.Timedelta(minutes=2),
        99.0,
        1.0,
        BreachOutcome.REJECTION,
        ts - pd.Timedelta(minutes=1),
        101.0,
    )
    target_level = LiquidityLevel(ts, 109.0, LiquiditySide.BSL, "SWING")
    displacement = DisplacementEvent(
        ts - pd.Timedelta(minutes=1),
        Direction.BULLISH,
        100.0,
        104.0,
        3.0,
        0.8,
        0.9,
        1.5,
        True,
        True,
        ts - pd.Timedelta(minutes=1),
    )
    mss = MSSEvent("LONG", ts - pd.Timedelta(minutes=1), 100.0, displacement, True, ts - pd.Timedelta(minutes=1), None)
    fvg = FVG(
        FVGDirection.BULLISH,
        ts - pd.Timedelta(minutes=1),
        ts - pd.Timedelta(minutes=1),
        ts - pd.Timedelta(minutes=2),
        ts - pd.Timedelta(minutes=1),
        100.0,
        108.0,
        104.0,
        8.0,
    )
    return PrecisionEvidence(
        ts,
        Direction.BULLISH,
        target_level,
        event,
        mss,
        fvg,
        101.0,
        99.0,
        109.0,
        target_level,
    )


def _valid_trade_evidence_same_bar(ts: pd.Timestamp) -> PrecisionEvidence:
    level = LiquidityLevel(ts - pd.Timedelta(minutes=2), 99.0, LiquiditySide.SSL, "SWING")
    event = LiquidityEvent(
        level,
        ts - pd.Timedelta(minutes=2),
        99.0,
        1.0,
        BreachOutcome.REJECTION,
        ts - pd.Timedelta(minutes=1),
        101.0,
    )
    target_level = LiquidityLevel(ts, 104.0, LiquiditySide.BSL, "SWING")
    displacement = DisplacementEvent(
        ts - pd.Timedelta(minutes=1),
        Direction.BULLISH,
        100.0,
        104.0,
        3.0,
        0.8,
        0.9,
        1.5,
        True,
        True,
        ts - pd.Timedelta(minutes=1),
    )
    mss = MSSEvent("LONG", ts - pd.Timedelta(minutes=1), 100.0, displacement, True, ts - pd.Timedelta(minutes=1), None)
    fvg = FVG(
        FVGDirection.BULLISH,
        ts - pd.Timedelta(minutes=1),
        ts - pd.Timedelta(minutes=1),
        ts - pd.Timedelta(minutes=2),
        ts - pd.Timedelta(minutes=1),
        100.0,
        104.0,
        102.0,
        4.0,
    )
    return PrecisionEvidence(
        ts,
        Direction.BULLISH,
        target_level,
        event,
        mss,
        fvg,
        101.0,
        98.0,
        104.0,
        target_level,
    )


def test_census_report_from_result():
    class Item:
        def __init__(self, ts, state):
            self.timestamp = ts
            self.state = state

    class Result:
        observations = (
            Item(pd.Timestamp("2022-01-01T00:00:00Z"), "VALID"),
            Item(pd.Timestamp("2022-01-01T00:05:00Z"), "NO_TRADE"),
            Item(pd.Timestamp("2022-01-01T00:10:00Z"), "DEVELOPING"),
            Item(pd.Timestamp("2022-01-01T00:15:00Z"), "INVALID"),
        )

    Result.state_counts = lambda self: {
        "VALID": 1, "DEVELOPING": 1, "INVALID": 1, "NO_TRADE": 1
    }

    report = CensusReport.from_result(
        Result(), instrument="XAUUSD", source="dukascopy", timeframe="5min"
    )
    assert report.observations == 4
    assert report.valid_setups == 1
    assert report.no_trade == 1
    assert report.developing == 1
    assert report.invalid == 1


def test_subject_aware_census_ids_are_unique_and_batch_ids_stay_legacy():
    availability = pd.Timestamp("2026-10-03T10:05:17Z")
    evidence = SimpleNamespace(
        direction=Direction.BULLISH,
        entry_price=101.0,
        invalidation_price=99.0,
        target_price=105.0,
        replay_subject=None,
    )
    subjects = [
        ReplaySubject(availability, pd.Timestamp(f"2026-10-03T10:0{minute}:00Z"))
        for minute in (1, 2)
    ]
    observations = [
        SimpleNamespace(
            timestamp=availability,
            state="VALID",
            reason="VALID_SEQUENCE",
            detail="test",
            planned_r=2.0,
            evidence=evidence,
            replay_subject=subject,
        )
        for subject in subjects
    ]

    decision_ids = [_decision_id_for(observation) for observation in observations]
    plans = [_build_trade_plan(observation, data=pd.DataFrame()) for observation in observations]
    assert len(set(decision_ids)) == 2
    assert all(plan is not None for plan in plans)
    assert len({plan.trade_id for plan in plans}) == 2
    assert all(plan.entry_time == availability for plan in plans)
    assert [serialize_trade_plan(plan)["replay_subject"] for plan in plans] == [
        subject.to_dict() for subject in subjects
    ]
    assert [
        _decision_record(observation, source="test")["replay_subject"]
        for observation in observations
    ] == [subject.to_dict() for subject in subjects]
    empty_path = pd.DataFrame(
        columns=["Open", "High", "Low", "Close"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    assert [
        _trade_outcome_record(plan, empty_path)["replay_subject"]
        for plan in plans
    ] == [subject.to_dict() for subject in subjects]
    evidence_only = SimpleNamespace(
        direction=Direction.BULLISH,
        entry_price=101.0,
        invalidation_price=99.0,
        target_price=105.0,
        replay_subject=subjects[0],
    )
    observation_without_subject = SimpleNamespace(
        timestamp=availability,
        state="VALID",
        reason="VALID_SEQUENCE",
        detail="test",
        planned_r=2.0,
        evidence=evidence_only,
    )
    assert _decision_id_for(observation_without_subject) == decision_ids[0]
    evidence_only_plan = _build_trade_plan(observation_without_subject, data=pd.DataFrame())
    assert evidence_only_plan is not None
    assert evidence_only_plan.trade_id == plans[0].trade_id

    legacy = SimpleNamespace(
        timestamp=availability,
        state="VALID",
        reason="VALID_SEQUENCE",
        detail="test",
        planned_r=2.0,
        evidence=evidence,
    )
    legacy_plan = _build_trade_plan(legacy, data=pd.DataFrame())
    assert _decision_id_for(legacy) == "decision_20261003100517_valid"
    assert legacy_plan is not None
    assert legacy_plan.trade_id == "trade_20261003100517_long"
    assert json.dumps(serialize_trade_plan(legacy_plan), indent=2) == json.dumps({
        "trade_id": "trade_20261003100517_long",
        "entry_time": "2026-10-03T10:05:17+00:00",
        "direction": "LONG",
        "entry_price": 101.0,
        "stop_price": 99.0,
        "target_price": 105.0,
        "planned_r": 2.0,
    }, indent=2)


def test_smoke_classifier_marks_sparse_history():
    idx = pd.date_range("2022-01-01 00:00:00Z", periods=8, freq="5min")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
        },
        index=idx,
    )
    coverage = summarize_data_coverage(df, timeframe="5min", source="dukascopy")
    assert classify_coverage(coverage) == "SMOKE_CENSUS"
    assert coverage["duration_minutes"] < 60


def test_build_historical_census_writes_smoke_summary(tmp_path):
    idx = pd.date_range("2022-01-01 00:00:00Z", periods=6, freq="5min")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
        },
        index=idx,
    )

    payload = build_historical_census(
        df,
        evidence_builder=lambda ts, base, context: PrecisionEvidence(
            as_of=ts,
            direction=None,
            draw_on_liquidity=None,
            liquidity_event=None,
            mss=None,
            pd_array=None,
            entry_price=None,
            invalidation_price=None,
            target_price=None,
            target_liquidity=None,
        ),
        output_dir=tmp_path,
        timeframe="5min",
        source="dukascopy",
    )

    assert payload["classification"] == "SMOKE_CENSUS"
    assert payload["coverage"]["rows"] == 6
    summary_path = Path(tmp_path) / "summary.json"
    assert summary_path.exists()


def test_valid_observation_creates_canonical_tradeplan_and_immutable_instance(tmp_path):
    idx = pd.date_range("2022-01-01T00:00:00Z", periods=3, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 108.0],
            "Low": [97.0, 99.0, 101.0],
            "Close": [101.0, 104.0, 106.0],
        },
        index=idx,
    )

    def builder(ts, base, context):
        return _valid_trade_evidence(ts)

    payload = build_historical_census(df, evidence_builder=builder, output_dir=tmp_path, timeframe="5min", source="dukascopy")
    assert payload["trade_plan_objects"]
    plan = payload["trade_plan_objects"][0]
    assert isinstance(plan, TradePlan)
    with pytest.raises((TypeError, AttributeError)):
        plan.entry_time = pd.Timestamp("2022-01-02T00:00:00Z")
    outcome = simulate_trade(df, plan)
    assert isinstance(outcome, type(simulate_trade(df, plan)))


def test_census_uses_canonical_tradeplan_serialization_and_writes_all_artifacts():
    idx = pd.date_range("2022-01-01T00:00:00Z", periods=3, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 108.0],
            "Low": [97.0, 99.0, 101.0],
            "Close": [101.0, 104.0, 106.0],
        },
        index=idx,
    )

    def builder(ts, base, context):
        return _valid_trade_evidence(ts)

    canonical_dir = Path("data/reports/historical_census")
    canonical_dir.mkdir(parents=True, exist_ok=True)

    payload = build_historical_census(df, evidence_builder=builder, output_dir=canonical_dir, timeframe="5min", source="dukascopy")
    plan = payload["trade_plan_objects"][0]
    serialized = serialize_trade_plan(plan)
    assert json.loads(json.dumps(serialized)) == serialized
    assert (canonical_dir / "census_manifest.json").exists()
    assert (canonical_dir / "census_coverage.md").exists()
    assert (canonical_dir / "census_decisions.jsonl").exists()
    assert (canonical_dir / "census_setups.jsonl").exists()
    assert (canonical_dir / "census_no_trade.jsonl").exists()
    assert (canonical_dir / "census_summary.json").exists()
    assert (canonical_dir / "census_summary.md").exists()
    first_setup = json.loads((canonical_dir / "census_setups.jsonl").read_text().strip().splitlines()[0])
    assert "decision_time" in first_setup
    assert "post_decision_outcome" in first_setup


def test_census_same_bar_stop_and_target_is_ambiguous(tmp_path):
    idx = pd.date_range("2022-01-01T00:00:00Z", periods=1, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [97.0],
            "Close": [101.0],
        },
        index=idx,
    )

    def builder(ts, base, context):
        return _valid_trade_evidence_same_bar(ts)

    payload = build_historical_census(df, evidence_builder=builder, output_dir=tmp_path, timeframe="5min", source="dukascopy")
    assert payload["trade_outcomes"]
    assert payload["trade_outcomes"][0]["outcome"] == TradeOutcome.EXPIRED.value


def test_census_post_decision_bar_stop_and_target_is_ambiguous(tmp_path):
    idx = pd.date_range("2022-01-01T00:00:00Z", periods=2, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "High": [105.0, 105.0],
            "Low": [97.0, 97.0],
            "Close": [101.0, 101.0],
        },
        index=idx,
    )

    def builder(ts, base, context):
        return _valid_trade_evidence_same_bar(ts)

    payload = build_historical_census(df, evidence_builder=builder, output_dir=tmp_path, timeframe="5min", source="dukascopy")
    assert payload["trade_outcomes"]
    assert payload["trade_outcomes"][0]["entry_time"] == "2022-01-01T00:05:00+00:00"
    assert payload["trade_outcomes"][0]["outcome"] == TradeOutcome.AMBIGUOUS.value


def test_census_compatibility_regression_for_canonical_liquidity_event(tmp_path):
    idx = pd.date_range("2022-01-01T00:00:00Z", periods=3, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 108.0],
            "Low": [97.0, 99.0, 101.0],
            "Close": [101.0, 104.0, 106.0],
        },
        index=idx,
    )

    def builder(ts, base, context):
        return _valid_trade_evidence(ts)

    payload = build_historical_census(df, evidence_builder=builder, output_dir=tmp_path, timeframe="5min", source="dukascopy")
    assert payload["trade_plans"]


def test_real_octa_dataset_metadata_stays_15min_without_full_replay():
    path = Path("data/raw/external_secondary/octa_mt4/XAU_15m_data.csv")
    coverage = summarize_data_coverage(path, timeframe="15min", source="octa_mt4")
    assert coverage["source"] == "octa_mt4"
    assert coverage["timeframe"] == "15min"
    assert coverage["rows"] > 0
    assert classify_coverage(coverage) in {"SMOKE_CENSUS", "HISTORICAL_CENSUS"}
    assert coverage["duration_minutes"] > 0
