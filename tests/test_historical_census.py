from pathlib import Path

import pandas as pd

from src.historical_census import (
    CensusReport,
    classify_coverage,
    summarize_data_coverage,
    build_historical_census,
)
from src.sniper_setup import PrecisionEvidence


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
