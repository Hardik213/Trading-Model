import pandas as pd
from src.historical_census import CensusReport


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
