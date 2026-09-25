import pandas as pd

from src.dukascopy_csv import quality_report, read_many, read_csv, to_ohlc


def _make_csv(path):
    rows = [
        {
            "Etc/UTC": "2022-03-01T12:00:00+00:00",
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 10,
        },
        {
            "Etc/UTC": "2022-03-01T12:00:00+00:00",
            "Open": 101.0,
            "High": 101.5,
            "Low": 100.0,
            "Close": 101.0,
            "Volume": 20,
        },
        {
            "Etc/UTC": "2022-03-01T12:00:01+00:00",
            "Open": 101.0,
            "High": 102.0,
            "Low": 100.5,
            "Close": 101.5,
            "Volume": 30,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_read_many_keeps_duplicate_timestamps(tmp_path):
    one = tmp_path / "one.csv"
    two = tmp_path / "two.csv"
    _make_csv(one)
    pd.DataFrame([
        {
            "Etc/UTC": "2022-03-01T12:00:01+00:00",
            "Open": 101.5,
            "High": 103.0,
            "Low": 101.0,
            "Close": 102.5,
            "Volume": 40,
        }
    ]).to_csv(two, index=False)

    df = read_many([one, two])
    assert len(df) == 4
    assert df["timestamp"].duplicated().sum() == 2


def test_quality_report_counts_duplicates_and_range(tmp_path):
    path = tmp_path / "sample.csv"
    _make_csv(path)
    df = read_many([path])
    q = quality_report(df, files=1)
    assert q["files"] == 1
    assert q["duplicate_timestamps"] == 2
    assert q["timezone"] == "UTC"
    assert q["start"] == "2022-03-01T12:00:00+00:00"
    assert q["end"] == "2022-03-01T12:00:01+00:00"


def test_to_ohlc_aggregates_duplicate_ticks_without_dropping_them():
    raw = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2022-03-01T12:00:00Z"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10},
            {"timestamp": pd.Timestamp("2022-03-01T12:00:00Z"), "open": 101.0, "high": 101.5, "low": 100.0, "close": 101.0, "volume": 20},
            {"timestamp": pd.Timestamp("2022-03-01T12:00:01Z"), "open": 101.0, "high": 102.0, "low": 100.5, "close": 101.5, "volume": 30},
        ]
    )
    bars = to_ohlc(raw, "1s")
    assert len(bars) == 2
    assert bars.iloc[0]["volume"] == 30
    assert bars.iloc[0]["high"] == 101.5
    assert bars.iloc[1]["close"] == 101.5
