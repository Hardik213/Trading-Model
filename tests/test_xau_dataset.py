import pandas as pd
from src.xau_dataset import TIMEFRAMES, build_timeframes, discover_csv


def test_discover_csv_is_deterministic(tmp_path):
    (tmp_path / "b.csv").write_text("x")
    (tmp_path / "a.csv").write_text("x")
    assert [p.name for p in discover_csv(tmp_path)] == ["a.csv", "b.csv"]


def test_build_timeframes_preserves_raw_quality(tmp_path):
    raw = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-03-01T12:00:01Z",
            "2022-03-01T12:00:02Z",
            "2022-03-01T12:01:01Z",
        ]),
        "open": [10.0, 10.5, 11.0],
        "high": [10.0, 11.0, 12.0],
        "low": [10.0, 10.5, 10.8],
        "close": [10.0, 11.0, 11.5],
        "volume": [1, 2, 3],
    })
    report = build_timeframes(raw, tmp_path / "out")
    assert report.raw_rows == 3
    assert report.invalid_ohlc_rows == 0
    assert "5min" in report.bars
    assert (tmp_path / "out" / "dataset_manifest.json").exists()
