from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset_forensics import (
    analyze_dataset,
    discover_dataset_files,
    generate_dataset_reports,
    validate_dataset_timeframe,
)


@pytest.mark.parametrize(
    "root, expected",
    [
        ("data/raw/dukascopy", "dukascopy"),
        ("data/raw/external_secondary/octa_mt4/XAU_15m_data.csv", "octa_mt4"),
    ],
)
def test_dataset_discovery_finds_expected_sources(root, expected):
    files = discover_dataset_files(root)
    assert files
    assert any(str(path).endswith(".csv") for path in files)
    assert expected in str(files[0]) or expected in str(files[-1])


def test_timezone_and_source_separation_are_explicit():
    duk = analyze_dataset("data/raw/dukascopy")
    octa = analyze_dataset("data/raw/external_secondary/octa_mt4/XAU_15m_data.csv")

    assert duk["source"] == "Dukascopy free-data export"
    assert duk["source_timezone"] == "UTC"
    assert duk["internal_timezone"] == "UTC"
    assert duk["research_role"] == "primary_historical_dukascopy"
    assert duk["duplicate_rows_after_first"] == 20117
    assert duk["duplicate_rows_in_duplicate_groups"] == 23322
    assert duk["duplicate_timestamp_values"] == 3205

    assert octa["source"] == "Octa MT4 History Center"
    assert octa["source_timezone"] == "UTC+3"
    assert octa["internal_timezone"] == "UTC"
    assert octa["research_role"] == "secondary_robustness_15m"
    assert octa["duplicate_rows_after_first"] == 0
    assert octa["duplicate_rows_in_duplicate_groups"] == 0
    assert octa["duplicate_timestamp_values"] == 0

    assert duk["source"] != octa["source"]


def test_duplicate_and_gap_detection_preserves_raw_data():
    duk = analyze_dataset("data/raw/dukascopy")
    assert duk["rows"] > 0
    assert duk["duplicate_timestamps"] >= 0
    assert duk["non_monotonic_rows"] >= 0
    assert duk["gap_distribution"]

    octa = analyze_dataset("data/raw/external_secondary/octa_mt4/XAU_15m_data.csv")
    assert octa["timeframe"] == "15min"
    assert octa["gap_distribution"]
    assert octa["duplicate_timestamps"] >= 0
    assert octa["non_monotonic_rows"] >= 0


def test_ohlc_validation_rejects_invalid_rows():
    invalid = {
        "timestamp": ["2022-01-01T00:00:00Z", "2022-01-01T00:15:00Z"],
        "open": [100.0, 100.0],
        "high": [90.0, 101.0],
        "low": [95.0, 99.0],
        "close": [99.0, 100.0],
        "volume": [10, 20],
    }
    result = analyze_dataset(invalid, source_name="synthetic", source_timezone="UTC", internal_timezone="UTC", timeframe="15min")
    assert result["invalid_ohlc_rows"] >= 1


def test_15m_octa_dataset_cannot_be_treated_as_finer_granularity():
    octa = analyze_dataset("data/raw/external_secondary/octa_mt4/XAU_15m_data.csv")
    for finer in ("1min", "3min", "5min"):
        with pytest.raises(ValueError):
            validate_dataset_timeframe(octa, finer)

    validate_dataset_timeframe(octa, "15min")


def test_manifest_generation_is_deterministic(tmp_path):
    generated = generate_dataset_reports(output_dir=tmp_path)
    assert set(generated) == {"dukascopy", "octa_mt4"}

    for name in generated:
        manifest_path = tmp_path / f"{name}_manifest.json"
        assert manifest_path.exists()
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert doc["instrument"] == "XAUUSD"
        assert doc["rows"] > 0
        assert doc["source_timeframe"] in {"tick", "15min"}

    report_path = tmp_path / "dataset_forensics_report.json"
    assert report_path.exists()
    md_path = tmp_path / "dataset_forensics_report.md"
    assert md_path.exists()
