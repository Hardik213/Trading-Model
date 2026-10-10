from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "evaluation_protocol.md"
VARIANT_LOG = ROOT / "docs" / "strategy_variant_log.csv"
EXPECTED_COLUMNS = (
    "variant_id",
    "date_created_utc",
    "code_commit",
    "configuration_sha256",
    "development_dataset_version",
    "rule_parameter_changes",
    "hypothesis",
    "metrics_examined",
    "result_location",
    "decision",
    "decision_rationale",
    "holdout_outcomes_accessed",
    "parent_variant_id",
)


def test_protocol_freezes_periods_metrics_and_holdout_rules():
    text = PROTOCOL.read_text(encoding="utf-8")
    required = (
        "**Protocol:** v1",
        "**Status:** FROZEN",
        "[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)",
        "[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)",
        "only raw-data presence and integrity were checked on 2025 data",
        "no strategy outcomes, performance metrics, or charts were inspected",
        "must not inspect 2025 strategy results",
        "requires explicit user authorization",
        "100 completed out-of-sample trades",
        "moving-block bootstrap",
        "Profit factor",
        "maximum drawdown",
        "commission",
        "slippage",
        "spread",
        "INSUFFICIENT EVIDENCE",
        "No strategy backtest, strategy experiment",
    )
    for phrase in required:
        assert phrase in text


def test_variant_log_schema_is_stable_and_ids_are_unique():
    with VARIANT_LOG.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == EXPECTED_COLUMNS
        rows = list(reader)

    variant_ids = [row["variant_id"] for row in rows]
    assert all(variant_ids)
    assert len(variant_ids) == len(set(variant_ids))
    assert all(
        row["holdout_outcomes_accessed"] in {"NO", "YES", "UNKNOWN"}
        for row in rows
    )
