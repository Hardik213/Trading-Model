import json
from pathlib import Path

import pytest

from src.core_v1_freeze_gate import assess_freeze_readiness, canonical_config_sha256


def test_canonical_hash_ignores_mapping_insertion_order():
    assert canonical_config_sha256({"b": 2, "a": 1}) == canonical_config_sha256(
        {"a": 1, "b": 2}
    )


def test_canonical_hash_rejects_non_finite_json_numbers():
    with pytest.raises(ValueError):
        canonical_config_sha256({"bad": float("nan")})


def test_development_configuration_fails_closed_without_evaluation():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/core_v1_development.json").read_text())
    result = assess_freeze_readiness(config)
    assert result["status"] == "NOT_READY"
    assert result["holdout_run_authorized"] is False
    assert "Configuration status is not FROZEN." in result["blockers"]
    assert "Execution configuration is incomplete." in result["blockers"]


def _valid_frozen_configuration():
    return {
        "schema_version": 1,
        "configuration_status": "FROZEN",
        "freeze_blockers": [],
        "periods_utc": {"holdout": {"authorization": "AUTHORIZED"}},
        "execution_configuration": {
            "starting_equity": 100000,
            "account_currency": "USD",
            "contract_size": 1,
            "minimum_quantity": 1,
            "quantity_step": 1,
            "account_currency_conversion_policy": "NONE",
            "commission_per_unit_per_side": 0,
            "slippage_per_unit_per_side": 0,
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "expected_blocker"),
    [
        (
            "periods_utc",
            [],
            "Holdout authorization is not explicitly AUTHORIZED.",
        ),
        (
            "periods_utc",
            {"holdout": []},
            "Holdout authorization is not explicitly AUTHORIZED.",
        ),
        (
            "freeze_blockers",
            None,
            "freeze_blockers must be a list of non-empty strings.",
        ),
        (
            "freeze_blockers",
            [""],
            "freeze_blockers must be a list of non-empty strings.",
        ),
    ],
)
def test_malformed_freeze_configuration_fails_closed(
    field, value, expected_blocker
):
    config = _valid_frozen_configuration()
    config[field] = value

    result = assess_freeze_readiness(config)

    assert result["status"] == "NOT_READY"
    assert result["holdout_run_authorized"] is False
    assert expected_blocker in result["blockers"]


def test_missing_freeze_blockers_fails_closed_without_holdout_authorization():
    config = _valid_frozen_configuration()
    del config["freeze_blockers"]

    result = assess_freeze_readiness(config)

    assert result["status"] == "NOT_READY"
    assert result["holdout_run_authorized"] is False
    assert "freeze_blockers must be a list of non-empty strings." in result[
        "blockers"
    ]
