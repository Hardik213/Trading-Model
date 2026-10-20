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
