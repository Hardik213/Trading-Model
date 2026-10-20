"""Read-only configuration hashing and freeze-readiness assessment.

This module never reads market data, starts a replay, or evaluates outcomes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 for a JSON-compatible configuration mapping."""
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_blockers(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return explicit blockers, failing closed for malformed/unfrozen input."""
    blockers: list[str] = []
    if config.get("schema_version") != 1:
        blockers.append("Unsupported or missing configuration schema_version.")
    if config.get("configuration_status") != "FROZEN":
        blockers.append("Configuration status is not FROZEN.")
    declared = config.get("freeze_blockers")
    if not isinstance(declared, list) or any(
        not isinstance(item, str) or not item.strip() for item in declared
    ):
        blockers.append("freeze_blockers must be a list of non-empty strings.")
    else:
        blockers.extend(declared)
    periods = config.get("periods_utc")
    holdout = periods.get("holdout", {}) if isinstance(periods, Mapping) else {}
    if not isinstance(holdout, Mapping) or holdout.get("authorization") != "AUTHORIZED":
        blockers.append("Holdout authorization is not explicitly AUTHORIZED.")
    execution = config.get("execution_configuration")
    required = (
        "starting_equity",
        "account_currency",
        "contract_size",
        "minimum_quantity",
        "quantity_step",
        "account_currency_conversion_policy",
        "commission_per_unit_per_side",
        "slippage_per_unit_per_side",
    )
    if not isinstance(execution, Mapping) or any(
        execution.get(key) is None for key in required
    ):
        blockers.append("Execution configuration is incomplete.")
    return tuple(dict.fromkeys(blockers))


def assess_freeze_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize readiness; readiness is not permission to run a holdout."""
    blockers = freeze_blockers(config)
    return {
        "status": "READY_FOR_REVIEW" if not blockers else "NOT_READY",
        "configuration_sha256": canonical_config_sha256(config),
        "blockers": blockers,
        "holdout_run_authorized": False,
    }
