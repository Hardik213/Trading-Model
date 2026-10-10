"""Canonical local paths for raw trading datasets."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_ENV = "TRADING_DATA_ROOT"


def data_root() -> Path:
    """Resolve raw data root; relative overrides are repository-root relative."""
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured is None or not configured.strip():
        return (REPOSITORY_ROOT / "data" / "raw").resolve()
    candidate = Path(configured.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def data_root_is_configured() -> bool:
    value = os.environ.get(DATA_ROOT_ENV)
    return value is not None and bool(value.strip())


def canonical_dataset_paths(root: Path | None = None) -> dict[str, Path]:
    raw_root = data_root() if root is None else root.resolve()
    return {
        "dukascopy": raw_root / "dukascopy",
        "octa_mt4": raw_root / "external_secondary" / "octa_mt4" / "XAU_15m_data.csv",
    }


def require_dataset_path(name: str, root: Path | None = None) -> Path:
    try:
        path = canonical_dataset_paths(root)[name]
    except KeyError as exc:
        raise ValueError(f"Unknown canonical dataset: {name!r}") from exc
    if not path.exists():
        configured = (
            f"{DATA_ROOT_ENV}={data_root()}"
            if data_root_is_configured()
            else f"{DATA_ROOT_ENV} is unset (default root: {data_root()})"
        )
        raise FileNotFoundError(
            f"Canonical {name} dataset is missing at {path}; {configured}. "
            f"Set {DATA_ROOT_ENV} to the directory containing dukascopy/ and "
            "external_secondary/."
        )
    return path
