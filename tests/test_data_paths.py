from __future__ import annotations

import pytest

from src.data_paths import (
    REPOSITORY_ROOT,
    canonical_dataset_paths,
    data_root,
    require_dataset_path,
)


def test_relative_data_root_is_resolved_from_repository_root(monkeypatch):
    monkeypatch.setenv("TRADING_DATA_ROOT", "external-data")
    assert data_root() == (REPOSITORY_ROOT / "external-data").resolve()


def test_default_data_root_is_repository_local(monkeypatch):
    monkeypatch.delenv("TRADING_DATA_ROOT", raising=False)
    assert data_root() == (REPOSITORY_ROOT / "data" / "raw").resolve()


def test_missing_configured_dataset_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_DATA_ROOT", str(tmp_path))
    with pytest.raises(
        FileNotFoundError,
        match="Canonical dukascopy dataset is missing.*TRADING_DATA_ROOT",
    ):
        require_dataset_path("dukascopy")


def test_canonical_paths_share_the_configured_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_DATA_ROOT", str(tmp_path))
    paths = canonical_dataset_paths()
    assert paths["dukascopy"] == tmp_path / "dukascopy"
    assert paths["octa_mt4"] == (
        tmp_path / "external_secondary" / "octa_mt4" / "XAU_15m_data.csv"
    )
