from __future__ import annotations

import pytest

from src.data_paths import (
    canonical_dataset_paths,
    data_root,
    data_root_is_configured,
    require_dataset_path,
)


@pytest.fixture
def require_canonical_dataset():
    def require(name: str):
        try:
            return require_dataset_path(name)
        except FileNotFoundError as exc:
            if data_root_is_configured() or any(data_root().glob("*")):
                pytest.fail(str(exc), pytrace=False)
            pytest.skip(str(exc))

    return require
