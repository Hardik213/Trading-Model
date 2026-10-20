"""Inspect Core v1 freeze readiness without loading data or running a replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core_v1_freeze_gate import assess_freeze_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/core_v1_development.json",
        help="JSON configuration path (default: development config)",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = assess_freeze_readiness(config)
    print(f"Status: {result['status']}")
    print(f"Configuration SHA-256: {result['configuration_sha256']}")
    print("Holdout run authorized: no (this checker never grants authorization)")
    for blocker in result["blockers"]:
        print(f"BLOCKER: {blocker}")
    return 0 if result["status"] == "READY_FOR_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
