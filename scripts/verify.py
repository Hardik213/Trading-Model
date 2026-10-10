from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_paths import DATA_ROOT_ENV, data_root  # noqa: E402
from src.dataset_manifest import (  # noqa: E402
    DatasetIntegrityError,
    MANIFEST_PATH,
    aggregate_manifest_hash,
    build_dataset_manifest,
    verify_pinned_manifest,
    write_dataset_manifest,
)


def _run_git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_baseline() -> tuple[str, str, bool]:
    branch = _run_git("branch", "--show-current")
    commit = _run_git("rev-parse", "--verify", "HEAD")
    status = _run_git("status", "--porcelain")
    if not branch or len(commit) != 40:
        raise RuntimeError("Git branch or commit cannot be resolved.")
    return branch, commit, not bool(status)


def _test_counts(output: str) -> str:
    for line in reversed(output.splitlines()):
        if " passed" in line or " failed" in line or " skipped" in line:
            return line.strip()
    return "Pytest did not emit a parseable summary."


def _run_pytest() -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-rs"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        print(result.stdout, end="")
    else:
        for line in result.stdout.splitlines():
            if " SKIPPED " in line:
                print(f"Dataset/test skip: {line.strip()}")
        collected = next(
            (
                line.strip()
                for line in result.stdout.splitlines()
                if "collected " in line
            ),
            "collected count unavailable",
        )
        summary = _test_counts(result.stdout)
        print(f"Pytest: {collected}; {summary}")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode, result.stdout + result.stderr


def _diff_check() -> tuple[int, str]:
    result = subprocess.run(
        ["git", "diff", "--check"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode == 0:
        print("git diff --check: clean")
    else:
        print("git diff --check: whitespace errors found")
    return result.returncode, result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Git, canonical datasets, pytest, and diff whitespace."
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="record the currently verified canonical dataset baseline",
    )
    args = parser.parse_args()

    print(f"Repository: {REPOSITORY_ROOT}")
    try:
        branch, commit, clean = _git_baseline()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Git baseline: INVALID ({exc})")
        return 2
    print(f"Branch: {branch}")
    print(f"Commit: {commit}")
    print(f"Working tree: {'clean' if clean else 'dirty'}")
    if _run_git("ls-files", "--", "data/raw"):
        print("Raw dataset files are tracked by Git; refusing verification.")
        return 2

    root = data_root()
    print(f"{DATA_ROOT_ENV}: {root}")
    manifest = None
    try:
        manifest = build_dataset_manifest(root)
    except (DatasetIntegrityError, OSError, UnicodeError, csv.Error) as exc:
        print(f"Dataset integrity: FAILED ({exc})")
    else:
        first_hash = manifest["aggregate_sha256"]
        second_hash = aggregate_manifest_hash(manifest)
        print(
            "Dukascopy: "
            f"{manifest['datasets']['dukascopy']['file_count']} monthly files, "
            f"{manifest['datasets']['dukascopy']['data_rows']} rows"
        )
        print(
            "Octa: "
            f"{manifest['datasets']['octa_mt4']['file']['data_rows']} rows, "
            f"sha256 {manifest['datasets']['octa_mt4']['file']['sha256']}"
        )
        print(f"Dataset manifest SHA-256: {first_hash}")
        if second_hash != first_hash:
            print("Dataset manifest determinism: FAILED")
            manifest = None
        else:
            print(f"Second canonical manifest hash: {second_hash} (identical)")
        if args.write_manifest and manifest is not None:
            write_dataset_manifest(manifest)
            print(f"Recorded manifest: {MANIFEST_PATH}")
            return 0
        if manifest is not None:
            try:
                verify_pinned_manifest(manifest)
            except (DatasetIntegrityError, OSError, json.JSONDecodeError) as exc:
                print(f"Dataset integrity: FAILED ({exc})")
                manifest = None
            else:
                print("Dataset integrity: PASS (pinned manifest matches)")

    pytest_code, _ = _run_pytest()
    diff_code, _ = _diff_check()
    if manifest is None:
        print("Overall verification: FAILED (dataset integrity)")
        return 3
    if pytest_code != 0:
        print(f"Overall verification: FAILED (pytest exit code {pytest_code})")
        return 4
    if diff_code != 0:
        print("Overall verification: FAILED (git diff --check)")
        return 5
    print("Overall verification: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Verification interrupted.")
