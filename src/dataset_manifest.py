"""Streaming integrity manifest for canonical Dukascopy and Octa datasets."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from .data_paths import REPOSITORY_ROOT, canonical_dataset_paths, data_root


MANIFEST_PATH = REPOSITORY_ROOT / "data" / "dataset_manifest.json"
EXPECTED_OCTA_SHA256 = "a48336b48c030bc407a5b53123fa46b328b74215bd7865f16890b3147a3d70b6"
EXPECTED_OCTA_ROWS = 494_235
EXPECTED_DUKASCOPY_FILES = 48
EXPECTED_DUKASCOPY_ROWS_HINT = 208_940_811
KNOWN_NONCANONICAL_DUKASCOPY_FILE = (
    "XAU-USD_100Tick_BID_2022-03-01_12_00-12_00_Etc_UTC.csv"
)
TICK_NAME = re.compile(
    r"^xauusd-tick-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})\.csv$",
    re.IGNORECASE,
)
TICK_HEADER = ["timestamp", "askPrice", "bidPrice"]
OCTA_HEADER = ["Date", "Open", "High", "Low", "Close", "Volume"]


class DatasetIntegrityError(ValueError):
    """Canonical dataset contents do not satisfy their pinned contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def aggregate_manifest_hash(manifest_body: dict[str, object]) -> str:
    """Hash sorted-key compact UTF-8 JSON, excluding the aggregate hash field."""
    body = {key: value for key, value in manifest_body.items() if key != "aggregate_sha256"}
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _hashed_lines(stream, digest):
    first_line = True
    for raw_line in stream:
        digest.update(raw_line)
        yield raw_line.decode("utf-8-sig" if first_line else "utf-8")
        first_line = False


def _month_epoch_bounds(start_text: str, end_text: str) -> tuple[int, int]:
    start = datetime.strptime(start_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (
        int((start - epoch).total_seconds() * 1000),
        int((end - epoch).total_seconds() * 1000),
    )


def _scan_tick_file(path: Path, root: Path) -> dict[str, object]:
    match = TICK_NAME.fullmatch(path.name)
    if match is None:
        raise DatasetIntegrityError(f"Unexpected Dukascopy tick filename: {path.name}")
    start_ms, end_ms = _month_epoch_bounds(*match.groups())
    digest = hashlib.sha256()
    rows = 0
    previous_timestamp: int | None = None

    with path.open("rb") as stream:
        records = csv.reader(_hashed_lines(stream, digest), strict=True)
        header = next(records, None)
        if header != TICK_HEADER:
            raise DatasetIntegrityError(
                f"{path}: expected exact header {TICK_HEADER!r}, got {header!r}"
            )
        for row_number, fields in enumerate(records, start=1):
            if len(fields) != len(TICK_HEADER):
                raise DatasetIntegrityError(
                    f"{path}: row {row_number} has {len(fields)} fields; expected 3"
                )
            try:
                timestamp_ms = int(fields[0])
                ask = float(fields[1])
                bid = float(fields[2])
            except (TypeError, ValueError, OverflowError) as exc:
                raise DatasetIntegrityError(
                    f"{path}: row {row_number} is not an integer-ms timestamp and numeric quote"
                ) from exc
            if not start_ms <= timestamp_ms < end_ms:
                raise DatasetIntegrityError(
                    f"{path}: row {row_number} timestamp falls outside filename month"
                )
            if previous_timestamp is not None and timestamp_ms < previous_timestamp:
                raise DatasetIntegrityError(
                    f"{path}: timestamp decreases at data row {row_number}"
                )
            if (
                not math.isfinite(ask)
                or not math.isfinite(bid)
                or ask <= 0
                or bid <= 0
                or ask < bid
            ):
                raise DatasetIntegrityError(
                    f"{path}: row {row_number} has invalid or crossed quotes"
                )
            previous_timestamp = timestamp_ms
            rows += 1

    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "data_rows": rows,
    }


def _scan_octa_file(path: Path, root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    rows = 0
    previous_timestamp: datetime | None = None

    with path.open("rb") as stream:
        records = csv.reader(
            _hashed_lines(stream, digest),
            delimiter=";",
            strict=True,
        )
        header = next(records, None)
        if header != OCTA_HEADER:
            raise DatasetIntegrityError(
                f"{path}: expected exact semicolon-delimited header {OCTA_HEADER!r}, "
                f"got {header!r}"
            )
        for row_number, fields in enumerate(records, start=1):
            if len(fields) != len(OCTA_HEADER):
                raise DatasetIntegrityError(
                    f"{path}: row {row_number} has {len(fields)} fields; expected 6"
                )
            try:
                timestamp = datetime.strptime(fields[0], "%Y.%m.%d %H:%M")
                prices = tuple(float(value) for value in fields[1:])
            except (TypeError, ValueError, OverflowError) as exc:
                raise DatasetIntegrityError(
                    f"{path}: row {row_number} has invalid native Octa values"
                ) from exc
            if timestamp.minute % 15 or timestamp.second or timestamp.microsecond:
                raise DatasetIntegrityError(
                    f"{path}: row {row_number} is not on native 15-minute granularity"
                )
            if not all(math.isfinite(value) for value in prices):
                raise DatasetIntegrityError(
                    f"{path}: row {row_number} contains a non-finite OHLCV value"
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise DatasetIntegrityError(
                    f"{path}: timestamps are not strictly increasing at row {row_number}"
                )
            previous_timestamp = timestamp
            rows += 1

    actual_hash = digest.hexdigest()
    if actual_hash != EXPECTED_OCTA_SHA256:
        raise DatasetIntegrityError(
            f"Octa SHA-256 mismatch: expected {EXPECTED_OCTA_SHA256}, got {actual_hash}"
        )
    if rows != EXPECTED_OCTA_ROWS:
        raise DatasetIntegrityError(
            f"Octa row count mismatch: expected {EXPECTED_OCTA_ROWS}, got {rows}"
        )
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": actual_hash,
        "data_rows": rows,
        "delimiter": ";",
        "source_timezone": "UTC+3",
        "native_granularity": "15min",
        "schema": OCTA_HEADER,
    }


def build_dataset_manifest(root: Path | None = None) -> dict[str, object]:
    raw_root = data_root() if root is None else root.resolve()
    paths = canonical_dataset_paths(raw_root)
    dukascopy_root = paths["dukascopy"]
    octa_path = paths["octa_mt4"]
    if not dukascopy_root.is_dir():
        raise DatasetIntegrityError(
            f"Canonical Dukascopy directory is missing: {dukascopy_root}"
        )
    if not octa_path.is_file():
        raise DatasetIntegrityError(f"Canonical Octa file is missing: {octa_path}")

    tick_files = sorted(
        (path for path in dukascopy_root.rglob("*.csv") if path.is_file()),
        key=lambda item: item.as_posix().lower(),
    )
    monthly_files = [
        path for path in tick_files if TICK_NAME.fullmatch(path.name) is not None
    ]
    if len(monthly_files) != EXPECTED_DUKASCOPY_FILES:
        raise DatasetIntegrityError(
            f"Expected {EXPECTED_DUKASCOPY_FILES} monthly Dukascopy tick files, "
            f"found {len(monthly_files)}"
        )
    unexpected = [
        path
        for path in tick_files
        if TICK_NAME.fullmatch(path.name) is None
        and not (
            path.parent == dukascopy_root
            and path.name == KNOWN_NONCANONICAL_DUKASCOPY_FILE
        )
    ]
    if unexpected:
        raise DatasetIntegrityError(
            "Unexpected CSVs under canonical Dukascopy root: "
            f"{[str(path.relative_to(raw_root)) for path in unexpected]!r}"
        )

    expected_months: list[tuple[str, str]] = []
    year, month = 2022, 1
    for _ in range(EXPECTED_DUKASCOPY_FILES):
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        expected_months.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        year, month = end.year, end.month

    actual_months = sorted(
        TICK_NAME.fullmatch(path.name).groups() for path in monthly_files
    )
    if actual_months != expected_months:
        raise DatasetIntegrityError(
            "Dukascopy monthly filename coverage must be Jan 2022 through Dec 2025"
        )

    tick_entries = [_scan_tick_file(path, raw_root) for path in monthly_files]
    total_rows = sum(int(item["data_rows"]) for item in tick_entries)
    if total_rows != EXPECTED_DUKASCOPY_ROWS_HINT:
        raise DatasetIntegrityError(
            "Dukascopy total row count mismatch: "
            f"expected {EXPECTED_DUKASCOPY_ROWS_HINT}, got {total_rows}"
        )
    octa_entry = _scan_octa_file(octa_path, raw_root)
    body: dict[str, object] = {
        "schema_version": 1,
        "datasets": {
            "dukascopy": {
                "relative_root": "dukascopy/",
                "instrument": "XAUUSD",
                "file_count": len(tick_entries),
                "data_rows": total_rows,
                "expected_rows_hint": EXPECTED_DUKASCOPY_ROWS_HINT,
                "schema": TICK_HEADER,
                "timestamp_convention": "Unix epoch milliseconds, UTC",
                "monthly_coverage": [
                    f"{start[:7]}" for start, _ in expected_months
                ],
                "excluded_noncanonical_file": {
                    "path": f"dukascopy/{KNOWN_NONCANONICAL_DUKASCOPY_FILE}",
                    "reason": "BID OHLCV export; not canonical bid/ask tick data",
                },
                "files": tick_entries,
            },
            "octa_mt4": {
                "instrument": "XAUUSD",
                "file": octa_entry,
            },
        },
    }
    body["aggregate_sha256"] = aggregate_manifest_hash(body)
    return body


def write_dataset_manifest(manifest: dict[str, object]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_pinned_manifest(
    actual: dict[str, object],
    path: Path = MANIFEST_PATH,
) -> None:
    if not path.is_file():
        raise DatasetIntegrityError(f"Pinned dataset manifest is missing: {path}")
    expected = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        expected_files = {
            item["path"]: item
            for item in expected["datasets"]["dukascopy"]["files"]
        }
        actual_files = {
            item["path"]: item
            for item in actual["datasets"]["dukascopy"]["files"]
        }
        mismatches = [
            name
            for name in sorted(set(expected_files) | set(actual_files))
            if expected_files.get(name) != actual_files.get(name)
        ]
        raise DatasetIntegrityError(
            "Live canonical datasets differ from the pinned manifest"
            + (f"; Dukascopy file mismatch: {mismatches!r}" if mismatches else "")
        )
    if aggregate_manifest_hash(actual) != actual["aggregate_sha256"]:
        raise DatasetIntegrityError("Pinned manifest aggregate hash is invalid")


__all__ = [
    "DatasetIntegrityError",
    "EXPECTED_DUKASCOPY_FILES",
    "EXPECTED_DUKASCOPY_ROWS_HINT",
    "EXPECTED_OCTA_ROWS",
    "EXPECTED_OCTA_SHA256",
    "MANIFEST_PATH",
    "aggregate_manifest_hash",
    "build_dataset_manifest",
    "verify_pinned_manifest",
    "write_dataset_manifest",
]
