from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATHS = OrderedDict(
    [
        (
            "dukascopy",
            REPO_ROOT / "data" / "raw" / "dukascopy",
        ),
        (
            "octa_mt4",
            REPO_ROOT / "data" / "raw" / "external_secondary" / "octa_mt4" / "XAU_15m_data.csv",
        ),
    ]
)


def discover_dataset_files(path_like: str | Path | None = None) -> list[Path]:
    """Return all CSV files associated with a dataset root or a single CSV path."""
    if path_like is None:
        files: list[Path] = []
        for candidate in DATASET_PATHS.values():
            files.extend(discover_dataset_files(candidate))
        return sorted(files, key=lambda p: str(p))

    candidate = Path(path_like)
    if candidate.is_dir():
        return sorted(candidate.rglob("*.csv"), key=lambda p: str(p))
    if candidate.is_file() and candidate.suffix.lower() == ".csv":
        return [candidate]
    return []


def _infer_source(path: Path) -> tuple[str, str, str, str, str, str]:
    name = path.name.lower()
    if "octa" in name or "xau_15m" in name or "15m" in name:
        return (
            "Octa MT4 History Center",
            "UTC+3",
            "UTC",
            "15min",
            "secondary_robustness_15m",
            "SECONDARY",
        )
    return (
        "Dukascopy free-data export",
        "UTC",
        "UTC",
        "tick",
        "primary_historical_dukascopy",
        "PRIMARY",
    )


def _normalise_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned.columns = [column.replace("\ufeff", "") for column in cleaned.columns]
    return cleaned


def _load_dataset_frame(path: Path) -> pd.DataFrame:
    if "octa" in path.name.lower() or "xau_15m" in path.name.lower() or "15m" in path.name.lower():
        frame = pd.read_csv(path, sep=";")
        frame = _normalise_source_frame(frame)
        frame["timestamp"] = pd.to_datetime(frame["Date"], format="%Y.%m.%d %H:%M", errors="coerce")
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("Etc/GMT-3").dt.tz_convert("UTC")
        return frame

    frame = pd.read_csv(path)
    frame = _normalise_source_frame(frame)
    frame["timestamp"] = pd.to_datetime(frame["Etc/UTC"], errors="coerce", utc=True)
    return frame


def _detect_invalid_ohlc(frame: pd.DataFrame) -> tuple[int, int, int]:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        return len(frame), 0, 0

    numeric = frame.copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    invalid = (
        numeric["High"].isna()
        | numeric["Low"].isna()
        | numeric["Open"].isna()
        | numeric["Close"].isna()
        | (numeric["High"] < numeric[["Open", "Close"]].max(axis=1))
        | (numeric["Low"] > numeric[["Open", "Close"]].min(axis=1))
        | (numeric["High"] < numeric["Low"])
        | (numeric["Volume"] < 0)
    )
    invalid_rows = int(invalid.sum())
    nan_total = int(numeric[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum())
    negative_volume = int((numeric["Volume"] < 0).sum())
    return invalid_rows, nan_total, negative_volume


def _summarise_gap_distribution(timestamps: pd.Series) -> dict[str, int]:
    positive = timestamps.diff().dropna()
    positive = positive[positive > pd.Timedelta(0)]
    if positive.empty:
        return {}
    counts = positive.value_counts().sort_index()
    return {str(value): int(count) for value, count in counts.items()}


def _largest_gaps(timestamps: pd.Series, top_n: int = 5) -> list[dict[str, Any]]:
    positive = timestamps.diff().dropna()
    positive = positive[positive > pd.Timedelta(0)]
    if positive.empty:
        return []

    items: list[dict[str, Any]] = []
    for delta in positive.sort_values(ascending=False).head(top_n).index:
        # The index above is a timedelta; the original gap is paired with the trailing timestamp.
        trailing = timestamps[timestamps.index[positive.index.get_loc(delta)]].__class__
        _ = trailing
    # Use a simpler, deterministic path based on the original series positions.
    series = timestamps.reset_index(drop=True)
    diffs = series.diff().dropna()
    positive_idx = diffs[diffs > pd.Timedelta(0)]
    gaps: list[dict[str, Any]] = []
    for idx, delta in positive_idx.sort_values(ascending=False).head(top_n).items():
        start = series.iloc[max(0, idx - 1)]
        end = series.iloc[idx]
        gaps.append(
            {
                "start_utc": pd.Timestamp(start).isoformat(),
                "end_utc": pd.Timestamp(end).isoformat(),
                "duration": str(delta),
            }
        )
    return gaps


def analyze_dataset(
    path_like: str | Path | dict[str, Any] | pd.DataFrame,
    *,
    source_name: str | None = None,
    source_timezone: str | None = None,
    internal_timezone: str | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Produce a forensic manifest for a single historical dataset source."""
    if isinstance(path_like, dict):
        frame = pd.DataFrame(path_like)
        source_name = source_name or "synthetic"
        source_timezone = source_timezone or "UTC"
        internal_timezone = internal_timezone or "UTC"
        timeframe = timeframe or "15min"
        if "timestamp" not in frame.columns:
            raise ValueError("Synthetic data must include a 'timestamp' column")
    elif isinstance(path_like, pd.DataFrame):
        frame = path_like.copy()
        source_name = source_name or "synthetic"
        source_timezone = source_timezone or "UTC"
        internal_timezone = internal_timezone or "UTC"
        timeframe = timeframe or "15min"
        if "timestamp" not in frame.columns:
            raise ValueError("Synthetic data must include a 'timestamp' column")
    else:
        csv_path = Path(path_like)
        if csv_path.is_dir():
            csv_path = discover_dataset_files(csv_path)[0]
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {csv_path}")
        inferred_source_name, inferred_source_timezone, inferred_internal_timezone, inferred_timeframe, _, _ = _infer_source(csv_path)
        source_name = source_name or inferred_source_name
        source_timezone = source_timezone or inferred_source_timezone
        internal_timezone = internal_timezone or inferred_internal_timezone
        timeframe = timeframe or inferred_timeframe
        frame = _load_dataset_frame(csv_path)

    frame = _normalise_source_frame(frame)
    column_map = {str(column).strip(): str(column).strip() for column in frame.columns}
    for lower_name, canonical_name in {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }.items():
        candidates = [key for key in frame.columns if str(key).lower() == lower_name]
        if candidates:
            frame[canonical_name] = pd.to_numeric(frame[candidates[0]], errors="coerce")
    if "timestamp" not in frame.columns and "Timestamp" in frame.columns:
        frame["timestamp"] = frame["Timestamp"]

    raw_timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    valid_timestamps = raw_timestamps.dropna().sort_values()
    gap_distribution = _summarise_gap_distribution(valid_timestamps)
    invalid_ohlc_rows, nan_total, negative_volume = _detect_invalid_ohlc(frame)
    duplicate_rows_after_first = int(valid_timestamps.duplicated(keep="first").sum())
    duplicate_rows_in_duplicate_groups = int(valid_timestamps.duplicated(keep=False).sum())
    duplicate_timestamp_values = int((valid_timestamps.value_counts() > 1).sum())

    metadata: dict[str, Any] = {
        "dataset_name": source_name,
        "source": source_name,
        "source_timezone": source_timezone,
        "internal_timezone": internal_timezone,
        "source_timeframe": timeframe,
        "timeframe": timeframe,
        "instrument": "XAUUSD",
        "research_role": "primary_historical_dukascopy" if "Dukascopy" in source_name else "secondary_robustness_15m",
        "research_classification": "PRIMARY" if "Dukascopy" in source_name else "SECONDARY",
        "rows": int(len(frame)),
        "timestamp_column": "timestamp",
        "first_timestamp_utc": valid_timestamps.min().isoformat() if not valid_timestamps.empty else None,
        "last_timestamp_utc": valid_timestamps.max().isoformat() if not valid_timestamps.empty else None,
        "duplicate_rows_after_first": duplicate_rows_after_first,
        "duplicate_rows_in_duplicate_groups": duplicate_rows_in_duplicate_groups,
        "duplicate_timestamp_values": duplicate_timestamp_values,
        "duplicate_timestamps": duplicate_rows_after_first,
        "non_monotonic_rows": int(valid_timestamps.diff().lt(pd.Timedelta(0)).sum()),
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "nan_ohlc_values": nan_total,
        "negative_volume_rows": negative_volume,
        "volume_semantics": "UNKNOWN",
        "gap_distribution": gap_distribution,
        "largest_gaps": _largest_gaps(valid_timestamps),
        "status": "forensic_review_only",
        "safe_for_forensic_use": invalid_ohlc_rows == 0 and negative_volume == 0,
        "notes": [
            "Raw observations are preserved without forward-filling or synthetic candle generation.",
            "Duplicate timestamps are kept for Dukascopy raw tick observations and are not silently dropped.",
            "Octa MT4 values are normalized from local UTC+3 provenance to internal UTC before use.",
            "The 15-minute Octa dataset is treated as a 15min source only and not coerced to a finer timeframe.",
            "No missing timestamps are filled or synthesized; timestamps are validated as-is from the originating raw file.",
            "Octa 15M history is never resampled downward into 5M, 3M, or 1M series for research use.",
        ],
    }
    return metadata


def validate_dataset_timeframe(dataset: dict[str, Any], requested_timeframe: str) -> None:
    actual = str(dataset.get("timeframe") or dataset.get("source_timeframe") or "").strip()
    if actual == "15min" and requested_timeframe in {"1min", "3min", "5min"}:
        raise ValueError("15min source data cannot be silently treated as a finer-granularity dataset.")
    if requested_timeframe == actual:
        return
    if actual == "tick" and requested_timeframe in {"1min", "3min", "5min", "15min"}:
        return
    if actual == "tick" and requested_timeframe == "tick":
        return
    if actual == "15min" and requested_timeframe == "15min":
        return
    raise ValueError(f"Requested timeframe {requested_timeframe!r} is incompatible with dataset timeframe {actual!r}.")


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Phase 10.6A Dataset Forensics / Validation",
        "",
        "This report intentionally performs provenance and raw-data validation only. It does not modify the canonical ICT 2022 strategy logic, sniper gate, execution logic, or live-trading configuration.",
        "",
        "## Summary",
        "",
        f"- Generated at UTC: {report.get('generated_at_utc', 'n/a')}",
        f"- Dataset count: {report.get('dataset_count', 0)}",
        f"- Status: {report.get('status', 'forensic_review_only')}",
        "",
    ]

    datasets = report.get("datasets", {})
    for source_name, metadata in datasets.items():
        lines.extend(
            [
                f"## {metadata.get('research_classification', 'UNKNOWN')} — {source_name}",
                "",
                f"- Source: {metadata.get('source', 'unknown')}",
                f"- Source timezone: {metadata.get('source_timezone', 'unknown')}",
                f"- Internal timezone: {metadata.get('internal_timezone', 'unknown')}",
                f"- Timeframe: {metadata.get('timeframe', 'unknown')}",
                f"- Rows: {metadata.get('rows', 0)}",
                f"- First timestamp (UTC): {metadata.get('first_timestamp_utc', 'n/a')}",
                f"- Last timestamp (UTC): {metadata.get('last_timestamp_utc', 'n/a')}",
                f"- Duplicate rows after first: {metadata.get('duplicate_rows_after_first', 0)}",
                f"- Duplicate rows in duplicate groups: {metadata.get('duplicate_rows_in_duplicate_groups', 0)}",
                f"- Duplicate timestamp values: {metadata.get('duplicate_timestamp_values', 0)}",
                f"- Non-monotonic rows: {metadata.get('non_monotonic_rows', 0)}",
                f"- Invalid OHLC rows: {metadata.get('invalid_ohlc_rows', 0)}",
                f"- Negative volume rows: {metadata.get('negative_volume_rows', 0)}",
                f"- Volume semantics: {metadata.get('volume_semantics', 'UNKNOWN')}",
                "",
            ]
        )

        gap_distribution = metadata.get("gap_distribution", {})
        if gap_distribution:
            lines.append("### Gap distribution")
            lines.append("")
            for bucket, count in gap_distribution.items():
                lines.append(f"- {bucket}: {count}")
            lines.append("")

        largest_gaps = metadata.get("largest_gaps", [])
        if largest_gaps:
            lines.append("### Largest gaps")
            lines.append("")
            for gap in largest_gaps:
                lines.append(f"- {gap['start_utc']} -> {gap['end_utc']} ({gap['duration']})")
            lines.append("")

        lines.append("### Notes")
        lines.append("")
        for note in metadata.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def generate_dataset_reports(output_dir: str | Path = "data/reports") -> dict[str, dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict[str, Any]] = {}
    for dataset_name, candidate in DATASET_PATHS.items():
        manifest = analyze_dataset(candidate)
        manifests[dataset_name] = manifest
        manifest_path = output_path / f"{dataset_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report_payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_count": len(manifests),
        "status": "forensic_review_only",
        "datasets": manifests,
    }
    report_path = output_path / "dataset_forensics_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    markdown = render_markdown_report(report_payload)
    (output_path / "dataset_forensics_report.md").write_text(markdown, encoding="utf-8")
    return manifests


if __name__ == "__main__":
    generate_dataset_reports()
