"""Census orchestration over a frozen causal evidence builder.

This module records classification counts and optional observations. It does
not invent evidence, targets, fills, or outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Callable, Optional

import pandas as pd

from .automatic_evidence import build_automatic_evidence
from .historical_sniper_replay import HistoricalPrecisionReplay, PrecisionReplayResult
from .replay_engine import ReplayConfig


HISTORICAL_CENSUS_MIN_ROWS = 1000
HISTORICAL_CENSUS_MIN_DURATION_MINUTES = 7 * 24 * 60


@dataclass(frozen=True)
class CensusReport:
    instrument: str
    source: str
    timeframe: str
    start: str
    end: str
    observations: int
    state_counts: dict[str, int]
    valid_setups: int
    no_trade: int
    developing: int
    invalid: int

    @classmethod
    def from_result(
        cls,
        result: PrecisionReplayResult,
        *,
        instrument: str,
        source: str,
        timeframe: str,
    ) -> "CensusReport":
        counts = result.state_counts()
        ts = [x.timestamp for x in result.observations]
        return cls(
            instrument=instrument,
            source=source,
            timeframe=timeframe,
            start=min(ts).isoformat() if ts else "",
            end=max(ts).isoformat() if ts else "",
            observations=len(result.observations),
            state_counts=counts,
            valid_setups=counts.get("VALID", 0),
            no_trade=counts.get("NO_TRADE", 0),
            developing=counts.get("DEVELOPING", 0),
            invalid=counts.get("INVALID", 0),
        )

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "source": self.source,
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "observations": self.observations,
            "state_counts": self.state_counts,
            "valid_setups": self.valid_setups,
            "no_trade": self.no_trade,
            "developing": self.developing,
            "invalid": self.invalid,
        }


def _coerce_frame(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, (str, Path)):
        path = Path(data)
        frame = None
        for sep in (None, ";", ","):
            try:
                if sep is None:
                    read_frame = pd.read_csv(path)
                else:
                    read_frame = pd.read_csv(path, sep=sep)
            except Exception:
                continue
            if not read_frame.empty:
                sample = read_frame.iloc[0].to_dict()
                if len(read_frame.columns) == 1 and any(";" in str(v) for v in sample.values()):
                    read_frame = pd.read_csv(path, sep=";")
                frame = read_frame
                break
        if frame is None:
            raise ValueError(f"Unable to read CSV data from: {path}")
    else:
        raise TypeError("data must be a DataFrame or a CSV path")

    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    timestamp_candidates = [
        "timestamp",
        "datetime",
        "date",
        "Date",
        "Etc/UTC",
        "time",
        "Time",
    ]
    ts_col = None
    for candidate in timestamp_candidates:
        if candidate in frame.columns:
            ts_col = candidate
            break
    if ts_col is None:
        for column in frame.columns:
            lower = column.lower()
            if "date" in lower or "time" in lower or "utc" in lower:
                ts_col = column
                break

    if ts_col is not None:
        frame = frame.rename(columns={ts_col: "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        frame["timestamp"] = pd.DatetimeIndex(frame["timestamp"]).tz_localize("UTC") if pd.DatetimeIndex(frame["timestamp"]).tz is None else pd.DatetimeIndex(frame["timestamp"]).tz_convert("UTC")
        frame = frame.set_index(pd.DatetimeIndex(frame["timestamp"]))
        frame = frame.drop(columns=["timestamp"])

    rename_map = {
        "Open": "Open",
        "open": "Open",
        "High": "High",
        "high": "High",
        "Low": "Low",
        "low": "Low",
        "Close": "Close",
        "close": "Close",
        "Volume": "Volume",
        "volume": "Volume",
    }
    frame = frame.rename(columns=rename_map)
    required = {"Open", "High", "Low", "Close"}
    present = set(frame.columns)
    missing = required.difference(present)
    if not missing:
        return frame
    if any(column.lower() in {"date", "time", "datetime"} for column in present):
        return frame
    return frame


def summarize_data_coverage(
    data: pd.DataFrame | str | Path,
    *,
    timeframe: str = "5min",
    source: str = "dukascopy",
) -> dict[str, Any]:
    frame = _coerce_frame(data)
    if frame.empty:
        return {
            "source": source,
            "timeframe": timeframe,
            "rows": 0,
            "start": None,
            "end": None,
            "duration_minutes": 0,
            "duration_hours": 0.0,
            "coverage_status": "INSUFFICIENT",
        }

    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    frame = frame.copy()
    frame.index = index
    frame = frame.sort_index()

    start = index.min()
    end = index.max()
    duration = end - start
    duration_minutes = int(duration.total_seconds() // 60)
    duplicate_timestamps = int(index.duplicated().sum())
    return {
        "source": source,
        "timeframe": timeframe,
        "rows": int(len(frame)),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "duration_minutes": duration_minutes,
        "duration_hours": round(duration_minutes / 60.0, 3),
        "duplicate_timestamps": duplicate_timestamps,
        "coverage_status": "SUFFICIENT" if duration_minutes >= HISTORICAL_CENSUS_MIN_DURATION_MINUTES and len(frame) >= HISTORICAL_CENSUS_MIN_ROWS else "INSUFFICIENT",
    }


def classify_coverage(coverage: dict[str, Any]) -> str:
    rows = int(coverage.get("rows", 0))
    duration_minutes = int(coverage.get("duration_minutes", 0))
    if rows < HISTORICAL_CENSUS_MIN_ROWS or duration_minutes < HISTORICAL_CENSUS_MIN_DURATION_MINUTES:
        return "SMOKE_CENSUS"
    return "HISTORICAL_CENSUS"


def run_census(
    data: pd.DataFrame,
    *,
    evidence_builder: Callable,
    timeframe: str = "5min",
    source: str = "dukascopy",
    min_planned_r: Optional[float] = None,
) -> tuple[PrecisionReplayResult, CensusReport]:
    replay = HistoricalPrecisionReplay(
        data,
        evidence_builder=evidence_builder,
        config=ReplayConfig(timeframe=timeframe, source=source),
        min_planned_r=min_planned_r,
    )
    result = replay.run()
    report = CensusReport.from_result(
        result,
        instrument="XAUUSD",
        source=source,
        timeframe=timeframe,
    )
    return result, report


def build_historical_census(
    data: pd.DataFrame | str | Path,
    *,
    evidence_builder: Optional[Callable] = None,
    output_dir: str | Path = "data/reports/historical_census",
    timeframe: str = "5min",
    source: str = "dukascopy",
    min_planned_r: Optional[float] = None,
) -> dict[str, Any]:
    frame = _coerce_frame(data)
    coverage = summarize_data_coverage(frame, timeframe=timeframe, source=source)
    classification = classify_coverage(coverage)

    replay_frame = frame.copy()
    if replay_frame.index.has_duplicates:
        replay_frame = replay_frame[~replay_frame.index.duplicated(keep="first")]

    builder = evidence_builder or (lambda ts, base, context: build_automatic_evidence(ts, base, context).evidence)
    try:
        result, report = run_census(
            replay_frame,
            evidence_builder=builder,
            timeframe=timeframe,
            source=source,
            min_planned_r=min_planned_r,
        )
        payload: dict[str, Any] = {
            "classification": classification,
            "coverage": coverage,
            "report": report.to_dict(),
            "observations": [
                {
                    "timestamp": obs.timestamp.isoformat(),
                    "state": obs.state,
                    "reason": obs.reason,
                    "detail": obs.detail,
                    "planned_r": obs.planned_r,
                }
                for obs in result.observations
            ],
            "warning": (
                "Smoke census: historical coverage is insufficient for a full historical census. "
                "This result is only a causal observation pass over the available sample."
                if classification == "SMOKE_CENSUS"
                else None
            ),
        }
    except Exception as exc:  # pragma: no cover - fail-closed safety path
        payload = {
            "classification": "SMOKE_CENSUS",
            "coverage": coverage,
            "report": None,
            "observations": [],
            "warning": (
                "Smoke census: the frozen evidence builder could not evaluate the available sample safely, "
                f"so the pipeline stopped without claiming historical completeness ({type(exc).__name__}: {exc})."
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "decision_records.json").write_text(
        json.dumps(payload["observations"], indent=2),
        encoding="utf-8",
    )
    return payload


def write_report(report: CensusReport, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
