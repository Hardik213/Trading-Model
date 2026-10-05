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
from .event_backtester import TradeDirection, TradePlan, simulate_trade
from .historical_sniper_replay import HistoricalPrecisionReplay, PrecisionReplayResult
from .mss import Direction, normalize_direction
from .replay_engine import ReplayConfig


def serialize_trade_plan(plan: TradePlan) -> dict[str, Any]:
    """Serialize the canonical immutable TradePlan into a deterministic JSON shape."""
    payload = {
        "trade_id": plan.trade_id,
        "entry_time": plan.entry_time.isoformat(),
        "direction": plan.direction.value,
        "entry_price": float(plan.entry_price),
        "stop_price": float(plan.stop_price),
        "target_price": float(plan.target_price),
        "planned_r": float(plan.planned_r),
    }
    if plan.replay_subject is not None:
        payload["replay_subject"] = plan.replay_subject.to_dict()
    return payload


CANONICAL_HISTORICAL_CENSUS_DIR = "data/reports/historical_census"

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


def _trade_direction_for(direction: Optional[Direction]) -> Optional[TradeDirection]:
    if direction is None:
        return None
    return TradeDirection.LONG if normalize_direction(direction) is Direction.BULLISH else TradeDirection.SHORT


def _replay_subject_for(obs):
    return getattr(obs, "replay_subject", None) or getattr(
        getattr(obs, "evidence", None),
        "replay_subject",
        None,
    )


def _decision_id_for(obs) -> str:
    subject = _replay_subject_for(obs)
    if subject is not None:
        return f"decision_{subject.subject_id}"
    ts = pd.Timestamp(obs.timestamp)
    return f"decision_{ts.strftime('%Y%m%d%H%M%S')}_{getattr(obs, 'state', 'UNKNOWN').lower()}"


def _build_trade_plan(obs, *, data: pd.DataFrame) -> Optional[TradePlan]:
    if obs.state != "VALID":
        return None
    evidence = getattr(obs, "evidence", None)
    if evidence is None:
        return None

    direction = evidence.direction
    if direction is None:
        return None
    trade_direction = _trade_direction_for(direction)
    entry = evidence.entry_price
    invalidation = evidence.invalidation_price
    target = evidence.target_price
    if entry is None or invalidation is None or target is None or trade_direction is None:
        return None

    subject = _replay_subject_for(obs)
    trade_id = (
        f"trade_{subject.subject_id}_{trade_direction.value.lower()}"
        if subject is not None
        else f"trade_{pd.Timestamp(obs.timestamp).strftime('%Y%m%d%H%M%S')}_{trade_direction.value.lower()}"
    )
    return TradePlan(
        trade_id=trade_id,
        entry_time=pd.Timestamp(obs.timestamp),
        direction=trade_direction,
        entry_price=float(entry),
        stop_price=float(invalidation),
        target_price=float(target),
        planned_r=float(obs.planned_r) if obs.planned_r is not None else abs(float(target) - float(entry)) / abs(float(entry) - float(invalidation)),
        replay_subject=subject,
    )


def _trade_outcome_record(plan: TradePlan, data: pd.DataFrame) -> dict[str, Any]:
    result = simulate_trade(data, plan)
    payload = {
        "trade_id": plan.trade_id,
        "entry_time": plan.entry_time.isoformat(),
        "direction": plan.direction.value,
        "outcome": result.outcome.value,
        "exit_time": result.exit_time.isoformat() if result.exit_time is not None else None,
        "entry_price": plan.entry_price,
        "exit_price": result.exit_price,
        "gross_r": result.gross_r,
        "net_r": result.net_r,
        "mfe_r": result.mfe_r,
        "mae_r": result.mae_r,
        "bars_held": result.bars_held,
        "reason": result.reason,
    }
    if result.replay_subject is not None:
        payload["replay_subject"] = result.replay_subject.to_dict()
    return payload


def _decision_record(obs, *, source: str, include_outcome: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    decision_time = {
        "timestamp_utc": pd.Timestamp(obs.timestamp).isoformat(),
        "classification": obs.state,
        "reason": obs.reason,
        "detail": obs.detail,
        "planned_r": obs.planned_r,
        "causal_cutoff_timestamp": pd.Timestamp(obs.timestamp).isoformat(),
    }
    payload = {
        "decision_id": _decision_id_for(obs),
        "timestamp_utc": decision_time["timestamp_utc"],
        "classification": decision_time["classification"],
        "reason": decision_time["reason"],
        "detail": decision_time["detail"],
        "planned_r": decision_time["planned_r"],
        "source_dataset": source,
        "causal_cutoff_timestamp": decision_time["causal_cutoff_timestamp"],
        "decision_time": decision_time,
    }
    if include_outcome is not None:
        payload["post_decision_outcome"] = include_outcome
    subject = _replay_subject_for(obs)
    if subject is not None:
        payload["replay_subject"] = subject.to_dict()
    return payload


def _observation_payload(obs) -> dict[str, Any]:
    payload = {
        "timestamp": obs.timestamp.isoformat(),
        "state": obs.state,
        "reason": obs.reason,
        "detail": obs.detail,
        "planned_r": obs.planned_r,
    }
    subject = _replay_subject_for(obs)
    if subject is not None:
        payload["decision_id"] = _decision_id_for(obs)
        payload["replay_subject"] = subject.to_dict()
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (pd.Timestamp, pd.Timedelta)):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    return value


def _json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    trade_plan_objects = safe.get("trade_plan_objects", [])
    if trade_plan_objects:
        safe["trade_plan_objects"] = [serialize_trade_plan(plan) for plan in trade_plan_objects]
    for key in ("trade_plans", "trade_outcomes", "observations", "decision_records", "setup_records", "no_trade_records"):
        if key in safe:
            safe[key] = _json_safe(safe[key])
    return safe


def build_historical_census(
    data: pd.DataFrame | str | Path,
    *,
    evidence_builder: Optional[Callable] = None,
    output_dir: str | Path = CANONICAL_HISTORICAL_CENSUS_DIR,
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
        trade_plans = []
        trade_outcomes = []
        trade_plan_objects: list[TradePlan] = []
        decision_records: list[dict[str, Any]] = []
        setup_records: list[dict[str, Any]] = []
        no_trade_records: list[dict[str, Any]] = []

        for obs in result.observations:
            decision_records.append(
                _decision_record(obs, source=source)
            )
            if obs.state == "NO_TRADE":
                no_trade_records.append(_decision_record(obs, source=source))
            elif obs.state == "VALID":
                plan = _build_trade_plan(obs, data=replay_frame)
                if plan is None:
                    continue
                trade_plan_objects.append(plan)
                trade_plans.append(serialize_trade_plan(plan))
                outcome = _trade_outcome_record(plan, replay_frame)
                trade_outcomes.append(outcome)
                decision_snapshot = _decision_record(obs, source=source)
                setup_record = {
                    "decision_id": decision_snapshot["decision_id"],
                    "decision_time": decision_snapshot["decision_time"],
                    "trade_plan": serialize_trade_plan(plan),
                    "post_decision_outcome": outcome,
                }
                if plan.replay_subject is not None:
                    setup_record["replay_subject"] = plan.replay_subject.to_dict()
                setup_records.append(setup_record)

        payload: dict[str, Any] = {
            "classification": classification,
            "coverage": coverage,
            "report": report.to_dict(),
            "observations": [_observation_payload(obs) for obs in result.observations],
            "trade_plan_objects": trade_plan_objects,
            "trade_plans": trade_plans,
            "trade_outcomes": trade_outcomes,
            "valid_trade_count": len(trade_outcomes),
            "warning": (
                "Smoke census: historical coverage is insufficient for a full historical census. "
                "This result is only a causal observation pass over the available sample."
                if classification == "SMOKE_CENSUS"
                else None
            ),
            "decision_records": decision_records,
            "setup_records": setup_records,
            "no_trade_records": no_trade_records,
        }
    except Exception as exc:  # pragma: no cover - fail-closed safety path
        payload = {
            "classification": "SMOKE_CENSUS",
            "coverage": coverage,
            "report": None,
            "observations": [],
            "trade_plan_objects": [],
            "trade_plans": [],
            "trade_outcomes": [],
            "valid_trade_count": 0,
            "warning": (
                "Smoke census: the frozen evidence builder could not evaluate the available sample safely, "
                f"so the pipeline stopped without claiming historical completeness ({type(exc).__name__}: {exc})."
            ),
            "error": f"{type(exc).__name__}: {exc}",
            "decision_records": [],
            "setup_records": [],
            "no_trade_records": [],
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "instrument": "XAUUSD",
        "source": source,
        "timeframe": timeframe,
        "classification": payload["classification"],
        "counts": {
            "observations": len(payload.get("observations", [])),
            "valid_trade_count": payload.get("valid_trade_count", 0),
            "no_trade_records": len(payload.get("no_trade_records", [])),
        },
        "artifacts": [
            "census_manifest.json",
            "census_coverage.md",
            "census_decisions.jsonl",
            "census_setups.jsonl",
            "census_no_trade.jsonl",
            "census_summary.json",
            "census_summary.md",
        ],
        "warning": payload.get("warning"),
        "error": payload.get("error"),
    }
    coverage_md = (
        "# Historical Census Coverage\n\n"
        f"- Classification: {payload['classification']}\n"
        f"- Source: {source}\n"
        f"- Timeframe: {timeframe}\n"
        f"- Rows: {coverage.get('rows', 0)}\n"
        f"- Start: {coverage.get('start')}\n"
        f"- End: {coverage.get('end')}\n"
        f"- Duration minutes: {coverage.get('duration_minutes', 0)}\n"
    )
    summary_md = (
        "# Historical Census Summary\n\n"
        f"- Classification: {payload['classification']}\n"
        f"- Valid setups: {payload.get('valid_trade_count', 0)}\n"
        f"- No-trade records: {len(payload.get('no_trade_records', []))}\n"
        f"- Observation count: {len(payload.get('observations', []))}\n"
        f"- Warning: {payload.get('warning') or 'None'}\n"
    )

    safe_payload = _json_safe_payload(payload)
    (out_dir / "summary.json").write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
    (out_dir / "decision_records.json").write_text(
        json.dumps(payload["observations"], indent=2),
        encoding="utf-8",
    )
    (out_dir / "trade_plans.json").write_text(
        json.dumps(payload["trade_plans"], indent=2),
        encoding="utf-8",
    )
    (out_dir / "trade_outcomes.json").write_text(
        json.dumps(_json_safe(payload["trade_outcomes"]), indent=2),
        encoding="utf-8",
    )
    (out_dir / "decision_records.jsonl").write_text(
        "\n".join(json.dumps(_json_safe(item)) for item in payload.get("observations", [])) + ("\n" if payload.get("observations") else ""),
        encoding="utf-8",
    )
    (out_dir / "census_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "census_coverage.md").write_text(coverage_md, encoding="utf-8")
    (out_dir / "census_decisions.jsonl").write_text(
        "\n".join(json.dumps(_json_safe(item)) for item in payload.get("decision_records", [])) + ("\n" if payload.get("decision_records") else ""),
        encoding="utf-8",
    )
    (out_dir / "census_setups.jsonl").write_text(
        "\n".join(json.dumps(_json_safe(item)) for item in payload.get("setup_records", [])) + ("\n" if payload.get("setup_records") else ""),
        encoding="utf-8",
    )
    (out_dir / "census_no_trade.jsonl").write_text(
        "\n".join(json.dumps(_json_safe(item)) for item in payload.get("no_trade_records", [])) + ("\n" if payload.get("no_trade_records") else ""),
        encoding="utf-8",
    )
    (out_dir / "census_summary.json").write_text(
        json.dumps({
            "classification": payload["classification"],
            "coverage": coverage,
            "valid_trade_count": payload.get("valid_trade_count", 0),
            "no_trade_count": len(payload.get("no_trade_records", [])),
            "warnings": [payload.get("warning")] if payload.get("warning") else [],
            "errors": [payload.get("error")] if payload.get("error") else [],
        }, indent=2),
        encoding="utf-8",
    )
    (out_dir / "census_summary.md").write_text(summary_md, encoding="utf-8")
    return payload


def write_report(report: CensusReport, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
