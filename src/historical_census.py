"""Census orchestration over a frozen causal evidence builder.

This module records classification counts and optional observations. It does
not invent evidence, targets, fills, or outcomes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Callable, Optional

import pandas as pd

from .historical_sniper_replay import HistoricalPrecisionReplay, PrecisionReplayResult
from .replay_engine import ReplayConfig


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


def write_report(report: CensusReport, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
