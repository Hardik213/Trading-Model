from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import pandas as pd

from .data_contract import normalize_ohlc
from .timeframe_context import TimeframeContext


class ReplayDecision(str, Enum):
    NO_TRADE = "NO_TRADE"
    DEVELOPING = "DEVELOPING"
    VALID = "VALID"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class ReplayObservation:
    timestamp: pd.Timestamp
    decision: ReplayDecision
    state: str
    reason: str


@dataclass(frozen=True)
class ReplayConfig:
    timeframe: str = "5M"
    source: str = "UNKNOWN"
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None


DecisionCallback = Callable[
    [pd.Timestamp, pd.DataFrame, TimeframeContext],
    ReplayObservation,
]


class HistoricalReplay:
    """
    Single chronological driver for the strategy stack.

    At decision time t it provides only:
      - base bars <= t
      - higher-timeframe bars <= t
      - no future rows

    The callback is where the already-built ICT-2022 state machine is connected.
    The replay engine itself does not invent a setup.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        context: TimeframeContext,
        config: ReplayConfig,
    ) -> None:
        self.data = normalize_ohlc(
            data,
            timeframe=config.timeframe,
            source=config.source,
        )
        self.context = context
        self.config = config

    def decision_times(self) -> pd.DatetimeIndex:
        idx = self.data.index

        if self.config.start is not None:
            idx = idx[idx >= self.config.start]
        if self.config.end is not None:
            idx = idx[idx <= self.config.end]

        return idx

    def run(self, callback: DecisionCallback) -> list[ReplayObservation]:
        observations: list[ReplayObservation] = []

        for timestamp in self.decision_times():
            # This is the no-lookahead boundary.
            visible_base = self.data.loc[self.data.index <= timestamp].copy()

            visible_frames = {
                name: frame.loc[frame.index <= timestamp].copy()
                for name, frame in self.context.frames.items()
            }
            visible_context = TimeframeContext(frames=visible_frames)

            observation = callback(
                timestamp,
                visible_base,
                visible_context,
            )

            if observation.timestamp != timestamp:
                raise ValueError(
                    "Replay callback returned an observation with the wrong timestamp."
                )

            observations.append(observation)

        return observations


def assert_no_future_data(
    visible: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> None:
    future = visible.index[visible.index > as_of]
    if len(future):
        raise AssertionError(
            f"Replay leaked {len(future)} future rows at {as_of}."
        )
