from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .data_contract import normalize_ohlc


@dataclass(frozen=True)
class TimeframeContext:
    """
    Timeframe views used by the replay engine.

    Each dataframe is independently normalized to UTC. Resampling is only
    performed from a lower-timeframe source and is anchored to UTC. The replay
    engine only exposes completed higher-timeframe bars as of a decision time.
    """

    frames: Mapping[str, pd.DataFrame]

    def available_as_of(self, timeframe: str, as_of: pd.Timestamp) -> pd.DataFrame:
        if timeframe not in self.frames:
            raise KeyError(f"Unknown timeframe: {timeframe}")

        frame = self.frames[timeframe]
        # A completed bar is available only after its timestamped interval has
        # closed. For ordinary timestamped OHLC datasets, the conservative
        # convention is to require the bar timestamp itself to be <= as_of.
        return frame.loc[frame.index <= as_of].copy()


def build_context(
    base: pd.DataFrame,
    *,
    base_timeframe: str = "5M",
    source: str = "UNKNOWN",
) -> TimeframeContext:
    """
    Build a deterministic context set from one normalized base frame.

    This intentionally keeps resampling conservative and explicit. The caller
    can replace these generated frames with feed-native 1H/15M/3M/1M data when
    available. No synthetic tick/order information is created.
    """
    base = normalize_ohlc(
        base,
        timeframe=base_timeframe,
        source=source,
    )

    rules = {
        "15M": "15min",
        "1H": "1h",
        "4H": "4h",
        "1D": "1D",
    }

    frames: dict[str, pd.DataFrame] = {base_timeframe: base}

    for label, rule in rules.items():
        if label == base_timeframe:
            continue

        resampled = (
            base[["Open", "High", "Low", "Close"]]
            .resample(rule, label="right", closed="right")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                }
            )
            .dropna()
        )
        resampled.attrs["timeframe"] = label
        resampled.attrs["source"] = source
        resampled.attrs["timezone"] = "UTC"
        frames[label] = resampled

    return TimeframeContext(frames=frames)
