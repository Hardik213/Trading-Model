import pandas as pd

from src.liquidity_replay import HistoricalLiquidityReplay
from src.replay_engine import ReplayConfig


def bars():
    idx = pd.date_range("2026-01-01 00:00", periods=14, freq="5min", tz="UTC")
    close = [100, 99, 101, 100, 102, 101, 103, 104, 102, 101, 100, 99, 100, 101]
    high = [x + 0.4 for x in close]
    low = [x - 0.4 for x in close]
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close}, index=idx
    )


def test_liquidity_replay_is_chronological_and_causal():
    result = HistoricalLiquidityReplay(
        bars(), config=ReplayConfig(timeframe="5M", source="TEST")
    ).run()
    assert len(result.observations) == len(bars())
    assert [x.timestamp for x in result.observations] == list(bars().index)
    for observation in result.observations:
        if observation.latest_rejection_timestamp is not None:
            assert observation.latest_rejection_timestamp < observation.timestamp
