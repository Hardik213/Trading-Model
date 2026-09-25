import pandas as pd

from src.historical_sniper_replay import HistoricalPrecisionReplay
from src.replay_engine import ReplayConfig
from src.sniper_setup import PrecisionEvidence


def bars(n=12):
    idx = pd.date_range("2026-01-01 00:00", periods=n, freq="5min", tz="UTC")
    close = [100 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [x + 0.2 for x in close],
            "Low": [x - 0.2 for x in close],
            "Close": close,
        },
        index=idx,
    )


def builder(ts, visible, context):
    return PrecisionEvidence(
        as_of=ts,
        direction=None,
        draw_on_liquidity=None,
        liquidity_event=None,
        mss=None,
        pd_array=None,
        entry_price=None,
        invalidation_price=None,
        target_price=None,
        target_liquidity=None,
    )


def test_replay_never_changes_evidence_timestamp():
    result = HistoricalPrecisionReplay(
        bars(),
        evidence_builder=builder,
        config=ReplayConfig(timeframe="5M", source="TEST"),
    ).run()
    assert len(result.observations) == 12
    assert all(x.state == "DEVELOPING" for x in result.observations)
    assert result.state_counts() == {
        "VALID": 0,
        "DEVELOPING": 12,
        "INVALID": 0,
        "NO_TRADE": 0,
    }


def test_replay_respects_start_end_window():
    data = bars()
    result = HistoricalPrecisionReplay(
        data,
        evidence_builder=builder,
        config=ReplayConfig(
            timeframe="5M",
            source="TEST",
            start=data.index[3],
            end=data.index[7],
        ),
    ).run()
    assert [x.timestamp for x in result.observations] == list(data.index[3:8])
