import pandas as pd
import pytest

from src.data_contract import normalize_ohlc
from src.pipeline import ICT2022Pipeline
from src.replay_engine import (
    HistoricalReplay,
    ReplayConfig,
    ReplayDecision,
    assert_no_future_data,
)
from src.timeframe_context import build_context


def frame(start="2026-01-01T10:00:00Z", n=6):
    idx = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame(
        {
            "Open": range(100, 100+n),
            "High": [x+1 for x in range(100, 100+n)],
            "Low": [x-1 for x in range(100, 100+n)],
            "Close": range(100, 100+n),
        },
        index=idx,
    )


def test_normalizer_requires_timezone():
    raw=frame().tz_localize(None)
    with pytest.raises(ValueError):
        normalize_ohlc(raw,timeframe="5M")


def test_normalizer_produces_utc():
    raw=frame()
    out=normalize_ohlc(raw,timeframe="5M")
    assert str(out.index.tz) == "UTC"


def test_normalizer_rejects_bad_ohlc():
    raw=frame()
    raw.loc[raw.index[0],"High"]=50
    with pytest.raises(ValueError):
        normalize_ohlc(raw,timeframe="5M")


def test_context_has_explicit_timeframes():
    ctx=build_context(frame(),base_timeframe="5M",source="TEST")
    assert "5M" in ctx.frames
    assert "15M" in ctx.frames
    assert "1H" in ctx.frames


def test_visible_context_never_contains_future_bars():
    base=frame(n=10)
    ctx=build_context(base,base_timeframe="5M",source="TEST")
    replay=HistoricalReplay(
        base,
        context=ctx,
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )
    seen=[]
    def callback(ts, visible, visible_context):
        assert_no_future_data(visible,as_of=ts)
        for f in visible_context.frames.values():
            assert_no_future_data(f,as_of=ts)
        seen.append(ts)
        return __import__("src.replay_engine",fromlist=["ReplayObservation"]).ReplayObservation(
            ts,ReplayDecision.DEVELOPING,"DEVELOPING","test"
        )
    out=replay.run(callback)
    assert len(out)==10
    assert len(seen)==10


def test_callback_cannot_change_replay_timestamp():
    base=frame(n=2)
    ctx=build_context(base,base_timeframe="5M",source="TEST")
    replay=HistoricalReplay(
        base,
        context=ctx,
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )
    def bad_callback(ts, visible, context):
        return __import__("src.replay_engine",fromlist=["ReplayObservation"]).ReplayObservation(
            ts+pd.Timedelta(minutes=5),ReplayDecision.DEVELOPING,"DEVELOPING","bad"
        )
    with pytest.raises(ValueError):
        replay.run(bad_callback)


def test_pipeline_runs_as_one_chronological_unit():
    base=frame(n=5)
    pipeline=ICT2022Pipeline.from_base_data(base,timeframe="5M",source="TEST")
    observations=pipeline.run_observation_pass()
    assert len(observations)==5
    assert all(x.decision is ReplayDecision.DEVELOPING for x in observations)
    assert [x.timestamp for x in observations] == list(base.index)
