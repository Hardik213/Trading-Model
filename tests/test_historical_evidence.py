import pandas as pd

from src.historical_evidence import HistoricalEvidenceProvider
from src.phase9_runner import run_historical_strategy
from src.replay_engine import HistoricalReplay, ReplayConfig
from src.reference_detectors import reference_detector_bundle
from src.timeframe_context import build_context
from src.ict2022_engine import SetupState


def data(n=8):
    idx = pd.date_range("2026-01-01T10:00:00Z", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "Open": [100+i for i in range(n)],
            "High": [101+i for i in range(n)],
            "Low": [99+i for i in range(n)],
            "Close": [100+i for i in range(n)],
        },
        index=idx,
    )


def make_replay(df):
    context=build_context(df,base_timeframe="5M",source="TEST")
    return HistoricalReplay(
        df,
        context=context,
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )


def test_reference_provider_is_chronological():
    df=data()
    provider=HistoricalEvidenceProvider(reference_detector_bundle())
    replay=make_replay(df)

    seen=[]

    def wrapped(timestamp, visible_base, visible_context):
        assert visible_base.index.max() <= timestamp
        for frame in visible_context.frames.values():
            if len(frame):
                assert frame.index.max() <= timestamp
        seen.append(timestamp)
        return provider.build(timestamp,visible_base,visible_context)

    # Directly prove the provider can be called only with visible data.
    for ts in replay.decision_times():
        visible=replay.data.loc[replay.data.index<=ts]
        context=build_context(visible,base_timeframe="5M",source="TEST")
        wrapped(ts,visible,context)

    assert seen == list(df.index)


def test_phase9_runner_is_single_chronological_unit():
    df=data()
    replay=make_replay(df)
    provider=HistoricalEvidenceProvider(reference_detector_bundle())

    run=run_historical_strategy(replay,provider=provider)

    assert len(run.result.observations)==len(df)
    assert all(
        x.state in {
            SetupState.DEVELOPING.value,
            SetupState.NO_TRADE.value,
            SetupState.ACTIVE_TRADE.value,
        }
        for x in run.result.observations
    )
    assert len(run.result.valid)==0
    assert len(run.result.no_trade)==0
    assert len(run.result.developing)==len(df)


def test_provider_does_not_invent_direction_or_trade():
    df=data()
    provider=HistoricalEvidenceProvider(reference_detector_bundle())
    replay=make_replay(df)
    run=run_historical_strategy(replay,provider=provider)

    for observation in run.result.observations:
        assert observation.direction is None
        assert observation.state == SetupState.DEVELOPING.value
        assert observation.entry_price is None
        assert observation.invalidation_price is None
        assert observation.target_price is None
