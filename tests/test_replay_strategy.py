import pandas as pd

from src.replay_engine import HistoricalReplay, ReplayConfig
from src.replay_strategy import ICT2022ReplayRunner
from src.strategy_adapter import EvidenceProvider, StrategyEvidence
from src.mss import Direction
from src.fvg import FVGDirection


class EmptyEvidenceProvider(EvidenceProvider):
    def build(self, timestamp, visible_base, visible_context):
        return StrategyEvidence(
            timestamp=timestamp,
            direction=None,
            dealing_range=None,
            draw_on_liquidity=None,
            liquidity_event=None,
            mss=None,
            pd_array=None,
            entry_price=None,
            invalidation_price=None,
            target_price=None,
            target_liquidity=None,
        )


def data():
    idx=pd.date_range("2026-01-01T10:00:00Z",periods=5,freq="5min")
    return pd.DataFrame(
        {"Open":[100]*5,"High":[101]*5,"Low":[99]*5,"Close":[100]*5},
        index=idx,
    )


def test_strategy_runner_processes_every_replay_timestamp():
    from src.timeframe_context import build_context
    base=data()
    ctx=build_context(base,base_timeframe="5M",source="TEST")
    replay=HistoricalReplay(
        base,
        context=ctx,
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )
    result=ICT2022ReplayRunner(
        replay,
        provider=EmptyEvidenceProvider(),
    ).run()

    assert len(result.observations)==5
    assert len(result.developing)==5
    assert len(result.valid)==0
    assert len(result.no_trade)==0


def test_replay_result_preserves_chronology():
    from src.timeframe_context import build_context
    base=data()
    ctx=build_context(base,base_timeframe="5M",source="TEST")
    replay=HistoricalReplay(
        base,
        context=ctx,
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )
    result=ICT2022ReplayRunner(
        replay,
        provider=EmptyEvidenceProvider(),
    ).run()

    assert [x.timestamp for x in result.observations] == list(base.index)
