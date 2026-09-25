import pandas as pd
import pytest

from src.event_backtester import TradeDirection, TradeOutcome, TradePlan, simulate_trade
from src.research_metrics import summarize_results


def frame(rows):
    # The declared TradePlan entry is 10:00 UTC, so the fixture must begin there.
    idx = pd.date_range("2026-01-01T10:00:00Z", periods=len(rows), freq="5min")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def plan(direction=TradeDirection.LONG):
    return TradePlan(
        trade_id="T1",
        entry_time=pd.Timestamp("2026-01-01T10:00:00Z"),
        direction=direction,
        entry_price=100,
        stop_price=98 if direction is TradeDirection.LONG else 102,
        target_price=104 if direction is TradeDirection.LONG else 96,
        planned_r=2.0,
    )


def test_long_target_before_stop():
    result=simulate_trade(frame([(100,101,99,100),(100,103,99,102),(102,105,101,104)]), plan())
    assert result.outcome is TradeOutcome.TARGET
    assert result.gross_r == 2.0


def test_long_stop_before_target():
    result=simulate_trade(frame([(100,101,99,100),(100,101,97,98),(98,105,97,104)]), plan())
    assert result.outcome is TradeOutcome.STOP
    assert result.gross_r == -1.0


def test_same_bar_stop_and_target_is_ambiguous():
    result=simulate_trade(frame([(100,105,97,101)]), plan())
    assert result.outcome is TradeOutcome.AMBIGUOUS
    assert result.exit_price is None


def test_short_target_before_stop():
    result=simulate_trade(frame([(100,101,99,100),(100,101,97,98),(98,97,95,96)]), plan(TradeDirection.SHORT))
    assert result.outcome is TradeOutcome.TARGET
    assert result.gross_r == 2.0


def test_short_stop_before_target():
    result=simulate_trade(frame([(100,103,99,101),(101,104,100,103)]), plan(TradeDirection.SHORT))
    assert result.outcome is TradeOutcome.STOP
    assert result.gross_r == -1.0


def test_expiry_is_not_a_win_or_loss():
    result=simulate_trade(frame([(100,101,99,100),(100,101,99,100)]), plan())
    assert result.outcome is TradeOutcome.EXPIRED
    assert result.net_r is None


def test_mfe_mae_are_recorded():
    result=simulate_trade(frame([(100,103,99,102),(102,105,101,104)]), plan())
    assert result.mfe_r >= 2.0
    assert result.mae_r <= -0.5


def test_invalid_plan_geometry_rejected():
    with pytest.raises(ValueError):
        TradePlan("bad", pd.Timestamp("2026-01-01T10:00:00Z"), TradeDirection.LONG, 100, 101, 104, 2)


def test_summary_uses_resolved_trades():
    df=frame([(100,101,99,100),(100,105,99,104)])
    results=[
        simulate_trade(df,plan()),
        simulate_trade(df,TradePlan("T2",pd.Timestamp("2026-01-01T10:00:00Z"),TradeDirection.LONG,100,98,110,5)),
    ]
    s=summarize_results(results)
    assert s.n_resolved == 1
    assert s.n_target == 1
    assert s.expectancy_r == 2.0
    assert s.win_rate == 1.0
