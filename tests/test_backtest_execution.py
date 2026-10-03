import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.event_backtester import TradeDirection, TradeOutcome, TradePlan, simulate_trade
from src.features import FEATURE_COLUMNS


class SignalOnMarker:
    def predict_proba(self, features):
        probability = 0.99 if features["ret_5"].iloc[0] == 1.0 else 0.5
        return np.array([[1.0 - probability, probability]])


class ShortSignalOnMarker:
    def predict_proba(self, features):
        probability = 0.01 if features["ret_5"].iloc[0] == 1.0 else 0.5
        return np.array([[1.0 - probability, probability]])


def _frame(decision_time, *, periods=8, ret_1=0.05):
    index = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="1h")
    frame = pd.DataFrame(1.0, index=index, columns=FEATURE_COLUMNS)
    frame["ret_1"] = 0.0
    frame.loc[index[1], "ret_1"] = ret_1
    frame["ret_5"] = 0.0
    frame.loc[index[1], "ret_5"] = 1.0
    frame["sma_fast"] = 2.0
    frame["sma_slow"] = 1.0
    frame["session_bias"] = 1.0
    frame["regime_bias"] = 1.0
    frame["macro_bias"] = 0.0
    frame["liquidity_sweep"] = 0.0
    frame["mtf_decision_time"] = index
    frame["regime_score_available_at"] = pd.Series(index, index=index)
    frame["regime_bias_available_at"] = pd.Series(index, index=index)
    frame.loc[index[1], "mtf_decision_time"] = decision_time
    frame["Open"] = 100.0
    frame["High"] = 101.0
    frame["Low"] = 99.0
    frame["Close"] = 100.0
    frame.attrs["bar_label"] = "left"
    for position, open_price in ((3, 200.0), (4, 220.0), (5, 240.0)):
        if position < periods:
            frame.loc[index[position], "Open"] = open_price
    return frame


def test_close_available_signal_uses_only_later_open_observations():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h")
    frame = _frame(index[2])

    result = run_backtest(frame, SignalOnMarker())

    expected_weight = min(0.20, 0.01 / 0.05)
    assert result["trades"] == 1
    assert np.isclose(result["final_capital"], 10000.0 * (1.0 + expected_weight * 0.10))


def test_after_close_signal_waits_for_a_strictly_later_bar():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=9, freq="1h")
    frame = _frame(index[3] + pd.Timedelta(minutes=30), periods=9)
    frame.loc[index[4], "Open"] = 200.0
    frame.loc[index[5], "Open"] = 240.0

    result = run_backtest(frame, SignalOnMarker())

    expected_weight = min(0.20, 0.01 / 0.05)
    assert result["trades"] == 1
    assert np.isclose(result["final_capital"], 10000.0 * (1.0 + expected_weight * 0.20))


def test_delayed_liquidity_confirmation_delays_execution():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=9, freq="1h")
    confirmation = index[3] + pd.Timedelta(minutes=30)
    frame = _frame(confirmation, periods=9)
    frame.loc[index[1], "liquidity_sweep"] = 1.0
    frame.loc[index[1], "regime_bias"] = 1.0
    frame["liquidity_confirmation_available_at"] = pd.Series(
        pd.NaT, index=index, dtype="datetime64[ns, UTC]"
    )
    frame.loc[index[1], "liquidity_confirmation_available_at"] = confirmation
    frame.loc[index[4], "Open"] = 200.0
    frame.loc[index[5], "Open"] = 240.0

    result = run_backtest(frame, SignalOnMarker())

    expected_weight = min(0.20, 0.01 / 0.05)
    assert result["trades"] == 1
    assert np.isclose(result["final_capital"], 10000.0 * (1.0 + expected_weight * 0.20))


def test_elapsed_nominal_return_is_not_attributed_to_later_entry():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=10, freq="1h")
    decision_time = index[5] + pd.Timedelta(minutes=30)
    frame = _frame(decision_time, periods=10, ret_1=0.5)
    frame.loc[index[6], "Open"] = 200.0
    frame.loc[index[7], "Open"] = 220.0

    result = run_backtest(frame, SignalOnMarker())

    expected_weight = min(0.20, 0.01 / 0.5)
    assert result["trades"] == 1
    assert np.isclose(result["final_capital"], 10000.0 * (1.0 + expected_weight * 0.10))


def test_signal_without_post_decision_entry_and_exit_is_unsupported():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1h")
    frame = _frame(index[-1] + pd.Timedelta(minutes=1), periods=5)

    result = run_backtest(frame, SignalOnMarker())

    assert result["trades"] == 0
    assert result["unsupported_signals"] == 1
    assert result["final_capital"] == result["initial_capital"]


def test_stop_loss_only_considers_bars_after_the_causal_entry():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1h")
    frame = _frame(index[2], periods=6)
    frame.loc[index[2], ["High", "Low"]] = [110.0, 90.0]
    frame.loc[index[3], ["Open", "High", "Low", "Close"]] = [100.0, 101.0, 99.0, 100.0]
    frame.loc[index[4], ["Open", "High", "Low", "Close"]] = [100.0, 101.0, 97.0, 98.0]
    entry_position = frame.index.searchsorted(index[2], side="right")
    entry_time = frame.index[entry_position]
    plan = TradePlan("stop", entry_time, TradeDirection.LONG, 100.0, 98.0, 102.0, 1.0)

    result = simulate_trade(frame[["Open", "High", "Low", "Close"]], plan)

    assert entry_time > index[2]
    assert result.outcome is TradeOutcome.STOP
    assert result.exit_time == index[4]


def test_take_profit_only_considers_bars_after_the_causal_entry():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1h")
    frame = _frame(index[2], periods=6)
    frame.loc[index[2], ["High", "Low"]] = [110.0, 90.0]
    frame.loc[index[3], ["Open", "High", "Low", "Close"]] = [100.0, 101.0, 99.0, 100.0]
    frame.loc[index[4], ["Open", "High", "Low", "Close"]] = [100.0, 103.0, 99.0, 102.0]
    entry_position = frame.index.searchsorted(index[2], side="right")
    entry_time = frame.index[entry_position]
    plan = TradePlan("target", entry_time, TradeDirection.LONG, 100.0, 98.0, 102.0, 1.0)

    result = simulate_trade(frame[["Open", "High", "Low", "Close"]], plan)

    assert entry_time > index[2]
    assert result.outcome is TradeOutcome.TARGET
    assert result.exit_time == index[4]


def test_backtest_rejects_mtf_decision_before_source_availability():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h")
    frame = _frame(index[2])
    frame["available_at"] = index + pd.Timedelta(hours=1)
    frame.loc[index[1], "available_at"] = index[3]
    model = SignalOnMarker()

    result = run_backtest(frame, model)

    assert result["trades"] == 0
    assert result["wins"] == 0
    assert result["final_capital"] == result["initial_capital"]


def test_equity_is_recognized_at_exit_not_signal_or_entry():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h")
    frame = _frame(index[2])

    result = run_backtest(frame, SignalOnMarker())
    equity = result["equity_curve"]

    assert equity[0] == result["initial_capital"]
    assert equity[2] == result["initial_capital"]
    assert equity[4] == result["initial_capital"]
    assert equity[5] == result["final_capital"]
    assert result["final_capital"] == 10200.0


def test_sequential_trade_sizing_uses_only_realized_equity():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h")
    frame = _frame(index[2])
    frame.loc[index[3], "ret_5"] = 1.0
    frame.loc[index[3], "ret_1"] = 0.05
    frame.loc[index[3], "mtf_decision_time"] = index[4]
    frame.loc[index[6], "Open"] = 260.0

    result = run_backtest(frame, SignalOnMarker())
    equity = result["equity_curve"]

    assert equity[4] == result["initial_capital"]
    assert equity[5] == 10200.0
    assert equity[7] == 10370.0
    assert result["trades"] == 2
    assert result["wins"] == 2


def test_unsupported_exit_does_not_change_equity_or_trade_metrics():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h")
    frame = _frame(index[2])
    frame.loc[index[4], "Open"] = np.nan

    result = run_backtest(frame, SignalOnMarker())

    assert result["trades"] == 0
    assert result["wins"] == 0
    assert result["unsupported_signals"] == 1
    assert result["final_capital"] == result["initial_capital"]


def test_short_return_is_realized_with_directional_sign_at_exit():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h")
    frame = _frame(index[2])
    frame.loc[index[1], "sma_fast"] = 0.0
    frame.loc[index[1], "sma_slow"] = 1.0
    frame.loc[index[1], "session_bias"] = -1.0
    frame.loc[index[1], "macro_bias"] = -1.0
    frame.loc[index[4], "Open"] = 180.0

    result = run_backtest(frame, ShortSignalOnMarker())

    assert result["equity_curve"][4] == result["initial_capital"]
    assert result["equity_curve"][5] == 10200.0
    assert result["final_capital"] == 10200.0