import pandas as pd

import src.liquidity_replay as liquidity_replay
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


def _capture_detector_inputs(monkeypatch):
    calls=[]

    def capture(frame, as_of, *args, **kwargs):
        available_at=(
            tuple(frame["available_at"])
            if "available_at" in frame.columns
            else None
        )
        calls.append((pd.Timestamp(as_of),tuple(frame.index),available_at))
        return []

    def capture_draw(frame, as_of, *args, **kwargs):
        capture(frame,as_of,*args,**kwargs)
        return None

    monkeypatch.setattr(liquidity_replay,"confirmed_liquidity_map",capture)
    monkeypatch.setattr(liquidity_replay,"liquidity_evidence_as_of",capture)
    monkeypatch.setattr(liquidity_replay,"draw_on_liquidity",capture_draw)
    return calls


def test_liquidity_replay_is_chronological_and_causal():
    result = HistoricalLiquidityReplay(
        bars(), config=ReplayConfig(timeframe="5M", source="TEST")
    ).run()
    assert len(result.observations) == len(bars())
    assert [x.timestamp for x in result.observations] == list(
        bars().index + pd.Timedelta(minutes=5)
    )
    assert pd.DatetimeIndex([x.timestamp for x in result.observations]).is_monotonic_increasing
    for observation in result.observations:
        if observation.latest_rejection_timestamp is not None:
            assert observation.latest_rejection_timestamp < observation.timestamp


def test_event_time_replay_filters_detector_inputs_and_groups_shared_availability(monkeypatch):
    data=bars().iloc[:3].copy()
    data["interval_end"]=data.index+pd.Timedelta(minutes=5)
    data["available_at"]=pd.to_datetime(
        ["2026-01-01T00:12:30Z","2026-01-01T00:12:30Z",None],utc=True
    )
    data["availability_ts"]=data["available_at"]
    data["historical_complete"]=[True,True,False]
    data["is_complete"]=[True,True,False]
    calls=_capture_detector_inputs(monkeypatch)

    result=HistoricalLiquidityReplay(
        data,config=ReplayConfig(timeframe="5M",source="TEST")
    ).run()

    available_at=pd.Timestamp("2026-01-01T00:12:30Z")
    assert [item.timestamp for item in result.observations] == [available_at]
    assert calls
    for timestamp, visible_labels, visible_availability in calls:
        assert timestamp == available_at
        assert visible_labels == tuple(data.index[:2])
        assert visible_availability == (available_at,available_at)
        assert all(value <= timestamp for value in visible_availability)
    assert data.index[2] not in tuple(label for _, labels, _ in calls for label in labels)


def test_nominal_close_fallback_filters_detector_inputs(monkeypatch):
    left=bars().iloc[:3].copy()
    left["historical_complete"]=[True,True,False]
    left["is_complete"]=[True,True,False]
    calls=_capture_detector_inputs(monkeypatch)

    result=HistoricalLiquidityReplay(
        left,config=ReplayConfig(timeframe="5M",source="TEST")
    ).run()

    expected=[pd.Timestamp("2026-01-01T00:05:00Z"),pd.Timestamp("2026-01-01T00:10:00Z")]
    assert [item.timestamp for item in result.observations] == expected
    for timestamp in expected:
        visible=tuple(label for call_time, labels, _ in calls if call_time == timestamp for label in labels)
        expected_labels=tuple(left.index[(left.index+pd.Timedelta(minutes=5))<=timestamp][:2])
        assert visible
        assert all(labels == expected_labels for call_time, labels, _ in calls if call_time == timestamp)
        assert left.index[2] not in visible
    right=bars().iloc[:2].copy()
    right.attrs["bar_label"]="right"
    right_calls=_capture_detector_inputs(monkeypatch)
    right_result=HistoricalLiquidityReplay(
        right,config=ReplayConfig(timeframe="5M",source="TEST")
    ).run()
    assert [item.timestamp for item in right_result.observations] == list(right.index)
    for timestamp in right.index:
        visible=tuple(label for call_time, labels, _ in right_calls if call_time == timestamp for label in labels)
        expected_labels=tuple(right.index[right.index<=timestamp])
        assert visible
        assert all(labels == expected_labels for call_time, labels, _ in right_calls if call_time == timestamp)


def test_replay_reports_resolution_label_and_delayed_availability_separately():
    data=pd.DataFrame(
        [
            (100,102,99,101),
            (101,102,95,100),
            (100,101,97,99),
            (99,100,94,96),
            (96,101,95,100),
        ],
        index=pd.date_range("2026-01-01T10:00:00Z",periods=5,freq="5min"),
        columns=["Open","High","Low","Close"],
    )
    data["interval_end"]=data.index+pd.Timedelta(minutes=5)
    data["available_at"]=data["interval_end"]
    data.loc[data.index[4],"available_at"]+=pd.Timedelta(minutes=10)
    data["historical_complete"]=True
    data["is_complete"]=True

    result=HistoricalLiquidityReplay(
        data,
        config=ReplayConfig(timeframe="5M",source="TEST"),
        left_bars=1,
        right_bars=1,
    ).run()
    observation=next(item for item in result.observations if item.resolved_rejections)

    assert observation.timestamp==pd.Timestamp("2026-01-01T10:35:00Z")
    assert observation.latest_rejection_timestamp==data.index[4]
    assert observation.latest_rejection_available_at==pd.Timestamp("2026-01-01T10:35:00Z")
    exported=next(row for row in HistoricalLiquidityReplay.event_rows(result) if row["resolved_rejections"])
    assert exported["latest_rejection_timestamp"]==data.index[4].isoformat()
    assert exported["latest_rejection_available_at"]=="2026-01-01T10:35:00+00:00"
