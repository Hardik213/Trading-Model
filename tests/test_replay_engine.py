import pandas as pd
import pytest

from src.data_contract import normalize_ohlc
from src.pipeline import ICT2022Pipeline
from src.replay_engine import (
    HistoricalReplay,
    ReplayConfig,
    ReplayDecision,
    ReplayObservation,
    assert_no_future_data,
)
from src.tick_aggregation import TickAggregator
from src.timeframe_context import TimeframeContext, bars_available_as_of, build_context


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


def test_normalizer_preserves_event_time_availability():
    raw=frame(n=2)
    raw["available_at"] = pd.to_datetime(
        ["2026-01-01T10:05:00Z", "2026-01-01T10:12:30Z"], utc=True
    )
    raw["availability_ts"] = raw["available_at"]
    raw["interval_end"] = pd.to_datetime(
        ["2026-01-01T10:05:00Z", "2026-01-01T10:10:00Z"], utc=True
    )
    out=normalize_ohlc(raw,timeframe="5M")
    assert out["available_at"].tolist() == raw["available_at"].tolist()
    assert out["availability_ts"].tolist() == raw["availability_ts"].tolist()
    assert out["interval_end"].tolist() == raw["interval_end"].tolist()
    assert out.index.tolist() == raw.index.tolist()


def test_normalizer_rejects_event_time_mode_without_availability():
    raw=frame(n=1)
    raw.attrs["availability_mode"]="event_time"
    with pytest.raises(ValueError,match="availability"):
        normalize_ohlc(raw,timeframe="5M")


def test_normalizer_rejects_availability_before_nominal_close():
    raw=frame(n=1)
    raw["available_at"]=pd.to_datetime(["2026-01-01T10:02:00Z"],utc=True)
    with pytest.raises(ValueError,match="close"):
        normalize_ohlc(raw,timeframe="5M")


def test_normalizer_rejects_availability_before_nominal_close_without_interval_end():
    raw=frame(n=1)
    raw["available_at"]=pd.to_datetime(["2026-01-01T10:02:00Z"],utc=True)
    with pytest.raises(ValueError,match="close"):
        normalize_ohlc(raw,timeframe="5M")


def test_normalizer_rejects_decreasing_event_availability():
    raw=frame(n=2)
    raw["interval_end"]=raw.index+pd.Timedelta(minutes=5)
    raw["available_at"]=pd.to_datetime(
        ["2026-01-01T10:15:00Z","2026-01-01T10:12:00Z"],utc=True
    )
    with pytest.raises(ValueError,match="non-decreasing"):
        normalize_ohlc(raw,timeframe="5M")


@pytest.mark.parametrize("column",["available_at","availability_ts","interval_end"])
def test_normalizer_rejects_naive_availability_timestamps(column):
    raw=frame(n=1)
    raw[column]=["2026-01-01T10:05:00"]
    with pytest.raises(ValueError,match="timezone-aware"):
        normalize_ohlc(raw,timeframe="5M")


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


def test_replay_uses_event_availability_and_hides_incomplete_bars():
    aggregator=TickAggregator("1min")
    emitted=[]
    for timestamp, bid, ask in [
        ("2026-01-01T00:00:00Z", 100.0, 101.0),
        ("2026-01-01T00:01:00Z", 101.0, 102.0),
        ("2026-01-01T00:03:30Z", 103.0, 104.0),
    ]:
        emitted.extend(aggregator.add_tick(timestamp, bid, ask))
    emitted.extend(aggregator.flush(include_incomplete=True))

    bars=pd.DataFrame(emitted).set_index("interval_start")
    bars=bars.rename(columns={
        "bid_open": "Open",
        "bid_high": "High",
        "bid_low": "Low",
        "bid_close": "Close",
    })
    context=TimeframeContext(frames={"1min": bars.copy()})
    replay=HistoricalReplay(
        bars,
        context=context,
        config=ReplayConfig(timeframe="1min",source="TEST"),
    )

    assert replay.data["available_at"].iloc[0] == pd.Timestamp("2026-01-01T00:01:00Z")
    assert pd.isna(replay.data["available_at"].iloc[-1])
    assert replay.decision_times().tolist() == [
        pd.Timestamp("2026-01-01T00:01:00Z"),
        pd.Timestamp("2026-01-01T00:03:30Z"),
    ]

    seen=[]
    def callback(timestamp, visible, visible_context):
        assert visible["available_at"].notna().all()
        assert (visible["available_at"] <= timestamp).all()
        context_bars=visible_context.frames["1min"]
        assert context_bars["available_at"].notna().all()
        assert (context_bars["available_at"] <= timestamp).all()
        seen.append((timestamp, len(visible), len(context_bars)))
        return ReplayObservation(timestamp,ReplayDecision.DEVELOPING,"DEVELOPING","test")

    replay.run(callback)
    assert seen == [
        (pd.Timestamp("2026-01-01T00:01:00Z"), 1, 1),
        (pd.Timestamp("2026-01-01T00:03:30Z"), 2, 2),
    ]


def test_delayed_bar_is_visible_only_at_its_finalizing_tick():
    aggregator=TickAggregator("1min")
    emitted=[]
    for timestamp, bid, ask in [
        ("2026-01-01T00:00:00Z", 100.0, 101.0),
        ("2026-01-01T00:00:30Z", 100.5, 101.5),
        ("2026-01-01T00:02:30Z", 102.0, 103.0),
    ]:
        emitted.extend(aggregator.add_tick(timestamp, bid, ask))
    emitted.extend(aggregator.flush(include_incomplete=True))
    bars=pd.DataFrame(emitted).set_index("interval_start").rename(columns={
        "bid_open": "Open", "bid_high": "High", "bid_low": "Low", "bid_close": "Close",
    })
    replay=HistoricalReplay(
        bars,
        context=TimeframeContext(frames={"1min": bars.copy()}),
        config=ReplayConfig(timeframe="1min",source="TEST"),
    )

    completed=bars.loc[pd.Timestamp("2026-01-01T00:00:00Z")]
    assert completed["interval_end"] == pd.Timestamp("2026-01-01T00:01:00Z")
    assert completed["available_at"] == pd.Timestamp("2026-01-01T00:02:30Z")
    assert pd.Timestamp("2026-01-01T00:01:00Z") not in replay.decision_times()
    assert replay.decision_times().tolist() == [pd.Timestamp("2026-01-01T00:02:30Z")]

    seen=[]
    replay.run(lambda ts, base, context: (
        seen.append((ts, len(base), len(context.frames["1min"])))
        or ReplayObservation(ts,ReplayDecision.DEVELOPING,"DEVELOPING","test")
    ))
    assert seen == [(pd.Timestamp("2026-01-01T00:02:30Z"), 1, 1)]


def test_exact_boundary_finalizes_bar_at_interval_end():
    aggregator=TickAggregator("1min")
    assert aggregator.add_tick("2026-01-01T00:00:00Z",100.0,101.0) == []
    emitted=aggregator.add_tick("2026-01-01T00:01:00Z",101.0,102.0)

    assert len(emitted) == 1
    assert emitted[0]["interval_start"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert emitted[0]["interval_end"] == pd.Timestamp("2026-01-01T00:01:00Z")
    assert emitted[0]["available_at"] == pd.Timestamp("2026-01-01T00:01:00Z")
    assert emitted[0]["historical_complete"] is True


def test_multiple_timeframes_share_delayed_finalizing_event_time():
    ticks=[
        ("2026-01-01T00:00:00Z", 100.0, 101.0),
        ("2026-01-01T00:07:30Z", 101.0, 102.0),
    ]
    minute=TickAggregator("1min")
    five_minute=TickAggregator("5min")
    minute_rows=[]
    five_minute_rows=[]
    for timestamp, bid, ask in ticks:
        minute_rows.extend(minute.add_tick(timestamp,bid,ask))
        five_minute_rows.extend(five_minute.add_tick(timestamp,bid,ask))

    minute_bar=minute_rows[0]
    context_bar=five_minute_rows[0]
    assert len(minute_rows) == len(five_minute_rows) == 1
    assert minute_bar["interval_end"] == pd.Timestamp("2026-01-01T00:01:00Z")
    assert minute_bar["available_at"] == pd.Timestamp("2026-01-01T00:07:30Z")
    assert context_bar["interval_end"] == pd.Timestamp("2026-01-01T00:05:00Z")
    assert context_bar["available_at"] == minute_bar["available_at"]

    base=pd.DataFrame([minute_bar]).set_index("interval_start").rename(columns={
        "bid_open": "Open", "bid_high": "High", "bid_low": "Low", "bid_close": "Close",
    })
    higher=pd.DataFrame([context_bar]).set_index("interval_start").rename(columns={
        "bid_open": "Open", "bid_high": "High", "bid_low": "Low", "bid_close": "Close",
    })
    replay=HistoricalReplay(
        base,
        context=TimeframeContext(frames={"5min": higher}),
        config=ReplayConfig(timeframe="1min",source="TEST"),
    )
    assert replay.decision_times().tolist() == [pd.Timestamp("2026-01-01T00:07:30Z")]
    seen=[]
    replay.run(lambda ts, visible, context: (
        seen.append((ts,len(visible),len(context.available_as_of("5min",ts))))
        or ReplayObservation(ts,ReplayDecision.DEVELOPING,"DEVELOPING","test")
    ))
    assert seen == [(pd.Timestamp("2026-01-01T00:07:30Z"),1,1)]


def test_legacy_ohlc_replay_uses_explicit_nominal_close_fallback():
    base=frame(start="2026-01-01T00:00:00Z",n=2)
    replay=HistoricalReplay(
        base,
        context=TimeframeContext(frames={"5min":base.copy()}),
        config=ReplayConfig(timeframe="5min",source="TEST"),
    )
    assert replay.data.attrs["availability_mode"] == "nominal_close_fallback"
    assert replay.decision_times().tolist() == [
        pd.Timestamp("2026-01-01T00:05:00Z"),
        pd.Timestamp("2026-01-01T00:10:00Z"),
    ]

    seen=[]
    replay.run(lambda ts, visible, context: (
        seen.append((ts,len(visible),len(context.available_as_of("5min",ts))))
        or ReplayObservation(ts,ReplayDecision.DEVELOPING,"DEVELOPING","test")
    ))
    assert seen == [
        (pd.Timestamp("2026-01-01T00:05:00Z"),1,1),
        (pd.Timestamp("2026-01-01T00:10:00Z"),2,2),
    ]


def test_normalizer_preserves_explicit_right_label_fallback():
    base=frame(start="2026-01-01T00:05:00Z",n=2)
    base.attrs["bar_label"] = "right"
    replay=HistoricalReplay(
        base,
        context=TimeframeContext(frames={"5min":base.copy()}),
        config=ReplayConfig(timeframe="5min",source="TEST"),
    )

    assert replay.data.attrs["bar_label"] == "right"
    assert replay.decision_times().tolist() == list(base.index)


def test_fallback_context_uses_only_nominally_available_constituents():
    base=frame(start="2026-01-01T10:00:00Z",n=4)
    context=build_context(base,base_timeframe="5M",source="TEST")
    replay=HistoricalReplay(
        base,
        context=context,
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )
    snapshots={}

    def callback(timestamp, visible_base, visible_context):
        if timestamp in {
            pd.Timestamp("2026-01-01T10:15:00Z"),
            pd.Timestamp("2026-01-01T10:20:00Z"),
        }:
            htf=visible_context.frames["15M"]
            snapshots[timestamp]=(
                visible_base.index.tolist(),
                htf.index.tolist(),
                htf["Close"].tolist(),
            )
        return ReplayObservation(timestamp,ReplayDecision.DEVELOPING,"DEVELOPING","fallback context")

    replay.run(callback)

    boundary=pd.Timestamp("2026-01-01T10:15:00Z")
    next_base_close=pd.Timestamp("2026-01-01T10:20:00Z")
    assert context.available_as_of("15M",boundary-pd.Timedelta(microseconds=1)).empty
    assert snapshots[boundary] == (
        list(base.index[:3]),
        [boundary],
        [102],
    )
    assert snapshots[next_base_close] == (
        list(base.index),
        [boundary],
        [102],
    )


def test_fallback_hour_context_waits_for_all_five_minute_constituents():
    base=frame(start="2026-01-01T10:00:00Z",n=12)
    context=build_context(base,base_timeframe="5M",source="TEST")
    hour_close=pd.Timestamp("2026-01-01T11:00:00Z")

    assert context.available_as_of(
        "1H",hour_close-pd.Timedelta(microseconds=1)
    ).empty
    visible=context.available_as_of("1H",hour_close)
    assert visible.index.tolist() == [hour_close]
    assert visible.loc[hour_close,"Open"] == 100
    assert visible.loc[hour_close,"Close"] == 111


def test_fallback_context_withholds_partial_and_gapped_groups():
    partial=frame(start="2026-01-01T10:00:00Z",n=2)
    partial_context=build_context(partial,base_timeframe="5M",source="TEST")
    boundary=pd.Timestamp("2026-01-01T10:15:00Z")
    assert partial_context.available_as_of("15M",boundary).empty

    gapped=frame(start="2026-01-01T10:00:00Z",n=4).drop(
        pd.Timestamp("2026-01-01T10:05:00Z")
    )
    gapped_context=build_context(gapped,base_timeframe="5M",source="TEST")
    assert gapped_context.available_as_of("15M",boundary).empty

    flagged_incomplete=frame(start="2026-01-01T10:00:00Z",n=3)
    flagged_incomplete["historical_complete"]=[True,True,False]
    flagged_incomplete["is_complete"]=[True,True,False]
    incomplete_context=build_context(flagged_incomplete,base_timeframe="5M",source="TEST")
    assert incomplete_context.available_as_of("15M",boundary).empty


def test_timeframe_context_filters_by_delayed_event_availability():
    context_bars=frame(start="2026-01-01T00:00:00Z",n=2)
    context_bars["interval_end"] = context_bars.index + pd.Timedelta(minutes=5)
    context_bars["available_at"] = pd.to_datetime(
        ["2026-01-01T00:07:30Z", "2026-01-01T00:12:30Z"], utc=True
    )
    context=TimeframeContext(frames={"5min":context_bars})

    assert context.available_as_of("5min",pd.Timestamp("2026-01-01T00:07:29Z")).empty
    visible=context.available_as_of("5min",pd.Timestamp("2026-01-01T00:07:30Z"))
    assert visible.index.tolist() == [pd.Timestamp("2026-01-01T00:00:00Z")]


def test_supplied_context_rejects_conflicting_availability_aliases():
    context_bars=frame(start="2026-01-01T00:00:00Z",n=1)
    context_bars["available_at"]=pd.to_datetime(["2026-01-01T00:05:00Z"],utc=True)
    context_bars["availability_ts"]=pd.to_datetime(["2026-01-01T00:06:00Z"],utc=True)
    context=TimeframeContext(frames={"5min":context_bars})

    with pytest.raises(ValueError,match="must match"):
        context.available_as_of("5min",pd.Timestamp("2026-01-01T00:06:00Z"))


def test_supplied_context_rejects_availability_before_interval_end():
    context_bars=frame(start="2026-01-01T00:00:00Z",n=1)
    context_bars["interval_end"]=pd.to_datetime(["2026-01-01T00:05:00Z"],utc=True)
    context_bars["available_at"]=pd.to_datetime(["2026-01-01T00:02:00Z"],utc=True)
    context=TimeframeContext(frames={"5min":context_bars})

    with pytest.raises(ValueError,match="interval end"):
        context.available_as_of("5min",pd.Timestamp("2026-01-01T00:05:00Z"))


@pytest.mark.parametrize("column",["available_at","availability_ts","interval_end"])
def test_supplied_event_context_rejects_naive_timing_metadata(column):
    context_bars=frame(start="2026-01-01T00:00:00Z",n=1)
    context_bars[column]=["2026-01-01T00:05:00"]
    context=TimeframeContext(frames={"5min":context_bars})

    with pytest.raises(ValueError,match="timezone-aware"):
        context.available_as_of("5min",pd.Timestamp("2026-01-01T00:05:00Z"))


def test_event_time_context_omits_groups_with_missing_constituents():
    labels=pd.DatetimeIndex(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:10:00Z",
            "2026-01-01T00:15:00Z",
            "2026-01-01T00:20:00Z",
            "2026-01-01T00:25:00Z",
        ]
    )
    event_bars=pd.DataFrame(
        {
            "Open":[100,101,102,103,104],
            "High":[101,102,103,104,105],
            "Low":[99,100,101,102,103],
            "Close":[100,101,102,103,104],
            "interval_end":labels+pd.Timedelta(minutes=5),
            "available_at":labels+pd.Timedelta(minutes=5),
            "historical_complete":[True]*5,
            "is_complete":[True]*5,
        },
        index=labels,
    )
    context=build_context(event_bars,base_timeframe="5M",source="TEST")
    fifteen=context.frames["15M"]

    assert pd.Timestamp("2026-01-01T00:15:00Z") not in fifteen.index
    assert pd.Timestamp("2026-01-01T00:30:00Z") in fifteen.index
    assert pd.Timestamp("2026-01-01T00:15:00Z") not in context.available_as_of(
        "15M",pd.Timestamp("2026-01-01T00:30:00Z")
    ).index
    assert fifteen.loc[pd.Timestamp("2026-01-01T00:30:00Z"),"available_at"] == pd.Timestamp(
        "2026-01-01T00:30:00Z"
    )


def test_nominal_close_decisions_skip_incomplete_rows():
    base=frame(start="2026-01-01T10:00:00Z",n=3)
    base["historical_complete"]=[True,True,False]
    base["is_complete"]=[True,True,False]
    replay=HistoricalReplay(
        base,
        context=TimeframeContext(frames={}),
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )
    seen=[]

    def callback(timestamp,visible,context):
        seen.append((timestamp,tuple(visible.index)))
        return ReplayObservation(timestamp,ReplayDecision.DEVELOPING,"DEVELOPING","test")

    replay.run(callback)
    assert [timestamp for timestamp,_ in seen] == [
        pd.Timestamp("2026-01-01T10:05:00Z"),
        pd.Timestamp("2026-01-01T10:10:00Z"),
    ]
    assert seen == [
        (pd.Timestamp("2026-01-01T10:05:00Z"),(base.index[0],)),
        (pd.Timestamp("2026-01-01T10:10:00Z"),(base.index[0],base.index[1])),
    ]


def test_build_context_propagates_delayed_and_incomplete_event_times():
    aggregator=TickAggregator("1min")
    emitted=[]
    for minute in range(15):
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z")+pd.Timedelta(minutes=minute)
        emitted.extend(aggregator.add_tick(timestamp,100.0+minute,101.0+minute))
    emitted.extend(aggregator.add_tick("2026-01-01T00:16:30Z",115.0,116.0))
    complete=pd.DataFrame(emitted).set_index("interval_start").rename(columns={
        "bid_open":"Open", "bid_high":"High", "bid_low":"Low", "bid_close":"Close",
    })
    complete_context=build_context(complete,base_timeframe="1min",source="TEST")
    higher=complete_context.frames["15M"]
    higher_close=pd.Timestamp("2026-01-01T00:15:00Z")
    assert higher.loc[higher_close,"available_at"] == pd.Timestamp("2026-01-01T00:16:30Z")
    assert complete_context.available_as_of("15M",pd.Timestamp("2026-01-01T00:15:00Z")).empty
    assert higher_close in complete_context.available_as_of(
        "15M",pd.Timestamp("2026-01-01T00:16:30Z")
    ).index

    partial=complete.iloc[:6].copy()
    partial_context=build_context(partial,base_timeframe="1min",source="TEST")
    partial_higher=partial_context.frames["15M"]
    partial_close=pd.Timestamp("2026-01-01T00:15:00Z")
    assert partial_close not in partial_higher.index
    assert partial_context.available_as_of(
        "15M",pd.Timestamp("2026-01-01T00:20:00Z")
    ).empty


def test_assert_no_future_data_checks_event_availability():
    visible=frame(n=2)
    visible["available_at"] = pd.to_datetime(
        ["2026-01-01T10:05:00Z", "2026-01-01T10:12:30Z"], utc=True
    )
    with pytest.raises(AssertionError, match="availability"):
        assert_no_future_data(visible,as_of=pd.Timestamp("2026-01-01T10:10:00Z"))


def test_pipeline_runs_as_one_chronological_unit():
    base=frame(n=5)
    pipeline=ICT2022Pipeline.from_base_data(base,timeframe="5M",source="TEST")
    observations=pipeline.run_observation_pass()
    assert len(observations)==5
    assert all(x.decision is ReplayDecision.DEVELOPING for x in observations)
    assert [x.timestamp for x in observations] == list(
        base.index + pd.Timedelta(minutes=5)
    )


def test_endpoint_only_metadata_controls_normalization_and_visibility():
    raw=frame(start="2026-01-01T10:00:00Z",n=2)
    raw["interval_end"]=pd.to_datetime(
        ["2026-01-01T10:20:00Z","2026-01-01T10:25:00Z"],utc=True
    )

    normalized=normalize_ohlc(raw,timeframe="5M")

    assert normalized["available_at"].tolist()==raw["interval_end"].tolist()
    assert normalized.attrs["availability_mode"]=="event_time"
    assert bars_available_as_of(
        normalized,pd.Timestamp("2026-01-01T10:05:00Z"),timeframe="5M"
    ).empty
    assert bars_available_as_of(
        normalized,pd.Timestamp("2026-01-01T10:20:00Z"),timeframe="5M"
    ).index.tolist()==[raw.index[0]]


def test_endpoint_only_metadata_rejects_decreasing_interval_ends():
    raw=frame(start="2026-01-01T10:00:00Z",n=2)
    raw["interval_end"]=pd.to_datetime(
        ["2026-01-01T10:20:00Z","2026-01-01T10:10:00Z"],utc=True
    )

    with pytest.raises(ValueError,match="non-decreasing"):
        normalize_ohlc(raw,timeframe="5M")


def test_endpoint_only_metadata_rejects_end_before_label():
    raw=frame(start="2026-01-01T10:00:00Z",n=1)
    raw["interval_end"]=pd.to_datetime(["2026-01-01T09:59:00Z"],utc=True)

    with pytest.raises(ValueError,match="interval end"):
        normalize_ohlc(raw,timeframe="5M")


def test_explicit_interval_end_rejects_missing_endpoint():
    for with_availability in (False,True):
        raw=frame(start="2026-01-01T10:00:00Z",n=1)
        raw["interval_end"]=pd.to_datetime([None],utc=True)
        if with_availability:
            raw["available_at"]=pd.to_datetime(["2026-01-01T10:05:00Z"],utc=True)

        with pytest.raises(ValueError,match="interval end"):
            normalize_ohlc(raw,timeframe="5M")


def test_endpoint_only_metadata_rejects_naive_endpoints():
    raw=frame(start="2026-01-01T10:00:00Z",n=1)
    raw["interval_end"]=["2026-01-01T10:05:00"]

    with pytest.raises(ValueError,match="timezone-aware"):
        normalize_ohlc(raw,timeframe="5M")


def test_availability_cannot_precede_explicit_interval_end():
    raw=frame(start="2026-01-01T10:00:00Z",n=1)
    raw["interval_end"]=pd.to_datetime(["2026-01-01T10:10:00Z"],utc=True)
    raw["available_at"]=pd.to_datetime(["2026-01-01T10:05:00Z"],utc=True)

    with pytest.raises(ValueError,match="interval end"):
        normalize_ohlc(raw,timeframe="5M")


def test_endpoint_only_supplied_context_waits_for_explicit_interval_end():
    context_bars=frame(start="2026-01-01T10:00:00Z",n=1)
    context_bars["interval_end"]=pd.to_datetime(["2026-01-01T10:20:00Z"],utc=True)
    context=TimeframeContext(frames={"5M":context_bars})

    assert context.available_as_of(
        "5M",pd.Timestamp("2026-01-01T10:05:00Z")
    ).empty
    assert context.available_as_of(
        "5M",pd.Timestamp("2026-01-01T10:20:00Z")
    ).index.tolist()==[context_bars.index[0]]


def test_endpoint_only_replay_schedules_decisions_at_interval_ends():
    raw=frame(start="2026-01-01T10:00:00Z",n=2)
    raw["interval_end"]=pd.to_datetime(
        ["2026-01-01T10:20:00Z","2026-01-01T10:25:00Z"],utc=True
    )
    replay=HistoricalReplay(
        raw,
        context=TimeframeContext(frames={"5M":raw.copy()}),
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )
    seen=[]

    def callback(timestamp,visible,context):
        seen.append((timestamp,tuple(visible.index)))
        return ReplayObservation(timestamp,ReplayDecision.DEVELOPING,"DEVELOPING","test")

    replay.run(callback)

    assert [timestamp for timestamp,_ in seen]==[
        pd.Timestamp("2026-01-01T10:20:00Z"),
        pd.Timestamp("2026-01-01T10:25:00Z"),
    ]
    assert seen[0][1]==(raw.index[0],)


def test_endpoint_only_incomplete_bar_never_becomes_a_replay_decision():
    raw=frame(start="2026-01-01T10:00:00Z",n=2)
    raw["interval_end"]=pd.to_datetime(
        ["2026-01-01T10:05:00Z","2026-01-01T10:10:00Z"],utc=True
    )
    raw["historical_complete"]=[True,False]
    raw["is_complete"]=[True,False]
    replay=HistoricalReplay(
        raw,
        context=TimeframeContext(frames={}),
        config=ReplayConfig(timeframe="5M",source="TEST"),
    )

    assert replay.decision_times().tolist()==[pd.Timestamp("2026-01-01T10:05:00Z")]
