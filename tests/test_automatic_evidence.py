import pandas as pd
from src.automatic_evidence import EvidenceBuildConfig, build_automatic_evidence
from src.data_contract import normalize_ohlc
from src.replay_subject import ReplaySubject
from src.timeframe_context import TimeframeContext


def _bars(n=40):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    rows = []
    price = 100.0
    for i, ts in enumerate(idx):
        # Deterministic quiet market with no qualifying displacement/liquidity.
        o = price
        c = price + (0.01 if i % 2 == 0 else -0.01)
        h = max(o, c) + 0.02
        l = min(o, c) - 0.02
        rows.append((o,h,l,c))
        price = c
    result = pd.DataFrame(rows, index=idx, columns=["Open","High","Low","Close"])
    result.attrs["timeframe"] = "5min"
    result.attrs["bar_label"] = "left"
    result.attrs["availability_mode"] = "nominal_close_fallback"
    return result


def test_empty_visible_data_is_developing_evidence():
    df = _bars()
    ts = df.index[0] - pd.Timedelta(minutes=5)
    result = build_automatic_evidence(ts, df.iloc[:0], None)
    assert result.reason == "EMPTY_VISIBLE_DATA"
    assert result.evidence.as_of == ts
    assert result.evidence.direction is None


def test_automatic_evidence_preserves_optional_replay_subject():
    df = _bars()
    as_of = df.index[0] - pd.Timedelta(minutes=5)
    subject = ReplaySubject(as_of, as_of)
    context = TimeframeContext(frames={}, replay_subject=subject)

    result = build_automatic_evidence(as_of, df.iloc[:0], context)

    assert result.evidence.as_of == as_of
    assert result.evidence.replay_subject == subject


def test_no_future_data_is_used_when_no_rejection_exists():
    df = _bars()
    ts = df.index[20]
    result = build_automatic_evidence(ts, df.loc[:ts], None)
    assert result.reason == "NO_CONFIRMED_LIQUIDITY_REJECTION"
    assert result.evidence.as_of == ts


def test_builder_rejects_missing_post_mss_array_in_safe_configuration(monkeypatch):
    # This is an integration seam test: a builder must never invent an FVG.
    # We monkeypatch the expensive detector chain to return no MSS.
    import src.automatic_evidence as ae
    monkeypatch.setattr(ae, "latest_reversal_liquidity", lambda *a, **k: None)
    df = _bars()
    ts = df.index[-1]
    result = ae.build_automatic_evidence(ts, df, None)
    assert result.reason == "NO_CONFIRMED_LIQUIDITY_REJECTION"


def test_builder_filters_event_time_and_incomplete_bars(monkeypatch):
    import src.automatic_evidence as ae

    df=_bars(n=3)
    df["interval_end"]=df.index+pd.Timedelta(minutes=5)
    df["available_at"]=pd.to_datetime(
        ["2026-01-01T00:05:00Z","2026-01-01T00:12:30Z",None],utc=True
    )
    df["historical_complete"]=[True,True,False]
    df["is_complete"]=[True,True,False]
    df.attrs["availability_mode"]="event_time"
    observed=[]

    def no_reversal(visible,*args,**kwargs):
        observed.append(tuple(visible.index))
        return None

    monkeypatch.setattr(ae,"latest_reversal_liquidity",no_reversal)

    early=build_automatic_evidence(pd.Timestamp("2026-01-01T00:10:00Z"),df,None)
    assert early.reason == "NO_CONFIRMED_LIQUIDITY_REJECTION"
    assert observed == [(df.index[0],),(df.index[0],)]

    observed.clear()
    exact=build_automatic_evidence(pd.Timestamp("2026-01-01T00:12:30Z"),df,None)
    assert exact.reason == "NO_CONFIRMED_LIQUIDITY_REJECTION"
    assert observed == [(df.index[0],df.index[1]),(df.index[0],df.index[1])]


def test_builder_uses_nominal_close_for_normalized_fallback(monkeypatch):
    import src.automatic_evidence as ae

    df=normalize_ohlc(_bars(n=3),timeframe="5M",source="TEST")
    observed=[]

    def no_reversal(visible,*args,**kwargs):
        observed.append(tuple(visible.index))
        return None

    monkeypatch.setattr(ae,"latest_reversal_liquidity",no_reversal)

    build_automatic_evidence(pd.Timestamp("2026-01-01T00:11:00Z"),df,None)
    assert observed == [(df.index[0],df.index[1]),(df.index[0],df.index[1])]
