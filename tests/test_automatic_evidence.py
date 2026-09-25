import pandas as pd
from src.automatic_evidence import EvidenceBuildConfig, build_automatic_evidence


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
    return pd.DataFrame(rows, index=idx, columns=["Open","High","Low","Close"])


def test_empty_visible_data_is_developing_evidence():
    df = _bars()
    ts = df.index[0] - pd.Timedelta(minutes=5)
    result = build_automatic_evidence(ts, df.iloc[:0], None)
    assert result.reason == "EMPTY_VISIBLE_DATA"
    assert result.evidence.as_of == ts
    assert result.evidence.direction is None


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
