import pandas as pd
from src.displacement import detect_displacement, is_confirmed_as_of as disp_confirmed
from src.mss import detect_mss, is_confirmed_as_of as mss_confirmed

def df(rows):
    idx = pd.date_range("2026-01-01 10:00", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx)

def test_displacement_cannot_be_consumed_at_impulse():
    rows = [{"open":100,"high":101,"low":99,"close":100.2} for _ in range(5)]
    rows += [{"open":100,"high":106,"low":99.5,"close":105.5},
             {"open":105,"high":106.5,"low":104.5,"close":106.2}]
    d = df(rows)
    e = detect_displacement(d, 5, "LONG", follow_through_bars=2)
    assert e and e.follow_through
    assert e.timestamp == d.index[5]
    assert e.confirmation_timestamp == d.index[6]
    assert not disp_confirmed(e, d.index[5])
    assert disp_confirmed(e, d.index[6])

def test_displacement_without_follow_through_is_unconfirmed():
    rows = [{"open":100,"high":101,"low":99,"close":100.2} for _ in range(5)]
    rows += [{"open":100,"high":106,"low":99.5,"close":105.5},
             {"open":105,"high":105.5,"low":104,"close":104.8}]
    d = df(rows)
    e = detect_displacement(d, 5, "LONG", follow_through_bars=2)
    assert e and not e.follow_through and e.confirmation_timestamp is None
    assert not disp_confirmed(e, d.index[-1])

def test_mss_confirmation_is_later_than_break():
    rows = [
        {"open":100,"high":101,"low":99,"close":100},
        {"open":100,"high":102,"low":100,"close":101},
        {"open":101,"high":103,"low":101,"close":102},
        {"open":102,"high":102.2,"low":98.5,"close":98.8},
        {"open":98.8,"high":99,"low":97.5,"close":98},
    ]
    d = df(rows)
    e = detect_mss(d, 3, follow_through_bars=2)
    assert e and e.direction == "SHORT"
    assert e.timestamp == d.index[3]
    assert e.confirmation_timestamp == d.index[4]
    assert not mss_confirmed(e, d.index[3])
    assert mss_confirmed(e, d.index[4])
