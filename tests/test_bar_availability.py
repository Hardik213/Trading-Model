import pandas as pd
from src.bar_availability import completed_bar_cutoff

def test_left_label_not_visible_until_close():
    idx = pd.date_range("2022-01-01 12:00", periods=2, freq="5min", tz="UTC")
    assert pd.isna(completed_bar_cutoff(idx, pd.Timestamp("2022-01-01 12:04:59Z"), "5min"))
    assert completed_bar_cutoff(idx, pd.Timestamp("2022-01-01 12:05:00Z"), "5min") == idx[0]

def test_second_bar_visible_only_after_second_close():
    idx = pd.date_range("2022-01-01 12:00", periods=3, freq="5min", tz="UTC")
    assert completed_bar_cutoff(idx, pd.Timestamp("2022-01-01 12:10:00Z"), "5min") == idx[1]
