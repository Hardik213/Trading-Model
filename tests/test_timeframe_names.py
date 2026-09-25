from src.timeframe_names import FILE_LABELS
def test_minute_labels_are_unambiguous():
    assert FILE_LABELS["1min"] == "1MIN"
    assert FILE_LABELS["3min"] == "3MIN"
    assert FILE_LABELS["5min"] == "5MIN"
    assert FILE_LABELS["1D"] == "1D"
