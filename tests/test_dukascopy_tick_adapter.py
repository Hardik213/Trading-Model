import pandas as pd
import pytest

from src.dukascopy_ticks import (
    TICK_BATCH_COLUMNS,
    discover_dukascopy_tick_files,
    iter_dukascopy_ticks,
)
from src.tick_aggregation import TickAggregator


HEADER = "timestamp,askPrice,bidPrice\n"


def _write(path, rows):
    path.write_text(HEADER + "".join(f"{row}\n" for row in rows), encoding="utf-8")
    return path


def _collect(paths, *, chunk_size=100):
    batches = list(iter_dukascopy_ticks(paths, chunk_size=chunk_size))
    if not batches:
        return pd.DataFrame(columns=TICK_BATCH_COLUMNS)
    return pd.concat(batches, ignore_index=True)


def test_maps_millisecond_timestamps_quotes_and_source_provenance(tmp_path):
    source = _write(
        tmp_path / "month.csv",
        [
            "1641164400245,1829.656,1828.604",
            "1641164400295,1829.866,1828.544",
        ],
    )

    result = _collect([source], chunk_size=1)

    assert result.columns.tolist() == list(TICK_BATCH_COLUMNS)
    assert result["timestamp"].tolist() == [
        pd.Timestamp(1641164400245, unit="ms", tz="UTC"),
        pd.Timestamp(1641164400295, unit="ms", tz="UTC"),
    ]
    assert str(result["timestamp"].dt.tz) == "UTC"
    assert result[["bid", "ask"]].values.tolist() == [
        [1828.604, 1829.656],
        [1828.544, 1829.866],
    ]
    assert result["source_file"].tolist() == [str(source), str(source)]
    assert result["source_row"].tolist() == [1, 2]
    assert result["source_order"].tolist() == [0, 1]
    assert result["raw_timestamp_ms"].tolist() == ["1641164400245", "1641164400295"]


def test_preserves_duplicate_timestamps_and_stable_file_row_order(tmp_path):
    timestamp = "1641164400245"
    first = _write(tmp_path / "2022-01.csv", [f"{timestamp},101,100", f"{timestamp},102,101"])
    second = _write(tmp_path / "2022-02.csv", [f"{timestamp},103,102"])

    result = _collect([first, second], chunk_size=2)

    assert result["timestamp"].duplicated(keep=False).all()
    assert result["bid"].tolist() == [100.0, 101.0, 102.0]
    assert result["source_file"].tolist() == [str(first), str(first), str(second)]
    assert result["source_row"].tolist() == [1, 2, 1]
    assert result["source_order"].tolist() == [0, 1, 2]


def test_directory_discovery_is_deterministic_and_does_not_mutate_input(tmp_path):
    second = _write(
        tmp_path / "xauusd-tick-2022-02-01-2022-03-01.csv",
        ["1641164400300,102,101"],
    )
    first = _write(
        tmp_path / "xauusd-tick-2022-01-01-2022-02-01.csv",
        ["1641164400200,101,100"],
    )
    _write(tmp_path / "XAU-USD_100Tick_BID_2022-03-01.csv", ["not,a,tick"])

    assert discover_dukascopy_tick_files(tmp_path) == [first, second]
    once = _collect([first, second])
    twice = _collect([first, second])
    pd.testing.assert_frame_equal(once, twice)


def test_rejects_decreasing_timestamps_within_file_and_across_files(tmp_path):
    decreasing = _write(
        tmp_path / "decreasing.csv",
        ["1641164400300,102,101", "1641164400200,101,100"],
    )
    with pytest.raises(ValueError, match="decreasing timestamp"):
        _collect([decreasing])

    first = _write(tmp_path / "first.csv", ["1641164400300,102,101"])
    second = _write(tmp_path / "second.csv", ["1641164400200,101,100"])
    with pytest.raises(ValueError, match="decreasing timestamp"):
        _collect([first, second])


@pytest.mark.parametrize(
    "row",
    [
        "1641164400245,101",  # too few fields
        "1641164400245,101,100,extra",  # too many fields
        "not-a-timestamp,101,100",
        "-1,101,100",
        "1641164400245,,100",
        "1641164400245,NaN,100",
        "1641164400245,Infinity,100",
        "1641164400245,101,0",
        "1641164400245,101,-1",
        "1641164400245,100,101",  # crossed quote
    ],
)
def test_rejects_invalid_tick_records_with_row_provenance(tmp_path, row):
    source = _write(tmp_path / "bad.csv", [row])

    with pytest.raises(ValueError, match=r"bad.csv.*row 1"):
        _collect([source])


def test_rejects_blank_and_malformed_csv_rows(tmp_path):
    blank = tmp_path / "blank.csv"
    blank.write_text(HEADER + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank.csv.*row 1"):
        _collect([blank])

    malformed = tmp_path / "malformed.csv"
    malformed.write_text(HEADER + '1641164400245,"101,100\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed.csv"):
        _collect([malformed])


def test_rejects_noncanonical_header_and_missing_files(tmp_path):
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("timestamp,bid,ask\n1641164400245,100,101\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact schema"):
        _collect([wrong])

    with pytest.raises(FileNotFoundError):
        _collect([tmp_path / "missing.csv"])


def test_chunking_is_bounded_and_lazy(tmp_path):
    source = tmp_path / "stream.csv"
    source.write_text(
        HEADER
        + "1641164400000,101,100\n"
        + "1641164400100,102,101\n"
        + "1641164400200,103,102\n"
        + "bad-timestamp,104,103\n",
        encoding="utf-8",
    )

    batches = iter_dukascopy_ticks([source], chunk_size=2)
    first = next(batches)
    assert len(first) == 2
    assert len(first) <= 2
    with pytest.raises(ValueError, match="row 4"):
        next(batches)


def test_adapter_output_is_accepted_by_existing_tick_aggregator(tmp_path):
    source = _write(
        tmp_path / "boundary.csv",
        [
            "1641164400000,101,100",
            "1641164430000,102,101",
            "1641164460000,103,102",
            "1641164490000,104,103",
            "1641164520000,105,104",
        ],
    )
    aggregator = TickAggregator("1min")
    completed = []
    for batch in iter_dukascopy_ticks([source], chunk_size=2):
        for tick in batch.itertuples(index=False):
            completed.extend(aggregator.add_tick(tick.timestamp, tick.bid, tick.ask))
    completed.extend(aggregator.flush(include_incomplete=True))

    assert len(completed) == 3
    assert completed[0]["bid_open"] == 100.0
    assert completed[0]["ask_close"] == 102.0
    assert completed[0]["available_at"] == pd.Timestamp("2022-01-02T23:01:00Z")
    assert completed[0]["historical_complete"] is True
    assert completed[-1]["historical_complete"] is False
    assert pd.isna(completed[-1]["available_at"])
