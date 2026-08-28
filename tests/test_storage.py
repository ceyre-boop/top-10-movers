from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from top10 import storage


def _df_with_as_of(as_of_values):
    return pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(len(as_of_values))],
            "trade_date": [dt.datetime(2024, 1, 5)] * len(as_of_values),
            "close": [1.0] * len(as_of_values),
            "as_of": as_of_values,
        }
    )


def test_write_parquet_requires_as_of_column(tmp_path):
    df = pd.DataFrame({"ticker": ["A"], "close": [1.0]})
    with pytest.raises(storage.LeakageError):
        storage.write_parquet(df, tmp_path / "out.parquet")


def test_write_parquet_allows_missing_as_of_when_not_required(tmp_path):
    df = pd.DataFrame({"ticker": ["A"], "close": [1.0]})
    path = tmp_path / "out.parquet"
    storage.write_parquet(df, path, as_of_required=False)
    assert path.exists()


def test_write_parquet_creates_parent_dirs(tmp_path):
    df = _df_with_as_of([dt.datetime(2024, 1, 1)])
    path = tmp_path / "nested" / "dir" / "out.parquet"
    storage.write_parquet(df, path)
    assert path.exists()


def test_read_parquet_as_of_filters_future_rows(tmp_path):
    df = _df_with_as_of(
        [
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 1, 5),
            dt.datetime(2024, 1, 10),
        ]
    )
    path = tmp_path / "bars.parquet"
    storage.write_parquet(df, path)

    result = storage.read_parquet(path, as_of=dt.datetime(2024, 1, 5))
    assert len(result) == 2
    assert result["as_of"].max() <= dt.datetime(2024, 1, 5)
    assert dt.datetime(2024, 1, 10) not in result["as_of"].tolist()


def test_read_parquet_without_as_of_returns_all_rows(tmp_path):
    df = _df_with_as_of([dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 10)])
    path = tmp_path / "bars.parquet"
    storage.write_parquet(df, path)

    result = storage.read_parquet(path)
    assert len(result) == 2


def test_assert_as_of_le_raises_on_leaking_row():
    df = _df_with_as_of([dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 10)])
    with pytest.raises(storage.LeakageError):
        storage.assert_as_of_le(df, dt.datetime(2024, 1, 5))


def test_assert_as_of_le_passes_when_clean():
    df = _df_with_as_of([dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 4)])
    storage.assert_as_of_le(df, dt.datetime(2024, 1, 5))


def test_append_only_write_refuses_second_write(tmp_path):
    path = tmp_path / "predictions" / "pred.json"
    storage.append_only_write({"ticker": "AAPL", "score": 0.9}, path)
    assert path.exists()

    with pytest.raises(FileExistsError):
        storage.append_only_write({"ticker": "AAPL", "score": 0.95}, path)


def test_spec_dir_creates_and_returns_path(tmp_path):
    d = storage.spec_dir(tmp_path, "abc123")
    assert d == tmp_path / "abc123"
    assert d.is_dir()
