"""Offline tests for top10/data adapters. No network access, no API keys."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from top10.data import _conform, get_source
from top10.data.base import DAILY_BARS_COLUMNS, EARNINGS_COLUMNS
from top10.data.cache import cached_call
from top10.data.databento import DatabentoSource
from top10.data.polygon import PolygonSource

FIXTURES = Path(__file__).parent / "fixtures"


# --- import / construction --------------------------------------------------


def test_adapters_import_and_construct_with_no_api_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    polygon_source = PolygonSource()
    databento_source = DatabentoSource()

    assert polygon_source.name == "polygon"
    assert databento_source.name == "databento"


def test_get_source_defaults_to_polygon(monkeypatch):
    monkeypatch.delenv("MARKET_DATA_VENDOR", raising=False)
    src = get_source()
    assert src.name == "polygon"


def test_get_source_reads_env_vendor(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_VENDOR", "databento")
    src = get_source()
    assert src.name == "databento"


def test_get_source_rejects_unknown_vendor():
    with pytest.raises(ValueError, match="valid vendors"):
        get_source("not-a-real-vendor")


# --- _conform ----------------------------------------------------------------


def test_conform_raises_on_missing_column():
    df = pd.DataFrame({"ticker": ["AAPL"], "open": [1.0]})
    with pytest.raises(ValueError, match="missing required column"):
        _conform(df, DAILY_BARS_COLUMNS)


def test_conform_reorders_and_casts_dtypes():
    df = pd.DataFrame(
        {
            "as_of": ["2024-01-05"],
            "volume": ["100"],
            "trade_date": ["2024-01-05"],
            "ticker": ["AAPL"],
            "open": ["1.0"],
            "high": ["2.0"],
            "low": ["0.5"],
            "close": ["1.5"],
            "dollar_volume": ["150"],
        }
    )
    out = _conform(df, DAILY_BARS_COLUMNS)
    assert list(out.columns) == DAILY_BARS_COLUMNS
    assert out["volume"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(out["trade_date"])


# --- polygon daily_bars -------------------------------------------------------


def test_polygon_daily_bars_parses_grouped_daily_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    fixture = json.loads(
        (FIXTURES / "polygon_grouped_daily_2024-01-05.json").read_text()
    )

    calls = []

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return fixture

    def _fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _FakeResponse()

    monkeypatch.setattr("top10.data.polygon.requests.get", _fake_get)

    source = PolygonSource()
    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))

    assert list(df.columns) == DAILY_BARS_COLUMNS
    assert set(df["ticker"]) == {"AAPL", "ZVZZT", "DLST"}
    # Delisted-looking ticker (DLST) must not be dropped -- P2 guardrail.
    assert "DLST" in set(df["ticker"])
    assert len(calls) == 1

    _url, params = calls[0]
    assert params["adjusted"] == "false"


def test_polygon_daily_bars_caches_and_avoids_second_network_call(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    fixture = json.loads(
        (FIXTURES / "polygon_grouped_daily_2024-01-05.json").read_text()
    )
    calls = []

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return fixture

    def _fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _FakeResponse()

    monkeypatch.setattr("top10.data.polygon.requests.get", _fake_get)

    source = PolygonSource()
    df1 = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    assert len(calls) == 1

    # Second adapter instance, same cache dir: must re-read from disk.
    source2 = PolygonSource()
    df2 = source2.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    assert len(calls) == 1

    pd.testing.assert_frame_equal(df1, df2)


# --- cache module --------------------------------------------------------------


def test_cached_call_writes_then_rereads_without_second_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)

    fetch_count = {"n": 0}

    def _fetch():
        fetch_count["n"] += 1
        return {"results": [{"a": 1}]}

    first = cached_call("vendor/ns", "key1", _fetch)
    second = cached_call("vendor/ns", "key1", _fetch)

    assert first == second == {"results": [{"a": 1}]}
    assert fetch_count["n"] == 1
    assert (tmp_path / "vendor" / "ns" / "key1.json").exists()


def test_cached_call_never_caches_empty_payload(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)

    fetch_count = {"n": 0}

    def _fetch():
        fetch_count["n"] += 1
        return {"results": []}

    cached_call("vendor/ns", "key2", _fetch)
    cached_call("vendor/ns", "key2", _fetch)

    assert fetch_count["n"] == 2
    assert not (tmp_path / "vendor" / "ns" / "key2.json").exists()


# --- databento --------------------------------------------------------------


def test_databento_unsupported_methods_raise_not_implemented(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    with pytest.raises(NotImplementedError, match="earnings"):
        source.earnings(dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    with pytest.raises(NotImplementedError, match="corporate-actions") as excinfo:
        source.corporate_actions(dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    # P2: splits/dividends/ticker-changes are still unimplemented, but
    # delistings are now inferrable -- the message must point at the
    # helpers that do it (see tests/test_databento_universe.py).
    assert "infer_delistings" in str(excinfo.value)

    with pytest.raises(NotImplementedError, match="listing-metadata"):
        source.ticker_meta(dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    with pytest.raises(NotImplementedError, match="short-interest"):
        source.short_interest(dt.date(2024, 1, 1), dt.date(2024, 1, 5))


# --- Defect 2: as_of on daily bars must be 16:00 ET, not midnight -----------


def test_polygon_daily_bars_as_of_is_16_00_not_midnight(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    fixture = json.loads(
        (FIXTURES / "polygon_grouped_daily_2024-01-05.json").read_text()
    )

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return fixture

    monkeypatch.setattr(
        "top10.data.polygon.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(),
    )

    source = PolygonSource()
    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))

    expected_as_of = pd.Timestamp("2024-01-05 16:00:00")
    assert (df["as_of"] == expected_as_of).all()
    # Explicitly not midnight -- a midnight as_of would satisfy
    # `as_of <= decision_time` for a T2 09:25 decision on the SAME day,
    # leaking the day's own close into a premarket decision.
    assert not (df["as_of"] == pd.Timestamp("2024-01-05")).any()


def test_databento_daily_bars_as_of_is_16_00_not_midnight(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource(venues=["XNAS.ITCH"])

    records = [
        {
            "ts_event": "2024-01-05T00:00:00Z",
            "instrument_id": 27,
            "open": 189.5,
            "high": 192.0,
            "low": 188.0,
            "close": 191.0,
            "volume": 1_000_000.0,
        }
    ]

    def _fake_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        return [] if schema == "definition" else records

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)

    class _FakeSymbology:
        def resolve(self, **kwargs):
            return {"result": {"27": [{"d0": kwargs["start_date"], "d1": kwargs["end_date"], "s": "AAPL"}]}}

    class _FakeMetadata:
        def get_dataset_condition(self, **kwargs):
            return []

    class _FakeClient:
        symbology = _FakeSymbology()
        metadata = _FakeMetadata()

    monkeypatch.setattr(source, "_get_client", lambda: _FakeClient())

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))

    as_of_naive = df["as_of"].dt.tz_localize(None) if df["as_of"].dt.tz is not None else df["as_of"]
    assert (as_of_naive == pd.Timestamp("2024-01-05 16:00:00")).all()
    assert df["ticker"].iloc[0] == "AAPL"


# --- Defect 2 / boundary agreement: the 09:25 bar must be EXCLUDED ----------


def _premarket_payload(minutes_and_ts: list[tuple[str, int]]) -> dict:
    return {
        "results": [
            {"t": ts_ms, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0, "v": 100.0, "n": 3}
            for _minute_label, ts_ms in minutes_and_ts
        ]
    }


def test_polygon_premarket_bars_excludes_09_25_bar(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    # 09:24 ET and 09:25 ET on 2024-01-05, as UTC epoch-ms (ET = UTC-5 in Jan).
    ts_09_24 = int(pd.Timestamp("2024-01-05 09:24", tz="America/New_York").timestamp() * 1000)
    ts_09_25 = int(pd.Timestamp("2024-01-05 09:25", tz="America/New_York").timestamp() * 1000)
    payload = _premarket_payload([("09:24", ts_09_24), ("09:25", ts_09_25)])

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(
        "top10.data.polygon.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(),
    )

    source = PolygonSource()
    df = source.premarket_bars(dt.date(2024, 1, 5), ["AAPL"])

    minutes = set(df["minute"].dt.strftime("%H:%M"))
    assert minutes == {"09:24"}
    assert "09:25" not in minutes


def test_databento_premarket_bars_excludes_09_25_bar(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    records = [
        {
            "ts_event": pd.Timestamp("2024-01-05 09:24", tz="America/New_York").tz_convert("UTC").isoformat(),
            "symbol": "AAPL",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 100.0,
            "count": 3,
        },
        {
            "ts_event": pd.Timestamp("2024-01-05 09:25", tz="America/New_York").tz_convert("UTC").isoformat(),
            "symbol": "AAPL",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 100.0,
            "count": 3,
        },
    ]
    monkeypatch.setattr(source, "_fetch_bars", lambda *a, **k: records)

    df = source.premarket_bars(dt.date(2024, 1, 5), ["AAPL"])

    minutes = set(df["minute"].dt.strftime("%H:%M"))
    assert minutes == {"09:24"}


# --- Defect 1: ticker_meta must be point-in-time, not active-only ----------


def _tickers_payload(results: list[dict]) -> dict:
    return {"results": results, "status": "OK"}


def test_polygon_ticker_meta_includes_delisted_names_earlier_in_history(monkeypatch, tmp_path):
    """P2: a ticker delisted mid-range (say 2019) must still appear in
    `ticker_meta` metadata usable to build an earlier (2017) universe."""
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    active_payload = _tickers_payload(
        [
            {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "type": "CS",
                "primary_exchange": "XNAS",
                "list_date": "1980-12-12",
                "delisted_utc": None,
            }
        ]
    )
    delisted_payload = _tickers_payload(
        [
            {
                "ticker": "DLST",
                "name": "Delisted Co",
                "type": "CS",
                "primary_exchange": "XNYS",
                "list_date": "2010-01-04",
                "delisted_utc": "2019-06-01",
            }
        ]
    )

    def _fake_get(url, params=None, timeout=None):
        class _FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                if params is not None and params.get("active") == "false":
                    return delisted_payload
                return active_payload

        return _FakeResponse()

    monkeypatch.setattr("top10.data.polygon.requests.get", _fake_get)

    source = PolygonSource()
    # `end` post-dates DLST's 2019 delisting; we're building a 2017 universe.
    df = source.ticker_meta(dt.date(2010, 1, 1), dt.date(2020, 1, 1))

    assert set(df["ticker"]) == {"AAPL", "DLST"}

    dlst = df[df["ticker"] == "DLST"].iloc[0]
    assert dlst["active_from"] == pd.Timestamp("2010-01-04")
    assert dlst["active_to"] == pd.Timestamp("2019-06-01")

    # The row must be usable (as_of < trade_date) for a 2017 trade_date --
    # i.e. NOT stamped at the query's range-end (2020-01-01).
    trade_date_2017 = pd.Timestamp("2017-03-01")
    assert dlst["as_of"] < trade_date_2017
    assert dlst["as_of"] != pd.Timestamp(dt.date(2020, 1, 1))


def test_polygon_ticker_meta_as_of_never_equals_range_end(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    payload = _tickers_payload(
        [
            {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "type": "CS",
                "primary_exchange": "XNAS",
                "list_date": "1980-12-12",
                "delisted_utc": None,
            }
        ]
    )
    empty_payload = _tickers_payload([])

    def _fake_get(url, params=None, timeout=None):
        class _FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                if params is not None and params.get("active") == "false":
                    return empty_payload
                return payload

        return _FakeResponse()

    monkeypatch.setattr("top10.data.polygon.requests.get", _fake_get)

    source = PolygonSource()
    end = dt.date(2024, 6, 1)
    df = source.ticker_meta(dt.date(2020, 1, 1), end)

    assert not df.empty
    assert (df["as_of"] != pd.Timestamp(end)).all()
    # A row usable for a 2021 trade_date must exist -- this is the exact
    # repro the audit called out ("universe rows with polygon-shaped meta
    # (as_of=end): 0").
    trade_date = pd.Timestamp("2021-01-04")
    usable = df[df["as_of"] < trade_date]
    assert not usable.empty


def test_polygon_ticker_meta_populates_market_cap_and_float_shares(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    payload = _tickers_payload(
        [
            {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "type": "CS",
                "primary_exchange": "XNAS",
                "list_date": "1980-12-12",
                "delisted_utc": None,
                "market_cap": 3_000_000_000_000.0,
                "weighted_shares_outstanding": 15_000_000_000.0,
            }
        ]
    )
    empty_payload = _tickers_payload([])

    def _fake_get(url, params=None, timeout=None):
        class _FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                if params is not None and params.get("active") == "false":
                    return empty_payload
                return payload

        return _FakeResponse()

    monkeypatch.setattr("top10.data.polygon.requests.get", _fake_get)

    source = PolygonSource()
    df = source.ticker_meta(dt.date(2020, 1, 1), dt.date(2024, 1, 1))

    assert "market_cap" in df.columns
    assert "float_shares" in df.columns
    row = df[df["ticker"] == "AAPL"].iloc[0]
    assert row["market_cap"] == 3_000_000_000_000.0
    assert row["float_shares"] == 15_000_000_000.0


# --- Defect 3: short_interest uses PUBLISH date, never settlement_date -----


def test_polygon_short_interest_uses_publish_date_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    payload = {
        "results": [
            {
                "ticker": "AAPL",
                "settlement_date": "2024-01-15",
                "publish_date": "2024-01-24",
                "short_interest": 100_000_000.0,
                "short_interest_pct_float": 0.8,
                "days_to_cover": 1.5,
            }
        ]
    }

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(
        "top10.data.polygon.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(),
    )

    source = PolygonSource()
    df = source.short_interest(dt.date(2024, 1, 1), dt.date(2024, 2, 1))

    row = df.iloc[0]
    assert row["settlement_date"] == pd.Timestamp("2024-01-15")
    # `as_of` must be the PUBLISH date, not the (earlier) settlement date.
    assert row["as_of"] == pd.Timestamp("2024-01-24")
    assert row["as_of"] != row["settlement_date"]


def test_polygon_short_interest_falls_back_to_conservative_lag_when_no_publish_date(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    payload = {
        "results": [
            {
                "ticker": "AAPL",
                "settlement_date": "2024-01-15",
                "short_interest": 100_000_000.0,
                "short_interest_pct_float": 0.8,
                "days_to_cover": 1.5,
            }
        ]
    }

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(
        "top10.data.polygon.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(),
    )

    source = PolygonSource()
    df = source.short_interest(dt.date(2024, 1, 1), dt.date(2024, 2, 1))

    row = df.iloc[0]
    settlement = pd.Timestamp("2024-01-15")
    # Never equal to (or before) settlement_date -- that would be the
    # exact look-ahead the audit flagged.
    assert row["as_of"] > settlement
    assert row["as_of"] == settlement + source._PUBLISH_LAG_DAYS


def test_databento_short_interest_raises_not_implemented(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    with pytest.raises(NotImplementedError, match="short-interest"):
        source.short_interest(dt.date(2024, 1, 1), dt.date(2024, 2, 1))


# --- Defect 4: date_is_revisable must survive to the consumer --------------


def test_polygon_earnings_unknown_announced_on_keeps_row_usable(monkeypatch, tmp_path):
    """When `announced_on` is unknown, the row must not self-erase: `as_of`
    must be populated (not NaT) and <= decision_time_t1(report_date), and
    `date_is_revisable` must remain True through `_conform`'s bool cast --
    this is what lets a PIT-filtering consumer (`as_of <= decision_time`)
    keep the row and read `date_is_revisable=True` off of it."""
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    payload = {
        "results": [
            {
                "ticker": "AAPL",
                "report_date": "2024-03-01",
                "session": "amc",
                "announced_on": None,
            }
        ]
    }

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(
        "top10.data.polygon.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(),
    )

    source = PolygonSource()
    df = source.earnings(dt.date(2024, 1, 1), dt.date(2024, 3, 31))

    row = df.iloc[0]
    assert row["date_is_revisable"] is True or row["date_is_revisable"] == True  # noqa: E712
    assert pd.notna(row["as_of"])

    # decision_time_t1(report_date) == report_date - 8h (16:00 ET the prior
    # day). The row must be usable at that boundary -- reproducing the B3
    # `LeakageError` the audit found (as_of == report_date used to fail
    # this exact check on every row with an unknown announcement date).
    report_date = pd.Timestamp("2024-03-01")
    decision_time_t1 = report_date - pd.Timedelta(hours=8)
    assert row["as_of"] <= decision_time_t1


def test_conform_bool_column_never_silently_coerces_missing_value():
    """P4 tripwire: `.astype("bool")` on an object column silently turns
    `None` -> False but `np.nan` -> True -- an ambiguous, direction-
    flipping coercion `_conform` must refuse rather than propagate."""
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "report_date": [pd.Timestamp("2024-01-01")],
            "session": ["amc"],
            "announced_on": [pd.NaT],
            "date_is_revisable": [float("nan")],
            "as_of": [pd.Timestamp("2024-01-01")],
        }
    )
    with pytest.raises(ValueError, match="date_is_revisable"):
        _conform(df, EARNINGS_COLUMNS)
