"""Offline tests for top10/data/composite.py. No network access, no API keys."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from top10.data import get_source
from top10.data.base import (
    CORPORATE_ACTIONS_COLUMNS,
    DAILY_BARS_COLUMNS,
    EARNINGS_COLUMNS,
)
from top10.data.composite import ROUTING, CapabilityUnavailable, CompositeSource, describe_routing
from top10.data.databento import DatabentoSource
from top10.data.free_tier import FinnhubEarnings
from top10.data.polygon import PolygonSource


def _daily_bars_df(ticker: str, trade_date: str, close: float = 10.0) -> pd.DataFrame:
    ts = pd.Timestamp(trade_date)
    return pd.DataFrame(
        [
            {
                "trade_date": ts,
                "ticker": ticker,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
                "dollar_volume": close * 1000.0,
                "as_of": ts + pd.Timedelta(hours=16),
            }
        ],
        columns=DAILY_BARS_COLUMNS,
    )


def _corporate_actions_df(ticker: str, ex_date: str, ratio: float = 0.05) -> pd.DataFrame:
    ts = pd.Timestamp(ex_date)
    return pd.DataFrame(
        [
            {
                "ex_date": ts,
                "ticker": ticker,
                "action_type": "reverse_split",
                "ratio": ratio,
                "cash_amount": None,
                "new_ticker": None,
                "as_of": ts,
            }
        ],
        columns=CORPORATE_ACTIONS_COLUMNS,
    )


def _earnings_df(ticker: str, report_date: str) -> pd.DataFrame:
    ts = pd.Timestamp(report_date)
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "report_date": ts,
                "session": "amc",
                "announced_on": pd.NaT,
                "date_is_revisable": True,
                "as_of": ts - pd.Timedelta(days=1),
            }
        ],
        columns=EARNINGS_COLUMNS,
    )


@pytest.fixture(autouse=True)
def _cache_to_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.setattr("top10.data.symbology.DATA_RAW", tmp_path)


# --- routing table -----------------------------------------------------------


def test_routing_table_names_expected_vendor_per_capability():
    assert ROUTING["daily_bars"]["vendor"] == "databento"
    assert ROUTING["premarket_bars"]["vendor"] == "databento"
    assert ROUTING["corporate_actions"]["vendor"] == "polygon"
    assert ROUTING["ticker_meta"]["vendor"] == "polygon"
    assert ROUTING["earnings"]["vendor"] == "finnhub"
    assert ROUTING["short_interest"]["vendor"] == "polygon"


def test_describe_routing_is_a_readable_table_mentioning_every_capability():
    table = describe_routing()
    assert isinstance(table, str)
    for capability in ROUTING:
        assert capability in table
    for info in ROUTING.values():
        assert info["vendor"] in table


def test_composite_registered_as_vendor(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    src = get_source("composite")
    assert src.name == "composite"


def test_daily_bars_dispatches_to_databento_delegate(monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    calls = []

    def _fake_daily_bars(self, start, end, **kwargs):
        calls.append((start, end))
        return _daily_bars_df("AAPL", "2024-01-05")

    monkeypatch.setattr(DatabentoSource, "daily_bars", _fake_daily_bars)

    source = CompositeSource()
    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))

    assert len(calls) == 1
    assert set(df["ticker"]) == {"AAPL"}


def test_corporate_actions_dispatches_to_polygon_delegate(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    calls = []

    def _fake_corporate_actions(self, start, end):
        calls.append((start, end))
        return _corporate_actions_df("AAPL", "2024-01-05")

    monkeypatch.setattr(PolygonSource, "corporate_actions", _fake_corporate_actions)

    source = CompositeSource()
    df = source.corporate_actions(dt.date(2024, 1, 1), dt.date(2024, 1, 31))

    assert len(calls) == 1
    assert set(df["ticker"]) == {"AAPL"}


def test_earnings_dispatches_to_finnhub_delegate(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    calls = []

    def _fake_earnings(self, start, end):
        calls.append((start, end))
        return _earnings_df("AAPL", "2024-03-01")

    monkeypatch.setattr(FinnhubEarnings, "earnings", _fake_earnings)

    source = CompositeSource()
    df = source.earnings(dt.date(2024, 1, 1), dt.date(2024, 3, 31))

    assert len(calls) == 1
    assert set(df["ticker"]) == {"AAPL"}


# --- missing key breaks only the routed capability --------------------------


def test_missing_finnhub_key_breaks_only_earnings_not_daily_bars(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")

    monkeypatch.setattr(
        DatabentoSource, "daily_bars", lambda self, start, end, **kw: _daily_bars_df("AAPL", "2024-01-05")
    )

    source = CompositeSource()

    with pytest.raises(CapabilityUnavailable, match="FINNHUB_API_KEY"):
        source.earnings(dt.date(2024, 1, 1), dt.date(2024, 3, 31))

    # daily_bars is unaffected -- a missing Finnhub key must not break it.
    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    assert set(df["ticker"]) == {"AAPL"}


# --- unconfigured / unimplemented capability raises, never returns empty ----


def test_unconfigured_corporate_actions_delegate_raises_not_empty(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    source = CompositeSource()
    with pytest.raises(CapabilityUnavailable, match="corporate_actions"):
        source.corporate_actions(dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_delegate_not_implemented_error_is_wrapped_not_swallowed(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def _raise_not_implemented(self, start, end):
        raise NotImplementedError("Polygon short-interest unavailable for this plan")

    monkeypatch.setattr(PolygonSource, "short_interest", _raise_not_implemented)

    source = CompositeSource()
    with pytest.raises(CapabilityUnavailable, match="short_interest"):
        source.short_interest(dt.date(2024, 1, 1), dt.date(2024, 1, 31))


# --- cross-vendor ticker alignment: reused ticker must not get the wrong split --


def test_reused_ticker_does_not_receive_the_other_companys_split(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")

    # "AAA" was company X (instrument_id 100) through 2019-12-31, then
    # reassigned to unrelated company Y (instrument_id 200) starting 2020.
    def _fake_corporate_actions(self, start, end):
        old_co_split = _corporate_actions_df("AAA", "2019-06-01", ratio=0.05)
        new_co_split = _corporate_actions_df("AAA", "2021-06-01", ratio=2.0)
        return pd.concat([old_co_split, new_co_split], ignore_index=True)

    monkeypatch.setattr(PolygonSource, "corporate_actions", _fake_corporate_actions)

    source = CompositeSource()
    resolver = source._get_resolver()

    # Resolution is now PER TRADING DAY: Databento reassigns equity
    # instrument_ids daily, so a persisted range-wide interval map is
    # wrong (it produced a +8702% median top-10 gainer on live data).
    # Stub the public single-day resolver rather than any internal state.
    def _resolve_at(symbol, date, *, client=None):
        if symbol != "AAA":
            return None
        return "100" if pd.Timestamp(date) < pd.Timestamp("2020-01-01") else "200"

    resolver.resolve_at = _resolve_at

    df = source.corporate_actions(dt.date(2019, 1, 1), dt.date(2022, 1, 1))

    old_row = df[df["ex_date"] == pd.Timestamp("2019-06-01")].iloc[0]
    new_row = df[df["ex_date"] == pd.Timestamp("2021-06-01")].iloc[0]

    assert old_row["instrument_id"] == "100"
    assert new_row["instrument_id"] == "200"
    assert old_row["instrument_id"] != new_row["instrument_id"]
    assert old_row["ratio"] == 0.05
    assert new_row["ratio"] == 2.0


def test_alignment_report_surfaces_unaligned_tickers_without_dropping_rows(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")

    monkeypatch.setattr(
        PolygonSource,
        "corporate_actions",
        lambda self, start, end: _corporate_actions_df("ZZZZ", "2024-01-05"),
    )

    source = CompositeSource()
    # No symbology data at all -- "ZZZZ" cannot be resolved.
    report = source.alignment_report(dt.date(2024, 1, 1), dt.date(2024, 1, 31))

    assert report["total_rows"] == 1
    assert "ZZZZ" in report["unaligned_tickers"]
    assert report["unaligned_ticker_count"] == 1

    # The row itself must still be present -- never silently dropped.
    df = source.corporate_actions(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert set(df["ticker"]) == {"ZZZZ"}


# --- fetch-once caching: second identical range makes zero network calls ---


def test_second_identical_daily_bars_request_makes_zero_network_calls(monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    calls = []

    def _fake_daily_bars(self, start, end, **kwargs):
        calls.append((start, end))
        return _daily_bars_df("AAPL", "2024-01-05")

    monkeypatch.setattr(DatabentoSource, "daily_bars", _fake_daily_bars)

    source = CompositeSource()
    df1 = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    assert len(calls) == 1

    df2 = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    assert len(calls) == 1  # no second network/delegate call

    pd.testing.assert_frame_equal(
        df1[DAILY_BARS_COLUMNS].reset_index(drop=True),
        df2[DAILY_BARS_COLUMNS].reset_index(drop=True),
    )


def test_second_identical_corporate_actions_request_makes_zero_network_calls(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    calls = []

    def _fake_corporate_actions(self, start, end):
        calls.append((start, end))
        return _corporate_actions_df("AAPL", "2024-01-05")

    monkeypatch.setattr(PolygonSource, "corporate_actions", _fake_corporate_actions)

    source = CompositeSource()
    source.corporate_actions(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert len(calls) == 1

    # Even a fresh CompositeSource instance re-reads the same disk cache.
    source2 = CompositeSource()
    source2.corporate_actions(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert len(calls) == 1


def test_second_identical_earnings_request_makes_zero_network_calls(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    calls = []

    def _fake_earnings(self, start, end):
        calls.append((start, end))
        return _earnings_df("AAPL", "2024-03-01")

    monkeypatch.setattr(FinnhubEarnings, "earnings", _fake_earnings)

    source = CompositeSource()
    source.earnings(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    source.earnings(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    assert len(calls) == 1
