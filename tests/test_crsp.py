"""Offline tests for top10/data/crsp.py. No network access, no WRDS
connection, no `wrds` package required."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from top10.data import get_source
from top10.data.base import (
    CORPORATE_ACTIONS_COLUMNS,
    DAILY_BARS_COLUMNS,
    TICKER_META_COLUMNS,
)
from top10.data.crsp import CRSPSource


# --- import / construction --------------------------------------------------


def test_crsp_source_imports_and_constructs_without_wrds_installed(monkeypatch):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    source = CRSPSource()
    assert source.name == "crsp"


def test_get_source_reads_crsp_vendor(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_VENDOR", "crsp")
    src = get_source()
    assert src.name == "crsp"
    assert isinstance(src, CRSPSource)


# --- daily_bars ---------------------------------------------------------------


def _daily_bars_raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_daily_bars_negative_prc_becomes_positive_and_flagged(monkeypatch):
    source = CRSPSource()
    raw = _daily_bars_raw(
        [
            {
                "permno": 10001.0,
                "date": pd.Timestamp("2024-01-05"),
                "prc": -50.25,
                "openprc": 49.0,
                "askhi": 51.0,
                "bidlo": 48.5,
                "vol": 100000.0,
                "cfacpr": 1.0,
                "cfacshr": 1.0,
                "ticker": "AAPL",
                "exchcd": 3,
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_daily_bars_raw", lambda start, end: raw)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))

    assert list(df.columns[: len(DAILY_BARS_COLUMNS)]) == DAILY_BARS_COLUMNS
    row = df.iloc[0]
    assert row["close"] == 50.25
    assert row["price_is_midpoint"] == True  # noqa: E712
    assert row["permno"] == 10001.0


def test_daily_bars_positive_prc_not_flagged_as_midpoint(monkeypatch):
    source = CRSPSource()
    raw = _daily_bars_raw(
        [
            {
                "permno": 10001.0,
                "date": pd.Timestamp("2024-01-05"),
                "prc": 50.25,
                "openprc": 49.0,
                "askhi": 51.0,
                "bidlo": 48.5,
                "vol": 100000.0,
                "cfacpr": 1.0,
                "cfacshr": 1.0,
                "ticker": "AAPL",
                "exchcd": 3,
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_daily_bars_raw", lambda start, end: raw)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    row = df.iloc[0]
    assert row["close"] == 50.25
    assert row["price_is_midpoint"] == False  # noqa: E712


def test_daily_bars_dollar_volume_uses_abs_price(monkeypatch):
    source = CRSPSource()
    raw = _daily_bars_raw(
        [
            {
                "permno": 10001.0,
                "date": pd.Timestamp("2024-01-05"),
                "prc": -10.0,
                "openprc": 10.0,
                "askhi": 10.5,
                "bidlo": 9.5,
                "vol": 1000.0,
                "cfacpr": 1.0,
                "cfacshr": 1.0,
                "ticker": "ZVZZT",
                "exchcd": 1,
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_daily_bars_raw", lambda start, end: raw)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    assert df.iloc[0]["dollar_volume"] == 10.0 * 1000.0


def test_daily_bars_does_not_apply_cfacpr_adjustment(monkeypatch):
    """P3: `cfacpr`/`cfacshr` must never be multiplied into price/volume --
    an adjustment factor != 1.0 must not change `close`/`open`/`volume`."""
    source = CRSPSource()
    raw = _daily_bars_raw(
        [
            {
                "permno": 10001.0,
                "date": pd.Timestamp("2024-01-05"),
                "prc": 100.0,
                "openprc": 99.0,
                "askhi": 101.0,
                "bidlo": 98.0,
                "vol": 5000.0,
                "cfacpr": 2.0,  # would halve price if (mis)applied
                "cfacshr": 2.0,
                "ticker": "AAPL",
                "exchcd": 3,
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_daily_bars_raw", lambda start, end: raw)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    row = df.iloc[0]
    assert row["close"] == 100.0
    assert row["open"] == 99.0
    assert row["volume"] == 5000.0


def test_daily_bars_as_of_is_16_00_not_midnight(monkeypatch):
    source = CRSPSource()
    raw = _daily_bars_raw(
        [
            {
                "permno": 10001.0,
                "date": pd.Timestamp("2024-01-05"),
                "prc": 100.0,
                "openprc": 99.0,
                "askhi": 101.0,
                "bidlo": 98.0,
                "vol": 5000.0,
                "cfacpr": 1.0,
                "cfacshr": 1.0,
                "ticker": "AAPL",
                "exchcd": 3,
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_daily_bars_raw", lambda start, end: raw)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))
    assert (df["as_of"] == pd.Timestamp("2024-01-05 16:00:00")).all()
    assert not (df["as_of"] == pd.Timestamp("2024-01-05")).any()


def test_daily_bars_includes_delisted_permnos_before_delisting(monkeypatch):
    """P2: a PERMNO delisted later must still appear for dates before its
    delisting -- the entire reason this adapter exists."""
    source = CRSPSource()
    raw = _daily_bars_raw(
        [
            {
                "permno": 99999.0,
                "date": pd.Timestamp("2015-03-10"),
                "prc": 12.0,
                "openprc": 11.5,
                "askhi": 12.5,
                "bidlo": 11.0,
                "vol": 2000.0,
                "cfacpr": 1.0,
                "cfacshr": 1.0,
                "ticker": "DLST",
                "exchcd": 1,
            },
            {
                "permno": 10001.0,
                "date": pd.Timestamp("2015-03-10"),
                "prc": 100.0,
                "openprc": 99.0,
                "askhi": 101.0,
                "bidlo": 98.0,
                "vol": 5000.0,
                "cfacpr": 1.0,
                "cfacshr": 1.0,
                "ticker": "AAPL",
                "exchcd": 3,
            },
        ]
    )
    monkeypatch.setattr(source, "_fetch_daily_bars_raw", lambda start, end: raw)

    # DLST is delisted well after this range; the range only reaches up
    # to a date BEFORE the delisting, and the row must still be present.
    df = source.daily_bars(dt.date(2015, 3, 1), dt.date(2015, 3, 31))
    assert 99999.0 in set(df["permno"])
    assert "DLST" in set(df["ticker"])


# --- corporate_actions ---------------------------------------------------------


def test_corporate_actions_splits_use_facpr_ratio(monkeypatch):
    source = CRSPSource()
    dist = pd.DataFrame(
        [
            {
                "permno": 10001.0,
                "dclrdt": pd.Timestamp("2024-01-01"),
                "exdt": pd.Timestamp("2024-01-10"),
                "distcd": 5523,
                "divamt": None,
                "facshr": 1.0,
                "facpr": 1.0,  # 2-for-1 split
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_splits_dividends_raw", lambda start, end: dist)
    monkeypatch.setattr(source, "_fetch_ticker_changes_raw", lambda start, end: pd.DataFrame())
    monkeypatch.setattr(source, "_fetch_delistings_raw", lambda start, end: pd.DataFrame())

    df = source.corporate_actions(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert list(df.columns[: len(CORPORATE_ACTIONS_COLUMNS)]) == CORPORATE_ACTIONS_COLUMNS
    row = df.iloc[0]
    assert row["action_type"] == "split"
    assert row["ratio"] == 2.0
    assert row["as_of"] == pd.Timestamp("2024-01-01")


def test_corporate_actions_delistings_included_and_marked(monkeypatch):
    """P2: delisting events must surface via `corporate_actions`."""
    source = CRSPSource()
    monkeypatch.setattr(source, "_fetch_splits_dividends_raw", lambda start, end: pd.DataFrame())
    monkeypatch.setattr(source, "_fetch_ticker_changes_raw", lambda start, end: pd.DataFrame())
    delist = pd.DataFrame(
        [
            {
                "permno": 99999.0,
                "dlstdt": pd.Timestamp("2015-06-01"),
                "dlstcd": 560,
                "dlret": -0.35,
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_delistings_raw", lambda start, end: delist)

    df = source.corporate_actions(dt.date(2015, 5, 1), dt.date(2015, 6, 30))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["action_type"] == "delisting"
    assert row["ex_date"] == pd.Timestamp("2015-06-01")
    assert row["as_of"] == pd.Timestamp("2015-06-01")
    assert row["permno"] == 99999.0


def test_corporate_actions_ticker_change_detected_from_dsenames(monkeypatch):
    source = CRSPSource()
    monkeypatch.setattr(source, "_fetch_splits_dividends_raw", lambda start, end: pd.DataFrame())
    monkeypatch.setattr(source, "_fetch_delistings_raw", lambda start, end: pd.DataFrame())
    changes = pd.DataFrame(
        [
            {
                "permno": 20000.0,
                "ticker": "OLDT",
                "namedt": pd.Timestamp("2010-01-01"),
                "nameendt": pd.Timestamp("2020-05-14"),
            },
            {
                "permno": 20000.0,
                "ticker": "NEWT",
                "namedt": pd.Timestamp("2020-05-15"),
                "nameendt": None,
            },
        ]
    )
    monkeypatch.setattr(source, "_fetch_ticker_changes_raw", lambda start, end: changes)

    df = source.corporate_actions(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["action_type"] == "ticker_change"
    assert row["ticker"] == "OLDT"
    assert row["new_ticker"] == "NEWT"
    assert row["ex_date"] == pd.Timestamp("2020-05-15")


# --- ticker_meta ---------------------------------------------------------------


def _ticker_meta_raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_ticker_meta_shrcd_common_stock_and_adr_mapping(monkeypatch):
    source = CRSPSource()
    raw = _ticker_meta_raw(
        [
            {
                "permno": 1.0,
                "ticker": "AAPL",
                "comnam": "Apple Inc",
                "namedt": pd.Timestamp("1980-12-12"),
                "nameendt": None,
                "exchcd": 3,
                "shrcd": 11,
                "prc": 100.0,
                "shrout": 1000.0,
            },
            {
                "permno": 2.0,
                "ticker": "BABA",
                "comnam": "Alibaba ADR",
                "namedt": pd.Timestamp("2014-09-19"),
                "nameendt": None,
                "exchcd": 3,
                "shrcd": 30,
                "prc": 80.0,
                "shrout": 500.0,
            },
            {
                "permno": 3.0,
                "ticker": "WEIRD",
                "comnam": "Something Else",
                "namedt": pd.Timestamp("2005-01-01"),
                "nameendt": None,
                "exchcd": 1,
                "shrcd": 73,  # not in mapping
                "prc": 5.0,
                "shrout": 200.0,
            },
        ]
    )
    monkeypatch.setattr(source, "_fetch_ticker_meta_raw", lambda start, end: raw)

    df = source.ticker_meta(dt.date(1980, 1, 1), dt.date(2024, 1, 1))
    assert list(df.columns[: len(TICKER_META_COLUMNS)]) == TICKER_META_COLUMNS

    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["security_type"] == "CS"

    baba = df[df["ticker"] == "BABA"].iloc[0]
    assert baba["security_type"] == "ADR"

    weird = df[df["ticker"] == "WEIRD"].iloc[0]
    assert weird["security_type"] == "OTHER"


def test_ticker_meta_market_cap_uses_shrout_in_thousands(monkeypatch):
    source = CRSPSource()
    raw = _ticker_meta_raw(
        [
            {
                "permno": 1.0,
                "ticker": "AAPL",
                "comnam": "Apple Inc",
                "namedt": pd.Timestamp("1980-12-12"),
                "nameendt": None,
                "exchcd": 3,
                "shrcd": 11,
                "prc": -100.0,  # abs() must be used for market cap too
                "shrout": 1000.0,
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_ticker_meta_raw", lambda start, end: raw)

    df = source.ticker_meta(dt.date(1980, 1, 1), dt.date(2024, 1, 1))
    row = df.iloc[0]
    # shrout is in THOUSANDS: market_cap = abs(prc) * shrout * 1000
    assert row["market_cap"] == 100.0 * 1000.0 * 1000.0
    assert pd.isna(row["float_shares"])


def test_ticker_meta_as_of_pinned_to_active_from(monkeypatch):
    source = CRSPSource()
    raw = _ticker_meta_raw(
        [
            {
                "permno": 1.0,
                "ticker": "AAPL",
                "comnam": "Apple Inc",
                "namedt": pd.Timestamp("1980-12-12"),
                "nameendt": None,
                "exchcd": 3,
                "shrcd": 11,
                "prc": 100.0,
                "shrout": 1000.0,
            }
        ]
    )
    monkeypatch.setattr(source, "_fetch_ticker_meta_raw", lambda start, end: raw)

    end = dt.date(2024, 6, 1)
    df = source.ticker_meta(dt.date(1980, 1, 1), end)
    row = df.iloc[0]
    assert row["as_of"] == pd.Timestamp("1980-12-12")
    assert row["as_of"] != pd.Timestamp(end)


def test_ticker_meta_exchcd_mapping(monkeypatch):
    source = CRSPSource()
    raw = _ticker_meta_raw(
        [
            {
                "permno": 1.0,
                "ticker": "A",
                "comnam": "NYSE Co",
                "namedt": pd.Timestamp("2000-01-01"),
                "nameendt": None,
                "exchcd": 1,
                "shrcd": 11,
                "prc": 10.0,
                "shrout": 100.0,
            },
            {
                "permno": 2.0,
                "ticker": "B",
                "comnam": "AMEX Co",
                "namedt": pd.Timestamp("2000-01-01"),
                "nameendt": None,
                "exchcd": 2,
                "shrcd": 11,
                "prc": 10.0,
                "shrout": 100.0,
            },
            {
                "permno": 3.0,
                "ticker": "C",
                "comnam": "NASDAQ Co",
                "namedt": pd.Timestamp("2000-01-01"),
                "nameendt": None,
                "exchcd": 3,
                "shrcd": 11,
                "prc": 10.0,
                "shrout": 100.0,
            },
        ]
    )
    monkeypatch.setattr(source, "_fetch_ticker_meta_raw", lambda start, end: raw)

    df = source.ticker_meta(dt.date(2000, 1, 1), dt.date(2024, 1, 1))
    exch = dict(zip(df["ticker"], df["exchange"]))
    assert exch == {"A": "XNYS", "B": "XASE", "C": "XNAS"}


# --- unsupported methods -------------------------------------------------------


def test_crsp_unsupported_methods_raise_not_implemented_with_actionable_messages():
    source = CRSPSource()

    with pytest.raises(NotImplementedError, match="Finnhub"):
        source.earnings(dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    with pytest.raises(NotImplementedError, match="Alpaca|Databento"):
        source.premarket_bars(dt.date(2024, 1, 1), ["AAPL"])

    with pytest.raises(NotImplementedError, match="Polygon"):
        source.short_interest(dt.date(2024, 1, 1), dt.date(2024, 2, 1))


def test_crsp_premarket_bars_error_states_t1_can_run_t2_cannot():
    source = CRSPSource()
    with pytest.raises(NotImplementedError, match="T1"):
        source.premarket_bars(dt.date(2024, 1, 1), ["AAPL"])
