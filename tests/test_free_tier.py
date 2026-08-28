"""Offline tests for top10.data.free_tier. No network access, no API keys."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from top10.data.base import DAILY_BARS_COLUMNS, EARNINGS_COLUMNS, PREMARKET_BARS_COLUMNS
from top10.data.free_tier import AlpacaPremarket, FinnhubEarnings, MissingApiKey, TiingoDaily


# --- import / construction never requires a key -------------------------------


def test_free_tier_module_imports_and_constructs_with_no_keys(monkeypatch):
    for var in ("FINNHUB_API_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "TIINGO_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    finnhub = FinnhubEarnings()
    alpaca = AlpacaPremarket()
    tiingo = TiingoDaily()

    assert finnhub.name == "finnhub_earnings"
    assert alpaca.name == "alpaca_premarket"
    assert tiingo.name == "tiingo_daily"


def test_finnhub_earnings_raises_clearly_with_no_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    source = FinnhubEarnings()

    with pytest.raises(MissingApiKey, match="FINNHUB_API_KEY"):
        source.earnings(dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_alpaca_premarket_raises_clearly_with_no_key(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    source = AlpacaPremarket()

    with pytest.raises(MissingApiKey, match="APCA_API_KEY_ID"):
        source.premarket_bars(dt.date(2024, 1, 5), ["AAPL"])


def test_tiingo_daily_raises_clearly_with_no_key(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    source = TiingoDaily()

    with pytest.raises(MissingApiKey, match="TIINGO_API_KEY"):
        source.daily_bars(dt.date(2024, 1, 1), dt.date(2024, 1, 5), ["AAPL"])


# --- FinnhubEarnings ------------------------------------------------------------


def test_finnhub_probe_lookback_reports_real_answer_not_docs_claim(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    source = FinnhubEarnings()

    today = dt.date(2024, 6, 15)
    # Free-tier-like payload: only ~20 days of real history despite a
    # 10-year request window -- this is exactly the gap between Finnhub's
    # documented "back to 2003" claim and a real free-tier key's behavior.
    payload = {
        "earningsCalendar": [
            {"symbol": "AAPL", "date": "2024-05-28", "hour": "amc"},
            {"symbol": "MSFT", "date": "2024-06-10", "hour": "bmo"},
        ]
    }

    calls = []

    def _fake_get(url, params=None, timeout=None):
        calls.append((url, params))

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return payload

        return _Resp()

    monkeypatch.setattr("top10.data.free_tier.requests.get", _fake_get)

    result = source.probe_lookback(today=today)

    assert result["earliest_report_date"] == "2024-05-28"
    assert result["row_count"] == 2
    assert result["suspiciously_recent_lookback"] is True
    assert len(calls) == 1
    # Requested a wide (documented-depth) window even though the real
    # answer came back much shallower.
    _url, params = calls[0]
    assert params["from"] == (today - dt.timedelta(days=365 * 10)).isoformat()


def test_finnhub_earnings_conforms_to_column_contract(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    source = FinnhubEarnings()

    payload = {
        "earningsCalendar": [
            {"symbol": "AAPL", "date": "2024-03-01", "hour": "amc"},
        ]
    }

    monkeypatch.setattr(
        "top10.data.free_tier.requests.get",
        lambda url, params=None, timeout=None: type(
            "R", (), {"status_code": 200, "raise_for_status": lambda self: None, "json": lambda self: payload}
        )(),
    )

    df = source.earnings(dt.date(2024, 1, 1), dt.date(2024, 3, 31))

    assert list(df.columns) == EARNINGS_COLUMNS
    row = df.iloc[0]
    assert row["date_is_revisable"] == True  # noqa: E712
    assert pd.notna(row["as_of"])
    assert row["as_of"] < row["report_date"]


# --- AlpacaPremarket -------------------------------------------------------------


def test_alpaca_premarket_excludes_09_25_bar_and_conforms(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    source = AlpacaPremarket()

    ts_09_24 = pd.Timestamp("2024-01-05 09:24", tz="America/New_York").tz_convert("UTC").isoformat()
    ts_09_25 = pd.Timestamp("2024-01-05 09:25", tz="America/New_York").tz_convert("UTC").isoformat()
    payload = {
        "bars": [
            {"t": ts_09_24, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0, "v": 100.0, "n": 3},
            {"t": ts_09_25, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0, "v": 100.0, "n": 3},
        ]
    }

    monkeypatch.setattr(
        "top10.data.free_tier.requests.get",
        lambda url, headers=None, params=None, timeout=None: type(
            "R", (), {"status_code": 200, "raise_for_status": lambda self: None, "json": lambda self: payload}
        )(),
    )

    df = source.premarket_bars(dt.date(2024, 1, 5), ["AAPL"])

    assert list(df.columns) == PREMARKET_BARS_COLUMNS
    minutes = set(df["minute"].dt.strftime("%H:%M"))
    assert minutes == {"09:24"}
    assert "09:25" not in minutes


def test_alpaca_premarket_rejects_date_before_2016(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    source = AlpacaPremarket()

    with pytest.raises(ValueError, match="2016-01-01"):
        source.premarket_bars(dt.date(2015, 12, 1), ["AAPL"])


def test_alpaca_docstring_documents_iex_volume_caveat():
    assert "2.5%" in AlpacaPremarket.__doc__
    assert "IEX" in AlpacaPremarket.__doc__


# --- TiingoDaily -----------------------------------------------------------------


def test_tiingo_daily_conforms_to_column_contract(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "test-key")
    source = TiingoDaily()

    payload = [
        {"date": "2024-01-05T00:00:00.000Z", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1000.0},
    ]

    monkeypatch.setattr(
        "top10.data.free_tier.requests.get",
        lambda url, params=None, timeout=None: type(
            "R", (), {"status_code": 200, "raise_for_status": lambda self: None, "json": lambda self: payload}
        )(),
    )

    df = source.daily_bars(dt.date(2024, 1, 1), dt.date(2024, 1, 5), ["AAPL"])

    assert list(df.columns) == DAILY_BARS_COLUMNS
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["as_of"] == pd.Timestamp("2024-01-05 16:00:00")


def test_tiingo_docstring_documents_survivor_only_constraint():
    assert "SURVIVOR-ONLY" in TiingoDaily.__doc__
    assert "sole source" in TiingoDaily.__doc__
