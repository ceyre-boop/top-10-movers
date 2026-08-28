from __future__ import annotations

import datetime as dt
import json
import sys
import types

import pytest

from top10.collect import rh_movers


# --- helpers -----------------------------------------------------------------


def _install_fake_robin_stocks(monkeypatch, *, movers=None, sp500_movers=None, raises=None):
    """Install a fake `robin_stocks.robinhood.markets` module in sys.modules
    so the lazy import in rh_movers picks it up without the real SDK being
    installed.
    """

    def get_top_movers(*args):
        # get_top_movers(direction) for the S&P 500 path, get_top_movers()
        # (no args) for the true top-movers path.
        if raises is not None:
            raise raises
        return movers

    def get_top_movers_sp500(direction):
        return sp500_movers

    fake_markets_module = types.ModuleType("robin_stocks.robinhood.markets")
    fake_markets_module.get_top_movers = get_top_movers
    fake_markets_module.get_top_movers_sp500 = get_top_movers_sp500

    fake_robinhood_module = types.ModuleType("robin_stocks.robinhood")
    fake_robinhood_module.markets = fake_markets_module

    monkeypatch.setitem(sys.modules, "robin_stocks", types.ModuleType("robin_stocks"))
    monkeypatch.setitem(sys.modules, "robin_stocks.robinhood", fake_robinhood_module)
    monkeypatch.setitem(sys.modules, "robin_stocks.robinhood.markets", fake_markets_module)


_RS_SP500_PAYLOAD = [
    {"symbol": "AAA", "price_movement": {"market_hours_last_movement_pct": "10.0"}},
    {"symbol": "BBB", "price_movement": {"market_hours_last_movement_pct": "9.0"}},
    {"symbol": "CCC", "price_movement": {"market_hours_last_movement_pct": "8.0"}},
]

_HTTPS_SP500_PAYLOAD = {
    "count": 3,
    "next": None,
    "previous": None,
    "results": [
        {"symbol": "XXX"},
        {"symbol": "YYY"},
        {"symbol": "ZZZ"},
    ],
}

_RS_TOP_MOVERS_PAYLOAD = [
    {"symbol": "SML1"},
    {"symbol": "SML2"},
    {"symbol": "SML3"},
]

_TOP_MOVERS_TAG_PAYLOAD = {
    "name": "Top Movers",
    "slug": "top-movers",
    "instruments": [
        "https://api.robinhood.com/instruments/11111111-1111-1111-1111-111111111111/",
        "https://api.robinhood.com/instruments/22222222-2222-2222-2222-222222222222/",
        "https://api.robinhood.com/instruments/33333333-3333-3333-3333-333333333333/",
    ],
}

_INSTRUMENT_SYMBOLS = {
    "11111111-1111-1111-1111-111111111111": "MOV1",
    "22222222-2222-2222-2222-222222222222": "MOV2",
    "33333333-3333-3333-3333-333333333333": "MOV3",
}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _requests_get_router(*, sp500_payload=None, tag_payload=None, instrument_symbols=None):
    """A `requests.get` stand-in that routes based on URL: the sp500
    endpoint, the top-movers tag endpoint, or an /instruments/<id>/ URL."""
    instrument_symbols = instrument_symbols or {}

    def _get(url, params=None, headers=None, timeout=None):
        if "movers/sp500" in url:
            return _FakeResponse(sp500_payload)
        if "tags/tag/top-movers" in url:
            return _FakeResponse(tag_payload)
        if "/instruments/" in url:
            inst_id = url.rstrip("/").rsplit("/", 1)[-1]
            symbol = instrument_symbols.get(inst_id)
            return _FakeResponse({"id": inst_id, "symbol": symbol})
        raise AssertionError(f"unexpected URL requested: {url}")

    return _get


# --- fetch_sp500_movers ----------------------------------------------------


def test_fetch_sp500_uses_robin_stocks_path_when_available(monkeypatch):
    _install_fake_robin_stocks(monkeypatch, movers=_RS_SP500_PAYLOAD)

    def _boom(*a, **k):
        raise AssertionError("https fallback should not be called when robin_stocks succeeds")

    monkeypatch.setattr(rh_movers.requests, "get", _boom)

    result = rh_movers.fetch_sp500_movers(direction="up")
    assert result["_source"] == "robinhood_sp500"
    assert result["_fetch_path"] == "robin_stocks"
    assert result["payload"] == _RS_SP500_PAYLOAD


def test_fetch_sp500_falls_back_to_https_when_robin_stocks_raises(monkeypatch):
    _install_fake_robin_stocks(monkeypatch, raises=RuntimeError("boom"))
    monkeypatch.setattr(
        rh_movers.requests, "get", _requests_get_router(sp500_payload=_HTTPS_SP500_PAYLOAD)
    )

    result = rh_movers.fetch_sp500_movers(direction="up")
    assert result["_source"] == "robinhood_sp500"
    assert result["_fetch_path"] == "https"
    assert result["payload"] == _HTTPS_SP500_PAYLOAD


def test_fetch_sp500_raises_when_both_paths_fail(monkeypatch):
    _install_fake_robin_stocks(monkeypatch, raises=RuntimeError("sdk down"))

    def _fail(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(rh_movers.requests, "get", _fail)

    with pytest.raises(RuntimeError):
        rh_movers.fetch_sp500_movers(direction="up")


# --- fetch_top_movers -------------------------------------------------------


def test_fetch_top_movers_uses_robin_stocks_path_when_available(monkeypatch):
    _install_fake_robin_stocks(monkeypatch, movers=_RS_TOP_MOVERS_PAYLOAD)

    def _boom(*a, **k):
        raise AssertionError("https fallback should not be called when robin_stocks succeeds")

    monkeypatch.setattr(rh_movers.requests, "get", _boom)

    result = rh_movers.fetch_top_movers()
    assert result["_source"] == "robinhood_top_movers"
    assert result["_fetch_path"] == "robin_stocks"
    assert result["symbols"] == ["SML1", "SML2", "SML3"]


def test_fetch_top_movers_falls_back_to_https_and_resolves_instruments(monkeypatch, tmp_path):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, raises=RuntimeError("sdk down"))
    monkeypatch.setattr(
        rh_movers.requests,
        "get",
        _requests_get_router(tag_payload=_TOP_MOVERS_TAG_PAYLOAD, instrument_symbols=_INSTRUMENT_SYMBOLS),
    )

    result = rh_movers.fetch_top_movers()
    assert result["_source"] == "robinhood_top_movers"
    assert result["_fetch_path"] == "https"
    assert result["payload"] == _TOP_MOVERS_TAG_PAYLOAD
    # Order preserved from the tag's instrument list.
    assert result["symbols"] == ["MOV1", "MOV2", "MOV3"]


def test_fetch_top_movers_raises_when_both_paths_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, raises=RuntimeError("sdk down"))

    def _fail(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(rh_movers.requests, "get", _fail)

    with pytest.raises(RuntimeError):
        rh_movers.fetch_top_movers()


def test_no_top_movers_direction_variant_endpoints_exist():
    # Regression guard for the live investigation finding: there is a
    # single combined "Top Movers" tag, no up/down split. Anything that
    # tries to hit a direction-specific top-movers URL is wrong.
    assert "direction" not in rh_movers._TOP_MOVERS_TAG_URL


# --- instrument URL resolution + caching -----------------------------------


def test_resolve_instrument_symbols_preserves_order(monkeypatch, tmp_path):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    monkeypatch.setattr(
        rh_movers.requests,
        "get",
        _requests_get_router(instrument_symbols=_INSTRUMENT_SYMBOLS),
    )

    urls = [
        "https://api.robinhood.com/instruments/33333333-3333-3333-3333-333333333333/",
        "https://api.robinhood.com/instruments/11111111-1111-1111-1111-111111111111/",
        "https://api.robinhood.com/instruments/22222222-2222-2222-2222-222222222222/",
    ]
    symbols = rh_movers._resolve_instrument_symbols(urls)
    assert symbols == ["MOV3", "MOV1", "MOV2"]


def test_resolve_instrument_symbols_caches_and_avoids_refetch(monkeypatch, tmp_path):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    call_count = {"n": 0}
    router = _requests_get_router(instrument_symbols=_INSTRUMENT_SYMBOLS)

    def _counting_get(*a, **k):
        call_count["n"] += 1
        return router(*a, **k)

    monkeypatch.setattr(rh_movers.requests, "get", _counting_get)

    urls = ["https://api.robinhood.com/instruments/11111111-1111-1111-1111-111111111111/"]
    first = rh_movers._resolve_instrument_symbols(urls)
    assert first == ["MOV1"]
    assert call_count["n"] == 1

    cache_path = tmp_path / "rh_movers" / "instrument_symbol_cache.json"
    assert cache_path.exists()
    assert json.loads(cache_path.read_text()) == {"11111111-1111-1111-1111-111111111111": "MOV1"}

    second = rh_movers._resolve_instrument_symbols(urls)
    assert second == ["MOV1"]
    # No new HTTP call -- served entirely from the on-disk cache.
    assert call_count["n"] == 1


# --- parse_sp500_tickers / parse_top_movers_tickers -------------------------


def test_parse_sp500_tickers_preserves_order_robin_stocks_shape():
    envelope = {"sp500": {"_source": "robinhood_sp500", "payload": _RS_SP500_PAYLOAD}}
    assert rh_movers.parse_sp500_tickers(envelope) == ["AAA", "BBB", "CCC"]


def test_parse_sp500_tickers_preserves_order_https_shape():
    envelope = {"sp500": {"_source": "robinhood_sp500", "payload": _HTTPS_SP500_PAYLOAD}}
    assert rh_movers.parse_sp500_tickers(envelope) == ["XXX", "YYY", "ZZZ"]


def test_parse_top_movers_tickers_reads_resolved_symbols_in_order():
    envelope = {
        "top_movers_available": True,
        "top_movers": {
            "_source": "robinhood_top_movers",
            "payload": _TOP_MOVERS_TAG_PAYLOAD,
            "symbols": ["MOV1", "MOV2", "MOV3"],
        },
    }
    assert rh_movers.parse_top_movers_tickers(envelope) == ["MOV1", "MOV2", "MOV3"]


def test_parse_top_movers_tickers_empty_when_unavailable():
    envelope = {
        "top_movers_available": False,
        "top_movers": {"_source": "robinhood_top_movers", "payload": None, "symbols": None},
    }
    assert rh_movers.parse_top_movers_tickers(envelope) == []


# --- capture ----------------------------------------------------------------


def test_capture_writes_both_feeds_in_separated_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, movers=_RS_SP500_PAYLOAD)
    # robin_stocks path is used for both feeds here since get_top_movers()
    # with no args also returns _RS_SP500_PAYLOAD-shaped data via the fake;
    # to distinguish, drive the https path for top_movers explicitly by
    # making robin_stocks raise for the no-arg call and use https+resolve.

    def get_top_movers(*args):
        if len(args) == 0:
            raise RuntimeError("no-op for this test's robin_stocks top_movers call")
        return _RS_SP500_PAYLOAD

    fake_markets = types.ModuleType("robin_stocks.robinhood.markets")
    fake_markets.get_top_movers = get_top_movers
    fake_markets.get_top_movers_sp500 = lambda direction: None
    fake_robinhood = types.ModuleType("robin_stocks.robinhood")
    fake_robinhood.markets = fake_markets
    monkeypatch.setitem(sys.modules, "robin_stocks", types.ModuleType("robin_stocks"))
    monkeypatch.setitem(sys.modules, "robin_stocks.robinhood", fake_robinhood)
    monkeypatch.setitem(sys.modules, "robin_stocks.robinhood.markets", fake_markets)

    monkeypatch.setattr(
        rh_movers.requests,
        "get",
        _requests_get_router(tag_payload=_TOP_MOVERS_TAG_PAYLOAD, instrument_symbols=_INSTRUMENT_SYMBOLS),
    )

    trade_date = dt.date(2026, 8, 27)
    path = rh_movers.capture(date=trade_date, direction="up")
    envelope = json.loads(path.read_text())

    assert envelope["sp500"]["available"] is True
    assert envelope["sp500"]["_source"] == "robinhood_sp500"
    assert envelope["sp500"]["payload"] == _RS_SP500_PAYLOAD

    assert envelope["top_movers"]["available"] is True
    assert envelope["top_movers"]["_source"] == "robinhood_top_movers"
    assert envelope["top_movers"]["symbols"] == ["MOV1", "MOV2", "MOV3"]
    assert envelope["top_movers_available"] is True

    # The two feeds must never share tickers/keys that could be conflated.
    assert set(envelope["sp500"]["payload"][0].keys()) != set()
    assert envelope["top_movers"]["symbols"] != rh_movers.parse_sp500_tickers(envelope)


def test_capture_succeeds_sp500_only_with_loud_warning_when_top_movers_unavailable(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, raises=RuntimeError("sdk down"))

    def _get(url, params=None, headers=None, timeout=None):
        if "movers/sp500" in url:
            return _FakeResponse(_HTTPS_SP500_PAYLOAD)
        if "tags/tag/top-movers" in url:
            raise RuntimeError("tag endpoint down")
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(rh_movers.requests, "get", _get)

    with caplog.at_level("WARNING"):
        path = rh_movers.capture(date=dt.date(2026, 8, 27), direction="up")

    envelope = json.loads(path.read_text())
    assert envelope["sp500"]["available"] is True
    assert envelope["top_movers"]["available"] is False
    assert envelope["top_movers"]["symbols"] is None
    assert envelope["top_movers_available"] is False

    assert any("PRIMARY PROXY-VALIDATION SOURCE UNAVAILABLE" in msg for msg in caplog.messages)


def test_capture_refuses_overwrite_existing_day(tmp_path, monkeypatch):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, movers=_RS_SP500_PAYLOAD)
    monkeypatch.setattr(
        rh_movers.requests,
        "get",
        _requests_get_router(tag_payload=_TOP_MOVERS_TAG_PAYLOAD, instrument_symbols=_INSTRUMENT_SYMBOLS),
    )

    trade_date = dt.date(2026, 8, 27)
    path = rh_movers.capture(date=trade_date, direction="up")
    original = json.loads(path.read_text())

    def _boom_sp500(*a, **k):
        raise AssertionError("fetch_sp500_movers should not be called for an already-captured day")

    def _boom_top_movers(*a, **k):
        raise AssertionError("fetch_top_movers should not be called for an already-captured day")

    monkeypatch.setattr(rh_movers, "fetch_sp500_movers", _boom_sp500)
    monkeypatch.setattr(rh_movers, "fetch_top_movers", _boom_top_movers)

    same_path = rh_movers.capture(date=trade_date, direction="up")
    assert same_path == path
    assert json.loads(path.read_text()) == original


def test_capture_raises_when_both_feeds_fail_entirely(tmp_path, monkeypatch):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, raises=RuntimeError("sdk down"))

    def _fail(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(rh_movers.requests, "get", _fail)

    with pytest.raises(RuntimeError):
        rh_movers.capture(date=dt.date(2026, 8, 27), direction="up")

    assert not (tmp_path / "rh_movers" / "2026-08-27.json").exists()


# --- load_captured_movers ----------------------------------------------------


def test_load_captured_movers_returns_none_for_missing_date(tmp_path, monkeypatch):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    assert rh_movers.load_captured_movers(dt.date(2099, 1, 1)) is None


def test_load_captured_movers_returns_envelope_with_tickers_when_captured(tmp_path, monkeypatch):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, movers=_RS_SP500_PAYLOAD)
    monkeypatch.setattr(
        rh_movers.requests,
        "get",
        _requests_get_router(tag_payload=_TOP_MOVERS_TAG_PAYLOAD, instrument_symbols=_INSTRUMENT_SYMBOLS),
    )

    def get_top_movers(*args):
        if len(args) == 0:
            raise RuntimeError("force https path for top_movers")
        return _RS_SP500_PAYLOAD

    fake_markets = types.ModuleType("robin_stocks.robinhood.markets")
    fake_markets.get_top_movers = get_top_movers
    fake_markets.get_top_movers_sp500 = lambda direction: None
    fake_robinhood = types.ModuleType("robin_stocks.robinhood")
    fake_robinhood.markets = fake_markets
    monkeypatch.setitem(sys.modules, "robin_stocks", types.ModuleType("robin_stocks"))
    monkeypatch.setitem(sys.modules, "robin_stocks.robinhood", fake_robinhood)
    monkeypatch.setitem(sys.modules, "robin_stocks.robinhood.markets", fake_markets)

    trade_date = dt.date(2026, 8, 27)
    rh_movers.capture(date=trade_date, direction="up")

    loaded = rh_movers.load_captured_movers(trade_date)
    assert loaded is not None
    assert loaded["top_movers_available"] is True
    assert loaded["top_movers_tickers"] == ["MOV1", "MOV2", "MOV3"]


def test_load_captured_movers_signals_unavailability_for_sp500_only_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, raises=RuntimeError("sdk down"))

    def _get(url, params=None, headers=None, timeout=None):
        if "movers/sp500" in url:
            return _FakeResponse(_HTTPS_SP500_PAYLOAD)
        if "tags/tag/top-movers" in url:
            raise RuntimeError("tag endpoint down")
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(rh_movers.requests, "get", _get)

    trade_date = dt.date(2026, 8, 27)
    rh_movers.capture(date=trade_date, direction="up")

    loaded = rh_movers.load_captured_movers(trade_date)
    assert loaded is not None
    assert loaded["top_movers_available"] is False
    assert loaded["top_movers_tickers"] is None
    # The absence must be unmistakable, not silently substituted.
    assert loaded["sp500"]["available"] is True


# --- CLI ---------------------------------------------------------------------


def test_main_returns_1_on_total_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, raises=RuntimeError("sdk down"))

    def _fail(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(rh_movers.requests, "get", _fail)

    exit_code = rh_movers.main(["--direction", "up", "--date", "2026-08-27"])
    assert exit_code == 1


def test_main_returns_0_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(rh_movers, "DATA_RAW", tmp_path)
    _install_fake_robin_stocks(monkeypatch, movers=_RS_SP500_PAYLOAD)
    monkeypatch.setattr(
        rh_movers.requests,
        "get",
        _requests_get_router(tag_payload=_TOP_MOVERS_TAG_PAYLOAD, instrument_symbols=_INSTRUMENT_SYMBOLS),
    )

    exit_code = rh_movers.main(["--direction", "up", "--date", "2026-08-27"])
    assert exit_code == 0
    assert (tmp_path / "rh_movers" / "2026-08-27.json").exists()
