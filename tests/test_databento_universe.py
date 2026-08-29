"""Offline tests for the Databento P2 (survivorship-bias) defense: the
full-cross-section daily pull, point-in-time symbol resolution, inferred
delistings, and the loud survivorship-verification check.

No network access, no API keys, no live paid Databento calls -- every
Databento SDK object used here is a fake/mock. Free metadata-style calls
(`metadata.get_cost`, `symbology.resolve`) are simulated the same way.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from top10.data.databento import (
    DatabentoSource,
    SurvivorshipReport,
    delistings_to_corporate_actions,
    infer_delistings,
    verify_no_survivorship,
)
from top10.data.symbology import AmbiguousSymbolError, SymbolResolver


# --- daily_bars: full cross-section, never a resolved symbol list ----------


def test_daily_bars_requests_all_symbols_never_a_resolved_ticker_list(monkeypatch, tmp_path):
    """The P2 defense: `daily_bars` must request `symbols="ALL_SYMBOLS"` for
    every chunk, NOT a pre-resolved list of ticker strings. Inverting this
    (resolving "today's" tickers first, then pulling history only for
    those) is exactly how survivorship bias enters."""
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    calls: list[dict] = []

    def _fake_fetch_bars(schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        calls.append({"schema": schema, "start": start, "end": end, "symbols": symbols})
        return []

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)

    source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 10))

    assert len(calls) >= 1
    for call in calls:
        assert call["symbols"] == "ALL_SYMBOLS"


def test_daily_bars_carries_instrument_id_as_extra_column(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    records = [
        {
            "ts_event": "2024-01-05T00:00:00Z",
            "symbol": "AAPL",
            "instrument_id": 12345,
            "open": 189.5,
            "high": 192.0,
            "low": 188.0,
            "close": 191.0,
            "volume": 1_000_000.0,
        }
    ]
    monkeypatch.setattr(source, "_fetch_bars", lambda *a, **k: records)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))

    assert "instrument_id" in df.columns
    assert df["instrument_id"].iloc[0] == 12345


def test_daily_bars_chunks_by_month_and_resumes_without_respending(monkeypatch, tmp_path):
    """A mid-pull failure must not re-fetch (and therefore not re-spend on)
    an already-completed month when the same range is requested again."""
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    calls: list[tuple[str, str]] = []

    # `cached_call` deliberately never caches an EMPTY payload (see
    # top10/data/cache.py) so a transient failure can't masquerade as "no
    # data" -- each chunk here returns a non-empty record so a successful
    # chunk actually gets checkpointed to disk.
    _record = [
        {
            "ts_event": "2024-01-05T00:00:00Z",
            "symbol": "AAPL",
            "instrument_id": 1,
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        }
    ]

    def _flaky_fetch_bars(schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        calls.append((start, end))
        if start == "2024-02-01":
            raise RuntimeError("simulated mid-pull network failure")
        return _record

    monkeypatch.setattr(source, "_fetch_bars", _flaky_fetch_bars)

    with pytest.raises(RuntimeError, match="simulated mid-pull"):
        source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 3, 5))

    # January's chunk succeeded and must be cached to disk before the
    # February chunk's failure -- i.e. it was checkpointed, not lost.
    assert len(calls) == 2  # January (succeeded), February (raised)

    # Fix the flakiness and re-run the SAME range: January must not be
    # re-fetched (it's already on disk), only February and March.
    calls.clear()

    def _fixed_fetch_bars(schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        calls.append((start, end))
        return _record

    monkeypatch.setattr(source, "_fetch_bars", _fixed_fetch_bars)

    source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 3, 5))

    starts = [c[0] for c in calls]
    assert "2024-01-05" not in starts  # already cached -- not re-spent on
    assert "2024-02-01" in starts
    assert "2024-03-01" in starts


def test_daily_bars_pulled_frame_keeps_a_2022_delisted_ticker_in_the_2019_universe(
    monkeypatch, tmp_path
):
    """The exact P2 repro: a ticker present in 2019 and delisted (absent)
    by 2022 must still appear in the subset of the pulled frame usable to
    build a 2019 universe -- i.e. the pull is NOT built from a 2022
    ("today's") ticker list."""
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    def _fake_fetch_bars(schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        records = [
            {
                "ts_event": f"{start}T00:00:00Z",
                "symbol": "SURVIVOR",
                "instrument_id": 1,
                "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0,
            }
        ]
        # DELISTED_2020 only trades during 2019 chunks.
        if start.startswith("2019"):
            records.append(
                {
                    "ts_event": f"{start}T00:00:00Z",
                    "symbol": "DELISTED_2020",
                    "instrument_id": 2,
                    "open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 50.0,
                }
            )
        return records

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)

    df = source.daily_bars(dt.date(2019, 1, 2), dt.date(2022, 1, 3))

    universe_2019 = df[df["trade_date"].dt.year == 2019]
    universe_2022 = df[df["trade_date"].dt.year == 2022]

    assert "DELISTED_2020" in set(universe_2019["ticker"])
    assert "DELISTED_2020" not in set(universe_2022["ticker"])
    assert "SURVIVOR" in set(universe_2022["ticker"])


# --- cost estimator: no download -------------------------------------------


class _FakeMetadata:
    def __init__(self, cost=0.25):
        self.cost = cost
        self.calls: list[dict] = []

    def get_cost(self, **kwargs):
        self.calls.append(kwargs)
        return self.cost


class _FakeTimeseriesNoCalls:
    def get_range(self, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("estimate_universe_pull_cost must never download data")


class _FakeClient:
    def __init__(self, cost=0.25):
        self.metadata = _FakeMetadata(cost=cost)
        self.timeseries = _FakeTimeseriesNoCalls()


def test_estimate_universe_pull_cost_performs_no_download(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()
    client = _FakeClient(cost=0.5)
    monkeypatch.setattr(source, "_get_client", lambda: client)

    result = source.estimate_universe_pull_cost(dt.date(2024, 1, 5), dt.date(2024, 3, 5))

    # Jan / Feb / Mar -- 3 calendar-month chunks.
    assert len(client.metadata.calls) == 3
    assert result["total_cost_usd"] == pytest.approx(1.5)
    assert len(result["chunks"]) == 3


# --- symbology: point-in-time resolution + reuse detection -----------------


class _FakeSymbology:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def resolve(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeSymbologyClient:
    def __init__(self, response):
        self.symbology = _FakeSymbology(response)


def test_symbol_resolver_resolves_reused_symbol_to_different_instrument_ids(tmp_path, monkeypatch):
    """Same raw symbol string ("ABC") reassigned to a different, unrelated
    issuer over time must resolve to two DIFFERENT instrument_ids, never
    merge into a single series."""
    monkeypatch.setattr("top10.data.symbology.DATA_RAW", tmp_path)

    response = {
        "result": {
            "ABC": [
                {"d0": "2015-01-01", "d1": "2016-06-01", "s": "1001"},
                {"d0": "2020-01-01", "d1": "2021-01-01", "s": "2002"},
            ]
        }
    }
    client = _FakeSymbologyClient(response)

    resolver = SymbolResolver("XNAS.ITCH")
    resolver.resolve_range(["ABC"], dt.date(2015, 1, 1), dt.date(2021, 1, 1), client=client)

    assert resolver.resolve_at("ABC", dt.date(2015, 6, 1)) == "1001"
    assert resolver.resolve_at("ABC", dt.date(2020, 6, 1)) == "2002"
    # Between the two intervals: unresolved, not silently one or the other.
    assert resolver.resolve_at("ABC", dt.date(2018, 1, 1)) is None

    # instrument_id -> symbol inverse lookup, per-era.
    assert resolver.symbol_at("1001", dt.date(2015, 6, 1)) == "ABC"
    assert resolver.symbol_at("2002", dt.date(2020, 6, 1)) == "ABC"


def test_symbol_resolver_detect_reuse_flags_multi_instrument_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr("top10.data.symbology.DATA_RAW", tmp_path)

    response = {
        "result": {
            "ABC": [
                {"d0": "2015-01-01", "d1": "2016-06-01", "s": "1001"},
                {"d0": "2020-01-01", "d1": "2021-01-01", "s": "2002"},
            ],
            "STABLE": [
                {"d0": "2015-01-01", "d1": "2021-01-01", "s": "3003"},
            ],
        }
    }
    client = _FakeSymbologyClient(response)

    resolver = SymbolResolver("XNAS.ITCH")
    resolver.resolve_range(
        ["ABC", "STABLE"], dt.date(2015, 1, 1), dt.date(2021, 1, 1), client=client
    )

    reuse = resolver.detect_reuse(dt.date(2015, 1, 1), dt.date(2021, 1, 1))

    assert set(reuse["raw_symbol"]) == {"ABC"}
    assert set(reuse["instrument_id"]) == {"1001", "2002"}
    assert "STABLE" not in set(reuse["raw_symbol"])


def test_symbol_resolver_persists_and_avoids_reresolving(tmp_path, monkeypatch):
    monkeypatch.setattr("top10.data.symbology.DATA_RAW", tmp_path)

    response = {"result": {"ABC": [{"d0": "2015-01-01", "d1": "2021-01-01", "s": "1001"}]}}
    client = _FakeSymbologyClient(response)

    resolver1 = SymbolResolver("XNAS.ITCH")
    resolver1.resolve_range(["ABC"], dt.date(2015, 1, 1), dt.date(2021, 1, 1), client=client)
    assert len(client.symbology.calls) == 1

    # A fresh resolver instance, same dataset/disk: must load from disk and
    # skip re-resolving an already-covered symbol.
    resolver2 = SymbolResolver("XNAS.ITCH")
    resolver2.resolve_range(["ABC"], dt.date(2015, 1, 1), dt.date(2021, 1, 1), client=client)
    assert len(client.symbology.calls) == 1  # unchanged -- no re-resolve
    assert resolver2.resolve_at("ABC", dt.date(2016, 1, 1)) == "1001"


def test_symbol_resolver_ambiguous_overlapping_intervals_raise(tmp_path, monkeypatch):
    monkeypatch.setattr("top10.data.symbology.DATA_RAW", tmp_path)

    resolver = SymbolResolver("XNAS.ITCH")
    resolver.load()
    resolver._intervals["ABC"] = [
        {"d0": "2015-01-01", "d1": "2016-01-01", "s": "1001"},
        {"d0": "2015-06-01", "d1": "2016-06-01", "s": "2002"},  # overlaps
    ]

    with pytest.raises(AmbiguousSymbolError):
        resolver.resolve_at("ABC", dt.date(2015, 7, 1))


# --- infer_delistings: distinguishes a halt from a true stop ---------------


def _bars_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def test_infer_delistings_flags_a_true_stop():
    sessions = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for d in sessions:
        rows.append({"trade_date": d, "ticker": "SURVIVOR", "instrument_id": 1, "close": 10.0})
    # DELISTED trades the first 10 sessions only, then never again.
    for d in sessions[:10]:
        rows.append({"trade_date": d, "ticker": "DELISTED", "instrument_id": 2, "close": 5.0})

    daily_bars = _bars_frame(rows)
    out = infer_delistings(daily_bars, min_gap_days=5)

    assert set(out["ticker"]) == {"DELISTED"}
    row = out[out["ticker"] == "DELISTED"].iloc[0]
    assert row["last_trade_date"] == sessions[9]
    assert row["inferred_delist_date"] == sessions[10]
    assert row["confidence"] > 0


def test_infer_delistings_does_not_flag_a_short_halt_that_resumes():
    sessions = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for d in sessions:
        rows.append({"trade_date": d, "ticker": "SURVIVOR", "instrument_id": 1, "close": 10.0})
    # HALTED misses a 3-session gap in the middle, then resumes trading
    # through to the end of the window -- never absent from the FINAL
    # session, so it must not be flagged.
    halted_sessions = [d for i, d in enumerate(sessions) if i not in (10, 11, 12)]
    for d in halted_sessions:
        rows.append({"trade_date": d, "ticker": "HALTED", "instrument_id": 2, "close": 5.0})

    daily_bars = _bars_frame(rows)
    out = infer_delistings(daily_bars, min_gap_days=5)

    assert "HALTED" not in set(out["ticker"])


def test_infer_delistings_short_gap_below_threshold_is_not_flagged():
    sessions = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for d in sessions:
        rows.append({"trade_date": d, "ticker": "SURVIVOR", "instrument_id": 1, "close": 10.0})
    # QUIET_STOP disappears only 3 sessions before the window's end --
    # below min_gap_days=5, more consistent with a brief halt than a
    # confirmed delisting.
    for d in sessions[:17]:
        rows.append({"trade_date": d, "ticker": "QUIET_STOP", "instrument_id": 2, "close": 5.0})

    daily_bars = _bars_frame(rows)
    out = infer_delistings(daily_bars, min_gap_days=5)

    assert "QUIET_STOP" not in set(out["ticker"])


def test_delistings_to_corporate_actions_shapes_output_and_never_backdates_as_of():
    delistings = pd.DataFrame(
        {
            "ticker": ["DELISTED"],
            "instrument_id": [2],
            "last_trade_date": [pd.Timestamp("2024-01-15")],
            "inferred_delist_date": [pd.Timestamp("2024-01-16")],
            "confidence": [0.8],
        }
    )

    out = delistings_to_corporate_actions(delistings)

    row = out.iloc[0]
    assert row["action_type"] == "delisting"
    assert row["as_of"] == pd.Timestamp("2024-01-16")
    # as_of must equal the OBSERVABLE (inferred_delist_date), never the
    # last_trade_date itself -- that would be knowledge before the absence
    # that triggered the inference had happened.
    assert row["as_of"] != pd.Timestamp("2024-01-15")
    assert out["confidence"].iloc[0] == 0.8


# --- verify_no_survivorship: loud FAIL on a survivor-only frame ------------


def test_verify_no_survivorship_fails_on_survivor_only_frame():
    """A frame where every ticker present early is STILL present late is
    exactly the survivorship-bias signature -- must FAIL loudly, not pass
    silently."""
    sessions = pd.date_range("2018-06-01", periods=400, freq="B")
    rows = []
    for d in sessions:
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            rows.append({"trade_date": d, "ticker": ticker, "close": 100.0})
    daily_bars = _bars_frame(rows)

    report = verify_no_survivorship(daily_bars)

    assert isinstance(report, SurvivorshipReport)
    assert report.passed is False
    assert not report  # __bool__ reflects passed
    assert report.disappeared_tickers == set()


def test_verify_no_survivorship_passes_when_names_genuinely_disappear():
    sessions = pd.date_range("2018-06-01", periods=400, freq="B")
    rows = []
    for d in sessions:
        rows.append({"trade_date": d, "ticker": "SURVIVOR", "close": 100.0})
    # DELISTED_CO only trades the first half of the window.
    for d in sessions[: len(sessions) // 2]:
        rows.append({"trade_date": d, "ticker": "DELISTED_CO", "close": 5.0})

    daily_bars = _bars_frame(rows)
    report = verify_no_survivorship(daily_bars)

    assert report.passed is True
    assert bool(report) is True
    assert "DELISTED_CO" in report.disappeared_tickers
