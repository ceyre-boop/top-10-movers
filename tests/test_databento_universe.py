"""Offline tests for the Databento P2 (survivorship-bias) defense AND for
the live-data-verified fixes to this adapter's model of Databento equities
(see the module docstring of `top10/data/databento.py` and
`top10/data/symbology.py` for the full write-up of each finding).

No network access, no API keys, no live paid Databento calls -- every
Databento SDK object used here is a fake/mock. Free metadata-style calls
(`metadata.get_cost`, `metadata.get_dataset_condition`, `symbology.resolve`)
are simulated the same way.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from top10.data.databento import (
    DatabentoSource,
    FIRST_AVAILABLE_DATE,
    LISTING_VENUE_DATASETS,
    SurvivorshipReport,
    delistings_to_corporate_actions,
    infer_delistings,
    verify_no_survivorship,
)
from top10.data.symbology import AmbiguousSymbolError, SymbolResolver


# --- shared fakes ------------------------------------------------------------


class _FakeMetadata:
    """Fakes the two free metadata endpoints this adapter calls:
    `get_cost` (cost estimate, used by `CostGuard`) and
    `get_dataset_condition` (degraded/pending day flags)."""

    def __init__(self, cost: float = 0.01, condition: dict[str, list[dict]] | None = None):
        self.cost = cost
        self.condition = condition or {}
        self.get_cost_calls: list[dict] = []
        self.get_dataset_condition_calls: list[dict] = []

    def get_cost(self, **kwargs):
        self.get_cost_calls.append(kwargs)
        return self.cost

    def get_record_count(self, **kwargs):
        return 0

    def get_dataset_condition(self, dataset, start_date, end_date):
        self.get_dataset_condition_calls.append(
            {"dataset": dataset, "start_date": start_date, "end_date": end_date}
        )
        return self.condition.get(dataset, [])


class _FakeSymbology:
    """Fakes `client.symbology.resolve`. `mapping` is keyed
    `(dataset, start_date_iso)` -> `{input_symbol_str: output_symbol_str}`,
    mirroring exactly the per-DAY request `SymbolResolver.resolve_day` /
    `resolve_at` make (`start_date`, `end_date=start_date+1`) -- there is no
    way to configure a multi-day interval here, which is the point."""

    def __init__(self, mapping: dict[tuple[str, str], dict[str, str]] | None = None):
        self.mapping = mapping or {}
        self.calls: list[dict] = []

    def resolve(self, *, dataset, symbols, stype_in, stype_out, start_date, end_date):
        self.calls.append(
            dict(
                dataset=dataset,
                symbols=list(symbols),
                stype_in=stype_in,
                stype_out=stype_out,
                start_date=start_date,
                end_date=end_date,
            )
        )
        day_map = self.mapping.get((dataset, start_date), {})
        result: dict[str, list[dict[str, str]]] = {}
        for s in symbols:
            if s in day_map:
                result[s] = [{"d0": start_date, "d1": end_date, "s": day_map[s]}]
        return {"result": result}


class _FakeClient:
    def __init__(
        self,
        cost: float = 0.01,
        symbology_mapping: dict[tuple[str, str], dict[str, str]] | None = None,
        condition: dict[str, list[dict]] | None = None,
    ):
        self.metadata = _FakeMetadata(cost=cost, condition=condition)
        self.symbology = _FakeSymbology(symbology_mapping)


def _make_source(monkeypatch, tmp_path, venues=None) -> DatabentoSource:
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    return DatabentoSource(venues=venues)


# --- daily_bars: full cross-section, never a resolved symbol list ----------


def test_daily_bars_requests_all_symbols_never_a_resolved_ticker_list(monkeypatch, tmp_path):
    """The P2 defense: `daily_bars` must request `symbols="ALL_SYMBOLS"` for
    every chunk/venue, NOT a pre-resolved list of ticker strings."""
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH"])

    calls: list[dict] = []

    def _fake_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        calls.append({"dataset": dataset, "schema": schema, "start": start, "end": end, "symbols": symbols})
        return []

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)
    monkeypatch.setattr(source, "_get_client", lambda: _FakeClient())

    source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 10))

    assert len(calls) >= 1
    for call in calls:
        assert call["symbols"] == "ALL_SYMBOLS"


def test_daily_bars_unions_the_three_listing_venue_datasets_by_default():
    """No consolidated US equities dataset exists before 2023 -- finding
    (1). The default venue set must be exactly the three listing-venue
    feeds verified to have `ohlcv-1d` history from 2018-05-01."""
    source = DatabentoSource()
    assert source._venues == LISTING_VENUE_DATASETS
    assert LISTING_VENUE_DATASETS == ["XNAS.ITCH", "XNYS.PILLAR", "XASE.PILLAR"]


def test_daily_bars_carries_instrument_id_as_extra_informational_column(monkeypatch, tmp_path):
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH"])

    records = [
        {
            "ts_event": "2024-01-05T00:00:00Z",
            "instrument_id": 27,
            "open": 189.5, "high": 192.0, "low": 188.0, "close": 191.0, "volume": 1_000_000.0,
        }
    ]

    def _fake_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        return [] if schema == "definition" else records

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)
    fake_client = _FakeClient(symbology_mapping={("XNAS.ITCH", "2024-01-05"): {"27": "AAPL"}})
    monkeypatch.setattr(source, "_get_client", lambda: fake_client)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))

    assert "instrument_id" in df.columns
    assert df["ticker"].iloc[0] == "AAPL"
    assert str(df["instrument_id"].iloc[0]) == "27"


def test_daily_bars_chunks_by_month_and_resumes_without_respending(monkeypatch, tmp_path):
    """A mid-pull failure must not re-fetch (and therefore not re-spend on)
    an already-completed month when the same range is requested again."""
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH"])
    monkeypatch.setattr(source, "_get_client", lambda: _FakeClient(
        symbology_mapping={
            ("XNAS.ITCH", "2024-01-05"): {"1": "AAPL"},
            ("XNAS.ITCH", "2024-02-01"): {"1": "AAPL"},
            ("XNAS.ITCH", "2024-03-01"): {"1": "AAPL"},
        }
    ))

    calls: list[tuple[str, str]] = []

    _record = [
        {
            "ts_event": "PLACEHOLDER",
            "instrument_id": 1,
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        }
    ]

    def _flaky_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        if schema == "definition":
            return []
        calls.append((start, end))
        if start == "2024-02-01":
            raise RuntimeError("simulated mid-pull network failure")
        rec = dict(_record[0])
        rec["ts_event"] = f"{start}T00:00:00Z"
        return [rec]

    monkeypatch.setattr(source, "_fetch_bars", _flaky_fetch_bars)

    with pytest.raises(RuntimeError, match="simulated mid-pull"):
        source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 3, 5))

    # January's chunk succeeded and must be cached to disk before the
    # February chunk's failure -- i.e. it was checkpointed, not lost.
    assert len(calls) == 2  # January (succeeded), February (raised)

    calls.clear()

    def _fixed_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        if schema == "definition":
            return []
        calls.append((start, end))
        rec = dict(_record[0])
        rec["ts_event"] = f"{start}T00:00:00Z"
        return [rec]

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
    build a 2019 universe."""
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH"])

    symbology_mapping = {
        ("XNAS.ITCH", "2019-01-02"): {"1": "SURVIVOR", "2": "DELISTED_2020"},
        ("XNAS.ITCH", "2020-01-01"): {"1": "SURVIVOR"},
        ("XNAS.ITCH", "2021-01-01"): {"1": "SURVIVOR"},
        ("XNAS.ITCH", "2022-01-01"): {"1": "SURVIVOR"},
    }
    monkeypatch.setattr(source, "_get_client", lambda: _FakeClient(symbology_mapping=symbology_mapping))

    def _fake_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        if schema == "definition":
            return []
        records = [
            {
                "ts_event": f"{start}T00:00:00Z",
                "instrument_id": 1,
                "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0,
            }
        ]
        if start.startswith("2019"):
            records.append(
                {
                    "ts_event": f"{start}T00:00:00Z",
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


# --- THE +8702% REGRESSION: instrument_id reassigned daily -----------------


def test_daily_bars_resolves_ticker_per_day_not_range_wide(monkeypatch, tmp_path):
    """THE verified live-data finding: `instrument_id` is REASSIGNED DAILY
    by Databento (LULU: 6844, 6843, 6839... across consecutive sessions).
    The SAME `instrument_id` here (6844) is bound to a DIFFERENT company on
    each of two different days -- a range-wide resolve would have labeled
    every row "LULU" (whichever ticker its interval happened to cover),
    exactly the bug that produced a live +8702% median top-10-gainer
    return. Per-day resolution must produce the CORRECT ticker each day.
    """
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH"])

    def _fake_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        if schema == "definition":
            return []
        return [
            {
                "ts_event": "2024-01-05T00:00:00Z",
                "instrument_id": 6844,
                "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0,
            },
            {
                "ts_event": "2024-01-08T00:00:00Z",
                "instrument_id": 6844,
                "open": 900.0, "high": 900.0, "low": 900.0, "close": 900.0, "volume": 5.0,
            },
        ]

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)
    fake_client = _FakeClient(
        symbology_mapping={
            ("XNAS.ITCH", "2024-01-05"): {"6844": "LULU"},
            ("XNAS.ITCH", "2024-01-08"): {"6844": "MRNA"},
        }
    )
    monkeypatch.setattr(source, "_get_client", lambda: fake_client)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 8))

    jan5 = df[df["trade_date"] == pd.Timestamp("2024-01-05")]
    jan8 = df[df["trade_date"] == pd.Timestamp("2024-01-08")]

    assert set(jan5["ticker"]) == {"LULU"}
    assert set(jan8["ticker"]) == {"MRNA"}
    assert jan5["close"].iloc[0] == 10.0
    assert jan8["close"].iloc[0] == 900.0
    # The bug this guards against: labeling id 6844 "LULU" on BOTH days.
    assert "LULU" not in set(jan8["ticker"])
    assert "MRNA" not in set(jan5["ticker"])

    # And the resolve calls actually asked for ONE day at a time, never a
    # multi-day span.
    for call in fake_client.symbology.calls:
        d0 = pd.Timestamp(call["start_date"])
        d1 = pd.Timestamp(call["end_date"])
        assert (d1 - d0) == pd.Timedelta(days=1)


def test_symbol_resolver_has_no_range_wide_resolve_method():
    """`resolve_range` (which persisted one interval spanning an arbitrary
    date range) is REMOVED, not merely deprecated -- it cannot be reached
    by accident."""
    assert not hasattr(SymbolResolver, "resolve_range")


def test_symbol_resolver_resolve_day_batches_ids_in_groups_of_500(tmp_path, monkeypatch):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)

    ids = list(range(1200))
    mapping = {("XNAS.ITCH", "2024-01-05"): {str(i): f"T{i}" for i in ids}}
    client = _FakeClient(symbology_mapping=mapping)

    resolver = SymbolResolver("XNAS.ITCH")
    result = resolver.resolve_day(dt.date(2024, 1, 5), ids, client=client)

    assert len(result) == 1200
    assert result["0"] == "T0"
    assert result["1199"] == "T1199"
    # 1200 ids in batches of <=500 -> 3 resolve() calls.
    assert len(client.symbology.calls) == 3
    for call in client.symbology.calls:
        assert len(call["symbols"]) <= 500


def test_symbol_resolver_resolve_day_caches_per_dataset_and_day(tmp_path, monkeypatch):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)

    mapping = {("XNAS.ITCH", "2024-01-05"): {"1": "AAPL"}}
    client = _FakeClient(symbology_mapping=mapping)

    resolver1 = SymbolResolver("XNAS.ITCH")
    resolver1.resolve_day(dt.date(2024, 1, 5), [1], client=client)
    assert len(client.symbology.calls) == 1

    # A fresh resolver, same dataset/day/disk: cached, no re-resolve.
    resolver2 = SymbolResolver("XNAS.ITCH")
    result = resolver2.resolve_day(dt.date(2024, 1, 5), [1], client=client)
    assert len(client.symbology.calls) == 1  # unchanged
    assert result["1"] == "AAPL"


def test_symbol_resolver_resolve_day_never_reused_across_different_days(tmp_path, monkeypatch):
    """A DIFFERENT day for the SAME dataset/id must trigger its OWN
    resolve -- the disk cache is keyed per (dataset, day), never merged
    across days into one map."""
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)

    mapping = {
        ("XNAS.ITCH", "2024-01-05"): {"6844": "LULU"},
        ("XNAS.ITCH", "2024-01-08"): {"6844": "MRNA"},
    }
    client = _FakeClient(symbology_mapping=mapping)
    resolver = SymbolResolver("XNAS.ITCH")

    r1 = resolver.resolve_day(dt.date(2024, 1, 5), [6844], client=client)
    r2 = resolver.resolve_day(dt.date(2024, 1, 8), [6844], client=client)

    assert r1["6844"] == "LULU"
    assert r2["6844"] == "MRNA"
    assert len(client.symbology.calls) == 2


def test_symbol_resolver_resolve_day_ambiguous_result_raises():
    resolver = SymbolResolver("XNAS.ITCH")

    class _AmbiguousSymbology:
        def resolve(self, **kwargs):
            return {"result": {"1": [{"d0": kwargs["start_date"], "d1": kwargs["end_date"], "s": "AAA"},
                                       {"d0": kwargs["start_date"], "d1": kwargs["end_date"], "s": "BBB"}]}}

    class _Client:
        symbology = _AmbiguousSymbology()

    with pytest.raises(AmbiguousSymbolError):
        resolver.resolve_day(dt.date(2024, 1, 5), [1], client=_Client())


# --- ts_event: UTC midnight OF the trade date, never tz-converted ----------


def test_daily_bars_ts_event_is_utc_date_not_tz_shifted(monkeypatch, tmp_path):
    """Converting `ts_event` to America/New_York shifts every bar back a
    day and manufactures phantom weekend sessions (verified live: a
    "2022-06-05" row -- a Sunday -- appeared from a Monday bar). This must
    use the UTC date directly."""
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH"])

    # 2022-06-06 is a Monday. Its ohlcv-1d ts_event is UTC midnight of that
    # SAME date.
    def _fake_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        if schema == "definition":
            return []
        return [
            {
                "ts_event": "2022-06-06T00:00:00Z",
                "instrument_id": 1,
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            }
        ]

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)
    fake_client = _FakeClient(symbology_mapping={("XNAS.ITCH", "2022-06-06"): {"1": "AAPL"}})
    monkeypatch.setattr(source, "_get_client", lambda: fake_client)

    df = source.daily_bars(dt.date(2022, 6, 6), dt.date(2022, 6, 6))

    trade_dates = set(df["trade_date"].dt.strftime("%Y-%m-%d"))
    assert trade_dates == {"2022-06-06"}
    # Explicitly not shifted back a day, and no weekend date manufactured.
    assert "2022-06-05" not in trade_dates
    weekday = pd.Timestamp(df["trade_date"].iloc[0]).weekday()
    assert weekday < 5  # Monday=0 .. Friday=4, never Sat/Sun


# --- volume summed across venues, close from the listing venue only -------


def test_daily_bars_sums_volume_across_venues_close_from_listing_venue_only(monkeypatch, tmp_path):
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH", "XNYS.PILLAR"])

    def _fake_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        if schema == "definition":
            if dataset == "XNAS.ITCH":
                return [{"raw_symbol": "LULU", "exchange": "XNAS", "instrument_class": "C"}]
            return []
        if dataset == "XNAS.ITCH":
            return [
                {
                    "ts_event": "2024-01-05T00:00:00Z",
                    "instrument_id": 1,
                    "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0,
                }
            ]
        # XNYS.PILLAR -- a stray print of the same, Nasdaq-listed, name.
        return [
            {
                "ts_event": "2024-01-05T00:00:00Z",
                "instrument_id": 999,
                "open": 100.2, "high": 101.0, "low": 99.0, "close": 100.9, "volume": 500.0,
            }
        ]

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)
    fake_client = _FakeClient(
        symbology_mapping={
            ("XNAS.ITCH", "2024-01-05"): {"1": "LULU"},
            ("XNYS.PILLAR", "2024-01-05"): {"999": "LULU"},
        }
    )
    monkeypatch.setattr(source, "_get_client", lambda: fake_client)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5))

    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "LULU"
    # Volume SUMS across both venues' prints of the same name.
    assert row["volume"] == 1500.0
    # Close comes from the LISTING venue (XNAS.ITCH, per the definition
    # pull's exchange field) -- NEVER averaged/last-across-venues.
    assert row["close"] == 100.5


# --- pre-2018-05-01 start rejected loudly -----------------------------------


def test_daily_bars_rejects_start_before_first_available_date():
    source = DatabentoSource()
    with pytest.raises(ValueError, match="2018-05-01"):
        source.daily_bars(FIRST_AVAILABLE_DATE - dt.timedelta(days=1), FIRST_AVAILABLE_DATE)


def test_estimate_universe_pull_cost_rejects_start_before_first_available_date():
    source = DatabentoSource()
    with pytest.raises(ValueError, match="2018-05-01"):
        source.estimate_universe_pull_cost(
            FIRST_AVAILABLE_DATE - dt.timedelta(days=1), FIRST_AVAILABLE_DATE
        )


# --- degraded days are surfaced and excluded --------------------------------


def test_daily_bars_excludes_and_surfaces_degraded_dates(monkeypatch, tmp_path):
    """A degraded day silently produces wrong returns -- Databento flags
    it via the free `metadata.get_dataset_condition` endpoint, and this
    must both drop the day from the output AND record it for the caller."""
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH"])

    def _fake_fetch_bars(dataset, schema, start, end, symbols="ALL_SYMBOLS", *, confirm=False):
        if schema == "definition":
            return []
        return [
            {
                "ts_event": "2024-01-05T00:00:00Z",
                "instrument_id": 1,
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 10.0,
            },
            {
                "ts_event": "2024-01-08T00:00:00Z",
                "instrument_id": 1,
                "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 20.0,
            },
        ]

    monkeypatch.setattr(source, "_fetch_bars", _fake_fetch_bars)
    fake_client = _FakeClient(
        symbology_mapping={
            ("XNAS.ITCH", "2024-01-05"): {"1": "AAPL"},
            ("XNAS.ITCH", "2024-01-08"): {"1": "AAPL"},
        },
        condition={"XNAS.ITCH": [{"date": "2024-01-05", "condition": "degraded"}]},
    )
    monkeypatch.setattr(source, "_get_client", lambda: fake_client)

    df = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 8))

    trade_dates = set(df["trade_date"].dt.strftime("%Y-%m-%d"))
    assert "2024-01-05" not in trade_dates
    assert "2024-01-08" in trade_dates
    assert any("2024-01-05" in entry for entry in source.last_degraded_dates)


# --- cost estimator: no download -------------------------------------------


class _FakeMetadataNoCalls(_FakeMetadata):
    pass


class _FakeTimeseriesNoCalls:
    def get_range(self, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("estimate_universe_pull_cost must never download data")


class _FullFakeClient:
    def __init__(self, cost=0.25):
        self.metadata = _FakeMetadataNoCalls(cost=cost)
        self.timeseries = _FakeTimeseriesNoCalls()


def test_estimate_universe_pull_cost_performs_no_download(monkeypatch, tmp_path):
    source = _make_source(monkeypatch, tmp_path, venues=["XNAS.ITCH", "XNYS.PILLAR"])
    client = _FullFakeClient(cost=0.5)
    monkeypatch.setattr(source, "_get_client", lambda: client)

    result = source.estimate_universe_pull_cost(dt.date(2024, 1, 5), dt.date(2024, 3, 5))

    # 3 calendar-month chunks x 2 venues.
    assert len(client.metadata.get_cost_calls) == 6
    assert result["total_cost_usd"] == pytest.approx(3.0)
    assert len(result["chunks"]) == 6


# --- symbology: legacy resolve_at shim (CompositeSource backward compat) ---


def test_symbol_resolver_resolve_at_resolves_a_single_day_only(tmp_path, monkeypatch):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)

    mapping = {("XNAS.ITCH", "2024-01-05"): {"ABC": "1001"}}
    client = _FakeClient(symbology_mapping=mapping)

    resolver = SymbolResolver("XNAS.ITCH")
    assert resolver.resolve_at("ABC", dt.date(2024, 1, 5), client=client) == "1001"
    # A DIFFERENT day, even for the same symbol, must resolve fresh -- no
    # persisted range-wide interval is consulted.
    assert resolver.resolve_at("ABC", dt.date(2024, 1, 8), client=client) is None


def test_symbol_resolver_resolve_at_returns_none_without_a_client():
    """Matches the historical behavior CompositeSource depends on: no
    client available -> None, no network call attempted."""
    resolver = SymbolResolver("XNAS.ITCH")
    assert resolver.resolve_at("ABC", dt.date(2024, 1, 5)) is None


def test_symbol_resolver_resolve_at_ambiguous_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)

    class _AmbiguousSymbology:
        def resolve(self, **kwargs):
            return {
                "result": {
                    "ABC": [
                        {"d0": kwargs["start_date"], "d1": kwargs["end_date"], "s": "1001"},
                        {"d0": kwargs["start_date"], "d1": kwargs["end_date"], "s": "2002"},
                    ]
                }
            }

    class _Client:
        symbology = _AmbiguousSymbology()

    resolver = SymbolResolver("XNAS.ITCH")
    with pytest.raises(AmbiguousSymbolError):
        resolver.resolve_at("ABC", dt.date(2024, 1, 5), client=_Client())


# --- infer_delistings: keyed on ticker (instrument_id is unsafe now) -------


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


def test_infer_delistings_keys_on_ticker_even_when_instrument_id_changes_daily():
    """`instrument_id` is reassigned DAILY by Databento -- keying
    `infer_delistings` on it (the historical CRSP-style assumption) would
    treat almost every ordinary trading day as a brand-new "instrument"
    and flag nearly everything as delisted. `ticker` (correctly resolved
    per day by `SymbolResolver.resolve_day`) is the stable identity here.
    """
    sessions = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for i, d in enumerate(sessions):
        # SURVIVOR's instrument_id changes EVERY session, exactly like the
        # live-verified LULU/MRNA finding -- but its ticker never does.
        rows.append({"trade_date": d, "ticker": "SURVIVOR", "instrument_id": 1000 + i, "close": 10.0})

    daily_bars = _bars_frame(rows)
    out = infer_delistings(daily_bars, min_gap_days=5)

    # SURVIVOR is present through the final session -- must NOT be flagged,
    # even though every single day carried a different instrument_id.
    assert "SURVIVOR" not in set(out["ticker"])


def test_infer_delistings_does_not_flag_a_short_halt_that_resumes():
    sessions = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for d in sessions:
        rows.append({"trade_date": d, "ticker": "SURVIVOR", "instrument_id": 1, "close": 10.0})
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
    assert row["as_of"] != pd.Timestamp("2024-01-15")
    assert out["confidence"].iloc[0] == 0.8


# --- verify_no_survivorship: loud FAIL on a survivor-only frame ------------


def test_verify_no_survivorship_fails_on_survivor_only_frame():
    sessions = pd.date_range("2018-06-01", periods=400, freq="B")
    rows = []
    for d in sessions:
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            rows.append({"trade_date": d, "ticker": ticker, "close": 100.0})
    daily_bars = _bars_frame(rows)

    report = verify_no_survivorship(daily_bars)

    assert isinstance(report, SurvivorshipReport)
    assert report.passed is False
    assert not report
    assert report.disappeared_tickers == set()


def test_verify_no_survivorship_passes_when_names_genuinely_disappear():
    sessions = pd.date_range("2018-06-01", periods=400, freq="B")
    rows = []
    for d in sessions:
        rows.append({"trade_date": d, "ticker": "SURVIVOR", "close": 100.0})
    for d in sessions[: len(sessions) // 2]:
        rows.append({"trade_date": d, "ticker": "DELISTED_CO", "close": 5.0})

    daily_bars = _bars_frame(rows)
    report = verify_no_survivorship(daily_bars)

    assert report.passed is True
    assert bool(report) is True
    assert "DELISTED_CO" in report.disappeared_tickers
