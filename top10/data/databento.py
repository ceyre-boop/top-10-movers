"""Databento implementation of :class:`~top10.data.base.MarketDataSource`.

Databento's core market-data schemas (OHLCV, trades, etc.) cover bars, but
it has no bulk corporate-actions or earnings-calendar feed comparable to
Polygon's reference endpoints. Per the P2 "silent survivorship trap" rule,
those methods raise ``NotImplementedError`` naming the supplemental source
needed rather than returning an empty frame that would look like "there
were no corporate actions/earnings" to a caller.

No network call and no API key lookup happens at import time or in
``__init__`` -- the key is only resolved the first time a request needs to
go out, and even the ``databento`` SDK client itself is constructed lazily.

COST GUARD: every network path in this class that could incur a paid
Databento request goes through :class:`top10.data.cost_guard.CostGuard`
(see ``_fetch_bars``). There is deliberately no method here that calls
``client.timeseries.get_range`` directly -- that is the whole point, since
the user's $125 promotional credit auto-bills real money once exhausted.
Use ``dry_run=True`` on ``daily_bars``/``premarket_bars`` to see the cost
estimate and record count for a pull before paying for it.

=============================================================================
VERIFIED-AGAINST-LIVE-DATA FINDINGS (see docs/DATA_SOURCES.md for the full
write-up). A LIVE test run against real Databento data found the model of
Databento equities this adapter previously used was wrong in several ways
that silently produced garbage (a median top-10-gainer return of +8702%).
Every finding below is confirmed against real API responses, not theorized.
=============================================================================

1. NO CONSOLIDATED US EQUITIES DATASET EXISTS BEFORE 2023. Verified via
   `metadata.get_dataset_range`: the three LISTING-VENUE feeds --
   `XNAS.ITCH` (Nasdaq), `XNYS.PILLAR` (NYSE), `XASE.PILLAR` (AMEX) -- all
   have `ohlcv-1d` starting **2018-05-01**. Every CONSOLIDATED feed
   (`EQUS.SUMMARY`, `DBEQ.BASIC`, `EQUS.MINI`, `IEXG.TOPS`, `XCIS`, `XCHI`)
   starts in 2023 or later -- entirely inside this project's 2023-01-01
   holdout, useless for training. `LISTING_VENUE_DATASETS` is therefore the
   UNION of the three listing-venue feeds, and `DatabentoSource` takes an
   optional `venues=` list defaulting to exactly those three.

2. PER-VENUE FEEDS ARE PARTIAL: each dataset above contains only trades
   EXECUTED ON that one exchange, not every trade in the name. So:
   - VOLUME: summed across venues for a (date, ticker) -- still an
     UNDERSTATE of true consolidated (SIP) volume, since ARCX/IEXG/MEMX/
     etc. venues are not fetched at all. This directly affects the $1M ADV
     universe filter in LABEL_SPEC (admits fewer names than a SIP-based
     filter would) -- see docs/DATA_SOURCES.md.
   - CLOSE: taken from the LISTING venue's bar ONLY (the official closing
     auction runs on the listing venue) -- never averaged/last-across-
     venues. The listing venue per ticker comes from the `definition`
     schema's `exchange` field (see `_get_listing_venue_map`).

3. `ohlcv-1d` `ts_event` IS UTC MIDNIGHT *OF* THE TRADE DATE -- converting
   it to America/New_York shifts every bar back one day and manufactures
   phantom weekend sessions (verified: tz-converting produced a
   "2022-06-05" row, a Sunday). `_utc_midnight_trade_date` takes the UTC
   date directly, no tz conversion, then `as_of` is stamped at 16:00 ET of
   that date per the P4 point-in-time contract.

4. `to_df(map_symbols=True)` does NOT populate `symbol` for `ALL_SYMBOLS`
   requests (verified: all 30,379 rows null in a live pull). This adapter
   therefore NEVER reads `r["symbol"]` for a daily-bars row -- ticker
   resolution goes exclusively through `SymbolResolver.resolve_day`
   (`top10/data/symbology.py`), which is mandatory, not a fallback.

5. `instrument_id` IS REASSIGNED DAILY -- see `top10/data/symbology.py`
   module docstring for the full write-up (LULU: 6844, 6843, 6839, 6840,
   6841, 6837, 6837 across seven consecutive sessions). This is THE fix:
   ticker resolution must happen per trading day, never range-wide.

6. DATABENTO FLAGS DEGRADED DAYS (verified: a live pull warned
   "2022-07-18 (degraded), 2022-07-25 (degraded)"). `_degraded_dates`
   calls the free `metadata.get_dataset_condition` endpoint and
   `daily_bars` EXCLUDES degraded/pending days from its output by default
   (a degraded day silently produces wrong returns) while recording which
   dates were dropped on `self.last_degraded_dates`.

7. The `definition` schema gives what `ticker_meta`-style consumers need:
   `raw_symbol`, `exchange` (listing venue), `security_type`/
   `instrument_class`. Verified `security_type` values: `C` (common
   stock), `E` (ETF), `W` (warrant), `P` (preferred), `U` (unit), plus
   `Q`/`O`/`I`/`A`/`H`/`M`. `_SECURITY_TYPE_LABELS` maps `C` to "common
   stock" and flags the rest. Definition pulls cost ~$0.065/venue/day, so
   `_get_listing_venue_map` pulls ONE representative day per calendar
   month (never daily) and carries the result forward for that whole
   month -- documented staleness, not a bug: a listing-venue change
   mid-month will misclassify the close venue for the rest of that month.

HISTORY START: Databento's equities history for these three listing-venue
datasets starts in May 2018 (see ``FIRST_AVAILABLE_DATE``). Requesting an
earlier ``start`` raises rather than silently returning a truncated frame,
which would quietly bias every walk-forward result built on it.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from top10.config import ET, get_api_key
from top10.data import _conform
from top10.data.base import CORPORATE_ACTIONS_COLUMNS, DAILY_BARS_COLUMNS, PREMARKET_BARS_COLUMNS
from top10.data.cache import cached_call
from top10.data.cost_guard import CostGuard, estimate_cost
from top10.data.symbology import SymbolResolver

logger = logging.getLogger(__name__)

# The three listing-venue feeds this adapter UNIONS for a full daily
# cross-section -- see finding (1) above. There is no consolidated US
# equities dataset that reaches back before 2023 on Databento.
LISTING_VENUE_DATASETS = ["XNAS.ITCH", "XNYS.PILLAR", "XASE.PILLAR"]

# Kept for backward compatibility ONLY: `top10.data.composite.CompositeSource`
# imports this single-dataset name for its (already-narrow, ticker-string-
# keyed) cross-vendor alignment resolver. `DatabentoSource` itself never
# pulls just this one dataset -- see `LISTING_VENUE_DATASETS`.
DATASET = LISTING_VENUE_DATASETS[0]

# Databento's equities history for all three listing-venue datasets starts
# in May 2018 (verified via `metadata.get_dataset_range`). A request with
# an earlier `start` must be rejected loudly -- see module docstring.
FIRST_AVAILABLE_DATE = dt.date(2018, 5, 1)

# `definition` schema `exchange` field -> the `LISTING_VENUE_DATASETS` entry
# that dataset's `ohlcv-1d` feed corresponds to. ARCX (NYSE Arca) prints are
# not separately pulled by this adapter (see finding 2's volume-understate
# note) and are mapped onto the NYSE feed as the closest listing-venue
# proxy; this is a documented approximation, not a claim of exact coverage.
_EXCHANGE_TO_VENUE_DATASET = {
    "XNAS": "XNAS.ITCH",
    "XNYS": "XNYS.PILLAR",
    "ARCX": "XNYS.PILLAR",
    "XASE": "XASE.PILLAR",
}

# Verified `definition` schema `security_type`/`instrument_class` values
# (finding 7). Only "C" maps to a plain common stock; every other code is
# flagged rather than silently treated as one, per LABEL_SPEC.
_SECURITY_TYPE_LABELS: dict[str, str] = {
    "C": "common_stock",
    "E": "etf",
    "W": "warrant",
    "P": "preferred",
    "U": "unit",
    "Q": "other_q",
    "O": "other_o",
    "I": "other_i",
    "A": "other_a",
    "H": "other_h",
    "M": "other_m",
}


def _month_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Split the INCLUSIVE range ``[start, end]`` into calendar-month
    chunks, also inclusive on both ends of each chunk.

    A full-universe daily pull from 2018-05-01 to today is the single
    largest spend in this project (see module docstring / `CostGuard`).
    Chunking by month, with each chunk going through the existing
    `cached_call` disk cache, gives checkpoint-and-resume for free: a
    mid-pull failure leaves already-fetched months cached on disk, and
    re-running the same `daily_bars(start, end)` call re-fetches only the
    months that never completed, never re-spending on ones that did.
    """
    chunks: list[tuple[dt.date, dt.date]] = []
    cur = start
    while cur <= end:
        if cur.month == 12:
            next_month_start = dt.date(cur.year + 1, 1, 1)
        else:
            next_month_start = dt.date(cur.year, cur.month + 1, 1)
        month_end = next_month_start - dt.timedelta(days=1)
        chunk_end = min(month_end, end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + dt.timedelta(days=1)
    return chunks


def _utc_midnight_trade_date(record: dict[str, Any]) -> pd.Timestamp:
    """`ohlcv-1d` `ts_event` is UTC MIDNIGHT *OF* the trade date -- see
    finding (3) in the module docstring. This deliberately takes the UTC
    calendar date AS-IS and never tz-converts it: converting to
    America/New_York shifts every bar back a day and manufactures phantom
    weekend sessions (verified live: a "2022-06-05" row, a Sunday)."""
    raw = record.get("ts_event") or record.get("index")
    ts = pd.Timestamp(raw)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


class DatabentoSource:
    """Databento adapter. Implements ``MarketDataSource`` structurally."""

    name = "databento"

    def __init__(self, venues: list[str] | None = None) -> None:
        # Union of listing-venue datasets this adapter pulls -- see finding
        # (1). Defaults to all three; a caller may narrow this (e.g. for a
        # cheaper single-venue smoke test) but `daily_bars` is only P2/P3
        # correct against the full union.
        self._venues: list[str] = list(venues) if venues else list(LISTING_VENUE_DATASETS)
        self._api_key: str | None = None
        self._client: Any = None
        self._guard: CostGuard | None = None
        self._resolvers: dict[str, SymbolResolver] = {}
        # "YYYY-MM" -> {raw_symbol: listing_venue_dataset}. Carried forward
        # for the whole month -- see finding (7) / `_get_listing_venue_map`.
        self._listing_venue_cache: dict[str, dict[str, str]] = {}
        # "YYYY-MM" -> {raw_symbol: security_type_label}.
        self._security_type_cache: dict[str, dict[str, str]] = {}
        # Populated by the most recent `daily_bars` call: sorted list of
        # "<venue> <date>" strings for every degraded/pending trading day
        # that was excluded from that call's output -- see finding (6).
        self.last_degraded_dates: list[str] = []

    # -- internals ---------------------------------------------------------

    def _key(self) -> str:
        if self._api_key is None:
            self._api_key = get_api_key("databento")
        return self._api_key

    def _get_client(self) -> Any:
        if self._client is None:
            import databento as db

            self._client = db.Historical(self._key())
        return self._client

    def _cost_guard(self) -> CostGuard:
        if self._guard is None:
            self._guard = CostGuard()
        return self._guard

    def _resolver_for(self, dataset: str) -> SymbolResolver:
        if dataset not in self._resolvers:
            self._resolvers[dataset] = SymbolResolver(dataset)
        return self._resolvers[dataset]

    @staticmethod
    def _require_start_after_history_begins(start: dt.date) -> None:
        if start < FIRST_AVAILABLE_DATE:
            raise ValueError(
                f"requested start {start.isoformat()} is before Databento's "
                f"actual first available date for the listing-venue equities "
                f"history this adapter unions ({', '.join(LISTING_VENUE_DATASETS)}) "
                f"({FIRST_AVAILABLE_DATE.isoformat()}). No consolidated US "
                "equities dataset on Databento reaches back before 2023 -- "
                "see the module docstring finding (1). Requesting an earlier "
                "start would silently return a shorter history than asked "
                "for, quietly biasing every walk-forward result built on it "
                f"-- pick a start on or after {FIRST_AVAILABLE_DATE.isoformat()}."
            )

    def _dry_run_result(
        self, client: Any, dataset: str, schema: str, start: str, end: str, symbols: str
    ) -> dict[str, Any]:
        """Cost estimate + record count WITHOUT downloading any data.

        Calls Databento's free metadata endpoints only (`metadata.get_cost`,
        `metadata.get_record_count`) -- no `timeseries.get_range` call, no
        cache write, and no entry in the spend ledger, since nothing was
        actually paid for.
        """
        request_params = dict(
            dataset=dataset,
            schema=schema,
            symbols=symbols,
            stype_in="raw_symbol",
            start=start,
            end=end,
        )
        cost = estimate_cost(client, **request_params)
        record_count = int(client.metadata.get_record_count(**request_params))
        return {
            "dry_run": True,
            "dataset": dataset,
            "schema": schema,
            "symbols": symbols,
            "start": start,
            "end": end,
            "cost_usd": cost,
            "record_count": record_count,
        }

    def _fetch_bars(
        self,
        dataset: str,
        schema: str,
        start: str,
        end: str,
        symbols: str = "ALL_SYMBOLS",
        *,
        confirm: bool = False,
    ) -> list[dict]:
        """The ONLY method in this class that may call
        ``client.timeseries.get_range`` -- always routed through
        :class:`CostGuard`. Estimates cost via Databento's own
        ``metadata.get_cost()`` before ever requesting data, refuses when
        the estimate would breach the budget ceiling or (unconfirmed)
        the per-request confirmation threshold, and records actual spend
        to the persisted ledger only after a successful fetch.
        """
        client = self._get_client()
        request_params = dict(
            dataset=dataset,
            schema=schema,
            symbols=symbols,
            stype_in="raw_symbol",
            start=start,
            end=end,
        )
        cost = estimate_cost(client, **request_params)
        description = f"{dataset} {schema} {symbols} {start}->{end}"

        def _do_fetch() -> list[dict]:
            store = client.timeseries.get_range(**request_params)
            # Convert to plain records so the cache layer can persist raw
            # JSON, same as every other vendor's payload. `map_symbols` is
            # deliberately NOT requested here -- finding (4): it does not
            # populate `symbol` for `ALL_SYMBOLS` requests anyway, so this
            # adapter never reads `record["symbol"]` for daily bars.
            return store.to_df().reset_index().to_dict("records")

        return self._cost_guard().guarded_request(
            _do_fetch, cost, description, confirm=confirm
        )

    def _degraded_dates(self, dataset: str, start: dt.date, end: dt.date) -> set[str]:
        """Free metadata call (never through `CostGuard`): every date in
        `[start, end]` Databento itself flags as degraded/pending for
        `dataset` -- finding (6). A degraded day silently produces wrong
        returns, so `daily_bars` excludes these dates by default."""
        client = self._get_client()
        try:
            condition = client.metadata.get_dataset_condition(
                dataset=dataset, start_date=start.isoformat(), end_date=end.isoformat()
            )
        except AttributeError:  # pragma: no cover - defensive, real SDK has this
            return set()
        degraded: set[str] = set()
        for entry in condition or []:
            state = entry.get("condition")
            date = entry.get("date")
            if date and state in ("degraded", "pending"):
                degraded.add(str(date))
        return degraded

    def _get_listing_venue_map(
        self, month_start: dt.date, *, confirm: bool = False
    ) -> tuple[dict[str, str], dict[str, str]]:
        """`(raw_symbol -> listing_venue_dataset, raw_symbol ->
        security_type_label)` for the calendar month containing
        `month_start`, via the `definition` schema's `exchange` and
        `instrument_class`/`security_type` fields -- finding (7).

        Pulled for ONE representative day (the first calendar day of the
        month) per venue, per month -- never daily, since `definition`
        pulls cost ~$0.065/venue/day and this adapter's own history spans
        years. The result is carried forward for the WHOLE month. This is
        a documented staleness tradeoff: a listing-venue change mid-month
        will misclassify that ticker's close-venue (and therefore its
        `close` price) for the rest of that month.
        """
        key = f"{month_start.year:04d}-{month_start.month:02d}"
        if key in self._listing_venue_cache:
            return self._listing_venue_cache[key], self._security_type_cache[key]

        day_start = dt.date(month_start.year, month_start.month, 1)
        day_end = day_start + dt.timedelta(days=1)

        venue_map: dict[str, str] = {}
        type_map: dict[str, str] = {}
        for venue in self._venues:

            def _fetch(venue: str = venue) -> list[dict]:
                return self._fetch_bars(
                    venue,
                    "definition",
                    day_start.isoformat(),
                    day_end.isoformat(),
                    symbols="ALL_SYMBOLS",
                    confirm=confirm,
                )

            cache_key = f"{venue}_{key}"
            records = cached_call(f"{self.name}/definitions", cache_key, _fetch)
            for r in records or []:
                raw_symbol = r.get("raw_symbol")
                if raw_symbol is None:
                    continue
                exchange = r.get("exchange")
                venue_map[raw_symbol] = _EXCHANGE_TO_VENUE_DATASET.get(exchange, venue)
                type_code = r.get("instrument_class") or r.get("security_type")
                type_map[raw_symbol] = _SECURITY_TYPE_LABELS.get(type_code, f"unknown_{type_code}")

        self._listing_venue_cache[key] = venue_map
        self._security_type_cache[key] = type_map
        return venue_map, type_map

    def resolve_symbols(self, tickers: list[str], as_of: dt.date) -> dict[str, str]:
        """Point-in-time raw_symbol resolution, INCLUDING delisted/reused
        tickers -- NOT IMPLEMENTED.

        Databento's `client.symbology.resolve()` (stype_in="raw_symbol",
        stype_out="instrument_id", pinned to a single historical day --
        see `top10/data/symbology.py`) can in principle correctly resolve a
        raw ticker symbol as of a specific past date even when that same
        symbol string was later reused by a different, unrelated issuer.
        That resolution + collision-disambiguation logic has not been
        implemented here.

        `daily_bars` (via ``symbols="ALL_SYMBOLS"``) is unaffected: an
        ALL_SYMBOLS request returns every instrument that traded in range,
        delisted names included, with no symbol-string ambiguity -- it
        resolves via `SymbolResolver.resolve_day` (instrument_id ->
        symbol), never this method. `premarket_bars`, however, is given a
        caller-supplied list of raw ticker strings and passes them straight
        through to `stype_in="raw_symbol"` without this resolution step --
        correct only for symbols that were never reused by a different
        issuer. Do not treat `premarket_bars` results for a reused/renamed
        ticker as reliable until this is implemented.
        """
        raise NotImplementedError(
            "Databento point-in-time symbol resolution "
            "(client.symbology.resolve() for raw_symbol -> instrument_id as "
            "of a historical date, including delisted/reused tickers) is not "
            "implemented. premarket_bars() passes raw ticker strings straight "
            "through to stype_in='raw_symbol', which is only safe for "
            "symbols that were never reused by a different, unrelated issuer."
        )

    # -- MarketDataSource ----------------------------------------------------

    def daily_bars(
        self,
        start: dt.date,
        end: dt.date,
        *,
        dry_run: bool = False,
        confirm: bool = False,
    ) -> pd.DataFrame | dict[str, Any]:
        """UNADJUSTED daily OHLCV, unioned across `self._venues` (the three
        listing-venue datasets by default -- finding 1), with a mandatory
        per-trading-day symbol resolution pass (finding 5) and degraded-day
        exclusion (finding 6).

        P2 DEFENSE (THE core survivorship-bias fix): this pulls the FULL
        daily cross-section -- every instrument that traded, delisted names
        included -- for each requested trading day, via
        ``symbols="ALL_SYMBOLS"`` on every venue. It deliberately does NOT
        resolve a ticker list first (e.g. "today's active symbols") and
        then fetch history for those names; that inversion is exactly how
        survivorship bias enters a dataset. Do NOT "optimize" this into a
        per-symbol pull over a pre-resolved list. See
        :func:`verify_no_survivorship` for the loud check that this
        invariant is actually holding for a given pulled frame.

        `instrument_id` is carried through as an extra column beyond the
        frozen `DAILY_BARS_COLUMNS` contract -- but see finding (5): unlike
        CRSP's `permno`, Databento's `instrument_id` is REASSIGNED DAILY
        and is therefore INFORMATIONAL ONLY (the id that happened to be
        bound to this row's ticker on this one trade_date). It is NOT safe
        to `groupby`/join across more than one trading day -- `ticker` (now
        correctly, per-day resolved via `SymbolResolver.resolve_day`) is
        the safe cross-day identity in this frame, not `instrument_id`.

        VOLUME is summed across every venue's bar for a (trade_date,
        ticker); CLOSE (and open/high/low) is taken from the LISTING
        venue's bar only, per `_get_listing_venue_map` -- never averaged or
        "last one wins" across venues (finding 2). A ticker whose listing
        venue could not be determined for its month falls back to
        whichever venue's bar happened to resolve first for that
        (day, ticker), which is a documented approximation, not a claim of
        exact closing-auction accuracy.

        The pull is chunked by calendar month and each chunk (per venue) is
        independently disk-cached (see `_month_chunks`), which makes a
        multi-year pull resumable. Cumulative Databento spend is logged
        after every chunk.

        ``dry_run=True`` returns a dict with the cost estimate and record
        count for the WHOLE `[start, end]` range across every venue,
        WITHOUT downloading -- see :meth:`_dry_run_result`. Use
        :meth:`estimate_universe_pull_cost` for a per-month, per-venue cost
        breakdown that mirrors how the pull is actually chunked.
        ``confirm=True`` is required to proceed with any single chunk's
        request estimated above the guard's confirmation threshold
        (default $5).
        """
        self._require_start_after_history_begins(start)

        if dry_run:
            client = self._get_client()
            per_venue = [
                self._dry_run_result(
                    client, venue, "ohlcv-1d", start.isoformat(), end.isoformat(), "ALL_SYMBOLS"
                )
                for venue in self._venues
            ]
            return {
                "dry_run": True,
                "venues": per_venue,
                "cost_usd": sum(v["cost_usd"] for v in per_venue),
                "record_count": sum(v["record_count"] for v in per_venue),
                "start": start.isoformat(),
                "end": end.isoformat(),
            }

        guard = self._cost_guard()
        # (trade_date, venue) -> raw records fetched for that venue+chunk.
        by_day_venue: dict[tuple[pd.Timestamp, str], list[dict]] = {}
        degraded: set[str] = set()

        for chunk_start, chunk_end in _month_chunks(start, end):
            for venue in self._venues:
                key = f"{venue}_{chunk_start.isoformat()}_{chunk_end.isoformat()}"

                def _fetch(
                    venue: str = venue, chunk_start: dt.date = chunk_start, chunk_end: dt.date = chunk_end
                ) -> list[dict]:
                    # P2 DEFENSE: ALL_SYMBOLS, never a resolved ticker list.
                    return self._fetch_bars(
                        venue,
                        "ohlcv-1d",
                        chunk_start.isoformat(),
                        chunk_end.isoformat(),
                        symbols="ALL_SYMBOLS",
                        confirm=confirm,
                    )

                records = cached_call(f"{self.name}/daily_bars", key, _fetch)
                logger.info(
                    "databento daily_bars chunk %s %s -> %s: cumulative spend $%.2f / $%.2f ceiling",
                    venue, chunk_start.isoformat(), chunk_end.isoformat(), guard.spent, guard.ceiling_usd,
                )

                for r in records or []:
                    trade_date = _utc_midnight_trade_date(r)
                    by_day_venue.setdefault((trade_date, venue), []).append(r)

            degraded |= {
                f"{venue} {d}"
                for venue in self._venues
                for d in self._degraded_dates(venue, chunk_start, chunk_end)
            }

        self.last_degraded_dates = sorted(degraded)
        degraded_dates_only = {entry.split(" ", 1)[1] for entry in degraded}

        # Per (trade_date, venue): resolve instrument_id -> ticker for
        # EXACTLY that day (finding 5) -- never range-wide.
        client = self._get_client()
        resolved: dict[tuple[pd.Timestamp, str, str], str] = {}
        for (trade_date, venue), recs in by_day_venue.items():
            ids = [r.get("instrument_id") for r in recs if r.get("instrument_id") is not None]
            mapping = self._resolver_for(venue).resolve_day(
                trade_date.date(), ids, client=client
            )
            for instrument_id, ticker in mapping.items():
                resolved[(trade_date, venue, instrument_id)] = ticker

        # Aggregate across venues: sum volume, keep every venue's own bar
        # so the listing venue's close can be picked per ticker/day.
        agg: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
        for (trade_date, venue), recs in by_day_venue.items():
            if trade_date.date().isoformat() in degraded_dates_only:
                continue  # finding (6): exclude degraded days entirely
            for r in recs:
                instrument_id = str(r.get("instrument_id"))
                ticker = resolved.get((trade_date, venue, instrument_id))
                if ticker is None:
                    continue  # unresolved for this exact day -- dropped, not guessed
                bucket = agg.setdefault(
                    (trade_date, ticker),
                    {"volume": 0.0, "bars_by_venue": {}, "instrument_id": instrument_id},
                )
                bucket["volume"] += float(r.get("volume") or 0.0)
                bucket["bars_by_venue"][venue] = r

        venue_map, _ = ({}, {})
        rows: list[dict[str, Any]] = []
        current_month_key: str | None = None
        for (trade_date, ticker), bucket in agg.items():
            month_key = f"{trade_date.year:04d}-{trade_date.month:02d}"
            if month_key != current_month_key:
                venue_map, _ = self._get_listing_venue_map(trade_date.date(), confirm=confirm)
                current_month_key = month_key

            listing_venue = venue_map.get(ticker)
            close_bar = bucket["bars_by_venue"].get(listing_venue) if listing_venue else None
            if close_bar is None:
                # Listing venue unknown/stale for this ticker -- fall back
                # to whichever venue's bar is available (documented
                # approximation, see method docstring).
                close_bar = next(iter(bucket["bars_by_venue"].values()))

            close = float(close_bar.get("close", 0.0))
            volume = bucket["volume"]
            as_of_ts = trade_date + pd.Timedelta(hours=16)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "open": close_bar.get("open"),
                    "high": close_bar.get("high"),
                    "low": close_bar.get("low"),
                    "close": close,
                    "volume": volume,
                    "dollar_volume": close * volume,
                    "as_of": as_of_ts,
                    "instrument_id": bucket["instrument_id"],
                }
            )

        df = pd.DataFrame(rows, columns=DAILY_BARS_COLUMNS + ["instrument_id"])
        conformed = _conform(df, DAILY_BARS_COLUMNS)
        conformed["instrument_id"] = (
            df["instrument_id"].values if not df.empty else pd.Series(dtype="object")
        )
        return conformed

    def estimate_universe_pull_cost(self, start: dt.date, end: dt.date) -> dict[str, Any]:
        """Total USD cost estimate for a full ``daily_bars(start, end)``
        cross-section pull ACROSS EVERY VENUE, WITHOUT downloading any data.

        Calls Databento's free `metadata.get_cost()` once per (venue,
        calendar-month chunk) pair (mirroring exactly how `daily_bars`
        itself chunks the pull -- see `_month_chunks`), never
        `timeseries.get_range`. This is the number the user should see
        BEFORE committing to a multi-year full-universe pull, which is the
        single largest spend this project can make.
        """
        self._require_start_after_history_begins(start)
        client = self._get_client()

        chunks: list[dict[str, Any]] = []
        total = 0.0
        for chunk_start, chunk_end in _month_chunks(start, end):
            for venue in self._venues:
                request_params = dict(
                    dataset=venue,
                    schema="ohlcv-1d",
                    symbols="ALL_SYMBOLS",
                    stype_in="raw_symbol",
                    start=chunk_start.isoformat(),
                    end=chunk_end.isoformat(),
                )
                cost = estimate_cost(client, **request_params)
                chunks.append(
                    {
                        "venue": venue,
                        "start": chunk_start.isoformat(),
                        "end": chunk_end.isoformat(),
                        "cost_usd": cost,
                    }
                )
                total += cost

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "total_cost_usd": total,
            "chunks": chunks,
        }

    def corporate_actions(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Databento still has no splits/dividends/ticker-change feed, so
        this method continues to raise -- see the message below for the
        reference-data vendor to use instead.

        Delistings ARE now inferrable from `daily_bars` alone (no separate
        feed needed) via the module-level :func:`infer_delistings` +
        :func:`delistings_to_corporate_actions` helpers, which produce
        `action_type="delisting"` rows shaped exactly like
        `CORPORATE_ACTIONS_COLUMNS`. That pipeline is deliberately NOT
        wired into this method: `corporate_actions(start, end)` has no
        `daily_bars` parameter, and calling `daily_bars` implicitly from
        here would be an unguarded-looking paid full-universe pull hidden
        inside what looks like a free metadata call -- the caller must
        fetch `daily_bars` explicitly (through the normal cost-guarded
        path) and pass it to `infer_delistings` themselves.
        """
        raise NotImplementedError(
            "Databento has no corporate-actions feed (splits/dividends/ticker "
            "changes). Use a reference-data vendor for this method, e.g. "
            "top10.data.polygon.PolygonSource.corporate_actions. Delistings "
            "specifically ARE inferrable from daily_bars -- see "
            "top10.data.databento.infer_delistings and "
            "top10.data.databento.delistings_to_corporate_actions."
        )

    def ticker_meta(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError(
            "Databento has no point-in-time listing-metadata feed (security "
            "type / exchange / active_from / active_to). Use a reference-data "
            "vendor for this method, e.g. top10.data.polygon.PolygonSource.ticker_meta. "
            "Note: DatabentoSource._get_listing_venue_map does expose the "
            "`definition` schema's exchange/security_type fields for its own "
            "internal daily_bars close-venue resolution, but that is a "
            "monthly-snapshot, carried-forward, non-point-in-time view -- not "
            "a substitute for this method's contract."
        )

    def earnings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError(
            "Databento has no earnings-calendar feed. Use a supplemental "
            "earnings-calendar source, e.g. top10.data.polygon.PolygonSource.earnings."
        )

    def short_interest(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError(
            "Databento has no FINRA short-interest feed. Use a supplemental "
            "short-interest source, e.g. top10.data.polygon.PolygonSource.short_interest."
        )

    def premarket_bars(
        self,
        trade_date: dt.date,
        tickers: list[str],
        *,
        venue: str | None = None,
        dry_run: bool = False,
        confirm: bool = False,
    ) -> pd.DataFrame | dict[str, Any]:
        """04:00-09:25 ET minute bars via the ``ohlcv-1m`` schema, on a
        SINGLE listing-venue dataset (`venue`, default `DATASET` ==
        `"XNAS.ITCH"`) -- unlike `daily_bars`, this is not unioned across
        all three venues, so it inherits the same partial-tape caveat as
        any one of them (finding 2): a name that trades premarket mostly
        on a different venue will be undercounted here.

        CAUTION: `tickers` are passed straight through to
        ``stype_in="raw_symbol"`` with no point-in-time resolution -- see
        :meth:`resolve_symbols` for exactly what is missing (safe only for
        symbols never reused by a different issuer).

        ``dry_run=True`` returns a dict with the cost estimate and record
        count WITHOUT downloading. ``confirm=True`` is required to proceed
        with any single request estimated above the guard's confirmation
        threshold (default $5).
        """
        self._require_start_after_history_begins(trade_date)
        venue = venue or DATASET

        d_str = trade_date.isoformat()
        window_start = pd.Timestamp(f"{d_str} 04:00", tz=ET)
        window_end = pd.Timestamp(f"{d_str} 09:25", tz=ET)
        symbols = ",".join(tickers)

        if dry_run:
            client = self._get_client()
            return self._dry_run_result(client, venue, "ohlcv-1m", d_str, d_str, symbols)

        def _fetch() -> list[dict]:
            return self._fetch_bars(venue, "ohlcv-1m", d_str, d_str, symbols=symbols, confirm=confirm)

        records = cached_call(f"{self.name}/premarket_bars", f"{venue}_{symbols}_{d_str}", _fetch)

        rows: list[dict[str, Any]] = []
        for r in records or []:
            minute_utc = pd.Timestamp(r.get("ts_event") or r.get("index"), tz="UTC")
            minute_et = minute_utc.tz_convert(ET)
            # `window_end` (09:25) EXCLUSIVE -- see PolygonSource.premarket_bars.
            if not (window_start <= minute_et < window_end):
                continue
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "ticker": r.get("symbol"),
                    "minute": minute_et.tz_localize(None),
                    "open": r.get("open"),
                    "high": r.get("high"),
                    "low": r.get("low"),
                    "close": r.get("close"),
                    "volume": r.get("volume"),
                    "trade_count": r.get("count"),
                    "as_of": minute_et.tz_localize(None),
                }
            )

        df = pd.DataFrame(rows, columns=PREMARKET_BARS_COLUMNS)
        return _conform(df, PREMARKET_BARS_COLUMNS)


# --- Delisting inference (P2: no CRSP-style delisting feed exists here) ----


INFER_DELISTINGS_COLUMNS = [
    "ticker",
    "instrument_id",
    "last_trade_date",
    "inferred_delist_date",
    "confidence",
]


def infer_delistings(daily_bars: pd.DataFrame, min_gap_days: int = 5) -> pd.DataFrame:
    """Derive delistings from a `daily_bars` frame, since Databento has no
    CRSP-style delisting-event table (`crsp.dsedelist`) to query.

    THIS IS INFERRED, NOT AUTHORITATIVE. An identity that trades
    continuously and then stops appearing, while the broader market (every
    OTHER identity in `daily_bars`) keeps trading, is flagged as a
    candidate delisting -- but a trading halt, a multi-day data outage on
    Databento's end, or a symbol rename onto a NEW identity for the same
    underlying company can each masquerade as a delisting under this
    heuristic just as easily as a real one.

    KEYED ON `ticker`, ALWAYS -- NOT `instrument_id`. `instrument_id` is
    reassigned DAILY by Databento (see `top10/data/symbology.py` /
    `DatabentoSource.daily_bars` finding 5); grouping on it would treat
    almost every ordinary trading day as a fresh "instrument" and flag
    nearly everything as delisted. `ticker`, by contrast, IS the correctly
    resolved, stable identity in a `daily_bars` frame produced by this
    module's own `SymbolResolver.resolve_day` pipeline -- this is the
    opposite of the historical CRSP-style advice ("prefer the stable id
    over the ticker string") precisely because Databento's id is not
    stable and its ticker (post-fix) now is.

    An identity is flagged when its last observed `trade_date` across the
    WHOLE `daily_bars` frame is more than `min_gap_days` market sessions
    before the LAST session present anywhere in the frame, AND it never
    reappears before that final session. A short gap that closes again is
    treated as a halt, not a delisting, and is excluded entirely.

    `inferred_delist_date` is the first market session STRICTLY AFTER the
    ticker's last trade -- i.e. the earliest date on which the ticker's
    absence actually became observable.

    `confidence` scales with how many market sessions have elapsed with no
    reappearance by the end of the frame.
    """
    if daily_bars.empty:
        return pd.DataFrame(columns=INFER_DELISTINGS_COLUMNS)

    key_col = "ticker"

    sessions = sorted(pd.Index(daily_bars["trade_date"].unique()))
    if len(sessions) < 2:
        # A single-session frame has no "later sessions" to be absent from
        # -- nothing is inferrable.
        return pd.DataFrame(columns=INFER_DELISTINGS_COLUMNS)
    last_session = sessions[-1]
    session_pos = {d: i for i, d in enumerate(sessions)}

    has_instrument_id = "instrument_id" in daily_bars.columns

    rows: list[dict[str, Any]] = []
    for key, group in daily_bars.groupby(key_col):
        last_trade_date = group["trade_date"].max()
        if last_trade_date == last_session:
            continue  # still present at the end of the window -- not a candidate

        gap_sessions = len(sessions) - 1 - session_pos[last_trade_date]
        if gap_sessions < min_gap_days:
            continue  # short gap -- more consistent with a halt than a delisting

        # `instrument_id` here is INFORMATIONAL ONLY (this ticker's id on
        # its last trade_date, not a stable identifier -- see finding 5)
        # and carried through purely for human debugging, never for a join.
        last_row = group.loc[group["trade_date"] == last_trade_date]
        instrument_id = last_row["instrument_id"].iloc[0] if has_instrument_id else None
        next_pos = session_pos[last_trade_date] + 1
        inferred_delist_date = sessions[next_pos] if next_pos < len(sessions) else last_trade_date

        # Monotonic, capped confidence: reaches 1.0 once the gap is 4x
        # `min_gap_days` or more. The 4x multiplier is a deliberately
        # conservative, documented choice (not tuned/backtested) -- treat
        # this as a coarse signal for triage, not a calibrated probability.
        confidence = min(1.0, gap_sessions / (min_gap_days * 4))

        rows.append(
            {
                "ticker": key,
                "instrument_id": instrument_id,
                "last_trade_date": last_trade_date,
                "inferred_delist_date": inferred_delist_date,
                "confidence": confidence,
            }
        )

    return pd.DataFrame(rows, columns=INFER_DELISTINGS_COLUMNS)


def delistings_to_corporate_actions(delistings: pd.DataFrame) -> pd.DataFrame:
    """Reshape :func:`infer_delistings` output into `CORPORATE_ACTIONS_COLUMNS`
    rows with `action_type="delisting"` -- the same value `CRSPSource.
    corporate_actions` already established for a real, authoritative
    delisting event (`crsp.dsedelist`).

    `as_of` is set to `inferred_delist_date` (the date the absence first
    became OBSERVABLE), never `last_trade_date` -- backdating `as_of` to
    the last trade would make the row look knowable before the absence
    that triggered the inference had actually happened, which is exactly
    the kind of retroactive knowledge P4/point-in-time filtering exists to
    prevent. `ratio`/`cash_amount`/`new_ticker` are always NaN/None:
    Databento gives no delisting-return or reason-code analog to CRSP's
    `dlret`/`dlstcd`. `confidence` is carried through as an extra column
    beyond the frozen contract (same pattern as CRSP's `permno`) so a
    consumer can filter out low-confidence inferences before treating a
    row as a real delisting.
    """
    if delistings.empty:
        df = pd.DataFrame(columns=CORPORATE_ACTIONS_COLUMNS)
        df["confidence"] = pd.Series(dtype="float64")
        df["instrument_id"] = pd.Series(dtype="object")
        return df

    out = pd.DataFrame(
        {
            "ex_date": delistings["inferred_delist_date"],
            "ticker": delistings["ticker"],
            "action_type": "delisting",
            "ratio": float("nan"),
            "cash_amount": float("nan"),
            "new_ticker": None,
            "as_of": delistings["inferred_delist_date"],
        }
    )
    conformed = _conform(out, CORPORATE_ACTIONS_COLUMNS)
    conformed["confidence"] = delistings["confidence"].values
    conformed["instrument_id"] = delistings["instrument_id"].values
    return conformed


# --- Survivorship verification (the P2 kill-criterion check) ---------------


@dataclass
class SurvivorshipReport:
    """Result of :func:`verify_no_survivorship`.

    `passed=False` means the survivorship-bias defense is NOT holding for
    the checked frame -- either because the check found zero names that
    disappear between the early and late windows (the survivorship
    signature itself: a real US equity cross-section loses hundreds of
    names a year), or because the frame was too small to check at all.
    """

    passed: bool
    message: str
    early_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    late_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    early_tickers: set[str] = field(default_factory=set)
    late_tickers: set[str] = field(default_factory=set)
    disappeared_tickers: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        return self.passed


def verify_no_survivorship(
    daily_bars: pd.DataFrame,
    *,
    early_frac: float = 0.1,
    late_frac: float = 0.1,
    min_disappeared: int = 1,
) -> SurvivorshipReport:
    """Verify that `daily_bars` actually exhibits survivorship (i.e. is
    NOT survivorship-biased): pick tickers present in the EARLY window and
    check they are ABSENT by the LATE window.

    This is the exact check the plan's P2 kill criterion demands and is
    otherwise invisible: a survivorship-biased pull (built from today's
    ticker list rather than each day's true cross-section) looks
    completely normal in every other respect -- it has plausible prices,
    plausible volumes, a plausible date range. The only observable symptom
    is that its cast of tickers barely changes over time.

    If ZERO tickers present early disappear by the end of the window
    across a genuinely multi-year span, THAT ABSENCE IS ITSELF THE
    SURVIVORSHIP SIGNATURE -- a real US equity cross-section loses
    hundreds of names a year to delisting, M&A, bankruptcy, etc. This
    function therefore treats "nothing disappeared" as a loud FAIL
    (`passed=False`), never as a quiet pass-by-default.

    `early_frac`/`late_frac` define the early/late windows as the first/
    last `frac` of the DISTINCT trading sessions present in `daily_bars`
    (not calendar time), so the check is meaningful regardless of the
    actual date range passed in.
    """
    if daily_bars.empty:
        return SurvivorshipReport(
            passed=False,
            message="verify_no_survivorship: daily_bars is empty -- nothing to verify.",
        )

    sessions = sorted(pd.Index(daily_bars["trade_date"].unique()))
    if len(sessions) < 2:
        return SurvivorshipReport(
            passed=False,
            message=(
                f"verify_no_survivorship: only {len(sessions)} distinct trading "
                "session(s) in daily_bars -- too small a window to check "
                "survivorship at all."
            ),
        )

    n = len(sessions)
    early_end_idx = max(0, int(n * early_frac) - 1)
    late_start_idx = min(n - 1, int(n * (1 - late_frac)))

    early_start, early_end = sessions[0], sessions[early_end_idx]
    late_start, late_end = sessions[late_start_idx], sessions[-1]

    early_window_df = daily_bars[
        (daily_bars["trade_date"] >= early_start) & (daily_bars["trade_date"] <= early_end)
    ]
    late_window_df = daily_bars[
        (daily_bars["trade_date"] >= late_start) & (daily_bars["trade_date"] <= late_end)
    ]

    early_tickers = set(early_window_df["ticker"].unique())
    late_tickers = set(late_window_df["ticker"].unique())
    disappeared = early_tickers - late_tickers

    passed = len(disappeared) >= min_disappeared

    if passed:
        message = (
            f"verify_no_survivorship PASSED: {len(disappeared)} of "
            f"{len(early_tickers)} early-window ticker(s) "
            f"({early_start.date()}..{early_end.date()}) are absent from the "
            f"late window ({late_start.date()}..{late_end.date()}) -- the "
            "frame is exhibiting genuine survivorship (names leaving the "
            "universe), consistent with a true daily cross-section."
        )
    else:
        message = (
            "verify_no_survivorship FAILED: 0 early-window ticker(s) "
            f"({early_start.date()}..{early_end.date()}, {len(early_tickers)} "
            f"names) disappeared by the late window "
            f"({late_start.date()}..{late_end.date()}). A real US equity "
            "cross-section loses hundreds of names a year to delisting, "
            "M&A, and bankruptcy -- zero disappearances over this window is "
            "the survivorship-bias signature itself (this frame was likely "
            "built from a single point-in-time ticker list, not each day's "
            "true cross-section). DO NOT use this frame for backtesting."
        )
        logger.error(message)

    return SurvivorshipReport(
        passed=passed,
        message=message,
        early_window=(early_start, early_end),
        late_window=(late_start, late_end),
        early_tickers=early_tickers,
        late_tickers=late_tickers,
        disappeared_tickers=disappeared,
    )
