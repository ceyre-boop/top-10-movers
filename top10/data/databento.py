"""Databento implementation of :class:`~top10.data.base.MarketDataSource`.

Databento's core market-data schemas (OHLCV, trades, etc.) cover bars, but
it has no bulk corporate-actions, listing-metadata, or earnings-calendar
feed comparable to Polygon's reference endpoints. Per the P2 "silent
survivorship trap" rule, those methods raise ``NotImplementedError`` naming
the supplemental source needed rather than returning an empty frame that
would look like "there were no corporate actions/earnings" to a caller.

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

HISTORY START: Databento's equities history for this dataset starts in
May 2018 (see ``FIRST_AVAILABLE_DATE``). Requesting an earlier ``start``
raises rather than silently returning a truncated frame, which would
quietly bias every walk-forward result built on it.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from top10.config import ET, get_api_key
from top10.data import _conform
from top10.data.base import DAILY_BARS_COLUMNS, PREMARKET_BARS_COLUMNS
from top10.data.cache import cached_call
from top10.data.cost_guard import CostGuard, estimate_cost

DATASET = "XNAS.ITCH"

# Databento's equities history for XNAS.ITCH starts in May 2018. A request
# with an earlier `start` must be rejected loudly -- see module docstring.
FIRST_AVAILABLE_DATE = dt.date(2018, 5, 1)


class DatabentoSource:
    """Databento adapter. Implements ``MarketDataSource`` structurally."""

    name = "databento"

    def __init__(self) -> None:
        self._api_key: str | None = None
        self._client: Any = None
        self._guard: CostGuard | None = None

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

    @staticmethod
    def _require_start_after_history_begins(start: dt.date) -> None:
        if start < FIRST_AVAILABLE_DATE:
            raise ValueError(
                f"requested start {start.isoformat()} is before Databento's "
                f"actual first available date for {DATASET} equities history "
                f"({FIRST_AVAILABLE_DATE.isoformat()}). Requesting an earlier "
                "start would silently return a shorter history than asked "
                "for, quietly biasing every walk-forward result built on it "
                "-- pick a start on or after "
                f"{FIRST_AVAILABLE_DATE.isoformat()}."
            )

    def _dry_run_result(
        self, client: Any, schema: str, start: str, end: str, symbols: str
    ) -> dict[str, Any]:
        """Cost estimate + record count WITHOUT downloading any data.

        Calls Databento's free metadata endpoints only (`metadata.get_cost`,
        `metadata.get_record_count`) -- no `timeseries.get_range` call, no
        cache write, and no entry in the spend ledger, since nothing was
        actually paid for.
        """
        request_params = dict(
            dataset=DATASET,
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
            "dataset": DATASET,
            "schema": schema,
            "symbols": symbols,
            "start": start,
            "end": end,
            "cost_usd": cost,
            "record_count": record_count,
        }

    def _fetch_bars(
        self,
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
            dataset=DATASET,
            schema=schema,
            symbols=symbols,
            stype_in="raw_symbol",
            start=start,
            end=end,
        )
        cost = estimate_cost(client, **request_params)
        description = f"{schema} {symbols} {start}->{end}"

        def _do_fetch() -> list[dict]:
            store = client.timeseries.get_range(**request_params)
            # Convert to plain records so the cache layer can persist raw
            # JSON, same as every other vendor's payload.
            return store.to_df().reset_index().to_dict("records")

        return self._cost_guard().guarded_request(
            _do_fetch, cost, description, confirm=confirm
        )

    def resolve_symbols(self, tickers: list[str], as_of: dt.date) -> dict[str, str]:
        """Point-in-time raw_symbol resolution, INCLUDING delisted/reused
        tickers -- NOT IMPLEMENTED.

        Databento's `client.symbology.resolve()` (stype_in="raw_symbol",
        stype_out="instrument_id", pinned to a historical date range) can
        in principle correctly resolve a raw ticker symbol as of a specific
        past date even when that same symbol string was later reused by a
        different, unrelated issuer. That resolution + collision-
        disambiguation logic has not been implemented here.

        `daily_bars` (via ``symbols="ALL_SYMBOLS"``) is unaffected: an
        ALL_SYMBOLS request returns every instrument that traded in range,
        delisted names included, with no symbol-string ambiguity.
        `premarket_bars`, however, is given a caller-supplied list of raw
        ticker strings and passes them straight through to
        `stype_in="raw_symbol"` without this resolution step -- correct
        only for symbols that were never reused by a different issuer. Do
        not treat `premarket_bars` results for a reused/renamed ticker as
        reliable until this is implemented.
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
        """UNADJUSTED daily OHLCV via the ``ohlcv-1d`` schema.

        P2 NOTE: Databento's OHLCV schemas are unadjusted by construction
        and include every symbol that traded (delisted names included), so
        no post-hoc filtering by "still active" must ever be applied here.

        ``dry_run=True`` returns a dict with the cost estimate and record
        count WITHOUT downloading -- see :meth:`_dry_run_result`.
        ``confirm=True`` is required to proceed with any single request
        estimated above the guard's confirmation threshold (default $5).
        """
        self._require_start_after_history_begins(start)

        if dry_run:
            client = self._get_client()
            return self._dry_run_result(
                client, "ohlcv-1d", start.isoformat(), end.isoformat(), "ALL_SYMBOLS"
            )

        key = f"{start.isoformat()}_{end.isoformat()}"

        def _fetch() -> list[dict]:
            return self._fetch_bars(
                "ohlcv-1d", start.isoformat(), end.isoformat(), confirm=confirm
            )

        records = cached_call(f"{self.name}/daily_bars", key, _fetch)

        rows: list[dict[str, Any]] = []
        for r in records or []:
            volume = float(r.get("volume", 0.0))
            close = float(r.get("close", 0.0))
            trade_ts = pd.Timestamp(r.get("ts_event") or r.get("index")).normalize()
            # Same P4 fix as PolygonSource.daily_bars: a daily bar becomes
            # knowable at the 16:00 ET close, not at midnight.
            as_of_ts = trade_ts + pd.Timedelta(hours=16)
            rows.append(
                {
                    "trade_date": trade_ts,
                    "ticker": r.get("symbol"),
                    "open": r.get("open"),
                    "high": r.get("high"),
                    "low": r.get("low"),
                    "close": close,
                    "volume": volume,
                    "dollar_volume": close * volume,
                    "as_of": as_of_ts,
                }
            )

        df = pd.DataFrame(rows, columns=DAILY_BARS_COLUMNS)
        return _conform(df, DAILY_BARS_COLUMNS)

    def corporate_actions(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError(
            "Databento has no corporate-actions feed (splits/dividends/ticker "
            "changes). Use a reference-data vendor for this method, e.g. "
            "top10.data.polygon.PolygonSource.corporate_actions."
        )

    def ticker_meta(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError(
            "Databento has no point-in-time listing-metadata feed (security "
            "type / exchange / active_from / active_to). Use a reference-data "
            "vendor for this method, e.g. top10.data.polygon.PolygonSource.ticker_meta."
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
        dry_run: bool = False,
        confirm: bool = False,
    ) -> pd.DataFrame | dict[str, Any]:
        """04:00-09:25 ET minute bars via the ``ohlcv-1m`` schema.

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

        d_str = trade_date.isoformat()
        window_start = pd.Timestamp(f"{d_str} 04:00", tz=ET)
        window_end = pd.Timestamp(f"{d_str} 09:25", tz=ET)
        symbols = ",".join(tickers)

        if dry_run:
            client = self._get_client()
            return self._dry_run_result(client, "ohlcv-1m", d_str, d_str, symbols)

        def _fetch() -> list[dict]:
            return self._fetch_bars("ohlcv-1m", d_str, d_str, symbols=symbols, confirm=confirm)

        records = cached_call(f"{self.name}/premarket_bars", f"{symbols}_{d_str}", _fetch)

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
