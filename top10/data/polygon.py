"""Polygon.io implementation of :class:`~top10.data.base.MarketDataSource`.

No network call and no API key lookup happens at import time or in
``__init__`` -- the key is only resolved (via ``top10.config.get_api_key``)
the first time a request actually needs to go out.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import pandas as pd
import requests

from top10.config import ET, get_api_key
from top10.data import _conform
from top10.data.base import (
    CORPORATE_ACTIONS_COLUMNS,
    DAILY_BARS_COLUMNS,
    EARNINGS_COLUMNS,
    PREMARKET_BARS_COLUMNS,
    SHORT_INTEREST_COLUMNS,
    TICKER_META_COLUMNS,
)
from top10.data.cache import cached_call

BASE_URL = "https://api.polygon.io"

# Polygon's "type" field -> our TICKER_META_COLUMNS security_type enum.
_SECURITY_TYPE_MAP = {
    "CS": "CS",
    "ADRC": "ADR",
    "ADR": "ADR",
    "ETF": "ETF",
    "ETN": "ETF",
    "WARRANT": "WARRANT",
    "RIGHT": "RIGHT",
    "UNIT": "UNIT",
    "SP": "SPAC",
}

# Polygon's "primary_exchange" MIC codes we expect to see for US equities.
_KNOWN_EXCHANGES = {"XNYS", "XNAS", "XASE"}


class PolygonSource:
    """Polygon.io REST adapter. Implements ``MarketDataSource`` structurally."""

    name = "polygon"

    def __init__(self, calls_per_min: int = 100, max_retries: int = 5) -> None:
        self.calls_per_min = calls_per_min
        self.max_retries = max_retries
        self._min_interval = 60.0 / calls_per_min if calls_per_min > 0 else 0.0
        self._last_call_ts: float | None = None
        self._api_key: str | None = None

    # -- internals ---------------------------------------------------------

    def _key(self) -> str:
        if self._api_key is None:
            self._api_key = get_api_key("polygon")
        return self._api_key

    def _throttle(self) -> None:
        if self._min_interval <= 0 or self._last_call_ts is None:
            return
        elapsed = time.monotonic() - self._last_call_ts
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """GET ``path`` with exponential backoff on HTTP 429."""
        params = dict(params or {})
        params["apiKey"] = self._key()
        url = f"{BASE_URL}{path}"

        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            self._last_call_ts = time.monotonic()
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                last_exc = RuntimeError(f"Polygon 429 rate-limited on {path}")
                continue
            resp.raise_for_status()
            return resp.json()
        assert last_exc is not None
        raise last_exc

    # -- MarketDataSource ----------------------------------------------------

    def daily_bars(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Grouped-daily UNADJUSTED bars, one call per trading day.

        P2 NOTE: the grouped-daily endpoint returns every ticker that
        traded that day -- including names that are later delisted -- so we
        must never filter rows out by "is this ticker still active today".
        """
        rows: list[dict[str, Any]] = []
        for trade_date in pd.date_range(start, end, freq="D"):
            d = trade_date.date()
            key = d.isoformat()

            def _fetch(d=d) -> dict:
                return self._request(
                    f"/v2/aggs/grouped/locale/us/market/stocks/{d.isoformat()}",
                    # CRITICAL (P3 leakage): adjusted MUST be false. Back-adjusted
                    # prices leak future corporate-action knowledge into the past.
                    params={"adjusted": "false"},
                )

            payload = cached_call(f"{self.name}/daily_bars", key, _fetch)
            results = payload.get("results") if isinstance(payload, dict) else None
            if not results:
                continue

            trade_ts = pd.Timestamp(d)
            # Grouped-daily results become knowable at the CLOSE of the
            # trading day (16:00 ET) -- NOT at midnight. Stamping midnight
            # would make a bar containing `close_t` (the label numerator)
            # look "knowable" hours before the market even opened, which
            # defangs any `as_of <= decision_time` PIT check for a
            # decision_time between midnight and the real close (e.g. T2's
            # 09:25 ET premarket cutoff).
            as_of_ts = trade_ts + pd.Timedelta(hours=16)
            for r in results:
                volume = float(r.get("v", 0.0))
                vwap = r.get("vw")
                dollar_volume = float(vwap) * volume if vwap is not None else float(r.get("c", 0.0)) * volume
                rows.append(
                    {
                        "trade_date": trade_ts,
                        "ticker": r.get("T"),
                        "open": r.get("o"),
                        "high": r.get("h"),
                        "low": r.get("l"),
                        "close": r.get("c"),
                        "volume": volume,
                        "dollar_volume": dollar_volume,
                        "as_of": as_of_ts,
                    }
                )

        df = pd.DataFrame(rows, columns=DAILY_BARS_COLUMNS)
        return _conform(df, DAILY_BARS_COLUMNS)

    def corporate_actions(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        def _fetch_splits() -> dict:
            return self._request(
                "/v3/reference/splits",
                params={
                    "execution_date.gte": start.isoformat(),
                    "execution_date.lte": end.isoformat(),
                    "limit": 1000,
                },
            )

        splits = cached_call(
            f"{self.name}/splits", f"{start.isoformat()}_{end.isoformat()}", _fetch_splits
        )
        for r in (splits.get("results") or []):
            split_from = r.get("split_from")
            split_to = r.get("split_to")
            ratio = (split_to / split_from) if split_from else None
            execution_date = r.get("execution_date")
            # Prefer the announcement date where the payload provides one;
            # Polygon's splits payload does not always include it, so we
            # fall back to execution_date (documented best-effort).
            as_of = r.get("announcement_date") or execution_date
            rows.append(
                {
                    "ex_date": execution_date,
                    "ticker": r.get("ticker"),
                    "action_type": "split" if (ratio is None or ratio >= 1) else "reverse_split",
                    "ratio": ratio,
                    "cash_amount": None,
                    "new_ticker": None,
                    "as_of": as_of,
                }
            )

        def _fetch_dividends() -> dict:
            return self._request(
                "/v3/reference/dividends",
                params={
                    "ex_dividend_date.gte": start.isoformat(),
                    "ex_dividend_date.lte": end.isoformat(),
                    "limit": 1000,
                },
            )

        dividends = cached_call(
            f"{self.name}/dividends", f"{start.isoformat()}_{end.isoformat()}", _fetch_dividends
        )
        for r in (dividends.get("results") or []):
            ex_date = r.get("ex_dividend_date")
            as_of = r.get("declaration_date") or ex_date
            rows.append(
                {
                    "ex_date": ex_date,
                    "ticker": r.get("ticker"),
                    "action_type": "dividend",
                    "ratio": None,
                    "cash_amount": r.get("cash_amount"),
                    "new_ticker": None,
                    "as_of": as_of,
                }
            )

        # Ticker-change events: Polygon has no single bulk endpoint for
        # this, so this is best-effort and may legitimately return zero
        # rows for a given range.

        df = pd.DataFrame(rows, columns=CORPORATE_ACTIONS_COLUMNS)
        return _conform(df, CORPORATE_ACTIONS_COLUMNS)

    def _ticker_meta_page(
        self, end: dt.date, active_flag: str, rows: list[dict[str, Any]]
    ) -> None:
        """Fetch every page of `/v3/reference/tickers?date=end&active=active_flag`
        into `rows`. Split out of `ticker_meta` so it can be called once for
        "currently active as of end" and once for "already delisted as of
        end" -- see `ticker_meta` docstring for why both calls are required.
        """
        url_path: str | None = "/v3/reference/tickers"
        params: dict[str, Any] | None = {
            "market": "stocks",
            "date": end.isoformat(),
            "active": active_flag,
            "limit": 1000,
        }
        page = 0
        next_key = f"{end.isoformat()}_active{active_flag}_p0"

        while url_path is not None:

            def _fetch(url_path=url_path, params=params) -> dict:
                return self._request(url_path, params=params)

            payload = cached_call(f"{self.name}/ticker_meta", next_key, _fetch)
            for r in (payload.get("results") or []):
                raw_type = (r.get("type") or "").upper()
                security_type = _SECURITY_TYPE_MAP.get(raw_type, "OTHER")
                exchange = r.get("primary_exchange") or "OTC"
                if exchange not in _KNOWN_EXCHANGES:
                    exchange = exchange if exchange else "OTC"
                list_date = r.get("list_date")
                rows.append(
                    {
                        "ticker": r.get("ticker"),
                        "name": r.get("name"),
                        "security_type": security_type,
                        "exchange": exchange,
                        "active_from": list_date,
                        "active_to": r.get("delisted_utc"),
                        # Best-effort: Polygon's bulk ticker-list endpoint
                        # does not reliably return market cap / share
                        # counts (those live on the per-ticker "ticker
                        # details" endpoint, which is one call per symbol
                        # and out of scope for a bulk point-in-time fetch).
                        # When the vendor DOES include these fields, we
                        # take them as-is; otherwise they are legitimately
                        # NaN, never silently omitted from the contract.
                        "market_cap": r.get("market_cap"),
                        "float_shares": r.get("weighted_shares_outstanding")
                        or r.get("share_class_shares_outstanding"),
                        # `as_of` is pinned to `active_from` (list_date),
                        # NOT `end` -- see `ticker_meta` docstring.
                        "as_of": list_date,
                    }
                )

            next_url = payload.get("next_url")
            if next_url:
                # next_url already carries its own query string (incl. cursor).
                url_path = next_url.replace(BASE_URL, "")
                params = None
                page += 1
                next_key = f"{end.isoformat()}_active{active_flag}_p{page}"
            else:
                url_path = None

    def ticker_meta(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Point-in-time listing metadata across `[start, end]`.

        P2 FIX: `/v3/reference/tickers?date=end&active=true` alone only
        returns names still listed as of `end` -- every name delisted
        before `end` would silently vanish from the universe (and, via
        `labels.build_universe`'s inner join, from every label). We
        therefore ALSO fetch `active=false` at the same `date=end`
        snapshot, which surfaces every ticker Polygon already has a
        recorded delisting for as of `end`. This is what makes a ticker
        delisted in 2019 show up in a universe built for a 2017
        `trade_date`: its `active_from`/`active_to` still correctly
        bracket 2017 -- we've simply learned about the 2019 delisting by
        the time this call was made. Call cost: 2x the single-call
        version (one `active=true` sweep, one `active=false` sweep, each
        paginated and cached per-page), which is negligible next to the
        survivorship bug it fixes.

        `as_of` is pinned to each row's `active_from` (`list_date`), never
        to `end` -- see TICKER_META_COLUMNS / `_ticker_meta_page`. This
        also fixes the second half of the P2/P4 bug: with `as_of == end`
        on every row, `labels.build_universe`'s `meta["as_of"] <
        trade_date` filter rejected every row for every `trade_date <=
        end`, silently producing an empty universe (and therefore empty
        labels that make every downstream sanity check vacuously PASS).

        KNOWN LIMITATION: Polygon's ticker-detail fields (name /
        security_type / exchange / market_cap / float_shares) reflect the
        vendor's LATEST known classification as of `end`, not necessarily
        what applied on every historical `trade_date` in range -- Polygon
        has no bulk endpoint for point-in-time ticker-attribute history.
        A ticker that changed exchange or security type mid-life will show
        its current classification for all historical trade_dates. A true
        day-by-day snapshot would require one `date=<d>` call per trading
        day in `[start, end]` (thousands of calls for a multi-year
        backfill); `active_from`/`active_to` (the fields the P2 rule
        actually depends on) are unaffected by this limitation, so we
        accept it rather than pay that call cost.
        """
        rows: list[dict[str, Any]] = []
        self._ticker_meta_page(end, "true", rows)
        self._ticker_meta_page(end, "false", rows)

        df = pd.DataFrame(rows, columns=TICKER_META_COLUMNS)
        # A ticker should only appear in one of the two `active` buckets
        # for a given `date=end` snapshot; de-dup defensively in case
        # Polygon's current-status semantics ever overlap.
        df = df.drop_duplicates(subset=["ticker", "active_from"], keep="first")
        return _conform(df, TICKER_META_COLUMNS)

    # When Polygon gives us no announcement timestamp for a report date,
    # `as_of` cannot be `report_date` itself (that is >= decision_time_t1
    # for that same trade_date -- see `earnings` docstring) and it must
    # not be left NaT either (NaT fails every `as_of <= decision_time`
    # filter, silently erasing the row before `date_is_revisable=True`
    # can ever be read downstream). One calendar day before `report_date`
    # is the latest bound that still respects `decision_time_t1`'s
    # prior-close cutoff for a trade_date == report_date, without
    # asserting any earlier knowledge we don't actually have.
    _UNKNOWN_ANNOUNCED_ASOF_LEAD = pd.Timedelta(days=1)

    def earnings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Best-effort earnings calendar.

        Availability/shape depends on the Polygon plan tier. When the
        vendor doesn't give us an announcement timestamp for the report
        date, we mark ``date_is_revisable=True`` per the contract.

        P4 FIX: `as_of` used to fall back to `report_date` when
        `announced_on` was missing. `report_date` (midnight) is AFTER
        `decision_time_t1(report_date) == report_date - 8h` (16:00 the
        prior day), so every such row failed
        `baselines.b3_earnings_x_vol`'s `_assert_safe_per_day` check --
        B3 could not run on any real Polygon earnings row with an unknown
        announcement date. See `_UNKNOWN_ANNOUNCED_ASOF_LEAD` for the
        conservative bound used instead, and EARNINGS_COLUMNS for the
        matching change required in `top10.features.t1._earnings_features`
        (which must gate on `as_of`, not `announced_on`, to avoid
        dropping every `date_is_revisable=True` row before its flag is
        readable).
        """
        rows: list[dict[str, Any]] = []

        def _fetch() -> dict:
            return self._request(
                "/vX/reference/earnings",
                params={
                    "report_date.gte": start.isoformat(),
                    "report_date.lte": end.isoformat(),
                    "limit": 1000,
                },
            )

        payload = cached_call(
            f"{self.name}/earnings", f"{start.isoformat()}_{end.isoformat()}", _fetch
        )
        for r in (payload.get("results") or []):
            announced_on = r.get("announced_on")
            report_date = r.get("report_date")
            date_is_revisable = announced_on is None
            if announced_on is not None:
                as_of = announced_on
            elif report_date is not None:
                as_of = pd.Timestamp(report_date) - self._UNKNOWN_ANNOUNCED_ASOF_LEAD
            else:
                as_of = None
            rows.append(
                {
                    "ticker": r.get("ticker"),
                    "report_date": report_date,
                    "session": r.get("session", "unknown"),
                    "announced_on": announced_on,
                    "date_is_revisable": date_is_revisable,
                    "as_of": as_of,
                }
            )

        df = pd.DataFrame(rows, columns=EARNINGS_COLUMNS)
        return _conform(df, EARNINGS_COLUMNS)

    def short_interest(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Bi-monthly FINRA short-interest figures via Polygon's short
        interest endpoint.

        `as_of` MUST be the FINRA PUBLISH date, never `settlement_date` --
        see SHORT_INTEREST_COLUMNS. FINRA publishes short-interest reports
        roughly 8 BUSINESS days after `settlement_date` (a fixed biweekly
        schedule); when Polygon's payload includes an actual publish
        timestamp we use it directly, otherwise we fall back to a
        conservative CALENDAR-day estimate (`_PUBLISH_LAG_DAYS`, chosen
        deliberately larger than the real ~8-business-day lag so this
        estimate never UNDERSTATES the lag and therefore never leaks).
        """
        rows: list[dict[str, Any]] = []
        url_path: str | None = "/stocks/v1/short-interest"
        params: dict[str, Any] | None = {
            "settlement_date.gte": start.isoformat(),
            "settlement_date.lte": end.isoformat(),
            "limit": 1000,
        }
        page = 0
        next_key = f"{start.isoformat()}_{end.isoformat()}_p0"
        saw_any_page = False

        while url_path is not None:

            def _fetch(url_path=url_path, params=params) -> dict:
                return self._request(url_path, params=params)

            try:
                payload = cached_call(f"{self.name}/short_interest", next_key, _fetch)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if not saw_any_page and status in (403, 404):
                    # Plan/account genuinely lacks this endpoint -- name the
                    # gap explicitly rather than returning an empty frame
                    # that would look like "no short interest this period".
                    raise NotImplementedError(
                        "Polygon short-interest endpoint (/stocks/v1/short-interest) "
                        f"returned HTTP {status} -- unavailable for this account/plan. "
                        "A FINRA short-interest feed is required for `short_interest`; "
                        "no other source is wired up."
                    ) from exc
                raise
            saw_any_page = True

            for r in (payload.get("results") or []):
                settlement_date = r.get("settlement_date")
                publish_date = r.get("publish_date") or r.get("date_published")
                if publish_date is not None:
                    as_of = publish_date
                elif settlement_date is not None:
                    as_of = pd.Timestamp(settlement_date) + self._PUBLISH_LAG_DAYS
                else:
                    as_of = None
                float_pct = r.get("short_interest_pct_float") or r.get("percent_of_float")
                rows.append(
                    {
                        "ticker": r.get("ticker"),
                        "settlement_date": settlement_date,
                        "short_interest_shares": r.get("short_interest") or r.get("settlement_shares"),
                        "short_interest_pct_float": float_pct,
                        "days_to_cover": r.get("days_to_cover") or r.get("avg_daily_volume_days_to_cover"),
                        "as_of": as_of,
                    }
                )

            next_url = payload.get("next_url")
            if next_url:
                url_path = next_url.replace(BASE_URL, "")
                params = None
                page += 1
                next_key = f"{start.isoformat()}_{end.isoformat()}_p{page}"
            else:
                url_path = None

        df = pd.DataFrame(rows, columns=SHORT_INTEREST_COLUMNS)
        return _conform(df, SHORT_INTEREST_COLUMNS)

    # Conservative calendar-day estimate of the FINRA settlement -> publish
    # lag, used only when Polygon's payload omits an explicit publish
    # date. Real lag is ~8 BUSINESS days (~11-12 calendar days across a
    # weekend); 14 is chosen deliberately larger so this fallback can only
    # ever overstate the lag (safe direction), never understate it.
    _PUBLISH_LAG_DAYS = pd.Timedelta(days=14)

    def premarket_bars(self, trade_date: dt.date, tickers: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        d_str = trade_date.isoformat()
        window_start = pd.Timestamp(f"{d_str} 04:00", tz=ET)
        window_end = pd.Timestamp(f"{d_str} 09:25", tz=ET)

        for ticker in tickers:

            def _fetch(ticker=ticker) -> dict:
                return self._request(
                    f"/v2/aggs/ticker/{ticker}/range/1/minute/{d_str}/{d_str}",
                    params={"adjusted": "false", "sort": "asc", "limit": 50000},
                )

            payload = cached_call(f"{self.name}/premarket_bars", f"{ticker}_{d_str}", _fetch)
            for r in (payload.get("results") or []):
                minute_utc = pd.Timestamp(r["t"], unit="ms", tz="UTC")
                minute_et = minute_utc.tz_convert(ET)
                # `window_end` (09:25) is EXCLUSIVE -- the 09:25:00-09:25:59
                # bar has not fully closed until the T2 decision boundary
                # (09:25 ET) has already passed, so it must not be treated
                # as knowable at that boundary. `t2.decision_time_t2` /
                # its consumers filter with `minute < cutoff`; this adapter
                # must agree, or a caller that trusts this method's own
                # windowing (rather than re-filtering) would see one bar of
                # look-ahead.
                if not (window_start <= minute_et < window_end):
                    continue
                rows.append(
                    {
                        "trade_date": pd.Timestamp(trade_date),
                        "ticker": ticker,
                        "minute": minute_et.tz_localize(None),
                        "open": r.get("o"),
                        "high": r.get("h"),
                        "low": r.get("l"),
                        "close": r.get("c"),
                        "volume": r.get("v"),
                        "trade_count": r.get("n"),
                        "as_of": minute_et.tz_localize(None),
                    }
                )

        df = pd.DataFrame(rows, columns=PREMARKET_BARS_COLUMNS)
        return _conform(df, PREMARKET_BARS_COLUMNS)
