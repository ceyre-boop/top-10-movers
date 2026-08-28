"""Free ($0) supplementary market-data adapters.

Databento alone does not cover everything this project needs (earnings
calendar, premarket bars pre-dating a paid pull, survivor-only daily bars
for sanity-checking). These three adapters plug those specific gaps at
zero cost, using free tiers of Finnhub, Alpaca, and Tiingo respectively.

Each adapter:
- reads its API key from the environment lazily (never at import time, and
  never in ``__init__``), so simply importing this module never requires
  a key or makes a network call;
- raises a clear, named error when its key is unconfigured, rather than
  silently returning an empty/looks-like-real-data result;
- conforms its output to the EXISTING column contracts in
  ``top10.data.base`` (``EARNINGS_COLUMNS`` / ``PREMARKET_BARS_COLUMNS`` /
  ``DAILY_BARS_COLUMNS``) via ``top10.data._conform``.

None of these are registered as a full ``MarketDataSource`` -- each only
implements the single method its free tier actually covers.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import pandas as pd
import requests

from top10.config import ET
from top10.data import _conform
from top10.data.base import DAILY_BARS_COLUMNS, EARNINGS_COLUMNS, PREMARKET_BARS_COLUMNS


class MissingApiKey(RuntimeError):
    """Raised when an adapter's env var is unset -- never a silent None
    from a data-fetching method."""


def _env_key(var_name: str) -> str | None:
    return os.environ.get(var_name) or None


# =============================================================================
# Finnhub earnings calendar
# =============================================================================


class FinnhubEarnings:
    """Finnhub ``/calendar/earnings`` free-tier adapter.

    Free tier is rate-limited to 60 calls/min (caller's responsibility to
    throttle across calls -- this adapter makes one call per ``earnings()``
    invocation and does not itself batch/paginate).

    IMPORTANT: Finnhub's docs claim earnings-calendar coverage back to
    2003, but the free tier attached to a given key has, in practice,
    been observed to only return roughly the last ~1 month of history
    regardless of the requested date range. Never trust the docs' claimed
    depth for planning a historical pull -- call :meth:`probe_lookback`
    first and use ITS answer, since it queries Finnhub directly with the
    configured key and reports what actually comes back.
    """

    name = "finnhub_earnings"
    _BASE_URL = "https://finnhub.io/api/v1"
    _ENV_VAR = "FINNHUB_API_KEY"

    def __init__(self) -> None:
        self._api_key: str | None = None

    def _key(self) -> str:
        if self._api_key is None:
            key = _env_key(self._ENV_VAR)
            if key is None:
                raise MissingApiKey(
                    f"{self._ENV_VAR} is not set -- FinnhubEarnings requires it. "
                    "Get a free key at https://finnhub.io/register."
                )
            self._api_key = key
        return self._api_key

    def _get(self, path: str, params: dict[str, Any]) -> dict:
        params = dict(params)
        params["token"] = self._key()
        resp = requests.get(f"{self._BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def probe_lookback(self, *, today: dt.date | None = None) -> dict[str, Any]:
        """Query the configured key's ACTUAL earnings-calendar lookback
        depth, rather than trusting Finnhub's documented "back to 2003"
        claim.

        Requests a wide range (10 years back from ``today``) and reports
        the earliest ``report_date`` Finnhub actually returned, plus the
        row count and whether that earliest date is suspiciously recent
        (< ~45 days back), which would indicate a free-tier lookback cap
        rather than genuine data absence.
        """
        today = today or dt.date.today()
        wide_start = today - dt.timedelta(days=365 * 10)
        payload = self._get(
            "/calendar/earnings",
            {"from": wide_start.isoformat(), "to": today.isoformat()},
        )
        rows = payload.get("earningsCalendar") or []
        report_dates = sorted(
            r["date"] for r in rows if r.get("date") is not None
        )
        earliest = report_dates[0] if report_dates else None
        earliest_dt = dt.date.fromisoformat(earliest) if earliest else None
        suspiciously_recent = (
            earliest_dt is not None and (today - earliest_dt).days < 45
        )
        return {
            "requested_from": wide_start.isoformat(),
            "requested_to": today.isoformat(),
            "row_count": len(rows),
            "earliest_report_date": earliest,
            "suspiciously_recent_lookback": suspiciously_recent,
        }

    def earnings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """``EARNINGS_COLUMNS``-conformant frame from Finnhub's calendar.

        Finnhub's payload does not distinguish "date announced" from
        "date of the report" -- there is no separate announcement
        timestamp, so ``as_of`` is conservatively set equal to
        ``report_date`` minus one day and ``date_is_revisable=True`` on
        every row (Finnhub calendar dates are known to shift as the real
        report date firms up), consistent with how
        ``PolygonSource.earnings`` treats an unknown announcement date.
        """
        payload = self._get(
            "/calendar/earnings",
            {"from": start.isoformat(), "to": end.isoformat()},
        )
        rows: list[dict[str, Any]] = []
        for r in payload.get("earningsCalendar") or []:
            report_date = r.get("date")
            if report_date is None:
                continue
            session_map = {"bmo": "bmo", "amc": "amc"}
            session = session_map.get((r.get("hour") or "").lower(), "unknown")
            report_ts = pd.Timestamp(report_date)
            rows.append(
                {
                    "ticker": r.get("symbol"),
                    "report_date": report_ts,
                    "session": session,
                    "announced_on": pd.NaT,
                    "date_is_revisable": True,
                    "as_of": report_ts - pd.Timedelta(days=1),
                }
            )

        df = pd.DataFrame(rows, columns=EARNINGS_COLUMNS)
        return _conform(df, EARNINGS_COLUMNS)


# =============================================================================
# Alpaca premarket bars (IEX feed)
# =============================================================================


class AlpacaPremarket:
    """Alpaca 1-minute premarket bars, IEX feed, 2016+.

    *** IEX-VOLUME CAVEAT (read before using dollar-volume features) ***
    Alpaca's free-tier market data is sourced from the IEX exchange only,
    which carries roughly ~2.5% of consolidated (SIP) US equity volume.
    Premarket dollar-volume figures computed from this adapter are NOT
    comparable in magnitude to SIP-consolidated volume from a paid feed
    (e.g. Polygon/Databento). This matters concretely for this project:
    baseline B4 thresholds top-10-by-premarket-gap candidates on premarket
    dollar volume (see docs/PREREG_TOP10.md) -- an IEX-sourced dollar-
    volume figure means something categorically different from a SIP
    figure, and a threshold tuned on one is not portable to the other.
    Never mix IEX-sourced and SIP-sourced dollar-volume figures within the
    same B4 threshold comparison.
    """

    name = "alpaca_premarket"
    _BASE_URL = "https://data.alpaca.markets/v2"
    _KEY_ID_ENV_VAR = "APCA_API_KEY_ID"
    _SECRET_ENV_VAR = "APCA_API_SECRET_KEY"
    FIRST_AVAILABLE_DATE = dt.date(2016, 1, 1)

    def __init__(self) -> None:
        self._key_id: str | None = None
        self._secret: str | None = None

    def _headers(self) -> dict[str, str]:
        if self._key_id is None or self._secret is None:
            key_id = _env_key(self._KEY_ID_ENV_VAR)
            secret = _env_key(self._SECRET_ENV_VAR)
            if key_id is None or secret is None:
                raise MissingApiKey(
                    f"{self._KEY_ID_ENV_VAR} and {self._SECRET_ENV_VAR} must both "
                    "be set -- AlpacaPremarket requires them. Get free keys at "
                    "https://alpaca.markets/."
                )
            self._key_id = key_id
            self._secret = secret
        return {
            "APCA-API-KEY-ID": self._key_id,
            "APCA-API-SECRET-KEY": self._secret,
        }

    def premarket_bars(self, trade_date: dt.date, tickers: list[str]) -> pd.DataFrame:
        """``PREMARKET_BARS_COLUMNS``-conformant frame, IEX feed only.

        See the class docstring's IEX-volume caveat before using the
        resulting ``volume``/dollar-volume figures alongside a SIP-sourced
        feed.
        """
        if trade_date < self.FIRST_AVAILABLE_DATE:
            raise ValueError(
                f"requested trade_date {trade_date.isoformat()} is before "
                f"Alpaca's IEX-feed history start "
                f"({self.FIRST_AVAILABLE_DATE.isoformat()})."
            )

        d_str = trade_date.isoformat()
        window_start = pd.Timestamp(f"{d_str} 04:00", tz=ET)
        window_end = pd.Timestamp(f"{d_str} 09:25", tz=ET)

        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            resp = requests.get(
                f"{self._BASE_URL}/stocks/{ticker}/bars",
                headers=self._headers(),
                params={
                    "timeframe": "1Min",
                    "start": window_start.tz_convert("UTC").isoformat(),
                    "end": window_end.tz_convert("UTC").isoformat(),
                    "feed": "iex",
                    "limit": 10000,
                },
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            for bar in payload.get("bars") or []:
                minute_utc = pd.Timestamp(bar["t"], tz="UTC")
                minute_et = minute_utc.tz_convert(ET)
                if not (window_start <= minute_et < window_end):
                    continue
                rows.append(
                    {
                        "trade_date": pd.Timestamp(trade_date),
                        "ticker": ticker,
                        "minute": minute_et.tz_localize(None),
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "volume": bar.get("v"),
                        "trade_count": bar.get("n"),
                        "as_of": minute_et.tz_localize(None),
                    }
                )

        df = pd.DataFrame(rows, columns=PREMARKET_BARS_COLUMNS)
        return _conform(df, PREMARKET_BARS_COLUMNS)


# =============================================================================
# Tiingo daily bars (survivor-only)
# =============================================================================


class TiingoDaily:
    """Tiingo supplementary daily OHLCV -- SURVIVOR-ONLY.

    *** SURVIVOR-ONLY: NEVER the sole source for label construction (P2) ***
    Tiingo's free/EOD daily-bars endpoint is queried per-ticker for a
    caller-supplied ticker list. It has no bulk "every symbol that traded,
    including delisted names" endpoint the way Databento's ``ohlcv-1d``
    (``symbols=ALL_SYMBOLS``) or Polygon's grouped-daily endpoint does --
    a delisted ticker simply isn't in the list the caller thinks to ask
    for, which is exactly the P2 "silent survivorship trap" this project's
    label spec forbids. This adapter exists ONLY to supplement/cross-check
    daily bars for tickers already known to still be listed; it must never
    be the sole/primary source feeding label construction, which requires
    a source that genuinely includes delisted names (e.g. Databento's
    ``daily_bars`` with ``ALL_SYMBOLS``, per ``DatabentoSource``).
    """

    name = "tiingo_daily"
    _BASE_URL = "https://api.tiingo.com/tiingo/daily"
    _ENV_VAR = "TIINGO_API_KEY"

    def __init__(self) -> None:
        self._api_key: str | None = None

    def _key(self) -> str:
        if self._api_key is None:
            key = _env_key(self._ENV_VAR)
            if key is None:
                raise MissingApiKey(
                    f"{self._ENV_VAR} is not set -- TiingoDaily requires it. "
                    "Get a free key at https://www.tiingo.com/."
                )
            self._api_key = key
        return self._api_key

    def daily_bars(
        self, start: dt.date, end: dt.date, tickers: list[str]
    ) -> pd.DataFrame:
        """``DAILY_BARS_COLUMNS``-conformant frame for the GIVEN
        (survivor) ``tickers`` only -- see class docstring."""
        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            resp = requests.get(
                f"{self._BASE_URL}/{ticker}/prices",
                params={
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                    "token": self._key(),
                    "format": "json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            for r in payload or []:
                trade_ts = pd.Timestamp(r.get("date"))
                if trade_ts.tzinfo is not None:
                    # Tiingo returns an ISO date with a "Z"/UTC suffix even
                    # though it means the calendar trade_date, not a real
                    # intraday instant -- strip tz rather than convert, so
                    # the date component itself never shifts.
                    trade_ts = trade_ts.tz_localize(None)
                trade_ts = trade_ts.normalize()
                close = float(r.get("close", 0.0))
                volume = float(r.get("volume", 0.0))
                # Same P4 fix as every other adapter: knowable at the
                # 16:00 ET close, not midnight.
                as_of_ts = trade_ts + pd.Timedelta(hours=16)
                rows.append(
                    {
                        "trade_date": trade_ts,
                        "ticker": ticker,
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
