"""Vendor-agnostic market data contract.

Every adapter returns pandas DataFrames with the exact columns documented
below. Adapters MUST NOT return rows whose information was not knowable at
the row's `as_of`. See docs/LABEL_SPEC.md "Point-in-time invariants".
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

import pandas as pd

# --- Frozen column contracts -------------------------------------------------

DAILY_BARS_COLUMNS = [
    "trade_date",   # datetime64[ns], midnight ET-naive
    "ticker",       # str
    "open",         # float, UNADJUSTED
    "high",
    "low",
    "close",
    "volume",       # float
    "dollar_volume",  # float
    "as_of",        # datetime64[ns], when this row became knowable.
                    # For a daily bar this is 16:00 ET on `trade_date`
                    # (the close), NOT midnight -- midnight would make
                    # the row look knowable *before the trading day even
                    # started*, defanging every `as_of <= decision_time`
                    # PIT check for any decision_time between midnight and
                    # the real close (e.g. T2's 09:25 ET).
]

CORPORATE_ACTIONS_COLUMNS = [
    "ex_date",      # datetime64[ns]
    "ticker",
    "action_type",  # {"split", "reverse_split", "dividend", "ticker_change"}
    "ratio",        # float; split ratio (new/old). NaN for dividends.
    "cash_amount",  # float; NaN for splits
    "new_ticker",   # str or None, for ticker_change
    "as_of",        # announcement/knowable date, NOT ex_date
]

TICKER_META_COLUMNS = [
    "ticker",
    "name",
    "security_type",  # {"CS", "ADR", "ETF", "WARRANT", "RIGHT", "UNIT", "SPAC", "OTHER"}
    "exchange",       # {"XNYS", "XNAS", "XASE", "OTC", ...}
    "active_from",    # datetime64[ns], the ticker's list_date
    "active_to",      # datetime64[ns], NaT if still listed
    "market_cap",     # float, NaN if the vendor didn't supply it for this row
    "float_shares",   # float, NaN if the vendor didn't supply it for this row
    "as_of",          # datetime64[ns], the date THIS ROW became knowable.
                       # MUST be pinned to `active_from` (list_date), never
                       # to the query's range-end date: a ticker's
                       # existence/classification became knowable the day
                       # it was listed, so any `trade_date` strictly after
                       # `active_from` may legitimately use the row. Do
                       # NOT derive `as_of` from `active_to` (delisting
                       # date) either -- learning a delisting date later
                       # must never retroactively un-know that the ticker
                       # existed and traded before it (see
                       # docs/LABEL_SPEC.md "Include names that are later
                       # delisted").
]

# Short interest is a distinct, bi-monthly FINRA feed with its own
# publish-lag semantics -- NOT part of TICKER_META_COLUMNS. See
# `MarketDataSource.short_interest`.
SHORT_INTEREST_COLUMNS = [
    "ticker",
    "settlement_date",           # datetime64[ns]; FINRA's "as of" record/settlement date
    "short_interest_shares",     # float
    "short_interest_pct_float",  # float, percent (0-100) of float shares short
    "days_to_cover",             # float
    "as_of",                     # datetime64[ns]. MUST be the FINRA PUBLISH
                                   # date -- the day FINRA actually released
                                   # the report -- and NEVER `settlement_date`.
                                   # `settlement_date` is a look-ahead: FINRA
                                   # publishes short-interest figures roughly
                                   # 8 business days AFTER the settlement
                                   # date, so stamping `as_of=settlement_date`
                                   # would make the row look knowable ~2
                                   # calendar weeks before it actually was.
]

EARNINGS_COLUMNS = [
    "ticker",
    "report_date",       # datetime64[ns]
    "session",           # {"bmo", "amc", "unknown"}
    "announced_on",      # datetime64[ns]; when the DATE was published. NaT => revised-risk
    "date_is_revisable", # bool; True when `announced_on` is unknown
    "as_of",              # datetime64[ns], when THIS ROW (not necessarily
                           # `announced_on`) became usable. When
                           # `announced_on` is known, `as_of == announced_on`.
                           # When it is genuinely unknown (`date_is_revisable
                           # == True`), `as_of` is still populated with a
                           # defensible conservative bound (see
                           # `PolygonSource.earnings`) rather than left NaT --
                           # a NaT `as_of` would make the row unusable by
                           # `as_of <= decision_time` PIT filters and
                           # silently erase every `date_is_revisable=True`
                           # row before its flag could ever be read by a
                           # consumer. Consumers that currently gate on
                           # `announced_on <= decision_time` (e.g.
                           # `top10.features.t1._earnings_features`) must
                           # gate on `as_of <= decision_time` instead so
                           # revisable rows survive.
]

PREMARKET_BARS_COLUMNS = [
    "trade_date",
    "ticker",
    "minute",        # datetime64[ns], ET-naive, within 04:00-09:25
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "as_of",
]


@runtime_checkable
class MarketDataSource(Protocol):
    """Implemented by each vendor adapter (Polygon, Databento, ...)."""

    name: str

    def daily_bars(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """UNADJUSTED daily OHLCV for ALL listed names, including symbols
        delisted after `start`. Columns == DAILY_BARS_COLUMNS."""
        ...

    def corporate_actions(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Columns == CORPORATE_ACTIONS_COLUMNS."""
        ...

    def ticker_meta(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Point-in-time listing metadata, INCLUDING names delisted by
        `end` (P2: never active-only). Columns == TICKER_META_COLUMNS."""
        ...

    def earnings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Columns == EARNINGS_COLUMNS."""
        ...

    def short_interest(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Bi-monthly FINRA short-interest figures. Columns ==
        SHORT_INTEREST_COLUMNS. `as_of` MUST be the FINRA publish date,
        never `settlement_date` -- see SHORT_INTEREST_COLUMNS."""
        ...

    def premarket_bars(
        self, trade_date: dt.date, tickers: list[str]
    ) -> pd.DataFrame:
        """04:00-09:25 ET minute bars. Columns == PREMARKET_BARS_COLUMNS."""
        ...


DECISION_TIME_T1 = "prior_close"   # 16:00 ET on t-1
DECISION_TIME_T2 = "premarket"     # 09:25 ET on t
