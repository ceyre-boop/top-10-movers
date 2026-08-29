"""CRSP (via WRDS) implementation of :class:`~top10.data.base.MarketDataSource`.

CRSP is the only source wired up here that solves P2 (survivorship bias) at
research grade: `crsp.dsf` / `crsp.dsenames` include delisted securities
with proper delisting-return handling going back decades, unlike the
vendor snapshot approach the other adapters have to fall back on. This
adapter is the intended long-run fix for P2 once WRDS/CRSP access is
confirmed -- see docs/DATA_SOURCES.md for the full vendor comparison.

Connects via the ``wrds`` python package (``wrds.Connection()``), which
authenticates against WRDS's PostgreSQL database. Credentials come from the
``WRDS_USERNAME`` environment variable plus the user's ``~/.pgpass`` file
(WRDS's own convention) -- this module never hardcodes, prompts for, or
logs a password. ``wrds`` is imported LAZILY inside ``_get_connection`` so
this module (and simply constructing ``CRSPSource``) never requires the
package to be installed.

No network/database call happens at import time, in ``__init__``, or in
any query-building helper -- only ``_get_connection()`` (invoked lazily,
the first time a query actually needs to run) touches the network.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import pandas as pd

from top10.data import _conform
from top10.data.base import (
    CORPORATE_ACTIONS_COLUMNS,
    DAILY_BARS_COLUMNS,
    EARNINGS_COLUMNS,
    PREMARKET_BARS_COLUMNS,
    SHORT_INTEREST_COLUMNS,
    TICKER_META_COLUMNS,
)

# CRSP `shrcd` (share code) -> our TICKER_META_COLUMNS `security_type` enum.
# Per LABEL_SPEC.md's universe definition (US common stocks and ADRs on
# NYSE/NASDAQ/AMEX): 10/11 = ordinary US common stock, 30/31 = ADR.
# Everything else is left "OTHER" so callers can filter/flag it per
# LABEL_SPEC rather than have this adapter silently decide for them.
_SHRCD_SECURITY_TYPE = {
    10: "CS",
    11: "CS",
    30: "ADR",
    31: "ADR",
}

# CRSP `exchcd` (exchange code) -> MIC. Negative exchcd values denote a
# security that WAS listed on that exchange but is no longer actively
# trading there (CRSP's own convention for capturing delisting status
# inline on the same code); we map the absolute value so a delisted NYSE
# name still reports XNYS rather than falling through to "OTC".
_EXCHCD_EXCHANGE = {
    1: "XNYS",
    2: "XASE",
    3: "XNAS",
}


class CRSPSource:
    """WRDS/CRSP adapter. Implements ``MarketDataSource`` structurally."""

    name = "crsp"

    def __init__(self) -> None:
        self._connection: Any = None

    # -- internals ---------------------------------------------------------

    def _get_connection(self) -> Any:
        """Lazily open (and cache) the WRDS PostgreSQL connection.

        Credentials: ``WRDS_USERNAME`` env var (if set) plus the user's
        ``~/.pgpass`` file, both handled entirely by the ``wrds`` package
        itself -- this method never reads, stores, or logs a password.
        """
        if self._connection is None:
            import wrds  # lazy: module must import without wrds installed

            username = os.environ.get("WRDS_USERNAME")
            kwargs: dict[str, Any] = {"wrds_username": username} if username else {}
            self._connection = wrds.Connection(**kwargs)
        return self._connection

    # -- raw query helpers (split out so tests can monkeypatch them
    #    directly, same pattern as DatabentoSource._fetch_bars) -----------

    def _fetch_daily_bars_raw(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """`crsp.dsf` (daily stock file) joined to `crsp.dsenames` for the
        ticker/exchange that was in effect on each trade date.

        P2: this join intentionally does NOT filter on any "still active"
        flag -- `dsf` carries every PERMNO that traded in range, delisted
        names included, and `dsenames`'s `namedt`/`nameendt` range simply
        picks the ticker string that was correct *on that date* rather than
        excluding anything.
        """
        sql = """
            SELECT dsf.permno,
                   dsf.date,
                   dsf.prc,
                   dsf.openprc,
                   dsf.askhi,
                   dsf.bidlo,
                   dsf.vol,
                   dsf.cfacpr,
                   dsf.cfacshr,
                   names.ticker,
                   names.exchcd
              FROM crsp.dsf AS dsf
              JOIN crsp.dsenames AS names
                ON dsf.permno = names.permno
               AND dsf.date BETWEEN names.namedt AND COALESCE(names.nameendt, dsf.date)
             WHERE dsf.date BETWEEN %(start)s AND %(end)s
        """
        return self._get_connection().raw_sql(
            sql, params={"start": start, "end": end}, date_cols=["date"]
        )

    def _fetch_splits_dividends_raw(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """`crsp.dsedist` distribution events (splits + cash dividends).

        `facpr` is CRSP's price-adjustment factor: non-zero/non-null marks
        a split (or reverse split, when `facpr` is negative). `divamt` is
        the cash-dividend amount. A row legitimately has one or the other,
        never authoritatively both in this simplified mapping -- CRSP's
        `distcd` distribution-code taxonomy has finer-grained categories
        (spinoffs, rights offerings, etc.) that are out of scope here.
        """
        sql = """
            SELECT permno, dclrdt, exdt, distcd, divamt, facshr, facpr
              FROM crsp.dsedist
             WHERE exdt BETWEEN %(start)s AND %(end)s
        """
        return self._get_connection().raw_sql(
            sql, params={"start": start, "end": end}, date_cols=["dclrdt", "exdt"]
        )

    def _fetch_ticker_changes_raw(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """`crsp.dsenames` name/ticker history, used to derive ticker-change
        events from consecutive rows for the same PERMNO. Pulled with a
        wide-open date filter (`namedt <= end` and `nameendt >= start` or
        NULL) so a transition that straddles the requested range is not
        clipped out of either side.
        """
        sql = """
            SELECT permno, ticker, namedt, nameendt
              FROM crsp.dsenames
             WHERE namedt <= %(end)s
               AND (nameendt IS NULL OR nameendt >= %(start)s)
             ORDER BY permno, namedt
        """
        return self._get_connection().raw_sql(
            sql, params={"start": start, "end": end}, date_cols=["namedt", "nameendt"]
        )

    def _fetch_delistings_raw(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """`crsp.dsedelist` -- the whole reason this adapter exists (P2).

        `dlstdt` is the delisting date, `dlstcd` the delisting reason code,
        `dlret` the delisting return (the partial/final return investors
        actually realized around the delisting event, distinct from a
        share-count or price ratio).
        """
        sql = """
            SELECT permno, dlstdt, dlstcd, dlret
              FROM crsp.dsedelist
             WHERE dlstdt BETWEEN %(start)s AND %(end)s
        """
        return self._get_connection().raw_sql(
            sql, params={"start": start, "end": end}, date_cols=["dlstdt"]
        )

    def _fetch_ticker_meta_raw(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """`crsp.dsenames` joined to `crsp.dsf` (on `namedt`) for a
        best-effort market-cap snapshot at each row's `active_from` date.

        `namedt`/`nameendt` give genuinely point-in-time `active_from`/
        `active_to` ranges (unlike the other adapters' "vendor's latest
        known classification" limitation) -- see `ticker_meta` docstring.
        """
        sql = """
            SELECT names.permno,
                   names.ticker,
                   names.comnam,
                   names.namedt,
                   names.nameendt,
                   names.exchcd,
                   names.shrcd,
                   dsf.prc,
                   dsf.shrout
              FROM crsp.dsenames AS names
              LEFT JOIN crsp.dsf AS dsf
                ON dsf.permno = names.permno
               AND dsf.date = names.namedt
             WHERE names.namedt <= %(end)s
               AND (names.nameendt IS NULL OR names.nameendt >= %(start)s)
        """
        return self._get_connection().raw_sql(
            sql, params={"start": start, "end": end}, date_cols=["namedt", "nameendt"]
        )

    # -- MarketDataSource ----------------------------------------------------

    def daily_bars(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """UNADJUSTED daily OHLCV, keyed internally on PERMNO (the stable
        CRSP identifier) but returned as `ticker` for contract
        compatibility. PERMNO is carried through as an extra column beyond
        the frozen contract -- ticker reuse across unrelated companies over
        the decades CRSP covers is a real source of silent joins-gone-wrong
        that PERMNO (unlike `ticker`) is immune to.

        P2: delisted PERMNOs are included -- see `_fetch_daily_bars_raw`.

        P3: `cfacpr`/`cfacshr` (CRSP's split/distribution adjustment
        factors) are pulled through for visibility but deliberately NEVER
        multiplied into price/volume here. This pipeline requires
        as-traded, unadjusted prices; applying them would leak future
        corporate-action knowledge into historical rows exactly like a
        vendor's "adjusted=true" bar would.
        """
        raw = self._fetch_daily_bars_raw(start, end)
        if raw is None or raw.empty:
            df = pd.DataFrame(columns=DAILY_BARS_COLUMNS)
            df["permno"] = pd.Series(dtype="float64")
            df["price_is_midpoint"] = pd.Series(dtype="bool")
            return df

        raw = raw.reset_index(drop=True)
        prc = pd.to_numeric(raw["prc"], errors="coerce")
        # CRSP sets `prc` NEGATIVE when it is a bid/ask midpoint rather than
        # an actual trade close (no last trade that day). Silently taking a
        # negative price would poison every return computed off of it, so
        # we take the absolute value and flag it via `price_is_midpoint`
        # rather than pretend it's an ordinary close.
        close = prc.abs()
        price_is_midpoint = prc < 0
        volume = pd.to_numeric(raw["vol"], errors="coerce")

        trade_ts = pd.to_datetime(raw["date"]).dt.normalize()
        # Same P4 fix as the other adapters: a daily bar becomes knowable
        # at the 16:00 ET close, not at midnight.
        as_of_ts = trade_ts + pd.Timedelta(hours=16)

        # NOTE on `vol` units: CRSP's own documentation notes that reported
        # daily volume conventions have changed across exchanges/eras (most
        # infamously, some historical periods report share counts in
        # round-lot units rather than raw shares). We deliberately do NOT
        # apply a blind multiplier here -- guessing the wrong conversion
        # for an unknown date range would silently corrupt volume rather
        # than leaving it verifiably unadjusted. Verify the `crsp.dsf`
        # documentation for the specific date range being pulled before
        # trusting `dollar_volume`/`volume` at face value across eras.
        dollar_volume = close * volume

        out = pd.DataFrame(
            {
                "trade_date": trade_ts,
                "ticker": raw["ticker"],
                "open": pd.to_numeric(raw["openprc"], errors="coerce"),
                "high": pd.to_numeric(raw["askhi"], errors="coerce"),
                "low": pd.to_numeric(raw["bidlo"], errors="coerce"),
                "close": close,
                "volume": volume,
                "dollar_volume": dollar_volume,
                "as_of": as_of_ts,
            }
        )
        conformed = _conform(out, DAILY_BARS_COLUMNS)
        # Extra columns beyond the frozen contract -- see docstring.
        conformed["permno"] = raw["permno"].values
        conformed["price_is_midpoint"] = price_is_midpoint.values
        return conformed

    def corporate_actions(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Splits/dividends (`crsp.dsedist`), ticker changes (derived from
        `crsp.dsenames` transitions), and delisting events
        (`crsp.dsedelist`) -- all mapped to CORPORATE_ACTIONS_COLUMNS.

        `as_of` is set to when each fact was knowable, not the effective
        date: `dclrdt` (declaration date) for distributions when present,
        the ticker-transition's own `namedt` for ticker changes (CRSP has
        no separate announcement date for this in `dsenames`), and
        `dlstdt` for delistings (the delisting date is itself the date the
        fact became known).

        NOTE: delisting events use `action_type="delisting"`, which
        extends the enum documented in `CORPORATE_ACTIONS_COLUMNS` (that
        comment predates CRSP being wired up and only the other adapters,
        which have no delisting feed at all, informed it). Consumers that
        switch on `action_type` must handle this value or explicitly
        filter it out.
        """
        rows: list[dict[str, Any]] = []

        dist = self._fetch_splits_dividends_raw(start, end)
        if dist is not None and not dist.empty:
            dist = dist.reset_index(drop=True)
            for _, r in dist.iterrows():
                facpr = r.get("facpr")
                divamt = r.get("divamt")
                exdt = r.get("exdt")
                dclrdt = r.get("dclrdt")
                as_of = dclrdt if pd.notna(dclrdt) else exdt
                if pd.notna(facpr) and facpr != 0:
                    ratio = 1.0 + float(facpr)
                    rows.append(
                        {
                            "ex_date": exdt,
                            "ticker": None,
                            "permno": r.get("permno"),
                            "action_type": "split" if ratio >= 1 else "reverse_split",
                            "ratio": ratio,
                            "cash_amount": None,
                            "new_ticker": None,
                            "as_of": as_of,
                        }
                    )
                elif pd.notna(divamt):
                    rows.append(
                        {
                            "ex_date": exdt,
                            "ticker": None,
                            "permno": r.get("permno"),
                            "action_type": "dividend",
                            "ratio": None,
                            "cash_amount": float(divamt),
                            "new_ticker": None,
                            "as_of": as_of,
                        }
                    )

        changes = self._fetch_ticker_changes_raw(start, end)
        if changes is not None and not changes.empty:
            changes = changes.sort_values(["permno", "namedt"]).reset_index(drop=True)
            for permno, group in changes.groupby("permno"):
                group = group.sort_values("namedt").reset_index(drop=True)
                for i in range(1, len(group)):
                    old_ticker = group.loc[i - 1, "ticker"]
                    new_ticker = group.loc[i, "ticker"]
                    ex_date = group.loc[i, "namedt"]
                    if old_ticker == new_ticker:
                        continue
                    if not (pd.Timestamp(start) <= pd.Timestamp(ex_date) <= pd.Timestamp(end)):
                        continue
                    rows.append(
                        {
                            "ex_date": ex_date,
                            "ticker": old_ticker,
                            "permno": permno,
                            "action_type": "ticker_change",
                            "ratio": None,
                            "cash_amount": None,
                            "new_ticker": new_ticker,
                            # CRSP's `dsenames` has no separate
                            # announcement date for a ticker change -- the
                            # transition's own effective date (`namedt`) is
                            # the earliest date this adapter can defensibly
                            # claim the fact was knowable.
                            "as_of": ex_date,
                        }
                    )

        delist = self._fetch_delistings_raw(start, end)
        if delist is not None and not delist.empty:
            delist = delist.reset_index(drop=True)
            for _, r in delist.iterrows():
                dlstdt = r.get("dlstdt")
                rows.append(
                    {
                        "ex_date": dlstdt,
                        "ticker": None,
                        "permno": r.get("permno"),
                        "action_type": "delisting",
                        "ratio": None,
                        # `dlret` is a RETURN (fraction), not a dollar
                        # amount -- reusing `cash_amount` here is a
                        # best-effort placement since the frozen contract
                        # has no dedicated field for a delisting return.
                        "cash_amount": r.get("dlret"),
                        "new_ticker": None,
                        "as_of": dlstdt,
                    }
                )

        df = pd.DataFrame(
            rows, columns=CORPORATE_ACTIONS_COLUMNS + ["permno"]
        )
        conformed = _conform(df, CORPORATE_ACTIONS_COLUMNS)
        conformed["permno"] = df["permno"].values if not df.empty else pd.Series(dtype="float64")
        return conformed

    def ticker_meta(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Point-in-time listing metadata from `crsp.dsenames`.

        `namedt`/`nameendt` are genuinely point-in-time `active_from`/
        `active_to` ranges (CRSP updates them exactly when a name/ticker/
        exchange/share-code change takes effect) -- unlike Polygon's bulk
        endpoint, which only reflects each ticker's LATEST known
        classification. `as_of` is pinned to `active_from` (`namedt`),
        never to `end`, per TICKER_META_COLUMNS.

        `market_cap` is `abs(prc) * shrout * 1000` (CRSP `shrout` is in
        THOUSANDS of shares) using the `crsp.dsf` row on `namedt` itself --
        a best-effort snapshot at listing time, NaN when no `dsf` row
        exists for that exact date (e.g. a non-trading day). `float_shares`
        is always NaN: CRSP's standard files do not carry a float-shares
        figure.
        """
        raw = self._fetch_ticker_meta_raw(start, end)
        if raw is None or raw.empty:
            df = pd.DataFrame(columns=TICKER_META_COLUMNS)
            df["permno"] = pd.Series(dtype="float64")
            return df

        raw = raw.reset_index(drop=True)
        shrcd = pd.to_numeric(raw["shrcd"], errors="coerce")
        security_type = shrcd.map(_SHRCD_SECURITY_TYPE).fillna("OTHER")

        exchcd = pd.to_numeric(raw["exchcd"], errors="coerce")
        exchange = exchcd.abs().map(_EXCHCD_EXCHANGE).fillna("OTC")

        prc = pd.to_numeric(raw["prc"], errors="coerce")
        shrout = pd.to_numeric(raw["shrout"], errors="coerce")
        # `shrout` is in THOUSANDS of shares in CRSP's file spec.
        market_cap = prc.abs() * shrout * 1000.0

        namedt = pd.to_datetime(raw["namedt"])
        nameendt = pd.to_datetime(raw["nameendt"])

        out = pd.DataFrame(
            {
                "ticker": raw["ticker"],
                "name": raw["comnam"],
                "security_type": security_type,
                "exchange": exchange,
                "active_from": namedt,
                "active_to": nameendt,
                "market_cap": market_cap,
                # CRSP's standard daily/monthly files do not carry a
                # float-shares figure -- always NaN, never omitted.
                "float_shares": float("nan"),
                # Pinned to `active_from` (`namedt`), never to `end` --
                # see docstring / TICKER_META_COLUMNS.
                "as_of": namedt,
            }
        )
        conformed = _conform(out, TICKER_META_COLUMNS)
        conformed["permno"] = raw["permno"].values
        return conformed

    def earnings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError(
            "CRSP does not carry an earnings calendar. Use a supplemental "
            "earnings-calendar source -- e.g. "
            "top10.data.free_tier.FinnhubEarnings or an Alpha Vantage "
            "earnings-calendar adapter -- for this method."
        )

    def premarket_bars(self, trade_date: dt.date, tickers: list[str]) -> pd.DataFrame:
        raise NotImplementedError(
            "CRSP's standard equity database has no intraday data, and the "
            "separate CRSP intraday product is often not included in a "
            "university subscription. T1 (prior-close decisions) can run "
            "entirely on CRSP; T2 (09:25 ET premarket decisions) cannot. "
            "Use top10.data.free_tier's Alpaca adapter (free, IEX feed, "
            "2016+) or top10.data.databento.DatabentoSource.premarket_bars "
            "as the T2 source."
        )

    def short_interest(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError(
            "CRSP does not carry FINRA short-interest figures. Use "
            "top10.data.polygon.PolygonSource.short_interest (Polygon's "
            "FINRA short-interest feed) for this method."
        )
