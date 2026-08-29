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
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from top10.config import ET, get_api_key
from top10.data import _conform
from top10.data.base import CORPORATE_ACTIONS_COLUMNS, DAILY_BARS_COLUMNS, PREMARKET_BARS_COLUMNS
from top10.data.cache import cached_call
from top10.data.cost_guard import CostGuard, estimate_cost

logger = logging.getLogger(__name__)

DATASET = "XNAS.ITCH"

# Databento's equities history for XNAS.ITCH starts in May 2018. A request
# with an earlier `start` must be rejected loudly -- see module docstring.
FIRST_AVAILABLE_DATE = dt.date(2018, 5, 1)


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
        """UNADJUSTED daily OHLCV via the ``ohlcv-1d`` schema, ``ALL_SYMBOLS``.

        P2 DEFENSE (THE core survivorship-bias fix): this pulls the FULL
        daily cross-section -- every instrument that traded, delisted names
        included -- for each requested trading day, via
        ``symbols="ALL_SYMBOLS"``. It deliberately does NOT resolve a
        ticker list first (e.g. "today's active symbols") and then fetch
        history for those names; that inversion is exactly how
        survivorship bias enters a dataset, since today's ticker list
        silently excludes every name that no longer exists today but
        traded during the requested window. Do NOT "optimize" this into a
        per-symbol pull over a pre-resolved list. See
        :func:`verify_no_survivorship` for the loud check that this
        invariant is actually holding for a given pulled frame.

        `instrument_id` is carried through as an extra column beyond the
        frozen `DAILY_BARS_COLUMNS` contract, exactly as `CRSPSource`
        carries `permno` -- Databento's raw ticker strings are reused
        across unrelated issuers over time (see `top10/data/symbology.py`),
        so `instrument_id` (not `ticker`) is the identifier safe to
        `groupby`/join on across a multi-year window.

        The pull is chunked by calendar month and each chunk is
        independently disk-cached (see `_month_chunks`), which makes a
        multi-year pull resumable: a mid-pull failure (network error,
        budget refusal, etc.) leaves completed months cached, and calling
        `daily_bars` again with the same `start`/`end` only re-fetches the
        months that never finished. Cumulative Databento spend is logged
        after every chunk.

        ``dry_run=True`` returns a dict with the cost estimate and record
        count for the WHOLE `[start, end]` range WITHOUT downloading -- see
        :meth:`_dry_run_result`. Use :meth:`estimate_universe_pull_cost` for
        a per-month cost breakdown that mirrors how the pull is actually
        chunked. ``confirm=True`` is required to proceed with any single
        chunk's request estimated above the guard's confirmation threshold
        (default $5).
        """
        self._require_start_after_history_begins(start)

        if dry_run:
            client = self._get_client()
            return self._dry_run_result(
                client, "ohlcv-1d", start.isoformat(), end.isoformat(), "ALL_SYMBOLS"
            )

        guard = self._cost_guard()
        rows: list[dict[str, Any]] = []

        for chunk_start, chunk_end in _month_chunks(start, end):
            key = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}"

            def _fetch(chunk_start: dt.date = chunk_start, chunk_end: dt.date = chunk_end) -> list[dict]:
                # P2 DEFENSE: ALL_SYMBOLS, never a resolved ticker list --
                # see the method docstring. This is the ONLY `symbols=`
                # value this method may ever pass to `_fetch_bars`.
                return self._fetch_bars(
                    "ohlcv-1d",
                    chunk_start.isoformat(),
                    chunk_end.isoformat(),
                    symbols="ALL_SYMBOLS",
                    confirm=confirm,
                )

            records = cached_call(f"{self.name}/daily_bars", key, _fetch)
            logger.info(
                "databento daily_bars chunk %s -> %s: cumulative spend $%.2f / $%.2f ceiling",
                chunk_start.isoformat(), chunk_end.isoformat(), guard.spent, guard.ceiling_usd,
            )

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
                        "instrument_id": r.get("instrument_id"),
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
        cross-section pull, WITHOUT downloading any data.

        Calls Databento's free `metadata.get_cost()` once per calendar-
        month chunk (mirroring exactly how `daily_bars` itself chunks the
        pull -- see `_month_chunks`), never `timeseries.get_range`. This is
        the number the user should see BEFORE committing to a multi-year
        full-universe pull, which is the single largest spend this project
        can make.
        """
        self._require_start_after_history_begins(start)
        client = self._get_client()

        chunks: list[dict[str, Any]] = []
        total = 0.0
        for chunk_start, chunk_end in _month_chunks(start, end):
            request_params = dict(
                dataset=DATASET,
                schema="ohlcv-1d",
                symbols="ALL_SYMBOLS",
                stype_in="raw_symbol",
                start=chunk_start.isoformat(),
                end=chunk_end.isoformat(),
            )
            cost = estimate_cost(client, **request_params)
            chunks.append(
                {"start": chunk_start.isoformat(), "end": chunk_end.isoformat(), "cost_usd": cost}
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

    THIS IS INFERRED, NOT AUTHORITATIVE. An `instrument_id` that trades
    continuously and then stops appearing, while the broader market (every
    OTHER instrument in `daily_bars`) keeps trading, is flagged as a
    candidate delisting -- but a trading halt, a multi-day data outage on
    Databento's end, or a symbol rename onto a NEW `instrument_id` for the
    same underlying company can each masquerade as a delisting under this
    heuristic just as easily as a real one. There is no dedicated "reason"
    or "delisting return" field here the way CRSP's `dlstcd`/`dlret` carry
    -- only a bare inference and a `confidence` score, never a boolean, so
    a caller cannot mistake "we detected an absence" for "we know why".

    An `instrument_id` is flagged when its last observed `trade_date`
    across the WHOLE `daily_bars` frame is more than `min_gap_days` market
    sessions before the LAST session present anywhere in the frame, AND it
    never reappears before that final session. A short gap that closes
    again (the instrument trades again later in the frame) is treated as a
    halt, not a delisting, and is excluded entirely -- this is what
    distinguishes an ordinary multi-day halt from a genuine stop.

    `inferred_delist_date` is the first market session STRICTLY AFTER the
    instrument's last trade -- i.e. the earliest date on which the
    instrument's absence actually became observable. It is never
    backdated to `last_trade_date` itself (the name legitimately traded
    that day) and never forward-dated past the point where the gap first
    became visible.

    `confidence` scales with how many market sessions have elapsed with no
    reappearance by the end of the frame: a name that vanished 4 sessions
    before the window's last observed session and a name that vanished 400
    sessions before it are both technically "absent", but the latter is
    far more likely a genuine delisting than an in-progress halt that
    simply hadn't resumed by the time this `daily_bars` frame was pulled.
    """
    if daily_bars.empty:
        return pd.DataFrame(columns=INFER_DELISTINGS_COLUMNS)

    has_instrument_id = "instrument_id" in daily_bars.columns and daily_bars["instrument_id"].notna().any()
    key_col = "instrument_id" if has_instrument_id else "ticker"

    sessions = sorted(pd.Index(daily_bars["trade_date"].unique()))
    if len(sessions) < 2:
        # A single-session frame has no "later sessions" to be absent from
        # -- nothing is inferrable.
        return pd.DataFrame(columns=INFER_DELISTINGS_COLUMNS)
    last_session = sessions[-1]
    session_pos = {d: i for i, d in enumerate(sessions)}

    rows: list[dict[str, Any]] = []
    for key, group in daily_bars.groupby(key_col):
        last_trade_date = group["trade_date"].max()
        if last_trade_date == last_session:
            continue  # still present at the end of the window -- not a candidate

        gap_sessions = len(sessions) - 1 - session_pos[last_trade_date]
        if gap_sessions < min_gap_days:
            continue  # short gap -- more consistent with a halt than a delisting

        ticker = group.loc[group["trade_date"] == last_trade_date, "ticker"].iloc[0]
        next_pos = session_pos[last_trade_date] + 1
        inferred_delist_date = sessions[next_pos] if next_pos < len(sessions) else last_trade_date

        # Monotonic, capped confidence: reaches 1.0 once the gap is 4x
        # `min_gap_days` or more. The 4x multiplier is a deliberately
        # conservative, documented choice (not tuned/backtested) -- treat
        # this as a coarse signal for triage, not a calibrated probability.
        confidence = min(1.0, gap_sessions / (min_gap_days * 4))

        rows.append(
            {
                "ticker": ticker,
                "instrument_id": key if key_col == "instrument_id" else None,
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
