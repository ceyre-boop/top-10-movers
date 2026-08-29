"""Composite market-data source: routes each `MarketDataSource` capability
to whichever already-wired-up vendor actually covers it, explicitly and
inspectably.

WHY THIS EXISTS (closing the P4 gap): the user has no WRDS/CRSP access.
Databento (`top10.data.databento.DatabentoSource`) solves P2 (survivorship
bias) for daily bars via a full `ALL_SYMBOLS` cross-section, but it carries
no splits/dividends/ticker-change feed at all --
`DatabentoSource.corporate_actions` raises `NotImplementedError`. Without
corporate actions, `top10.labels.build_labels` cannot exclude split days
(docs/LABEL_SPEC.md "Corporate-action exclusions"), and a 1:20 reverse
split reads as a +1900% return -- every label set built without it is
contaminated. No single free source covers everything this project needs,
so `CompositeSource` composes several, one method at a time:

    daily_bars, premarket_bars -> Databento   (P2-safe cross-section, 2018-05-01+)
    corporate_actions          -> Polygon     (free-tier reference endpoints)
    ticker_meta                -> Polygon     (free reference list, cross-checked
                                                against Databento symbology)
    earnings                   -> Finnhub     (`top10.data.free_tier.FinnhubEarnings`)
    short_interest              -> Polygon    (if the plan allows it, else raises)

See `ROUTING` / `describe_routing()` for the exact, inspectable table --
silent routing is how you end up not knowing what your data actually is.

RATE LIMITS / CACHING: every delegate call in this module goes through
`top10.data.cache.cached_call` (see `_cached_frame`), same as every other
adapter in this package -- a second call for an already-fetched
`(capability, start, end)` range performs ZERO network requests, it is
re-read from disk. The `PolygonSource` delegate used here is constructed
with `calls_per_min=5` to match Polygon's actual free-tier ceiling (see
`_get_polygon`), on top of the disk cache -- fetch once, cache to disk,
respect the limiter, never re-download.

CROSS-VENDOR TICKER ALIGNMENT (the real risk): Databento identifies
instruments by `instrument_id`; Polygon identifies them by ticker string.
Joining Polygon's splits onto Databento's bars by ticker string alone, with
no regard for ticker reuse, applies the wrong split to the wrong company
whenever a ticker has been reassigned to a different, unrelated issuer.
`corporate_actions()` and `ticker_meta()` therefore attach a point-in-time
`instrument_id` column via `top10.data.symbology.SymbolResolver`
(resolved AT each row's own date, never a single blanket resolution for
the whole range), and `alignment_report()` surfaces every ticker that
could NOT be confidently resolved -- those rows are never silently
dropped, only flagged, since dropping them would look exactly like "this
ticker had no corporate actions", the single most dangerous silent
failure available here.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import Any, Callable

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
from top10.data.cache import cached_call
from top10.data.free_tier import MissingApiKey

logger = logging.getLogger(__name__)


class CapabilityUnavailable(RuntimeError):
    """Raised by `CompositeSource` when a routed capability cannot actually
    be served -- either the delegate vendor has no implementation
    (`NotImplementedError`) or its API key/env var is unconfigured.

    `CompositeSource` NEVER falls back to returning an empty frame in this
    situation. An empty `corporate_actions` frame means, to every
    downstream consumer, "no splits ever happened" -- which would silently
    disable the P4 split-exclusion tripwire in `top10.labels.build_labels`
    and is the single most dangerous silent failure this module can make.
    """


# --- ROUTING: the explicit, inspectable dispatch table ----------------------

ROUTING: dict[str, dict[str, str]] = {
    "daily_bars": {
        "vendor": "databento",
        "env_var": "DATABENTO_API_KEY",
        "reason": (
            "P2-safe full daily cross-section (ALL_SYMBOLS, incl. delisted "
            "names), 2018-05-01+."
        ),
    },
    "premarket_bars": {
        "vendor": "databento",
        "env_var": "DATABENTO_API_KEY",
        "reason": "full-tape 04:00-09:25 ET minute bars, 2018-05-01+.",
    },
    "corporate_actions": {
        "vendor": "polygon",
        "env_var": "POLYGON_API_KEY",
        "reason": (
            "/v3/reference/splits + /v3/reference/dividends -- small "
            "reference endpoints, available on Polygon's free tier."
        ),
    },
    "ticker_meta": {
        "vendor": "polygon",
        "env_var": "POLYGON_API_KEY",
        "reason": (
            "free bulk reference ticker list; cross-checked against "
            "Databento's point-in-time symbology for instrument_id alignment."
        ),
    },
    "earnings": {
        "vendor": "finnhub",
        "env_var": "FINNHUB_API_KEY",
        "reason": "FinnhubEarnings free-tier calendar (top10.data.free_tier).",
    },
    "short_interest": {
        "vendor": "polygon",
        "env_var": "POLYGON_API_KEY",
        "reason": (
            "/stocks/v1/short-interest, plan-dependent -- raises "
            "NotImplementedError (via the Polygon delegate itself) when the "
            "account's plan does not include it."
        ),
    },
}


def describe_routing() -> str:
    """Human-readable routing table: which vendor answers which question,
    and why. See module docstring -- silent routing is how you end up not
    knowing what your data is."""
    header = f"{'capability':<20}{'vendor':<12}{'env var':<24}reason"
    lines = [header, "-" * len(header) * 2]
    for capability, info in ROUTING.items():
        lines.append(
            f"{capability:<20}{info['vendor']:<12}{info['env_var']:<24}{info['reason']}"
        )
    return "\n".join(lines)


class CompositeSource:
    """Routes each `MarketDataSource` method to the best available vendor.
    Implements `MarketDataSource` structurally, same pattern as every other
    adapter in this package.

    Every delegate (`DatabentoSource`, `PolygonSource`, `FinnhubEarnings`,
    `SymbolResolver`) is constructed LAZILY -- simply constructing
    `CompositeSource` or calling `daily_bars` never requires a Finnhub key,
    and a missing Finnhub key breaks ONLY `earnings`.
    """

    name = "composite"
    ROUTING = ROUTING

    def __init__(self) -> None:
        self._databento: Any = None
        self._polygon: Any = None
        self._finnhub: Any = None
        self._resolver: Any = None
        # (start_key, end_key) -> sorted list of tickers that could not be
        # confidently aligned to an instrument_id on their last
        # `corporate_actions` computation -- see `alignment_report`.
        self._last_unaligned: dict[tuple[str, str], list[str]] = {}

    def describe_routing(self) -> str:
        return describe_routing()

    # -- lazy delegate construction ------------------------------------------

    def _get_databento(self) -> Any:
        if self._databento is None:
            from top10.data.databento import DatabentoSource

            self._databento = DatabentoSource()
        return self._databento

    def _get_polygon(self) -> Any:
        if self._polygon is None:
            from top10.data.polygon import PolygonSource

            # Free-tier reference endpoints are rate-limited to 5 calls/min
            # -- see module docstring.
            self._polygon = PolygonSource(calls_per_min=5)
        return self._polygon

    def _get_finnhub(self) -> Any:
        if self._finnhub is None:
            from top10.data.free_tier import FinnhubEarnings

            self._finnhub = FinnhubEarnings()
        return self._finnhub

    def _get_resolver(self) -> Any:
        if self._resolver is None:
            from top10.data.databento import DATASET
            from top10.data.symbology import SymbolResolver

            self._resolver = SymbolResolver(DATASET)
        return self._resolver

    # -- capability dispatch: never silently degrade to empty ---------------

    def _require_env(self, capability: str) -> dict[str, str]:
        info = ROUTING[capability]
        env_var = info["env_var"]
        if env_var and not os.environ.get(env_var):
            raise CapabilityUnavailable(
                f"CompositeSource.{capability} routes to {info['vendor']} "
                f"(see CompositeSource.describe_routing()), which requires "
                f"{env_var} to be set -- it is not. A missing key must never "
                "silently degrade to an empty result here: for "
                "`corporate_actions` in particular, an empty frame would "
                "look exactly like 'no splits ever happened' and would "
                "silently disable the P4 split-exclusion tripwire in "
                "top10.labels.build_labels."
            )
        return info

    def _call_delegate(self, capability: str, fn: Callable[[], Any]) -> Any:
        """Invoke `fn` (a bound delegate method call), translating a
        delegate's own `NotImplementedError`/`MissingApiKey` into a
        `CapabilityUnavailable` that names the ROUTED capability and
        vendor, never returning an empty frame in their place."""
        info = self._require_env(capability)
        try:
            return fn()
        except NotImplementedError as exc:
            raise CapabilityUnavailable(
                f"CompositeSource.{capability} routes to {info['vendor']}, "
                f"but the delegate raised NotImplementedError: {exc}"
            ) from exc
        except MissingApiKey as exc:
            raise CapabilityUnavailable(
                f"CompositeSource.{capability} routes to {info['vendor']}, "
                f"but its API key is not configured: {exc}"
            ) from exc

    # -- disk-cached delegate calls (fetch-once, never re-download) ---------

    def _cached_frame(
        self,
        capability: str,
        key: str,
        fetch_fn: Callable[[], pd.DataFrame],
        required_columns: list[str],
    ) -> pd.DataFrame:
        """Round-trip `fetch_fn()`'s DataFrame through the existing on-disk
        JSON cache (`top10.data.cache.cached_call`) -- a second call for the
        same `(capability, key)` re-reads from disk and makes ZERO network
        calls. Columns beyond `required_columns` (e.g. `instrument_id`) are
        preserved through the round trip.
        """

        def _fetch_records() -> list[dict[str, Any]]:
            df = self._call_delegate(capability, fetch_fn)
            if df is None:
                df = pd.DataFrame(columns=required_columns)
            return json.loads(df.to_json(orient="records", date_format="iso"))

        records = cached_call(f"composite/{capability}", key, _fetch_records)
        df = pd.DataFrame(records) if records else pd.DataFrame(columns=required_columns)

        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
        conformed = _conform(df[required_columns], required_columns)
        for col in df.columns:
            if col not in required_columns:
                conformed[col] = df[col].values
        return conformed

    # -- cross-vendor ticker alignment (Databento instrument_id, point-in-time) --

    def _resolve_instrument_id(self, ticker: Any, as_of_date: Any) -> str | None:
        from top10.data.symbology import AmbiguousSymbolError

        if ticker is None or (isinstance(ticker, float) and pd.isna(ticker)):
            return None
        if as_of_date is None or pd.isna(as_of_date):
            return None
        resolver = self._get_resolver()
        try:
            return resolver.resolve_at(str(ticker), pd.Timestamp(as_of_date).date())
        except AmbiguousSymbolError:
            return None

    def _attach_corporate_action_alignment(
        self, df: pd.DataFrame, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        """Attach a point-in-time `instrument_id` column to `df` (a
        `corporate_actions` frame), resolved AT each row's own `ex_date` --
        NOT a single blanket resolution for the whole range. This is what
        prevents a ticker reused across two unrelated companies from
        having the wrong company's split applied when later joined onto
        Databento's `instrument_id`-keyed `daily_bars`.

        Rows that cannot be confidently aligned are kept (never dropped)
        and recorded for `alignment_report`.
        """
        instrument_ids: list[str | None] = []
        unaligned: set[str] = set()
        for ticker, ex_date in zip(df["ticker"], df["ex_date"]):
            iid = self._resolve_instrument_id(ticker, ex_date)
            instrument_ids.append(iid)
            if iid is None and ticker is not None and not (
                isinstance(ticker, float) and pd.isna(ticker)
            ):
                unaligned.add(str(ticker))

        out = df.copy()
        out["instrument_id"] = instrument_ids
        self._last_unaligned[self._range_key(start, end)] = sorted(unaligned)
        return out

    def _attach_ticker_meta_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same idea as `_attach_corporate_action_alignment`, applied to
        `ticker_meta` rows, resolved AT each row's own `active_from`."""
        instrument_ids = [
            self._resolve_instrument_id(ticker, active_from)
            for ticker, active_from in zip(df["ticker"], df["active_from"])
        ]
        out = df.copy()
        out["instrument_id"] = instrument_ids
        return out

    @staticmethod
    def _range_key(start: dt.date, end: dt.date) -> tuple[str, str]:
        return (pd.Timestamp(start).isoformat(), pd.Timestamp(end).isoformat())

    def alignment_report(self, start: dt.date, end: dt.date) -> dict[str, Any]:
        """Run (or re-read the cached result of) `corporate_actions(start,
        end)` and report which tickers' rows could NOT be confidently
        mapped to a Databento `instrument_id`. These rows are still
        present in `corporate_actions`'s own output (never silently
        dropped) -- this is how a caller finds out which ones need manual
        review before joining corporate actions onto `daily_bars`.
        """
        df = self.corporate_actions(start, end)
        unaligned = self._last_unaligned.get(self._range_key(start, end), [])
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "total_rows": int(len(df)),
            "aligned_row_count": int(df["instrument_id"].notna().sum()) if "instrument_id" in df else 0,
            "unaligned_ticker_count": len(unaligned),
            "unaligned_tickers": unaligned,
        }

    # -- MarketDataSource -----------------------------------------------------

    def daily_bars(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        delegate = self._get_databento()
        key = f"{start.isoformat()}_{end.isoformat()}"
        return self._cached_frame(
            "daily_bars", key, lambda: delegate.daily_bars(start, end), DAILY_BARS_COLUMNS
        )

    def corporate_actions(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        delegate = self._get_polygon()
        key = f"{start.isoformat()}_{end.isoformat()}"
        df = self._cached_frame(
            "corporate_actions",
            key,
            lambda: delegate.corporate_actions(start, end),
            CORPORATE_ACTIONS_COLUMNS,
        )
        return self._attach_corporate_action_alignment(df, start, end)

    def ticker_meta(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        delegate = self._get_polygon()
        key = f"{start.isoformat()}_{end.isoformat()}"
        df = self._cached_frame(
            "ticker_meta", key, lambda: delegate.ticker_meta(start, end), TICKER_META_COLUMNS
        )
        return self._attach_ticker_meta_alignment(df)

    def earnings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        delegate = self._get_finnhub()
        key = f"{start.isoformat()}_{end.isoformat()}"
        return self._cached_frame(
            "earnings", key, lambda: delegate.earnings(start, end), EARNINGS_COLUMNS
        )

    def short_interest(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        delegate = self._get_polygon()
        key = f"{start.isoformat()}_{end.isoformat()}"
        return self._cached_frame(
            "short_interest",
            key,
            lambda: delegate.short_interest(start, end),
            SHORT_INTEREST_COLUMNS,
        )

    def premarket_bars(self, trade_date: dt.date, tickers: list[str]) -> pd.DataFrame:
        delegate = self._get_databento()
        key = f"{trade_date.isoformat()}_{'-'.join(sorted(tickers))}"
        return self._cached_frame(
            "premarket_bars",
            key,
            lambda: delegate.premarket_bars(trade_date, tickers),
            PREMARKET_BARS_COLUMNS,
        )
