"""Point-in-time raw_symbol <-> instrument_id resolution for Databento.

VERIFIED FINDING (live run against real Databento data): Databento
reassigns `instrument_id` DAILY for equities. Spot-checking one liquid,
rarely-reassigned name (AAPL stayed `27` throughout) hid this; checking
less liquid names exposed it immediately -- LULU's `instrument_id` across
seven consecutive trading sessions was `6844, 6843, 6839, 6840, 6841, 6837,
6837`; MRNA's was `7345, 7343, 7339, 7340, 7342, 7338, 7340`. There is no
sense in which "the `instrument_id` for LULU" is a single, range-stable
value the way CRSP's PERMNO is.

**This means a RANGE-WIDE `symbology.resolve(start_date=..., end_date=...)`
that returns one `{d0, d1, instrument_id}` interval spanning many days is
WRONG for equities on this vendor.** Labeling `instrument_id` 6844 as
"LULU" for every day in a multi-day range is exactly as wrong as labeling
it "LULU" forever -- on the very next day, `6844` may be assigned to a
completely different company. In a live test this produced a median
top-10-gainer return of **+8702%** and rows where two unrelated tickers
shared an identical close, because the wrong company's price series got
spliced onto the wrong ticker.

THE FIX: resolution must happen PER TRADING DAY -- `start_date=D`,
`end_date=D+1` -- and the result must be keyed `(trade_date,
instrument_id) -> raw_symbol`, never `instrument_id -> raw_symbol` alone.
:meth:`SymbolResolver.resolve_day` is the ONLY sanctioned way to do this
resolution in this codebase; there is deliberately no range-wide
`resolve_range` method left to reach for by accident (see "REMOVED" note
below). Instrument ids are batched in groups of 500 per day (verified
working against the live API) since a full day's cross-section can run
into the thousands of distinct ids.

Each call goes through Databento's `client.symbology.resolve()` HTTP
endpoint, which is a free metadata-style call -- it does NOT return
timeseries data and therefore does not go through
:class:`top10.data.cost_guard.CostGuard`. It is, however, slow (one HTTP
round trip per batch of <=500 ids per day), so every `(dataset, day)`
result is cached to disk via :func:`top10.data.cache.cached_call` under
namespace ``databento/symbology/<dataset>`` and NEVER re-resolved once
cached -- a cached day is reused forever, exactly like every other vendor
payload in this project.

REMOVED: the old `resolve_range(symbols, start, end)` method, which
persisted a `raw_symbol -> [{"d0", "d1", "s"}, ...]` interval map spanning
an arbitrary date range and treated each interval as valid for its entire
span. That is the exact shape of the +8702% bug once inverted to
`instrument_id -> raw_symbol` for a whole-universe pull, and even in its
original `raw_symbol -> instrument_id` direction it is not safe to trust
for more than a single day for the same reason (a `raw_symbol`'s bound
`instrument_id` can change day to day, not just when the company itself
changes). It has been deleted, not merely deprecated, so it cannot be
reached by accident. :meth:`resolve_at` (kept only for
`top10.data.composite.CompositeSource` backward compatibility) is now a
thin, explicitly single-day shim -- see its docstring.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

# Re-exported (not used directly in this module any more -- resolution now
# goes through `top10.data.cache.cached_call`, which reads its own
# `top10.data.cache.DATA_RAW`) purely so existing test suites that
# monkeypatch `top10.data.symbology.DATA_RAW` (e.g. `tests/test_composite.py`,
# not owned by this change) keep working unchanged.
from top10.config import DATA_RAW  # noqa: F401
from top10.data.cache import cached_call


class AmbiguousSymbolError(RuntimeError):
    """Raised when a point-in-time lookup for a SINGLE trading day matches
    more than one distinct counterpart (e.g. an `instrument_id` bound to
    two different raw symbols within the same one-day resolve window).

    This should not happen with well-formed Databento symbology data for a
    single-day range, but a corrupted/hand-edited cache file could still
    produce it -- refusing loudly here is safer than silently picking one.
    """


def _batched(items: list[str], size: int = 500) -> list[list[str]]:
    """Split ``items`` into chunks of at most ``size`` -- Databento's
    `symbology.resolve` is verified working with batches of 500 ids."""
    return [items[i : i + size] for i in range(0, len(items), size)]


class SymbolResolver:
    """Per-trading-day `instrument_id <-> raw_symbol` resolution for one
    Databento dataset (e.g. ``"XNAS.ITCH"``).

    Nothing touches the network at construction time -- resolution only
    happens when :meth:`resolve_day` (or the legacy single-day
    :meth:`resolve_at` shim) is called, and even then only for days not
    already covered by the on-disk cache.
    """

    def __init__(self, dataset: str, client: Any = None) -> None:
        self.dataset = dataset
        self._client = client

    # -- the sanctioned per-day resolution path -------------------------------

    def resolve_day(
        self,
        day: dt.date,
        instrument_ids: list[Any],
        *,
        client: Any = None,
    ) -> dict[str, str]:
        """Resolve ``instrument_ids`` to their raw ticker symbols AS OF
        ``day`` ONLY -- see module docstring for why this must never span
        more than one trading day for Databento equities.

        Returns ``{str(instrument_id): raw_symbol}``, omitting any id
        Databento could not resolve for this exact day (never guessing).
        Cached to disk per ``(dataset, day)`` -- a repeat call for the same
        day makes zero network calls, and days already fully cached (see
        the cache-key note below) are read straight from disk.
        """
        client = client or self._client
        if client is None:
            raise ValueError("resolve_day requires a Databento client (pass client=...)")

        ids = sorted({str(i) for i in instrument_ids if i is not None})
        if not ids:
            return {}

        cache_key = day.isoformat()

        def _fetch() -> dict[str, str]:
            mapping: dict[str, str] = {}
            for batch in _batched(ids, 500):
                response = client.symbology.resolve(
                    dataset=self.dataset,
                    symbols=batch,
                    stype_in="instrument_id",
                    stype_out="raw_symbol",
                    start_date=day.isoformat(),
                    end_date=(day + dt.timedelta(days=1)).isoformat(),
                )
                result = response.get("result", response)
                for instrument_id, intervals in result.items():
                    if not intervals:
                        continue
                    if len(intervals) > 1:
                        raise AmbiguousSymbolError(
                            f"instrument_id {instrument_id!r} resolved to "
                            f"{len(intervals)} distinct raw symbols within the "
                            f"single day {day.isoformat()}: {intervals} -- the "
                            "cached/returned symbology data is corrupted for "
                            "this day."
                        )
                    interval = intervals[0]
                    symbol = interval.get("s") if isinstance(interval, dict) else interval
                    if symbol is not None:
                        mapping[str(instrument_id)] = symbol
            return mapping

        return cached_call(f"databento/symbology/{self.dataset}", cache_key, _fetch)

    # -- legacy shim: kept ONLY for CompositeSource backward compatibility --

    def resolve_at(
        self, symbol: str, date: dt.date, *, client: Any = None
    ) -> str | None:
        """Point-in-time raw_symbol -> instrument_id, resolved fresh for
        exactly ``[date, date + 1 day)`` -- NEVER a persisted range-wide
        interval (see module docstring / REMOVED note for why that is
        wrong). Kept only so `top10.data.composite.CompositeSource`, which
        calls this for cross-vendor `instrument_id` alignment, keeps
        working; it is a single-symbol, single-day resolve, not a bulk
        path, and is NOT how `DatabentoSource.daily_bars` resolves tickers
        (that uses :meth:`resolve_day`, the id -> symbol direction, in
        daily batches of <=500).

        Returns ``None`` (and makes NO network call) if no client is
        available -- matching the historical behavior of this method,
        which only ever read a persisted map that nothing in this codebase
        currently populates outside of tests.
        """
        client = client or self._client
        if client is None:
            return None

        cache_key = f"{date.isoformat()}_{symbol}"

        def _fetch() -> dict[str, Any]:
            response = client.symbology.resolve(
                dataset=self.dataset,
                symbols=[symbol],
                stype_in="raw_symbol",
                stype_out="instrument_id",
                start_date=date.isoformat(),
                end_date=(date + dt.timedelta(days=1)).isoformat(),
            )
            result = response.get("result", response)
            intervals = result.get(symbol, [])
            if len(intervals) > 1:
                raise AmbiguousSymbolError(
                    f"symbol {symbol!r} resolved to {len(intervals)} distinct "
                    f"instrument_ids within the single day {date.isoformat()}: "
                    f"{intervals}."
                )
            interval = intervals[0] if intervals else None
            instrument_id = (
                interval.get("s") if isinstance(interval, dict) else interval
            )
            return {"instrument_id": instrument_id}

        payload = cached_call(
            f"databento/symbology_reverse/{self.dataset}", cache_key, _fetch
        )
        return payload.get("instrument_id") if payload else None
