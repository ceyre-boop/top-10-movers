"""Point-in-time raw_symbol <-> instrument_id resolution for Databento.

Databento identifies instruments by an integer ``instrument_id``, NOT by the
raw ticker string. Raw symbols are reused across unrelated companies over
time -- a delisted ticker is routinely reassigned to a new, unrelated
issuer years later. Resolving purely on ticker string therefore risks
silently splicing two different companies' price histories into one
series -- the same class of error CRSP's PERMNO exists to prevent (see
``top10/data/crsp.py``).

:class:`SymbolResolver` builds and persists a point-in-time map of
``raw_symbol -> [{"d0": start_date, "d1": end_date, "s": instrument_id}, ...]``
via Databento's ``symbology.resolve`` HTTP endpoint (``client.symbology.
resolve()``), which is a free metadata-style call -- it does NOT return
timeseries data and therefore does not go through
:class:`top10.data.cost_guard.CostGuard`. Once resolved, the map is
persisted to ``data/raw/databento/symbology/<dataset>.json`` and reused --
re-resolving an already-covered ``(symbol, date range)`` is wasted spend
(rate-limited API calls) even though the calls themselves are free.

Each interval's ``d1`` (end date) is treated as EXCLUSIVE, matching
Databento's own ``symbology.resolve`` date-range convention.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from top10.config import DATA_RAW


class AmbiguousSymbolError(RuntimeError):
    """Raised when a point-in-time lookup matches more than one interval.

    This should not happen with well-formed Databento symbology data (a
    raw symbol's resolved intervals are expected to be disjoint in time),
    but a corrupted/hand-edited persisted map could produce overlapping
    intervals -- refusing loudly here is safer than silently picking one.
    """


def _map_path(dataset: str) -> Path:
    # `DATA_RAW` is looked up at call time (not baked into a module-level
    # constant at import time) so tests can monkeypatch
    # `top10.data.symbology.DATA_RAW`, same pattern as `top10.data.cache`.
    safe = dataset.replace("/", "_").replace(":", "_")
    return Path(DATA_RAW) / "databento" / "symbology" / f"{safe}.json"


class SymbolResolver:
    """Point-in-time raw_symbol <-> instrument_id map for one dataset.

    Backed by Databento's ``definition``/``symbology.resolve`` data.
    Nothing touches the network or the filesystem at construction time --
    call :meth:`load` (or let :meth:`resolve_range` do it implicitly) to
    read a previously-persisted map, and :meth:`resolve_range` to fetch and
    persist new coverage.
    """

    def __init__(self, dataset: str, client: Any = None) -> None:
        self.dataset = dataset
        self._client = client
        # raw_symbol -> [{"d0": "YYYY-MM-DD", "d1": "YYYY-MM-DD", "s": "<instrument_id>"}, ...]
        self._intervals: dict[str, list[dict[str, str]]] = {}
        self._loaded = False

    # -- persistence ---------------------------------------------------------

    def _path(self) -> Path:
        return _map_path(self.dataset)

    def load(self) -> None:
        """Read the persisted map from disk, if present. Never raises for a
        missing file -- an empty/unresolved map is the correct starting
        state for a dataset that has never been resolved before."""
        path = self._path()
        if path.exists():
            with path.open("r") as f:
                data = json.load(f)
            self._intervals = {k: list(v) for k, v in data.get("intervals", {}).items()}
        self._loaded = True

    def _save(self) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"dataset": self.dataset, "intervals": self._intervals}, f, indent=2)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # -- resolution ------------------------------------------------------------

    def resolve_range(
        self,
        symbols: list[str],
        start: dt.date,
        end: dt.date,
        *,
        client: Any = None,
        force: bool = False,
    ) -> None:
        """Resolve ``symbols`` for ``[start, end)`` and persist the result.

        Symbols already covered by a persisted interval are skipped unless
        ``force=True`` -- re-resolving is wasted spend (see module
        docstring), even though each individual call is free.
        """
        self._ensure_loaded()
        client = client or self._client
        if client is None:
            raise ValueError("resolve_range requires a Databento client (pass client=...)")

        to_resolve = symbols if force else [s for s in symbols if s not in self._intervals]
        if not to_resolve:
            return

        response = client.symbology.resolve(
            dataset=self.dataset,
            symbols=to_resolve,
            stype_in="raw_symbol",
            stype_out="instrument_id",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        result = response.get("result", response)
        for symbol, intervals in result.items():
            self._intervals[symbol] = list(intervals)
        self._save()

    def resolve_at(self, symbol: str, date: dt.date) -> str | None:
        """Return the ``instrument_id`` bound to ``symbol`` on ``date``, or
        ``None`` if unresolved for that date. Raises
        :class:`AmbiguousSymbolError` if more than one interval matches."""
        self._ensure_loaded()
        matches = self._matching_intervals(self._intervals.get(symbol, []), date)
        if not matches:
            return None
        if len(matches) > 1:
            raise AmbiguousSymbolError(
                f"symbol {symbol!r} resolves to {len(matches)} distinct "
                f"instrument_ids on {date.isoformat()}: {matches} -- the "
                "persisted symbology map has overlapping intervals."
            )
        return matches[0]

    def symbol_at(self, instrument_id: str, date: dt.date) -> str | None:
        """Inverse of :meth:`resolve_at`: the raw symbol bound to
        ``instrument_id`` on ``date``, or ``None`` if unresolved. Raises
        :class:`AmbiguousSymbolError` if more than one raw symbol maps to
        the same ``instrument_id`` on that date (should not happen -- an
        instrument_id is Databento's stable identifier)."""
        self._ensure_loaded()
        instrument_id = str(instrument_id)
        candidates: set[str] = set()
        for symbol, intervals in self._intervals.items():
            for iv in intervals:
                if str(iv.get("s")) == instrument_id and self._in_interval(iv, date):
                    candidates.add(symbol)
        if not candidates:
            return None
        if len(candidates) > 1:
            raise AmbiguousSymbolError(
                f"instrument_id {instrument_id!r} resolves to {len(candidates)} "
                f"distinct raw symbols on {date.isoformat()}: {sorted(candidates)}"
            )
        return next(iter(candidates))

    @staticmethod
    def _in_interval(interval: dict[str, str], date: dt.date) -> bool:
        d = pd.Timestamp(date)
        d0 = pd.Timestamp(interval["d0"])
        d1 = pd.Timestamp(interval["d1"])
        # `d1` is EXCLUSIVE, matching Databento's own convention.
        return d0 <= d < d1

    def _matching_intervals(self, intervals: list[dict[str, str]], date: dt.date) -> list[str]:
        return [str(iv["s"]) for iv in intervals if self._in_interval(iv, date)]

    # -- reuse detection ---------------------------------------------------

    def detect_reuse(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Every raw symbol bound to more than one ``instrument_id`` within
        ``[start, end)``. Symbol reuse (a delisted ticker reassigned to a
        new, unrelated issuer) is a real phenomenon on US exchanges, not an
        edge case -- this makes it visible rather than silently correct.

        Columns: ``raw_symbol``, ``instrument_id``, ``interval_start``,
        ``interval_end``.
        """
        self._ensure_loaded()
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

        rows: list[dict[str, Any]] = []
        for symbol, intervals in self._intervals.items():
            in_window = []
            distinct_ids: set[str] = set()
            for iv in intervals:
                d0, d1 = pd.Timestamp(iv["d0"]), pd.Timestamp(iv["d1"])
                # Overlap test against [start_ts, end_ts).
                if d1 <= start_ts or d0 >= end_ts:
                    continue
                in_window.append((d0, d1, str(iv["s"])))
                distinct_ids.add(str(iv["s"]))
            if len(distinct_ids) > 1:
                for d0, d1, instrument_id in in_window:
                    rows.append(
                        {
                            "raw_symbol": symbol,
                            "instrument_id": instrument_id,
                            "interval_start": d0,
                            "interval_end": d1,
                        }
                    )

        return pd.DataFrame(
            rows, columns=["raw_symbol", "instrument_id", "interval_start", "interval_end"]
        ).sort_values(["raw_symbol", "interval_start"]).reset_index(drop=True)
