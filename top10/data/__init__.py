"""Vendor-agnostic market data package.

Swapping vendors is a one-line config change: set ``MARKET_DATA_VENDOR`` (or
pass ``vendor=`` explicitly) and call :func:`get_source`. Adapter modules are
imported lazily inside :func:`get_source` so that simply importing
``top10.data`` never requires an API key or a vendor SDK to be installed.
"""

from __future__ import annotations

import pandas as pd

from top10.data.base import (
    CORPORATE_ACTIONS_COLUMNS,
    DAILY_BARS_COLUMNS,
    EARNINGS_COLUMNS,
    PREMARKET_BARS_COLUMNS,
    SHORT_INTEREST_COLUMNS,
    TICKER_META_COLUMNS,
    MarketDataSource,
)

__all__ = [
    "CORPORATE_ACTIONS_COLUMNS",
    "DAILY_BARS_COLUMNS",
    "EARNINGS_COLUMNS",
    "PREMARKET_BARS_COLUMNS",
    "SHORT_INTEREST_COLUMNS",
    "TICKER_META_COLUMNS",
    "MarketDataSource",
    "get_source",
]

_VALID_VENDORS = ("polygon", "databento", "crsp")

# Known dtype for every column name that appears across the frozen contracts
# in top10/data/base.py. Columns not listed here are left as-is (object).
_DTYPES: dict[str, str] = {
    "trade_date": "datetime64[ns]",
    "ex_date": "datetime64[ns]",
    "as_of": "datetime64[ns]",
    "active_from": "datetime64[ns]",
    "active_to": "datetime64[ns]",
    "report_date": "datetime64[ns]",
    "announced_on": "datetime64[ns]",
    "minute": "datetime64[ns]",
    "settlement_date": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "dollar_volume": "float64",
    "ratio": "float64",
    "cash_amount": "float64",
    "trade_count": "float64",
    "market_cap": "float64",
    "float_shares": "float64",
    "short_interest_shares": "float64",
    "short_interest_pct_float": "float64",
    "days_to_cover": "float64",
    "date_is_revisable": "bool",
}


def _conform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reindex ``df`` to exactly ``columns`` (that order) with contract dtypes.

    Raises ``ValueError`` if a required column is missing from ``df`` --
    adapters must build a fully-populated frame (using NaN/None for
    legitimately-absent-per-row fields like ``new_ticker``) rather than
    omitting a column outright.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"adapter contract violation: missing required column(s) {missing}"
        )

    out = df.reindex(columns=columns).copy()
    for col in columns:
        dtype = _DTYPES.get(col)
        if dtype is None:
            continue
        if dtype == "datetime64[ns]":
            out[col] = pd.to_datetime(out[col])
        elif dtype == "bool":
            # `.astype("bool")` on an object column is NOT a safe way to
            # coerce missing values: `None` silently becomes `False` but
            # `np.nan` silently becomes `True` (pandas treats a float NaN
            # as "truthy" under `bool()`). A bool-contract column
            # (currently only `date_is_revisable`) must never carry an
            # ambiguous missing value in the first place -- fail loud here
            # rather than let an adapter bug flip a row's meaning
            # depending on which "missing" sentinel it happened to use.
            if out[col].isna().any():
                raise ValueError(
                    f"adapter contract violation: column {col!r} has missing "
                    "value(s) in a bool-typed contract column; the adapter "
                    "must populate an explicit True/False for every row "
                    "(np.nan/None are not safely coercible to bool)."
                )
            out[col] = out[col].astype("bool")
        else:
            out[col] = out[col].astype(dtype)
    return out


def get_source(vendor: str | None = None) -> MarketDataSource:
    """Return a :class:`MarketDataSource` for ``vendor``.

    ``vendor`` defaults to the ``MARKET_DATA_VENDOR`` environment variable,
    falling back to ``"polygon"``. No network call or API key lookup happens
    here -- adapters only touch the network on their first real request.
    """
    import os

    resolved = vendor or os.environ.get("MARKET_DATA_VENDOR") or "polygon"
    resolved = resolved.lower()

    if resolved == "polygon":
        from top10.data.polygon import PolygonSource

        return PolygonSource()
    if resolved == "databento":
        from top10.data.databento import DatabentoSource

        return DatabentoSource()
    if resolved == "crsp":
        from top10.data.crsp import CRSPSource

        return CRSPSource()

    raise ValueError(
        f"unknown market data vendor {resolved!r}; valid vendors: {', '.join(_VALID_VENDORS)}"
    )
