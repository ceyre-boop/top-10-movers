"""Bars-only T1 feature builder — the EXP-003/EXP-004 reproducibility repro.

EXP-003 and EXP-004 ran from scratch scripts that no longer exist on disk.
Their 15 feature names (`xs_vol_rank`, `xs_ret_rank`, `top90`, `logpx`,
`dist52`, `vol20`, `vol5`, `relvol_prev`, `ret5`, `ret20`, `volofvol`,
`logdv`, `prev_ret1`, `was_top`, `top30`) appear nowhere else in the repo.
This module rebuilds them from a SINGLE input --
`data/raw/databento/preholdout/universe_liquidity.parquet` -- pure pandas,
no vendor calls, no `ticker_meta`/`earnings`/`short_interest` dependency
(none of that exists on disk for this universe, which is exactly why
`top10.features.t1.build_t1_features` cannot be reused here).

Decision time is T1: 16:00 ET on the trade date's own prior TRADING day
(`prev_date`, not a naive calendar-day subtraction -- across a weekend or
holiday, `trade_date - 1 calendar day` is not a trading day at all).
`prev_date`/`prev_close` are already carried on the input frame per-row,
so this is stamped directly rather than recomputed.

Every feature is built to be knowable strictly before that decision time:
any column on the input frame that represents a quantity REALIZED during
`trade_date` itself (`ret`, `dollar_volume`, `volume`, `label`, `close`)
is shifted by one row (per ticker, sorted by `trade_date`) before it
contributes to a feature for that row -- so a feature stamped for
`trade_date` only ever reflects data through `prev_date`'s close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed order -- `top10.features.spec.T1B_COLUMNS` locks this same order.
BARS_T1_FEATURES: tuple[str, ...] = (
    "xs_vol_rank",
    "xs_ret_rank",
    "top90",
    "logpx",
    "dist52",
    "vol20",
    "vol5",
    "relvol_prev",
    "ret5",
    "ret20",
    "volofvol",
    "logdv",
    "prev_ret1",
    "was_top",
    "top30",
)

_REQUIRED_COLUMNS = (
    "trade_date",
    "ticker",
    "close",
    "dollar_volume",
    "adv20",
    "ret",
    "label",
    "prev_date",
)


def _per_ticker_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute all 15 features for one ticker's rows, sorted by trade_date.

    Every rolling/pct_change computation is built on a series that is
    itself shifted by one row FIRST wherever the base column represents a
    same-day-realized quantity (`close`, `ret`, `dollar_volume`, `label`)
    -- so the value landing on row `trade_date=T` only ever reflects data
    through `T`'s own `prev_date`.
    """
    grp = grp.sort_values("trade_date")

    close = grp["close"]
    ret = grp["ret"]
    dollar_volume = grp["dollar_volume"]
    adv20 = grp["adv20"]
    label = grp["label"].astype(float)

    shifted_close = close.shift(1)
    shifted_ret = ret.shift(1)

    out = pd.DataFrame(index=grp.index)

    # -- return-based ---------------------------------------------------
    out["ret5"] = close.pct_change(5).shift(1)
    out["ret20"] = close.pct_change(20).shift(1)
    out["prev_ret1"] = shifted_ret

    # -- volatility -------------------------------------------------------
    out["vol5"] = shifted_ret.rolling(window=5, min_periods=3).std()
    out["vol20"] = shifted_ret.rolling(window=20, min_periods=10).std()
    out["volofvol"] = out["vol5"].rolling(window=20, min_periods=10).std()

    # -- liquidity / volume -------------------------------------------------
    out["relvol_prev"] = (dollar_volume / adv20).shift(1)
    out["logdv"] = np.log1p(dollar_volume.shift(1))

    # -- price level / distance from high ------------------------------
    out["logpx"] = np.log(shifted_close)
    roll_max_252 = shifted_close.rolling(window=252, min_periods=60).max()
    out["dist52"] = shifted_close / roll_max_252 - 1.0

    # -- label history ----------------------------------------------------
    out["was_top"] = label.shift(1)
    out["top30"] = label.rolling(window=30, min_periods=1).sum().shift(1)
    out["top90"] = label.rolling(window=90, min_periods=1).sum().shift(1)

    return out


def build_bars_t1(universe: pd.DataFrame) -> pd.DataFrame:
    """Build the 15 EXP-003/EXP-004 T1 features from `universe` alone.

    `universe`: the `universe_liquidity.parquet`-shaped frame -- one row
    per (trade_date, ticker), already point-in-time filtered upstream
    (liquidity/price floors, split days, degraded sessions all excluded
    before this function ever sees a row -- see EXP-003's "Setup").

    Returns one row per input row, columns `trade_date, ticker,
    <BARS_T1_FEATURES...>, as_of` -- `as_of` is 16:00 ET on that row's own
    `prev_date` (the T1 decision time), stamped directly from the frame's
    own `prev_date` column rather than a naive calendar-day subtraction,
    since `trade_date - 1 calendar day` is not always a trading day.

    Early rows (insufficient trailing history for a given ticker, e.g. the
    first 60 sessions for `dist52`) get NaN for the features that need
    more history than is yet available -- this function does not drop
    them; a caller doing model fit/eval is expected to drop NaN rows
    itself (this mirrors EXP-003's own "after feature warmup" step).
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in universe.columns]
    if missing:
        raise KeyError(f"build_bars_t1: universe frame is missing required column(s) {missing}")

    if universe.empty:
        return pd.DataFrame(columns=["trade_date", "ticker", *BARS_T1_FEATURES, "as_of"])

    parts = []
    for _, grp in universe.groupby("ticker", sort=False):
        parts.append(_per_ticker_features(grp))
    features = pd.concat(parts)

    out = universe[["trade_date", "ticker", "prev_date"]].join(features)

    # -- cross-sectional ranks (within trade_date, across tickers) --------
    out["xs_ret_rank"] = out.groupby("trade_date")["prev_ret1"].rank(pct=True)
    out["xs_vol_rank"] = out.groupby("trade_date")["vol20"].rank(pct=True)

    out["as_of"] = pd.to_datetime(out["prev_date"]) + pd.Timedelta(hours=16)

    out = out[["trade_date", "ticker", *BARS_T1_FEATURES, "as_of"]].reset_index(drop=True)
    return out
