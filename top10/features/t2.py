"""T2 feature builder -- decision time = 09:25 ET on trade_date.

Per plan §4.2. T2 output is T1's columns plus premarket-derived columns,
all as of 09:25 ET. The hard requirement here: every premarket
aggregation MUST be clipped to `minute < 09:25 ET` -- this module never
trusts the vendor's own 04:00-09:25 windowing contract (see
`top10/data/base.py` PREMARKET_BARS_COLUMNS docstring) and re-filters
itself, because a single leaked 09:30+ bar would be a direct P3 leak
into what's supposed to be a pre-open decision.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from top10.features.spec import PRIOR_CLOSE_COLUMNS, T2_COLUMNS
from top10.features.t1 import decision_time_t1
from top10.leakage import assert_decision_time_safe, assert_self_exclusion

_PREMARKET_CUTOFF = dt.time(9, 25)


def decision_time_t2(trade_date: dt.date | dt.datetime | pd.Timestamp) -> pd.Timestamp:
    """09:25 ET on `trade_date`."""
    return pd.Timestamp(trade_date) + pd.Timedelta(hours=9, minutes=25)


def _prior_close_lookup(
    prior_close: pd.DataFrame | pd.Series, trade_date_ts: pd.Timestamp, max_as_of: pd.Timestamp
) -> pd.Series:
    """Column contract: `top10.features.spec.PRIOR_CLOSE_COLUMNS` -- `close`
    is the PRIOR trading day's close, indexed by the trade_date being
    PREDICTED (already shifted forward by the caller so a simple join
    works). This is the SAME contract `top10.baselines.b4_premarket_gap`
    requires (see its docstring).

    Defect 2 (CONFIRMED): `pipeline.build_features_step` and this function
    previously disagreed on the column name (`prior_close` vs `close`),
    which crashed the only T2 orchestration path with `KeyError:
    'prior_close'` on every real call. `close` is now the single agreed
    contract for both consumers -- `baselines.b4_premarket_gap` already
    required it, so `top10.baselines` needs no change.

    A bare `pd.Series` (ticker -> close) is still accepted directly for
    callers/tests that already have a per-ticker lookup and don't need the
    `as_of` guard below.
    """
    if isinstance(prior_close, pd.Series):
        return prior_close

    required = set(PRIOR_CLOSE_COLUMNS)
    missing = required - set(prior_close.columns)
    if missing:
        raise ValueError(f"_prior_close_lookup: prior_close frame missing columns {missing}")

    day_rows = prior_close[prior_close["trade_date"] == trade_date_ts]
    # Defect 2 (CONFIRMED): `build_t2_features` never validated `prior_close`'s
    # `as_of`, so a caller passing day-t closes as "prior_close" would
    # silently get a wrong `premarket_gap_pct` and no error. A prior close
    # must have been knowable by the T1 decision time (16:00 ET on t-1);
    # anything stamped later cannot legitimately be "prior" close.
    if not day_rows.empty:
        assert_decision_time_safe(day_rows, max_as_of)
    return day_rows.set_index("ticker")["close"]


def _halted_tickers(halts: pd.DataFrame, trade_date_ts: pd.Timestamp, decision_time: pd.Timestamp) -> set:
    if halts is None or halts.empty:
        return set()
    df = halts[halts["trade_date"] == trade_date_ts]
    if "as_of" in df.columns:
        df = df[df["as_of"] <= decision_time]
    return set(df["ticker"])


def _premarket_features(pm: pd.DataFrame, trade_date_ts: pd.Timestamp, adv_20: float, prior_close_px: float) -> dict:
    empty = {
        "premarket_gap_pct": np.nan,
        "premarket_dollar_volume": np.nan,
        "premarket_rel_volume": np.nan,
        "premarket_high_to_last_drawdown": np.nan,
        "premarket_trade_count": 0,
        "premarket_first_trade_minutes": np.nan,
    }
    if pm.empty:
        return empty

    pm = pm.sort_values("minute")
    last_price = float(pm["close"].iloc[-1])
    pm_high = float(pm["high"].max())
    total_volume = float(pm["volume"].sum())
    dollar_volume = float((pm["close"] * pm["volume"]).sum())
    trade_count = int(pm["trade_count"].sum())

    gap_pct = (
        last_price / prior_close_px - 1.0
        if prior_close_px not in (None,) and not pd.isna(prior_close_px) and prior_close_px != 0
        else np.nan
    )
    rel_volume = (
        total_volume / adv_20 if adv_20 is not None and not pd.isna(adv_20) and adv_20 != 0 else np.nan
    )
    drawdown = last_price / pm_high - 1.0 if pm_high != 0 else np.nan

    session_open = trade_date_ts + pd.Timedelta(hours=4)
    first_minute = pm["minute"].iloc[0]
    first_trade_minutes = float((first_minute - session_open).total_seconds() / 60.0)

    return {
        "premarket_gap_pct": gap_pct,
        "premarket_dollar_volume": dollar_volume,
        "premarket_rel_volume": rel_volume,
        "premarket_high_to_last_drawdown": drawdown,
        "premarket_trade_count": trade_count,
        "premarket_first_trade_minutes": first_trade_minutes,
    }


def build_t2_features(
    t1_features: pd.DataFrame,
    premarket_bars: pd.DataFrame,
    prior_close: pd.DataFrame | pd.Series,
    halts: pd.DataFrame,
    trade_date: dt.date | dt.datetime | pd.Timestamp,
    *,
    labels_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build T2 features: T1 columns plus premarket-derived columns, all
    re-stamped to the T2 decision time (09:25 ET on `trade_date`).

    `labels_history`, if given and it carries rows for `trade_date`, is
    used to run `assert_self_exclusion` against the output before
    returning (TOP FINDING fix -- see the call site below). Optional and
    keyword-only so existing positional call sites are unaffected.
    """
    trade_date_ts = pd.Timestamp(trade_date)
    decision_time = decision_time_t2(trade_date_ts)

    cutoff = trade_date_ts + pd.Timedelta(hours=9, minutes=25)
    pm_all = premarket_bars[premarket_bars["trade_date"] == trade_date_ts]
    pm_all = pm_all[pm_all["minute"] < cutoff]
    if "as_of" in pm_all.columns:
        pm_all = pm_all[pm_all["as_of"] <= decision_time]

    prior_close_series = _prior_close_lookup(prior_close, trade_date_ts, decision_time_t1(trade_date_ts))
    halted = _halted_tickers(halts, trade_date_ts, decision_time)

    out = t1_features.copy()
    out["as_of"] = decision_time

    pm_by_ticker = dict(tuple(pm_all.groupby("ticker")))

    extra_rows = []
    for _, r in out.iterrows():
        ticker = r["ticker"]
        pm = pm_by_ticker.get(ticker, pm_all.iloc[0:0])
        adv_20 = r.get("adv_20", np.nan)
        prior_close_px = prior_close_series.get(ticker, np.nan)

        pm_feats = _premarket_features(pm, trade_date_ts, adv_20, prior_close_px)
        pm_feats["overnight_halt_flag"] = int(ticker in halted)
        extra_rows.append(pm_feats)

    extra_df = pd.DataFrame(extra_rows, index=out.index)
    out = pd.concat([out, extra_df], axis=1)

    out = out[list(T2_COLUMNS)].reset_index(drop=True)

    # TOP FINDING (adversarial audit): `assert_decision_time_safe` and
    # `assert_self_exclusion` had ZERO production call sites -- the
    # anti-leakage harness was dead code. This is the production gate: no
    # T2 feature row may leave this function with `as_of` after
    # `decision_time`, or self-exclusion-correlated with a same-day label.
    assert_decision_time_safe(out, decision_time)

    if labels_history is not None and not labels_history.empty:
        same_day_labels = labels_history[labels_history["trade_date"] == trade_date_ts]
        if not same_day_labels.empty:
            assert_self_exclusion(out, same_day_labels)

    return out
