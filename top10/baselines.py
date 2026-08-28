"""Baselines to beat — docs/PREREG_TOP10.md "Baselines to beat" and plan §3.1.

Every baseline returns a predictions frame with columns
`trade_date, ticker, score` (higher score == more likely to be a top-10
mover; `top10.metrics` takes the top-k by score descending).

Every baseline uses ONLY information available at its decision time and
enforces that via `top10.leakage.assert_decision_time_safe` before scoring.

Decision-time convention (docs/LABEL_SPEC.md / PREREG_TOP10.md):
- T1 ("prior_close"): 16:00 ET on trade_date - 1 calendar day.
- T2 ("premarket"): 09:25 ET on trade_date itself.

These are calendar-day approximations of the real trading-calendar cutoff
(weekends/holidays aren't modeled here); they are strictly conservative
for the leakage check's purpose -- a row that passes `as_of <= t1(trade_date)`
also passes the true trading-calendar cutoff, since the true prior trading
day's close is never later than the calendar day before.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from top10.experiment import assert_frame_holdout_sealed
from top10.leakage import assert_decision_time_safe

K_DEFAULT = 10

# T2 decision time is 09:25 ET -- the 09:25:00-09:25:59 bar itself is NOT
# yet knowable at the decision instant, so the cutoff is exclusive. Must
# agree exactly with `top10.features.t2._PREMARKET_CUTOFF` / the
# `minute < cutoff` clip in `top10.features.t2.build_t2_features`.
_PREMARKET_CUTOFF_HOUR = 9
_PREMARKET_CUTOFF_MINUTE = 25


def _t1_decision_time(trade_date: pd.Timestamp) -> dt.datetime:
    ts = pd.Timestamp(trade_date)
    return (ts - pd.Timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0).to_pydatetime()


def _t2_decision_time(trade_date: pd.Timestamp) -> dt.datetime:
    ts = pd.Timestamp(trade_date)
    return ts.replace(hour=9, minute=25, second=0, microsecond=0).to_pydatetime()


def _close_decision_time(trade_date: pd.Timestamp) -> dt.datetime:
    """A row dated `trade_date` cannot be knowable before that day's own
    16:00 ET close -- used to sanity-check frames indexed by their own
    "as of this day's close" semantics (e.g. a labels frame), as opposed to
    `_t1_decision_time` which is the cutoff for USING that day's data to
    predict the NEXT trading day."""
    ts = pd.Timestamp(trade_date)
    return ts.replace(hour=16, minute=0, second=0, microsecond=0).to_pydatetime()


def _assert_safe_per_day(df: pd.DataFrame, decision_time_fn, date_col: str = "trade_date") -> None:
    """Group `df` by `date_col` and check each group against its own
    decision time (T1 or T2 depending on `decision_time_fn`)."""
    if df.empty:
        return
    for trade_date, group in df.groupby(date_col):
        assert_decision_time_safe(group, decision_time_fn(trade_date))


def _top_n_per_day(scored: pd.DataFrame, n: int) -> pd.DataFrame:
    if scored.empty:
        return scored[["trade_date", "ticker", "score"]].copy()
    ranked = scored.sort_values(["trade_date", "score", "ticker"], ascending=[True, False, True])
    ranked["_pos"] = ranked.groupby("trade_date").cumcount()
    return ranked[ranked["_pos"] < n][["trade_date", "ticker", "score"]].reset_index(drop=True)


# --- B0: random -------------------------------------------------------------


def b0_random(universe: pd.DataFrame, seed: int, k: int = K_DEFAULT, *, unseal_token: str | None = None) -> pd.DataFrame:
    """Uniform-random top-k per day. Reproducible via `seed`.

    `universe`: trade_date, ticker, as_of (candidate universe for the day,
    already point-in-time filtered per docs/LABEL_SPEC.md "Universe").
    """
    assert_frame_holdout_sealed(universe, unseal_token=unseal_token)
    _assert_safe_per_day(universe, _t1_decision_time)

    rows = []
    for trade_date, group in universe.groupby("trade_date"):
        # Per-day, per-seed reproducible RNG so results don't depend on
        # global RNG state or row order across days.
        day_seed = (hash((seed, pd.Timestamp(trade_date).value)) % (2**32))
        rng = np.random.default_rng(day_seed)
        scores = rng.random(len(group))
        rows.append(pd.DataFrame({"trade_date": trade_date, "ticker": group["ticker"].to_numpy(), "score": scores}))

    if not rows:
        return pd.DataFrame(columns=["trade_date", "ticker", "score"])
    scored = pd.concat(rows, ignore_index=True)
    return _top_n_per_day(scored, k)


# --- B1: yesterday's top 10 repeated -----------------------------------------


def b1_yesterday_repeat(labels: pd.DataFrame, k: int = K_DEFAULT, *, unseal_token: str | None = None) -> pd.DataFrame:
    """Predict day t's top-10 as day t-1's top-10, scored by day t-1's rank
    (rank 1 -> highest score). `labels`: trade_date, ticker, rank, return_t,
    label, label_spec_version, as_of.

    Day 1 (no prior trading day present in `labels`) degrades gracefully:
    it is simply omitted from the output rather than raising.
    """
    assert_frame_holdout_sealed(labels, unseal_token=unseal_token)
    _assert_safe_per_day(labels, _close_decision_time)

    trade_dates = sorted(labels["trade_date"].unique())
    if len(trade_dates) < 2:
        return pd.DataFrame(columns=["trade_date", "ticker", "score"])

    rows = []
    for prev_date, next_date in zip(trade_dates[:-1], trade_dates[1:]):
        prev_top = labels[(labels["trade_date"] == prev_date) & (labels["label"] == 1)]
        if prev_top.empty:
            continue
        max_rank = prev_top["rank"].max()
        score = (max_rank + 1) - prev_top["rank"]
        rows.append(pd.DataFrame({"trade_date": next_date, "ticker": prev_top["ticker"].to_numpy(), "score": score.to_numpy()}))

    if not rows:
        return pd.DataFrame(columns=["trade_date", "ticker", "score"])
    scored = pd.concat(rows, ignore_index=True)
    return _top_n_per_day(scored, k)


# --- B2: highest 5-day realized volatility -----------------------------------


def _rolling_realized_vol_scores(bars: pd.DataFrame, window: int) -> pd.DataFrame:
    """For each ticker, compute realized vol (std of daily returns) over a
    trailing `window`-day window ending at bar date `d`, and attribute that
    score to the NEXT trading day in the shared calendar (the earliest day
    it could legally be used to predict, per T1).

    Returns trade_date, ticker, score, as_of (as_of == bar date `d`, i.e.
    when the score became knowable).
    """
    if bars.empty:
        return pd.DataFrame(columns=["trade_date", "ticker", "score", "as_of"])

    trade_dates = sorted(bars["trade_date"].unique())
    next_date = {trade_dates[i]: trade_dates[i + 1] for i in range(len(trade_dates) - 1)}

    bars = bars.sort_values(["ticker", "trade_date"]).copy()
    bars["return_1d"] = bars.groupby("ticker")["close"].pct_change()
    bars["realized_vol"] = (
        bars.groupby("ticker")["return_1d"].rolling(window=window, min_periods=window).std().reset_index(level=0, drop=True)
    )

    scored = bars.dropna(subset=["realized_vol"]).copy()
    scored["predict_for"] = scored["trade_date"].map(next_date)
    scored = scored.dropna(subset=["predict_for"])

    out = scored[["predict_for", "ticker", "realized_vol", "as_of"]].rename(
        columns={"predict_for": "trade_date", "realized_vol": "score"}
    )
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out.reset_index(drop=True)


def b2_realized_vol(
    features_or_bars: pd.DataFrame, k: int = K_DEFAULT, window: int = 5, *, unseal_token: str | None = None
) -> pd.DataFrame:
    """Top-k by trailing `window`-day realized volatility of daily returns.
    `features_or_bars`: DAILY_BARS_COLUMNS-shaped frame (trade_date, ticker,
    ..., close, ..., as_of)."""
    assert_frame_holdout_sealed(features_or_bars, unseal_token=unseal_token)
    scored = _rolling_realized_vol_scores(features_or_bars, window)
    _assert_safe_per_day(scored, _t1_decision_time)
    return _top_n_per_day(scored, k)


# --- B3: earnings-today x highest 20-day vol ---------------------------------


def b3_earnings_x_vol(
    bars: pd.DataFrame,
    earnings: pd.DataFrame,
    k: int = K_DEFAULT,
    window: int = 20,
    *,
    unseal_token: str | None = None,
) -> pd.DataFrame:
    """Restrict the candidate set to tickers reporting earnings ON the
    predicted trade_date, then rank by trailing `window`-day realized vol.

    `earnings`: EARNINGS_COLUMNS-shaped frame (ticker, report_date, ...,
    as_of).
    """
    assert_frame_holdout_sealed(bars, unseal_token=unseal_token)
    if not earnings.empty:
        assert_frame_holdout_sealed(
            earnings.rename(columns={"report_date": "trade_date"}), unseal_token=unseal_token
        )
    scored = _rolling_realized_vol_scores(bars, window)
    _assert_safe_per_day(scored, _t1_decision_time)

    if not earnings.empty:
        _assert_safe_per_day(
            earnings.rename(columns={"report_date": "trade_date"}), _t1_decision_time
        )

    earnings_today = earnings[["ticker", "report_date"]].rename(columns={"report_date": "trade_date"})
    earnings_today = earnings_today.astype({"trade_date": "datetime64[ns]", "ticker": "object"})
    restricted = scored.merge(earnings_today, on=["trade_date", "ticker"], how="inner")
    return _top_n_per_day(restricted, k)


# --- B4: premarket gap % (T2 only) -------------------------------------------


def b4_premarket_gap(
    premarket_bars: pd.DataFrame,
    prior_close: pd.DataFrame,
    min_premarket_dollar_vol: float = 500_000,
    k: int = K_DEFAULT,
    *,
    unseal_token: str | None = None,
) -> pd.DataFrame:
    """T2 baseline: top-k by premarket gap %, subject to a premarket
    dollar-volume floor. This is the baseline the model must beat -- keep
    it exact.

    `premarket_bars`: PREMARKET_BARS_COLUMNS-shaped (trade_date, ticker,
    minute, open, high, low, close, volume, trade_count, as_of), NOMINALLY
    restricted to 04:00-09:25 ET by the caller/vendor -- but this baseline
    NEVER trusts that windowing contract (see `top10.features.t2`, which
    states the same policy) and always re-clips to `minute < 09:25 ET`
    itself before aggregating. PREREG_TOP10's primary success claim is
    "beats B4 by >= 1.0 average hits/day", so a single leaked post-cutoff
    bar silently inflating B4 would invalidate that claim in either
    direction -- the clip is therefore enforced here unconditionally,
    independent of whatever the caller happens to hand in.
    `prior_close`: trade_date, ticker, close, as_of -- the PRIOR trading
    day's close, indexed by the trade_date being PREDICTED (i.e. already
    shifted forward by the caller so a simple join works).
    """
    assert_frame_holdout_sealed(premarket_bars, unseal_token=unseal_token)
    assert_frame_holdout_sealed(prior_close, unseal_token=unseal_token)
    _assert_safe_per_day(premarket_bars, _t2_decision_time)
    _assert_safe_per_day(prior_close, _t1_decision_time)

    if premarket_bars.empty:
        return pd.DataFrame(columns=["trade_date", "ticker", "score"])

    pm = premarket_bars.sort_values(["trade_date", "ticker", "minute"]).copy()
    # Re-clip to `minute < 09:25 ET` ourselves -- the boundary is exclusive
    # (the 09:25:00-09:25:59 bar is not yet knowable at the 09:25 decision
    # instant) and must agree with `top10.features.t2.build_t2_features`'s
    # `cutoff = trade_date_ts + pd.Timedelta(hours=9, minutes=25)` clip.
    cutoff = pd.to_datetime(pm["trade_date"]) + pd.Timedelta(
        hours=_PREMARKET_CUTOFF_HOUR, minutes=_PREMARKET_CUTOFF_MINUTE
    )
    pm = pm[pm["minute"] < cutoff]
    pm["dollar_vol"] = pm["close"] * pm["volume"]

    agg = pm.groupby(["trade_date", "ticker"]).agg(
        last_premarket_close=("close", "last"),
        premarket_dollar_vol=("dollar_vol", "sum"),
    ).reset_index()

    merged = agg.merge(prior_close[["trade_date", "ticker", "close"]].rename(columns={"close": "prior_close"}), on=["trade_date", "ticker"], how="inner")
    merged = merged[merged["premarket_dollar_vol"] >= min_premarket_dollar_vol]
    merged["score"] = (merged["last_premarket_close"] / merged["prior_close"] - 1.0) * 100.0

    return _top_n_per_day(merged, k)


def run_all_baselines(
    *,
    universe: pd.DataFrame,
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    earnings: pd.DataFrame,
    premarket_bars: pd.DataFrame,
    prior_close: pd.DataFrame,
    seed: int = 0,
    min_premarket_dollar_vol: float = 500_000,
    k: int = K_DEFAULT,
    unseal_token: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every baseline and return {"B0": ..., "B1": ..., ..., "B4": ...}.

    `unseal_token` is forwarded to every individual baseline -- each one
    independently refuses (`top10.experiment.assert_holdout_sealed`) to
    compute on holdout-dated (>= 2023-01-01) input without it, so this
    aggregator cannot be used to read the sealed holdout for free.
    """
    return {
        "B0": b0_random(universe, seed=seed, k=k, unseal_token=unseal_token),
        "B1": b1_yesterday_repeat(labels, k=k, unseal_token=unseal_token),
        "B2": b2_realized_vol(bars, k=k, unseal_token=unseal_token),
        "B3": b3_earnings_x_vol(bars, earnings, k=k, unseal_token=unseal_token),
        "B4": b4_premarket_gap(
            premarket_bars,
            prior_close,
            min_premarket_dollar_vol=min_premarket_dollar_vol,
            k=k,
            unseal_token=unseal_token,
        ),
    }
