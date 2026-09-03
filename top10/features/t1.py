"""T1 feature builder -- decision time = prior close (16:00 ET on t-1).

Per plan §4.1. Every feature here must be computable using ONLY
information knowable strictly before `trade_date` -- daily bars, ticker
metadata, earnings, and label history all get filtered to
`as_of <= decision_time_t1` (and, for calendar-day-keyed frames, to
`trade_date < trade_date` as well) before anything is derived from them.

`labels_history` is explicitly called out as the most dangerous input:
a same-day label leaking into "days since last top-10 appearance" would
be a direct instance of the label appearing in its own features (P3).
This module filters `labels_history` to `trade_date < trade_date` (a
strict inequality, not `<=`) before it touches any appearance-derived
column.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from top10.data.base import SHORT_INTEREST_COLUMNS
from top10.features.spec import T1_COLUMNS
from top10.leakage import assert_decision_time_safe, assert_self_exclusion

# --- Constants ----------------------------------------------------------

_PRICE_BINS = [-np.inf, 5, 10, 25, 50, 100, 250, np.inf]
_MCAP_BINS = [-np.inf, 50e6, 300e6, 2e9, 10e9, 200e9, np.inf]
_FLOAT_BINS = [-np.inf, 10e6, 50e6, 200e6, 1e9, np.inf]

# Fixed sector universe -> one-hot column suffix. Anything not in this map
# (including missing/unknown sector) falls into `sector_other`.
_SECTOR_TO_SUFFIX = {
    "communication services": "communication",
    "consumer": "consumer",
    "consumer discretionary": "consumer",
    "consumer staples": "consumer",
    "energy": "energy",
    "financials": "financials",
    "healthcare": "healthcare",
    "health care": "healthcare",
    "industrials": "industrials",
    "materials": "materials",
    "real estate": "realestate",
    "technology": "technology",
    "information technology": "technology",
    "utilities": "utilities",
}
_SECTOR_SUFFIXES = (
    "communication",
    "consumer",
    "energy",
    "financials",
    "healthcare",
    "industrials",
    "materials",
    "realestate",
    "technology",
    "utilities",
)

_APPEARANCE_30D = pd.Timedelta(days=30)
_APPEARANCE_90D = pd.Timedelta(days=90)
_MAX_52W_BARS = 252


def decision_time_t1(trade_date: dt.date | dt.datetime | pd.Timestamp) -> pd.Timestamp:
    """16:00 ET on t-1, expressed relative to `trade_date` midnight
    (which is 8 hours after that prior close)."""
    return pd.Timestamp(trade_date) - pd.Timedelta(hours=8)


def _bucket(value: float, bins: list[float]) -> float:
    if value is None or pd.isna(value):
        return np.nan
    coded = pd.cut(pd.Series([value]), bins=bins, labels=False, right=False)
    return float(coded.iloc[0])


def _sector_onehot_and_biotech(sector: object, industry: object) -> dict[str, float]:
    row = {f"sector_{s}": 0 for s in _SECTOR_SUFFIXES}
    row["sector_other"] = 0

    sector_str = str(sector).strip().lower() if sector is not None and not pd.isna(sector) else None
    industry_str = str(industry).strip().lower() if industry is not None and not pd.isna(industry) else ""

    suffix = _SECTOR_TO_SUFFIX.get(sector_str) if sector_str else None
    if suffix is not None:
        row[f"sector_{suffix}"] = 1
    else:
        row["sector_other"] = 1

    row["is_biotech"] = int("biotech" in industry_str)
    return row


def _streak(rets: np.ndarray) -> float:
    if rets.size == 0:
        return np.nan
    last = rets[-1]
    if pd.isna(last) or last == 0:
        return 0.0
    sign = 1 if last > 0 else -1
    count = 0
    for r in rets[::-1]:
        if pd.isna(r):
            break
        s = 1 if r > 0 else (-1 if r < 0 else 0)
        if s != sign:
            break
        count += 1
    return float(sign * count)


def _latest_pit_row(frame: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
    """One row per ticker: the most recent row with `as_of <= decision_time`.

    This is the single mechanism that both (a) selects the current-best-
    known sector/market-cap/float classification and (b) implements
    "forward-fill only from the publish date, never the settlement date"
    for short-interest fields -- a row published after `decision_time` is
    simply not in the filtered set, so nothing forward-fills from it.
    """
    if frame.empty:
        return frame
    filtered = frame[frame["as_of"] <= decision_time]
    if filtered.empty:
        return filtered
    filtered = filtered.sort_values("as_of")
    return filtered.groupby("ticker", as_index=True).tail(1).set_index("ticker")


def _per_ticker_bar_features(bars: pd.DataFrame) -> dict:
    bars = bars.sort_values("trade_date")
    closes = bars["close"].to_numpy(dtype=float)
    volumes = bars["volume"].to_numpy(dtype=float)
    n = closes.size

    rets = closes[1:] / closes[:-1] - 1.0 if n >= 2 else np.array([])

    ret_1d = rets[-1] if rets.size >= 1 else np.nan
    ret_5d = closes[-1] / closes[-6] - 1.0 if n >= 6 else np.nan
    ret_20d = closes[-1] / closes[-21] - 1.0 if n >= 21 else np.nan

    rvol_5d = float(np.std(rets[-5:], ddof=1)) if rets.size >= 5 else np.nan
    rvol_20d = float(np.std(rets[-20:], ddof=1)) if rets.size >= 20 else np.nan

    vol_of_vol = np.nan
    if rets.size >= 14:
        roll = pd.Series(rets).rolling(5).std(ddof=1).dropna()
        tail = roll.tail(10)
        if len(tail) >= 5:
            vol_of_vol = float(tail.std(ddof=1))

    adv_20 = float(np.mean(volumes[-20:])) if n >= 20 else np.nan
    rel_volume_1d = (
        float(volumes[-1] / adv_20) if n >= 20 and not np.isnan(adv_20) and adv_20 != 0 else np.nan
    )

    rel_volume_5d_trend = np.nan
    if n >= 24:
        rel_vols = []
        for offset in range(5):
            end = n - offset
            window = volumes[end - 20:end]
            day_vol = volumes[end - 1]
            adv_i = np.mean(window)
            rel_vols.append(day_vol / adv_i if adv_i != 0 else np.nan)
        rel_vols = list(reversed(rel_vols))
        if not any(pd.isna(v) for v in rel_vols):
            slope = np.polyfit(np.arange(5), rel_vols, 1)[0]
            rel_volume_5d_trend = float(slope)

    price_bucket = _bucket(closes[-1], _PRICE_BINS) if n >= 1 else np.nan

    window52 = closes[-_MAX_52W_BARS:]
    high52 = float(np.max(window52))
    low52 = float(np.min(window52))
    dist_from_52w_high = closes[-1] / high52 - 1.0 if high52 != 0 else np.nan
    dist_from_52w_low = closes[-1] / low52 - 1.0 if low52 != 0 else np.nan

    consecutive_streak = _streak(rets)

    return {
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "rvol_5d": rvol_5d,
        "rvol_20d": rvol_20d,
        "vol_of_vol": vol_of_vol,
        "rel_volume_1d": rel_volume_1d,
        "rel_volume_5d_trend": rel_volume_5d_trend,
        "adv_20": adv_20,
        "price_bucket": price_bucket,
        "dist_from_52w_high": dist_from_52w_high,
        "dist_from_52w_low": dist_from_52w_low,
        "consecutive_streak": consecutive_streak,
    }


def _appearance_features(
    labels_history: pd.DataFrame,
    ticker: str,
    trade_date_ts: pd.Timestamp,
    decision_time: pd.Timestamp,
) -> dict:
    if labels_history.empty:
        return {"days_since_last_top10": np.nan, "appearances_30d": 0, "appearances_90d": 0}

    # STRICT prior days only: trade_date < trade_date_ts. A same-day label
    # planted for `ticker` must never be visible here (P3 tripwire).
    #
    # Defect 4: `as_of` was never checked here, so a rebuilt/revised label
    # vintage stamped with an `as_of` after `decision_time` would be
    # consumed silently even though it wasn't yet knowable at decision
    # time. The labels agent has made label `as_of` meaningful specifically
    # so this filter can be added.
    prior = labels_history[
        (labels_history["ticker"] == ticker)
        & (labels_history["trade_date"] < trade_date_ts)
        & (labels_history["as_of"] <= decision_time)
    ]
    positives = prior[prior["label"] == 1]

    if positives.empty:
        days_since = np.nan
    else:
        last_appearance = positives["trade_date"].max()
        days_since = float((trade_date_ts - last_appearance).days)

    appearances_30d = int((positives["trade_date"] >= trade_date_ts - _APPEARANCE_30D).sum())
    appearances_90d = int((positives["trade_date"] >= trade_date_ts - _APPEARANCE_90D).sum())

    return {
        "days_since_last_top10": days_since,
        "appearances_30d": appearances_30d,
        "appearances_90d": appearances_90d,
    }


def _earnings_features(
    earnings: pd.DataFrame, ticker: str, trade_date_ts: pd.Timestamp, decision_time: pd.Timestamp
) -> dict:
    empty = {
        "earnings_today": 0,
        "earnings_tomorrow": 0,
        "days_to_earnings": np.nan,
        "earnings_date_revisable": False,
    }
    if earnings.empty:
        return empty

    # Defect 3 (CONFIRMED): filtering on `announced_on <= decision_time`
    # is False for NaT, so every REVISABLE row (the data adapter sets
    # `date_is_revisable = (announced_on is None)`, i.e. `announced_on`
    # is NaT exactly when the row is revisable) was silently dropped
    # BEFORE `earnings_date_revisable` could ever be read -- revisable
    # earnings were invisible rather than flagged. Filter on `as_of`
    # instead: it is the conservative, always-populated knowability
    # timestamp the data adapter stamps on every row (revisable or not),
    # so a revisable row with `announced_on == NaT` is correctly kept
    # (and flagged) as long as it was knowable by `decision_time`.
    known = earnings[
        (earnings["ticker"] == ticker)
        & (earnings["as_of"] <= decision_time)
        & (earnings["report_date"] >= trade_date_ts)
    ]
    if known.empty:
        return empty

    known = known.sort_values("report_date")
    nearest = known.iloc[0]
    days_to_earnings = float((nearest["report_date"] - trade_date_ts).days)

    return {
        "earnings_today": int(nearest["report_date"] == trade_date_ts),
        "earnings_tomorrow": int(nearest["report_date"] == trade_date_ts + pd.Timedelta(days=1)),
        "days_to_earnings": days_to_earnings,
        "earnings_date_revisable": bool(nearest.get("date_is_revisable", False)),
    }


def _market_context_row(
    market_context: pd.DataFrame, trade_date_ts: pd.Timestamp, decision_time: pd.Timestamp
) -> dict:
    empty = {
        "mkt_spy_ret_1d": np.nan,
        "mkt_spy_ret_5d": np.nan,
        "mkt_vix_level": np.nan,
        "mkt_iwm_minus_spy_1d": np.nan,
        "mkt_attention_regime_count": np.nan,
    }
    if market_context.empty:
        return empty

    prior = market_context[
        (market_context["trade_date"] < trade_date_ts) & (market_context["as_of"] <= decision_time)
    ]
    if prior.empty:
        return empty

    row = prior.sort_values("trade_date").iloc[-1]
    return {
        "mkt_spy_ret_1d": float(row.get("spy_ret_1d", np.nan)),
        "mkt_spy_ret_5d": float(row.get("spy_ret_5d", np.nan)),
        "mkt_vix_level": float(row.get("vix_level", np.nan)),
        "mkt_iwm_minus_spy_1d": float(row.get("iwm_minus_spy_1d", np.nan)),
        "mkt_attention_regime_count": float(row.get("movers_10pct_count", np.nan)),
    }


def build_t1_features(
    daily_bars: pd.DataFrame,
    ticker_meta: pd.DataFrame,
    earnings: pd.DataFrame,
    labels_history: pd.DataFrame,
    market_context: pd.DataFrame,
    trade_date: dt.date | dt.datetime | pd.Timestamp,
    short_interest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build T1 features for every ticker with prior-day bar history.

    Decision time is 16:00 ET on t-1 (`decision_time_t1`). Every input
    frame is filtered to that boundary before it contributes to a
    feature. The universe of tickers is every ticker with at least one
    `daily_bars` row strictly before `trade_date`.

    `short_interest` is its OWN frame (`SHORT_INTEREST_COLUMNS`), never
    merged into `ticker_meta` -- see `top10/data/base.py`'s comment: FINRA
    short-interest is a distinct, bi-monthly feed with its own publish-lag
    semantics, deliberately excluded from `TICKER_META_COLUMNS`. Defaults
    to an empty, correctly-shaped frame so callers that have no short
    interest data yet degrade to NaN short-interest features rather than
    crash.
    """
    trade_date_ts = pd.Timestamp(trade_date)
    decision_time = decision_time_t1(trade_date_ts)

    if short_interest is None:
        short_interest = pd.DataFrame(columns=list(SHORT_INTEREST_COLUMNS))

    bars = daily_bars[
        (daily_bars["trade_date"] < trade_date_ts) & (daily_bars["as_of"] <= decision_time)
    ]

    if bars.empty:
        return pd.DataFrame(columns=list(T1_COLUMNS))

    # Defect 6 (CONFIRMED): these columns were previously read via
    # `.get(...)` on a per-ticker Series, which returns None/NaN just as
    # quietly whether the VALUE is missing (legitimate) or the COLUMN
    # itself is absent entirely (a broken/incomplete adapter contract) --
    # the latter must raise loudly, not manufacture permanent silent NaN
    # in production.
    _REQUIRED_META_COLUMNS = (
        "market_cap",
        "float_shares",
    )
    if not ticker_meta.empty:
        missing_meta_cols = [c for c in _REQUIRED_META_COLUMNS if c not in ticker_meta.columns]
        if missing_meta_cols:
            raise KeyError(
                f"build_t1_features: ticker_meta frame is missing required column(s) "
                f"{missing_meta_cols} -- Defect 6 (CONFIRMED): a missing meta column must "
                "raise, not silently produce permanently-NaN features."
            )

    # Defect 3 (CONFIRMED): `short_interest_pct_float` / `days_to_cover`
    # were read off `ticker_meta`, but `TICKER_META_COLUMNS` deliberately
    # excludes them -- they live in `SHORT_INTEREST_COLUMNS`, its own
    # frame. Unlike the `ticker_meta` check above, this check is
    # UNCONDITIONAL on emptiness: an empty `short_interest` frame with the
    # wrong columns is still a contract violation, not "no data yet".
    _REQUIRED_SHORT_INTEREST_COLUMNS = (
        "short_interest_pct_float",
        "days_to_cover",
    )
    missing_si_cols = [
        c for c in _REQUIRED_SHORT_INTEREST_COLUMNS if c not in short_interest.columns
    ]
    if missing_si_cols:
        raise KeyError(
            f"build_t1_features: short_interest frame is missing required column(s) "
            f"{missing_si_cols} -- Defect 3 (CONFIRMED): a missing short_interest column "
            "must raise, even for an empty frame, not silently produce permanently-NaN "
            "features."
        )

    meta_pit = _latest_pit_row(ticker_meta, decision_time)
    si_pit = _latest_pit_row(short_interest, decision_time)
    mkt_row = _market_context_row(market_context, trade_date_ts, decision_time)

    rows = []
    for ticker, grp in bars.groupby("ticker", sort=False):
        bar_feats = _per_ticker_bar_features(grp)
        appear_feats = _appearance_features(labels_history, ticker, trade_date_ts, decision_time)
        earn_feats = _earnings_features(earnings, ticker, trade_date_ts, decision_time)

        meta_row = meta_pit.loc[ticker] if ticker in meta_pit.index else None
        sector = meta_row.get("sector") if meta_row is not None else None
        industry = meta_row.get("industry") if meta_row is not None else None
        market_cap = meta_row.get("market_cap") if meta_row is not None else np.nan
        float_shares = meta_row.get("float_shares") if meta_row is not None else np.nan

        si_row = si_pit.loc[ticker] if ticker in si_pit.index else None
        short_interest_pct = (
            si_row.get("short_interest_pct_float") if si_row is not None else np.nan
        )
        days_to_cover = si_row.get("days_to_cover") if si_row is not None else np.nan

        sector_feats = _sector_onehot_and_biotech(sector, industry)

        row = {
            "trade_date": trade_date_ts,
            "ticker": ticker,
            **bar_feats,
            "mcap_bucket": _bucket(market_cap, _MCAP_BINS),
            "float_bucket": _bucket(float_shares, _FLOAT_BINS),
            **appear_feats,
            **earn_feats,
            **sector_feats,
            "short_interest_pct_float": float(short_interest_pct) if pd.notna(short_interest_pct) else np.nan,
            "days_to_cover": float(days_to_cover) if pd.notna(days_to_cover) else np.nan,
            **mkt_row,
            "as_of": decision_time,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["ret_1d_rank"] = out.groupby("trade_date")["ret_1d"].rank(
        method="min", ascending=False, na_option="keep"
    )

    out = out[list(T1_COLUMNS)].reset_index(drop=True)

    # TOP FINDING (adversarial audit): `assert_decision_time_safe` and
    # `assert_self_exclusion` had ZERO production call sites -- the
    # anti-leakage harness was dead code. This is the production gate:
    # no T1 feature row may leave this function with `as_of` after
    # `decision_time`, or self-exclusion-correlated with a same-day label.
    assert_decision_time_safe(out, decision_time)

    # `labels_history` is the RAW frame the caller passed in, not the
    # `trade_date < trade_date_ts`-filtered `prior` used above -- if it
    # happens to also carry the current day's real labels (e.g. a caller
    # re-checking already-realized history), verify none of today's
    # features are a same-day label leak before returning.
    if not labels_history.empty and "trade_date" in labels_history.columns:
        same_day_labels = labels_history[labels_history["trade_date"] == trade_date_ts]
        if not same_day_labels.empty:
            assert_self_exclusion(out, same_day_labels)

    return out
