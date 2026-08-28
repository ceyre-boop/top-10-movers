"""Universe construction and label building per docs/LABEL_SPEC.md.

See docs/LABEL_SPEC.md §Universe, §Label, and §Corporate-action exclusions.
This module MUST NOT be edited to relax any of those rules; edit the label
spec (frozen, hashed) instead and regenerate labels under the new hash.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Iterable

import pandas as pd

from top10.config import DATA_LABELS, DOCS
from top10.hashing import hash_file
from top10.storage import read_parquet, spec_dir, write_parquet

logger = logging.getLogger(__name__)

# --- Universe eligibility constants -----------------------------------------

_ALLOWED_EXCHANGES = {"XNYS", "XNAS", "XASE"}

# Common stock / ADR are unconditionally eligible (subject to price/volume
# filters). ETFs, warrants, rights, units, and pre-merger SPACs are still
# candidate-evaluated (per spec: "Flag and evaluate, rather than
# automatically keep") -- they are FLAGGED via `flagged_type`, not dropped.
_CORE_TYPES = {"CS", "ADR"}
_FLAGGED_TYPES = {"ETF", "WARRANT", "RIGHT", "UNIT", "SPAC"}
_ALLOWED_TYPES = _CORE_TYPES | _FLAGGED_TYPES

_MIN_PRIOR_CLOSE = 1.00
_MIN_AVG_DOLLAR_VOLUME = 1_000_000.0
_DOLLAR_VOLUME_WINDOW = 20

_SPLIT_ACTION_TYPES = {"split", "reverse_split"}

# P4 (staleness): default number of most-recent trading sessions (strictly
# before `trade_date`) within which a ticker's last bar must fall to be
# treated as a genuine "prior close". `1` means the bar must be dated the
# single most-recent session before `trade_date` -- i.e. no gap at all.
_DEFAULT_MAX_STALENESS_SESSIONS = 1


def _label_spec_version() -> str:
    """Read the frozen label-spec hash at runtime. Never hardcode the hex."""
    return hash_file(DOCS / "LABEL_SPEC.md")


def _prior_close_as_of(trade_date_ts: pd.Timestamp) -> pd.Timestamp:
    """(t-1) 16:00 ET -- the moment the prior close (and therefore the
    day-`t` universe built from it) genuinely becomes determinable.

    `trade_date_ts` is always midnight-naive-ET (per this module's
    convention), so subtracting 8 hours lands on 16:00 ET the calendar day
    before -- the SAME arithmetic used by
    `top10.baselines._t1_decision_time` and
    `top10.features.t1.decision_time_t1`. Universe `as_of` MUST match this
    convention exactly, or `top10.baselines.b0_random`'s
    `assert_decision_time_safe(..., _t1_decision_time)` gate raises on
    every real universe frame (this was Defect 1: stamping `as_of` at
    midnight of `t` is AFTER the prior close, not "as of" it).
    """
    return trade_date_ts - pd.Timedelta(hours=8)


def _close_as_of(trade_date_ts: pd.Timestamp) -> pd.Timestamp:
    """16:00 ET on `trade_date` itself -- the moment day-`t`'s own close
    (and therefore day-`t`'s label) first becomes knowable. Matches
    `top10.baselines._close_decision_time`, which is the gate
    `b1_yesterday_repeat` (and any future label-history consumer) asserts
    labels against.
    """
    return trade_date_ts + pd.Timedelta(hours=16)


def build_universe(
    daily_bars: pd.DataFrame,
    ticker_meta: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    trade_date: dt.date | dt.datetime | pd.Timestamp,
    max_staleness_sessions: int = _DEFAULT_MAX_STALENESS_SESSIONS,
) -> pd.DataFrame:
    """Build the day-`trade_date` candidate universe using ONLY information
    knowable as of the prior close (docs/LABEL_SPEC.md §Universe).

    `corporate_actions` is accepted for point-in-time ticker-change /
    delisting-boundary tracking (§Corporate-action exclusions: "Track
    ticker changes and delisting boundaries point-in-time"); it is not
    otherwise used to filter the universe itself, since split exclusion is
    a label-time (post-ranking-input) rule, not a universe rule.

    `max_staleness_sessions` (Defect 4): a ticker's prior bar must fall
    within this many of the most-recent trading sessions strictly before
    `trade_date` to be treated as a genuine "prior close". Without this
    bound, a name that last traded 5 sessions ago yields a 5-day return at
    label time, competing against one-day returns for a top-10 slot --
    a phantom-mover source. Default `1` requires the bar be dated the
    single most-recent session before `trade_date` (no gap). Names whose
    only prior bar is stale are EXCLUDED (not merely flagged): a stale
    prior_close cannot support the spec's one-day `return_t` definition at
    all, so keeping it in the universe as a normal candidate would just
    reintroduce the phantom-mover bug one layer up; excluded/dropped
    counts are logged for visibility.
    """
    trade_date_ts = pd.Timestamp(trade_date)
    prior_close_cutoff = _prior_close_as_of(trade_date_ts)

    # --- Listing eligibility, point-in-time -----------------------------
    # "as of the prior close" (§Universe) means as_of <= (t-1) 16:00 ET, NOT
    # merely "before midnight of t" -- the latter would admit metadata
    # revisions stamped hours after the close that genuinely defines the
    # universe (Defect 1's root cause, applied consistently here too).
    meta = ticker_meta.copy()
    meta = meta[meta["as_of"] <= prior_close_cutoff]
    meta = meta[meta["exchange"].isin(_ALLOWED_EXCHANGES)]
    meta = meta[meta["security_type"].isin(_ALLOWED_TYPES)]
    meta = meta[meta["active_from"] <= trade_date_ts]
    # P2: include names that are LATER delisted -- only require the name be
    # active as of `trade_date`, never that it remains active afterward.
    still_active = meta["active_to"].isna() | (meta["active_to"] >= trade_date_ts)
    meta = meta[still_active]

    if meta.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "ticker",
                "security_type",
                "exchange",
                "flagged_type",
                "prior_close",
                "avg_dollar_volume_20d",
                "as_of",
            ]
        )

    # De-dup: keep the most-recent-as_of metadata row per ticker.
    meta = meta.sort_values("as_of").drop_duplicates("ticker", keep="last")

    # --- Prior-close / prior 20-day dollar volume, using ONLY bars strictly
    # before `trade_date` AND knowable as of the prior close -- same
    # `<= prior_close_cutoff` reasoning as the metadata filter above.
    bars = daily_bars.copy()
    bars = bars[
        (bars["trade_date"] < trade_date_ts) & (bars["as_of"] <= prior_close_cutoff)
    ]
    bars = bars.sort_values(["ticker", "trade_date"])

    # P4 staleness: the set of the `max_staleness_sessions` most-recent
    # trading sessions (as observed across the whole market in `bars`)
    # strictly before `trade_date`. A ticker's own last bar must land in
    # this set to count as a genuine (non-stale) prior close.
    all_sessions = sorted(bars["trade_date"].unique())
    if max_staleness_sessions > 0:
        allowed_recent_sessions = set(all_sessions[-max_staleness_sessions:])
    else:
        allowed_recent_sessions = set(all_sessions)

    prior_rows = []
    stale_tickers: list[str] = []
    for ticker, grp in bars.groupby("ticker", sort=False):
        grp = grp.sort_values("trade_date")
        last = grp.iloc[-1]
        if last["trade_date"] not in allowed_recent_sessions:
            stale_tickers.append(ticker)
            continue
        window = grp.tail(_DOLLAR_VOLUME_WINDOW)
        prior_rows.append(
            {
                "ticker": ticker,
                "prior_close": last["close"],
                "avg_dollar_volume_20d": window["dollar_volume"].mean(),
            }
        )

    if stale_tickers:
        logger.warning(
            "build_universe[%s]: %d ticker(s) excluded for a stale prior close "
            "(last bar outside the most-recent %d session(s)): %s",
            trade_date_ts.date(), len(stale_tickers), max_staleness_sessions,
            sorted(stale_tickers),
        )

    prior = pd.DataFrame(
        prior_rows, columns=["ticker", "prior_close", "avg_dollar_volume_20d"]
    )

    universe = meta.merge(prior, on="ticker", how="inner")

    # --- Price / liquidity filters ---------------------------------------
    universe = universe[universe["prior_close"] >= _MIN_PRIOR_CLOSE]
    universe = universe[universe["avg_dollar_volume_20d"] >= _MIN_AVG_DOLLAR_VOLUME]

    universe = universe.copy()
    universe["trade_date"] = trade_date_ts
    universe["flagged_type"] = universe["security_type"].isin(_FLAGGED_TYPES)
    # Defect 1: the universe becomes determinable at the prior close, i.e.
    # (t-1) 16:00 ET -- NEVER midnight of `t` (that is AFTER the prior
    # close and fails every downstream `as_of <= (t-1) 16:00` decision-time
    # gate, e.g. `top10.baselines.b0_random`).
    universe["as_of"] = prior_close_cutoff

    return universe[
        [
            "trade_date",
            "ticker",
            "security_type",
            "exchange",
            "flagged_type",
            "prior_close",
            "avg_dollar_volume_20d",
            "as_of",
        ]
    ].reset_index(drop=True)


def build_labels(
    universe: pd.DataFrame,
    daily_bars: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    trade_date: dt.date | dt.datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Build day-`trade_date` labels from `universe` and same-day bars.

    docs/LABEL_SPEC.md §Label + §Corporate-action exclusions:
      - return_t computed on UNADJUSTED close-over-close prices.
      - Any ticker with a split/reverse_split ex_date == trade_date is
        excluded BEFORE ranking (P4 tripwire).
      - Rank descending by return_t; label = 1 for rank <= 10.

    Defect 3: a universe name with NO day-`trade_date` bar (halted all day,
    or delisted intraday) can never be silently dropped by the join -- it
    is EXCLUDED from ranking (there is no `close_t` to compute `return_t`
    from), but the drop is explicit, counted, and logged so a spike is
    visible to `top10.sanity.check_universe_coverage(labels, universe)`.
    We exclude rather than carry a null return because (a) the spec's
    `return_t` definition is a real close-over-close ratio -- there is no
    principled `return_t` to impute for a name that did not trade, and
    (b) a null-valued `return_t` would either sort as NaN (silently
    dropping to the bottom, indistinguishable from "no bar") or crash a
    naive descending sort, without adding any real information; the
    ticker's continued UNIVERSE eligibility (§Universe "include names that
    are later delisted") is already fully preserved upstream in
    `build_universe`, which is the correct place for "is this name
    eligible" -- `build_labels` only decides "is this name RANKABLE today".
    """
    trade_date_ts = pd.Timestamp(trade_date)
    close_as_of_cutoff = _close_as_of(trade_date_ts)

    today_bars = daily_bars[daily_bars["trade_date"] == trade_date_ts]
    today_bars = today_bars[["ticker", "close", "as_of"]].rename(
        columns={"close": "close_t", "as_of": "close_as_of"}
    )

    merged_all = universe.merge(today_bars, on="ticker", how="left", indicator=True)
    no_bar_mask = merged_all["_merge"] == "left_only"
    if no_bar_mask.any():
        no_bar_tickers = sorted(merged_all.loc[no_bar_mask, "ticker"].tolist())
        logger.warning(
            "build_labels[%s]: %d/%d universe name(s) had no day-t bar "
            "(halted all day / delisted intraday) and are EXCLUDED from "
            "ranking (no return_t is computable without a close): %s",
            trade_date_ts.date(), len(no_bar_tickers), len(universe), no_bar_tickers,
        )
    merged = merged_all[~no_bar_mask].drop(columns="_merge").copy()

    # P4: exclude split / reverse-split days BEFORE ranking.
    if not corporate_actions.empty:
        split_today = corporate_actions[
            (corporate_actions["ex_date"] == trade_date_ts)
            & (corporate_actions["action_type"].isin(_SPLIT_ACTION_TYPES))
        ]
        excluded_tickers = set(split_today["ticker"])
    else:
        excluded_tickers = set()

    merged = merged[~merged["ticker"].isin(excluded_tickers)].copy()

    merged["return_t"] = merged["close_t"] / merged["prior_close"] - 1.0

    merged = merged.sort_values("return_t", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1
    merged["label"] = (merged["rank"] <= 10).astype(int)
    merged["label_spec_version"] = _label_spec_version()
    merged["trade_date"] = trade_date_ts
    # Defect 1/5: a label becomes knowable no earlier than day-t's own
    # 16:00 ET close. Prefer the bar's own `close_as_of` (it may be later,
    # e.g. a delayed/late-reported print), but never let a missing or
    # implausibly-early upstream `as_of` understate this -- and NEVER
    # hardcode midnight, which predates the very close the label is
    # derived from.
    merged["as_of"] = merged["close_as_of"].fillna(close_as_of_cutoff)
    merged["as_of"] = merged["as_of"].clip(lower=close_as_of_cutoff)

    return merged[
        [
            "trade_date",
            "ticker",
            "rank",
            "return_t",
            "label",
            "label_spec_version",
            "as_of",
        ]
    ].reset_index(drop=True)


def _output_path(trade_date_ts: pd.Timestamp, spec_hash: str) -> Any:
    directory = spec_dir(DATA_LABELS, spec_hash)
    return directory / f"{trade_date_ts.date().isoformat()}.parquet"


def _resolve_frames(source_or_frames: Any, start: dt.date, end: dt.date):
    """Accept either a MarketDataSource-like object (with `.daily_bars`,
    `.ticker_meta`, `.corporate_actions` methods) or a pre-loaded mapping /
    3-tuple of (daily_bars, ticker_meta, corporate_actions) DataFrames.
    """
    if hasattr(source_or_frames, "daily_bars") and callable(
        source_or_frames.daily_bars
    ):
        daily_bars = source_or_frames.daily_bars(start, end)
        corporate_actions = source_or_frames.corporate_actions(start, end)
        ticker_meta = source_or_frames.ticker_meta(start, end)
        return daily_bars, ticker_meta, corporate_actions

    if isinstance(source_or_frames, dict):
        return (
            source_or_frames["daily_bars"],
            source_or_frames["ticker_meta"],
            source_or_frames["corporate_actions"],
        )

    daily_bars, ticker_meta, corporate_actions = source_or_frames
    return daily_bars, ticker_meta, corporate_actions


def build_label_range(
    source_or_frames: Any,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Build and persist labels for every trading day in [start, end].

    Resumable: a day already written under `data/labels/<spec-hash>/` is
    skipped (not recomputed). Logs progress as it goes.
    """
    daily_bars, ticker_meta, corporate_actions = _resolve_frames(
        source_or_frames, start, end
    )

    spec_hash = _label_spec_version()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    trade_dates = sorted(
        d
        for d in daily_bars["trade_date"].unique()
        if start_ts <= pd.Timestamp(d) <= end_ts
    )

    all_labels: list[pd.DataFrame] = []
    total = len(trade_dates)
    for i, trade_date in enumerate(trade_dates, start=1):
        trade_date_ts = pd.Timestamp(trade_date)
        out_path = _output_path(trade_date_ts, spec_hash)

        if out_path.exists():
            logger.info(
                "[%d/%d] %s already written, skipping", i, total, trade_date_ts.date()
            )
            all_labels.append(read_parquet(out_path))
            continue

        logger.info("[%d/%d] building labels for %s", i, total, trade_date_ts.date())
        universe = build_universe(daily_bars, ticker_meta, corporate_actions, trade_date_ts)
        labels = build_labels(universe, daily_bars, corporate_actions, trade_date_ts)

        write_parquet(labels, out_path)
        all_labels.append(labels)

    if not all_labels:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "ticker",
                "rank",
                "return_t",
                "label",
                "label_spec_version",
                "as_of",
            ]
        )

    return pd.concat(all_labels, ignore_index=True)
