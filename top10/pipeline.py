"""The ingest -> labels -> features -> baselines -> walkforward spine.

Each `*_step` function is idempotent: re-running it against a range that
was already built is safe and cheap -- it detects what's already on disk
and skips it, logging what it skipped rather than silently doing nothing.

Per docs/LABEL_SPEC.md §"Point-in-time invariants" this module never reads
or writes 2023-01-01+ data except through the single sealed chokepoint
`top10.experiment.assert_holdout_sealed` -- see `run_all`.

Vendor adapters (`top10.data.polygon`, `top10.data.databento`), the
Robinhood collector (`top10.collect`), and the T1/T2 feature builders
(`top10.features.t1`, `top10.features.t2`) are all imported LAZILY inside
functions here, never at module scope -- this module must import cleanly
even while those modules are mid-flight/incomplete.
"""

from __future__ import annotations

import datetime as dt
import inspect
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from top10.baselines import run_all_baselines
from top10.config import DATA_PIT
from top10.experiment import assert_holdout_sealed
from top10.features.spec import PRIOR_CLOSE_COLUMNS, T1_SPEC, T2_SPEC, FeatureSpec, feature_output_path, write_features
from top10.labels import build_label_range
from top10.leakage import shuffle_label_test
from top10.sanity import run_all as sanity_run_all
from top10.storage import LeakageError, read_parquet, write_parquet
from top10.walkforward import expanding_window_splits, run_walkforward

logger = logging.getLogger(__name__)

_PIT_DATASETS = ("daily_bars", "corporate_actions", "ticker_meta", "earnings")


class PipelineAbort(Exception):
    """Raised when a pipeline step fails a hard gate (e.g. sanity checks).

    A sanity failure aborts the pipeline -- it never merely warns, since a
    bad label set silently poisons every downstream step (labels, features,
    baselines, walk-forward all read from the same corrupted file).
    """


# --- ingest ------------------------------------------------------------


def _pit_path(dataset: str, vendor: str, start: dt.date, end: dt.date) -> Path:
    return Path(DATA_PIT) / vendor / dataset / f"{start.isoformat()}_{end.isoformat()}.parquet"


def ingest(vendor: str, start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
    """Pull daily bars, corporate actions, ticker meta, and earnings for
    `[start, end]` from `vendor` via `top10.data.get_source`, and persist
    each as a point-in-time parquet table under `data/pit/<vendor>/<dataset>/`.

    Resumable: if the exact `(vendor, dataset, start, end)` range is
    already on disk, the vendor fetch is skipped entirely and the cached
    frame is read back instead.

    Raw vendor payloads are cached verbatim by the adapter itself (via
    `top10.data.cache.cached_call`, per docs/LABEL_SPEC.md "raw dumps are
    untouched") as a side effect of the fetch calls below -- this function
    is only responsible for the point-in-time layer.
    """
    from top10.data import get_source  # lazy: top10/data/__init__.py may be mid-flight

    source = get_source(vendor)

    fetchers = {
        "daily_bars": source.daily_bars,
        "corporate_actions": source.corporate_actions,
        "ticker_meta": source.ticker_meta,
        "earnings": source.earnings,
    }

    frames: dict[str, pd.DataFrame] = {}
    for dataset, fetch_fn in fetchers.items():
        path = _pit_path(dataset, vendor, start, end)
        if path.exists():
            logger.info(
                "ingest: %s %s..%s already on disk at %s, skipping fetch",
                dataset, start, end, path,
            )
            frames[dataset] = read_parquet(path)
            continue

        logger.info("ingest: fetching %s %s..%s from vendor=%s", dataset, start, end, vendor)
        df = fetch_fn(start, end)
        write_parquet(df, path)
        frames[dataset] = df

    # TOP FINDING (adversarial audit): `assert_no_adjusted_prices` and
    # `verify_unadjusted` had ZERO production call sites. Back-adjusted
    # prices are P3 -- the leak that yields 7/10 in backtest and 1/10
    # live -- and they are undetectable downstream, because every return
    # computed from them looks perfectly plausible. Ingest is the only
    # place the raw vendor feed is still visible, so this is the only
    # place the check can run. Do not remove it.
    bars = frames.get("daily_bars")
    actions = frames.get("corporate_actions")
    if bars is not None and not bars.empty:
        from top10.leakage import assert_no_adjusted_prices, verify_unadjusted

        # Both raise on detection; re-raise as PipelineAbort so the caller
        # sees a single, unambiguous "stop, your prices are wrong" failure
        # rather than a LeakageError from three layers down.
        try:
            assert_no_adjusted_prices(bars, actions)
            if actions is not None and not actions.empty:
                verify_unadjusted(bars, actions)
        except LeakageError as exc:
            raise PipelineAbort(
                f"ingest: vendor feed appears BACK-ADJUSTED -- {exc}. "
                "Unadjusted prices are mandatory (docs/LABEL_SPEC.md). "
                "Refusing to build a point-in-time layer on adjusted prices."
            ) from exc

    return frames


# --- labels --------------------------------------------------------------


def build_labels_step(
    source_or_frames: Any,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Build (or resume) labels for `[start, end]`.

    Delegates to `top10.labels.build_label_range`, which is itself
    per-trading-day idempotent and logs each day it skips.
    """
    logger.info("build_labels_step: building labels for %s..%s", start, end)
    return build_label_range(source_or_frames, start, end)


# --- features --------------------------------------------------------------


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def build_features_step(
    task: str,
    start: dt.date,
    end: dt.date,
    *,
    daily_bars: pd.DataFrame | None = None,
    ticker_meta: pd.DataFrame | None = None,
    earnings: pd.DataFrame | None = None,
    labels_history: pd.DataFrame | None = None,
    market_context: pd.DataFrame | None = None,
    premarket_bars: pd.DataFrame | None = None,
    prior_close: pd.DataFrame | None = None,
    halts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build (or resume) T1 or T2 features for every trading day in
    `[start, end]`.

    Idempotent per trading day: a day whose feature file already exists
    under `data/features/<feature-spec-hash>/` is skipped (not
    recomputed), and the skip is logged.

    Inputs default to empty frames when not supplied -- callers running
    this against real data are expected to pass frames loaded from the
    `data/pit/` tables written by `ingest`; tests inject synthetic frames
    directly so this step never has to touch the filesystem or network.
    """
    if task not in ("T1", "T2"):
        raise ValueError(f"build_features_step: task must be 'T1' or 'T2', got {task!r}")

    from top10.features.t1 import build_t1_features, decision_time_t1  # lazy: may be mid-flight
    from top10.features.t2 import build_t2_features, decision_time_t2  # lazy: may be mid-flight

    daily_bars = daily_bars if daily_bars is not None else _empty_frame(["trade_date", "ticker", "as_of"])
    ticker_meta = ticker_meta if ticker_meta is not None else _empty_frame(["ticker", "as_of"])
    earnings = earnings if earnings is not None else _empty_frame(["ticker", "as_of"])
    labels_history = labels_history if labels_history is not None else _empty_frame(["trade_date", "ticker", "label", "as_of"])
    market_context = market_context if market_context is not None else pd.DataFrame()
    premarket_bars = premarket_bars if premarket_bars is not None else _empty_frame(["trade_date", "ticker", "minute", "as_of"])
    # Defect 2 (CONFIRMED): must agree with `top10.features.t2`'s
    # `_prior_close_lookup` column contract (see `PRIOR_CLOSE_COLUMNS`
    # docstring in `features/spec.py`) -- this used to be `prior_close`
    # here vs `close` in t2, which crashed the only T2 orchestration path.
    prior_close = prior_close if prior_close is not None else _empty_frame(list(PRIOR_CLOSE_COLUMNS))
    halts = halts if halts is not None else pd.DataFrame()

    spec: FeatureSpec = T1_SPEC if task == "T1" else T2_SPEC

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    trade_dates = sorted(
        d for d in daily_bars["trade_date"].unique() if start_ts <= pd.Timestamp(d) <= end_ts
    )

    all_features: list[pd.DataFrame] = []
    total = len(trade_dates)
    for i, trade_date in enumerate(trade_dates, start=1):
        trade_date_ts = pd.Timestamp(trade_date)
        out_path = feature_output_path(spec, trade_date_ts)

        # TOP FINDING (adversarial audit): `read_parquet(as_of=...)` is
        # documented as the PIT chokepoint but was never once called with
        # `as_of=` in production -- every real call site passed nothing.
        # This is that fix: a cached feature file is re-filtered to this
        # task's own decision time for `trade_date_ts` on every read.
        decision_time = decision_time_t1(trade_date_ts) if task == "T1" else decision_time_t2(trade_date_ts)

        if out_path.exists():
            logger.info(
                "build_features_step[%s]: [%d/%d] %s already written, skipping",
                task, i, total, trade_date_ts.date(),
            )
            all_features.append(read_parquet(out_path, as_of=decision_time))
            continue

        logger.info(
            "build_features_step[%s]: [%d/%d] building features for %s",
            task, i, total, trade_date_ts.date(),
        )

        if task == "T1":
            features = build_t1_features(
                daily_bars, ticker_meta, earnings, labels_history, market_context, trade_date_ts
            )
        else:
            t1_features = build_t1_features(
                daily_bars, ticker_meta, earnings, labels_history, market_context, trade_date_ts
            )
            features = build_t2_features(
                t1_features, premarket_bars, prior_close, halts, trade_date_ts,
                labels_history=labels_history,
            )

        write_features(features, spec, trade_date_ts)
        all_features.append(features)

    if not all_features:
        return _empty_frame(list(spec.columns))
    return pd.concat(all_features, ignore_index=True)


# --- sanity ------------------------------------------------------------


def run_sanity_step(labels: pd.DataFrame, corporate_actions: pd.DataFrame, universe: pd.DataFrame | None = None):
    """Run `top10.sanity.run_all` and raise `PipelineAbort` on failure.

    §2.5: a sanity failure must ABORT the pipeline, not warn.
    """
    report = sanity_run_all(labels, corporate_actions, universe=universe)
    logger.info("run_sanity_step:\n%s", report)
    if not report.passed:
        raise PipelineAbort(
            f"run_sanity_step: sanity checks FAILED ({len(report.failures)} failure(s)); "
            f"aborting pipeline before any downstream step reads these labels.\n{report}"
        )
    return report


# --- baselines ------------------------------------------------------------


def run_baselines_step(
    *,
    universe: pd.DataFrame,
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    earnings: pd.DataFrame,
    premarket_bars: pd.DataFrame,
    prior_close: pd.DataFrame,
    seed: int = 0,
    min_premarket_dollar_vol: float = 500_000,
    k: int = 10,
) -> dict[str, pd.DataFrame]:
    """Compute every baseline (B0-B4). Baselines are pure functions of
    their inputs -- nothing is persisted here, so there is nothing to
    resume; each call simply (re)computes deterministically."""
    logger.info("run_baselines_step: computing B0-B4")
    return run_all_baselines(
        universe=universe,
        labels=labels,
        bars=bars,
        earnings=earnings,
        premarket_bars=premarket_bars,
        prior_close=prior_close,
        seed=seed,
        min_premarket_dollar_vol=min_premarket_dollar_vol,
        k=k,
    )


# --- walkforward ------------------------------------------------------------


def run_walkforward_step(
    model_factory,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    retrain: str = "yearly",
    min_train_years: int = 2,
    k: int = 10,
    unseal_token: str | None = None,
):
    """Build expanding-window splits over `features`' trade dates and run
    walk-forward evaluation. Every split is checked against the sealed
    holdout via `top10.experiment.assert_holdout_sealed` (through
    `expanding_window_splits`) -- this step never needs its own separate
    guard, but forwards `unseal_token` explicitly so the caller's intent
    is visible at the call site.
    """
    dates = sorted(pd.Timestamp(d) for d in features["trade_date"].unique())
    splits = expanding_window_splits(
        dates, retrain=retrain, min_train_years=min_train_years, unseal_token=unseal_token
    )
    logger.info("run_walkforward_step: %d split(s) (%s retrain)", len(splits), retrain)
    return run_walkforward(model_factory, features, labels, splits, k=k)


# --- run_all ------------------------------------------------------------


def run_all(
    vendor: str,
    start: dt.date,
    end: dt.date,
    *,
    model_factory=None,
    unseal_token: str | None = None,
) -> dict[str, Any]:
    """Run the full ingest -> labels -> sanity -> features -> baselines ->
    walk-forward spine for `[start, end]`.

    Never touches 2023-01-01+ except through
    `top10.experiment.assert_holdout_sealed` -- this is checked up front,
    against the full requested range, before a single byte is fetched.
    """
    assert_holdout_sealed([start, end], unseal_token=unseal_token)

    frames = ingest(vendor, start, end)

    labels = build_labels_step(frames, start, end)

    run_sanity_step(labels, frames.get("corporate_actions", pd.DataFrame()))

    t1_features = build_features_step(
        "T1",
        start,
        end,
        daily_bars=frames.get("daily_bars"),
        ticker_meta=frames.get("ticker_meta"),
        earnings=frames.get("earnings"),
        labels_history=labels,
    )

    results: dict[str, Any] = {
        "frames": frames,
        "labels": labels,
        "t1_features": t1_features,
    }

    if model_factory is not None and not t1_features.empty:
        # TOP FINDING (adversarial audit): `shuffle_label_test` had zero
        # production call sites, even though docs/PREREG_TOP10.md line 37
        # cites it as an anti-leakage requirement. Run it as a gate before
        # any walk-forward is permitted to proceed, and ABORT (never merely
        # warn) on failure -- a leaking feature set must never reach
        # walk-forward evaluation.
        shuffle_result = shuffle_label_test(
            _shuffle_fit_predict(model_factory, unseal_token),
            t1_features,
            labels,
        )
        logger.info("run_all: shuffle_label_test result: %s", shuffle_result)
        if not shuffle_result["passed"]:
            raise PipelineAbort(
                "run_all: shuffle_label_test FAILED "
                f"(observed_precision={shuffle_result['observed_precision']!r} > "
                f"tolerance={shuffle_result['tolerance']!r}) -- one or more features "
                f"leak the true label; aborting before walk-forward. {shuffle_result}"
            )
        results["shuffle_label_test"] = shuffle_result

        results["walkforward"] = run_walkforward_step(
            model_factory, t1_features, labels, unseal_token=unseal_token
        )

    return results


def _shuffle_fit_predict(model_factory, unseal_token: str | None):
    """Adapt `model_factory` (duck-typed `.fit(features, labels)` /
    `.predict(features) -> DataFrame`) into the
    `fit_predict_fn(features, shuffled_labels) -> predictions` shape
    `top10.leakage.shuffle_label_test` expects, forwarding `unseal_token`
    iff the model's own `.fit` declares that parameter -- same convention
    as `top10.walkforward._fit_model`."""

    def _fit_predict(features: pd.DataFrame, shuffled_labels: pd.DataFrame) -> pd.DataFrame:
        model = model_factory()
        if "unseal_token" in inspect.signature(model.fit).parameters:
            model.fit(features, shuffled_labels, unseal_token=unseal_token)
        else:
            model.fit(features, shuffled_labels)
        return model.predict(features)

    return _fit_predict
