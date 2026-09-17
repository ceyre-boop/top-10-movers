"""The committed walk-forward runner EXP-003/EXP-004 lacked.

`run_variant("t1b")` is the STEP-1 repro gate: it must reproduce EXP-003's
walk-forward result (mean ~0.660 hits/day, per-year vs B2 approximately
+0.100 / +0.076 / +0.045) using ONLY features rebuilt from
`data/raw/databento/preholdout/universe_liquidity.parquet` through the
real, guarded pipeline (`top10.features.bars_t1.build_bars_t1` ->
`top10.features.spec.write_features` -> `top10.walkforward`), not a
scratch script.

**No `unseal_token` parameter anywhere in this module.** The holdout
(>= 2023-01-01) stays sealed structurally: `_load_universe` asserts the
loaded frame's own `trade_date.max()` is strictly before the holdout
start, and every downstream call (`expanding_window_splits`,
`run_walkforward`, the `top10.baselines` functions) is left at its
default `unseal_token=None` -- there is no code path in this module that
could ever supply the token, by construction, not by convention.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import pandas as pd

from top10.baselines import (
    _rolling_realized_vol_scores,
    _top_n_per_day,
    b0_random,
    b1_yesterday_repeat,
)
from top10.config import DATA_RAW
from top10.experiment import assert_frame_holdout_sealed
from top10.features.bars_t1 import BARS_T1_FEATURES, build_bars_t1
from top10.features.spec import T1B_SPEC, T1B_TFM_SPEC, write_features
from top10.features.t1 import decision_time_t1
from top10.leakage import assert_decision_time_safe
from top10.metrics import compare_to_baseline, per_year_report
from top10.model import Top10Ranker
from top10.walkforward import expanding_window_splits, run_walkforward

_VALID_VARIANTS = ("t1b", "t1b_tfm")

_DEFAULT_UNIVERSE_PATH = (
    DATA_RAW / "databento" / "preholdout" / "universe_liquidity.parquet"
)

# Plan §6: the holdout is 2023-01-01 onward. This module has no unseal
# path at all -- a frame that reaches this far is refused unconditionally.
_HOLDOUT_START = pd.Timestamp("2023-01-01")

_TEST_YEARS = (2020, 2021, 2022)

# Frozen a priori, per the plan, so only the feature set varies between
# `t1b` and `t1b_tfm`.
_LGBM_PARAMS: dict[str, Any] = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "num_boost_round": 300,
    "verbosity": -1,
}


def _load_universe(universe_path: str | Path) -> pd.DataFrame:
    universe = pd.read_parquet(universe_path)

    max_date = pd.Timestamp(universe["trade_date"].max())
    if max_date >= _HOLDOUT_START:
        raise ValueError(
            f"run_variant: universe frame at {universe_path} contains dates "
            f"through {max_date.date()}, on/after the sealed holdout start "
            f"({_HOLDOUT_START.date()}). This module has no unseal path -- "
            "the holdout stays sealed structurally, not by convention."
        )
    return universe


def _labels_frame(universe: pd.DataFrame) -> pd.DataFrame:
    """trade_date, ticker, rank, label, return_t, label_spec_version, as_of
    -- `as_of` is the CLOSE decision time (16:00 ET on `trade_date` itself),
    since a label for `trade_date` is only knowable after that day's own
    close, matching `top10.baselines`'s `_close_decision_time` convention.
    """
    labels = universe[["trade_date", "ticker", "rank", "label"]].copy()
    labels["return_t"] = universe["ret"]
    labels["label_spec_version"] = "bars_t1_repro_v1"
    labels["as_of"] = pd.to_datetime(universe["trade_date"]) + pd.Timedelta(hours=16)
    return labels


def _bars_close_frame(universe: pd.DataFrame) -> pd.DataFrame:
    """trade_date, ticker, close, as_of -- for `top10.baselines.b2_realized_vol`.
    `as_of` is the close decision time, same convention as `_labels_frame`.
    """
    bars = universe[["trade_date", "ticker", "close"]].copy()
    bars["as_of"] = pd.to_datetime(universe["trade_date"]) + pd.Timedelta(hours=16)
    return bars


def _load_tfm_quantiles(year: int) -> pd.DataFrame:
    """Load the TimesFM quantile cache for `year`, if it exists.

    Per the plan, the cache lives under `data/raw/timesfm/<model_tag>/<year>.parquet`,
    written by a SEPARATE interpreter/module (`top10/forecast/timesfm_cache.py`,
    out of scope here) so this project's dependency graph never imports
    `timesfm` itself. Raises `FileNotFoundError` with a clear, actionable
    message when the cache is absent -- callers must not silently degrade
    to an empty/zero TFM feature set.
    """
    tfm_dir = Path(DATA_RAW) / "timesfm"
    matches = sorted(glob.glob(str(tfm_dir / "*" / f"{year}.parquet")))
    if not matches:
        raise FileNotFoundError(
            f"run_variant('t1b_tfm'): no TimesFM quantile cache found for year "
            f"{year} under {tfm_dir}/<model_tag>/{year}.parquet. The cache is "
            "built by a separate interpreter (see top10/forecast/timesfm_cache.py "
            "-- deliberately out of this module's scope, so this project's "
            "dependency graph never imports `timesfm` itself). Build the cache "
            "first; this variant cannot run without it."
        )
    return pd.read_parquet(matches[0])


def _build_features(variant: str, universe: pd.DataFrame) -> pd.DataFrame:
    features = build_bars_t1(universe)

    if variant == "t1b":
        return features

    # t1b_tfm: merge the TimesFM quantile-derived features in.
    # `top10.features.tfm` (the 7-feature builder from a quantile frame) is
    # out of this module's owned scope -- if/when that cache exists, the
    # merge point is here. Absent the cache, this fails loudly per year
    # rather than silently dropping the TFM half of the feature set.
    years = sorted(pd.to_datetime(features["trade_date"]).dt.year.unique())
    quantile_frames = [_load_tfm_quantiles(int(year)) for year in years]
    quantiles = pd.concat(quantile_frames, ignore_index=True)

    merged = features.merge(quantiles, on=["trade_date", "ticker"], how="left")
    return merged


def _write_features_per_day(features: pd.DataFrame, spec) -> None:
    """Write `features` through `top10.features.spec.write_features`, one
    call per `trade_date`, so the leakage gate fires on every write --
    exactly the persistence path EXP-003/EXP-004 bypassed."""
    for trade_date, day_df in features.groupby("trade_date"):
        write_features(day_df.reset_index(drop=True), spec, trade_date)


def run_variant(variant: str, *, universe_path: str | Path | None = None) -> dict:
    """Run the full walk-forward evaluation for `variant` in {"t1b", "t1b_tfm"}.

    Returns a structured result:
    {
        "variant": str,
        "feature_spec_hash": str,
        "n_rows": int,
        "n_days": int,
        "per_year": pd.DataFrame,
        "mean_hits_per_day": float,
        "vs_baseline": {"B0": {...}, "B1": {...}, "B2": {...}},
        "predictions": pd.DataFrame,
    }
    """
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"run_variant: variant must be one of {_VALID_VARIANTS}, got {variant!r}")

    universe = _load_universe(universe_path or _DEFAULT_UNIVERSE_PATH)

    spec = T1B_SPEC if variant == "t1b" else T1B_TFM_SPEC

    features = _build_features(variant, universe)
    _write_features_per_day(features, spec)

    feature_columns = list(spec.columns[2:-1])  # drop trade_date, ticker, as_of
    # "After feature warmup" (EXP-003's own term): rows lacking enough
    # trailing history for one or more features (e.g. `dist52`'s 60-day
    # minimum) are NaN and must be dropped before fit/eval -- LightGBM can
    # technically tolerate NaN, but this is a faithfulness requirement, not
    # a modeling one: reproducing EXP-003 means reproducing what it fed the
    # model, not merely something numerically similar.
    fit_features = features.dropna(subset=feature_columns).reset_index(drop=True)

    labels = _labels_frame(universe)

    dates = fit_features["trade_date"].unique()
    splits = expanding_window_splits(dates, retrain="yearly", min_train_years=2)

    def model_factory() -> Top10Ranker:
        return Top10Ranker(objective="binary", params=dict(_LGBM_PARAMS), feature_spec=spec)

    result = run_walkforward(model_factory, fit_features, labels, splits, k=10)

    # The candidate pool every baseline picks from must be the SAME
    # post-warmup (trade_date, ticker) universe the model itself competed
    # over -- not the full, unrestricted universe. A ticker with fewer
    # than `dist52`'s 60-day minimum history is invisible to the model
    # (dropped by the `fit_features` warmup filter above) but would still
    # be visible to an unrestricted B2, which needs only 5 days of
    # history -- letting B2 pick from young, still-thinly-traded names the
    # model never had a chance at. Confirmed empirically: unrestricted B2
    # measures ~0.72-0.78 hits/day flat across every year (implausible,
    # and nothing like EXP-004's reported 0.534/0.514/0.713); restricting
    # to `allowed_pairs` below brings B1/B2 back in line with EXP-003's/
    # EXP-004's reported baseline numbers.
    allowed_pairs = set(zip(fit_features["trade_date"], fit_features["ticker"]))

    bars_close = _bars_close_frame(universe)
    universe_for_b0 = fit_features[["trade_date", "ticker", "as_of"]].copy()

    b1_full = b1_yesterday_repeat(labels, k=10)
    b1_restricted = b1_full[
        [(td, tk) in allowed_pairs for td, tk in zip(b1_full["trade_date"], b1_full["ticker"])]
    ].reset_index(drop=True)

    # B2 needs the SAME two-step treatment as B1 (score first over full
    # history so trailing windows stay correct, restrict candidates
    # second) -- `top10.baselines.b2_realized_vol` doesn't expose a
    # candidate-restriction hook, so its own scoring/leakage-check/top-N
    # internals are reused directly here rather than duplicated.
    assert_frame_holdout_sealed(bars_close)
    b2_scored = _rolling_realized_vol_scores(bars_close, window=5)
    b2_scored = b2_scored[
        [(td, tk) in allowed_pairs for td, tk in zip(b2_scored["trade_date"], b2_scored["ticker"])]
    ]
    for trade_date, group in b2_scored.groupby("trade_date"):
        assert_decision_time_safe(group, decision_time_t1(trade_date))
    b2_restricted = _top_n_per_day(b2_scored, 10)

    baseline_preds = {
        "B0": b0_random(universe_for_b0, seed=0, k=10),
        "B1": b1_restricted,
        "B2": b2_restricted,
    }

    vs_baseline = {
        name: compare_to_baseline(result.predictions, preds, labels, k=10)
        for name, preds in baseline_preds.items()
    }

    per_year = per_year_report(result.predictions, labels, k=10)
    per_year = per_year[per_year["year"].isin(_TEST_YEARS)].reset_index(drop=True)

    mean_hits_per_day = float(per_year["mean_hits"].mean()) if not per_year.empty else float("nan")

    return {
        "variant": variant,
        "feature_spec_hash": spec.spec_hash,
        "n_rows": int(len(fit_features)),
        "n_days": int(fit_features["trade_date"].nunique()),
        "per_year": per_year,
        "mean_hits_per_day": mean_hits_per_day,
        "vs_baseline": vs_baseline,
        "predictions": result.predictions,
    }
