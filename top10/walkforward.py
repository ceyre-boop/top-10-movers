"""Walk-forward evaluation — docs/PREREG_TOP10.md §5.2.

Plan §5.2 is explicit: **walk-forward only, expanding window.** A random
train/test split is a P6 violation, so this module intentionally does not
expose one, even as an option -- there is no `test_size` / `random_state`
knob anywhere in this file.

Every split is guarded against the sealed holdout (Plan §6 / P12): building
a split whose window reaches into `2023-01-01+` raises unless the caller
holds the same `unseal_token="PREREG_FROZEN"` gate used everywhere else
(`top10.experiment.assert_holdout_sealed`).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import pandas as pd

from top10.experiment import assert_frame_holdout_sealed, assert_holdout_sealed
from top10.metrics import compare_to_baseline, per_year_report

_VALID_RETRAIN = ("yearly", "quarterly")


@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        # P6: an expanding-window split must never let train touch or cross
        # into its own test window. No gap-free overlap is possible.
        if not (self.train_end < self.test_start):
            raise ValueError(
                f"Split: train_end ({self.train_end}) must be strictly before "
                f"test_start ({self.test_start}) -- overlapping train/test "
                "windows are a P6 leakage violation."
            )


def _fit_model(model: Any, features: pd.DataFrame, labels: pd.DataFrame, unseal_token: str | None) -> None:
    """Call `model.fit(features, labels)`, forwarding `unseal_token` iff
    `model.fit` declares that parameter (e.g. `top10.model.Top10Ranker.fit`'s
    own holdout guard). `model_factory` is duck-typed and injected test
    doubles (e.g. a trivial identity-score model with no lightgbm
    dependency) are not required to know about the seal at all -- this
    split's own boundaries were already checked by `run_walkforward`
    before this is ever called, so a plain `.fit(features, labels)` there
    is not a leakage gap, just a model that doesn't participate in the
    token forwarding."""
    if "unseal_token" in inspect.signature(model.fit).parameters:
        model.fit(features, labels, unseal_token=unseal_token)
    else:
        model.fit(features, labels)


def _period_key(ts: pd.Timestamp, retrain: str) -> tuple[int, int]:
    if retrain == "yearly":
        return (ts.year, 0)
    # quarterly
    return (ts.year, (ts.month - 1) // 3)


def _period_bounds(dates: Sequence[pd.Timestamp], retrain: str) -> list[tuple[tuple[int, int], list[pd.Timestamp]]]:
    """Group sorted unique `dates` into (period_key, dates_in_period) pairs,
    in chronological order."""
    grouped: dict[tuple[int, int], list[pd.Timestamp]] = {}
    order: list[tuple[int, int]] = []
    for d in dates:
        key = _period_key(d, retrain)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(d)
    return [(key, grouped[key]) for key in order]


def expanding_window_splits(
    dates: Sequence[Any],
    *,
    retrain: str = "yearly",
    min_train_years: int = 2,
    unseal_token: str | None = None,
) -> list[Split]:
    """Build expanding-window, walk-forward splits over `dates`.

    `retrain`: "yearly" or "quarterly" (Plan §5.2 says test both).
    `min_train_years`: how many calendar years of initial history are
    reserved as the first training window before the first test period
    begins.

    Every constructed split is checked via
    `top10.experiment.assert_holdout_sealed` -- a split whose train or test
    window reaches into the sealed 2023-01-01+ holdout raises unless
    `unseal_token="PREREG_FROZEN"` is supplied.

    Only the four boundary timestamps (`train_start`, `train_end`,
    `test_start`, `test_end`) are passed to the seal check, not every date
    in between -- this IS the general check, not an approximation, given
    this function's own invariant: `train_end`/`test_end` are always the
    actual maximum real date of their date set (`train_dates[-1]` /
    `period_dates[-1]`, both drawn from `unique_dates`, never a synthetic
    range endpoint), and the holdout is a one-sided, open-ended range
    (`date >= HOLDOUT_START`). For a one-sided threshold, "the maximum of
    a set is >= the threshold" is logically equivalent to "some element of
    the set is >= the threshold" -- so checking the max is checking every
    date. This equivalence would NOT hold if a caller fed non-monotonic
    boundaries that don't reflect the true max/min of the underlying date
    set; it holds here only because `train_end`/`test_end` are always
    real, drawn-from-`unique_dates` values.
    """
    if retrain not in _VALID_RETRAIN:
        raise ValueError(f"expanding_window_splits: retrain must be one of {_VALID_RETRAIN}, got {retrain!r}")

    unique_dates = sorted({pd.Timestamp(d) for d in dates})
    if not unique_dates:
        return []

    periods = _period_bounds(unique_dates, retrain)

    first_year = unique_dates[0].year
    train_cutoff_year = first_year + min_train_years  # first year eligible to be tested

    splits: list[Split] = []
    for period_key, period_dates in periods:
        period_year = period_key[0]
        if period_year < train_cutoff_year:
            continue

        test_start = period_dates[0]
        test_end = period_dates[-1]

        train_dates = [d for d in unique_dates if d < test_start]
        if not train_dates:
            continue
        train_start = train_dates[0]
        train_end = train_dates[-1]

        split = Split(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)

        assert_holdout_sealed(
            pd.Series([split.train_start, split.train_end, split.test_start, split.test_end]),
            unseal_token=unseal_token,
        )

        splits.append(split)

    return splits


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    labels: pd.DataFrame
    k: int = 10
    unseal_token: str | None = None

    def __post_init__(self) -> None:
        # `metrics.per_year_report` / `compare_to_baseline` (called below)
        # are themselves unguarded -- they live outside this module -- so
        # the seal is enforced here, at construction, at the
        # walkforward/baselines boundary this module DOES own. A
        # `WalkForwardResult` built directly (bypassing `run_walkforward`,
        # which seals every split) with holdout-dated predictions/labels
        # is exactly the kind of free peek at the scoring step Plan §6
        # forbids.
        assert_frame_holdout_sealed(self.predictions, unseal_token=self.unseal_token)
        assert_frame_holdout_sealed(self.labels, unseal_token=self.unseal_token)

    @property
    def per_year(self) -> pd.DataFrame:
        return per_year_report(self.predictions, self.labels, self.k)

    def vs_baseline(self, baseline_preds: pd.DataFrame) -> dict:
        assert_frame_holdout_sealed(baseline_preds, unseal_token=self.unseal_token)
        return compare_to_baseline(self.predictions, baseline_preds, self.labels, self.k)

    def decay_plot_data(self) -> dict[str, list]:
        """Per-year series for the §5.2 decay plot (data only -- rendering
        is left to the caller)."""
        report = self.per_year
        return {
            "year": report["year"].tolist(),
            "precision_at_k": report["precision_at_k"].tolist(),
            "map_at_k": report["map_at_k"].tolist(),
            "mean_hits": report["mean_hits"].tolist(),
            "n_days": report["n_days"].tolist(),
        }


def run_walkforward(
    model_factory: Callable[[], Any],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: Sequence[Split],
    k: int = 10,
    *,
    unseal_token: str | None = None,
) -> WalkForwardResult:
    """Refit a fresh model (via `model_factory()`) per split, predict
    out-of-sample on that split's test window only, and concatenate all
    out-of-sample predictions.

    `model_factory` is duck-typed: it must return an object with
    `.fit(features, labels)` and `.predict(features) -> DataFrame[trade_date,
    ticker, score]`. This lets tests inject a trivial deterministic model
    with no lightgbm dependency.

    `splits` is NOT trusted to have already been sealed -- `expanding_window_splits`
    checks the splits it builds, but a caller can hand-build a `Sequence[Split]`
    directly (skipping that function entirely) with a `test_start` reaching
    into 2023-01-01+. So every split is re-checked here too, independent of
    provenance, unless `unseal_token="PREREG_FROZEN"` is supplied.
    """
    for split in splits:
        assert_holdout_sealed(
            pd.Series([split.train_start, split.train_end, split.test_start, split.test_end]),
            unseal_token=unseal_token,
        )

    all_predictions: list[pd.DataFrame] = []

    for split in splits:
        train_features = features[
            (features["trade_date"] >= split.train_start) & (features["trade_date"] <= split.train_end)
        ]
        train_labels = labels[
            (labels["trade_date"] >= split.train_start) & (labels["trade_date"] <= split.train_end)
        ]
        test_features = features[
            (features["trade_date"] >= split.test_start) & (features["trade_date"] <= split.test_end)
        ]

        if train_features.empty or test_features.empty:
            continue

        model = model_factory()
        _fit_model(model, train_features, train_labels, unseal_token)
        predictions = model.predict(test_features)
        all_predictions.append(predictions)

    if not all_predictions:
        predictions = pd.DataFrame(columns=["trade_date", "ticker", "score"])
    else:
        predictions = pd.concat(all_predictions, ignore_index=True)

    return WalkForwardResult(predictions=predictions, labels=labels, k=k, unseal_token=unseal_token)


@dataclass
class GateResult:
    passed: bool
    years_won: int
    years_total: int
    verdict: str


def beats_baseline_gate(
    result: WalkForwardResult,
    baseline_preds: pd.DataFrame,
    *,
    min_years_won: int = 5,
    min_years: int = 7,
) -> GateResult:
    """Plan §5.2 / checklist 4.2 decision gate: the model must beat B4 in
    at least `min_years_won` of `min_years` evaluated years.

    This is a §10 KILL CRITERION -- a stop condition, not a nice-to-have --
    so the failure path is made just as prominent as the success path via
    `GateResult.verdict`.
    """
    comparison = result.vs_baseline(baseline_preds)
    years_won = int(comparison["years_won"])
    years_total = int(comparison["years_total"])

    enough_years = years_total >= min_years
    passed = bool(enough_years and years_won >= min_years_won)

    if not enough_years:
        verdict = (
            f"GATE UNDECIDABLE: only {years_total} year(s) evaluated, need at least "
            f"{min_years} to apply the §5.2 gate."
        )
    elif passed:
        verdict = (
            f"GATE PASSED: model beat baseline B4 in {years_won}/{years_total} years "
            f"(threshold: >= {min_years_won})."
        )
    else:
        verdict = (
            f"GATE FAILED -- §10 KILL CRITERION TRIGGERED: model beat baseline B4 in only "
            f"{years_won}/{years_total} years (threshold: >= {min_years_won}). Per plan §10, "
            "this result does not support a primary success claim."
        )

    return GateResult(passed=passed, years_won=years_won, years_total=years_total, verdict=verdict)
