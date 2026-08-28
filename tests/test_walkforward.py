"""Tests for top10/walkforward.py.

Offline, synthetic only -- the injected `model_factory` in the end-to-end
test is a trivial deterministic scorer (score = a feature column), so
nothing here requires lightgbm.
"""

from __future__ import annotations

import pandas as pd
import pytest

from top10.storage import LeakageError
from top10.walkforward import (
    GateResult,
    Split,
    WalkForwardResult,
    beats_baseline_gate,
    expanding_window_splits,
    run_walkforward,
)


def _business_dates(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start, end))


# --- expanding_window_splits: no leakage -----------------------------------


def test_yearly_splits_never_leak():
    dates = _business_dates("2015-01-01", "2022-12-31")
    splits = expanding_window_splits(dates, retrain="yearly", min_train_years=2)
    assert len(splits) > 0
    for split in splits:
        assert split.train_end < split.test_start


def test_quarterly_splits_never_leak():
    dates = _business_dates("2015-01-01", "2022-12-31")
    splits = expanding_window_splits(dates, retrain="quarterly", min_train_years=2)
    assert len(splits) > 0
    for split in splits:
        assert split.train_end < split.test_start


def test_splits_are_expanding_not_shrinking():
    dates = _business_dates("2015-01-01", "2020-12-31")
    splits = expanding_window_splits(dates, retrain="yearly", min_train_years=2)
    # Every split's train_start is the same (expanding from the very start of
    # history), and train_end strictly increases across splits.
    assert all(s.train_start == splits[0].train_start for s in splits)
    train_ends = [s.train_end for s in splits]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)


def test_split_construction_rejects_overlap_directly():
    with pytest.raises(ValueError):
        Split(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-06-01"),
            test_start=pd.Timestamp("2020-05-01"),  # before train_end -- illegal
            test_end=pd.Timestamp("2020-12-01"),
        )


def test_invalid_retrain_raises():
    with pytest.raises(ValueError):
        expanding_window_splits(_business_dates("2020-01-01", "2020-12-31"), retrain="monthly")


# --- holdout seal ------------------------------------------------------------


def test_split_covering_holdout_raises_without_unseal():
    dates = _business_dates("2015-01-01", "2023-06-30")
    with pytest.raises(LeakageError):
        expanding_window_splits(dates, retrain="yearly", min_train_years=2)


def test_split_covering_holdout_passes_with_unseal_token():
    dates = _business_dates("2015-01-01", "2023-06-30")
    splits = expanding_window_splits(
        dates, retrain="yearly", min_train_years=2, unseal_token="PREREG_FROZEN"
    )
    assert len(splits) > 0
    assert any(s.test_end >= pd.Timestamp("2023-01-01") for s in splits)


def test_wrong_unseal_token_still_raises():
    dates = _business_dates("2015-01-01", "2023-06-30")
    with pytest.raises(LeakageError):
        expanding_window_splits(dates, retrain="yearly", min_train_years=2, unseal_token="not-the-token")


# --- beats_baseline_gate -----------------------------------------------------


def _p_vs_q_dataset(model_wins_years: list[int], all_years: list[int]):
    """For each year, a single trading day with a two-ticker universe
    {"P" (label=1), "Q" (label=0)}. Baseline always (wrongly) picks "Q".
    Model picks "P" (correct) in `model_wins_years`, else "Q" (matches
    baseline, so it does not "win" that year)."""
    label_rows, model_rows, baseline_rows = [], [], []
    for year in all_years:
        day = pd.Timestamp(f"{year}-06-01")
        label_rows.append({"trade_date": day, "ticker": "P", "rank": 1, "return_t": 0.5, "label": 1, "label_spec_version": "v1", "as_of": day})
        label_rows.append({"trade_date": day, "ticker": "Q", "rank": 2, "return_t": 0.0, "label": 0, "label_spec_version": "v1", "as_of": day})

        baseline_rows.append({"trade_date": day, "ticker": "Q", "score": 1.0})
        pick = "P" if year in model_wins_years else "Q"
        model_rows.append({"trade_date": day, "ticker": pick, "score": 1.0})

    labels = pd.DataFrame(label_rows)
    baseline_preds = pd.DataFrame(baseline_rows)
    model_preds = pd.DataFrame(model_rows)
    return model_preds, baseline_preds, labels


def test_beats_baseline_gate_fails_at_4_of_7():
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    model_preds, baseline_preds, labels = _p_vs_q_dataset(model_wins_years=years[:4], all_years=years)
    result = WalkForwardResult(predictions=model_preds, labels=labels, k=1)

    gate = beats_baseline_gate(result, baseline_preds, min_years_won=5, min_years=7)

    assert isinstance(gate, GateResult)
    assert gate.years_won == 4
    assert gate.years_total == 7
    assert gate.passed is False
    assert "FAILED" in gate.verdict


def test_beats_baseline_gate_passes_at_5_of_7():
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    model_preds, baseline_preds, labels = _p_vs_q_dataset(model_wins_years=years[:5], all_years=years)
    result = WalkForwardResult(predictions=model_preds, labels=labels, k=1)

    gate = beats_baseline_gate(result, baseline_preds, min_years_won=5, min_years=7)

    assert gate.years_won == 5
    assert gate.years_total == 7
    assert gate.passed is True
    assert "PASSED" in gate.verdict


def test_beats_baseline_gate_undecidable_with_too_few_years():
    years = [2015, 2016, 2017]
    model_preds, baseline_preds, labels = _p_vs_q_dataset(model_wins_years=years, all_years=years)
    result = WalkForwardResult(predictions=model_preds, labels=labels, k=1)

    gate = beats_baseline_gate(result, baseline_preds, min_years_won=5, min_years=7)

    assert gate.passed is False
    assert "UNDECIDABLE" in gate.verdict


# --- end-to-end: run_walkforward -> per_year precision@10, hand-computed ----


class _IdentityScoreModel:
    """Trivial deterministic model: score = the `f1` feature column,
    verbatim. `.fit` is a no-op -- injected as `model_factory` so this test
    needs no lightgbm."""

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> "_IdentityScoreModel":
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": features["trade_date"].to_numpy(),
                "ticker": features["ticker"].to_numpy(),
                "score": features["f1"].to_numpy(),
            }
        )


def _e2e_dataset():
    """2020 = train year (score/labels irrelevant, model is a no-op-fit
    identity scorer). 2021 = single test year with two days:

    Day 1 (2021-01-04): score == return exactly -> top-10 by score is
    exactly the top-10 by return -> 10/10 hits -> precision@10 = 1.0.

    Day 2 (2021-01-05): the model's top-10 by score wrongly includes the
    two true rank-11/12 tickers (T11, T12) in place of two true top-10
    tickers (T9, T10) -> 8/10 hits -> precision@10 = 0.8.

    Hand-computed mean precision@10 for 2021 = (1.0 + 0.8) / 2 = 0.9.
    """
    tickers = [f"T{i}" for i in range(1, 13)]  # T1..T12
    train_day = pd.Timestamp("2020-06-01")
    day1 = pd.Timestamp("2021-01-04")
    day2 = pd.Timestamp("2021-01-05")

    # Returns are strictly descending T1 (highest) .. T12 (lowest) both days
    # -> true label top-10 by return is always {T1..T10}.
    returns = {t: 12 - i for i, t in enumerate(tickers)}  # T1:12 ... T12:1

    feature_rows, label_rows = [], []

    for day in (train_day, day1, day2):
        for t in tickers:
            ret = float(returns[t])
            label = 1 if ret >= 3.0 else 0  # top-10 returns are 12..3 -> label 1
            label_rows.append(
                {
                    "trade_date": day,
                    "ticker": t,
                    "rank": 13 - int(ret),
                    "return_t": ret,
                    "label": label,
                    "label_spec_version": "v1",
                    "as_of": day,
                }
            )

    for t in tickers:
        # Train day + day1: score == true return (perfect predictor).
        feature_rows.append({"trade_date": train_day, "ticker": t, "f1": float(returns[t]), "as_of": train_day})
        feature_rows.append({"trade_date": day1, "ticker": t, "f1": float(returns[t]), "as_of": day1})

    # Day2: swap scores so T11/T12 (true non-hits) outrank T9/T10 (true
    # hits), producing exactly 8 correct top-10 picks.
    day2_scores = dict(returns)
    day2_scores["T11"], day2_scores["T9"] = 100, 1  # T11 now scores above everything
    day2_scores["T12"], day2_scores["T10"] = 99, 0
    for t in tickers:
        feature_rows.append({"trade_date": day2, "ticker": t, "f1": float(day2_scores[t]), "as_of": day2})

    features = pd.DataFrame(feature_rows)
    labels = pd.DataFrame(label_rows)
    return features, labels


def test_run_walkforward_end_to_end_precision_matches_hand_computed_value():
    features, labels = _e2e_dataset()
    dates = sorted(features["trade_date"].unique())

    splits = expanding_window_splits(dates, retrain="yearly", min_train_years=1)
    # Only 2021 should be tested (2020 is the single reserved training year).
    assert len(splits) == 1
    assert splits[0].test_start.year == 2021

    result = run_walkforward(_IdentityScoreModel, features, labels, splits, k=10)

    assert not result.predictions.empty

    per_year = result.per_year
    assert list(per_year["year"]) == [2021]
    assert per_year.loc[0, "n_days"] == 2
    assert per_year.loc[0, "precision_at_k"] == pytest.approx(0.9)


# --- Defect 3: run_walkforward is a holdout-seal chokepoint independent of
# how `splits` was built (a hand-built `Sequence[Split]` bypasses
# `expanding_window_splits` entirely) -----------------------------------


def test_run_walkforward_raises_on_hand_built_holdout_split_without_token():
    hand_built_split = Split(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2022-12-31"),
        test_start=pd.Timestamp("2023-06-01"),
        test_end=pd.Timestamp("2023-06-30"),
    )
    features = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2022-06-01"), pd.Timestamp("2023-06-05")],
            "ticker": ["A", "A"],
            "f1": [1.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2022-06-01"), pd.Timestamp("2023-06-05")],
            "ticker": ["A", "A"],
            "rank": [1, 1],
            "return_t": [0.1, 0.1],
            "label": [1, 1],
            "label_spec_version": ["v1", "v1"],
            "as_of": [pd.Timestamp("2022-06-01"), pd.Timestamp("2023-06-05")],
        }
    )

    with pytest.raises(LeakageError):
        run_walkforward(_IdentityScoreModel, features, labels, [hand_built_split], k=10)


def test_run_walkforward_succeeds_on_hand_built_holdout_split_with_token():
    hand_built_split = Split(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2022-12-31"),
        test_start=pd.Timestamp("2023-06-01"),
        test_end=pd.Timestamp("2023-06-30"),
    )
    features = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2022-06-01"), pd.Timestamp("2023-06-05")],
            "ticker": ["A", "A"],
            "f1": [1.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2022-06-01"), pd.Timestamp("2023-06-05")],
            "ticker": ["A", "A"],
            "rank": [1, 1],
            "return_t": [0.1, 0.1],
            "label": [1, 1],
            "label_spec_version": ["v1", "v1"],
            "as_of": [pd.Timestamp("2022-06-01"), pd.Timestamp("2023-06-05")],
        }
    )

    result = run_walkforward(
        _IdentityScoreModel, features, labels, [hand_built_split], k=10, unseal_token="PREREG_FROZEN"
    )
    assert not result.predictions.empty


def test_walkforward_result_direct_construction_raises_on_holdout_without_token():
    predictions = pd.DataFrame({"trade_date": [pd.Timestamp("2023-06-05")], "ticker": ["A"], "score": [1.0]})
    labels = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2023-06-05")],
            "ticker": ["A"],
            "rank": [1],
            "return_t": [0.1],
            "label": [1],
            "label_spec_version": ["v1"],
            "as_of": [pd.Timestamp("2023-06-05")],
        }
    )
    with pytest.raises(LeakageError):
        WalkForwardResult(predictions=predictions, labels=labels, k=10)


def test_decay_plot_data_shape_matches_per_year():
    features, labels = _e2e_dataset()
    dates = sorted(features["trade_date"].unique())
    splits = expanding_window_splits(dates, retrain="yearly", min_train_years=1)
    result = run_walkforward(_IdentityScoreModel, features, labels, splits, k=10)

    decay = result.decay_plot_data()
    assert decay["year"] == [2021]
    assert decay["precision_at_k"] == pytest.approx([0.9])
    assert set(decay.keys()) == {"year", "precision_at_k", "map_at_k", "mean_hits", "n_days"}
