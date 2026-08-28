from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from top10 import metrics


def _labels_day(trade_date, top10_tickers, all_tickers):
    """Build a labels frame for one day: top10_tickers ranked 1..10 with
    label=1, all remaining tickers in all_tickers with label=0 and rank NaN.
    """
    rows = []
    for i, ticker in enumerate(top10_tickers, start=1):
        rows.append(
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "rank": i,
                "return_t": 1.0 / i,
                "label": 1,
                "label_spec_version": "v1",
                "as_of": trade_date,
            }
        )
    for ticker in all_tickers:
        if ticker in top10_tickers:
            continue
        rows.append(
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "rank": np.nan,
                "return_t": 0.0,
                "label": 0,
                "label_spec_version": "v1",
                "as_of": trade_date,
            }
        )
    return pd.DataFrame(rows)


DAY1 = dt.datetime(2024, 1, 2)
DAY2 = dt.datetime(2024, 1, 3)


def test_precision_at_k_hand_computed():
    # Day 1: top-10 predictions hit 6 of the true top 10.
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(10)]
    labels = _labels_day(DAY1, top10, universe)

    predicted_tickers = top10[:6] + [f"N{i}" for i in range(4)]
    scores = list(range(10, 0, -1))
    predictions = pd.DataFrame({"trade_date": DAY1, "ticker": predicted_tickers, "score": scores})

    assert metrics.precision_at_k(predictions, labels, k=10) == pytest.approx(0.6)


def test_precision_at_k_averages_across_days():
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(5)]
    labels1 = _labels_day(DAY1, top10, universe)
    labels2 = _labels_day(DAY2, top10, universe)
    labels = pd.concat([labels1, labels2], ignore_index=True)

    # Day 1: 10/10 hits. Day 2: 5/10 hits (top10[:5] are true positives,
    # the 5 N-tickers are not).
    preds1 = pd.DataFrame({"trade_date": DAY1, "ticker": top10, "score": list(range(10, 0, -1))})
    preds2 = pd.DataFrame({"trade_date": DAY2, "ticker": [f"N{i}" for i in range(5)] + top10[:5], "score": list(range(10, 0, -1))})
    predictions = pd.concat([preds1, preds2], ignore_index=True)

    result = metrics.precision_at_k(predictions, labels, k=10)
    assert result == pytest.approx((1.0 + 0.5) / 2)


def test_map_at_k_hand_computed_perfect_order():
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(5)]
    labels = _labels_day(DAY1, top10, universe)

    # Perfect: all 10 hits, in order -> AP = 1.0
    predictions = pd.DataFrame({"trade_date": DAY1, "ticker": top10, "score": list(range(10, 0, -1))})
    assert metrics.map_at_k(predictions, labels, k=10) == pytest.approx(1.0)


def test_map_at_k_hand_computed_two_hits():
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(5)]
    labels = _labels_day(DAY1, top10, universe)

    # Hits at position 1 and 3 out of 10 predictions; misses elsewhere.
    # (positions 8-10 use tickers outside the labeled top10/universe so
    # they don't accidentally count as hits)
    ordered = ["T0", "N0", "T1", "N1", "N2", "N3", "N4", "M0", "M1", "M2"]
    scores = list(range(10, 0, -1))
    predictions = pd.DataFrame({"trade_date": DAY1, "ticker": ordered, "score": scores})

    # AP@10 = (1/1 + 2/3) / 10   (only first two hits counted; denom is
    # min(n_relevant=10, k=10) = 10)
    expected = (1 / 1 + 2 / 3) / 10
    assert metrics.map_at_k(predictions, labels, k=10) == pytest.approx(expected)


def test_map_at_k_rank_sensitive_same_hit_set():
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(5)]
    labels = _labels_day(DAY1, top10, universe)

    same_hits_early = ["T0", "T1", "N0", "N1", "N2", "N3", "N4", "T2", "T3", "T4"]
    same_hits_late = ["N0", "N1", "T0", "T1", "N2", "N3", "N4", "T2", "T3", "T4"]

    preds_early = pd.DataFrame({"trade_date": DAY1, "ticker": same_hits_early, "score": list(range(10, 0, -1))})
    preds_late = pd.DataFrame({"trade_date": DAY1, "ticker": same_hits_late, "score": list(range(10, 0, -1))})

    map_early = metrics.map_at_k(preds_early, labels, k=10)
    map_late = metrics.map_at_k(preds_late, labels, k=10)
    assert map_early > map_late


def test_hits_per_day_exact_counts():
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(5)]
    labels1 = _labels_day(DAY1, top10, universe)
    labels2 = _labels_day(DAY2, top10, universe)
    labels = pd.concat([labels1, labels2], ignore_index=True)

    preds1 = pd.DataFrame({"trade_date": DAY1, "ticker": top10[:7] + [f"N{i}" for i in range(3)], "score": list(range(10, 0, -1))})
    preds2 = pd.DataFrame({"trade_date": DAY2, "ticker": [f"N{i}" for i in range(5)] + top10[:5], "score": list(range(10, 0, -1))})
    predictions = pd.concat([preds1, preds2], ignore_index=True)

    hits = metrics.hits_per_day(predictions, labels, k=10)
    assert hits[DAY1] == 7
    assert hits[DAY2] == 5


def test_per_year_report_columns_and_grouping():
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(5)]
    day_2023 = dt.datetime(2023, 6, 1)
    day_2024 = dt.datetime(2024, 6, 1)
    labels = pd.concat([_labels_day(day_2023, top10, universe), _labels_day(day_2024, top10, universe)], ignore_index=True)

    preds_2023 = pd.DataFrame({"trade_date": day_2023, "ticker": top10, "score": list(range(10, 0, -1))})
    preds_2024 = pd.DataFrame({"trade_date": day_2024, "ticker": [f"N{i}" for i in range(5)] + top10[:5], "score": list(range(10, 0, -1))})
    predictions = pd.concat([preds_2023, preds_2024], ignore_index=True)

    report = metrics.per_year_report(predictions, labels, k=10)
    assert set(report.columns) == {"year", "precision_at_k", "map_at_k", "mean_hits", "median_hits", "n_days"}
    assert sorted(report["year"].tolist()) == [2023, 2024]
    row_2023 = report[report["year"] == 2023].iloc[0]
    assert row_2023["precision_at_k"] == pytest.approx(1.0)
    assert row_2023["n_days"] == 1


def _daily_hits_frames(model_hits, baseline_hits, start=DAY1):
    dates = [start + dt.timedelta(days=i) for i in range(len(model_hits))]
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(5)]

    label_frames = [_labels_day(d, top10, universe) for d in dates]
    labels = pd.concat(label_frames, ignore_index=True)

    def preds_for(hits_list):
        rows = []
        for d, n_hit in zip(dates, hits_list):
            tickers = top10[:n_hit] + [f"N{i}" for i in range(10 - n_hit)]
            rows.append(pd.DataFrame({"trade_date": d, "ticker": tickers, "score": list(range(10, 0, -1))}))
        return pd.concat(rows, ignore_index=True)

    return preds_for(model_hits), preds_for(baseline_hits), labels


def test_compare_to_baseline_mean_delta_and_years():
    model_hits = [8, 9, 7, 8]
    baseline_hits = [3, 4, 2, 3]
    model_preds, baseline_preds, labels = _daily_hits_frames(model_hits, baseline_hits)

    result = metrics.compare_to_baseline(model_preds, baseline_preds, labels, k=10)
    assert result["mean_hits_delta"] == pytest.approx(np.mean(model_hits) - np.mean(baseline_hits))
    assert result["n_days"] == 4
    assert "paired_t_test" in result and "p_value" in result["paired_t_test"]
    assert "wilcoxon_test" in result and "p_value" in result["wilcoxon_test"]
    assert result["years_total"] == 1
    assert result["years_won"] == 1


def test_paired_t_test_pins_scipy_pvalue_on_fixed_input():
    # Regression pin: if a future scipy upgrade changes ttest_rel's
    # behaviour, this catches the shift instead of it silently changing
    # the significance of the final PREREG_TOP10 claim.
    a = [8.0, 9.0, 7.0, 8.0, 6.0, 9.0, 5.0, 8.0, 7.0, 6.0, 8.0, 9.0]
    b = [3.0, 4.0, 2.0, 3.0, 3.0, 4.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0]

    result = metrics.paired_t_test(a, b)
    assert result["n"] == 12
    assert result["statistic"] == pytest.approx(16.911534525287763)
    assert result["p_value"] == pytest.approx(3.2048253110561305e-09)


def test_wilcoxon_pins_scipy_pvalue_on_fixed_input():
    a = [8.0, 9.0, 7.0, 8.0, 6.0, 9.0, 5.0, 8.0, 7.0, 6.0, 8.0, 9.0]
    b = [3.0, 4.0, 2.0, 3.0, 3.0, 4.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0]

    result = metrics.wilcoxon_signed_rank_test(a, b)
    assert result["n"] == 12
    assert result["statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(0.00048828125)


def test_paired_t_test_degenerate_n_less_than_2():
    result = metrics.paired_t_test([5.0], [3.0])
    assert result["n"] == 1
    assert np.isnan(result["statistic"])
    assert np.isnan(result["p_value"])

    result_empty = metrics.paired_t_test([], [])
    assert result_empty["n"] == 0
    assert np.isnan(result_empty["p_value"])


def test_wilcoxon_degenerate_all_zero_differences():
    # Many days where the model exactly ties the baseline's hit count --
    # this must not raise/warn into a holdout run, and must report the
    # honest null result (no evidence of a difference).
    a = [5.0, 5.0, 5.0, 5.0]
    b = [5.0, 5.0, 5.0, 5.0]

    result = metrics.wilcoxon_signed_rank_test(a, b)
    assert result["n"] == 4
    assert result["statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_wilcoxon_degenerate_empty_input():
    result = metrics.wilcoxon_signed_rank_test([], [])
    assert result["n"] == 0
    assert np.isnan(result["p_value"])


def test_family_wise_correction_holm_worked_example():
    pvalues = [0.01, 0.04, 0.03, 0.005]
    adjusted = metrics.family_wise_correction(pvalues, method="holm")
    expected = [0.03, 0.06, 0.06, 0.02]
    assert adjusted == pytest.approx(expected)


def test_family_wise_correction_holm_second_independent_worked_example():
    # Classic textbook example (already sorted ascending), computed by
    # hand independently of the implementation:
    #   m=5, p=[.01,.02,.03,.04,.05]
    #   raw_i = (m-i+1)*p_i -> [0.05, 0.08, 0.09, 0.08, 0.05]
    #   cumulative max (Holm monotonization) -> [0.05, 0.08, 0.09, 0.09, 0.09]
    pvalues = [0.01, 0.02, 0.03, 0.04, 0.05]
    adjusted = metrics.family_wise_correction(pvalues, method="holm")
    expected = [0.05, 0.08, 0.09, 0.09, 0.09]
    assert adjusted == pytest.approx(expected)


def test_family_wise_correction_never_decreases_relative_order_violation():
    # A later (larger) raw p-value must never end up with a smaller
    # adjusted p-value than an earlier, smaller raw p-value once monotonized.
    pvalues = [0.001, 0.2, 0.19, 0.3]
    adjusted = metrics.family_wise_correction(pvalues, method="holm")
    # sorted order of adjusted values by ascending raw p should be non-decreasing
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted_in_p_order = [adjusted[i] for i in order]
    assert adjusted_in_p_order == sorted(adjusted_in_p_order)


def test_module_has_no_accuracy_function():
    assert not hasattr(metrics, "accuracy")
    assert not hasattr(metrics, "accuracy_score")
