"""Evaluation metrics for the TOP10 daily top-movers predictor.

Per docs/PREREG_TOP10.md the primary metric is precision@10 and the
secondary metric is MAP@10. Per plan §5 (P5), the label set is extremely
imbalanced -- roughly 10 positives out of ~4,000 candidates per day -- so a
naive "accuracy" number is meaningless: a model that predicts all-negative
scores ~99.75% "accuracy" while being useless. This module deliberately
does NOT expose an accuracy function. Only precision@k and MAP@k (and the
raw hit counts they're built from) are reported.

`predictions` frames carry: trade_date, ticker, score.
`labels` frames carry: trade_date, ticker, rank, return_t, label,
label_spec_version, as_of (see docs/LABEL_SPEC.md).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _top_k_per_day(predictions: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return the top-`k` predictions per `trade_date`, ranked by score desc.

    Ties are broken deterministically by `ticker` ascending so results are
    reproducible regardless of input row order.
    """
    if predictions.empty:
        return predictions.copy()

    ranked = predictions.sort_values(
        ["trade_date", "score", "ticker"], ascending=[True, False, True]
    )
    ranked["_pos"] = ranked.groupby("trade_date").cumcount()
    return ranked[ranked["_pos"] < k].drop(columns="_pos")


def _labels_lookup(labels: pd.DataFrame) -> pd.DataFrame:
    return labels[["trade_date", "ticker", "label"]]


def hits_per_day(predictions: pd.DataFrame, labels: pd.DataFrame, k: int = 10) -> pd.Series:
    """Raw count of true positives within the top-`k` predictions, per day.

    Indexed by `trade_date`. This is the unit PREREG_TOP10's success
    criterion ("beats B4 by >= 1.0 average hits/day") is written in, so it
    must be exact -- no averaging or normalization here.
    """
    topk = _top_k_per_day(predictions, k)
    if topk.empty:
        return pd.Series(dtype=float, name="hits")

    merged = topk.merge(
        _labels_lookup(labels), on=["trade_date", "ticker"], how="left"
    )
    merged["label"] = merged["label"].fillna(0)
    series = merged.groupby("trade_date")["label"].sum()
    series.name = "hits"
    return series


def precision_at_k(predictions: pd.DataFrame, labels: pd.DataFrame, k: int = 10) -> float:
    """Mean precision@k across days.

    Per-day precision = (# hits in top-k) / (# predictions actually taken,
    capped at k). Averaged across all days present in `predictions`.
    """
    topk = _top_k_per_day(predictions, k)
    if topk.empty:
        return float("nan")

    merged = topk.merge(
        _labels_lookup(labels), on=["trade_date", "ticker"], how="left"
    )
    merged["label"] = merged["label"].fillna(0)

    per_day = merged.groupby("trade_date").agg(hits=("label", "sum"), n=("label", "size"))
    per_day["precision"] = per_day["hits"] / per_day["n"]
    return float(per_day["precision"].mean())


def _average_precision_for_day(day_topk: pd.DataFrame, n_relevant: int, k: int) -> float:
    """Rank-aware average precision for a single day's ranked top-k rows.

    `day_topk` must already be sorted best-score-first. Denominator is
    min(n_relevant, k) -- the standard AP@k convention.
    """
    denom = min(n_relevant, k)
    if denom <= 0:
        return float("nan")

    hits_so_far = 0
    precision_sum = 0.0
    for i, is_hit in enumerate(day_topk["label"].tolist(), start=1):
        if is_hit:
            hits_so_far += 1
            precision_sum += hits_so_far / i
    return precision_sum / denom


def map_at_k(predictions: pd.DataFrame, labels: pd.DataFrame, k: int = 10) -> float:
    """Mean average precision@k across days. Rank-sensitive: for a fixed
    hit set, ordering hits earlier in the ranking strictly increases AP.
    """
    topk = _top_k_per_day(predictions, k)
    if topk.empty:
        return float("nan")

    topk = topk.sort_values(["trade_date", "score", "ticker"], ascending=[True, False, True])
    topk = topk.merge(_labels_lookup(labels), on=["trade_date", "ticker"], how="left")
    topk["label"] = topk["label"].fillna(0).astype(int)

    n_relevant_by_day = labels[labels["label"] == 1].groupby("trade_date").size()

    aps = []
    for trade_date, day_rows in topk.groupby("trade_date", sort=False):
        n_relevant = int(n_relevant_by_day.get(trade_date, 0))
        ap = _average_precision_for_day(day_rows, n_relevant, k)
        if not math.isnan(ap):
            aps.append(ap)

    if not aps:
        return float("nan")
    return float(np.mean(aps))


def per_year_report(predictions: pd.DataFrame, labels: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Per-calendar-year precision@10, MAP@10, mean/median hits, n_days.

    Required by PREREG_TOP10 §5.2.
    """
    hits = hits_per_day(predictions, labels, k)
    if hits.empty:
        return pd.DataFrame(
            columns=["year", "precision_at_k", "map_at_k", "mean_hits", "median_hits", "n_days"]
        )

    years = pd.to_datetime(hits.index).year
    rows = []
    for year in sorted(set(years)):
        mask_dates = pd.to_datetime(hits.index).year == year
        year_dates = hits.index[mask_dates]

        year_preds = predictions[predictions["trade_date"].isin(year_dates)]
        year_labels = labels[labels["trade_date"].isin(year_dates)]
        year_hits = hits[mask_dates]

        rows.append(
            {
                "year": int(year),
                "precision_at_k": precision_at_k(year_preds, year_labels, k),
                "map_at_k": map_at_k(year_preds, year_labels, k),
                "mean_hits": float(year_hits.mean()),
                "median_hits": float(year_hits.median()),
                "n_days": int(len(year_hits)),
            }
        )

    return pd.DataFrame(rows)


# --- Significance testing ----------------------------------------------
#
# These feed the PREREG_TOP10 primary success criterion directly ("beats
# B4 by >= 1.0 average hits/day with corrected significance"), so the
# paired t-test and Wilcoxon signed-rank test are delegated to scipy.stats
# rather than hand-rolled: a subtle numerical bug in a hand-rolled
# incomplete-beta or normal-approximation Wilcoxon is exactly the kind of
# thing that would silently corrupt the one claim this project exists to
# make. `family_wise_correction` below stays hand-implemented -- it's
# simple, exact, and not worth a statsmodels dependency.


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired two-sided t-test on `a - b`, via `scipy.stats.ttest_rel`.
    Returns {statistic, p_value, n}.

    Degenerate case (n < 2): returns a clearly-marked null result rather
    than letting scipy raise/warn into a holdout run.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n}

    result = stats.ttest_rel(a, b)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n}


def wilcoxon_signed_rank_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test on `a - b`, via
    `scipy.stats.wilcoxon`. Returns {statistic, p_value, n}.

    `zero_method="pratt"` is chosen deliberately over the scipy default
    ("wilcox"): daily hit-count differences between a model and B4 will
    contain many exact ties at zero (identical hit counts on plenty of
    days), and "wilcox" simply drops every zero-difference day before
    ranking -- discarding real information about how often the model
    fails to beat the baseline and shrinking the effective sample size.
    "pratt" keeps zero differences in the ranking step (they contribute
    ranks that are then excluded only from the signed-rank sum), which is
    the standard recommendation when zeros are frequent rather than a
    rare tie-breaking edge case.

    Degenerate cases handled explicitly rather than letting scipy raise
    into a holdout run:
    - n < 1: null result.
    - all differences exactly zero: honest null result (no evidence of a
      difference), since scipy's normal-approximation path divides by a
      zero standard error in this case.
    - any other scipy `ValueError` (e.g. too few non-zero differences for
      the requested mode): null result, with the scipy message preserved
      under `error` for debugging.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    n = len(diffs)
    if n < 1:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": 0}

    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0, "n": n}

    try:
        result = stats.wilcoxon(a, b, zero_method="pratt", mode="auto")
    except ValueError as exc:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n, "error": str(exc)}

    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n}


def compare_to_baseline(
    model_preds: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    labels: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Compare model vs. baseline on daily hit counts.

    Returns mean hits delta, per-year win/loss record, `years_won` /
    `years_total` (for the ">= 5 of 7 years" gate), and paired significance
    (both a paired t-test and a distribution-free Wilcoxon signed-rank
    test) on the aligned daily hit counts.
    """
    model_hits = hits_per_day(model_preds, labels, k)
    baseline_hits = hits_per_day(baseline_preds, labels, k)

    common_dates = model_hits.index.intersection(baseline_hits.index)
    model_hits = model_hits.loc[common_dates].sort_index()
    baseline_hits = baseline_hits.loc[common_dates].sort_index()

    mean_hits_delta = float(model_hits.mean() - baseline_hits.mean())

    years = pd.to_datetime(model_hits.index).year
    per_year_record: dict[int, dict] = {}
    for year in sorted(set(years)):
        mask = years == year
        model_mean = float(model_hits[mask].mean())
        baseline_mean = float(baseline_hits[mask].mean())
        per_year_record[int(year)] = {
            "model_mean_hits": model_mean,
            "baseline_mean_hits": baseline_mean,
            "model_won": model_mean > baseline_mean,
        }

    years_won = sum(1 for rec in per_year_record.values() if rec["model_won"])
    years_total = len(per_year_record)

    t_result = paired_t_test(model_hits.to_numpy(), baseline_hits.to_numpy())
    wilcoxon_result = wilcoxon_signed_rank_test(model_hits.to_numpy(), baseline_hits.to_numpy())

    return {
        "mean_hits_delta": mean_hits_delta,
        "per_year_record": per_year_record,
        "years_won": years_won,
        "years_total": years_total,
        "paired_t_test": t_result,
        "wilcoxon_test": wilcoxon_result,
        "n_days": int(len(common_dates)),
    }


def family_wise_correction(pvalues: Sequence[float], method: str = "holm") -> list[float]:
    """Holm-Bonferroni family-wise error correction.

    Per PREREG_TOP10 P12, the final claim must be corrected for the number
    of model variants tested. Implemented directly (no statsmodels
    dependency). Returns adjusted p-values in the same order as `pvalues`.
    """
    if method != "holm":
        raise ValueError(f"family_wise_correction: unsupported method {method!r}")

    m = len(pvalues)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted_sorted = []
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, candidate)
        adjusted_sorted.append(running_max)

    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted
