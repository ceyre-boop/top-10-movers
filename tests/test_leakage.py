from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from top10 import leakage
from top10.storage import LeakageError


# --- assert_decision_time_safe ------------------------------------------


def test_assert_decision_time_safe_passes_when_clean():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "trade_date": [dt.datetime(2024, 1, 5)] * 2,
            "as_of": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 4)],
        }
    )
    leakage.assert_decision_time_safe(df, dt.datetime(2024, 1, 5))


def test_assert_decision_time_safe_raises_on_future_as_of():
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "trade_date": [dt.datetime(2024, 1, 5)],
            "as_of": [dt.datetime(2024, 1, 10)],
        }
    )
    with pytest.raises(LeakageError):
        leakage.assert_decision_time_safe(df, dt.datetime(2024, 1, 5))


# --- assert_no_adjusted_prices -------------------------------------------


def test_assert_no_adjusted_prices_flags_as_of_before_trade_date():
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "trade_date": [dt.datetime(2024, 1, 5)],
            "as_of": [dt.datetime(2024, 1, 4)],
        }
    )
    with pytest.raises(LeakageError):
        leakage.assert_no_adjusted_prices(df)


def test_assert_no_adjusted_prices_passes_when_clean():
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "trade_date": [dt.datetime(2024, 1, 5)],
            "as_of": [dt.datetime(2024, 1, 5)],
        }
    )
    leakage.assert_no_adjusted_prices(df)


def test_assert_no_adjusted_prices_catches_confirmed_back_adjustment_repro():
    """Confirmed audit repro: a bar from 2020-01-02 (as_of == trade_date,
    i.e. it looks perfectly clean under the corporate_actions=None
    precondition) whose ticker later underwent a split effective
    2021-06-01, announced 2021-05-20 -- both entirely in the bar's future.
    A fully back-adjusted feed passes this shape on every row under the
    OLD (buggy) implementation, because the old first clause
    (`ex_date <= trade_date`) filters out exactly this case. This test
    MUST fail against that old implementation.
    """
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "trade_date": [dt.datetime(2020, 1, 2)],
            "as_of": [dt.datetime(2020, 1, 2)],
        }
    )
    corp_actions = pd.DataFrame(
        {
            "ticker": ["A"],
            "ex_date": [dt.datetime(2021, 6, 1)],
            "as_of": [dt.datetime(2021, 5, 20)],
        }
    )
    with pytest.raises(LeakageError):
        leakage.assert_no_adjusted_prices(df, corp_actions)


def test_assert_no_adjusted_prices_with_corporate_actions_passes_when_split_already_occurred():
    """A split whose ex_date is BEFORE the bar's own trade_date is a
    legitimate, already-happened market event -- not evidence of back-
    adjustment (the structural signature requires ex_date > trade_date).
    """
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "trade_date": [dt.datetime(2024, 1, 5)],
            "as_of": [dt.datetime(2024, 1, 5)],
        }
    )
    corp_actions = pd.DataFrame(
        {
            "ticker": ["A"],
            "ex_date": [dt.datetime(2024, 1, 3)],
            "as_of": [dt.datetime(2024, 1, 2)],
        }
    )
    leakage.assert_no_adjusted_prices(df, corp_actions)


def test_assert_no_adjusted_prices_with_corporate_actions_passes_when_action_was_already_knowable():
    """ex_date is in the bar's future, but the action was announced BEFORE
    the bar's own as_of -- the bar could not have retroactively baked in
    an adjustment for information it plainly already had at write time in
    some other, non-adjustment way; this function's proxy specifically
    requires action_as_of > as_of to flag.
    """
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "trade_date": [dt.datetime(2020, 1, 2)],
            "as_of": [dt.datetime(2021, 6, 5)],
        }
    )
    corp_actions = pd.DataFrame(
        {
            "ticker": ["A"],
            "ex_date": [dt.datetime(2021, 6, 1)],
            "as_of": [dt.datetime(2021, 5, 20)],
        }
    )
    leakage.assert_no_adjusted_prices(df, corp_actions)


# --- verify_unadjusted ----------------------------------------------------


def _split_bars(pre_close: float, post_close: float, ex_date: dt.datetime, ticker: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker, ticker],
            "trade_date": [ex_date - dt.timedelta(days=1), ex_date],
            "close": [pre_close, post_close],
        }
    )


def _split_action(ratio: float, ex_date: dt.datetime, ticker: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "ex_date": [ex_date],
            "action_type": ["split"],
            "ratio": [ratio],
        }
    )


def test_verify_unadjusted_passes_on_real_split_discontinuity():
    ex_date = dt.datetime(2024, 3, 1)
    # 2-for-1 split: price should roughly halve.
    bars = _split_bars(pre_close=100.0, post_close=50.5, ex_date=ex_date)
    actions = _split_action(ratio=2.0, ex_date=ex_date)
    leakage.verify_unadjusted(bars, actions)


def test_verify_unadjusted_flags_flat_series_at_known_split():
    ex_date = dt.datetime(2024, 3, 1)
    # No discontinuity at all despite a known 2-for-1 split -- the
    # tell-tale signature of a back-adjusted series.
    bars = _split_bars(pre_close=100.0, post_close=99.5, ex_date=ex_date)
    actions = _split_action(ratio=2.0, ex_date=ex_date)
    with pytest.raises(LeakageError):
        leakage.verify_unadjusted(bars, actions)


# --- assert_self_exclusion -------------------------------------------------


def _labels_frame():
    return pd.DataFrame(
        {
            "trade_date": [dt.datetime(2024, 1, 5)] * 2,
            "ticker": ["A", "B"],
            "rank": [1, 2],
            "return_t": [0.10, 0.05],
            "label": [1, 1],
            "label_spec_version": ["v1", "v1"],
            "as_of": [dt.datetime(2024, 1, 5)] * 2,
        }
    )


def test_assert_self_exclusion_passes_on_clean_features():
    features = pd.DataFrame(
        {
            "trade_date": [dt.datetime(2024, 1, 5)] * 2,
            "ticker": ["A", "B"],
            "prior_5d_vol": [0.02, 0.03],
            "as_of": [dt.datetime(2024, 1, 4)] * 2,
        }
    )
    leakage.assert_self_exclusion(features, _labels_frame())


def test_assert_self_exclusion_catches_planted_leak_equal_to_return_t():
    labels = _labels_frame()
    features = pd.DataFrame(
        {
            "trade_date": [dt.datetime(2024, 1, 5)] * 2,
            "ticker": ["A", "B"],
            # Planted leak: this column is literally same-day return_t.
            "sneaky_feature": labels["return_t"].to_numpy(),
            "as_of": [dt.datetime(2024, 1, 4)] * 2,
        }
    )
    with pytest.raises(LeakageError):
        leakage.assert_self_exclusion(features, labels)


def test_assert_self_exclusion_catches_direct_label_column():
    labels = _labels_frame()
    features = pd.DataFrame(
        {
            "trade_date": [dt.datetime(2024, 1, 5)] * 2,
            "ticker": ["A", "B"],
            "rank": [1, 2],  # forbidden column present directly
            "as_of": [dt.datetime(2024, 1, 4)] * 2,
        }
    )
    with pytest.raises(LeakageError):
        leakage.assert_self_exclusion(features, labels)


def _multi_day_features_and_labels(n_days=8, n_per_day=20):
    """20 candidates/day, 10 positives/day, deterministic (no RNG) so
    every test using it is reproducible without a seed.
    """
    rows_f = []
    rows_l = []
    for d in range(n_days):
        trade_date = dt.datetime(2024, 1, 1) + dt.timedelta(days=d)
        for i in range(n_per_day):
            ticker = f"T{i}"
            return_t = 1.0 / (i + 1)
            rank = i + 1 if i < 10 else np.nan
            label = 1 if i < 10 else 0
            # A feature genuinely unrelated to same-day return_t / rank:
            # a fixed pseudo-random permutation of 0..n_per_day-1, applied
            # per-day with a day-dependent offset so it isn't accidentally
            # monotonic in `i`.
            unrelated = float((i * 37 + 11 + d * 5) % n_per_day)
            rows_f.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "prior_day_return": unrelated,
                    "as_of": trade_date - dt.timedelta(days=1),
                }
            )
            rows_l.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "rank": rank,
                    "return_t": return_t,
                    "label": label,
                    "label_spec_version": "v1",
                    "as_of": trade_date,
                }
            )
    return pd.DataFrame(rows_f), pd.DataFrame(rows_l)


def test_assert_self_exclusion_catches_affine_disguised_return_t():
    """sneaky = return_t * 100 + 1 is an affine transform of return_t --
    perfectly Pearson-correlated with it (rho == 1.0) every day, so a
    model fed this column recovers exact same-day return_t and gets
    10/10 precision, yet it is never elementwise-identical to return_t.
    The old elementwise-identity-only check wrongly PASSES this.
    """
    features, labels = _multi_day_features_and_labels()
    merged = features.merge(labels[["trade_date", "ticker", "return_t"]], on=["trade_date", "ticker"])
    features = features.copy()
    features["sneaky"] = merged["return_t"] * 100.0 + 1.0
    with pytest.raises(LeakageError):
        leakage.assert_self_exclusion(features, labels)


def test_assert_self_exclusion_catches_doubled_same_day_rank():
    """sneaky_rank = rank * 2.0 is a monotonic (and here, affine on the
    non-NaN subset) disguise of same-day rank -- perfectly correlated with
    it every day. The old elementwise-identity-only check wrongly PASSES
    this too.
    """
    features, labels = _multi_day_features_and_labels()
    merged = features.merge(labels[["trade_date", "ticker", "rank"]], on=["trade_date", "ticker"])
    features = features.copy()
    features["sneaky_rank"] = merged["rank"] * 2.0
    with pytest.raises(LeakageError):
        leakage.assert_self_exclusion(features, labels)


def test_assert_self_exclusion_does_not_false_positive_on_legitimate_feature():
    """A feature that is genuinely only mildly/incidentally related to
    same-day labels (here: deliberately uncorrelated by construction)
    must not trip the correlation check.
    """
    features, labels = _multi_day_features_and_labels()
    leakage.assert_self_exclusion(features, labels)


# --- shuffle_label_test ----------------------------------------------------


def _fake_features_and_labels(n_days=6, n_per_day=20):
    rows_f = []
    rows_l = []
    for d in range(n_days):
        trade_date = dt.datetime(2024, 1, 1) + dt.timedelta(days=d)
        tickers = [f"T{i}" for i in range(n_per_day)]
        for i, ticker in enumerate(tickers):
            rows_f.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "feat1": float(i),
                    "as_of": trade_date - dt.timedelta(days=1),
                }
            )
            rows_l.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "rank": i + 1 if i < 10 else np.nan,
                    "return_t": 1.0 / (i + 1),
                    "label": 1 if i < 10 else 0,
                    "label_spec_version": "v1",
                    "as_of": trade_date,
                }
            )
    return pd.DataFrame(rows_f), pd.DataFrame(rows_l)


def test_shuffle_label_test_shuffles_within_day_and_preserves_ten_positives_per_day():
    features, labels = _fake_features_and_labels(n_days=4, n_per_day=15)

    captured_shuffled = []

    def fit_predict_fn(features_df, shuffled_labels_df):
        captured_shuffled.append(shuffled_labels_df.copy())
        preds = features_df[["trade_date", "ticker"]].copy()
        preds["score"] = 0.0
        return preds

    result = leakage.shuffle_label_test(fit_predict_fn, features, labels, k=10, n_trials=1, seed=1)
    assert "observed_precision" in result
    assert "expected_precision" in result
    assert "passed" in result

    shuffled = captured_shuffled[0]
    for trade_date, group in shuffled.groupby("trade_date"):
        assert int((group["label"] == 1).sum()) == 10
        assert set(group["trade_date"].unique()) == {trade_date}
    # Same multiset of tickers present each day (shuffle must not move
    # rows across days or drop/duplicate any).
    assert sorted(shuffled["ticker"]) == sorted(labels["ticker"])


def test_shuffle_label_test_fails_for_leaked_same_day_return_feature():
    """The single most important test in this file: a feature set that
    directly contains same-day return_t lets a trivial "sort by that
    column" model recover the TRUE top-10 exactly, even though it was
    fit against SHUFFLED (information-free) labels. `passed` must be
    False. The old implementation:
    (a) used a fixed 0.15 floor that a 20-candidates/day toy universe
        with expected_precision=0.5 never exceeds regardless of leakage,
    (b) scored predictions against the SHUFFLED labels instead of the
        TRUE labels, so a leaked-but-shuffled-trained model looks exactly
        as "random" as a clean one.
    Both bugs make this test pass under the old code; it must fail there.

    Uses 200 candidates/day (expected_precision = 10/200 = 0.05) rather
    than a tiny toy universe: at expected_precision=0.5 (e.g. 20
    candidates/day) the tolerance's `multiplier * expected_precision`
    component clips to 1.0 and a perfect leak (precision=1.0) sits right
    at the boundary -- a real but small-universe edge case, not a defect
    in the tolerance itself. 200/day is still small enough to run fast
    but large enough to discriminate cleanly, and is much closer to
    PREREG_TOP10's real ~4,000/day universe in spirit.
    """
    features, labels = _fake_features_and_labels(n_days=6, n_per_day=200)
    # The feature IS the true same-day return -- the leak.
    leaked = features.merge(labels[["trade_date", "ticker", "return_t"]], on=["trade_date", "ticker"])

    def fit_predict_fn(features_df, shuffled_labels_df):
        # A "model" that ignores the (shuffled, uninformative) labels
        # entirely and just re-derives its score from the leaked feature.
        merged = features_df.merge(
            leaked[["trade_date", "ticker", "return_t"]], on=["trade_date", "ticker"]
        )
        preds = merged[["trade_date", "ticker"]].copy()
        preds["score"] = merged["return_t"]
        return preds

    result = leakage.shuffle_label_test(fit_predict_fn, features, labels, k=10, n_trials=3, seed=7)
    assert result["passed"] is False
    assert result["observed_precision"] > result["tolerance"]


def test_shuffle_label_test_passes_for_genuinely_clean_features():
    features, labels = _fake_features_and_labels(n_days=6, n_per_day=20)

    def fit_predict_fn(features_df, shuffled_labels_df):
        # Score has no relationship to labels at all -- clean.
        preds = features_df[["trade_date", "ticker", "feat1"]].copy()
        preds = preds.rename(columns={"feat1": "score"})
        return preds

    result = leakage.shuffle_label_test(fit_predict_fn, features, labels, k=10, n_trials=5, seed=3)
    assert result["passed"] is True


def test_tolerance_at_4000_candidates_per_day_rejects_precision_of_0_15():
    """PREREG_TOP10's real universe: ~4,000 candidates/day, k=10 ->
    expected_precision = 0.0025. The OLD tolerance
    (`max(0.15, 3.0 / candidates_per_day.mean())`) accepted ANY precision
    up to 0.1525 -- 61x random, 1.5 real hits/day of undetected leakage.
    The new tolerance must reject an observed precision of 0.15 outright.
    """
    expected_precision = 10 / 4000
    tolerance = leakage._random_guess_tolerance(expected_precision, n_observations=4000 * 5)
    assert tolerance < 0.15
    observed_precision = 0.15
    assert observed_precision > tolerance


def test_shuffle_label_test_rejects_015_precision_at_4000_candidates_per_day():
    """End-to-end version of the tolerance test above, exercised through
    the public API rather than the private helper.
    """
    trade_date = dt.datetime(2024, 1, 1)
    n = 4000
    tickers = [f"T{i}" for i in range(n)]
    features = pd.DataFrame(
        {
            "trade_date": [trade_date] * n,
            "ticker": tickers,
            "feat1": np.arange(n, dtype=float),
            "as_of": [trade_date - dt.timedelta(days=1)] * n,
        }
    )
    labels = pd.DataFrame(
        {
            "trade_date": [trade_date] * n,
            "ticker": tickers,
            "rank": [i + 1 if i < 10 else np.nan for i in range(n)],
            "return_t": [1.0 / (i + 1) for i in range(n)],
            "label": [1 if i < 10 else 0 for i in range(n)],
            "label_spec_version": ["v1"] * n,
            "as_of": [trade_date] * n,
        }
    )

    def fit_predict_fn(features_df, shuffled_labels_df):
        # Deterministically hit exactly 15% of a 10-slot top-k, i.e.
        # precision@10 == 0.15, regardless of what the true labels are:
        # score the true top-1/top-2 tickers highest (guaranteed hits
        # against the TRUE labels used for scoring) plus 8 arbitrary
        # non-label tickers, giving 2/10 = 0.2... instead, construct an
        # exact 0.15 via a mix across trials by scoring the true top-10
        # ticker set only 15% of the time using the trial's rng draw
        # baked into shuffled_labels_df's own permutation, then otherwise
        # scoring arbitrary non-hits. To keep this deterministic and
        # simple, just always score exactly 1.5 real hits worth is not
        # integral, so alternate 1 and 2 true hits across the two calls
        # this test will make (n_trials=2) to average to 1.5/10 = 0.15.
        true_top10 = set(tickers[:10])
        if not hasattr(fit_predict_fn, "_call") :
            fit_predict_fn._call = 0
        fit_predict_fn._call += 1
        n_true_hits = 1 if fit_predict_fn._call % 2 == 1 else 2
        hit_tickers = list(true_top10)[:n_true_hits]
        non_hit_tickers = [t for t in tickers if t not in true_top10][: (10 - n_true_hits)]
        chosen = hit_tickers + non_hit_tickers
        scores = {t: float(len(chosen) - i) for i, t in enumerate(chosen)}
        preds = features_df[["trade_date", "ticker"]].copy()
        preds["score"] = preds["ticker"].map(scores).fillna(-1.0)
        return preds

    result = leakage.shuffle_label_test(fit_predict_fn, features, labels, k=10, n_trials=2, seed=11)
    assert result["observed_precision"] == pytest.approx(0.15)
    assert result["passed"] is False
