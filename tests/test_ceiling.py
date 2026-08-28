"""Tests for top10/ceiling.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from top10.ceiling import (
    CLASSIFICATION_RUBRIC,
    classify,
    estimate_ceiling,
    render_ceiling_md,
    sample_positives,
)
from top10.experiment import HOLDOUT_START
from top10.storage import LeakageError


# --- fixtures ------------------------------------------------------------


def _labels_row(trade_date, ticker, label):
    return {
        "trade_date": pd.Timestamp(trade_date),
        "ticker": ticker,
        "rank": 1,
        "return_t": 0.3,
        "label": label,
        "label_spec_version": "test-hash",
        "as_of": pd.Timestamp(trade_date),
    }


@pytest.fixture
def mixed_labels():
    rows = []
    # 150 pre-holdout positives.
    for i in range(150):
        rows.append(_labels_row(f"2020-01-{(i % 28) + 1:02d}", f"PRE{i}", 1))
    # 50 pre-holdout negatives (must never be sampled).
    for i in range(50):
        rows.append(_labels_row(f"2020-01-{(i % 28) + 1:02d}", f"NEG{i}", 0))
    # 30 holdout (2023+) positives -- must never be sampled when pre_holdout_only=True.
    for i in range(30):
        rows.append(_labels_row(f"2023-02-{(i % 28) + 1:02d}", f"HOLD{i}", 1))
    return pd.DataFrame(rows)


# --- sample_positives ------------------------------------------------------


def test_sample_positives_never_samples_2023_plus(mixed_labels):
    sampled = sample_positives(mixed_labels, n=200, seed=0)
    assert (pd.to_datetime(sampled["trade_date"]) < HOLDOUT_START).all()


def test_sample_positives_never_samples_negatives(mixed_labels):
    sampled = sample_positives(mixed_labels, n=200, seed=0)
    assert (sampled["label"] == 1).all()


def test_sample_positives_is_reproducible(mixed_labels):
    a = sample_positives(mixed_labels, n=50, seed=42)
    b = sample_positives(mixed_labels, n=50, seed=42)
    assert list(a["ticker"]) == list(b["ticker"])


def test_sample_positives_caps_at_available_count(mixed_labels):
    sampled = sample_positives(mixed_labels, n=10_000, seed=0)
    assert len(sampled) == 150  # only 150 pre-holdout positives exist


def test_sample_positives_can_include_holdout_with_token(mixed_labels):
    sampled = sample_positives(mixed_labels, n=10_000, seed=0, include_holdout=True, unseal_token="PREREG_FROZEN")
    assert len(sampled) == 180  # 150 pre-holdout + 30 holdout positives


# --- Defect 3: sample_positives(include_holdout=True) is a holdout-seal
# chokepoint -- there used to be a plain `pre_holdout_only: bool` flag any
# caller could flip with no gate at all. -------------------------------


def test_sample_positives_include_holdout_without_token_raises(mixed_labels):
    with pytest.raises(LeakageError):
        sample_positives(mixed_labels, n=10_000, seed=0, include_holdout=True)


def test_sample_positives_include_holdout_wrong_token_raises(mixed_labels):
    with pytest.raises(LeakageError):
        sample_positives(mixed_labels, n=10_000, seed=0, include_holdout=True, unseal_token="not-the-token")


# --- CLASSIFICATION_RUBRIC ------------------------------------------------


def test_rubric_names_all_three_categories():
    assert "scheduled_catalyst" in CLASSIFICATION_RUBRIC
    assert "carryover" in CLASSIFICATION_RUBRIC
    assert "unscheduled" in CLASSIFICATION_RUBRIC


# --- classify ------------------------------------------------------------


def test_classify_appends_category_column(mixed_labels):
    samples = sample_positives(mixed_labels, n=10, seed=0)
    classified = classify(samples, lambda row: "unscheduled")
    assert (classified["category"] == "unscheduled").all()
    assert len(classified) == len(samples)


def test_classify_rejects_invalid_category(mixed_labels):
    samples = sample_positives(mixed_labels, n=5, seed=0)
    with pytest.raises(ValueError):
        classify(samples, lambda row: "not_a_real_category")


# --- estimate_ceiling / Wilson interval --------------------------------------


def test_estimate_ceiling_point_estimate():
    classified = pd.DataFrame({"category": ["unscheduled"] * 40 + ["carryover"] * 40 + ["scheduled_catalyst"] * 20})
    estimate = estimate_ceiling(classified)
    assert estimate.n == 100
    assert estimate.unscheduled_count == 40
    assert estimate.unscheduled_share == pytest.approx(0.40)


def test_estimate_ceiling_wilson_interval_matches_hand_computation():
    """n=20, successes=10 -> phat=0.5. Hand-computed 95% Wilson interval
    (z=1.959963984540054):

    denom = 1 + z^2/n = 1 + 3.84150.../20 = 1.1920751...
    center = phat + z^2/(2n) = 0.5 + 3.8415/(40) = 0.5960188...
    margin = z * sqrt(phat*(1-phat)/n + z^2/(4n^2))
           = 1.959963984540054 * sqrt(0.25/20 + 3.8415/1600)
           = 1.959963984540054 * sqrt(0.0125 + 0.00240094...)
           = 1.959963984540054 * 0.121867...
           ~= 0.238857...
    low = (center - margin) / denom, high = (center + margin) / denom
    """
    z = 1.959963984540054
    n, successes = 20, 10
    phat = successes / n
    denom = 1 + (z**2) / n
    center = phat + (z**2) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2) / (4 * n**2))
    expected_low = (center - margin) / denom
    expected_high = (center + margin) / denom

    classified = pd.DataFrame({"category": ["unscheduled"] * successes + ["carryover"] * (n - successes)})
    estimate = estimate_ceiling(classified)

    assert estimate.ci_low == pytest.approx(expected_low, abs=1e-6)
    assert estimate.ci_high == pytest.approx(expected_high, abs=1e-6)
    assert 0.0 <= estimate.ci_low <= estimate.unscheduled_share <= estimate.ci_high <= 1.0


def test_estimate_ceiling_requires_category_column():
    with pytest.raises(ValueError):
        estimate_ceiling(pd.DataFrame({"trade_date": [1, 2, 3]}))


def test_estimate_ceiling_n200_narrow_enough_to_be_informative():
    classified = pd.DataFrame({"category": ["unscheduled"] * 60 + ["carryover"] * 100 + ["scheduled_catalyst"] * 40})
    estimate = estimate_ceiling(classified)
    assert (estimate.ci_high - estimate.ci_low) < 0.20


# --- render_ceiling_md ------------------------------------------------------


def test_render_ceiling_md_returns_string_without_writing_to_docs():
    classified = pd.DataFrame({"category": ["unscheduled"] * 40 + ["carryover"] * 40 + ["scheduled_catalyst"] * 20})
    estimate = estimate_ceiling(classified)
    markdown = render_ceiling_md(estimate)
    assert isinstance(markdown, str)
    assert "40" in markdown
    assert "Ceiling estimate" in markdown
