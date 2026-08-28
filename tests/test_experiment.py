"""Tests for top10/experiment.py."""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
import pytest

from top10.experiment import (
    HOLDOUT_START,
    assert_frame_holdout_sealed,
    assert_holdout_sealed,
    count_corrected_variants,
    log_experiment,
)
from top10.storage import LeakageError


# --- assert_holdout_sealed ---------------------------------------------------


def test_assert_holdout_sealed_raises_without_token():
    dates = pd.to_datetime(["2022-01-01", "2023-06-01"])
    with pytest.raises(LeakageError):
        assert_holdout_sealed(dates)


def test_assert_holdout_sealed_passes_with_correct_token():
    dates = pd.to_datetime(["2022-01-01", "2023-06-01"])
    assert_holdout_sealed(dates, unseal_token="PREREG_FROZEN")  # must not raise


def test_assert_holdout_sealed_wrong_token_still_raises():
    dates = pd.to_datetime(["2023-06-01"])
    with pytest.raises(LeakageError):
        assert_holdout_sealed(dates, unseal_token="wrong-token")


def test_assert_holdout_sealed_pre_holdout_dates_never_raise():
    dates = pd.to_datetime(["2015-01-01", "2022-12-31"])
    assert_holdout_sealed(dates)  # entirely pre-holdout -- no token needed


def test_holdout_start_constant():
    assert HOLDOUT_START == pd.Timestamp("2023-01-01")


# --- assert_frame_holdout_sealed (the reusable per-entry-point guard) --------


def test_assert_frame_holdout_sealed_raises_without_token():
    df = pd.DataFrame({"trade_date": pd.to_datetime(["2022-01-01", "2023-06-01"])})
    with pytest.raises(LeakageError):
        assert_frame_holdout_sealed(df)


def test_assert_frame_holdout_sealed_passes_with_token():
    df = pd.DataFrame({"trade_date": pd.to_datetime(["2023-06-01"])})
    assert_frame_holdout_sealed(df, unseal_token="PREREG_FROZEN")  # must not raise


def test_assert_frame_holdout_sealed_noop_on_empty_or_missing_column():
    assert_frame_holdout_sealed(pd.DataFrame())  # empty -- no-op
    assert_frame_holdout_sealed(pd.DataFrame({"other_col": [1, 2]}))  # no date_col -- no-op


# --- Defect 3: cli.py's ingest/labels/features routes are holdout-seal
# chokepoints (`python -m top10.cli features --start 2023-01-01 --end
# 2025-12-31` previously built holdout features with no token at all). ------


def test_cli_ingest_route_raises_on_holdout_without_token(monkeypatch):
    from top10 import cli as cli_mod

    monkeypatch.setattr("top10.pipeline.ingest", lambda vendor, start, end: {})
    args = argparse.Namespace(
        vendor="polygon", start=dt.date(2023, 6, 1), end=dt.date(2023, 6, 30), unseal_token=None
    )
    with pytest.raises(LeakageError):
        cli_mod._cmd_ingest(args)


def test_cli_ingest_route_succeeds_on_holdout_with_token(monkeypatch):
    from top10 import cli as cli_mod

    monkeypatch.setattr("top10.pipeline.ingest", lambda vendor, start, end: {})
    args = argparse.Namespace(
        vendor="polygon", start=dt.date(2023, 6, 1), end=dt.date(2023, 6, 30), unseal_token="PREREG_FROZEN"
    )
    assert cli_mod._cmd_ingest(args) == 0


def test_cli_labels_route_raises_on_holdout_without_token(monkeypatch):
    from top10 import cli as cli_mod

    monkeypatch.setattr("top10.pipeline.ingest", lambda vendor, start, end: {})
    monkeypatch.setattr("top10.pipeline.build_labels_step", lambda frames, start, end: pd.DataFrame())
    args = argparse.Namespace(
        vendor="polygon", start=dt.date(2023, 6, 1), end=dt.date(2023, 6, 30), unseal_token=None
    )
    with pytest.raises(LeakageError):
        cli_mod._cmd_labels(args)


def test_cli_labels_route_succeeds_on_holdout_with_token(monkeypatch):
    from top10 import cli as cli_mod

    monkeypatch.setattr("top10.pipeline.ingest", lambda vendor, start, end: {})
    monkeypatch.setattr("top10.pipeline.build_labels_step", lambda frames, start, end: pd.DataFrame())
    args = argparse.Namespace(
        vendor="polygon", start=dt.date(2023, 6, 1), end=dt.date(2023, 6, 30), unseal_token="PREREG_FROZEN"
    )
    assert cli_mod._cmd_labels(args) == 0


def test_cli_features_route_raises_on_holdout_without_token(monkeypatch):
    from top10 import cli as cli_mod

    monkeypatch.setattr("top10.pipeline.ingest", lambda vendor, start, end: {})
    monkeypatch.setattr("top10.pipeline.build_features_step", lambda task, start, end, **kw: pd.DataFrame())
    args = argparse.Namespace(
        vendor="polygon", task="T1", start=dt.date(2023, 6, 1), end=dt.date(2023, 6, 30), unseal_token=None
    )
    with pytest.raises(LeakageError):
        cli_mod._cmd_features(args)


def test_cli_features_route_succeeds_on_holdout_with_token(monkeypatch):
    from top10 import cli as cli_mod

    monkeypatch.setattr("top10.pipeline.ingest", lambda vendor, start, end: {})
    monkeypatch.setattr("top10.pipeline.build_features_step", lambda task, start, end, **kw: pd.DataFrame())
    args = argparse.Namespace(
        vendor="polygon", task="T1", start=dt.date(2023, 6, 1), end=dt.date(2023, 6, 30), unseal_token="PREREG_FROZEN"
    )
    assert cli_mod._cmd_features(args) == 0


# --- Defect 3: Top10Ranker.fit is a holdout-seal chokepoint -----------------
#
# `predict()` / `rank_top_k()` are deliberately NOT guarded the same way --
# see the rationale in `top10.model.Top10Ranker.fit`'s docstring: they are
# also the live production inference path
# (`top10.predict_live.predict_for_date`), which runs forever on dates
# >= HOLDOUT_START once the plan is frozen, so gating them would misapply a
# one-time-read seal to routine, already-authorized production use.


def test_top10ranker_fit_raises_on_holdout_without_token():
    from top10.model import Top10Ranker

    features = pd.DataFrame({"trade_date": pd.to_datetime(["2023-06-01"]), "ticker": ["A"], "f1": [1.0]})
    labels = pd.DataFrame({"trade_date": pd.to_datetime(["2023-06-01"]), "ticker": ["A"], "label": [1]})
    ranker = Top10Ranker(objective="binary")
    with pytest.raises(LeakageError):
        ranker.fit(features, labels)


def test_top10ranker_fit_passes_holdout_guard_with_token():
    from top10.model import Top10Ranker

    features = pd.DataFrame({"trade_date": pd.to_datetime(["2023-06-01"]), "ticker": ["A"], "f1": [1.0]})
    labels = pd.DataFrame({"trade_date": pd.to_datetime(["2023-06-01"]), "ticker": ["A"], "label": [1]})
    ranker = Top10Ranker(objective="binary")
    try:
        ranker.fit(features, labels, unseal_token="PREREG_FROZEN")
    except ImportError as exc:
        # Expected in this offline test environment (no lightgbm install) --
        # what matters is that it is NOT the holdout guard that fired.
        assert "lightgbm" in str(exc).lower()
    except LeakageError:
        pytest.fail("Top10Ranker.fit raised LeakageError even with a valid unseal_token")


# --- log_experiment / count_corrected_variants -------------------------------


def _base_kwargs(experiments_dir, counts=True, p_value=0.02):
    per_year = pd.DataFrame(
        [{"year": 2021, "precision_at_k": 0.5, "map_at_k": 0.4, "mean_hits": 5.0, "median_hits": 5.0, "n_days": 250}]
    )
    vs_baseline = {
        "mean_hits_delta": 1.2,
        "years_won": 5,
        "years_total": 7,
        "paired_t_test": {"statistic": 2.1, "p_value": p_value, "n": 250},
        "wilcoxon_test": {"statistic": 100.0, "p_value": p_value, "n": 250},
    }
    return dict(
        author="test-author",
        hypothesis="Testing log_experiment.",
        feature_spec_hash="feat-hash-abc",
        label_spec_hash="label-hash-xyz",
        task="T1",
        model_family="LightGBM binary",
        hyperparameters={"num_leaves": 31},
        feature_set="T1_SPEC",
        train_window=("2015-01-01", "2020-12-31"),
        validation_window=("2021-01-01", "2022-12-31"),
        per_year=per_year,
        vs_baseline=vs_baseline,
        counts_toward_family_wise_correction=counts,
        p_value=p_value,
        experiments_dir=experiments_dir,
    )


def test_log_experiment_creates_exp_001_when_empty(tmp_path):
    path = log_experiment(**_base_kwargs(tmp_path))
    assert path.name == "EXP-001.md"
    assert path.exists()
    text = path.read_text()
    assert "EXP-001" in text
    assert "feat-hash-abc" in text
    assert "label-hash-xyz" in text
    assert "| 2021 | 0.5000 | 0.4000 | 5.00 | 5.00 | 250 |" in text


def test_log_experiment_increments_ids(tmp_path):
    p1 = log_experiment(**_base_kwargs(tmp_path))
    p2 = log_experiment(**_base_kwargs(tmp_path))
    p3 = log_experiment(**_base_kwargs(tmp_path))
    assert [p.name for p in (p1, p2, p3)] == ["EXP-001.md", "EXP-002.md", "EXP-003.md"]


def test_log_experiment_increments_past_gaps(tmp_path):
    # Pre-seed EXP-001 and EXP-005 (simulating some manually-deleted ids) --
    # next id must be 006, not 002.
    (tmp_path / "EXP-001.md").write_text("placeholder")
    (tmp_path / "EXP-005.md").write_text("placeholder")
    path = log_experiment(**_base_kwargs(tmp_path))
    assert path.name == "EXP-006.md"


def test_count_corrected_variants_counts_only_flagged(tmp_path):
    log_experiment(**_base_kwargs(tmp_path, counts=True))
    log_experiment(**_base_kwargs(tmp_path, counts=False))
    log_experiment(**_base_kwargs(tmp_path, counts=True))

    assert count_corrected_variants(tmp_path) == 2


def test_count_corrected_variants_empty_dir(tmp_path):
    assert count_corrected_variants(tmp_path / "does-not-exist") == 0


def test_count_corrected_variants_ignores_non_exp_files(tmp_path):
    (tmp_path / "README.md").write_text("**Counts toward family-wise correction? (y/n)**: y")
    assert count_corrected_variants(tmp_path) == 0
