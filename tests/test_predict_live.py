"""Tests for top10/predict_live.py.

Offline, synthetic, no network, no lightgbm, no git remote -- git
operations run against a real throwaway repo created in tmp_path so the
pre-commitment behavior is exercised for real, without touching the
actual project repo.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from top10.features.spec import T2_SPEC
from top10.predict_live import (
    STOP_DRIFT_HITS,
    STOP_WINDOW_DAYS,
    KILL_WINDOW_DAYS,
    GitCommitError,
    _load_captured_rh,
    predict_for_date,
    rolling_monitor,
    score_prior_day,
)
from top10.storage import LeakageError, append_only_write


# --- fixtures ------------------------------------------------------------


def _t2_features_row(trade_date, ticker) -> dict:
    return {c: (trade_date if c == "trade_date" else (ticker if c == "ticker" else (trade_date if c == "as_of" else 0.0))) for c in T2_SPEC.columns}


@pytest.fixture
def t2_features():
    trade_date = pd.Timestamp("2024-03-14")
    rows = [_t2_features_row(trade_date, f"T{i}") for i in range(15)]
    return pd.DataFrame(rows)


class _FakeModel:
    """Duck-typed Top10Ranker stand-in: no lightgbm dependency."""

    def rank_top_k(self, features, k=10, feature_spec=None):
        ranked = features.copy()
        ranked["score"] = range(len(ranked), 0, -1)
        return ranked[["trade_date", "ticker", "score"]].head(k).reset_index(drop=True)

    @classmethod
    def load(cls, path, feature_spec=None):
        return cls()


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("test repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


# --- predict_for_date ------------------------------------------------------


def test_predict_for_date_writes_and_commits(t2_features, git_repo, monkeypatch):
    monkeypatch.setattr("top10.model.Top10Ranker", _FakeModel, raising=False)

    predictions_dir = git_repo / "data" / "predictions"
    out_path = predict_for_date(
        dt.date(2024, 3, 14),
        "unused-model-path",
        features=t2_features,
        predictions_dir=predictions_dir,
        repo_root=git_repo,
    )

    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["trade_date"] == "2024-03-14"
    assert len(payload["predictions"]) == 10

    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"], cwd=git_repo, capture_output=True, text=True, check=True
    )
    assert "2024-03-14" in log.stdout


def test_predict_for_date_refuses_to_overwrite(t2_features, git_repo, monkeypatch):
    monkeypatch.setattr("top10.model.Top10Ranker", _FakeModel, raising=False)

    predictions_dir = git_repo / "data" / "predictions"
    predict_for_date(
        dt.date(2024, 3, 14),
        "unused-model-path",
        features=t2_features,
        predictions_dir=predictions_dir,
        repo_root=git_repo,
    )

    with pytest.raises(FileExistsError):
        predict_for_date(
            dt.date(2024, 3, 14),
            "unused-model-path",
            features=t2_features,
            predictions_dir=predictions_dir,
            repo_root=git_repo,
        )


def test_predict_for_date_requires_features():
    with pytest.raises(ValueError):
        predict_for_date(dt.date(2024, 3, 14), "unused-model-path")


def test_predict_for_date_git_failure_is_loud(t2_features, tmp_path, monkeypatch):
    """A directory that is NOT a git repo -> `git add` fails -> must raise,
    never silently succeed."""
    monkeypatch.setattr("top10.model.Top10Ranker", _FakeModel, raising=False)

    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    predictions_dir = not_a_repo / "data" / "predictions"

    with pytest.raises(GitCommitError):
        predict_for_date(
            dt.date(2024, 3, 14),
            "unused-model-path",
            features=t2_features,
            predictions_dir=predictions_dir,
            repo_root=not_a_repo,
        )


# --- Defect 5: predict_for_date must assert decision-time safety ------------------


def test_predict_for_date_raises_on_leaked_feature_frame(git_repo, monkeypatch):
    """The one place a leak becomes a permanent, git-committed artifact
    previously had NO PIT assertion at all -- only a column-name/order
    check. A feature row whose `as_of` is after the T2 decision time must
    now raise before a prediction is ever written."""
    monkeypatch.setattr("top10.model.Top10Ranker", _FakeModel, raising=False)

    trade_date = pd.Timestamp("2024-03-14")
    from top10.features.t2 import decision_time_t2

    leaked_as_of = decision_time_t2(trade_date) + pd.Timedelta(hours=1)
    rows = []
    for i in range(15):
        row = _t2_features_row(trade_date, f"T{i}")
        row["as_of"] = leaked_as_of
        rows.append(row)
    leaked_features = pd.DataFrame(rows)

    predictions_dir = git_repo / "data" / "predictions"
    with pytest.raises(LeakageError):
        predict_for_date(
            dt.date(2024, 3, 14),
            "unused-model-path",
            features=leaked_features,
            predictions_dir=predictions_dir,
            repo_root=git_repo,
        )
    # And, crucially, nothing was ever written.
    assert not (predictions_dir / "2024-03-14.json").exists()


# --- Defect 5: _load_captured_rh reads the envelope, refuses S&P 500-only ---------


def test_load_captured_rh_returns_real_tickers(tmp_path, monkeypatch):
    import top10.config as config_mod

    monkeypatch.setattr(config_mod, "DATA_RAW", tmp_path)
    import top10.collect.rh_movers as rh_movers_mod

    monkeypatch.setattr(rh_movers_mod, "DATA_RAW", tmp_path)

    envelope = {
        "captured_at_utc": "2024-03-14T21:05:00+00:00",
        "captured_at_et": "2024-03-14T16:05:00-05:00",
        "trade_date": "2024-03-14",
        "sp500": {"_source": "robinhood_sp500", "_fetch_path": "https", "available": True, "payload": {}},
        "top_movers": {
            "_source": "robinhood_top_movers",
            "_fetch_path": "https",
            "available": True,
            "payload": {},
            "symbols": ["AAA", "BBB", "CCC"],
        },
        "top_movers_available": True,
    }
    out_dir = tmp_path / "rh_movers"
    out_dir.mkdir(parents=True)
    (out_dir / "2024-03-14.json").write_text(json.dumps(envelope))

    tickers = _load_captured_rh(pd.Timestamp("2024-03-14"))
    assert tickers == ["AAA", "BBB", "CCC"]


def test_load_captured_rh_refuses_sp500_only_capture(tmp_path, monkeypatch):
    import top10.config as config_mod

    monkeypatch.setattr(config_mod, "DATA_RAW", tmp_path)
    import top10.collect.rh_movers as rh_movers_mod

    monkeypatch.setattr(rh_movers_mod, "DATA_RAW", tmp_path)

    envelope = {
        "captured_at_utc": "2024-03-14T21:05:00+00:00",
        "captured_at_et": "2024-03-14T16:05:00-05:00",
        "trade_date": "2024-03-14",
        "sp500": {"_source": "robinhood_sp500", "_fetch_path": "https", "available": True, "payload": {}},
        "top_movers": {
            "_source": "robinhood_top_movers",
            "_fetch_path": None,
            "available": False,
            "payload": None,
            "symbols": None,
        },
        "top_movers_available": False,
    }
    out_dir = tmp_path / "rh_movers"
    out_dir.mkdir(parents=True)
    (out_dir / "2024-03-14.json").write_text(json.dumps(envelope))

    with pytest.raises(ValueError):
        _load_captured_rh(pd.Timestamp("2024-03-14"))


def test_load_captured_rh_raises_on_missing_capture(tmp_path, monkeypatch):
    import top10.config as config_mod

    monkeypatch.setattr(config_mod, "DATA_RAW", tmp_path)
    import top10.collect.rh_movers as rh_movers_mod

    monkeypatch.setattr(rh_movers_mod, "DATA_RAW", tmp_path)

    with pytest.raises(FileNotFoundError):
        _load_captured_rh(pd.Timestamp("2024-03-14"))


# --- score_prior_day ------------------------------------------------------


def test_score_prior_day_computes_hits(tmp_path):
    predictions_dir = tmp_path / "predictions"
    scores_dir = tmp_path / "scores"

    payload = {
        "trade_date": "2024-03-14",
        "predictions": [{"ticker": t, "score": 10.0 - i} for i, t in enumerate(["A", "B", "C", "D"])],
    }
    append_only_write(payload, predictions_dir / "2024-03-14.json")

    proxy_labels = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2024-03-14"), "ticker": "A", "label": 1},
            {"trade_date": pd.Timestamp("2024-03-14"), "ticker": "Z", "label": 1},
            {"trade_date": pd.Timestamp("2024-03-14"), "ticker": "B", "label": 0},
        ]
    )

    record = score_prior_day(
        dt.date(2024, 3, 14),
        predictions_dir=predictions_dir,
        captured_rh=["A", "B", "X"],
        proxy_labels=proxy_labels,
        scores_dir=scores_dir,
    )

    assert record["rh_hits"] == 2  # A, B
    assert record["proxy_hits"] == 1  # A
    assert (scores_dir / "2024-03-14.json").exists()


def test_score_prior_day_missing_prediction_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_prior_day(
            dt.date(2024, 3, 14),
            predictions_dir=tmp_path / "predictions",
            captured_rh=[],
            proxy_labels=pd.DataFrame(columns=["trade_date", "ticker", "label"]),
            scores_dir=tmp_path / "scores",
        )


# --- rolling_monitor: §7 stop rule -------------------------------------------


def test_rolling_monitor_stops_at_exactly_20_days_of_drift():
    holdout_expectation = 5.0
    # Every day strictly more than 1.5 hits below expectation.
    live_hits = [holdout_expectation - STOP_DRIFT_HITS - 0.1] * STOP_WINDOW_DAYS
    verdict = rolling_monitor(live_hits=live_hits, holdout_expectation=holdout_expectation)
    assert verdict.stop is True


def test_rolling_monitor_does_not_stop_at_19_days():
    holdout_expectation = 5.0
    live_hits = [holdout_expectation - STOP_DRIFT_HITS - 0.1] * (STOP_WINDOW_DAYS - 1)
    verdict = rolling_monitor(live_hits=live_hits, holdout_expectation=holdout_expectation)
    assert verdict.stop is False


def test_rolling_monitor_does_not_stop_at_exactly_threshold_drift():
    """Drift of exactly 1.5 (not > 1.5) must not trigger STOP."""
    holdout_expectation = 5.0
    live_hits = [holdout_expectation - STOP_DRIFT_HITS] * STOP_WINDOW_DAYS
    verdict = rolling_monitor(live_hits=live_hits, holdout_expectation=holdout_expectation)
    assert verdict.stop is False


def test_rolling_monitor_no_stop_when_only_some_days_drift():
    holdout_expectation = 5.0
    live_hits = [holdout_expectation - STOP_DRIFT_HITS - 0.1] * (STOP_WINDOW_DAYS - 1) + [holdout_expectation]
    verdict = rolling_monitor(live_hits=live_hits, holdout_expectation=holdout_expectation)
    assert verdict.stop is False


# --- rolling_monitor: §10 kill criterion -------------------------------------


def test_rolling_monitor_kills_at_30_consecutive_days_below_b4():
    live_hits = [2.0] * KILL_WINDOW_DAYS
    b4_hits = [3.0] * KILL_WINDOW_DAYS
    verdict = rolling_monitor(live_hits=live_hits, b4_hits=b4_hits)
    assert verdict.kill is True


def test_rolling_monitor_does_not_kill_at_29_days():
    live_hits = [2.0] * (KILL_WINDOW_DAYS - 1)
    b4_hits = [3.0] * (KILL_WINDOW_DAYS - 1)
    verdict = rolling_monitor(live_hits=live_hits, b4_hits=b4_hits)
    assert verdict.kill is False


def test_rolling_monitor_does_not_kill_when_one_day_ties_or_beats_b4():
    live_hits = [2.0] * (KILL_WINDOW_DAYS - 1) + [3.0]
    b4_hits = [3.0] * KILL_WINDOW_DAYS
    verdict = rolling_monitor(live_hits=live_hits, b4_hits=b4_hits)
    assert verdict.kill is False


def test_rolling_monitor_ok_verdict_when_nothing_triggers():
    live_hits = [5.0] * 30
    b4_hits = [3.0] * 30
    verdict = rolling_monitor(live_hits=live_hits, b4_hits=b4_hits, holdout_expectation=5.0)
    assert verdict.stop is False
    assert verdict.kill is False
    assert "OK" in str(verdict)
