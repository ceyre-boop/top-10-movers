"""Tests for top10/model.py.

Deliberately offline / synthetic and require NO lightgbm install: every
case here exercises a guard rail that fires BEFORE lightgbm is touched
(`tune`'s holdout check, `load`'s feature-spec-hash check).

`top10.features.spec.FeatureSpec` is used where convenient, but the model
module only actually depends on `feature_spec.spec_hash` (duck typing), so
a couple of tests use a tiny local stand-in instead of importing the
concurrently-developed `top10.features.spec` module, to stay decoupled
from its in-flux state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import pytest

from top10 import model as model_mod
from top10.model import HOLDOUT_START, PARAM_GRID, Top10Ranker


@dataclass(frozen=True)
class _FakeSpec:
    """Duck-types `top10.features.spec.FeatureSpec` -- only `.spec_hash` is
    used by `top10.model`."""

    spec_hash: str


def test_invalid_objective_raises():
    with pytest.raises(ValueError):
        Top10Ranker(objective="not_a_real_objective")


def test_tune_raises_when_train_window_overlaps_holdout():
    train_features = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2022-06-01", "2023-01-05"]), "ticker": ["A", "B"], "f1": [1.0, 2.0]}
    )
    val_features = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2021-06-01"]), "ticker": ["A"], "f1": [1.0]}
    )
    train_labels = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2022-06-01", "2023-01-05"]), "ticker": ["A", "B"], "label": [0, 1]}
    )
    val_labels = pd.DataFrame({"trade_date": pd.to_datetime(["2021-06-01"]), "ticker": ["A"], "label": [0]})

    ranker = Top10Ranker(objective="binary")
    with pytest.raises(ValueError, match="holdout"):
        ranker.tune(train_features, train_labels, val_features, val_labels)


def test_tune_raises_when_validation_window_overlaps_holdout():
    train_features = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2019-06-01", "2020-01-05"]), "ticker": ["A", "B"], "f1": [1.0, 2.0]}
    )
    train_labels = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2019-06-01", "2020-01-05"]), "ticker": ["A", "B"], "label": [0, 1]}
    )
    val_features = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2023-03-01"]), "ticker": ["A"], "f1": [1.0]}
    )
    val_labels = pd.DataFrame({"trade_date": pd.to_datetime(["2023-03-01"]), "ticker": ["A"], "label": [0]})

    ranker = Top10Ranker(objective="binary")
    with pytest.raises(ValueError, match="holdout"):
        ranker.tune(train_features, train_labels, val_features, val_labels)


def test_tune_within_prereg_windows_does_not_raise_holdout_error():
    # 2015-2020 train / 2021-2022 validation is explicitly in-bounds per
    # plan §5.1 -- this should get past the holdout guard (it may still
    # raise ImportError if lightgbm isn't installed, which is fine and
    # expected in this offline test environment; we only assert it's NOT
    # the holdout guard that fires).
    train_features = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2015-06-01", "2020-12-01"]), "ticker": ["A", "B"], "f1": [1.0, 2.0]}
    )
    train_labels = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2015-06-01", "2020-12-01"]), "ticker": ["A", "B"], "label": [0, 1]}
    )
    val_features = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2021-06-01", "2022-06-01"]), "ticker": ["A", "B"], "f1": [1.0, 2.0]}
    )
    val_labels = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2021-06-01", "2022-06-01"]), "ticker": ["A", "B"], "label": [0, 1]}
    )

    ranker = Top10Ranker(objective="binary")
    try:
        ranker.tune(train_features, train_labels, val_features, val_labels, param_grid={"num_leaves": [7]})
    except ImportError as exc:
        assert "lightgbm" in str(exc).lower()
    except ValueError as exc:
        pytest.fail(f"tune() raised ValueError inside the allowed PREREG window: {exc}")


def test_load_refuses_prediction_on_mismatched_feature_spec_hash(tmp_path):
    model_dir = tmp_path / "saved_model"
    model_dir.mkdir()
    meta = {
        "objective": "binary",
        "params": {"objective": "binary"},
        "feature_spec_hash": "hash-trained-on",
        "label_spec_hash": "some-label-hash",
        "feature_columns": ["f1", "f2"],
        "use_focal_loss": False,
    }
    (model_dir / "meta.json").write_text(json.dumps(meta))
    # Deliberately NOT writing model.txt -- the hash mismatch must be
    # detected and raised before lightgbm.Booster(model_file=...) is ever
    # attempted, so this test needs no lightgbm install and no real booster.

    mismatched_spec = _FakeSpec(spec_hash="hash-different")
    with pytest.raises(ValueError, match="feature spec hash mismatch"):
        Top10Ranker.load(model_dir, feature_spec=mismatched_spec)


def test_load_missing_meta_raises_file_not_found(tmp_path):
    empty_dir = tmp_path / "nope"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        Top10Ranker.load(empty_dir)


def test_predict_refuses_mismatched_feature_spec_hash_without_touching_booster():
    ranker = Top10Ranker(objective="binary")
    ranker.feature_spec_hash = "trained-hash"
    # No booster set at all -- if the spec check didn't fire first, this
    # would raise a RuntimeError about the missing booster instead.
    mismatched_spec = _FakeSpec(spec_hash="other-hash")
    features = pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-01"]), "ticker": ["A"], "f1": [1.0]})

    with pytest.raises(ValueError, match="feature spec hash mismatch"):
        ranker.predict(features, feature_spec=mismatched_spec)


def test_predict_before_fit_raises_runtime_error():
    ranker = Top10Ranker(objective="binary")
    features = pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-01"]), "ticker": ["A"], "f1": [1.0]})
    with pytest.raises(RuntimeError):
        ranker.predict(features)


def test_holdout_start_matches_prereg():
    assert HOLDOUT_START == pd.Timestamp("2023-01-01")


def test_param_grid_is_nonempty_and_modest():
    assert isinstance(PARAM_GRID, dict)
    assert len(PARAM_GRID) > 0
    total_combos = 1
    for values in PARAM_GRID.values():
        total_combos *= len(values)
    assert total_combos <= 50  # "modest" per spec -- not an exhaustive search
