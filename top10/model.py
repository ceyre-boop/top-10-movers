"""Top10Ranker — the TOP10 model wrapper (docs/PREREG_TOP10.md "Model family").

Wraps LightGBM behind a small, spec-aware interface. Two first-class
objective variants are supported, per the plan's note that this is
fundamentally a ranking problem:

- ``"binary"``: binary classification, imbalance-corrected via
  ``scale_pos_weight`` (auto-computed from the fit data if not given) or,
  optionally, a focal-loss custom objective.
- ``"lambdarank"``: LightGBM's LambdaRank objective, grouped by
  ``trade_date`` (each trading day is one ranking group).

``lightgbm`` is imported LAZILY inside methods so this module -- and every
test that only exercises the pre-fit guard rails (`tune`'s holdout check,
`load`'s feature-spec-hash check) -- works without lightgbm installed.
Install it via ``pip install -e '.[model]'``.

Persistence (`save`/`load`) stores the trained booster alongside the
feature-spec hash and label-spec hash it was trained against. `load`
refuses to let a caller predict with a mismatched feature spec -- silent
feature drift between train and predict time is exactly the kind of bug
that manufactures fake edge (P12).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from top10.experiment import assert_frame_holdout_sealed

_LIGHTGBM_INSTALL_HINT = (
    "lightgbm is required for this operation. Install it with: "
    "pip install -e '.[model]'"
)

_ID_COLUMNS = ("trade_date", "ticker", "as_of")

_VALID_OBJECTIVES = ("binary", "lambdarank")

# Modest hyperparameter grid for `tune()`. Kept small deliberately -- every
# grid point tuned is a variant that must be logged and Holm-corrected
# (docs/PREREG_TOP10.md P12 / experiments/TEMPLATE.md).
PARAM_GRID: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_data_in_leaf": [10, 20, 50],
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "binary": {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    },
    "lambdarank": {
        "objective": "lambdarank",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "eval_at": [10],
        "verbosity": -1,
    },
}

# Plan §6 / P12: the holdout is 2023-01-01 onward and is sealed. Tuning
# (plan §5.1) is scoped to 2015-2020 train / 2021-2022 validation ONLY --
# there is no unseal path here at all, unlike walkforward/experiment's
# gated `assert_holdout_sealed`. Tuning must never touch holdout, full stop.
HOLDOUT_START = pd.Timestamp("2023-01-01")


def _focal_loss_binary(gamma: float = 2.0, alpha: float = 0.25):
    """Return (grad, hess) objective + eval functions for LightGBM focal
    loss, as an alternative to `scale_pos_weight` for class imbalance.
    """

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def focal_loss_objective(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        y = train_data.get_label()
        p = _sigmoid(preds)
        eps = 1e-9
        p = np.clip(p, eps, 1 - eps)

        pt = np.where(y == 1, p, 1 - p)
        alpha_t = np.where(y == 1, alpha, 1 - alpha)

        # Numerically-approximated gradient/hessian of the focal loss w.r.t.
        # the raw (pre-sigmoid) score, via finite-ish analytic derivatives
        # of -alpha_t * (1 - pt)^gamma * log(pt).
        grad = alpha_t * (1 - pt) ** (gamma - 1) * (gamma * pt * np.log(pt) + pt - 1)
        grad = np.where(y == 1, -grad, grad)
        hess = alpha_t * (1 - pt) ** gamma * p * (1 - p) * gamma  # positive-definite approximation
        hess = np.maximum(hess, eps)
        return grad, hess

    return focal_loss_objective


class Top10Ranker:
    """LightGBM-backed ranker for the TOP10 daily top-movers task."""

    def __init__(
        self,
        objective: str = "binary",
        params: dict[str, Any] | None = None,
        *,
        feature_spec: Any | None = None,
        label_spec_hash: str | None = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        if objective not in _VALID_OBJECTIVES:
            raise ValueError(
                f"Top10Ranker: objective must be one of {_VALID_OBJECTIVES}, got {objective!r}"
            )
        self.objective = objective
        self.params: dict[str, Any] = {**DEFAULT_PARAMS[objective], **(params or {})}
        self.feature_spec = feature_spec
        self.feature_spec_hash: str | None = (
            feature_spec.spec_hash if feature_spec is not None else None
        )
        self.label_spec_hash = label_spec_hash
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        self._booster = None
        self._feature_columns: list[str] | None = None

    # -- fit / predict --------------------------------------------------

    def _feature_columns_from(self, features: pd.DataFrame) -> list[str]:
        return [c for c in features.columns if c not in _ID_COLUMNS]

    def fit(
        self, features: pd.DataFrame, labels: pd.DataFrame, *, unseal_token: str | None = None
    ) -> "Top10Ranker":
        # Training/refitting is the leakage-relevant act here (it is what
        # `tune()` already guards, unconditionally, for its fixed
        # 2015-2020/2021-2022 windows) -- a fit call reaching into
        # holdout-dated (>= 2023-01-01) rows would contaminate the sealed
        # walk-forward evaluation Plan §6 protects. `predict()` is
        # deliberately NOT guarded the same way: it is also the live
        # production inference path (`top10.predict_live.predict_for_date`),
        # which by design runs forever on dates >= HOLDOUT_START once the
        # plan is frozen -- gating every live prediction behind the
        # one-time `PREREG_FROZEN` token would misapply a one-time-read
        # seal to routine, already-authorized production use.
        assert_frame_holdout_sealed(features, unseal_token=unseal_token)
        assert_frame_holdout_sealed(labels, unseal_token=unseal_token)

        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - exercised only w/o lightgbm
            raise ImportError(_LIGHTGBM_INSTALL_HINT) from exc

        merged = features.merge(
            labels[["trade_date", "ticker", "label"]], on=["trade_date", "ticker"], how="inner"
        )
        if merged.empty:
            raise ValueError("Top10Ranker.fit: no overlapping (trade_date, ticker) rows between features and labels.")

        merged = merged.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
        feature_columns = self._feature_columns_from(features)
        self._feature_columns = feature_columns

        X = merged[feature_columns]
        y = merged["label"].astype(int).to_numpy()

        params = dict(self.params)
        fit_kwargs: dict[str, Any] = {}
        objective_override = None

        if self.objective == "binary":
            if self.use_focal_loss:
                objective_override = _focal_loss_binary(self.focal_gamma, self.focal_alpha)
                params.pop("objective", None)
            elif "scale_pos_weight" not in params:
                n_pos = int(y.sum())
                n_neg = int(len(y) - n_pos)
                params["scale_pos_weight"] = float(n_neg / n_pos) if n_pos > 0 else 1.0

            train_set = lgb.Dataset(X, label=y)
        else:  # lambdarank: group by trade_date, preserving row order within each group.
            group_sizes = merged.groupby("trade_date", sort=False).size().to_numpy()
            train_set = lgb.Dataset(X, label=y, group=group_sizes)

        if objective_override is not None:
            self._booster = lgb.train(params, train_set, fobj=objective_override)
        else:
            self._booster = lgb.train(params, train_set)

        # Track label spec hash from the data itself if the caller didn't
        # pin one explicitly (labels.build_labels persists it as
        # `label_spec_version`, per docs/LABEL_SPEC.md).
        if self.label_spec_hash is None and "label_spec_version" in labels.columns and not labels.empty:
            self.label_spec_hash = str(labels["label_spec_version"].iloc[0])

        return self

    def predict(self, features: pd.DataFrame, *, feature_spec: Any | None = None) -> pd.DataFrame:
        self._assert_feature_spec_matches(feature_spec)

        if self._booster is None:
            raise RuntimeError("Top10Ranker.predict: model has not been fit or loaded.")
        if self._feature_columns is None:
            raise RuntimeError("Top10Ranker.predict: no feature columns recorded on this model.")

        missing = [c for c in self._feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Top10Ranker.predict: features frame is missing trained-on column(s) {missing}.")

        X = features[self._feature_columns]
        scores = self._booster.predict(X)

        return pd.DataFrame(
            {
                "trade_date": features["trade_date"].to_numpy(),
                "ticker": features["ticker"].to_numpy(),
                "score": np.asarray(scores, dtype=float),
            }
        )

    def rank_top_k(self, features: pd.DataFrame, k: int = 10, *, feature_spec: Any | None = None) -> pd.DataFrame:
        """Predict and return only the top-`k` scored rows per `trade_date`,
        ranked by score descending with ticker-ascending tie-breaking (same
        convention as `top10.metrics` / `top10.baselines`)."""
        predictions = self.predict(features, feature_spec=feature_spec)
        if predictions.empty:
            return predictions

        ranked = predictions.sort_values(["trade_date", "score", "ticker"], ascending=[True, False, True])
        ranked["_pos"] = ranked.groupby("trade_date").cumcount()
        return ranked[ranked["_pos"] < k].drop(columns="_pos").reset_index(drop=True)

    def _assert_feature_spec_matches(self, feature_spec: Any | None) -> None:
        if feature_spec is None or self.feature_spec_hash is None:
            return
        if feature_spec.spec_hash != self.feature_spec_hash:
            raise ValueError(
                "Top10Ranker: feature spec hash mismatch. This model was trained "
                f"on feature spec hash={self.feature_spec_hash!r}, but was asked to "
                f"predict on features matching spec hash={feature_spec.spec_hash!r}. "
                "Refusing -- silent feature drift is a P12 route to fake edge."
            )

    # -- tune -------------------------------------------------------------

    def tune(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.DataFrame,
        val_features: pd.DataFrame,
        val_labels: pd.DataFrame,
        *,
        param_grid: dict[str, list[Any]] | None = None,
        k: int = 10,
    ) -> dict[str, Any]:
        """Grid-search `param_grid` (default `PARAM_GRID`), scoring each
        candidate by precision@k on `val_features`/`val_labels`.

        Plan §5.1: tuning happens on 2015-2020 train / 2021-2022 validation
        ONLY. Raises `ValueError` immediately -- before touching lightgbm at
        all -- if either window overlaps the sealed 2023-01-01+ holdout.
        Returns {"best_params", "best_score", "results": [...]}.
        """
        self._assert_window_excludes_holdout(train_features, "train_features")
        self._assert_window_excludes_holdout(train_labels, "train_labels")
        self._assert_window_excludes_holdout(val_features, "val_features")
        self._assert_window_excludes_holdout(val_labels, "val_labels")

        try:
            import lightgbm  # noqa: F401  -- import check only; deferred to fit()
        except ImportError as exc:  # pragma: no cover - exercised only w/o lightgbm
            raise ImportError(_LIGHTGBM_INSTALL_HINT) from exc

        from top10.metrics import precision_at_k

        grid = param_grid or PARAM_GRID
        keys = list(grid.keys())
        combos = _cartesian_product(grid, keys)

        results = []
        best_score = float("-inf")
        best_params: dict[str, Any] | None = None

        for combo in combos:
            candidate = Top10Ranker(
                objective=self.objective,
                params=combo,
                use_focal_loss=self.use_focal_loss,
                focal_gamma=self.focal_gamma,
                focal_alpha=self.focal_alpha,
            )
            candidate.fit(train_features, train_labels)
            preds = candidate.predict(val_features)
            score = precision_at_k(preds, val_labels, k=k)
            results.append({"params": combo, "precision_at_k": score})
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_params = combo

        return {"best_params": best_params, "best_score": best_score, "results": results}

    @staticmethod
    def _assert_window_excludes_holdout(frame: pd.DataFrame, name: str) -> None:
        # Checking only `.max()` IS the general check here, not an
        # approximation: the holdout is a one-sided, open-ended range
        # (`trade_date >= HOLDOUT_START`), and for a one-sided threshold
        # "the maximum of a column is >= the threshold" is logically
        # equivalent to "some row in that column is >= the threshold" --
        # regardless of row order or gaps in the frame. This would stop
        # being sufficient only for a two-sided (bounded) holdout window,
        # which is not what Plan §6 defines.
        if frame.empty or "trade_date" not in frame.columns:
            return
        max_date = pd.Timestamp(frame["trade_date"].max())
        if max_date >= HOLDOUT_START:
            raise ValueError(
                f"Top10Ranker.tune: {name} contains dates on/after the sealed holdout "
                f"start ({HOLDOUT_START.date()}). Plan §5.1 restricts tuning to "
                "2015-2020 train / 2021-2022 validation -- there is no unseal path "
                "for tune()."
            )

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        if self._booster is None:
            raise RuntimeError("Top10Ranker.save: model has not been fit; nothing to save.")

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        meta = {
            "objective": self.objective,
            "params": self.params,
            "feature_spec_hash": self.feature_spec_hash,
            "label_spec_hash": self.label_spec_hash,
            "feature_columns": self._feature_columns,
            "use_focal_loss": self.use_focal_loss,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        self._booster.save_model(str(directory / "model.txt"))
        return directory

    @classmethod
    def load(cls, path: str | Path, *, feature_spec: Any | None = None) -> "Top10Ranker":
        """Load a saved model. If `feature_spec` is given, its hash is
        checked against the hash the model was trained on BEFORE lightgbm
        is touched at all -- a mismatch raises immediately rather than
        silently loading a booster that will be fed drifted features.
        """
        directory = Path(path)
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Top10Ranker.load: no meta.json found at {directory}.")

        meta = json.loads(meta_path.read_text())
        stored_hash = meta.get("feature_spec_hash")

        if feature_spec is not None and stored_hash is not None and feature_spec.spec_hash != stored_hash:
            raise ValueError(
                "Top10Ranker.load: feature spec hash mismatch. This model was "
                f"trained on feature spec hash={stored_hash!r}, but was loaded "
                f"against a feature spec with hash={feature_spec.spec_hash!r}. "
                "Refusing to load for prediction -- silent feature drift is a "
                "P12 route to fake edge."
            )

        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - exercised only w/o lightgbm
            raise ImportError(_LIGHTGBM_INSTALL_HINT) from exc

        instance = cls(
            objective=meta["objective"],
            params=meta["params"],
            label_spec_hash=meta.get("label_spec_hash"),
            use_focal_loss=meta.get("use_focal_loss", False),
        )
        instance.feature_spec_hash = stored_hash
        instance._feature_columns = meta.get("feature_columns")
        instance._booster = lgb.Booster(model_file=str(directory / "model.txt"))
        return instance


def _cartesian_product(grid: dict[str, list[Any]], keys: list[str]) -> list[dict[str, Any]]:
    if not keys:
        return [{}]
    combos: list[dict[str, Any]] = [{}]
    for key in keys:
        new_combos = []
        for combo in combos:
            for value in grid[key]:
                new_combos.append({**combo, key: value})
        combos = new_combos
    return combos
