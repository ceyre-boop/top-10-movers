from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import top10.runner as runner_mod
from top10.runner import run_variant


def _synthetic_universe(n_tickers: int = 3, start: str = "2018-01-01", end: str = "2020-12-31") -> pd.DataFrame:
    """A small but multi-year synthetic `universe_liquidity.parquet`-shaped
    frame -- enough business days/tickers for `expanding_window_splits`
    (`min_train_years=2`, default) to produce a real 2020 test split, and
    enough label diversity for `Top10Ranker.fit`'s non-degenerate path."""
    dates = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(0)

    rows = []
    for ticker_idx in range(n_tickers):
        ticker = f"TICK{ticker_idx}"
        close = 10.0
        prev_close = None
        prev_date = None
        for i, trade_date in enumerate(dates):
            ret = float(rng.normal(0.0005, 0.02))
            new_close = close * (1.0 + ret)
            if prev_close is None:
                prev_close = close
                prev_date = trade_date - pd.Timedelta(days=1)
            volume = float(rng.uniform(500_000, 1_500_000))
            rows.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "close": new_close,
                    "prev_close": close,
                    "prev_date": prev_date,
                    "volume": volume,
                    "dollar_volume": new_close * volume,
                    "adv20": new_close * volume,
                    "ret": new_close / close - 1.0,
                    "label": int(rng.random() < 0.02),
                    "rank": i % 20 + 1,
                    "high": new_close,
                    "low": new_close,
                    "is_split_day": False,
                    "is_degraded": False,
                    "_active": True,
                }
            )
            prev_date = trade_date
            close = new_close

    return pd.DataFrame(rows)


@pytest.fixture()
def isolated_features_dir(tmp_path, monkeypatch):
    import top10.features.spec as spec_mod

    monkeypatch.setattr(spec_mod, "DATA_FEATURES", tmp_path / "features")
    return tmp_path


# --- module-level invariant: no unseal_token anywhere -----------------------


def test_module_never_declares_or_forwards_unseal_token():
    """No function IN THIS MODULE accepts or forwards `unseal_token` --
    the module docstring itself names and explains that guarantee, so the
    check here is over live code objects, not a source-text grep (which
    would also match the docstring's own explanation)."""
    for name, obj in vars(runner_mod).items():
        if not inspect.isfunction(obj) or obj.__module__ != runner_mod.__name__:
            continue
        assert "unseal_token" not in inspect.signature(obj).parameters, name


def test_run_variant_signature_has_no_unseal_token_param():
    sig = inspect.signature(run_variant)
    assert "unseal_token" not in sig.parameters


# --- validation --------------------------------------------------------------


def test_run_variant_rejects_unknown_variant():
    with pytest.raises(ValueError):
        run_variant("bogus")


def test_run_variant_refuses_holdout_dated_universe(tmp_path):
    universe = _synthetic_universe(n_tickers=2, start="2022-06-01", end="2023-03-01")
    path = tmp_path / "universe.parquet"
    universe.to_parquet(path)

    with pytest.raises(ValueError):
        run_variant("t1b", universe_path=path)


def test_run_variant_t1b_tfm_raises_clear_error_without_cache(tmp_path, isolated_features_dir):
    universe = _synthetic_universe(n_tickers=2, start="2018-01-01", end="2018-06-01")
    path = tmp_path / "universe.parquet"
    universe.to_parquet(path)

    with pytest.raises(FileNotFoundError, match="TimesFM"):
        run_variant("t1b_tfm", universe_path=path)


# --- end-to-end (small synthetic universe) -----------------------------------


def test_run_variant_t1b_end_to_end(tmp_path, isolated_features_dir):
    universe = _synthetic_universe(n_tickers=3, start="2018-01-01", end="2020-12-31")
    path = tmp_path / "universe.parquet"
    universe.to_parquet(path)

    result = run_variant("t1b", universe_path=path)

    assert result["variant"] == "t1b"
    assert isinstance(result["feature_spec_hash"], str) and result["feature_spec_hash"]
    assert result["n_rows"] > 0
    assert result["n_days"] > 0
    assert set(result["per_year"]["year"]) <= {2020, 2021, 2022}
    assert not result["per_year"].empty
    assert set(result["vs_baseline"].keys()) == {"B0", "B1", "B2"}
    for comparison in result["vs_baseline"].values():
        assert "mean_hits_delta" in comparison
        assert "per_year_record" in comparison

    # Features were actually persisted through write_features -- not a
    # scratch computation.
    features_dir = isolated_features_dir / "features" / result["feature_spec_hash"]
    assert features_dir.exists()
    assert any(features_dir.glob("*.parquet"))
