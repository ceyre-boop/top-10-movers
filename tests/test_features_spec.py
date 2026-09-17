from __future__ import annotations

import pandas as pd
import pytest

from top10.features.bars_t1 import BARS_T1_FEATURES
from top10.features.spec import (
    FEATURE_SPEC_VERSION,
    T1B_COLUMNS,
    T1B_SPEC,
    T1B_TFM_COLUMNS,
    T1B_TFM_SPEC,
    T1_SPEC,
    T2_SPEC,
    FeatureSpec,
    write_features,
)
from top10.storage import LeakageError


@pytest.fixture()
def isolated_data_dirs(tmp_path, monkeypatch):
    import top10.features.spec as spec_mod

    monkeypatch.setattr(spec_mod, "DATA_FEATURES", tmp_path / "features")
    return tmp_path


# --- FEATURE_SPEC_VERSION bump ------------------------------------------------


def test_feature_spec_version_is_2():
    assert FEATURE_SPEC_VERSION == "2"


def test_version_bump_gives_t1_and_t2_fresh_hashes():
    # Sanity: T1/T2 specs pick up the bumped version too -- old feature
    # caches under the "1" hash are invalidated by design.
    assert T1_SPEC.version == "2"
    assert T2_SPEC.version == "2"


# --- T1B / T1B_TFM column contracts -------------------------------------------


def test_t1b_columns_are_trade_date_ticker_15_features_as_of():
    assert T1B_COLUMNS[0] == "trade_date"
    assert T1B_COLUMNS[1] == "ticker"
    assert T1B_COLUMNS[-1] == "as_of"
    assert T1B_COLUMNS[2:-1] == BARS_T1_FEATURES
    assert len(T1B_COLUMNS) == 2 + 15 + 1


def test_t1b_tfm_columns_are_t1b_plus_7_tfm_features_before_as_of():
    expected_extra = (
        "tfm_q50",
        "tfm_q90",
        "tfm_spread",
        "tfm_skew_norm",
        "tfm_upside_ratio",
        "tfm_q90_xs_rank",
        "tfm_skew_xs_rank",
    )
    assert T1B_TFM_COLUMNS[:-1] == T1B_COLUMNS[:-1] + expected_extra
    assert T1B_TFM_COLUMNS[-1] == "as_of"


def test_t1b_spec_task_and_hash():
    assert T1B_SPEC.task == "T1B"
    assert T1B_SPEC.columns == T1B_COLUMNS
    assert isinstance(T1B_SPEC.spec_hash, str) and T1B_SPEC.spec_hash


def test_t1b_tfm_spec_task_and_hash():
    assert T1B_TFM_SPEC.task == "T1B_TFM"
    assert T1B_TFM_SPEC.columns == T1B_TFM_COLUMNS
    assert T1B_TFM_SPEC.spec_hash != T1B_SPEC.spec_hash


# --- write_features: T1B/T1B_TFM are registered on the decision-time ladder --


def test_write_features_writes_t1b_at_t1_decision_time(isolated_data_dirs):
    from top10.features.t1 import decision_time_t1

    trade_date = pd.Timestamp("2024-01-10")
    safe_as_of = decision_time_t1(trade_date)
    df = pd.DataFrame(
        [{**{c: 0.0 for c in T1B_SPEC.columns}, "trade_date": trade_date, "ticker": "AAA", "as_of": safe_as_of}]
    )[list(T1B_SPEC.columns)]

    path = write_features(df, T1B_SPEC, trade_date)
    assert path.exists()


def test_write_features_refuses_leaked_t1b_frame(isolated_data_dirs):
    from top10.features.t1 import decision_time_t1

    trade_date = pd.Timestamp("2024-01-10")
    leaked_as_of = decision_time_t1(trade_date) + pd.Timedelta(hours=1)
    df = pd.DataFrame(
        [{**{c: 0.0 for c in T1B_SPEC.columns}, "trade_date": trade_date, "ticker": "AAA", "as_of": leaked_as_of}]
    )[list(T1B_SPEC.columns)]

    with pytest.raises(LeakageError):
        write_features(df, T1B_SPEC, trade_date)


def test_write_features_refuses_leaked_t1b_tfm_frame(isolated_data_dirs):
    from top10.features.t1 import decision_time_t1

    trade_date = pd.Timestamp("2024-01-10")
    leaked_as_of = decision_time_t1(trade_date) + pd.Timedelta(hours=1)
    df = pd.DataFrame(
        [{**{c: 0.0 for c in T1B_TFM_SPEC.columns}, "trade_date": trade_date, "ticker": "AAA", "as_of": leaked_as_of}]
    )[list(T1B_TFM_SPEC.columns)]

    with pytest.raises(LeakageError):
        write_features(df, T1B_TFM_SPEC, trade_date)


# --- the latent defect: an unregistered task must raise, not write unguarded --


def test_write_features_raises_on_unregistered_task(isolated_data_dirs):
    """TOP FINDING (T1B/T1B_TFM registration): the task dispatch previously
    fell through to `decision_time = None` for an unknown `spec.task`, and
    the leakage guard below only ran when `decision_time is not None` -- so
    registering ANY new task name silently disabled `assert_decision_time_safe`
    on write. This must now raise instead of writing an unguarded frame."""
    trade_date = pd.Timestamp("2024-01-10")
    bogus_spec = FeatureSpec(task="BOGUS_TASK", columns=("trade_date", "ticker", "as_of"))
    df = pd.DataFrame(
        [{"trade_date": trade_date, "ticker": "AAA", "as_of": trade_date + pd.Timedelta(days=999)}]
    )

    with pytest.raises(ValueError):
        write_features(df, bogus_spec, trade_date)

    # And nothing was written to disk for the bogus spec.
    from top10.features.spec import feature_output_path

    assert not feature_output_path(bogus_spec, trade_date).exists()
