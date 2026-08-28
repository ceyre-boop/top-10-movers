"""Feature spec registry — locks the column contract for T1/T2 features.

Per plan §2.4, feature frames are written under
`data/features/<feature-spec-hash>/`, and the spec hash is derived from
the task name, the ordered column list, and a version string (never the
raw column list alone -- bumping the version invalidates old caches even
if columns happen to be unchanged).

`validate_frame` is the pre-registration lock (P12): a frame's columns
must match a spec exactly -- same set, same order -- or it's rejected.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from top10.config import DATA_FEATURES
from top10.hashing import hash_spec
from top10.storage import spec_dir, write_parquet

FEATURE_SPEC_VERSION = "1"

# --- T1 (decision time: prior close, 16:00 ET on t-1) -----------------------

T1_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "ticker",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_1d_rank",
    "rvol_5d",
    "rvol_20d",
    "vol_of_vol",
    "rel_volume_1d",
    "rel_volume_5d_trend",
    "adv_20",
    "price_bucket",
    "mcap_bucket",
    "float_bucket",
    "days_since_last_top10",
    "appearances_30d",
    "appearances_90d",
    "earnings_today",
    "earnings_tomorrow",
    "days_to_earnings",
    "earnings_date_revisable",
    "sector_communication",
    "sector_consumer",
    "sector_energy",
    "sector_financials",
    "sector_healthcare",
    "sector_industrials",
    "sector_materials",
    "sector_realestate",
    "sector_technology",
    "sector_utilities",
    "sector_other",
    "is_biotech",
    "short_interest_pct_float",
    "days_to_cover",
    "dist_from_52w_high",
    "dist_from_52w_low",
    "consecutive_streak",
    "mkt_spy_ret_1d",
    "mkt_spy_ret_5d",
    "mkt_vix_level",
    "mkt_iwm_minus_spy_1d",
    "mkt_attention_regime_count",
    "as_of",
)

# --- T2 (decision time: 09:25 ET on t) --------------------------------------
# T2 == every T1 column PLUS premarket-derived columns, with `as_of`
# re-stamped to the T2 decision time and kept as the trailing column.

_T2_EXTRA_COLUMNS: tuple[str, ...] = (
    "premarket_gap_pct",
    "premarket_dollar_volume",
    "premarket_rel_volume",
    "premarket_high_to_last_drawdown",
    "premarket_trade_count",
    "premarket_first_trade_minutes",
    "overnight_halt_flag",
)

T2_COLUMNS: tuple[str, ...] = T1_COLUMNS[:-1] + _T2_EXTRA_COLUMNS + ("as_of",)

# --- prior_close input contract (T2 only) ------------------------------------
#
# Defect 2 (CONFIRMED): `pipeline.build_features_step`'s default prior_close
# frame and `top10.features.t2._prior_close_lookup` previously disagreed on
# the column name (`close` vs `prior_close`), which crashed the only T2
# orchestration path with `KeyError: 'prior_close'` on every real call.
# `top10.baselines.b4_premarket_gap` already required this exact contract
# (see its own docstring), so it needed no change -- `t2`/`pipeline` were
# the two consumers that had to agree, and now both use this single name.
#
# `close` is the PRIOR trading day's close, indexed by the trade_date being
# PREDICTED (already shifted forward by the caller so a simple join works).
# A bare `pd.Series` (ticker -> close) is also still accepted by
# `top10.features.t2.build_t2_features` directly, bypassing the `as_of`
# guard below.
PRIOR_CLOSE_COLUMNS: tuple[str, ...] = ("trade_date", "ticker", "close", "as_of")


@dataclass(frozen=True)
class FeatureSpec:
    """Locks a feature task's column contract for hashing + validation."""

    task: str  # "T1" | "T2"
    columns: tuple[str, ...] = field(default_factory=tuple)
    version: str = FEATURE_SPEC_VERSION

    @property
    def spec_hash(self) -> str:
        return hash_spec(
            {"task": self.task, "columns": list(self.columns), "version": self.version}
        )


T1_SPEC = FeatureSpec(task="T1", columns=T1_COLUMNS, version=FEATURE_SPEC_VERSION)
T2_SPEC = FeatureSpec(task="T2", columns=T2_COLUMNS, version=FEATURE_SPEC_VERSION)


def validate_frame(df: pd.DataFrame, spec: FeatureSpec) -> None:
    """Raise `ValueError` unless `df`'s columns match `spec.columns` exactly
    -- same set AND same order. Missing/extra/reordered columns all fail.
    """
    actual = list(df.columns)
    expected = list(spec.columns)
    if actual == expected:
        return

    actual_set, expected_set = set(actual), set(expected)
    missing = [c for c in expected if c not in actual_set]
    extra = [c for c in actual if c not in expected_set]

    if missing or extra:
        raise ValueError(
            f"validate_frame: frame does not match spec {spec.task!r} "
            f"(hash={spec.spec_hash}). missing={missing} extra={extra}"
        )

    raise ValueError(
        f"validate_frame: frame columns are out of order for spec "
        f"{spec.task!r} (hash={spec.spec_hash}). "
        f"expected={expected} actual={actual}"
    )


def feature_output_path(
    spec: FeatureSpec, trade_date: dt.date | dt.datetime | pd.Timestamp
) -> Path:
    """Return `data/features/<spec-hash>/<trade_date>.parquet`, per §2.4."""
    directory = spec_dir(DATA_FEATURES, spec.spec_hash)
    return directory / f"{pd.Timestamp(trade_date).date().isoformat()}.parquet"


def write_features(
    df: pd.DataFrame,
    spec: FeatureSpec,
    trade_date: dt.date | dt.datetime | pd.Timestamp,
) -> Path:
    """Validate `df` against `spec` and persist it under the spec's hash
    directory. Returns the path written.

    TOP FINDING (adversarial audit): the anti-leakage harness had zero
    production call sites. This is the persistence gate -- a frame whose
    `as_of` is after this task's own decision time for `trade_date` must
    never be written to disk, since a written feature file is exactly what
    downstream training/live-prediction code trusts blindly.
    """
    validate_frame(df, spec)

    # Lazy import: avoids a module-scope import-time dependency between the
    # spec registry (imported by t1.py/t2.py at module scope) and
    # top10.leakage (owned/being hardened concurrently by another agent).
    from top10.leakage import assert_decision_time_safe

    trade_date_ts = pd.Timestamp(trade_date)
    if spec.task == "T1":
        from top10.features.t1 import decision_time_t1

        decision_time = decision_time_t1(trade_date_ts)
    elif spec.task == "T2":
        from top10.features.t2 import decision_time_t2

        decision_time = decision_time_t2(trade_date_ts)
    else:
        decision_time = None

    if decision_time is not None:
        assert_decision_time_safe(df, decision_time)

    path = feature_output_path(spec, trade_date)
    write_parquet(df, path)
    return path
