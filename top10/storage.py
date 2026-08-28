"""Point-in-time storage guardrails.

This module is the single chokepoint that enforces §2.4 of
docs/LABEL_SPEC.md ("Point-in-time invariants"): no row whose `as_of` is
after the decision time may be read or persisted as usable.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class LeakageError(Exception):
    """Raised when point-in-time invariants are violated."""


def write_parquet(
    df: pd.DataFrame, path: Path | str, *, as_of_required: bool = True
) -> None:
    """Write `df` to `path` as parquet.

    Raises `LeakageError` if `as_of_required` and `df` lacks an `as_of`
    column. Creates parent directories as needed.
    """
    if as_of_required and "as_of" not in df.columns:
        raise LeakageError(
            f"write_parquet: DataFrame written to {path} has no 'as_of' "
            "column; point-in-time provenance cannot be verified."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


def read_parquet(
    path: Path | str, *, as_of: dt.datetime | None = None
) -> pd.DataFrame:
    """Read parquet at `path`.

    When `as_of` is given, only rows with `row.as_of <= as_of` are
    returned. This is the chokepoint that enforces point-in-time reads.
    """
    df = pd.read_parquet(path)
    if as_of is not None:
        if "as_of" not in df.columns:
            raise LeakageError(
                f"read_parquet: {path} has no 'as_of' column; cannot "
                "enforce as_of filtering."
            )
        df = df[df["as_of"] <= as_of].reset_index(drop=True)
    return df


def assert_as_of_le(df: pd.DataFrame, decision_time: dt.datetime) -> None:
    """Raise `LeakageError` if any row's `as_of` is after `decision_time`.

    The error message names the offending tickers/dates when identifying
    columns are present.
    """
    if "as_of" not in df.columns:
        raise LeakageError("assert_as_of_le: DataFrame has no 'as_of' column.")

    leaking = df[df["as_of"] > decision_time]
    if leaking.empty:
        return

    id_cols = [c for c in ("ticker", "trade_date", "ex_date") if c in leaking.columns]
    if id_cols:
        offenders = leaking[id_cols].to_dict(orient="records")
    else:
        offenders = leaking.index.tolist()

    raise LeakageError(
        f"assert_as_of_le: {len(leaking)} row(s) have as_of > "
        f"{decision_time!r}: {offenders!r}"
    )


def append_only_write(obj: object, path: Path | str) -> None:
    """Write `obj` as JSON to `path`, refusing to overwrite an existing file.

    Used for `data/predictions/` and raw Robinhood captures — temporal
    pre-commitment means a prediction, once written, can never be edited.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"append_only_write: {path} already exists and cannot be "
            "overwritten."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def spec_dir(base: Path, spec_hash: str) -> Path:
    """Return `base/<spec_hash>/`, creating it if needed."""
    directory = Path(base) / spec_hash
    directory.mkdir(parents=True, exist_ok=True)
    return directory
