"""Experiment logging and the holdout seal — docs/PREREG_TOP10.md §5.3, §6, P12.

Two responsibilities live here:

1. `assert_holdout_sealed` -- the ONE guarded chokepoint every
   holdout-touching code path (walk-forward splits, tuning windows, manual
   scripts) must call. It refuses unless the caller supplies the literal
   unseal token `"PREREG_FROZEN"`, per PREREG_TOP10's "One-time holdout
   rule": the plan must be finalized and committed before the holdout is
   ever read.
2. `log_experiment` / `count_corrected_variants` -- the P12 defense. Per
   §5.3, "an unlogged run does not count": only experiments filed here as
   `experiments/EXP-###.md` can be cited in the final claim, and only the
   ones explicitly flagged `counts_toward_family_wise_correction` feed the
   Holm correction in `top10.metrics.family_wise_correction`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from top10.config import EXPERIMENTS
from top10.storage import LeakageError

# Plan §6: holdout is 2023-01-01 through data end, and is sealed.
HOLDOUT_START = pd.Timestamp("2023-01-01")

UNSEAL_TOKEN = "PREREG_FROZEN"

_EXP_FILENAME_RE = re.compile(r"^EXP-(\d+)\.md$")

# Accepts both the canonical template line
#   **Counts toward family-wise correction? (y/n)**: y
# and the hand-written prose form actually used in EXP-003
#   **Counts toward family-wise correction?** **YES** -- first fitted model
# `(y/n)` and the surrounding `**` are optional on either side, and the
# answer may be `y` or `yes`, case-insensitive. This is deliberately
# permissive about formatting because the cost of a false negative here
# (a real logged variant silently not counted, corrupting the Holm
# denominator) is far worse than the cost of a false positive.
_COUNTS_LINE_RE = re.compile(
    r"\*{0,2}Counts toward family-wise correction\?(?:\s*\(y/n\))?\*{0,2}:?\s*\*{0,2}(?:yes|y)\b",
    re.IGNORECASE,
)
_PVALUE_LINE_RE = re.compile(r"p-value\s*=\s*([0-9.eE+-]+)")


class NoCitableClaimError(RuntimeError):
    """Raised when a caller attempts to build a final PREREG claim backed by
    zero experiments counting toward the family-wise correction.

    Per docs/PREREG_TOP10.md §5.3, "an unlogged run does not count" -- a
    count of 0 means there is no p-value to Holm-correct and therefore no
    citable claim, not an empty-but-valid correction. This must be a loud
    failure, not a null result silently rendered as a "discovery".
    """


def assert_holdout_sealed(dates: Sequence[Any], *, unseal_token: str | None = None) -> None:
    """Raise unless `unseal_token == "PREREG_FROZEN"` AND `dates` contains
    at least one date on/after the sealed holdout start.

    This is the single guarded function every holdout-touching path must
    route through (Plan §6 / P12). A caller with the correct token is
    explicitly asserting the PREREG plan is finalized/committed and the
    one-time holdout read is authorized.
    """
    if unseal_token == UNSEAL_TOKEN:
        return

    timestamps = [pd.Timestamp(d) for d in dates]
    touches_holdout = any(ts >= HOLDOUT_START for ts in timestamps)
    if touches_holdout:
        raise LeakageError(
            f"assert_holdout_sealed: one or more dates fall on/after the sealed "
            f"holdout start ({HOLDOUT_START.date()}) and no valid unseal_token was "
            "supplied. Per docs/PREREG_TOP10.md 'One-time holdout rule', the holdout "
            "may not be touched until the plan is frozen and this call is explicitly "
            f"unsealed with unseal_token={UNSEAL_TOKEN!r}."
        )


def assert_frame_holdout_sealed(
    df: pd.DataFrame, *, unseal_token: str | None = None, date_col: str = "trade_date"
) -> None:
    """Single reusable guard for every DataFrame-shaped entry point that
    touches the holdout (Plan §6 / P12): extract `date_col` from `df` and
    route it through `assert_holdout_sealed`.

    Every module that owns a chokepoint reachable with an arbitrary
    holdout-dated frame -- `baselines`, `ceiling`, `model`, the `cli` --
    should call this (or `assert_holdout_sealed` directly for a bare date
    sequence) rather than reimplementing the extraction, so the seal is
    structural rather than a convention each call site has to remember.

    A no-op if `df` is empty/None or lacks `date_col` (nothing to check).
    """
    if df is None or df.empty or date_col not in df.columns:
        return
    assert_holdout_sealed(df[date_col], unseal_token=unseal_token)


def _next_exp_id(experiments_dir: Path) -> int:
    existing_ids = []
    if experiments_dir.exists():
        for p in experiments_dir.glob("EXP-*.md"):
            match = _EXP_FILENAME_RE.match(p.name)
            if match:
                existing_ids.append(int(match.group(1)))
    return (max(existing_ids) + 1) if existing_ids else 1


def _per_year_table(per_year: pd.DataFrame) -> str:
    header = "| Year | precision@10 | MAP@10 | mean hits | median hits | n_days |"
    sep = "|------|--------------|--------|-----------|-------------|--------|"
    if per_year is None or per_year.empty:
        return "\n".join([header, sep, "|      |              |        |           |             |        |"])

    rows = [header, sep]
    for _, row in per_year.iterrows():
        rows.append(
            "| {year} | {precision:.4f} | {map_:.4f} | {mean_hits:.2f} | {median_hits:.2f} | {n_days} |".format(
                year=int(row["year"]),
                precision=row["precision_at_k"],
                map_=row["map_at_k"],
                mean_hits=row["mean_hits"],
                median_hits=row["median_hits"],
                n_days=int(row["n_days"]),
            )
        )
    return "\n".join(rows)


def log_experiment(
    *,
    author: str,
    hypothesis: str,
    feature_spec_hash: str,
    label_spec_hash: str,
    task: str,
    model_family: str,
    hyperparameters: dict,
    feature_set: str,
    train_window: tuple[str, str],
    validation_window: tuple[str, str],
    per_year: pd.DataFrame,
    vs_baseline: dict,
    counts_toward_family_wise_correction: bool,
    holdout_window: tuple[str, str] | None = None,
    p_value: float | None = None,
    notes: str = "",
    experiments_dir: Path | str = EXPERIMENTS,
    date: str | None = None,
) -> Path:
    """Write `experiments/EXP-###.md` from the structure of
    `experiments/TEMPLATE.md`, auto-incrementing the experiment id.

    `vs_baseline` is expected to be the dict shape returned by
    `top10.metrics.compare_to_baseline` / `WalkForwardResult.vs_baseline`.
    """
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    exp_id = _next_exp_id(experiments_dir)
    exp_name = f"EXP-{exp_id:03d}"
    out_path = experiments_dir / f"{exp_name}.md"

    date_str = date or pd.Timestamp.today().date().isoformat()

    counts_str = "y" if counts_toward_family_wise_correction else "n"
    p_value_str = "" if p_value is None else f"{p_value:.6g}"

    holdout_line = f"{holdout_window[0]} - {holdout_window[1]}" if holdout_window else "(not run -- not a holdout evaluation)"

    t_test = vs_baseline.get("paired_t_test", {}) if vs_baseline else {}
    wilcoxon = vs_baseline.get("wilcoxon_test", {}) if vs_baseline else {}

    content = f"""# Experiment Template — docs/PREREG_TOP10.md §5.3

## Identity

- **Experiment ID**: {exp_name}
- **Date**: {date_str}
- **Author**: {author}

## Hypothesis

{hypothesis}

## Spec hashes

- **Feature spec hash**: `{feature_spec_hash}`
- **Label spec hash**: `{label_spec_hash}`

## Model configuration

- **Task**: {task}
- **Model family**: {model_family}
- **Hyperparameters**: {hyperparameters}
- **Feature set**: {feature_set}

## Split / walk-forward window

- **Train window**: {train_window[0]} - {train_window[1]}
- **Validation window (walk-forward, expanding)**: {validation_window[0]} - {validation_window[1]}
- **Holdout window** (only if this run is the sealed holdout evaluation): {holdout_line}

## Results — per-year table

{_per_year_table(per_year)}

## Comparison vs B4

- **Mean hits/day delta vs B4**: {vs_baseline.get("mean_hits_delta", "") if vs_baseline else ""}
- **Years won vs B4** (out of years evaluated): {vs_baseline.get("years_won", "") if vs_baseline else ""} / {vs_baseline.get("years_total", "") if vs_baseline else ""}
- **Paired t-test**: statistic = {t_test.get("statistic", "")}, p-value = {t_test.get("p_value", "")}
- **Wilcoxon signed-rank test**: statistic = {wilcoxon.get("statistic", "")}, p-value = {wilcoxon.get("p_value", "")}

## Family-wise correction

- **Counts toward family-wise correction? (y/n)**: {counts_str}
- If yes, record this experiment's raw p-value here so it can be included
  in the `top10.metrics.family_wise_correction` call over all counted
  variants: p-value = {p_value_str}

## Notes / caveats

{notes}
"""

    out_path.write_text(content)
    return out_path


def count_corrected_variants(experiments_dir: Path | str = EXPERIMENTS) -> int:
    """Count logged experiments flagged
    `counts_toward_family_wise_correction? (y/n): y` -- the true number of
    variants tried, so the final PREREG claim can be Holm-corrected
    (`top10.metrics.family_wise_correction`) against it. Per §5.3, an
    unlogged run does not count -- only files actually present here count.
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        return 0

    count = 0
    for p in sorted(experiments_dir.glob("EXP-*.md")):
        if not _EXP_FILENAME_RE.match(p.name):
            continue
        text = p.read_text()
        if _COUNTS_LINE_RE.search(text):
            count += 1
    return count


def assert_citable_claim(experiments_dir: Path | str = EXPERIMENTS) -> int:
    """Guard for the final-claim chokepoint (§5.3): return
    `count_corrected_variants(experiments_dir)`, raising
    :class:`NoCitableClaimError` if it is 0.

    Every code path that assembles the final PREREG claim (e.g. feeding
    p-values into `top10.metrics.family_wise_correction`) must call this
    first rather than calling `count_corrected_variants` directly and
    trusting the caller to notice a 0. Zero logged variants means no
    citable claim can be built, and that must fail loudly rather than
    silently producing `family_wise_correction([]) == []`.
    """
    count = count_corrected_variants(experiments_dir)
    if count == 0:
        raise NoCitableClaimError(
            "count_corrected_variants() == 0: no experiment in "
            f"{Path(experiments_dir)} counts toward the family-wise "
            "correction. Per docs/PREREG_TOP10.md §5.3, an unlogged run "
            "does not count -- a final claim cannot be built on zero "
            "counted variants."
        )
    return count
