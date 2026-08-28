"""Sanity checks for built labels/universe, per plan §2.5.

Each check returns a structured `CheckResult` (never a bare bool) so
failures carry a human-readable reason. `run_all` aggregates them into a
`SanityReport` used to gate CI via the module CLI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import pandas as pd

from top10.storage import read_parquet

# --- P4 tripwire, upper bound only (contamination): docs/LABEL_SPEC.md
# §Corporate-action exclusions says explicitly that "+200%" is the
# contamination signature ("if median is +200%, corporate actions are
# leaking"), and the plan's stated EXPECTED range for a clean top-10
# median is +20% to +60%. We set the hard-FAIL threshold at +100% --
# comfortably above the +60% "typical" ceiling (so a merely hot day never
# aborts the pipeline) and comfortably below the +200% contamination
# example (so real split/reverse-split leakage is still caught with
# margin). This bound alone does real, safety-critical work: it is the
# only thing standing between a split-artifact-poisoned label set and
# every downstream consumer (features, baselines, walk-forward).
_MEDIAN_GAINER_CONTAMINATION_THRESHOLD = 1.00

# --- Quiet-market regime, lower bound only: the plan's +20%..+60%
# "typical" range is an EXPECTATION, not a hard floor. A universe filtered
# to ADV >= $1M routinely produces top-10 medians of 10-18% on quiet days
# -- that is normal market behavior, not a data defect, and must never
# abort the pipeline (Defect 2). It is still worth a human's attention
# (an unusually quiet regime, or a subtly too-narrow universe), so it is
# surfaced as a WARNING, never a FAILURE.
_MEDIAN_GAINER_QUIET_THRESHOLD = 0.20

_CONTINUITY_MAX_DRIFT = 0.05

# --- Defect 3 tripwire: fraction of a day's universe that may be dropped
# between `universe` and `labels` (no day-t bar -- halted all day /
# delisted intraday) before it is treated as a data-outage signature
# rather than ordinary single-name halts. A handful of halted/delisted
# names among thousands is routine; losing a large slice of the universe
# in one day is not a market event.
_COVERAGE_MAX_DROP_FRACTION = 0.05

_SPLIT_ACTION_TYPES = {"split", "reverse_split"}

# Severity values for CheckResult.severity.
_FAIL = "fail"
_WARN = "warn"


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)
    # "fail" (default): a failed result aborts the pipeline (SanityReport.
    # passed becomes False). "warn": a failed result is surfaced via
    # SanityReport.warnings but NEVER aborts the pipeline -- see Defect 2.
    severity: str = _FAIL

    def __str__(self) -> str:
        if self.passed:
            status = "PASS"
        elif self.severity == _WARN:
            status = "WARN"
        else:
            status = "FAIL"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class SanityReport:
    results: list[CheckResult]

    @property
    def failures(self) -> list[CheckResult]:
        """Hard failures only -- these are what gate the pipeline."""
        return [r for r in self.results if not r.passed and r.severity == _FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        """Soft warnings -- surfaced for a human, never abort the pipeline."""
        return [r for r in self.results if not r.passed and r.severity == _WARN]

    @property
    def passed(self) -> bool:
        """True iff there are no hard failures. Warnings do not affect this."""
        return len(self.failures) == 0

    def __str__(self) -> str:
        lines = [str(r) for r in self.results]
        summary = "ALL PASSED" if self.passed else f"{len(self.failures)} FAILED"
        if self.warnings:
            summary += f", {len(self.warnings)} WARNING(S)"
        lines.append(f"--- {summary} ({len(self.results)} checks) ---")
        return "\n".join(lines)


def check_universe_continuity(labels_or_universe: pd.DataFrame) -> CheckResult:
    """Ticker count per trading day should drift smoothly, not step-change.

    Flags any day-over-day change in unique ticker count > 5% as a
    suspected missing-delisted-data artifact.
    """
    if labels_or_universe.empty:
        return CheckResult(
            "universe_continuity", True, "no rows to check (empty input)"
        )

    counts = (
        labels_or_universe.groupby("trade_date")["ticker"]
        .nunique()
        .sort_index()
    )

    if len(counts) < 2:
        return CheckResult(
            "universe_continuity", True, "fewer than 2 trading days, nothing to compare"
        )

    offenders = []
    prev_date, prev_count = None, None
    for trade_date, count in counts.items():
        if prev_count is not None and prev_count > 0:
            pct_change = abs(count - prev_count) / prev_count
            if pct_change > _CONTINUITY_MAX_DRIFT:
                offenders.append(
                    {
                        "from_date": prev_date,
                        "to_date": trade_date,
                        "from_count": prev_count,
                        "to_count": count,
                        "pct_change": pct_change,
                    }
                )
        prev_date, prev_count = trade_date, count

    if offenders:
        return CheckResult(
            "universe_continuity",
            False,
            f"{len(offenders)} day(s) had a >{_CONTINUITY_MAX_DRIFT:.0%} step-change "
            "in unique ticker count -- suspected missing-delisted-data artifact",
            {"offenders": offenders},
        )

    return CheckResult(
        "universe_continuity", True, "ticker count drifts smoothly across days"
    )


def _median_top10_by_day(labels: pd.DataFrame) -> pd.Series:
    top = labels[labels["label"] == 1]
    if top.empty:
        return pd.Series(dtype=float)
    return top.groupby("trade_date")["return_t"].median()


def check_median_gainer_contamination(labels: pd.DataFrame) -> CheckResult:
    """Hard FAIL (severity="fail"): median top-10 return above
    `_MEDIAN_GAINER_CONTAMINATION_THRESHOLD` (+100%).

    P4 tripwire, UPPER bound only: an implausibly large median means
    corporate actions are leaking into unadjusted returns, and the label
    set is invalid per docs/LABEL_SPEC.md ("the label set is invalid and
    must be rebuilt"). This is the check that does real safety work and
    the only one of the two median-gainer checks allowed to abort the
    pipeline (Defect 2) -- see module-level threshold comment for the
    +60% typical / +200% contamination calibration.
    """
    if labels.empty:
        return CheckResult(
            "median_gainer_contamination", True, "no rows to check (empty input)"
        )

    medians = _median_top10_by_day(labels)
    if medians.empty:
        return CheckResult(
            "median_gainer_contamination", True, "no labeled (label==1) rows to check"
        )

    offenders = medians[medians > _MEDIAN_GAINER_CONTAMINATION_THRESHOLD]

    if not offenders.empty:
        worst_date = offenders.idxmax()
        return CheckResult(
            "median_gainer_contamination",
            False,
            f"{len(offenders)} day(s) have a top-10 median return above "
            f"+{_MEDIAN_GAINER_CONTAMINATION_THRESHOLD:.0%} "
            f"(worst: {worst_date} = {offenders[worst_date]:.1%}) -- "
            "corporate actions are leaking into unadjusted returns",
            {"offenders": offenders.to_dict()},
            severity=_FAIL,
        )

    return CheckResult(
        "median_gainer_contamination",
        True,
        f"top-10 median return at or below +{_MEDIAN_GAINER_CONTAMINATION_THRESHOLD:.0%} "
        "on every day",
        severity=_FAIL,
    )


def check_median_gainer_quiet_regime(labels: pd.DataFrame) -> CheckResult:
    """Soft WARN (severity="warn"): median top-10 return below
    `_MEDIAN_GAINER_QUIET_THRESHOLD` (+20%).

    A universe filtered to ADV >= $1M routinely produces top-10 medians of
    10-18% on ordinary quiet days -- this is normal market regime
    variation, not a defect, and must NEVER abort the pipeline (Defect 2).
    It is still logged as a warning so a human can notice an unusually
    quiet stretch or an overly narrow universe.
    """
    if labels.empty:
        return CheckResult(
            "median_gainer_quiet_regime", True, "no rows to check (empty input)",
            severity=_WARN,
        )

    medians = _median_top10_by_day(labels)
    if medians.empty:
        return CheckResult(
            "median_gainer_quiet_regime", True, "no labeled (label==1) rows to check",
            severity=_WARN,
        )

    offenders = medians[medians < _MEDIAN_GAINER_QUIET_THRESHOLD]

    if not offenders.empty:
        worst_date = offenders.idxmin()
        return CheckResult(
            "median_gainer_quiet_regime",
            False,
            f"{len(offenders)} day(s) have a top-10 median return below "
            f"+{_MEDIAN_GAINER_QUIET_THRESHOLD:.0%} "
            f"(lowest: {worst_date} = {offenders[worst_date]:.1%}) -- "
            "quiet-market regime, NOT aborting the pipeline",
            {"offenders": offenders.to_dict()},
            severity=_WARN,
        )

    return CheckResult(
        "median_gainer_quiet_regime",
        True,
        f"top-10 median return at or above +{_MEDIAN_GAINER_QUIET_THRESHOLD:.0%} "
        "on every day",
        severity=_WARN,
    )


def check_label_cardinality(labels: pd.DataFrame) -> CheckResult:
    """Exactly 10 positives (label==1) per trading day, unless the day's
    universe itself has fewer than 10 candidates."""
    if labels.empty:
        return CheckResult("label_cardinality", True, "no rows to check (empty input)")

    per_day = labels.groupby("trade_date").agg(
        n_positive=("label", "sum"), n_total=("ticker", "count")
    )

    offenders = per_day[
        (per_day["n_positive"] != 10) & (per_day["n_total"] >= 10)
    ]

    if not offenders.empty:
        return CheckResult(
            "label_cardinality",
            False,
            f"{len(offenders)} day(s) do not have exactly 10 positives despite "
            "a universe of >= 10 candidates",
            {"offenders": offenders.to_dict(orient="index")},
        )

    return CheckResult(
        "label_cardinality",
        True,
        "every day has exactly 10 positives (or fewer only when the universe "
        "itself was < 10)",
    )


def check_no_split_days(
    labels: pd.DataFrame, corporate_actions: pd.DataFrame
) -> CheckResult:
    """No labeled row may share a ticker+date with a split/reverse_split."""
    if labels.empty or corporate_actions.empty:
        return CheckResult(
            "no_split_days", True, "no rows to check (empty labels or corporate_actions)"
        )

    splits = corporate_actions[
        corporate_actions["action_type"].isin(_SPLIT_ACTION_TYPES)
    ][["ticker", "ex_date"]].drop_duplicates()

    merged = labels.merge(
        splits, left_on=["ticker", "trade_date"], right_on=["ticker", "ex_date"], how="inner"
    )

    if not merged.empty:
        offenders = merged[["ticker", "trade_date"]].drop_duplicates().to_dict(
            orient="records"
        )
        return CheckResult(
            "no_split_days",
            False,
            f"{len(offenders)} labeled row(s) share a ticker+date with a "
            "split/reverse_split -- P4 exclusion did not run",
            {"offenders": offenders},
        )

    return CheckResult(
        "no_split_days", True, "no labeled row shares a ticker+date with a split"
    )


def check_universe_coverage(labels: pd.DataFrame, universe: pd.DataFrame) -> CheckResult:
    """Hard FAIL: the fraction of a day's universe dropped between
    `universe` and `labels` should be small.

    Defect 3: `top10.labels.build_labels` explicitly excludes (and logs)
    any universe name with no day-t bar (halted all day / delisted
    intraday) -- it can no longer silently vanish. This check is the
    downstream tripwire on that explicit count: a handful of halted or
    intraday-delisted names among a large universe is routine and
    expected, but losing an unusually large slice of the day's universe in
    one shot is a data-outage signature (a vendor gap), not a market
    event, per the same reasoning `check_universe_continuity` already
    applies to day-over-day ticker counts.

    Note some of the universe-to-labels gap on any given day is also
    ordinary P4 split/reverse-split exclusion (`check_no_split_days`
    already covers that separately) -- both sources are folded into one
    fraction here deliberately, since either one spiking is worth a human
    look.
    """
    if labels.empty or universe.empty:
        return CheckResult(
            "universe_coverage", True, "no rows to check (empty labels or universe)"
        )

    uni_counts = universe.groupby("trade_date")["ticker"].nunique()
    lbl_counts = labels.groupby("trade_date")["ticker"].nunique()

    offenders = {}
    for trade_date, uni_n in uni_counts.items():
        if uni_n <= 0:
            continue
        lbl_n = int(lbl_counts.get(trade_date, 0))
        dropped = int(uni_n) - lbl_n
        frac = dropped / uni_n
        if frac > _COVERAGE_MAX_DROP_FRACTION:
            offenders[trade_date] = {
                "universe_n": int(uni_n),
                "labeled_n": lbl_n,
                "dropped": dropped,
                "dropped_fraction": frac,
            }

    if offenders:
        worst_date = max(offenders, key=lambda d: offenders[d]["dropped_fraction"])
        worst = offenders[worst_date]
        return CheckResult(
            "universe_coverage",
            False,
            f"{len(offenders)} day(s) dropped more than "
            f"{_COVERAGE_MAX_DROP_FRACTION:.0%} of the universe between "
            f"universe and labels (worst: {worst_date} dropped "
            f"{worst['dropped']}/{worst['universe_n']} = "
            f"{worst['dropped_fraction']:.1%}) -- suspected data outage "
            "(missing day-t bars), not simultaneous halts",
            {"offenders": offenders},
        )

    return CheckResult(
        "universe_coverage",
        True,
        f"no day dropped more than {_COVERAGE_MAX_DROP_FRACTION:.0%} of the "
        "universe between universe and labels",
    )


def run_all(
    labels: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    universe: pd.DataFrame | None = None,
) -> SanityReport:
    """Run every sanity check and aggregate into a `SanityReport`.

    `universe` is optional; when omitted, `check_universe_continuity` runs
    against `labels` itself (still useful, just coarser), and
    `check_universe_coverage` is skipped entirely (it has nothing to
    compare `labels` against).
    """
    continuity_input = universe if universe is not None else labels

    results = [
        check_universe_continuity(continuity_input),
        check_median_gainer_contamination(labels),
        check_median_gainer_quiet_regime(labels),
        check_label_cardinality(labels),
        check_no_split_days(labels, corporate_actions),
    ]
    if universe is not None and not universe.empty:
        results.append(check_universe_coverage(labels, universe))

    return SanityReport(results)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run sanity checks against built labels and exit non-zero on failure."
    )
    parser.add_argument(
        "--labels", required=True, help="Path to a labels parquet file (or directory)."
    )
    parser.add_argument(
        "--corporate-actions",
        required=False,
        default=None,
        help="Path to a corporate_actions parquet file, if available.",
    )
    args = parser.parse_args(argv)

    labels = read_parquet(args.labels)
    corporate_actions = (
        read_parquet(args.corporate_actions)
        if args.corporate_actions
        else pd.DataFrame(columns=["ticker", "ex_date", "action_type"])
    )

    report = run_all(labels, corporate_actions)
    print(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(_main())
