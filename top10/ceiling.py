"""Ceiling estimation -- plan §3.2, checklist 2.2, docs/CEILING.md.

Estimates the practical recall ceiling: the share of positive-label days
that were driven by genuinely unscheduled news, as opposed to a scheduled
catalyst (an earnings date, a known FDA PDUFA date, etc.) or carryover
momentum from a prior day's move. Only the unscheduled share is, in
principle, predictable ahead of time from the kind of point-in-time
features this project builds -- scheduled-catalyst days are trivially
knowable and carryover days are largely captured by B1.

This module has NO LLM SDK dependency and makes NO network calls: the
classifier is injected as `classifier_fn`, so a human, a scripted
heuristic, or an LLM call living entirely in the CALLER's code can be
plugged in without this module knowing or caring which.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from scipy import stats

from top10.experiment import HOLDOUT_START, assert_holdout_sealed

# --- Rubric ------------------------------------------------------------
#
# Frozen per docs/CEILING.md "Goal": every positive label instance must be
# classified into exactly one of these three categories. Do not edit this
# string without also updating docs/CEILING.md -- it is the operative
# definition, not just documentation of one.

CLASSIFICATION_RUBRIC = """\
CEILING CLASSIFICATION RUBRIC (docs/CEILING.md)

Classify the ticker's top-10 appearance on `trade_date` into EXACTLY ONE
of the following three categories:

1. scheduled_catalyst
   A known, calendar-visible event whose timing was public knowledge
   before the move -- an earnings report, a scheduled FDA decision date
   (e.g. PDUFA), a scheduled investor day, a scheduled index
   inclusion/exclusion effective date, or similar.

2. carryover
   The move is largely a continuation of a prior trading day's move --
   the ticker was already trending sharply (up or down, gap-continuation,
   short-squeeze continuation) with no new information event identifiable
   on `trade_date` itself.

3. unscheduled
   A genuinely unscheduled news event drove the move -- an unscheduled
   FDA action, an M&A announcement/rumor, an unscheduled guidance
   update/preannouncement, activist-investor news, litigation news,
   regulatory action, or any other event whose timing was not knowable in
   advance.

If more than one category plausibly applies, classify by the PRIMARY
driver of the move. When genuinely ambiguous, prefer `carryover` over
`unscheduled` -- the ceiling estimate should be conservative (an
under-estimate of the unscheduled share, not an over-estimate).
"""

_VALID_CATEGORIES = ("scheduled_catalyst", "carryover", "unscheduled")


# --- Sampling ------------------------------------------------------------


def sample_positives(
    labels: pd.DataFrame,
    n: int = 200,
    seed: int = 0,
    *,
    include_holdout: bool = False,
    unseal_token: str | None = None,
) -> pd.DataFrame:
    """Sample `n` positive (label==1) label instances, reproducibly via
    `seed`.

    `include_holdout=False` (the default, and the only mode
    docs/CEILING.md's protocol condones) restricts sampling to
    `trade_date < 2023-01-01` -- the sealed holdout is never touched by
    ceiling estimation, since that would itself be a holdout read.

    Reading holdout-dated (>= 2023-01-01) positives requires BOTH
    `include_holdout=True` AND `unseal_token="PREREG_FROZEN"`
    (`top10.experiment.assert_holdout_sealed`) -- there used to be a plain
    `pre_holdout_only: bool` flag here that any caller could flip to
    `False` with no gate at all, despite this docstring's earlier claim
    that the holdout "is never touched"; that flag is gone, replaced by
    the same unseal-token mechanism used everywhere else the holdout is
    reachable (Plan §6 / P12).
    """
    positives = labels[labels["label"] == 1]
    if not include_holdout:
        positives = positives[pd.to_datetime(positives["trade_date"]) < HOLDOUT_START]
    else:
        assert_holdout_sealed(pd.to_datetime(positives["trade_date"]), unseal_token=unseal_token)

    if positives.empty:
        return positives.reset_index(drop=True)

    n_sample = min(n, len(positives))
    return positives.sample(n=n_sample, random_state=seed).reset_index(drop=True)


# --- Classification ------------------------------------------------------


def classify(
    samples: pd.DataFrame,
    classifier_fn: Callable[[pd.Series], str],
) -> pd.DataFrame:
    """Apply `classifier_fn` (human or LLM, injected by the caller) to
    every row of `samples`, returning `samples` with a `category` column
    appended.

    `classifier_fn(row) -> one of CLASSIFICATION_RUBRIC's three category
    names`. Raises `ValueError` immediately if `classifier_fn` returns
    anything else -- a typo'd category would silently corrupt the ceiling
    estimate.
    """
    categories = []
    for _, row in samples.iterrows():
        category = classifier_fn(row)
        if category not in _VALID_CATEGORIES:
            raise ValueError(
                f"classify: classifier_fn returned {category!r}, which is not one "
                f"of {_VALID_CATEGORIES}."
            )
        categories.append(category)

    out = samples.copy()
    out["category"] = categories
    return out


# --- Ceiling estimate ------------------------------------------------------


@dataclass(frozen=True)
class CeilingEstimate:
    n: int
    unscheduled_count: int
    unscheduled_share: float
    ci_low: float
    ci_high: float
    confidence: float
    category_counts: dict[str, int]


def _wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Preferred over a naive normal-approximation interval here because
    `n=200` (docs/CEILING.md) is small enough that the naive interval can
    produce nonsensical bounds outside `[0, 1]` near the extremes -- Wilson
    stays within `[0, 1]` by construction and is the standard
    recommendation for this sample size.
    """
    if n == 0:
        return (0.0, 0.0)

    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    phat = successes / n
    denom = 1 + (z**2) / n
    center = phat + (z**2) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2) / (4 * n**2))

    low = (center - margin) / denom
    high = (center + margin) / denom
    return (max(0.0, low), min(1.0, high))


def estimate_ceiling(classified: pd.DataFrame, confidence: float = 0.95) -> CeilingEstimate:
    """Compute the unscheduled share plus its Wilson score confidence
    interval from a `classify`-output frame (must carry a `category`
    column)."""
    if "category" not in classified.columns:
        raise ValueError("estimate_ceiling: `classified` frame has no 'category' column; run classify() first.")

    n = len(classified)
    counts = {cat: int((classified["category"] == cat).sum()) for cat in _VALID_CATEGORIES}
    unscheduled_count = counts["unscheduled"]

    unscheduled_share = (unscheduled_count / n) if n > 0 else float("nan")
    ci_low, ci_high = _wilson_interval(unscheduled_count, n, confidence=confidence)

    return CeilingEstimate(
        n=n,
        unscheduled_count=unscheduled_count,
        unscheduled_share=unscheduled_share,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        category_counts=counts,
    )


def render_ceiling_md(estimate: CeilingEstimate) -> str:
    """Render `estimate` as markdown to paste into `docs/CEILING.md`.

    Does NOT write to docs/CEILING.md itself -- callers decide when/how to
    commit the estimate."""
    pct = lambda x: f"{x:.1%}"  # noqa: E731

    lines = [
        "## Ceiling estimate",
        "",
        f"- **Sample size (n)**: {estimate.n}",
        f"- **Unscheduled count**: {estimate.unscheduled_count}",
        f"- **Unscheduled share (point estimate)**: {pct(estimate.unscheduled_share)}",
        f"- **{estimate.confidence:.0%} Wilson score CI**: "
        f"[{pct(estimate.ci_low)}, {pct(estimate.ci_high)}]",
        "",
        "| Category | Count | Share |",
        "|----------|-------|-------|",
    ]
    for cat in _VALID_CATEGORIES:
        count = estimate.category_counts.get(cat, 0)
        share = (count / estimate.n) if estimate.n > 0 else float("nan")
        lines.append(f"| {cat} | {count} | {pct(share)} |")

    return "\n".join(lines) + "\n"
