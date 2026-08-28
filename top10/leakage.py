"""Anti-leakage harness — docs/PREREG_TOP10.md §"Anti-leakage requirements"
and plan §4.3.

Reusable assertions that other modules (features, baselines, model
training) call before trusting a frame. Every check here fails loud
(`LeakageError`) rather than silently dropping rows -- a silent drop hides
a bug in the caller's pipeline.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from top10.metrics import precision_at_k
from top10.storage import LeakageError, assert_as_of_le

__all__ = [
    "LeakageError",
    "assert_decision_time_safe",
    "assert_no_adjusted_prices",
    "verify_unadjusted",
    "assert_self_exclusion",
    "shuffle_label_test",
]

_SPLIT_ACTION_TYPES = {"split", "reverse_split"}


def assert_decision_time_safe(df: pd.DataFrame, decision_time: dt.datetime) -> None:
    """Wrap `storage.assert_as_of_le` with a TOP10-specific error message.

    Raises `LeakageError` if any row's `as_of` is after `decision_time` --
    i.e. the row was not knowable at decision time.
    """
    try:
        assert_as_of_le(df, decision_time)
    except LeakageError as exc:
        raise LeakageError(
            "assert_decision_time_safe: one or more rows were not yet "
            f"knowable at decision_time={decision_time!r}. This frame "
            f"cannot be used to make a prediction at that decision time. {exc}"
        ) from exc


def assert_no_adjusted_prices(df: pd.DataFrame, corporate_actions: pd.DataFrame | None = None) -> None:
    """Fail if a row's OHLCV plausibly reflects a corporate action it could
    not yet have known about -- i.e. a back-adjusted price series.

    Back-adjustment restates a HISTORICAL bar using a split ratio whose
    `ex_date` is in the FUTURE relative to that bar's own `trade_date`
    (that is the entire mechanism of a back-adjusted feed: today's series
    is divided through by every split that has since occurred, including
    ones that hadn't happened yet as of the bar's own date). So the
    structural tell for this leak is `ex_date > trade_date`, not
    `ex_date <= trade_date` -- flagging the latter (the earlier, buggy
    version of this function) filters out exactly the rows that back-
    adjustment produces and lets a fully back-adjusted feed pass on every
    row.

    A row is flagged when, for its ticker, there exists a corporate action
    where:
    - `ex_date > row.trade_date` (the split is in the row's future), AND
    - `action.as_of > row.as_of` (the row was written before the split was
      even knowable, so it cannot have been LEGITIMATELY split-adjusted at
      write time -- if its price nonetheless reflects that ratio, that is
      exactly what back-adjustment looks like).

    Honesty about this heuristic's limits: this is necessarily a
    conservative, metadata-only proxy. It cannot inspect the vendor's
    arithmetic directly, so it will flag rows for tickers that later
    happen to split -- which is extremely common for any long-lived
    ticker and is NOT on its own proof of adjustment. It is a "this row is
    suspect, go verify" signal, not a "this row is guilty" signal. Use
    `verify_unadjusted` alongside it: that function inspects the actual
    price series around a known split's `ex_date` for the discontinuity an
    unadjusted series must show, which is the closest this module can get
    to confirming (rather than merely suspecting) back-adjustment.

    When `corporate_actions` is not supplied, only a much weaker
    structural precondition on `df` itself is checked: that no row's
    `as_of` is before its own `trade_date` (impossible for a same-day
    unadjusted bar). **This branch provides essentially no protection
    against back-adjustment** -- the confirmed back-adjustment repro this
    function exists to catch (bar `as_of == trade_date`, split announced
    and effective years later) has `as_of == trade_date`, which passes
    this precondition cleanly. Do not treat a pass on the
    `corporate_actions=None` branch as "prices are unadjusted"; it only
    means the row was not lazily backdated, and it says nothing about
    genuine split back-adjustment. Pass `corporate_actions` whenever it is
    available.
    """
    if "as_of" not in df.columns:
        raise LeakageError("assert_no_adjusted_prices: frame has no 'as_of' column.")

    if corporate_actions is None:
        if "trade_date" not in df.columns:
            return
        bad = df[df["as_of"] < df["trade_date"]]
        if not bad.empty:
            raise LeakageError(
                "assert_no_adjusted_prices: "
                f"{len(bad)} row(s) have as_of before their own trade_date, "
                "which is impossible for an unadjusted same-day bar: "
                f"{bad[['ticker', 'trade_date', 'as_of']].to_dict(orient='records')!r}"
            )
        return

    required = {"ticker", "trade_date", "as_of"}
    if not required.issubset(df.columns):
        raise LeakageError(f"assert_no_adjusted_prices: frame missing columns {required - set(df.columns)}")
    if not {"ticker", "ex_date", "as_of"}.issubset(corporate_actions.columns):
        raise LeakageError("assert_no_adjusted_prices: corporate_actions missing required columns")

    merged = df.merge(
        corporate_actions[["ticker", "ex_date", "as_of"]].rename(
            columns={"as_of": "action_as_of"}
        ),
        on="ticker",
        how="inner",
    )
    # `ex_date > trade_date`: the split is in this bar's future -- the
    # structural signature of back-adjustment (see docstring above).
    # `action_as_of > as_of`: the bar was recorded before that split was
    # even knowable, so it could not have been legitimately adjusted for
    # it at write time.
    leaking = merged[
        (merged["ex_date"] > merged["trade_date"]) & (merged["action_as_of"] > merged["as_of"])
    ]
    if not leaking.empty:
        offenders = leaking[["ticker", "trade_date", "as_of", "ex_date", "action_as_of"]]
        raise LeakageError(
            "assert_no_adjusted_prices: "
            f"{len(leaking)} row(s) predate a split/reverse_split on their own "
            "ticker whose ex_date lies in the row's future -- the structural "
            f"signature of a back-adjusted price series: {offenders.to_dict(orient='records')!r}"
        )


def verify_unadjusted(
    bars: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    flat_band: float = 0.10,
) -> None:
    """Detect back-adjustment by looking for the price discontinuity an
    UNADJUSTED series must show at a known split's `ex_date`, and that an
    ADJUSTED (back-adjusted) series must NOT show.

    Heuristic: for each split/reverse_split action with a known `ratio`
    (new/old shares -- e.g. 2.0 for a 2-for-1 split), take the last close
    strictly before `ex_date` and the first close on/after `ex_date` for
    that ticker. An unadjusted series has a real jump there: price should
    move by roughly `1 / ratio` (a 2-for-1 split roughly halves the
    price). A back-adjusted series has already smoothed that jump out of
    history, so the observed ratio at `ex_date` will sit close to 1.0
    (flat) instead. A row is flagged when the observed close-to-close
    ratio at `ex_date` falls within `flat_band` of 1.0 (default: within
    10%) despite a known split existing there -- i.e. the series is
    suspiciously flat exactly where an unadjusted series must jump.

    Honesty about this heuristic's limits, spelled out rather than implied:
    - It is silent (no signal either way, NOT a pass) whenever `bars`
      doesn't have a bar strictly before AND a bar on/after `ex_date` for
      that ticker -- e.g. gaps around the split, a ticker that only starts
      trading after the split, or an `ex_date` outside the queried date
      range. A silent gap here is a false negative, not a guarantee of
      cleanliness.
    - Genuine same-day price moves that happen to coincide with `ex_date`
      (e.g. an earnings gap on the exact split day) can mask a real
      adjustment jump and produce a false negative, or -- far less likely
      given how large real split ratios are (smallest common ratio, 3-for-
      2, is a 33% jump, well outside a 10% flat band) -- an unusually
      volatility-free unadjusted day could in principle look flat and
      produce a false positive.
    - `flat_band=0.10` is chosen because it is comfortably inside the jump
      produced by the smallest common split ratios while still tolerant of
      bid/ask noise; it is not derived from any formal false-positive
      analysis and should be tuned against real data before being trusted
      unattended.
    - Dividends and ticker changes (no `ratio`, or `ratio` is NaN) are
      skipped entirely -- this function only checks splits/reverse-splits.

    Raises `LeakageError` naming every offending ticker/ex_date/observed
    ratio. Does not raise merely because a check could not be evaluated.
    """
    required_bars = {"ticker", "trade_date", "close"}
    if not required_bars.issubset(bars.columns):
        raise LeakageError(f"verify_unadjusted: bars frame missing columns {required_bars - set(bars.columns)}")
    required_ca = {"ticker", "ex_date", "action_type", "ratio"}
    if not required_ca.issubset(corporate_actions.columns):
        raise LeakageError(
            f"verify_unadjusted: corporate_actions frame missing columns {required_ca - set(corporate_actions.columns)}"
        )

    splits = corporate_actions[
        corporate_actions["action_type"].isin(_SPLIT_ACTION_TYPES) & corporate_actions["ratio"].notna()
    ]

    offenders = []
    for _, action in splits.iterrows():
        ticker = action["ticker"]
        ex_date = action["ex_date"]
        ratio = float(action["ratio"])
        if ratio <= 0 or math.isclose(ratio, 1.0):
            continue

        ticker_bars = bars[bars["ticker"] == ticker].sort_values("trade_date")
        before = ticker_bars[ticker_bars["trade_date"] < ex_date]
        after = ticker_bars[ticker_bars["trade_date"] >= ex_date]
        if before.empty or after.empty:
            # Cannot evaluate -- documented false-negative case above.
            continue

        last_before = float(before.iloc[-1]["close"])
        first_after = float(after.iloc[0]["close"])
        if last_before <= 0 or first_after <= 0:
            continue

        observed_ratio = first_after / last_before
        if abs(observed_ratio - 1.0) <= flat_band:
            offenders.append(
                {
                    "ticker": ticker,
                    "ex_date": ex_date,
                    "split_ratio": ratio,
                    "expected_close_ratio_approx": 1.0 / ratio,
                    "observed_close_ratio": observed_ratio,
                }
            )

    if offenders:
        raise LeakageError(
            "verify_unadjusted: "
            f"{len(offenders)} ticker/ex_date pair(s) show no price discontinuity "
            f"at a known split, consistent with a back-adjusted series: {offenders!r}"
        )


def assert_self_exclusion(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    rho_threshold: float = 0.99,
    min_group_size: int = 5,
    min_hit_fraction: float = 0.5,
) -> None:
    """A ticker's own label for day t must never leak into its own
    features for day t.

    Three layers, from cheapest/most literal to most general:
    1. Feature frames must not themselves carry `return_t`, `rank`, or
       `label` columns at all -- those are label-only fields per
       docs/LABEL_SPEC.md and have no business being feature inputs.
    2. No feature column is literally elementwise-equal to same-day
       `return_t` / `rank` / `label` (once aligned on trade_date +
       ticker). This only catches identity copies.
    3. Within each `trade_date`, no numeric feature column may be near-
       perfectly (`|rho| >= rho_threshold`, default 0.99) correlated --
       Pearson OR Spearman -- with same-day `return_t`, `rank`, or
       `label`, on a material share (`min_hit_fraction`, default 0.5) of
       the days where the check is evaluable. Layer 2 alone misses any
       monotonic or affine disguise of the label (e.g. `return_t * 100 +
       1`, or `rank * 2`) -- both are perfectly rank- and/or
       linearly-preserving, so an elementwise-identity check trivially
       lets them through while a real model would recover the label from
       them just as easily as from the raw column.

    Threshold choices, spelled out rather than assumed:
    - `rho_threshold=0.99` is deliberately very high. Real, non-leaked
      features (e.g. prior-day return, or premarket gap) are frequently
      *mildly* correlated with same-day return_t -- momentum/mean-
      reversion are real phenomena this model is trying to exploit, not
      artifacts to suppress. A threshold near-perfect correlation is meant
      to catch near-tautological relationships, not "this feature is
      informative", which is the whole point of building it.
    - `min_group_size=5`: correlation with fewer than 5 paired points per
      day is not evaluated. With 2 points, Pearson correlation is always
      exactly +/-1 by construction regardless of any real relationship --
      a false-positive trap this exists to avoid. Days below the minimum
      are skipped, not counted as evaluable, and not counted as hits.
    - `min_hit_fraction=0.5`: requiring the correlation to hold on a
      majority of evaluable days (rather than raising on any single day)
      avoids a false alarm from one unusually correlated day; a real leak
      is structural and shows up on essentially every day, not one.
    """
    forbidden_cols = {"return_t", "rank", "label"}
    present_forbidden = forbidden_cols.intersection(features.columns)
    if present_forbidden:
        raise LeakageError(
            "assert_self_exclusion: features frame directly carries "
            f"label-only column(s) {sorted(present_forbidden)}; these must "
            "never appear as feature inputs."
        )

    if "trade_date" not in features.columns or "ticker" not in features.columns:
        raise LeakageError("assert_self_exclusion: features frame needs trade_date + ticker to align.")

    merged = features.merge(
        labels[["trade_date", "ticker", "return_t", "rank", "label"]],
        on=["trade_date", "ticker"],
        how="inner",
    )
    if merged.empty:
        return

    feature_cols = [c for c in features.columns if c not in ("trade_date", "ticker", "as_of")]

    # Layer 2: literal elementwise identity.
    for feature_col in feature_cols:
        col = merged[feature_col]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        for label_col in ("return_t", "rank", "label"):
            if np.allclose(
                col.to_numpy(dtype=float),
                merged[label_col].to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise LeakageError(
                    f"assert_self_exclusion: feature column {feature_col!r} is "
                    f"identical to same-day label column {label_col!r} -- this "
                    "is a same-day label leak into the features."
                )

    # Layer 3: near-perfect same-day correlation (catches monotonic /
    # affine disguises that layer 2 cannot see).
    offenders = []
    for feature_col in feature_cols:
        if not pd.api.types.is_numeric_dtype(merged[feature_col]):
            continue
        for label_col in ("return_t", "rank", "label"):
            hit_days = []
            evaluable_days = 0
            for trade_date, group in merged.groupby("trade_date"):
                pair = group[[feature_col, label_col]].dropna()
                if len(pair) < min_group_size:
                    continue
                x = pair[feature_col].to_numpy(dtype=float)
                y = pair[label_col].to_numpy(dtype=float)
                if np.std(x) == 0 or np.std(y) == 0:
                    # Constant column on this day -- correlation is
                    # undefined, not evidence of anything.
                    continue
                evaluable_days += 1
                pearson_rho = float(stats.pearsonr(x, y).statistic)
                spearman_rho = float(stats.spearmanr(x, y).statistic)
                worst = max(abs(pearson_rho), abs(spearman_rho))
                if not math.isnan(worst) and worst >= rho_threshold:
                    hit_days.append(
                        {"trade_date": trade_date, "pearson": pearson_rho, "spearman": spearman_rho}
                    )

            if evaluable_days == 0:
                continue
            if (len(hit_days) / evaluable_days) >= min_hit_fraction:
                offenders.append(
                    {
                        "feature": feature_col,
                        "label_col": label_col,
                        "evaluable_days": evaluable_days,
                        "hit_days": hit_days,
                    }
                )

    if offenders:
        raise LeakageError(
            "assert_self_exclusion: "
            f"{len(offenders)} feature/label pair(s) are near-perfectly "
            f"(|rho| >= {rho_threshold}) correlated with a same-day label column "
            f"on >= {min_hit_fraction:.0%} of evaluable days -- this is consistent "
            f"with a disguised (monotonic/affine) same-day label leak: {offenders!r}"
        )


def _random_guess_tolerance(
    expected_precision: float,
    n_observations: int,
    *,
    multiplier: float = 3.0,
    z: float = 3.0,
) -> float:
    """Upper bound above which observed shuffle-label precision is no
    longer plausible as random-permutation noise around
    `expected_precision`, and should instead be read as leakage.

    Deliberately NOT a fixed floor (docs/PREREG_TOP10.md's ~4,000
    candidates/day universe makes `expected_precision = 10/4000 =
    0.0025`; a fixed floor like the old `0.15` is 61x that rate -- 1.5
    real hits/day worth of leakage would sail through undetected). The
    bound is the LARGER of two components, so neither a big universe
    (tiny expected_precision) nor a small sample (`n_observations`) can
    manufacture a false pass or a false alarm on its own:

    - `multiplier * expected_precision` (default 3x): precision more than
      a small multiple of the true random-guessing rate is inherently
      suspicious, and stays proportional as candidates/day changes --
      unlike a fixed floor, which becomes meaningless once the universe is
      large.
    - A one-sided binomial upper confidence bound (normal approximation,
      `z=3` ~ 99.9% one-sided) around `expected_precision` over
      `n_observations` independent Bernoulli(`expected_precision`) draws.
      This absorbs genuine sampling noise when `n_observations` is small
      (few trials, few days, or k close to the universe size) without
      needing a second magic constant -- the multiplier term alone would
      be too tight for small samples and too loose for huge ones.

    Result is clipped to `[expected_precision, 1.0]`.
    """
    if math.isnan(expected_precision) or expected_precision <= 0:
        return expected_precision

    multiplier_bound = multiplier * expected_precision
    if n_observations > 0:
        se = math.sqrt(expected_precision * (1 - expected_precision) / n_observations)
        ci_bound = expected_precision + z * se
    else:
        ci_bound = expected_precision

    bound = max(multiplier_bound, ci_bound)
    return float(min(1.0, max(expected_precision, bound)))


def shuffle_label_test(
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    k: int = 10,
    n_trials: int = 5,
    seed: int = 0,
    multiplier: float = 3.0,
    z: float = 3.0,
) -> dict:
    """Permute labels WITHIN each trade_date (preserving the "10 positives
    per day" structure -- a naive global shuffle would break that
    invariant), refit via `fit_predict_fn` on the SHUFFLED labels, then
    score the resulting predictions against the TRUE labels.

    Scoring against the true labels (not the shuffled ones) is the whole
    point of this test: a model trained on permuted, information-free
    labels should recover nothing about the true labels UNLESS some
    feature leaks the true label directly (e.g. a same-day return
    column) -- in which case the model ignores the noisy shuffled target
    it was nominally trained on and reconstructs the true ranking from
    the leaked feature anyway. Scoring against the shuffled labels
    instead (the old, broken behavior) makes a leaked model look exactly
    as "random" as a clean one, because both are being graded against
    noise.

    `fit_predict_fn(features, shuffled_labels) -> predictions` is injected
    so this harness has no model dependency (e.g. no lightgbm import).

    Returns {observed_precision, expected_precision, trials, passed,
    tolerance}. `passed` is True when the mean observed precision across
    trials, scored against the TRUE labels, does not exceed
    `_random_guess_tolerance(...)` -- see that function's docstring for
    why the bound is proportional to `expected_precision` rather than a
    fixed floor.
    """
    rng = np.random.default_rng(seed)

    candidates_per_day = features.groupby("trade_date")["ticker"].nunique()
    n_days = len(candidates_per_day)
    if n_days == 0:
        expected_precision = float("nan")
    else:
        avg_n = float(candidates_per_day.mean())
        expected_precision = min(1.0, k / avg_n) if avg_n > 0 else float("nan")

    trial_precisions = []
    for _ in range(n_trials):
        shuffled = labels.copy()
        for trade_date, group in shuffled.groupby("trade_date"):
            idx = group.index.to_numpy()
            permuted = rng.permutation(idx)
            for col in ("rank", "return_t", "label"):
                if col in shuffled.columns:
                    shuffled.loc[idx, col] = group.loc[permuted, col].to_numpy()

            # Invariant: the within-day shuffle must be a genuine
            # permutation of that day's own values -- same multiset in,
            # same multiset out. (The old check compared a count derived
            # from `group`, pre-shuffle, against a count derived from
            # `shuffled.loc[idx]` in the SAME iteration immediately after
            # `shuffled.loc[idx]` was assigned FROM a permutation of
            # `group`'s own rows -- the two are mathematically identical
            # by construction no matter what upstream bug existed, so
            # that comparison could never fail. Comparing full sorted
            # value lists actually can fail, e.g. if a future change
            # permuted against the wrong day's rows or left rows
            # unassigned.)
            for col in ("rank", "return_t", "label"):
                if col not in shuffled.columns:
                    continue
                original_values = np.sort(group[col].to_numpy(dtype=float))
                shuffled_values = np.sort(shuffled.loc[idx, col].to_numpy(dtype=float))
                if not np.array_equal(original_values, shuffled_values, equal_nan=True):
                    raise LeakageError(
                        "shuffle_label_test: within-day shuffle changed the "
                        f"multiset of {col!r} values for {trade_date!r}; shuffle "
                        "must be a permutation within the day, not a global "
                        "reshuffle or a partial/incorrect reassignment."
                    )

        predictions = fit_predict_fn(features, shuffled)
        # Score against the TRUE labels, not the shuffled ones -- see
        # docstring.
        trial_precisions.append(precision_at_k(predictions, labels, k=k))

    observed_precision = float(np.mean(trial_precisions))
    n_observations = n_trials * n_days * k
    tolerance = _random_guess_tolerance(expected_precision, n_observations, multiplier=multiplier, z=z)
    passed = not math.isnan(expected_precision) and observed_precision <= tolerance

    return {
        "observed_precision": observed_precision,
        "expected_precision": expected_precision,
        "tolerance": tolerance,
        "trials": trial_precisions,
        "passed": bool(passed),
    }
