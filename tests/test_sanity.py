from __future__ import annotations

import pandas as pd

from top10 import sanity


def _labels_row(trade_date, ticker, rank, return_t, label):
    return {
        "trade_date": pd.Timestamp(trade_date),
        "ticker": ticker,
        "rank": rank,
        "return_t": return_t,
        "label": label,
        "label_spec_version": "test-hash",
        "as_of": pd.Timestamp(trade_date),
    }


def _make_day_labels(trade_date, n_universe=20, top10_returns=None):
    """Build a synthetic labels DataFrame for one trading day with an
    n_universe-sized universe and exactly 10 positives at the given
    returns (defaulting to a plausible +20%..+60% spread)."""
    if top10_returns is None:
        top10_returns = [0.20 + 0.04 * i for i in range(10)]  # 20%..56%
    rows = []
    for i, ret in enumerate(top10_returns):
        rows.append(_labels_row(trade_date, f"T{i}", i + 1, ret, 1))
    for i in range(len(top10_returns), n_universe):
        rows.append(_labels_row(trade_date, f"T{i}", i + 1, -0.01 * i, 0))
    return pd.DataFrame(rows)


# --- check_universe_continuity ------------------------------------------------

def test_continuity_passes_with_smooth_drift():
    day1 = _make_day_labels("2024-03-14", n_universe=100)
    day2 = _make_day_labels("2024-03-15", n_universe=102)  # 2% drift
    labels = pd.concat([day1, day2], ignore_index=True)

    result = sanity.check_universe_continuity(labels)

    assert result.passed is True


def test_continuity_fails_on_step_change():
    day1 = _make_day_labels("2024-03-14", n_universe=100)
    day2 = _make_day_labels("2024-03-15", n_universe=50)  # 50% drop
    labels = pd.concat([day1, day2], ignore_index=True)

    result = sanity.check_universe_continuity(labels)

    assert result.passed is False
    assert "offenders" in result.details


# --- check_median_gainer_contamination / check_median_gainer_quiet_regime -----
# Defect 2: the P4 median-gainer tripwire must be two checks with different
# severities -- a hard FAIL on the upper (contamination) bound, and a soft
# WARN (never aborting) on the lower (quiet-market) bound.

def test_median_gainer_passes_in_plausible_range():
    labels = _make_day_labels("2024-03-15")

    contamination = sanity.check_median_gainer_contamination(labels)
    quiet = sanity.check_median_gainer_quiet_regime(labels)

    assert contamination.passed is True
    assert quiet.passed is True


def test_median_gainer_contamination_fails_at_plus_200_pct():
    """P4 tripwire: a median top-10 return of +200% indicates corporate
    action leakage. Must be a hard FAILURE (severity='fail')."""
    huge_returns = [2.0 + 0.1 * i for i in range(10)]  # ~200%+
    labels = _make_day_labels("2024-03-15", top10_returns=huge_returns)

    result = sanity.check_median_gainer_contamination(labels)

    assert result.passed is False
    assert result.severity == "fail"
    assert "leaking" in result.message


def test_median_gainer_quiet_regime_warns_not_fails_at_plus_14_pct():
    """Defect 2: an ordinary quiet-market median (e.g. +14%) must produce a
    WARNING, never a hard FAILURE -- and must never abort the pipeline."""
    quiet_returns = [0.10 + 0.01 * i for i in range(10)]  # 10%..19%, median ~14.5%
    labels = _make_day_labels("2024-03-15", top10_returns=quiet_returns)

    contamination = sanity.check_median_gainer_contamination(labels)
    quiet = sanity.check_median_gainer_quiet_regime(labels)

    assert contamination.passed is True  # never a hard failure on the low side
    assert quiet.passed is False
    assert quiet.severity == "warn"


def test_median_gainer_contamination_does_not_fire_on_quiet_day():
    """The upper (contamination) bound must never fire just because a day
    is quiet -- only the lower/warn check should react to a low median."""
    small_returns = [0.001 * i for i in range(10)]
    labels = _make_day_labels("2024-03-15", top10_returns=small_returns)

    result = sanity.check_median_gainer_contamination(labels)

    assert result.passed is True


# --- check_label_cardinality --------------------------------------------------

def test_cardinality_passes_with_exactly_ten():
    labels = _make_day_labels("2024-03-15", n_universe=50)
    result = sanity.check_label_cardinality(labels)
    assert result.passed is True


def test_cardinality_fails_with_wrong_count():
    labels = _make_day_labels("2024-03-15", n_universe=50)
    # Corrupt: flip one positive to zero, leaving only 9.
    labels.loc[labels["label"] == 1, "label"] = [1] * 9 + [0]

    result = sanity.check_label_cardinality(labels)

    assert result.passed is False


def test_cardinality_allows_fewer_than_ten_when_universe_small():
    rows = [_labels_row("2024-03-15", f"T{i}", i + 1, 0.1 * (5 - i), 1) for i in range(5)]
    labels = pd.DataFrame(rows)

    result = sanity.check_label_cardinality(labels)

    assert result.passed is True


# --- check_no_split_days -------------------------------------------------------

def test_no_split_days_passes_when_clean():
    labels = _make_day_labels("2024-03-15")
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )
    result = sanity.check_no_split_days(labels, corporate_actions)
    assert result.passed is True


def test_no_split_days_fails_when_labeled_ticker_has_split_same_day():
    labels = _make_day_labels("2024-03-15")
    corporate_actions = pd.DataFrame(
        [
            {
                "ex_date": pd.Timestamp("2024-03-15"),
                "ticker": "T0",  # T0 is one of the labeled tickers above
                "action_type": "reverse_split",
                "ratio": 20.0,
                "cash_amount": None,
                "new_ticker": None,
                "as_of": pd.Timestamp("2024-03-14"),
            }
        ]
    )

    result = sanity.check_no_split_days(labels, corporate_actions)

    assert result.passed is False


# --- check_universe_coverage ---------------------------------------------------
# Defect 3: a universe-to-labels drop must be counted and reported, not
# silently absorbed -- and an anomalous spike must be flagged.

def _make_universe(trade_date, tickers):
    return pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date),
            "ticker": tickers,
            "security_type": "CS",
            "exchange": "XNYS",
            "flagged_type": False,
            "prior_close": 10.0,
            "avg_dollar_volume_20d": 5_000_000.0,
            "as_of": pd.Timestamp(trade_date) - pd.Timedelta(hours=8),
        }
    )


def test_universe_coverage_passes_with_small_ordinary_drop():
    tickers = [f"T{i}" for i in range(100)]
    universe = _make_universe("2024-03-15", tickers)
    # 1 halted name dropped out of 100 (1% < 5% threshold) -- ordinary.
    labels = _make_day_labels("2024-03-15", n_universe=99)

    result = sanity.check_universe_coverage(labels, universe)

    assert result.passed is True


def test_universe_coverage_fails_on_anomalous_spike():
    tickers = [f"T{i}" for i in range(100)]
    universe = _make_universe("2024-03-15", tickers)
    # 40 names missing from labels out of 100 (40% >> 5% threshold) --
    # a data-outage signature, not ordinary halts.
    labels = _make_day_labels("2024-03-15", n_universe=60)

    result = sanity.check_universe_coverage(labels, universe)

    assert result.passed is False
    assert "offenders" in result.details
    assert result.severity == "fail"


# --- run_all / SanityReport ----------------------------------------------------

def test_run_all_passes_for_clean_data():
    labels = _make_day_labels("2024-03-15")
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    report = sanity.run_all(labels, corporate_actions)

    assert report.passed is True
    assert report.failures == []
    assert "ALL PASSED" in str(report)


def test_run_all_reports_failures():
    huge_returns = [2.0 + 0.1 * i for i in range(10)]
    labels = _make_day_labels("2024-03-15", top10_returns=huge_returns)
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    report = sanity.run_all(labels, corporate_actions)

    assert report.passed is False
    assert any(f.name == "median_gainer_contamination" for f in report.failures)
    assert "FAILED" in str(report)


def test_run_all_quiet_regime_warns_but_still_passes():
    """Defect 2: a run_all report for an ordinary quiet day must PASS
    (report.passed True) while still surfacing the low median as a
    warning -- this is the core regression: constant spurious aborts on
    quiet days must stop."""
    quiet_returns = [0.10 + 0.01 * i for i in range(10)]  # median ~14.5%
    labels = _make_day_labels("2024-03-15", top10_returns=quiet_returns)
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    report = sanity.run_all(labels, corporate_actions)

    assert report.passed is True
    assert report.failures == []
    assert any(w.name == "median_gainer_quiet_regime" for w in report.warnings)


def test_run_all_wires_universe_coverage_when_universe_supplied():
    tickers = [f"T{i}" for i in range(100)]
    universe = _make_universe("2024-03-15", tickers)
    labels = _make_day_labels("2024-03-15", n_universe=60)  # anomalous drop
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    report = sanity.run_all(labels, corporate_actions, universe=universe)

    assert report.passed is False
    assert any(f.name == "universe_coverage" for f in report.failures)


def test_cli_exits_nonzero_on_failure(tmp_path):
    huge_returns = [2.0 + 0.1 * i for i in range(10)]
    labels = _make_day_labels("2024-03-15", top10_returns=huge_returns)

    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path)

    exit_code = sanity._main(["--labels", str(labels_path)])

    assert exit_code != 0


def test_cli_exits_zero_on_pass(tmp_path):
    labels = _make_day_labels("2024-03-15")

    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path)

    exit_code = sanity._main(["--labels", str(labels_path)])

    assert exit_code == 0
