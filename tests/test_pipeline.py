"""Tests for top10/pipeline.py.

Offline, synthetic, no network -- vendor access is mocked at the
`top10.data.get_source` boundary via monkeypatch.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import top10.data as data_pkg
import top10.pipeline as pipeline
from top10.features.spec import T1_SPEC, write_features
from top10.pipeline import (
    PipelineAbort,
    build_features_step,
    build_labels_step,
    ingest,
    run_all,
    run_sanity_step,
)
from top10.storage import LeakageError


# --- fixtures ------------------------------------------------------------


class _FakeSource:
    """Minimal MarketDataSource stand-in with call counters so tests can
    assert the vendor was (or wasn't) hit."""

    name = "fake"

    def __init__(self):
        self.calls = {"daily_bars": 0, "corporate_actions": 0, "ticker_meta": 0, "earnings": 0, "short_interest": 0}

    def daily_bars(self, start, end):
        self.calls["daily_bars"] += 1
        dates = pd.bdate_range(start, end)
        rows = []
        for day_idx, d in enumerate(dates):
            for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
                # Slight day/ticker-varying drift -- a dead-flat price series
                # (0.0 return_t every single day) is unrealistic and prone to
                # spuriously colliding with any legitimately-all-zero feature
                # under assert_self_exclusion's exact-identity layer.
                close = (10.0 + i) * (1 + 0.001 * (i + 1) * (day_idx + 1))
                rows.append(
                    {
                        "trade_date": pd.Timestamp(d),
                        "ticker": ticker,
                        "open": 10.0 + i,
                        "high": 11.0 + i,
                        "low": 9.0 + i,
                        "close": close,
                        "volume": 1_000_000.0,
                        "dollar_volume": 10_000_000.0,
                        "as_of": pd.Timestamp(d),
                    }
                )
        return pd.DataFrame(rows)

    def corporate_actions(self, start, end):
        self.calls["corporate_actions"] += 1
        return pd.DataFrame(columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"])

    def ticker_meta(self, start, end):
        self.calls["ticker_meta"] += 1
        rows = []
        for ticker in ["AAA", "BBB", "CCC"]:
            rows.append(
                {
                    "ticker": ticker,
                    "name": ticker,
                    "security_type": "CS",
                    "exchange": "XNYS",
                    "active_from": pd.Timestamp("2000-01-01"),
                    "active_to": pd.NaT,
                    # Defect 6: market_cap/float_shares (ticker-meta contract)
                    # and short_interest_pct_float/days_to_cover (separate
                    # short_interest() source) must all be present -- a
                    # missing column now raises loudly in build_t1_features.
                    "market_cap": 5e9,
                    "float_shares": 1e8,
                    "as_of": pd.Timestamp("2000-01-01"),
                }
            )
        return pd.DataFrame(rows)

    def earnings(self, start, end):
        self.calls["earnings"] += 1
        return pd.DataFrame(columns=["ticker", "report_date", "session", "announced_on", "date_is_revisable", "as_of"])

    def short_interest(self, start, end):
        self.calls["short_interest"] += 1
        from top10.data.base import SHORT_INTEREST_COLUMNS

        return pd.DataFrame(columns=list(SHORT_INTEREST_COLUMNS))


@pytest.fixture
def fake_source(monkeypatch):
    source = _FakeSource()
    monkeypatch.setattr(data_pkg, "get_source", lambda vendor: source, raising=False)
    return source


_LARGE_UNIVERSE_TICKERS = [f"T{i:02d}" for i in range(20)]
_LARGE_UNIVERSE_SECTORS = ["Technology", "Healthcare", "Energy", "Financials"]


class _LargeFakeSource:
    """Wider (20-ticker), varied-return/varied-sector MarketDataSource
    stand-in for `run_all` tests -- `_FakeSource`'s tiny 3-ticker universe
    (where every ticker is trivially top-10) makes EVERY row `label=1` and
    every dummy feature constant, which spuriously collides with
    `assert_self_exclusion`'s exact-identity layer by pure coincidence,
    not because of a real leak. A wider universe with a genuine 10/20
    label split and mixed sectors is what real production usage (and this
    check) actually looks like."""

    name = "fake-large"

    def __init__(self):
        self.calls = {"daily_bars": 0, "corporate_actions": 0, "ticker_meta": 0, "earnings": 0, "short_interest": 0}

    def daily_bars(self, start, end):
        self.calls["daily_bars"] += 1
        dates = pd.bdate_range(start, end)
        rows = []
        # Independent per-ticker random walk (fixed seed for determinism) --
        # a price function that is monotonic in ticker index on EVERY day
        # (e.g. `close = f(i) * g(day)`) makes yesterday's cross-sectional
        # rank (the `ret_1d_rank` feature, computed at the T1 decision time)
        # identical to today's cross-sectional rank (the label's `rank`)
        # every single day -- an artifact of the synthetic price function,
        # not a real leak, but indistinguishable from one to
        # `assert_self_exclusion`'s exact-identity layer. A genuine random
        # walk breaks that day-to-day rank coincidence.
        rng = np.random.default_rng(1234)
        closes = {t: 10.0 + i for i, t in enumerate(_LARGE_UNIVERSE_TICKERS)}
        for d in dates:
            for ticker in _LARGE_UNIVERSE_TICKERS:
                closes[ticker] = max(1.0, closes[ticker] * (1 + rng.normal(0, 0.02)))
                close = closes[ticker]
                rows.append(
                    {
                        "trade_date": pd.Timestamp(d),
                        "ticker": ticker,
                        "open": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 1_000_000.0,
                        "dollar_volume": 10_000_000.0,
                        "as_of": pd.Timestamp(d),
                    }
                )
        return pd.DataFrame(rows)

    def corporate_actions(self, start, end):
        self.calls["corporate_actions"] += 1
        return pd.DataFrame(columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"])

    def ticker_meta(self, start, end):
        self.calls["ticker_meta"] += 1
        rows = []
        for i, ticker in enumerate(_LARGE_UNIVERSE_TICKERS):
            rows.append(
                {
                    "ticker": ticker,
                    "name": ticker,
                    "security_type": "CS",
                    "exchange": "XNYS",
                    "active_from": pd.Timestamp("2000-01-01"),
                    "active_to": pd.NaT,
                    "market_cap": 5e9 + i * 1e8,
                    "float_shares": 1e8,
                    "as_of": pd.Timestamp("2000-01-01"),
                    "sector": _LARGE_UNIVERSE_SECTORS[i % len(_LARGE_UNIVERSE_SECTORS)],
                    "industry": "Misc",
                }
            )
        return pd.DataFrame(rows)

    def earnings(self, start, end):
        self.calls["earnings"] += 1
        return pd.DataFrame(columns=["ticker", "report_date", "session", "announced_on", "date_is_revisable", "as_of"])

    def short_interest(self, start, end):
        self.calls["short_interest"] += 1
        from top10.data.base import SHORT_INTEREST_COLUMNS

        return pd.DataFrame(columns=list(SHORT_INTEREST_COLUMNS))


@pytest.fixture
def large_fake_source(monkeypatch):
    source = _LargeFakeSource()
    monkeypatch.setattr(data_pkg, "get_source", lambda vendor: source, raising=False)
    return source


@pytest.fixture
def isolated_data_dirs(tmp_path, monkeypatch):
    """Point every DATA_* path pipeline touches at a tmp_path sandbox."""
    import top10.pipeline as pipeline_mod

    pit_dir = tmp_path / "pit"
    monkeypatch.setattr(pipeline_mod, "DATA_PIT", pit_dir)

    import top10.labels as labels_mod

    labels_dir = tmp_path / "labels"
    monkeypatch.setattr(labels_mod, "DATA_LABELS", labels_dir)

    import top10.features.spec as spec_mod

    features_dir = tmp_path / "features"
    monkeypatch.setattr(spec_mod, "DATA_FEATURES", features_dir)

    return tmp_path


# --- ingest ------------------------------------------------------------


def test_ingest_fetches_from_vendor_on_first_call(fake_source, isolated_data_dirs):
    frames = ingest("fake", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert fake_source.calls["daily_bars"] == 1
    assert not frames["daily_bars"].empty


def test_ingest_is_resumable_skips_vendor_on_second_call(fake_source, isolated_data_dirs):
    ingest("fake", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert fake_source.calls["daily_bars"] == 1

    ingest("fake", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    # Same exact range -> should be read back from the PIT cache, not refetched.
    assert fake_source.calls["daily_bars"] == 1


def test_ingest_different_range_refetches(fake_source, isolated_data_dirs):
    ingest("fake", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    ingest("fake", dt.date(2024, 2, 1), dt.date(2024, 2, 5))
    assert fake_source.calls["daily_bars"] == 2


# --- Defect 3: short_interest ingest path ------------------------------------


def test_ingest_fetches_and_persists_short_interest(fake_source, isolated_data_dirs):
    frames = ingest("fake", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert fake_source.calls["short_interest"] == 1
    assert "short_interest" in frames

    pit_path = pipeline._pit_path(
        "short_interest", "fake", dt.date(2024, 1, 2), dt.date(2024, 1, 5)
    )
    assert pit_path.exists()


def test_ingest_degrades_short_interest_on_not_implemented(isolated_data_dirs, monkeypatch):
    """A plan-limited vendor key (or an adapter without a short_interest
    feed) must degrade ONE feature family -- ingest must not abort the
    whole pull."""
    from top10.data.base import SHORT_INTEREST_COLUMNS

    class _NoShortInterestSource(_FakeSource):
        def short_interest(self, start, end):
            self.calls["short_interest"] += 1
            raise NotImplementedError("no short-interest feed for this vendor")

    source = _NoShortInterestSource()
    monkeypatch.setattr(data_pkg, "get_source", lambda vendor: source, raising=False)

    frames = ingest("fake", dt.date(2024, 1, 2), dt.date(2024, 1, 5))

    # Ingest completed for every OTHER dataset -- it did not abort.
    assert not frames["daily_bars"].empty
    # short_interest degrades to an empty, correctly-shaped frame.
    assert frames["short_interest"].empty
    assert list(frames["short_interest"].columns) == list(SHORT_INTEREST_COLUMNS)


# --- build_labels_step ------------------------------------------------------


def test_build_labels_step_produces_labels(fake_source, isolated_data_dirs):
    frames = ingest("fake", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    labels = build_labels_step(frames, dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert not labels.empty
    assert set(labels.columns) >= {"trade_date", "ticker", "rank", "return_t", "label"}


# --- run_sanity_step: ABORT, not warn ---------------------------------------


def _bad_labels() -> pd.DataFrame:
    """Labels whose median top-10 return is wildly outside the plausible
    [20%, 60%] band -- a P4 tripwire violation that must ABORT."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "trade_date": pd.Timestamp("2024-01-02"),
                "ticker": f"T{i}",
                "rank": i + 1,
                "return_t": 5.00,  # +500%, implausible
                "label": 1,
                "label_spec_version": "test-hash",
                "as_of": pd.Timestamp("2024-01-02"),
            }
        )
    return pd.DataFrame(rows)


def test_run_sanity_step_aborts_pipeline_on_failure():
    bad_labels = _bad_labels()
    corporate_actions = pd.DataFrame(columns=["ticker", "ex_date", "action_type"])

    with pytest.raises(PipelineAbort):
        run_sanity_step(bad_labels, corporate_actions)


def test_run_sanity_step_does_not_merely_warn(caplog):
    """A sanity failure must raise, not just log a warning and continue --
    assert the exception actually propagates (proven by pytest.raises
    above) AND that no return value is produced on failure."""
    bad_labels = _bad_labels()
    corporate_actions = pd.DataFrame(columns=["ticker", "ex_date", "action_type"])

    result = None
    raised = False
    try:
        result = run_sanity_step(bad_labels, corporate_actions)
    except PipelineAbort:
        raised = True

    assert raised is True
    assert result is None


def test_run_sanity_step_passes_on_good_labels():
    rows = []
    for i in range(10):
        rows.append(
            {
                "trade_date": pd.Timestamp("2024-01-02"),
                "ticker": f"T{i}",
                "rank": i + 1,
                "return_t": 0.20 + 0.04 * i,
                "label": 1,
                "label_spec_version": "test-hash",
                "as_of": pd.Timestamp("2024-01-02"),
            }
        )
    labels = pd.DataFrame(rows)
    corporate_actions = pd.DataFrame(columns=["ticker", "ex_date", "action_type"])
    report = run_sanity_step(labels, corporate_actions)
    assert report.passed


# --- build_features_step: idempotent ----------------------------------------


def test_build_features_step_is_idempotent(isolated_data_dirs, monkeypatch):
    calls = {"t1": 0}

    def _fake_build_t1(daily_bars, ticker_meta, earnings, labels_history, market_context, trade_date, short_interest=None):
        calls["t1"] += 1
        from top10.features.spec import T1_COLUMNS
        from top10.features.t1 import decision_time_t1

        # as_of must be decision-time-safe: write_features (via
        # assert_decision_time_safe) now refuses to persist a frame whose
        # as_of is after its own task's decision time.
        as_of = decision_time_t1(trade_date)
        return pd.DataFrame(
            [
                {
                    c: (trade_date if c == "trade_date" else ("AAA" if c == "ticker" else (as_of if c == "as_of" else 0.0)))
                    for c in T1_COLUMNS
                }
            ]
        )

    import top10.features.t1 as t1_mod

    monkeypatch.setattr(t1_mod, "build_t1_features", _fake_build_t1)

    daily_bars = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")],
            "ticker": ["AAA"],
            "as_of": [pd.Timestamp("2024-01-02")],
        }
    )

    build_features_step("T1", dt.date(2024, 1, 2), dt.date(2024, 1, 2), daily_bars=daily_bars)
    assert calls["t1"] == 1

    build_features_step("T1", dt.date(2024, 1, 2), dt.date(2024, 1, 2), daily_bars=daily_bars)
    assert calls["t1"] == 1  # second call must skip, not rebuild


# --- build_features_step: short_interest threaded through to build_t1_features -----


def test_build_features_step_threads_short_interest_to_build_t1_features(isolated_data_dirs, monkeypatch):
    from top10.data.base import SHORT_INTEREST_COLUMNS

    received = {}

    def _fake_build_t1(daily_bars, ticker_meta, earnings, labels_history, market_context, trade_date, short_interest=None):
        received["short_interest"] = short_interest
        from top10.features.spec import T1_COLUMNS
        from top10.features.t1 import decision_time_t1

        as_of = decision_time_t1(trade_date)
        return pd.DataFrame(
            [
                {
                    c: (trade_date if c == "trade_date" else ("AAA" if c == "ticker" else (as_of if c == "as_of" else 0.0)))
                    for c in T1_COLUMNS
                }
            ]
        )

    import top10.features.t1 as t1_mod

    monkeypatch.setattr(t1_mod, "build_t1_features", _fake_build_t1)

    daily_bars = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")],
            "ticker": ["AAA"],
            "as_of": [pd.Timestamp("2024-01-02")],
        }
    )
    short_interest = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "settlement_date": pd.Timestamp("2024-01-01"),
                "short_interest_shares": 1000.0,
                "short_interest_pct_float": 5.0,
                "days_to_cover": 1.5,
                "as_of": pd.Timestamp("2024-01-01"),
            }
        ]
    )

    build_features_step(
        "T1", dt.date(2024, 1, 2), dt.date(2024, 1, 2),
        daily_bars=daily_bars, short_interest=short_interest,
    )

    assert received["short_interest"] is short_interest


def test_build_features_step_defaults_short_interest_to_empty_correct_shape(isolated_data_dirs, monkeypatch):
    from top10.data.base import SHORT_INTEREST_COLUMNS

    received = {}

    def _fake_build_t1(daily_bars, ticker_meta, earnings, labels_history, market_context, trade_date, short_interest=None):
        received["short_interest"] = short_interest
        from top10.features.spec import T1_COLUMNS
        from top10.features.t1 import decision_time_t1

        as_of = decision_time_t1(trade_date)
        return pd.DataFrame(
            [
                {
                    c: (trade_date if c == "trade_date" else ("AAA" if c == "ticker" else (as_of if c == "as_of" else 0.0)))
                    for c in T1_COLUMNS
                }
            ]
        )

    import top10.features.t1 as t1_mod

    monkeypatch.setattr(t1_mod, "build_t1_features", _fake_build_t1)

    daily_bars = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")],
            "ticker": ["AAA"],
            "as_of": [pd.Timestamp("2024-01-02")],
        }
    )

    build_features_step("T1", dt.date(2024, 1, 2), dt.date(2024, 1, 2), daily_bars=daily_bars)

    assert received["short_interest"].empty
    assert list(received["short_interest"].columns) == list(SHORT_INTEREST_COLUMNS)


# --- build_features_step: T2 orchestration path (Defect 2) -------------------------


def test_build_features_step_t2_end_to_end(isolated_data_dirs):
    """Defect 2 (CONFIRMED): `build_features_step("T2", ...)` was the ONLY
    T2 orchestration path and crashed with `KeyError: 'prior_close'` on
    every real call -- 232 tests passed because every T2 test called
    `build_t2_features` directly with a hand-built frame. This exercises
    the actual pipeline entry point end to end."""
    trade_date = pd.Timestamp("2024-01-10")
    prior_date = trade_date - pd.Timedelta(days=1)

    daily_bars = pd.DataFrame(
        [
            {
                "trade_date": prior_date,
                "ticker": "AAA",
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "volume": 1_000_000.0,
                "dollar_volume": 10_000_000.0,
                "as_of": prior_date,
            },
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "open": 11.0,
                "high": 11.5,
                "low": 10.5,
                "close": 11.0,
                "volume": 1_000_000.0,
                "dollar_volume": 11_000_000.0,
                "as_of": trade_date,
            },
        ]
    )
    ticker_meta = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 5e9,
                "float_shares": 1e8,
                "as_of": pd.Timestamp("2020-01-01"),
            }
        ]
    )
    earnings = pd.DataFrame(columns=["ticker", "report_date", "session", "announced_on", "date_is_revisable", "as_of"])
    labels_history = pd.DataFrame(columns=["trade_date", "ticker", "rank", "return_t", "label", "label_spec_version", "as_of"])

    premarket_bars = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "minute": trade_date + pd.Timedelta(hours=9),
                "open": 11.5,
                "high": 11.6,
                "low": 11.4,
                "close": 11.5,
                "volume": 5000.0,
                "trade_count": 20,
                "as_of": trade_date + pd.Timedelta(hours=9),
            }
        ]
    )
    prior_close = pd.DataFrame(
        [{"trade_date": trade_date, "ticker": "AAA", "close": 10.0, "as_of": prior_date}]
    )
    halts = pd.DataFrame(columns=["ticker", "trade_date", "as_of"])

    out = build_features_step(
        "T2",
        trade_date.date(),
        trade_date.date(),
        daily_bars=daily_bars,
        ticker_meta=ticker_meta,
        earnings=earnings,
        labels_history=labels_history,
        premarket_bars=premarket_bars,
        prior_close=prior_close,
        halts=halts,
    )

    assert not out.empty
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["premarket_gap_pct"] == pytest.approx(11.5 / 10.0 - 1.0)


# --- write_features: refuses a leaked frame (Defect 1 / TOP FINDING) ---------------


def test_write_features_refuses_leaked_frame(isolated_data_dirs):
    from top10.features.t1 import decision_time_t1

    trade_date = pd.Timestamp("2024-01-10")
    leaked_as_of = decision_time_t1(trade_date) + pd.Timedelta(hours=1)  # AFTER decision time
    df = pd.DataFrame(
        [
            {
                **{c: 0.0 for c in T1_SPEC.columns},
                "trade_date": trade_date,
                "ticker": "AAA",
                "as_of": leaked_as_of,
            }
        ]
    )[list(T1_SPEC.columns)]

    with pytest.raises(LeakageError):
        write_features(df, T1_SPEC, trade_date)


# --- run_all: shuffle_label_test gate (TOP FINDING) --------------------------------


class _FakeWalkforwardModel:
    """Minimal duck-typed model -- no lightgbm dependency."""

    def fit(self, features, labels, *, unseal_token=None):
        return self

    def predict(self, features):
        return pd.DataFrame(
            {"trade_date": features["trade_date"], "ticker": features["ticker"], "score": 0.0}
        )


def test_run_all_aborts_on_shuffle_label_test_failure(large_fake_source, isolated_data_dirs, monkeypatch):
    import top10.pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod,
        "shuffle_label_test",
        lambda *a, **k: {"observed_precision": 1.0, "expected_precision": 0.01, "tolerance": 0.05, "trials": [1.0], "passed": False},
    )

    with pytest.raises(PipelineAbort):
        run_all("fake-large", dt.date(2022, 1, 3), dt.date(2022, 1, 6), model_factory=lambda: _FakeWalkforwardModel())


def test_run_all_shuffle_label_test_gate_runs_before_walkforward_when_passing(large_fake_source, isolated_data_dirs, monkeypatch):
    import top10.pipeline as pipeline_mod

    calls = {"shuffle": 0, "walkforward": 0}

    def _fake_shuffle(*a, **k):
        calls["shuffle"] += 1
        return {"observed_precision": 0.0, "expected_precision": 0.01, "tolerance": 0.05, "trials": [0.0], "passed": True}

    def _fake_run_walkforward_step(*a, **k):
        calls["walkforward"] += 1
        return "ok"

    monkeypatch.setattr(pipeline_mod, "shuffle_label_test", _fake_shuffle)
    monkeypatch.setattr(pipeline_mod, "run_walkforward_step", _fake_run_walkforward_step)

    result = run_all("fake-large", dt.date(2022, 1, 3), dt.date(2022, 1, 6), model_factory=lambda: _FakeWalkforwardModel())

    assert calls["shuffle"] == 1
    assert calls["walkforward"] == 1
    assert result["shuffle_label_test"]["passed"] is True
    assert result["walkforward"] == "ok"


# --- back-adjusted price detection at ingest ---------------------------------
# Regression for the adversarial audit's TOP FINDING: `assert_no_adjusted_prices`
# and `verify_unadjusted` existed but had ZERO production call sites. Ingest is
# the only layer where the raw vendor feed is still visible.


def _adjusted_feed_frames():
    """A back-adjusted feed: a 1:10 split on 2020-06-01, and the pre-split
    bars have ALREADY been divided through by 10, so the series shows no
    discontinuity at the ex_date. An unadjusted feed must show a ~10x drop.
    """
    dates = pd.date_range("2020-05-25", "2020-06-05", freq="B")
    bars = pd.DataFrame({
        "trade_date": dates,
        "ticker": "AAA",
        "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
        "volume": 1_000_000.0, "dollar_volume": 10_000_000.0,
        "as_of": dates + pd.Timedelta(hours=16),
    })
    actions = pd.DataFrame({
        "ex_date": [pd.Timestamp("2020-06-01")],
        "ticker": ["AAA"],
        "action_type": ["split"],
        "ratio": [10.0],
        "cash_amount": [float("nan")],
        "new_ticker": [None],
        "as_of": [pd.Timestamp("2020-05-20")],
    })
    return bars, actions


def test_ingest_aborts_on_back_adjusted_prices(tmp_path, monkeypatch):
    monkeypatch.setattr("top10.pipeline.DATA_PIT", tmp_path)
    bars, actions = _adjusted_feed_frames()

    class _FakeSource:
        name = "fake"
        def daily_bars(self, start, end): return bars
        def corporate_actions(self, start, end): return actions
        def ticker_meta(self, start, end):
            return pd.DataFrame(columns=["ticker", "as_of"])
        def earnings(self, start, end):
            return pd.DataFrame(columns=["ticker", "as_of"])
        def short_interest(self, start, end):
            from top10.data.base import SHORT_INTEREST_COLUMNS
            return pd.DataFrame(columns=list(SHORT_INTEREST_COLUMNS))

    monkeypatch.setattr("top10.data.get_source", lambda vendor=None: _FakeSource())

    with pytest.raises(PipelineAbort, match="BACK-ADJUSTED"):
        pipeline.ingest("fake", dt.date(2020, 5, 25), dt.date(2020, 6, 5))
