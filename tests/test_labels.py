from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from top10 import labels as labels_mod
from top10.hashing import hash_file
from top10.config import DOCS


# --- Synthetic fixture builders ----------------------------------------------

def _bar(ticker, trade_date, close, dollar_volume, as_of=None, o=None, h=None, l=None, v=None):
    trade_date = pd.Timestamp(trade_date)
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "open": o if o is not None else close,
        "high": h if h is not None else close,
        "low": l if l is not None else close,
        "close": close,
        "volume": v if v is not None else dollar_volume / max(close, 0.01),
        "dollar_volume": dollar_volume,
        "as_of": pd.Timestamp(as_of) if as_of is not None else trade_date,
    }


def _meta(
    ticker,
    security_type="CS",
    exchange="XNYS",
    active_from="2000-01-01",
    active_to=None,
    as_of="1999-01-01",
):
    return {
        "ticker": ticker,
        "name": ticker,
        "security_type": security_type,
        "exchange": exchange,
        "active_from": pd.Timestamp(active_from),
        "active_to": pd.Timestamp(active_to) if active_to is not None else pd.NaT,
        "as_of": pd.Timestamp(as_of),
    }


def _ca(ticker, ex_date, action_type, ratio=None, cash_amount=None, new_ticker=None, as_of=None):
    ex_date = pd.Timestamp(ex_date)
    return {
        "ex_date": ex_date,
        "ticker": ticker,
        "action_type": action_type,
        "ratio": ratio,
        "cash_amount": cash_amount,
        "new_ticker": new_ticker,
        "as_of": pd.Timestamp(as_of) if as_of is not None else ex_date - pd.Timedelta(days=1),
    }


TRADE_DATE = pd.Timestamp("2024-03-15")
PRIOR_DATE = pd.Timestamp("2024-03-14")


def _make_20_days_of_bars(ticker, prior_close, dollar_volume, end_date=PRIOR_DATE):
    """20 trading days of history ending at `end_date` (inclusive)."""
    rows = []
    dates = pd.bdate_range(end=end_date, periods=20)
    for d in dates:
        close = prior_close if d == end_date else prior_close * 0.99
        rows.append(_bar(ticker, d, close, dollar_volume))
    return rows


@pytest.fixture
def base_universe_inputs():
    """A clean, well-formed universe of 3 eligible common stocks."""
    bars = []
    for ticker, close, dvol in [("AAA", 10.0, 5_000_000), ("BBB", 20.0, 2_000_000), ("CCC", 5.0, 1_500_000)]:
        bars.extend(_make_20_days_of_bars(ticker, close, dvol))
    # Today's bars (t) -- close_t values chosen later per test.
    daily_bars = pd.DataFrame(bars)
    ticker_meta = pd.DataFrame(
        [_meta("AAA"), _meta("BBB"), _meta("CCC")]
    )
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )
    return daily_bars, ticker_meta, corporate_actions


# --- Universe filter tests ----------------------------------------------------

def test_price_filter_excludes_under_a_dollar(base_universe_inputs):
    daily_bars, ticker_meta, corporate_actions = base_universe_inputs
    bars = pd.concat(
        [daily_bars, pd.DataFrame(_make_20_days_of_bars("PENNY", 0.50, 5_000_000))],
        ignore_index=True,
    )
    meta = pd.concat([ticker_meta, pd.DataFrame([_meta("PENNY")])], ignore_index=True)

    universe = labels_mod.build_universe(bars, meta, corporate_actions, TRADE_DATE)

    assert "PENNY" not in set(universe["ticker"])
    assert "AAA" in set(universe["ticker"])


def test_dollar_volume_filter_excludes_illiquid(base_universe_inputs):
    daily_bars, ticker_meta, corporate_actions = base_universe_inputs
    bars = pd.concat(
        [daily_bars, pd.DataFrame(_make_20_days_of_bars("ILLIQ", 10.0, 100_000))],
        ignore_index=True,
    )
    meta = pd.concat([ticker_meta, pd.DataFrame([_meta("ILLIQ")])], ignore_index=True)

    universe = labels_mod.build_universe(bars, meta, corporate_actions, TRADE_DATE)

    assert "ILLIQ" not in set(universe["ticker"])


def test_otc_excluded(base_universe_inputs):
    daily_bars, ticker_meta, corporate_actions = base_universe_inputs
    bars = pd.concat(
        [daily_bars, pd.DataFrame(_make_20_days_of_bars("OTCCO", 10.0, 5_000_000))],
        ignore_index=True,
    )
    meta = pd.concat(
        [ticker_meta, pd.DataFrame([_meta("OTCCO", exchange="OTC")])], ignore_index=True
    )

    universe = labels_mod.build_universe(bars, meta, corporate_actions, TRADE_DATE)

    assert "OTCCO" not in set(universe["ticker"])


def test_etf_is_flagged_not_dropped(base_universe_inputs):
    daily_bars, ticker_meta, corporate_actions = base_universe_inputs
    bars = pd.concat(
        [daily_bars, pd.DataFrame(_make_20_days_of_bars("SPYX", 400.0, 50_000_000))],
        ignore_index=True,
    )
    meta = pd.concat(
        [ticker_meta, pd.DataFrame([_meta("SPYX", security_type="ETF")])], ignore_index=True
    )

    universe = labels_mod.build_universe(bars, meta, corporate_actions, TRADE_DATE)

    row = universe[universe["ticker"] == "SPYX"]
    assert not row.empty, "ETF must be evaluated, not auto-dropped"
    assert bool(row.iloc[0]["flagged_type"]) is True

    # Common stocks must NOT be flagged.
    aaa_row = universe[universe["ticker"] == "AAA"]
    assert bool(aaa_row.iloc[0]["flagged_type"]) is False


def test_later_delisted_ticker_still_in_earlier_universe(base_universe_inputs):
    """P2 regression: a ticker delisted in 2019 must still appear in an
    earlier-date (e.g. 2017) universe."""
    ticker = "DELIST"
    early_trade_date = pd.Timestamp("2017-06-01")
    bars = pd.DataFrame(
        _make_20_days_of_bars(ticker, 10.0, 5_000_000, end_date=pd.Timestamp("2017-05-31"))
    )
    meta = pd.DataFrame(
        [
            _meta(
                ticker,
                active_from="2010-01-01",
                active_to="2019-08-01",  # delisted LATER than the trade date under test
                as_of="2009-01-01",
            )
        ]
    )
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    universe = labels_mod.build_universe(bars, meta, corporate_actions, early_trade_date)

    assert ticker in set(universe["ticker"])


def test_delisted_before_trade_date_excluded(base_universe_inputs):
    ticker = "GONE"
    bars = pd.DataFrame(_make_20_days_of_bars(ticker, 10.0, 5_000_000))
    meta = pd.DataFrame(
        [_meta(ticker, active_from="2010-01-01", active_to="2020-01-01", as_of="2009-01-01")]
    )
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    # trade_date is AFTER active_to -> must be excluded.
    universe = labels_mod.build_universe(bars, meta, corporate_actions, pd.Timestamp("2024-03-15"))

    assert ticker not in set(universe["ticker"])


# --- Label / reverse-split trap tests ----------------------------------------

def _universe_and_bars_for_label_test():
    """3 candidates in-universe; today's closes chosen so ranking is clear."""
    daily_bars = []
    ticker_metas = []
    for ticker, prior_close in [("AAA", 10.0), ("BBB", 20.0), ("CCC", 5.0), ("DDD", 8.0)]:
        daily_bars.extend(_make_20_days_of_bars(ticker, prior_close, 5_000_000))
        ticker_metas.append(_meta(ticker))

    ticker_meta = pd.DataFrame(ticker_metas)
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )
    daily_bars_df = pd.DataFrame(daily_bars)
    universe = labels_mod.build_universe(daily_bars_df, ticker_meta, corporate_actions, TRADE_DATE)
    return universe, daily_bars_df


def test_reverse_split_trap_excluded_from_labels():
    """A ticker with a 1:20 reverse split on the trade date whose raw
    close-over-close is +1900% MUST be excluded from labels."""
    universe, daily_bars_df = _universe_and_bars_for_label_test()

    # REVSPL: prior_close (unadjusted, pre-split) = 1.0; today's raw close
    # after a 1:20 reverse split = 20.0 -> raw return = +1900%.
    revspl_history = _make_20_days_of_bars("REVSPL", 1.0, 5_000_000)
    revspl_meta = _meta("REVSPL")
    universe2 = labels_mod.build_universe(
        pd.concat([daily_bars_df, pd.DataFrame(revspl_history)], ignore_index=True),
        pd.concat([pd.DataFrame([revspl_meta]), pd.DataFrame([_meta(t) for t in universe["ticker"]])], ignore_index=True),
        pd.DataFrame(columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]),
        TRADE_DATE,
    )
    assert "REVSPL" in set(universe2["ticker"])

    today_rows = []
    for ticker in universe2["ticker"]:
        prior_close = universe2[universe2["ticker"] == ticker].iloc[0]["prior_close"]
        if ticker == "REVSPL":
            today_rows.append(_bar(ticker, TRADE_DATE, 20.0, 5_000_000))
        else:
            # modest +10% move for everyone else
            today_rows.append(_bar(ticker, TRADE_DATE, prior_close * 1.10, 5_000_000))

    today_bars = pd.DataFrame(today_rows)
    corporate_actions = pd.DataFrame(
        [_ca("REVSPL", TRADE_DATE, "reverse_split", ratio=20.0)]
    )

    labels_df = labels_mod.build_labels(universe2, today_bars, corporate_actions, TRADE_DATE)

    assert "REVSPL" not in set(labels_df["ticker"])

    # Direct P4 tripwire coverage lives in test_sanity.py
    # (test_median_gainer_contamination_fails_at_plus_200_pct); this small
    # 5-name fixture's own median stays low even with REVSPL's raw +1900%
    # mixed in (one extreme outlier diluted by four modest +10% movers), so
    # it is check_no_split_days -- not the median check -- that catches
    # THIS specific fixture. Confirm that directly instead.
    from top10 import sanity

    unfiltered = universe2.merge(
        today_bars[["ticker", "close"]].rename(columns={"close": "close_t"}), on="ticker"
    )
    unfiltered["return_t"] = unfiltered["close_t"] / unfiltered["prior_close"] - 1.0
    unfiltered = unfiltered.sort_values("return_t", ascending=False).reset_index(drop=True)
    unfiltered["rank"] = unfiltered.index + 1
    unfiltered["label"] = (unfiltered["rank"] <= 10).astype(int)
    unfiltered["label_spec_version"] = "unused"

    result = sanity.check_no_split_days(unfiltered, corporate_actions)
    assert result.passed is False


def test_exactly_ten_positives_when_universe_large_enough():
    tickers = [f"T{i}" for i in range(15)]
    daily_bars = []
    ticker_metas = []
    for i, ticker in enumerate(tickers):
        prior_close = 10.0 + i
        daily_bars.extend(_make_20_days_of_bars(ticker, prior_close, 5_000_000))
        ticker_metas.append(_meta(ticker))

    ticker_meta = pd.DataFrame(ticker_metas)
    daily_bars_df = pd.DataFrame(daily_bars)
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    universe = labels_mod.build_universe(daily_bars_df, ticker_meta, corporate_actions, TRADE_DATE)

    today_rows = []
    for i, ticker in enumerate(tickers):
        prior_close = universe[universe["ticker"] == ticker].iloc[0]["prior_close"]
        today_rows.append(_bar(ticker, TRADE_DATE, prior_close * (1 + 0.01 * i), 5_000_000))
    today_bars = pd.DataFrame(today_rows)

    labels_df = labels_mod.build_labels(universe, today_bars, corporate_actions, TRADE_DATE)

    assert labels_df["label"].sum() == 10
    assert len(labels_df[labels_df["rank"] <= 10]) == 10


def test_label_spec_version_matches_committed_hash():
    universe, daily_bars_df = _universe_and_bars_for_label_test()
    today_rows = []
    for ticker in universe["ticker"]:
        prior_close = universe[universe["ticker"] == ticker].iloc[0]["prior_close"]
        today_rows.append(_bar(ticker, TRADE_DATE, prior_close * 1.05, 5_000_000))
    today_bars = pd.DataFrame(today_rows)
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    labels_df = labels_mod.build_labels(universe, today_bars, corporate_actions, TRADE_DATE)

    expected = hash_file(DOCS / "LABEL_SPEC.md")
    assert (labels_df["label_spec_version"] == expected).all()

    committed = (DOCS / "LABEL_SPEC.sha256").read_text().strip().split()[0]
    assert expected == committed


def test_persisted_columns_exact():
    universe, daily_bars_df = _universe_and_bars_for_label_test()
    today_rows = []
    for ticker in universe["ticker"]:
        prior_close = universe[universe["ticker"] == ticker].iloc[0]["prior_close"]
        today_rows.append(_bar(ticker, TRADE_DATE, prior_close * 1.05, 5_000_000))
    today_bars = pd.DataFrame(today_rows)
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    labels_df = labels_mod.build_labels(universe, today_bars, corporate_actions, TRADE_DATE)

    expected_cols = {"trade_date", "ticker", "rank", "return_t", "label", "label_spec_version", "as_of"}
    assert set(labels_df.columns) == expected_cols


# --- Defect 1: universe as_of must be (t-1) 16:00 ET, not midnight of t -------

def test_universe_as_of_is_prior_close_not_midnight(base_universe_inputs):
    """P1: `as_of` must be (t-1) 16:00 ET -- midnight of t is AFTER the
    prior close and is not "as of" it."""
    daily_bars, ticker_meta, corporate_actions = base_universe_inputs

    universe = labels_mod.build_universe(daily_bars, ticker_meta, corporate_actions, TRADE_DATE)

    assert not universe.empty
    expected_as_of = TRADE_DATE - pd.Timedelta(hours=8)  # (t-1) 16:00 ET
    assert (universe["as_of"] == expected_as_of).all()
    assert (universe["as_of"] != TRADE_DATE).all()


def test_universe_passes_prior_close_decision_time_gate(base_universe_inputs):
    """A build_universe frame must pass a real (t-1) 16:00 ET decision-time
    assertion -- this is the exact contract top10.baselines.b0_random
    enforces via assert_decision_time_safe(universe, _t1_decision_time(t)).
    Before the fix, every real universe frame RAISED here."""
    from top10.leakage import assert_decision_time_safe

    daily_bars, ticker_meta, corporate_actions = base_universe_inputs
    universe = labels_mod.build_universe(daily_bars, ticker_meta, corporate_actions, TRADE_DATE)

    decision_time = (TRADE_DATE - pd.Timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
    # Must NOT raise.
    assert_decision_time_safe(universe, decision_time.to_pydatetime())


def test_b0_random_accepts_build_universe_output():
    """Direct regression for the confirmed defect: 'b0_random on
    build_universe output RAISES'. Uses a pre-holdout trade_date (the
    PREREG one-time holdout seal, owned by top10.experiment, is a
    separate concern from this defect and must stay sealed)."""
    from top10.baselines import b0_random

    pre_holdout_trade_date = pd.Timestamp("2019-06-14")
    pre_holdout_prior_date = pd.Timestamp("2019-06-13")

    bars = []
    for ticker, close, dvol in [("AAA", 10.0, 5_000_000), ("BBB", 20.0, 2_000_000)]:
        bars.extend(_make_20_days_of_bars(ticker, close, dvol, end_date=pre_holdout_prior_date))
    daily_bars = pd.DataFrame(bars)
    ticker_meta = pd.DataFrame(
        [_meta("AAA", as_of="1999-01-01"), _meta("BBB", as_of="1999-01-01")]
    )
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    universe = labels_mod.build_universe(daily_bars, ticker_meta, corporate_actions, pre_holdout_trade_date)

    # Must NOT raise.
    predictions = b0_random(universe, seed=0, k=2)
    assert set(predictions.columns) == {"trade_date", "ticker", "score"}


# --- Defect 3: no-day-t-bar universe names are excluded, counted, logged -----

def test_no_day_t_bar_ticker_excluded_and_logged(caplog):
    """A universe name with no day-t bar (halted all day / delisted
    intraday) must be dropped EXPLICITLY -- counted and logged -- never
    silently via a bare inner join."""
    universe, daily_bars_df = _universe_and_bars_for_label_test()

    # HALTED has no bar on TRADE_DATE at all.
    today_rows = []
    for ticker in universe["ticker"]:
        if ticker == "HALTED":
            continue
        prior_close = universe[universe["ticker"] == ticker].iloc[0]["prior_close"]
        today_rows.append(_bar(ticker, TRADE_DATE, prior_close * 1.05, 5_000_000))
    today_bars = pd.DataFrame(today_rows)

    universe_with_halted = pd.concat(
        [universe, pd.DataFrame([{**universe.iloc[0].to_dict(), "ticker": "HALTED"}])],
        ignore_index=True,
    )
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    with caplog.at_level("WARNING"):
        labels_df = labels_mod.build_labels(universe_with_halted, today_bars, corporate_actions, TRADE_DATE)

    assert "HALTED" not in set(labels_df["ticker"])
    assert len(labels_df) == len(universe_with_halted) - 1
    assert any("HALTED" in rec.message and "no day-t bar" in rec.message for rec in caplog.records)


# --- Defect 4: stale prior_close must not produce a multi-day return ---------

def test_stale_prior_close_excluded_from_universe():
    """A ticker whose last bar is 5 sessions old must not be admitted to
    the universe with a 5-day return competing for a top-10 slot."""
    bars = []
    for ticker, close, dvol in [("AAA", 10.0, 5_000_000), ("BBB", 20.0, 2_000_000), ("CCC", 5.0, 1_500_000)]:
        bars.extend(_make_20_days_of_bars(ticker, close, dvol))

    # STALE's most recent bar is 5 business days before PRIOR_DATE.
    stale_end = pd.bdate_range(end=PRIOR_DATE, periods=6)[0]
    bars.extend(_make_20_days_of_bars("STALE", 10.0, 5_000_000, end_date=stale_end))

    daily_bars = pd.DataFrame(bars)
    ticker_meta = pd.DataFrame(
        [_meta("AAA"), _meta("BBB"), _meta("CCC"), _meta("STALE")]
    )
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    universe = labels_mod.build_universe(daily_bars, ticker_meta, corporate_actions, TRADE_DATE)

    assert "STALE" not in set(universe["ticker"])
    assert "AAA" in set(universe["ticker"])

    # And even if STALE somehow had a day-t bar, it can never reach
    # build_labels since it never enters the universe -- confirm no
    # multi-day return is ever computed for it.
    today_rows = [_bar("STALE", TRADE_DATE, 11.0, 5_000_000)]
    for ticker in universe["ticker"]:
        prior_close = universe[universe["ticker"] == ticker].iloc[0]["prior_close"]
        today_rows.append(_bar(ticker, TRADE_DATE, prior_close * 1.02, 5_000_000))
    today_bars = pd.DataFrame(today_rows)

    labels_df = labels_mod.build_labels(universe, today_bars, corporate_actions, TRADE_DATE)
    assert "STALE" not in set(labels_df["ticker"])


def test_staleness_configurable_via_max_staleness_sessions():
    """A wider staleness window admits an otherwise-stale name."""
    bars = []
    for ticker, close, dvol in [("AAA", 10.0, 5_000_000), ("BBB", 20.0, 2_000_000)]:
        bars.extend(_make_20_days_of_bars(ticker, close, dvol))

    stale_end = pd.bdate_range(end=PRIOR_DATE, periods=3)[0]  # 2 sessions stale
    bars.extend(_make_20_days_of_bars("STALE2", 10.0, 5_000_000, end_date=stale_end))

    daily_bars = pd.DataFrame(bars)
    ticker_meta = pd.DataFrame([_meta("AAA"), _meta("BBB"), _meta("STALE2")])
    corporate_actions = pd.DataFrame(
        columns=["ex_date", "ticker", "action_type", "ratio", "cash_amount", "new_ticker", "as_of"]
    )

    default_universe = labels_mod.build_universe(daily_bars, ticker_meta, corporate_actions, TRADE_DATE)
    assert "STALE2" not in set(default_universe["ticker"])

    wide_universe = labels_mod.build_universe(
        daily_bars, ticker_meta, corporate_actions, TRADE_DATE, max_staleness_sessions=5
    )
    assert "STALE2" in set(wide_universe["ticker"])
