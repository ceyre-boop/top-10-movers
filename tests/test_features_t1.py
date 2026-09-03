from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from top10.data.base import SHORT_INTEREST_COLUMNS
from top10.features import t1 as t1_mod
from top10.features.spec import T1_COLUMNS, T1_SPEC, validate_frame
from top10.leakage import assert_decision_time_safe, assert_self_exclusion
from top10.storage import LeakageError

TRADE_DATE = pd.Timestamp("2024-03-15")


# --- Synthetic fixture builders ----------------------------------------------


def _bar(ticker, trade_date, close, volume=1_000_000.0, as_of=None):
    trade_date = pd.Timestamp(trade_date)
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "dollar_volume": close * volume,
        "as_of": pd.Timestamp(as_of) if as_of is not None else trade_date,
    }


def _n_days_of_bars(ticker, n, start_close=10.0, drift=0.0, end_date=None, volume=1_000_000.0):
    """`n` business days of bars ending at `end_date` (default: day before
    TRADE_DATE). Deterministic small drift so returns are well-defined."""
    end_date = pd.Timestamp(end_date) if end_date is not None else TRADE_DATE - pd.Timedelta(days=1)
    dates = pd.bdate_range(end=end_date, periods=n)
    rows = []
    close = start_close
    for d in dates:
        rows.append(_bar(ticker, d, close, volume=volume))
        close = close * (1 + drift)
    return rows


def _meta_row(ticker, sector=None, industry=None, market_cap=None, float_shares=None,
              as_of="2020-01-01"):
    return {
        "ticker": ticker,
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
        "float_shares": float_shares,
        "as_of": pd.Timestamp(as_of),
    }


def _si_row(ticker, short_interest_pct_float=None, days_to_cover=None,
            short_interest_shares=None, settlement_date="2020-01-01", as_of="2020-01-01"):
    """A row shaped like `SHORT_INTEREST_COLUMNS` -- short interest is its
    own frame, never part of `ticker_meta` (see top10/data/base.py)."""
    return {
        "ticker": ticker,
        "settlement_date": pd.Timestamp(settlement_date),
        "short_interest_shares": short_interest_shares,
        "short_interest_pct_float": short_interest_pct_float,
        "days_to_cover": days_to_cover,
        "as_of": pd.Timestamp(as_of),
    }


def _empty_short_interest():
    return pd.DataFrame(columns=list(SHORT_INTEREST_COLUMNS))


def _empty_earnings():
    return pd.DataFrame(
        columns=["ticker", "report_date", "session", "announced_on", "date_is_revisable", "as_of"]
    )


def _empty_labels_history():
    return pd.DataFrame(
        columns=["trade_date", "ticker", "rank", "return_t", "label", "label_spec_version", "as_of"]
    )


def _empty_market_context():
    return pd.DataFrame(
        columns=["trade_date", "spy_ret_1d", "spy_ret_5d", "vix_level", "iwm_minus_spy_1d", "movers_10pct_count", "as_of"]
    )


def _market_context_row(trade_date, **kwargs):
    trade_date = pd.Timestamp(trade_date)
    row = {
        "trade_date": trade_date,
        "spy_ret_1d": 0.01,
        "spy_ret_5d": 0.02,
        "vix_level": 15.0,
        "iwm_minus_spy_1d": -0.002,
        "movers_10pct_count": 12,
        "as_of": trade_date,
    }
    row.update(kwargs)
    return row


def _basic_inputs(n_days=25, tickers=("AAA", "BBB", "CCC")):
    bars = []
    for i, t in enumerate(tickers):
        bars.extend(_n_days_of_bars(t, n_days, start_close=10.0 + i, drift=0.001 * (i + 1)))
    daily_bars = pd.DataFrame(bars)
    ticker_meta = pd.DataFrame(
        [_meta_row(t, sector="Technology", industry="Software", market_cap=5e9, float_shares=1e8) for t in tickers]
    )
    return daily_bars, ticker_meta


def _build(daily_bars, ticker_meta, earnings=None, labels_history=None, market_context=None,
           trade_date=TRADE_DATE, short_interest=None):
    return t1_mod.build_t1_features(
        daily_bars,
        ticker_meta,
        earnings if earnings is not None else _empty_earnings(),
        labels_history if labels_history is not None else _empty_labels_history(),
        market_context if market_context is not None else _empty_market_context(),
        trade_date,
        short_interest=short_interest,
    )


# --- Structural invariants ----------------------------------------------------


def test_output_columns_match_spec_order():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)
    assert list(out.columns) == list(T1_COLUMNS)
    validate_frame(out, T1_SPEC)


def test_as_of_present_and_decision_time_safe():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)
    assert "as_of" in out.columns
    assert not out.empty
    decision_time = t1_mod.decision_time_t1(TRADE_DATE)
    assert_decision_time_safe(out, decision_time)
    assert (out["as_of"] == decision_time).all()


def test_self_exclusion_passes_against_real_labels():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)

    labels = pd.DataFrame(
        [
            {
                "trade_date": TRADE_DATE,
                "ticker": t,
                "rank": i + 1,
                "return_t": 0.05 * (i + 1),
                "label": int(i < 1),
                "label_spec_version": "test",
                "as_of": TRADE_DATE,
            }
            for i, t in enumerate(out["ticker"])
        ]
    )
    assert_self_exclusion(out, labels)


def test_features_never_reject_same_day_bar_as_leak():
    """Sanity: decision_time_t1 sits strictly before trade_date midnight,
    so even a same-day bar (as_of == trade_date) is correctly excluded
    upstream rather than silently accepted -- i.e. it never appears in
    `out` at all for the trailing-window computation."""
    daily_bars, ticker_meta = _basic_inputs()
    same_day_bar = pd.DataFrame([_bar("AAA", TRADE_DATE, 999.0, as_of=TRADE_DATE)])
    bars_with_leak = pd.concat([daily_bars, same_day_bar], ignore_index=True)

    out = _build(bars_with_leak, ticker_meta)
    aaa_ret_1d = out.loc[out["ticker"] == "AAA", "ret_1d"].iloc[0]
    # If the same-day bar (close=999) had leaked in, ret_1d would be huge.
    assert aaa_ret_1d < 5.0


# --- 20d history / NaN behavior ------------------------------------------------


def test_short_history_ticker_yields_nan_for_20d_features_not_wrong_value():
    daily_bars, ticker_meta = _basic_inputs(n_days=25)
    short_bars = pd.DataFrame(_n_days_of_bars("SHORT", 5, start_close=10.0))
    meta_short = pd.DataFrame([_meta_row("SHORT", sector="Technology", industry="Software")])

    out = _build(
        pd.concat([daily_bars, short_bars], ignore_index=True),
        pd.concat([ticker_meta, meta_short], ignore_index=True),
    )
    row = out[out["ticker"] == "SHORT"].iloc[0]
    assert pd.isna(row["ret_20d"])
    assert pd.isna(row["rvol_20d"])
    assert pd.isna(row["adv_20"])
    # But short-window features should still compute (n=5 >= 2).
    assert pd.notna(row["ret_1d"])


def test_full_history_ticker_has_20d_features():
    daily_bars, ticker_meta = _basic_inputs(n_days=25)
    out = _build(daily_bars, ticker_meta)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert pd.notna(row["ret_20d"])
    assert pd.notna(row["rvol_20d"])
    assert pd.notna(row["adv_20"])


# --- Cross-sectional rank -------------------------------------------------------


def test_ret_1d_rank_is_cross_sectional_within_trade_date():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)

    manual_rank = out["ret_1d"].rank(method="min", ascending=False)
    assert (out["ret_1d_rank"] == manual_rank).all()

    best = out.sort_values("ret_1d", ascending=False).iloc[0]
    assert best["ret_1d_rank"] == 1.0


def test_ret_1d_rank_computed_per_day_not_globally():
    """Build two separate single-day frames and confirm rank=1 in both,
    proving rank isn't computed against some accumulated global frame."""
    daily_bars, ticker_meta = _basic_inputs(tickers=("AAA", "BBB"))
    out_day1 = _build(daily_bars, ticker_meta, trade_date=TRADE_DATE)

    daily_bars2, ticker_meta2 = _basic_inputs(
        tickers=("CCC", "DDD"), n_days=25
    )
    trade_date2 = TRADE_DATE + pd.Timedelta(days=7)
    daily_bars2 = pd.DataFrame(
        _n_days_of_bars("CCC", 25, start_close=10.0, drift=0.001, end_date=trade_date2 - pd.Timedelta(days=1))
        + _n_days_of_bars("DDD", 25, start_close=20.0, drift=0.002, end_date=trade_date2 - pd.Timedelta(days=1))
    )
    out_day2 = _build(daily_bars2, ticker_meta2, trade_date=trade_date2)

    assert 1.0 in out_day1["ret_1d_rank"].values
    assert 1.0 in out_day2["ret_1d_rank"].values


# --- labels_history / appearance features (the dangerous ones) -----------------


def test_same_day_planted_label_is_not_consumed():
    daily_bars, ticker_meta = _basic_inputs()

    # Realistic same-day labels for the FULL universe (AAA/BBB/CCC), not
    # just a single planted ticker -- `build_t1_features` now also runs
    # `assert_self_exclusion` against same-day rows in `labels_history`
    # (Defect 1 production wiring), and that check's identity layer has no
    # statistical protection at n=1 (see `top10.leakage.assert_self_exclusion`
    # docstring); a single-ticker frame is not representative of how this
    # function is actually called in production (the full day's universe).
    labels_history = pd.DataFrame(
        [
            {
                "trade_date": TRADE_DATE,  # SAME day as trade_date under test
                "ticker": "AAA",
                "rank": 1,
                "return_t": 0.5,
                "label": 1,
                "label_spec_version": "test",
                "as_of": TRADE_DATE,
            },
            {
                "trade_date": TRADE_DATE,
                "ticker": "BBB",
                "rank": 2,
                "return_t": 0.2,
                "label": 0,
                "label_spec_version": "test",
                "as_of": TRADE_DATE,
            },
            {
                "trade_date": TRADE_DATE,
                "ticker": "CCC",
                "rank": 3,
                "return_t": 0.1,
                "label": 0,
                "label_spec_version": "test",
                "as_of": TRADE_DATE,
            },
        ]
    )

    out = _build(daily_bars, ticker_meta, labels_history=labels_history)
    row = out[out["ticker"] == "AAA"].iloc[0]

    # If the same-day label had leaked in, days_since_last_top10 would be 0
    # and appearances_30d/90d would be >= 1.
    assert pd.isna(row["days_since_last_top10"])
    assert row["appearances_30d"] == 0
    assert row["appearances_90d"] == 0


def test_prior_day_label_is_correctly_counted():
    daily_bars, ticker_meta = _basic_inputs()

    prior_date = TRADE_DATE - pd.Timedelta(days=5)
    labels_history = pd.DataFrame(
        [
            {
                "trade_date": prior_date,
                "ticker": "AAA",
                "rank": 1,
                "return_t": 0.5,
                "label": 1,
                "label_spec_version": "test",
                "as_of": prior_date,
            }
        ]
    )

    out = _build(daily_bars, ticker_meta, labels_history=labels_history)
    row = out[out["ticker"] == "AAA"].iloc[0]

    assert row["days_since_last_top10"] == 5.0
    assert row["appearances_30d"] == 1
    assert row["appearances_90d"] == 1


def test_appearance_outside_90d_window_not_counted():
    daily_bars, ticker_meta = _basic_inputs()

    old_date = TRADE_DATE - pd.Timedelta(days=200)
    labels_history = pd.DataFrame(
        [
            {
                "trade_date": old_date,
                "ticker": "AAA",
                "rank": 1,
                "return_t": 0.5,
                "label": 1,
                "label_spec_version": "test",
                "as_of": old_date,
            }
        ]
    )

    out = _build(daily_bars, ticker_meta, labels_history=labels_history)
    row = out[out["ticker"] == "AAA"].iloc[0]

    assert row["days_since_last_top10"] == 200.0
    assert row["appearances_30d"] == 0
    assert row["appearances_90d"] == 0


# --- earnings ------------------------------------------------------------------


def test_earnings_today_flag_and_days_to_earnings():
    daily_bars, ticker_meta = _basic_inputs()
    earnings = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "report_date": TRADE_DATE,
                "session": "bmo",
                "announced_on": TRADE_DATE - pd.Timedelta(days=10),
                "date_is_revisable": False,
                "as_of": TRADE_DATE - pd.Timedelta(days=10),
            }
        ]
    )
    out = _build(daily_bars, ticker_meta, earnings=earnings)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["earnings_today"] == 1
    assert row["earnings_tomorrow"] == 0
    assert row["days_to_earnings"] == 0.0
    assert row["earnings_date_revisable"] == False  # noqa: E712


def test_earnings_announced_after_decision_time_is_ignored():
    """P3 tripwire: an earnings row whose `announced_on` is AFTER
    decision_time_t1 must not be usable, even if report_date == trade_date."""
    daily_bars, ticker_meta = _basic_inputs()
    decision_time = t1_mod.decision_time_t1(TRADE_DATE)
    earnings = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "report_date": TRADE_DATE,
                "session": "bmo",
                "announced_on": decision_time + pd.Timedelta(hours=1),  # announced AFTER decision time
                "date_is_revisable": False,
                "as_of": decision_time + pd.Timedelta(hours=1),
            }
        ]
    )
    out = _build(daily_bars, ticker_meta, earnings=earnings)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["earnings_today"] == 0
    assert pd.isna(row["days_to_earnings"])


def test_revisable_earnings_date_marked_with_companion_column():
    """Defect 3 (CONFIRMED): the real data adapter sets
    `date_is_revisable = (announced_on is None)`, so an adapter-realistic
    revisable row has `announced_on = NaT` -- NEVER `announced_on` set
    alongside `date_is_revisable=True`, a combination the adapter cannot
    produce. Filtering on `announced_on <= decision_time` (the old, buggy
    behavior) is False for NaT and silently drops every revisable row
    before `earnings_date_revisable` can ever be read; this must be kept
    and flagged instead, gated on the always-populated `as_of`."""
    daily_bars, ticker_meta = _basic_inputs()
    earnings = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "report_date": TRADE_DATE + pd.Timedelta(days=3),
                "session": "amc",
                "announced_on": pd.NaT,
                "date_is_revisable": True,
                "as_of": TRADE_DATE - pd.Timedelta(days=20),
            }
        ]
    )
    out = _build(daily_bars, ticker_meta, earnings=earnings)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["earnings_date_revisable"] == True  # noqa: E712
    assert row["days_to_earnings"] == 3.0


def test_revisable_earnings_row_dropped_before_its_own_as_of_is_knowable():
    """The revisable row must still respect PIT: if its conservative
    `as_of` is after decision_time, it must not be visible yet."""
    daily_bars, ticker_meta = _basic_inputs()
    decision_time = t1_mod.decision_time_t1(TRADE_DATE)
    earnings = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "report_date": TRADE_DATE + pd.Timedelta(days=3),
                "session": "amc",
                "announced_on": pd.NaT,
                "date_is_revisable": True,
                "as_of": decision_time + pd.Timedelta(hours=1),
            }
        ]
    )
    out = _build(daily_bars, ticker_meta, earnings=earnings)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["earnings_date_revisable"] == False  # noqa: E712
    assert pd.isna(row["days_to_earnings"])


# --- short interest: forward-fill only from publish date ------------------------


def test_short_interest_not_visible_before_publish_date():
    daily_bars, ticker_meta = _basic_inputs()
    publish_date = TRADE_DATE - pd.Timedelta(days=2)

    short_interest = pd.DataFrame(
        [_si_row("AAA", short_interest_pct_float=25.0, days_to_cover=3.5, as_of=publish_date)]
    )

    # trade_date is BEFORE the publish date -> short interest must be NaN.
    out_before = _build(
        daily_bars, ticker_meta, trade_date=publish_date - pd.Timedelta(days=1),
        short_interest=short_interest,
    )
    row_before = out_before[out_before["ticker"] == "AAA"].iloc[0]
    assert pd.isna(row_before["short_interest_pct_float"])

    # trade_date is AFTER the publish date -> value must be forward-filled.
    out_after = _build(daily_bars, ticker_meta, trade_date=TRADE_DATE, short_interest=short_interest)
    row_after = out_after[out_after["ticker"] == "AAA"].iloc[0]
    assert row_after["short_interest_pct_float"] == 25.0
    assert row_after["days_to_cover"] == 3.5


def test_more_recent_knowable_short_interest_vintage_wins():
    """Rewritten -- the prior name/docstring
    (`test_short_interest_forward_fills_from_settlement_not_used`) claimed
    to test that forward-fill keys off `as_of` rather than a 'settlement
    date', but this module has no settlement-date concept at all, so that
    claim was untestable by construction; its own docstring conceded as
    much. What this test actually exercises: given two knowable vintages
    for the same ticker, `_latest_pit_row` selects the most recent one
    (by `as_of`), not an arbitrary or earliest one."""
    daily_bars, ticker_meta = _basic_inputs()
    old_publish = TRADE_DATE - pd.Timedelta(days=40)
    new_publish = TRADE_DATE - pd.Timedelta(days=5)

    short_interest = pd.DataFrame(
        [
            _si_row("AAA", short_interest_pct_float=10.0, days_to_cover=1.0, as_of=old_publish),
            _si_row("AAA", short_interest_pct_float=40.0, days_to_cover=6.0, as_of=new_publish),
        ]
    )

    out = _build(daily_bars, ticker_meta, trade_date=TRADE_DATE, short_interest=short_interest)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["short_interest_pct_float"] == 40.0
    assert row["days_to_cover"] == 6.0


# --- buckets ---------------------------------------------------------------------


def test_price_bucket_is_bucketed_not_raw():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)
    for v in out["price_bucket"]:
        assert v == int(v)  # bucket code, not a raw price


def test_mcap_and_float_bucketed():
    daily_bars, ticker_meta = _basic_inputs()
    ticker_meta = ticker_meta.copy()
    ticker_meta.loc[ticker_meta["ticker"] == "AAA", "market_cap"] = 5e9
    ticker_meta.loc[ticker_meta["ticker"] == "AAA", "float_shares"] = 5e7
    out = _build(daily_bars, ticker_meta)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["mcap_bucket"] in {0, 1, 2, 3, 4, 5}
    assert row["float_bucket"] in {0, 1, 2, 3, 4}


# --- sector one-hot + biotech -----------------------------------------------------


def test_sector_one_hot_and_is_biotech_flag():
    daily_bars, ticker_meta = _basic_inputs(tickers=("AAA",))
    ticker_meta = pd.DataFrame(
        [_meta_row("AAA", sector="Healthcare", industry="Biotechnology", market_cap=1e9, float_shares=5e7)]
    )
    out = _build(daily_bars, ticker_meta)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["sector_healthcare"] == 1
    assert row["sector_technology"] == 0
    assert row["is_biotech"] == 1


def test_unknown_sector_falls_into_other():
    daily_bars, ticker_meta = _basic_inputs(tickers=("AAA",))
    ticker_meta = pd.DataFrame([_meta_row("AAA", sector="Weird Made Up Sector", industry="Nothing")])
    out = _build(daily_bars, ticker_meta)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["sector_other"] == 1
    assert row["is_biotech"] == 0


# --- market context ---------------------------------------------------------------


def test_market_context_broadcast_to_all_tickers():
    daily_bars, ticker_meta = _basic_inputs()
    prior_date = TRADE_DATE - pd.Timedelta(days=1)
    market_context = pd.DataFrame([_market_context_row(prior_date, vix_level=22.5)])
    out = _build(daily_bars, ticker_meta, market_context=market_context)
    assert (out["mkt_vix_level"] == 22.5).all()


def test_market_context_same_day_row_excluded():
    daily_bars, ticker_meta = _basic_inputs()
    market_context = pd.DataFrame(
        [
            _market_context_row(TRADE_DATE - pd.Timedelta(days=1), vix_level=20.0),
            _market_context_row(TRADE_DATE, vix_level=999.0),  # same-day: must be excluded
        ]
    )
    out = _build(daily_bars, ticker_meta, market_context=market_context)
    assert (out["mkt_vix_level"] == 20.0).all()


# --- Defect 4: labels_history must respect as_of, not just trade_date -------------


def test_revised_label_vintage_with_later_as_of_is_not_yet_consumed():
    """A rebuilt/revised label vintage for a prior trade_date, stamped with
    an `as_of` AFTER decision_time, must not be silently consumed just
    because its `trade_date` is strictly prior."""
    daily_bars, ticker_meta = _basic_inputs()
    decision_time = t1_mod.decision_time_t1(TRADE_DATE)
    prior_date = TRADE_DATE - pd.Timedelta(days=5)

    labels_history = pd.DataFrame(
        [
            {
                "trade_date": prior_date,
                "ticker": "AAA",
                "rank": 1,
                "return_t": 0.5,
                "label": 1,
                "label_spec_version": "test",
                # Revised/rebuilt AFTER this decision time -- not yet knowable.
                "as_of": decision_time + pd.Timedelta(hours=1),
            }
        ]
    )

    out = _build(daily_bars, ticker_meta, labels_history=labels_history)
    row = out[out["ticker"] == "AAA"].iloc[0]

    assert pd.isna(row["days_since_last_top10"])
    assert row["appearances_30d"] == 0
    assert row["appearances_90d"] == 0


# --- Defect 1: production wiring of the anti-leakage harness ----------------------


def test_build_t1_features_output_is_always_decision_time_safe():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)
    # Would raise LeakageError if build_t1_features didn't already
    # self-check via assert_decision_time_safe before returning.
    assert_decision_time_safe(out, t1_mod.decision_time_t1(TRADE_DATE))


def test_build_t1_features_raises_on_same_day_label_identity_leak():
    """If the caller passes `labels_history` that ALSO carries today's real
    label rows (not just strictly-prior history), and a feature is
    literally identical to the same-day `return_t` column,
    `build_t1_features` must raise rather than silently return the
    leaking frame -- proves `assert_self_exclusion` is actually wired in,
    not just importable."""
    daily_bars, ticker_meta = _basic_inputs()
    out_preview = _build(daily_bars, ticker_meta)

    # Real same-day labels for TRADE_DATE, with `return_t` set IDENTICAL
    # (by construction) to the `ret_1d` feature this module produces --
    # an exact same-day label leak (assert_self_exclusion layer 2).
    labels_today = pd.DataFrame(
        [
            {
                "trade_date": TRADE_DATE,
                "ticker": row["ticker"],
                "rank": i + 1,
                "return_t": row["ret_1d"],
                "label": 1,
                "label_spec_version": "test",
                "as_of": TRADE_DATE,
            }
            for i, (_, row) in enumerate(out_preview.iterrows())
        ]
    )

    with pytest.raises(LeakageError):
        _build(daily_bars, ticker_meta, labels_history=labels_today)


# --- Defect 6: missing ticker_meta column must raise, not silently NaN ------------


def test_missing_ticker_meta_column_raises_loudly():
    daily_bars, ticker_meta = _basic_inputs()
    broken_meta = ticker_meta.drop(columns=["market_cap"])
    with pytest.raises(KeyError):
        _build(daily_bars, broken_meta)


# --- Defect 3: short_interest is its own frame, never merged into
# ticker_meta -- a missing required column must raise, even (especially)
# when the frame is empty. -------------------------------------------------


def test_missing_short_interest_column_raises_even_when_empty():
    """The negative case that matters most: an EMPTY short_interest frame
    missing `days_to_cover` must still raise -- if an empty frame passes
    silently, the fix did not land (this is exactly Defect 3's second bug:
    the old `if not ticker_meta.empty` gate let an empty frame NaN
    everything silently)."""
    daily_bars, ticker_meta = _basic_inputs()
    broken_short_interest = pd.DataFrame(
        columns=[c for c in SHORT_INTEREST_COLUMNS if c != "days_to_cover"]
    )
    assert broken_short_interest.empty
    with pytest.raises(KeyError):
        _build(daily_bars, ticker_meta, short_interest=broken_short_interest)


def test_missing_short_interest_column_raises_when_nonempty_too():
    daily_bars, ticker_meta = _basic_inputs()
    broken_short_interest = pd.DataFrame(
        [{"ticker": "AAA", "settlement_date": TRADE_DATE, "short_interest_shares": 1.0,
          "short_interest_pct_float": 5.0, "as_of": TRADE_DATE}]  # missing days_to_cover
    )
    with pytest.raises(KeyError):
        _build(daily_bars, ticker_meta, short_interest=broken_short_interest)


def test_well_formed_short_interest_produces_pit_gated_values():
    """Positive case: a well-formed short_interest frame produces non-null
    values via the `_latest_pit_row` lookup, and a row whose `as_of` is
    AFTER decision_time is not used."""
    daily_bars, ticker_meta = _basic_inputs()
    decision_time = t1_mod.decision_time_t1(TRADE_DATE)

    short_interest = pd.DataFrame(
        [
            # Knowable well before decision_time -- must be used.
            _si_row("AAA", short_interest_pct_float=12.5, days_to_cover=2.0,
                    as_of=TRADE_DATE - pd.Timedelta(days=10)),
            # Published AFTER decision_time -- must NOT be used.
            _si_row("BBB", short_interest_pct_float=99.0, days_to_cover=9.0,
                    as_of=decision_time + pd.Timedelta(hours=1)),
        ]
    )

    out = _build(daily_bars, ticker_meta, short_interest=short_interest)

    aaa = out[out["ticker"] == "AAA"].iloc[0]
    assert aaa["short_interest_pct_float"] == 12.5
    assert aaa["days_to_cover"] == 2.0

    bbb = out[out["ticker"] == "BBB"].iloc[0]
    assert pd.isna(bbb["short_interest_pct_float"])
    assert pd.isna(bbb["days_to_cover"])


def test_ticker_with_no_meta_row_at_all_still_degrades_to_nan():
    """A ticker simply absent from (non-empty) ticker_meta is a legitimate
    per-row gap, not a schema defect -- must still degrade to NaN, not raise."""
    daily_bars, ticker_meta = _basic_inputs(tickers=("AAA", "BBB"))
    ticker_meta_missing_ccc = ticker_meta[ticker_meta["ticker"] != "BBB"]
    out = _build(daily_bars, ticker_meta_missing_ccc)
    row = out[out["ticker"] == "BBB"].iloc[0]
    assert pd.isna(row["mcap_bucket"])
    assert pd.isna(row["float_bucket"])


# --- validate_frame ----------------------------------------------------------------


def test_validate_frame_accepts_matching_columns():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)
    validate_frame(out, T1_SPEC)


def test_validate_frame_rejects_reordered_columns():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)
    reordered = out[list(reversed(list(out.columns)))]
    with pytest.raises(ValueError):
        validate_frame(reordered, T1_SPEC)


def test_validate_frame_rejects_missing_column():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta).drop(columns=["ret_1d"])
    with pytest.raises(ValueError):
        validate_frame(out, T1_SPEC)


def test_validate_frame_rejects_extra_column():
    daily_bars, ticker_meta = _basic_inputs()
    out = _build(daily_bars, ticker_meta)
    out["bogus_extra"] = 1
    with pytest.raises(ValueError):
        validate_frame(out, T1_SPEC)
