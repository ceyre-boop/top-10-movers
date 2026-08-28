from __future__ import annotations

import pandas as pd
import pytest

from top10.features import t1 as t1_mod
from top10.features import t2 as t2_mod
from top10.features.spec import T1_COLUMNS, T2_COLUMNS, T2_SPEC, validate_frame
from top10.leakage import assert_decision_time_safe, assert_self_exclusion
from top10.storage import LeakageError

TRADE_DATE = pd.Timestamp("2024-03-15")


# --- Fixture builders ----------------------------------------------------------


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


def _n_days_of_bars(ticker, n, start_close=10.0, drift=0.001, end_date=None, volume=1_000_000.0):
    end_date = pd.Timestamp(end_date) if end_date is not None else TRADE_DATE - pd.Timedelta(days=1)
    dates = pd.bdate_range(end=end_date, periods=n)
    rows = []
    close = start_close
    for d in dates:
        rows.append(_bar(ticker, d, close, volume=volume))
        close = close * (1 + drift)
    return rows


def _meta_row(ticker, sector="Technology", industry="Software", market_cap=5e9, float_shares=1e8, as_of="2020-01-01"):
    return {
        "ticker": ticker,
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
        "float_shares": float_shares,
        "short_interest_pct_float": None,
        "days_to_cover": None,
        "as_of": pd.Timestamp(as_of),
    }


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


def _pm_bar(ticker, minute, close, volume=10_000.0, high=None, trade_count=5, as_of=None):
    minute = pd.Timestamp(minute)
    return {
        "trade_date": TRADE_DATE,
        "ticker": ticker,
        "minute": minute,
        "open": close,
        "high": high if high is not None else close,
        "low": close,
        "close": close,
        "volume": volume,
        "trade_count": trade_count,
        "as_of": pd.Timestamp(as_of) if as_of is not None else minute,
    }


def _t1_for(tickers=("AAA", "BBB"), n_days=25):
    bars = []
    for i, t in enumerate(tickers):
        bars.extend(_n_days_of_bars(t, n_days, start_close=10.0 + i))
    daily_bars = pd.DataFrame(bars)
    ticker_meta = pd.DataFrame([_meta_row(t) for t in tickers])
    t1_features = t1_mod.build_t1_features(
        daily_bars,
        ticker_meta,
        _empty_earnings(),
        _empty_labels_history(),
        _empty_market_context(),
        TRADE_DATE,
    )
    return t1_features


def _prior_close_df(t1_features, prices: dict, *, trade_date=TRADE_DATE, as_of=None):
    """Column contract: `trade_date, ticker, close, as_of` -- `close` is the
    PRIOR trading day's close, indexed by the trade_date being PREDICTED.
    Same contract `top10.baselines.b4_premarket_gap` requires (Defect 2).
    `as_of` defaults to the T1 decision time (16:00 ET on t-1) -- the
    latest a legitimate prior close could have been knowable."""
    trade_date = pd.Timestamp(trade_date)
    as_of = pd.Timestamp(as_of) if as_of is not None else t1_mod.decision_time_t1(trade_date)
    return pd.DataFrame(
        [{"trade_date": trade_date, "ticker": t, "close": p, "as_of": as_of} for t, p in prices.items()]
    )


def _empty_halts():
    return pd.DataFrame(columns=["ticker", "trade_date", "as_of"])


# --- Structural invariants -------------------------------------------------------


def test_output_columns_match_spec_order():
    t1_features = _t1_for()
    pm = pd.DataFrame(
        [_pm_bar("AAA", "2024-03-15 09:00", 10.5), _pm_bar("BBB", "2024-03-15 08:30", 11.2)]
    )
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0, "BBB": 11.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    assert list(out.columns) == list(T2_COLUMNS)
    validate_frame(out, T2_SPEC)


def test_t2_is_strict_superset_of_t1_columns():
    assert set(T1_COLUMNS).issubset(set(T2_COLUMNS))
    assert set(T2_COLUMNS) - set(T1_COLUMNS) == {
        "premarket_gap_pct",
        "premarket_dollar_volume",
        "premarket_rel_volume",
        "premarket_high_to_last_drawdown",
        "premarket_trade_count",
        "premarket_first_trade_minutes",
        "overnight_halt_flag",
    }


def test_as_of_present_and_decision_time_safe():
    t1_features = _t1_for()
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 09:00", 10.5)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0, "BBB": 11.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)

    decision_time = t2_mod.decision_time_t2(TRADE_DATE)
    assert "as_of" in out.columns
    assert (out["as_of"] == decision_time).all()
    assert_decision_time_safe(out, decision_time)


def test_self_exclusion_passes_against_real_labels():
    t1_features = _t1_for()
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 09:00", 10.5)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0, "BBB": 11.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)

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


# --- The core P3 tripwire: 09:30 bar must be excluded --------------------------


def test_0930_bar_is_excluded_from_premarket_aggregation():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame(
        [
            _pm_bar("AAA", "2024-03-15 09:00", 10.2, volume=1000),
            _pm_bar("AAA", "2024-03-15 09:30", 999.0, volume=999_999_999),  # PLANTED leak
        ]
    )
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]

    # If the 09:30 bar leaked in, gap% and dollar volume would be enormous.
    assert row["premarket_gap_pct"] < 1.0
    assert row["premarket_dollar_volume"] < 1_000_000


def test_bar_exactly_at_0925_is_excluded():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame(
        [
            _pm_bar("AAA", "2024-03-15 09:00", 10.2, volume=1000),
            _pm_bar("AAA", "2024-03-15 09:25", 500.0, volume=999_999),  # exactly at cutoff -> excluded
        ]
    )
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["premarket_dollar_volume"] < 1_000_000


def test_bar_before_0925_is_included():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 09:24", 10.5, volume=1000)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["premarket_trade_count"] == 5
    assert row["premarket_gap_pct"] == pytest.approx(0.05)


# --- premarket feature values ---------------------------------------------------


def test_premarket_gap_pct_computed_against_prior_close():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 08:00", 11.0)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["premarket_gap_pct"] == pytest.approx(0.10)


def test_premarket_high_to_last_drawdown():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame(
        [
            _pm_bar("AAA", "2024-03-15 07:00", 12.0, high=12.0),
            _pm_bar("AAA", "2024-03-15 08:00", 10.0, high=10.0),  # faded off the high
        ]
    )
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["premarket_high_to_last_drawdown"] == pytest.approx(10.0 / 12.0 - 1.0)


def test_no_premarket_bars_yields_nan_not_zero():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame(columns=["trade_date", "ticker", "minute", "open", "high", "low", "close", "volume", "trade_count", "as_of"])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert pd.isna(row["premarket_gap_pct"])
    assert pd.isna(row["premarket_rel_volume"])
    assert row["premarket_trade_count"] == 0


def test_first_trade_minutes_since_0400():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 04:15", 10.5)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["premarket_first_trade_minutes"] == pytest.approx(15.0)


# --- overnight halt --------------------------------------------------------------


def test_overnight_halt_flag_set_when_present():
    t1_features = _t1_for(tickers=("AAA", "BBB"))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 09:00", 10.5)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0, "BBB": 11.0})
    halts = pd.DataFrame([{"ticker": "AAA", "trade_date": TRADE_DATE, "as_of": TRADE_DATE}])
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, halts, TRADE_DATE)

    assert out.loc[out["ticker"] == "AAA", "overnight_halt_flag"].iloc[0] == 1
    assert out.loc[out["ticker"] == "BBB", "overnight_halt_flag"].iloc[0] == 0


def test_overnight_halt_flag_zero_when_no_halts():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 09:00", 10.5)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    assert out["overnight_halt_flag"].tolist() == [0]


# --- validate_frame on T2 ---------------------------------------------------------


def test_validate_frame_rejects_reordered_t2_columns():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 09:00", 10.5)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    reordered = out[list(reversed(list(out.columns)))]
    with pytest.raises(ValueError):
        validate_frame(reordered, T2_SPEC)


# --- Defect 2: prior_close column contract + as_of guard ---------------------------


def test_prior_close_frame_uses_close_column_not_prior_close():
    """Defect 2 (CONFIRMED): pipeline's default prior_close frame and
    baselines.b4_premarket_gap both use column `close`, not `prior_close`
    -- this used to crash the only T2 orchestration path with
    `KeyError: 'prior_close'`. Passing a `close`-column frame must work."""
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 08:00", 11.0)])
    prior_close = pd.DataFrame(
        [{"trade_date": TRADE_DATE, "ticker": "AAA", "close": 10.0, "as_of": t1_mod.decision_time_t1(TRADE_DATE)}]
    )
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["premarket_gap_pct"] == pytest.approx(0.10)


def test_prior_close_series_shortcut_still_accepted():
    """A bare per-ticker `pd.Series` (ticker -> close) is still accepted
    directly, bypassing the DataFrame column-contract/as_of guard."""
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 08:00", 11.0)])
    prior_close = pd.Series({"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    row = out[out["ticker"] == "AAA"].iloc[0]
    assert row["premarket_gap_pct"] == pytest.approx(0.10)


def test_prior_close_with_day_t_as_of_raises_instead_of_silently_wrong():
    """Defect 2 (CONFIRMED): `build_t2_features` never validated
    `prior_close`'s `as_of` -- a caller passing day-t closes (as_of ==
    trade_date's own close, long after the T1 decision time) as
    "prior_close" would silently get a wrong `premarket_gap_pct` and no
    error. Must now raise."""
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 08:00", 11.0)])
    prior_close = pd.DataFrame(
        [{"trade_date": TRADE_DATE, "ticker": "AAA", "close": 999.0, "as_of": TRADE_DATE + pd.Timedelta(hours=16)}]
    )
    with pytest.raises(LeakageError):
        t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)


# --- Defect 1: production wiring of the anti-leakage harness -----------------------


def test_build_t2_features_output_is_always_decision_time_safe():
    t1_features = _t1_for(tickers=("AAA",))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 09:00", 10.5)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0})
    out = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)
    # Would raise LeakageError if build_t2_features didn't already
    # self-check via assert_decision_time_safe before returning.
    assert_decision_time_safe(out, t2_mod.decision_time_t2(TRADE_DATE))


def test_build_t2_features_raises_on_same_day_label_identity_leak():
    t1_features = _t1_for(tickers=("AAA", "BBB"))
    pm = pd.DataFrame([_pm_bar("AAA", "2024-03-15 09:00", 10.5)])
    prior_close = _prior_close_df(t1_features, {"AAA": 10.0, "BBB": 11.0})
    out_preview = t2_mod.build_t2_features(t1_features, pm, prior_close, _empty_halts(), TRADE_DATE)

    labels_today = pd.DataFrame(
        [
            {
                "trade_date": TRADE_DATE,
                "ticker": row["ticker"],
                "rank": i + 1,
                # Exact same-day identity leak against premarket_gap_pct.
                "return_t": row["premarket_gap_pct"],
                "label": 1,
                "label_spec_version": "test",
                "as_of": TRADE_DATE,
            }
            for i, (_, row) in enumerate(out_preview.iterrows())
        ]
    )

    with pytest.raises(LeakageError):
        t2_mod.build_t2_features(
            t1_features, pm, prior_close, _empty_halts(), TRADE_DATE, labels_history=labels_today
        )
