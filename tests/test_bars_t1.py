from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from top10.features.bars_t1 import BARS_T1_FEATURES, build_bars_t1
from top10.features.spec import T1B_COLUMNS, T1B_SPEC, validate_frame
from top10.features.t1 import decision_time_t1
from top10.leakage import assert_decision_time_safe


def _make_ticker_frame(ticker: str, n: int, *, start: str = "2020-01-01", close0: float = 10.0, drift: float = 0.01, volume: float = 1_000_000.0) -> pd.DataFrame:
    """`n` consecutive business days for `ticker` with a deterministic,
    constant per-day pct-change `drift` (so `ret`/`ret5`/`ret20` etc. have
    exact, hand-computable values), constant volume/adv20 (so `relvol_prev`
    is trivially 1.0 throughout), and `label` all zero (overridable by the
    caller after construction)."""
    dates = pd.bdate_range(start=start, periods=n + 1)
    trade_dates = dates[1:]
    prev_dates = dates[:-1]

    closes = close0 * (1.0 + drift) ** np.arange(n + 1)
    close = closes[1:]
    prev_close = closes[:-1]
    ret = close / prev_close - 1.0

    return pd.DataFrame(
        {
            "trade_date": trade_dates,
            "ticker": ticker,
            "close": close,
            "prev_close": prev_close,
            "prev_date": prev_dates,
            "volume": volume,
            "dollar_volume": close * volume,
            "adv20": volume * close,  # constant -> relvol_prev == 1.0 everywhere
            "ret": ret,
            "label": 0,
            "rank": np.arange(1, n + 1),
            "high": close,
            "low": close,
        }
    )


def _universe(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


# --- schema / decision-time safety ------------------------------------------


def test_output_columns_match_t1b_spec_order():
    universe = _make_ticker_frame("AAA", 5)
    out = build_bars_t1(universe)
    assert list(out.columns) == list(T1B_COLUMNS)
    validate_frame(out, T1B_SPEC)


def test_feature_order_is_fixed_and_has_15_names():
    assert len(BARS_T1_FEATURES) == 15
    assert len(set(BARS_T1_FEATURES)) == 15


def test_as_of_is_16h_on_prev_date():
    universe = _make_ticker_frame("AAA", 5)
    out = build_bars_t1(universe)
    expected = pd.to_datetime(universe["prev_date"]) + pd.Timedelta(hours=16)
    assert (out["as_of"].to_numpy() == expected.to_numpy()).all()


def test_output_is_always_decision_time_safe():
    universe = _make_ticker_frame("AAA", 80)
    out = build_bars_t1(universe)
    for trade_date, group in out.groupby("trade_date"):
        assert_decision_time_safe(group, decision_time_t1(trade_date))


def test_missing_required_column_raises():
    universe = _make_ticker_frame("AAA", 5).drop(columns=["adv20"])
    with pytest.raises(KeyError):
        build_bars_t1(universe)


def test_empty_universe_returns_empty_frame_with_correct_columns():
    universe = _make_ticker_frame("AAA", 0)
    out = build_bars_t1(universe)
    assert out.empty
    assert list(out.columns) == list(T1B_COLUMNS)


# --- feature correctness -----------------------------------------------------


def test_prev_ret1_is_ret_shifted_one_day():
    universe = _make_ticker_frame("AAA", 10, drift=0.02)
    out = build_bars_t1(universe)
    expected = universe["ret"].shift(1).to_numpy()
    np.testing.assert_allclose(out["prev_ret1"].to_numpy()[1:], expected[1:])
    assert pd.isna(out["prev_ret1"].iloc[0])


def test_logpx_is_log_of_prior_close():
    universe = _make_ticker_frame("AAA", 10)
    out = build_bars_t1(universe)
    expected = np.log(universe["close"].shift(1))
    np.testing.assert_allclose(out["logpx"].to_numpy()[1:], expected.to_numpy()[1:])


def test_relvol_prev_is_prior_day_dollar_volume_over_adv20():
    universe = _make_ticker_frame("AAA", 10, volume=2_000_000.0)
    out = build_bars_t1(universe)
    # constant volume/close ratio in the fixture -> exactly 1.0 every day
    # except the very first (no prior day).
    np.testing.assert_allclose(out["relvol_prev"].to_numpy()[1:], 1.0)
    assert pd.isna(out["relvol_prev"].iloc[0])


def test_logdv_is_log1p_of_prior_dollar_volume():
    universe = _make_ticker_frame("AAA", 10)
    out = build_bars_t1(universe)
    expected = np.log1p(universe["dollar_volume"].shift(1))
    np.testing.assert_allclose(out["logdv"].to_numpy()[1:], expected.to_numpy()[1:])


def test_was_top_is_prior_day_label():
    universe = _make_ticker_frame("AAA", 10)
    universe.loc[universe.index[3], "label"] = 1
    out = build_bars_t1(universe)
    expected = universe["label"].shift(1).to_numpy()
    np.testing.assert_allclose(out["was_top"].to_numpy()[1:], expected[1:])


def test_top30_and_top90_are_shifted_rolling_sums_of_label():
    universe = _make_ticker_frame("AAA", 40)
    for i in (2, 5, 10, 20, 25):
        universe.loc[universe.index[i], "label"] = 1
    out = build_bars_t1(universe)

    expected_30 = universe["label"].rolling(window=30, min_periods=1).sum().shift(1)
    expected_90 = universe["label"].rolling(window=90, min_periods=1).sum().shift(1)
    np.testing.assert_allclose(out["top30"].to_numpy()[1:], expected_30.to_numpy()[1:])
    np.testing.assert_allclose(out["top90"].to_numpy()[1:], expected_90.to_numpy()[1:])
    # A day right after a labeled day must count it.
    day_after_label = out.iloc[3]
    assert day_after_label["was_top"] == 1.0


def test_ret5_and_ret20_are_shifted_pct_changes():
    universe = _make_ticker_frame("AAA", 30, drift=0.01)
    out = build_bars_t1(universe)

    expected_5 = universe["close"].pct_change(5).shift(1)
    expected_20 = universe["close"].pct_change(20).shift(1)
    np.testing.assert_allclose(
        out["ret5"].to_numpy()[~np.isnan(expected_5.to_numpy())],
        expected_5.dropna().to_numpy(),
    )
    np.testing.assert_allclose(
        out["ret20"].to_numpy()[~np.isnan(expected_20.to_numpy())],
        expected_20.dropna().to_numpy(),
    )


def test_vol5_and_vol20_are_nan_during_warmup_then_populated():
    universe = _make_ticker_frame("AAA", 30, drift=0.0)
    universe["ret"] = universe["ret"] + np.array(
        [0.01 if i % 2 == 0 else -0.01 for i in range(len(universe))]
    )
    out = build_bars_t1(universe)

    # vol5 needs >= 3 prior shifted-ret observations -> NaN for the first
    # few rows, populated well before row 10.
    assert out["vol5"].iloc[:2].isna().all()
    assert out["vol5"].iloc[10:].notna().all()

    # vol20 needs >= 10 prior shifted-ret observations.
    assert out["vol20"].iloc[:9].isna().all()
    assert out["vol20"].iloc[25:].notna().all()


def test_volofvol_is_rolling_std_of_vol5():
    universe = _make_ticker_frame("AAA", 60, drift=0.0)
    universe["ret"] = universe["ret"] + np.sin(np.arange(len(universe))) * 0.02
    out = build_bars_t1(universe)

    shifted_ret = universe["ret"].shift(1)
    vol5 = shifted_ret.rolling(window=5, min_periods=3).std()
    expected_volofvol = vol5.rolling(window=20, min_periods=10).std()

    valid = expected_volofvol.notna().to_numpy()
    np.testing.assert_allclose(
        out["volofvol"].to_numpy()[valid], expected_volofvol.to_numpy()[valid]
    )


def test_dist52_needs_60_days_min_and_is_distance_from_prior_rolling_max():
    n = 120
    universe = _make_ticker_frame("AAA", n, drift=0.0)
    # Force a known, single high point early, then flat -- prior close
    # should sit strictly below that historical max for the rest of the
    # series once enough history exists.
    universe.loc[universe.index[10], "close"] = 100.0
    out = build_bars_t1(universe)

    assert out["dist52"].iloc[:59].isna().all()
    later = out["dist52"].iloc[70:]
    assert later.notna().all()
    assert (later <= 0.0).all()


def test_xs_ret_rank_and_xs_vol_rank_are_cross_sectional_pct_within_day():
    a = _make_ticker_frame("AAA", 30, drift=0.05)
    b = _make_ticker_frame("BBB", 30, drift=0.001)
    universe = _universe(a, b)
    out = build_bars_t1(universe)

    for trade_date, group in out.groupby("trade_date"):
        valid = group.dropna(subset=["xs_ret_rank", "xs_vol_rank"])
        if valid.empty:
            continue
        assert (valid["xs_ret_rank"] <= 1.0).all()
        assert (valid["xs_ret_rank"] > 0.0).all()
        assert (valid["xs_vol_rank"] <= 1.0).all()
        assert (valid["xs_vol_rank"] > 0.0).all()

    # AAA has a much larger constant drift than BBB -> AAA's prior return
    # should rank strictly higher on every day both are valid.
    merged = out[out["ticker"] == "AAA"].merge(
        out[out["ticker"] == "BBB"], on="trade_date", suffixes=("_a", "_b")
    )
    valid = merged.dropna(subset=["xs_ret_rank_a", "xs_ret_rank_b"])
    assert (valid["xs_ret_rank_a"] > valid["xs_ret_rank_b"]).all()
