from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from top10 import baselines
from top10.storage import LeakageError


# Pre-holdout (< 2023-01-01) throughout -- the seal-guard tests below use
# their own explicit 2023+ dates. Using post-holdout dates here would make
# every ordinary computation test above also have to pass an unseal_token
# it has nothing to do with.
DAY1 = dt.datetime(2020, 1, 2)
DAY2 = dt.datetime(2020, 1, 3)
DAY3 = dt.datetime(2020, 1, 4)


def test_b0_random_reproducible_with_seed():
    universe = pd.DataFrame(
        {
            "trade_date": [DAY1] * 20,
            "ticker": [f"T{i}" for i in range(20)],
            "as_of": [DAY1 - dt.timedelta(days=1)] * 20,
        }
    )
    preds_a = baselines.b0_random(universe, seed=7)
    preds_b = baselines.b0_random(universe, seed=7)
    preds_c = baselines.b0_random(universe, seed=8)

    pd.testing.assert_frame_equal(
        preds_a.sort_values("ticker").reset_index(drop=True),
        preds_b.sort_values("ticker").reset_index(drop=True),
    )
    assert len(preds_a) == 10
    assert set(preds_a["ticker"]) != set(preds_c["ticker"]) or not preds_a["score"].equals(preds_c["score"])


def test_b0_random_raises_on_leaking_as_of():
    universe = pd.DataFrame(
        {
            "trade_date": [DAY1] * 5,
            "ticker": [f"T{i}" for i in range(5)],
            "as_of": [DAY1 + dt.timedelta(days=5)] * 5,  # future, illegal
        }
    )
    with pytest.raises(LeakageError):
        baselines.b0_random(universe, seed=1)


def _labels_day(trade_date, top10_tickers, all_tickers):
    rows = []
    for i, ticker in enumerate(top10_tickers, start=1):
        rows.append(
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "rank": i,
                "return_t": 1.0 / i,
                "label": 1,
                "label_spec_version": "v1",
                "as_of": trade_date,
            }
        )
    for ticker in all_tickers:
        if ticker in top10_tickers:
            continue
        rows.append(
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "rank": np.nan,
                "return_t": 0.0,
                "label": 0,
                "label_spec_version": "v1",
                "as_of": trade_date,
            }
        )
    return pd.DataFrame(rows)


def test_b1_yesterday_repeat_scores_by_prior_rank():
    top10_day1 = [f"T{i}" for i in range(10)]
    universe = top10_day1 + [f"N{i}" for i in range(5)]
    labels_day1 = _labels_day(DAY1, top10_day1, universe)
    labels_day2 = _labels_day(DAY2, top10_day1, universe)  # content irrelevant for B1 input
    labels = pd.concat([labels_day1, labels_day2], ignore_index=True)

    preds = baselines.b1_yesterday_repeat(labels)
    day2_preds = preds[preds["trade_date"] == DAY2].sort_values("score", ascending=False)

    assert set(day2_preds["ticker"]) == set(top10_day1)
    # rank 1 (T0) should have the highest score.
    assert day2_preds.iloc[0]["ticker"] == "T0"
    assert day2_preds.iloc[-1]["ticker"] == "T9"


def test_b1_yesterday_repeat_day_one_degrades_gracefully():
    top10 = [f"T{i}" for i in range(10)]
    universe = top10 + [f"N{i}" for i in range(5)]
    labels = _labels_day(DAY1, top10, universe)  # only one day present

    preds = baselines.b1_yesterday_repeat(labels)
    assert preds.empty
    assert list(preds.columns) == ["trade_date", "ticker", "score"]


def _bars_with_vol_pattern(n_days=10, n_tickers=5):
    """High-vol ticker (T0) alternates hard; others stay flat."""
    dates = [DAY1 + dt.timedelta(days=d) for d in range(n_days)]
    rows = []
    for ticker_idx in range(n_tickers):
        ticker = f"T{ticker_idx}"
        price = 100.0
        for d_idx, d in enumerate(dates):
            if ticker_idx == 0:
                price *= 1.10 if d_idx % 2 == 0 else 1.0 / 1.10
            else:
                price *= 1.001
            rows.append(
                {
                    "trade_date": d,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000.0,
                    "dollar_volume": price * 1_000_000.0,
                    "as_of": d,
                }
            )
    return pd.DataFrame(rows)


def test_b2_realized_vol_ranks_highest_vol_ticker_first():
    bars = _bars_with_vol_pattern(n_days=10, n_tickers=5)
    preds = baselines.b2_realized_vol(bars, k=3, window=5)

    assert not preds.empty
    last_date = preds["trade_date"].max()
    top_pick = preds[preds["trade_date"] == last_date].sort_values("score", ascending=False).iloc[0]
    assert top_pick["ticker"] == "T0"


def test_b2_realized_vol_raises_on_leaking_as_of():
    bars = _bars_with_vol_pattern(n_days=8, n_tickers=3)
    # Pick a row from a date that still has a "next trading day" mapping
    # (the very last date has no next-day mapping and would be dropped),
    # and push its as_of far into the future.
    trade_dates = sorted(bars["trade_date"].unique())
    target_date = trade_dates[-2]
    row_idx = bars[(bars["trade_date"] == target_date) & (bars["ticker"] == "T0")].index[0]
    bars.loc[row_idx, "as_of"] = bars["trade_date"].max() + dt.timedelta(days=30)
    with pytest.raises(LeakageError):
        baselines.b2_realized_vol(bars, k=3, window=5)


def test_b3_earnings_x_vol_restricts_to_earnings_today():
    bars = _bars_with_vol_pattern(n_days=25, n_tickers=5)
    trade_dates = sorted(bars["trade_date"].unique())
    predict_date = trade_dates[-1]

    # Only T1 (a flat, low-vol ticker) reports earnings on predict_date.
    earnings = pd.DataFrame(
        {
            "ticker": ["T1"],
            "report_date": [predict_date],
            "session": ["bmo"],
            "announced_on": [predict_date - dt.timedelta(days=10)],
            "date_is_revisable": [False],
            "as_of": [predict_date - dt.timedelta(days=10)],
        }
    )

    preds = baselines.b3_earnings_x_vol(bars, earnings, k=10, window=5)
    day_preds = preds[preds["trade_date"] == predict_date]
    assert set(day_preds["ticker"]) == {"T1"}


def test_b4_premarket_gap_respects_dollar_volume_floor():
    trade_date = DAY1

    prior_close = pd.DataFrame(
        {
            "trade_date": [trade_date, trade_date],
            "ticker": ["BIG", "SMALL"],
            "close": [10.0, 10.0],
            "as_of": [trade_date - dt.timedelta(days=1)] * 2,
        }
    )

    premarket_bars = pd.DataFrame(
        {
            "trade_date": [trade_date, trade_date],
            "ticker": ["BIG", "SMALL"],
            "minute": [dt.datetime(2020, 1, 2, 9, 0), dt.datetime(2020, 1, 2, 9, 0)],
            "open": [12.0, 20.0],
            "high": [12.0, 20.0],
            "low": [12.0, 20.0],
            "close": [12.0, 20.0],  # SMALL has a much bigger gap %...
            "volume": [100_000.0, 1.0],  # ...but tiny premarket dollar volume.
            "trade_count": [500, 1],
            "as_of": [dt.datetime(2020, 1, 2, 9, 25), dt.datetime(2020, 1, 2, 9, 25)],
        }
    )

    preds = baselines.b4_premarket_gap(premarket_bars, prior_close, min_premarket_dollar_vol=500_000)

    assert "SMALL" not in preds["ticker"].tolist()
    assert "BIG" in preds["ticker"].tolist()


def test_b4_premarket_gap_scores_by_gap_percent():
    trade_date = DAY1
    prior_close = pd.DataFrame(
        {
            "trade_date": [trade_date, trade_date],
            "ticker": ["A", "B"],
            "close": [10.0, 10.0],
            "as_of": [trade_date - dt.timedelta(days=1)] * 2,
        }
    )
    premarket_bars = pd.DataFrame(
        {
            "trade_date": [trade_date, trade_date],
            "ticker": ["A", "B"],
            "minute": [dt.datetime(2020, 1, 2, 9, 0)] * 2,
            "open": [11.0, 15.0],
            "high": [11.0, 15.0],
            "low": [11.0, 15.0],
            "close": [11.0, 15.0],  # A: +10% gap, B: +50% gap
            "volume": [1_000_000.0, 1_000_000.0],
            "trade_count": [1000, 1000],
            "as_of": [dt.datetime(2020, 1, 2, 9, 25)] * 2,
        }
    )

    preds = baselines.b4_premarket_gap(premarket_bars, prior_close, min_premarket_dollar_vol=100).sort_values(
        "score", ascending=False
    )
    assert preds.iloc[0]["ticker"] == "B"
    assert preds.iloc[0]["score"] == pytest.approx(50.0)
    assert preds.iloc[1]["score"] == pytest.approx(10.0)


def test_run_all_baselines_returns_all_keys():
    top10 = [f"T{i}" for i in range(10)]
    universe_tickers = top10 + [f"N{i}" for i in range(5)]
    labels_day1 = _labels_day(DAY1, top10, universe_tickers)
    labels_day2 = _labels_day(DAY2, top10, universe_tickers)
    labels = pd.concat([labels_day1, labels_day2], ignore_index=True)

    universe = pd.DataFrame(
        {
            "trade_date": [DAY1] * len(universe_tickers),
            "ticker": universe_tickers,
            "as_of": [DAY1 - dt.timedelta(days=1)] * len(universe_tickers),
        }
    )
    bars = _bars_with_vol_pattern(n_days=10, n_tickers=5)
    earnings = pd.DataFrame(
        columns=["ticker", "report_date", "session", "announced_on", "date_is_revisable", "as_of"]
    )
    premarket_bars = pd.DataFrame(
        {
            "trade_date": [DAY1],
            "ticker": ["T0"],
            "minute": [dt.datetime(2020, 1, 2, 9, 0)],
            "open": [11.0],
            "high": [11.0],
            "low": [11.0],
            "close": [11.0],
            "volume": [1_000_000.0],
            "trade_count": [100],
            "as_of": [dt.datetime(2020, 1, 2, 9, 25)],
        }
    )
    prior_close = pd.DataFrame(
        {
            "trade_date": [DAY1],
            "ticker": ["T0"],
            "close": [10.0],
            "as_of": [DAY1 - dt.timedelta(days=1)],
        }
    )

    result = baselines.run_all_baselines(
        universe=universe,
        labels=labels,
        bars=bars,
        earnings=earnings,
        premarket_bars=premarket_bars,
        prior_close=prior_close,
    )
    assert set(result.keys()) == {"B0", "B1", "B2", "B3", "B4"}
    for frame in result.values():
        assert list(frame.columns) == ["trade_date", "ticker", "score"]


# --- Defect 1: B4 must self-clip to minute < 09:25 ET -----------------------
# (docs/PREREG_TOP10.md's primary success claim is "beats B4 by >= 1.0
# average hits/day" -- a silently inflated B4 invalidates that claim in
# either direction, so B4 must never trust the caller's/vendor's own
# premarket windowing contract, exactly like top10.features.t2 doesn't.)


def test_b4_premarket_gap_excludes_planted_post_cutoff_bar():
    """Exact confirmed repro: a legitimate 09:00 premarket bar (close 10.0)
    plus a planted 15:59 bar (close 50.0), both stamped as_of = trade_date
    midnight (so `_assert_safe_per_day` -- which only inspects `as_of`, not
    `minute` -- lets both through). Before the fix, B4 aggregated both and
    returned score=400.0; it must return 0.0 (only the 09:00 bar counts)."""
    trade_date = DAY1
    prior_close = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ticker": ["X"],
            "close": [10.0],
            "as_of": [trade_date - dt.timedelta(days=1)],
        }
    )
    premarket_bars = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "X",
                "minute": trade_date + dt.timedelta(hours=9),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100_000.0,
                "trade_count": 10,
                "as_of": trade_date,
            },
            {
                "trade_date": trade_date,
                "ticker": "X",
                "minute": trade_date + dt.timedelta(hours=15, minutes=59),
                "open": 50.0,
                "high": 50.0,
                "low": 50.0,
                "close": 50.0,
                "volume": 100_000.0,
                "trade_count": 10,
                "as_of": trade_date,
            },
        ]
    )

    preds = baselines.b4_premarket_gap(premarket_bars, prior_close, min_premarket_dollar_vol=1.0)

    assert preds.loc[preds["ticker"] == "X", "score"].iloc[0] == pytest.approx(0.0)


def test_b4_premarket_gap_excludes_09_25_30_bar():
    """A bar timestamped 09:25:30 (inside the 09:25:00-09:25:59 minute) is
    NOT yet knowable at the 09:25 decision instant -- the boundary is
    exclusive."""
    trade_date = DAY1
    prior_close = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ticker": ["X"],
            "close": [10.0],
            "as_of": [trade_date - dt.timedelta(days=1)],
        }
    )
    decision_time = trade_date + dt.timedelta(hours=9, minutes=25)
    premarket_bars = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "X",
                "minute": trade_date + dt.timedelta(hours=9),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100_000.0,
                "trade_count": 10,
                "as_of": decision_time,
            },
            {
                "trade_date": trade_date,
                "ticker": "X",
                "minute": trade_date + dt.timedelta(hours=9, minutes=25, seconds=30),
                "open": 999.0,
                "high": 999.0,
                "low": 999.0,
                "close": 999.0,
                "volume": 100_000.0,
                "trade_count": 10,
                "as_of": decision_time,
            },
        ]
    )

    preds = baselines.b4_premarket_gap(premarket_bars, prior_close, min_premarket_dollar_vol=1.0)

    assert preds.loc[preds["ticker"] == "X", "score"].iloc[0] == pytest.approx(0.0)


def test_b4_and_t2_agree_on_09_25_boundary():
    """B4's re-clip must agree exactly with `top10.features.t2`'s
    `minute < cutoff` clip (cutoff = trade_date + 09:25) -- same last
    in-window bar, same excluded boundary bar, same resulting gap."""
    from top10.features import t1 as t1_mod
    from top10.features import t2 as t2_mod

    trade_date = pd.Timestamp("2020-03-16")
    ticker = "AAA"

    def _daily_bar(td, close):
        td = pd.Timestamp(td)
        return {
            "trade_date": td,
            "ticker": ticker,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000_000.0,
            "dollar_volume": close * 1_000_000.0,
            "as_of": td,
        }

    dates = pd.bdate_range(end=trade_date - pd.Timedelta(days=1), periods=25)
    daily_bars = pd.DataFrame([_daily_bar(d, 10.0) for d in dates])
    ticker_meta = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 5e9,
                "float_shares": 1e8,
                "short_interest_pct_float": None,
                "days_to_cover": None,
                "as_of": pd.Timestamp("2020-01-01"),
            }
        ]
    )
    empty_earnings = pd.DataFrame(
        columns=["ticker", "report_date", "session", "announced_on", "date_is_revisable", "as_of"]
    )
    empty_labels_history = pd.DataFrame(
        columns=["trade_date", "ticker", "rank", "return_t", "label", "label_spec_version", "as_of"]
    )
    empty_market_context = pd.DataFrame(
        columns=[
            "trade_date",
            "spy_ret_1d",
            "spy_ret_5d",
            "vix_level",
            "iwm_minus_spy_1d",
            "movers_10pct_count",
            "as_of",
        ]
    )
    empty_halts = pd.DataFrame(columns=["ticker", "trade_date", "as_of"])

    t1_features = t1_mod.build_t1_features(
        daily_bars, ticker_meta, empty_earnings, empty_labels_history, empty_market_context, trade_date
    )

    prior_close_val = 10.0
    prior_close_series = pd.Series({ticker: prior_close_val}, name="prior_close")

    in_window_minute = trade_date + pd.Timedelta(hours=9, minutes=24, seconds=59)
    boundary_minute = trade_date + pd.Timedelta(hours=9, minutes=25)
    premarket_bars = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "minute": in_window_minute,
                "open": 12.0,
                "high": 12.0,
                "low": 12.0,
                "close": 12.0,
                "volume": 100_000.0,
                "trade_count": 50,
                "as_of": boundary_minute,
            },
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "minute": boundary_minute,  # excluded by both -- `minute < cutoff` is strict
                "open": 50.0,
                "high": 50.0,
                "low": 50.0,
                "close": 50.0,
                "volume": 100_000.0,
                "trade_count": 50,
                "as_of": boundary_minute,
            },
        ]
    )

    t2_features = t2_mod.build_t2_features(t1_features, premarket_bars, prior_close_series, empty_halts, trade_date)
    t2_gap = t2_features.loc[t2_features["ticker"] == ticker, "premarket_gap_pct"].iloc[0]
    expected_gap = 12.0 / prior_close_val - 1.0
    assert t2_gap == pytest.approx(expected_gap)

    prior_close_df = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ticker": [ticker],
            "close": [prior_close_val],
            "as_of": [trade_date - dt.timedelta(days=1)],
        }
    )
    b4_preds = baselines.b4_premarket_gap(premarket_bars, prior_close_df, min_premarket_dollar_vol=1.0)
    b4_score = b4_preds.loc[b4_preds["ticker"] == ticker, "score"].iloc[0]
    assert b4_score == pytest.approx(expected_gap * 100.0)


# --- Defect 2: b0_random on a universe stamped as_of = (t-1) 16:00 ET -------


def test_b0_random_succeeds_on_universe_stamped_prior_close_as_of():
    """Coordinates with the concurrent fix to `labels.build_universe`,
    which is being changed to stamp `as_of` at the prior trading day's
    16:00 ET close (not `trade_date` midnight). This is the assumption
    this test bakes in: `as_of = (t-1) 16:00`, which must satisfy
    `_t1_decision_time` (`<=` (t-1) 16:00) and therefore must NOT raise."""
    trade_date = DAY2
    prior_close_16h = (pd.Timestamp(DAY2) - pd.Timedelta(days=1)).replace(hour=16, minute=0)
    universe = pd.DataFrame(
        {
            "trade_date": [trade_date] * 5,
            "ticker": [f"T{i}" for i in range(5)],
            "as_of": [prior_close_16h] * 5,
        }
    )

    preds = baselines.b0_random(universe, seed=1)

    assert not preds.empty
    assert set(preds["ticker"]).issubset({f"T{i}" for i in range(5)})


# --- Defect 3: baselines.run_all_baselines is a holdout-seal chokepoint ----


def _holdout_baseline_frames():
    hd1 = dt.datetime(2023, 6, 1)
    hd2 = dt.datetime(2023, 6, 2)
    top10 = [f"T{i}" for i in range(10)]
    universe_tickers = top10 + [f"N{i}" for i in range(5)]
    labels_day1 = _labels_day(hd1, top10, universe_tickers)
    labels_day2 = _labels_day(hd2, top10, universe_tickers)
    labels = pd.concat([labels_day1, labels_day2], ignore_index=True)

    universe = pd.DataFrame(
        {
            "trade_date": [hd1] * len(universe_tickers),
            "ticker": universe_tickers,
            "as_of": [hd1 - dt.timedelta(days=1)] * len(universe_tickers),
        }
    )
    bars = _bars_with_vol_pattern(n_days=10, n_tickers=5)
    bars["trade_date"] = bars["trade_date"] + (hd1 - DAY1)
    bars["as_of"] = bars["as_of"] + (hd1 - DAY1)
    earnings = pd.DataFrame(
        columns=["ticker", "report_date", "session", "announced_on", "date_is_revisable", "as_of"]
    )
    premarket_bars = pd.DataFrame(
        {
            "trade_date": [hd1],
            "ticker": ["T0"],
            "minute": [hd1 + dt.timedelta(hours=9)],
            "open": [11.0],
            "high": [11.0],
            "low": [11.0],
            "close": [11.0],
            "volume": [1_000_000.0],
            "trade_count": [100],
            "as_of": [hd1 + dt.timedelta(hours=9, minutes=25)],
        }
    )
    prior_close = pd.DataFrame(
        {
            "trade_date": [hd1],
            "ticker": ["T0"],
            "close": [10.0],
            "as_of": [hd1 - dt.timedelta(days=1)],
        }
    )
    return dict(
        universe=universe, labels=labels, bars=bars, earnings=earnings, premarket_bars=premarket_bars, prior_close=prior_close
    )


def test_run_all_baselines_raises_on_holdout_without_token():
    frames = _holdout_baseline_frames()
    with pytest.raises(LeakageError):
        baselines.run_all_baselines(**frames)


def test_run_all_baselines_succeeds_on_holdout_with_token():
    frames = _holdout_baseline_frames()
    result = baselines.run_all_baselines(**frames, unseal_token="PREREG_FROZEN")
    assert set(result.keys()) == {"B0", "B1", "B2", "B3", "B4"}
