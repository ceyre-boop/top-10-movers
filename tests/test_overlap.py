from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from top10.collect import overlap


# --- overlap_at_k -------------------------------------------------------


def test_overlap_at_k_full_match():
    proxy = ["AAA", "BBB", "CCC"]
    real = ["ccc", "bbb", "aaa"]  # order + case shouldn't matter for the set
    assert overlap.overlap_at_k(proxy, real, k=3) == 3


def test_overlap_at_k_partial_match():
    proxy = ["AAA", "BBB", "CCC", "DDD"]
    real = ["CCC", "DDD", "EEE", "FFF"]
    assert overlap.overlap_at_k(proxy, real, k=4) == 2


def test_overlap_at_k_no_match():
    assert overlap.overlap_at_k(["AAA"], ["BBB"], k=10) == 0


def test_overlap_at_k_respects_k_truncation():
    proxy = ["AAA", "BBB", "CCC"]
    real = ["CCC", "DDD", "EEE"]
    # Only top-2 of each considered.
    assert overlap.overlap_at_k(proxy, real, k=2) == 0


def test_overlap_at_k_empty_inputs():
    assert overlap.overlap_at_k([], [], k=10) == 0
    assert overlap.overlap_at_k(None, None, k=10) == 0


# --- meets_freeze_criterion ----------------------------------------------


def _df_with_constant_overlap(value, n=30, start=dt.date(2026, 1, 1)):
    dates = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame(
        {
            "trade_date": [d.date() for d in dates],
            "proxy_available": [True] * n,
            "real_available": [True] * n,
            "overlap_at_10": [value] * n,
        }
    )


def test_meets_freeze_criterion_false_at_median_7():
    df = _df_with_constant_overlap(7)
    assert overlap.meets_freeze_criterion(df, threshold=8, window_days=30) is False


def test_meets_freeze_criterion_true_at_median_8():
    df = _df_with_constant_overlap(8)
    assert overlap.meets_freeze_criterion(df, threshold=8, window_days=30) is True


def test_meets_freeze_criterion_false_on_empty_dataframe():
    df = pd.DataFrame(columns=["trade_date", "proxy_available", "real_available", "overlap_at_10"])
    assert overlap.meets_freeze_criterion(df) is False


def test_meets_freeze_criterion_false_when_all_missing():
    n = 30
    dates = pd.bdate_range(start=dt.date(2026, 1, 1), periods=n)
    df = pd.DataFrame(
        {
            "trade_date": [d.date() for d in dates],
            "proxy_available": [False] * n,
            "real_available": [False] * n,
            "overlap_at_10": [None] * n,
        }
    )
    assert overlap.meets_freeze_criterion(df) is False


def test_meets_freeze_criterion_uses_only_the_trailing_window():
    # 30 days of overlap=8 (meets), followed by 30 days of overlap=2
    # (doesn't). window_days=30 should look only at the most recent 30.
    dates_a = pd.bdate_range(start=dt.date(2026, 1, 1), periods=30)
    dates_b = pd.bdate_range(start=dates_a[-1] + pd.Timedelta(days=1), periods=30)
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "trade_date": [d.date() for d in dates_a],
                    "proxy_available": True,
                    "real_available": True,
                    "overlap_at_10": 8,
                }
            ),
            pd.DataFrame(
                {
                    "trade_date": [d.date() for d in dates_b],
                    "proxy_available": True,
                    "real_available": True,
                    "overlap_at_10": 2,
                }
            ),
        ],
        ignore_index=True,
    )
    assert overlap.meets_freeze_criterion(df, threshold=8, window_days=30) is False


# --- weekly_report --------------------------------------------------------


def _write_capture(data_raw: object, trade_date: dt.date, tickers: list[str]) -> None:
    """Write a capture envelope with a real TRUE top-movers feed present."""
    out_dir = data_raw / "rh_movers"
    out_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "captured_at_utc": "2026-08-27T20:05:00+00:00",
        "captured_at_et": "2026-08-27T16:05:00-04:00",
        "trade_date": trade_date.isoformat(),
        "sp500": {
            "_source": "robinhood_sp500",
            "_fetch_path": "https",
            "available": True,
            "payload": {"results": [{"symbol": "AAPL"}, {"symbol": "MSFT"}]},
        },
        "top_movers": {
            "_source": "robinhood_top_movers",
            "_fetch_path": "https",
            "available": True,
            "payload": {"instruments": [f"https://api.robinhood.com/instruments/{t}/" for t in tickers]},
            "symbols": tickers,
        },
        "top_movers_available": True,
    }
    (out_dir / f"{trade_date.isoformat()}.json").write_text(json.dumps(envelope))


def _write_sp500_only_capture(data_raw: object, trade_date: dt.date, sp500_tickers: list[str]) -> None:
    """Write a capture envelope where the TRUE top-movers feed was
    unavailable that day and only the S&P 500 context feed was captured."""
    out_dir = data_raw / "rh_movers"
    out_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "captured_at_utc": "2026-08-27T20:05:00+00:00",
        "captured_at_et": "2026-08-27T16:05:00-04:00",
        "trade_date": trade_date.isoformat(),
        "sp500": {
            "_source": "robinhood_sp500",
            "_fetch_path": "https",
            "available": True,
            "payload": {"results": [{"symbol": t} for t in sp500_tickers]},
        },
        "top_movers": {
            "_source": "robinhood_top_movers",
            "_fetch_path": None,
            "available": False,
            "payload": None,
            "symbols": None,
        },
        "top_movers_available": False,
    }
    (out_dir / f"{trade_date.isoformat()}.json").write_text(json.dumps(envelope))


def _write_label_file(data_labels: object, trade_date: dt.date, tickers: list[str]) -> None:
    spec_dir = data_labels / "fakehash"
    spec_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "trade_date": [trade_date] * len(tickers),
            "ticker": tickers,
            "rank": list(range(1, len(tickers) + 1)),
            "return_t": [0.1] * len(tickers),
            "label": [1] * len(tickers),
            "label_spec_version": ["v1"] * len(tickers),
        }
    )
    df.to_parquet(spec_dir / f"{trade_date.isoformat()}.parquet")


def test_weekly_report_tolerates_absent_labels(tmp_path, monkeypatch, caplog):
    data_raw = tmp_path / "raw"
    data_labels = tmp_path / "labels"  # never created
    monkeypatch.setattr(overlap, "DATA_RAW", data_raw)
    monkeypatch.setattr(overlap, "DATA_LABELS", data_labels)

    trade_date = dt.date(2026, 8, 24)  # a Monday
    _write_capture(data_raw, trade_date, ["AAA", "BBB"])

    with caplog.at_level("WARNING"):
        df = overlap.weekly_report(trade_date, trade_date)

    assert not df.empty
    row = df.iloc[0]
    assert bool(row["proxy_available"]) is False
    assert bool(row["real_available"]) is True
    assert pd.isna(row["overlap_at_10"])
    assert any("no proxy label" in msg for msg in caplog.messages)


def test_weekly_report_computes_overlap_when_both_present(tmp_path, monkeypatch):
    data_raw = tmp_path / "raw"
    data_labels = tmp_path / "labels"
    monkeypatch.setattr(overlap, "DATA_RAW", data_raw)
    monkeypatch.setattr(overlap, "DATA_LABELS", data_labels)

    trade_date = dt.date(2026, 8, 24)  # a Monday
    proxy_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
    real_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "ZZZ", "YYY", "XXX", "WWW", "VVV"]
    _write_label_file(data_labels, trade_date, proxy_tickers)
    _write_capture(data_raw, trade_date, real_tickers)

    df = overlap.weekly_report(trade_date, trade_date)
    row = df.iloc[0]
    assert bool(row["proxy_available"]) is True
    assert bool(row["real_available"]) is True
    assert row["overlap_at_10"] == 5


def test_weekly_report_marks_missing_capture(tmp_path, monkeypatch):
    data_raw = tmp_path / "raw"
    data_labels = tmp_path / "labels"
    monkeypatch.setattr(overlap, "DATA_RAW", data_raw)
    monkeypatch.setattr(overlap, "DATA_LABELS", data_labels)

    trade_date = dt.date(2026, 8, 24)
    _write_label_file(data_labels, trade_date, ["AAA", "BBB"])

    df = overlap.weekly_report(trade_date, trade_date)
    row = df.iloc[0]
    assert bool(row["proxy_available"]) is True
    assert bool(row["real_available"]) is False
    assert pd.isna(row["overlap_at_10"])


# --- refusal to score against the S&P 500 feed -----------------------------


def test_load_real_movers_refuses_sp500_only_capture(tmp_path, monkeypatch, caplog):
    data_raw = tmp_path / "raw"
    monkeypatch.setattr(overlap, "DATA_RAW", data_raw)

    trade_date = dt.date(2026, 8, 24)
    # sp500 tickers deliberately overlap heavily with what a naive fallback
    # would score as a "hit" -- if _load_real_movers ever silently fell
    # back to this feed, this test would start passing overlap scores it
    # must never produce.
    _write_sp500_only_capture(data_raw, trade_date, ["AAPL", "MSFT", "NVDA"])

    with caplog.at_level("WARNING"):
        real = overlap._load_real_movers(trade_date)

    assert real is None
    assert any("Refusing to substitute the S&P 500 feed" in msg for msg in caplog.messages)


def test_weekly_report_marks_sp500_only_day_as_real_unavailable(tmp_path, monkeypatch):
    data_raw = tmp_path / "raw"
    data_labels = tmp_path / "labels"
    monkeypatch.setattr(overlap, "DATA_RAW", data_raw)
    monkeypatch.setattr(overlap, "DATA_LABELS", data_labels)

    trade_date = dt.date(2026, 8, 24)
    proxy_tickers = [f"T{i}" for i in range(10)]
    _write_label_file(data_labels, trade_date, proxy_tickers)
    _write_sp500_only_capture(data_raw, trade_date, ["AAPL", "MSFT"])

    df = overlap.weekly_report(trade_date, trade_date)
    row = df.iloc[0]
    assert bool(row["proxy_available"]) is True
    assert bool(row["real_available"]) is False
    assert pd.isna(row["overlap_at_10"])


def test_meets_freeze_criterion_false_when_true_list_missing_even_with_high_sp500_overlap(tmp_path, monkeypatch):
    # Simulate 30 days where the true top-movers feed was NEVER captured
    # (sp500-only every day) -- the freeze gate must stay unreachable
    # rather than being (wrongly) satisfiable via the S&P 500 feed.
    data_raw = tmp_path / "raw"
    data_labels = tmp_path / "labels"
    monkeypatch.setattr(overlap, "DATA_RAW", data_raw)
    monkeypatch.setattr(overlap, "DATA_LABELS", data_labels)

    dates = [d.date() for d in pd.bdate_range(start=dt.date(2026, 1, 1), periods=30)]
    for d in dates:
        _write_label_file(data_labels, d, [f"T{i}" for i in range(10)])
        _write_sp500_only_capture(data_raw, d, ["AAPL", "MSFT", "NVDA"])

    df = overlap.weekly_report(dates[0], dates[-1])
    assert (df["real_available"] == False).all()  # noqa: E712
    assert overlap.meets_freeze_criterion(df, threshold=8, window_days=30) is False
