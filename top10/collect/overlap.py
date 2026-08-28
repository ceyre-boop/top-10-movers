"""Overlap validation between the proxy label and the captured real list.

Implements docs/LABEL_SPEC.md "Proxy validation":

- Weekly, compute overlap between the proxy top 10 and the captured
  Robinhood list.
- Tune only universe filters until median overlap across 30 days is at
  least 8/10.
- Once the overlap target is met, freeze the filters and log the freeze
  date.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from top10.collect.rh_movers import parse_top_movers_tickers
from top10.config import DATA_LABELS, DATA_RAW

logger = logging.getLogger(__name__)


def overlap_at_k(proxy: list[str], real: list[str], k: int = 10) -> int:
    """Set intersection size of the top-`k` of `proxy` and `real`."""
    proxy_top = {str(t).upper() for t in (proxy or [])[:k]}
    real_top = {str(t).upper() for t in (real or [])[:k]}
    return len(proxy_top & real_top)


def _find_label_files() -> list[Path]:
    """All proxy-label parquet files under `data/labels/<spec-hash>/`.

    Tolerates `data/labels/` not existing yet (no proxy labels have been
    generated) — returns an empty list rather than raising.
    """
    if not DATA_LABELS.exists():
        return []
    return sorted(DATA_LABELS.glob("*/*.parquet"))


def _load_proxy_top10(trade_date: dt.date) -> list[str] | None:
    """The proxy label's top-10 tickers (by `rank`) for `trade_date`.

    Returns None if no proxy label data is available for that day (or at
    all) — the caller is responsible for surfacing that absence rather
    than treating it as a zero-overlap day.
    """
    files = _find_label_files()
    if not files:
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as exc:  # noqa: BLE001 - skip unreadable files
            logger.warning("_load_proxy_top10: failed to read %s: %s", f, exc)
            continue
        if "trade_date" in df.columns and "ticker" in df.columns:
            frames.append(df)

    if not frames:
        return None

    labels = pd.concat(frames, ignore_index=True)
    labels["trade_date"] = pd.to_datetime(labels["trade_date"]).dt.date
    day = labels[labels["trade_date"] == trade_date]
    if day.empty:
        return None

    if "label" in day.columns:
        day = day[day["label"] == 1]
    if "rank" in day.columns:
        day = day.sort_values("rank")

    tickers = day["ticker"].astype(str).tolist()
    return tickers or None


def _load_real_movers(trade_date: dt.date) -> list[str] | None:
    """The captured TRUE Robinhood top-movers list for `trade_date` --
    NEVER the S&P 500 feed (see rh_movers.py module docstring for why:
    scoring the small/mid-cap-heavy proxy label against a mega-cap-only
    feed would make the freeze gate unreachable by construction).

    Returns None if no capture exists for that day, OR if that day's
    capture only obtained the S&P 500 feed -- this function REFUSES to
    silently fall back to it, and logs a clear warning explaining why.
    """
    path = DATA_RAW / "rh_movers" / f"{trade_date.isoformat()}.json"
    if not path.exists():
        return None

    envelope = json.loads(path.read_text())
    if not envelope.get("top_movers_available", False):
        logger.warning(
            "_load_real_movers: %s has no TRUE top-movers capture for %s "
            "(top_movers_available=False, sp500-only or empty). Refusing to "
            "substitute the S&P 500 feed -- overlap for this day is unscoreable.",
            path,
            trade_date.isoformat(),
        )
        return None

    return parse_top_movers_tickers(envelope) or None


def _date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def weekly_report(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Per-day overlap between the proxy label top-10 and the captured
    real Robinhood list, for each trading day in `[start, end]`.

    Tolerates proxy labels being entirely absent (nothing under
    `data/labels/` yet) and says so via a log warning; the returned
    DataFrame will simply have `proxy_available=False` for every row in
    that case.
    """
    if not _find_label_files():
        logger.warning(
            "weekly_report: no proxy label files found under %s; overlap "
            "cannot be computed until proxy labels exist.",
            DATA_LABELS,
        )

    rows = []
    for trade_date in _date_range(start, end):
        proxy = _load_proxy_top10(trade_date)
        real = _load_real_movers(trade_date)
        overlap = overlap_at_k(proxy, real) if proxy is not None and real is not None else None
        rows.append(
            {
                "trade_date": trade_date,
                "proxy_available": proxy is not None,
                "real_available": real is not None,
                "overlap_at_10": overlap,
            }
        )

    df = pd.DataFrame(rows, columns=["trade_date", "proxy_available", "real_available", "overlap_at_10"])
    return df


def meets_freeze_criterion(df: pd.DataFrame, threshold: int = 8, window_days: int = 30) -> bool:
    """The §"Proxy validation" freeze gate: median overlap >= `threshold`
    (default 8/10) over the most recent `window_days` (default 30).

    False if there isn't at least one day of overlap data in the window —
    an empty/all-missing window can never meet the criterion.
    """
    if df.empty or "overlap_at_10" not in df.columns:
        return False

    window = df.sort_values("trade_date").tail(window_days)
    overlaps = window["overlap_at_10"].dropna()
    if overlaps.empty:
        return False

    return bool(overlaps.median() >= threshold)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Weekly overlap report between the proxy label and the captured Robinhood movers list."
    )
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--window-days", type=int, default=30)
    args = parser.parse_args(argv)

    _configure_logging()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    df = weekly_report(start, end)
    print(df.to_string(index=False))

    frozen = meets_freeze_criterion(df, threshold=args.threshold, window_days=args.window_days)
    print(
        f"meets_freeze_criterion(threshold={args.threshold}, "
        f"window_days={args.window_days}) = {frozen}"
    )
    if frozen:
        print(f"FREEZE CRITERION MET as of {dt.date.today().isoformat()} — log the freeze date per LABEL_SPEC.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
