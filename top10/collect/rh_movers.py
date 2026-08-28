"""Capture Robinhood's real movers feeds.

Implements docs/LABEL_SPEC.md "Proxy validation": starting on live
collection day, at 16:05 ET, capture the real Robinhood top movers list and
store the raw response, timestamped, without editing. There is no
historical archive of this list — every day this doesn't run is a
permanently lost data point, so resilience matters more than elegance.

IMPORTANT — there are TWO different Robinhood feeds in play, and they must
never be confused:

- `fetch_sp500_movers` / `_source: "robinhood_sp500"` — Robinhood's S&P 500
  movers feed (`midlands/movers/sp500/`). Mega-cap only. This is captured
  because it's free and may be useful context, but it is NOT the list
  docs/LABEL_SPEC.md's "Proxy validation" gate is about. The label spec's
  universe (>=$1.00, >=$1M 20d ADV) is dominated by small/mid caps; scoring
  the proxy label against the S&P 500 feed would produce ~0/10 overlap
  forever BY CONSTRUCTION and make the freeze gate permanently
  unreachable. `overlap.py` refuses to score against this feed.
- `fetch_top_movers` / `_source: "robinhood_top_movers"` — Robinhood's
  curated "Top Movers" tag (`midlands/tags/tag/top-movers/`). THIS is the
  real target of the proxy-validation gate. It returns instrument URLs
  rather than tickers, which are resolved to symbols and cached (the
  instrument -> symbol mapping is stable, so re-resolving daily would be
  wasteful).

Each feed is fetched via two independent paths, tried in order, because
Robinhood's unofficial API breaks periodically:

1. the `robin_stocks` SDK (optional dependency, `pip install .[collect]`)
2. a direct HTTPS request against the public, unauthenticated endpoint,
   with a browser-like User-Agent

Never mutate or normalize a fetched payload before storing it — §1.3 of
the label spec requires the raw response, unedited. (Instrument-URL ->
symbol resolution for the top-movers feed is stored alongside the raw
payload as a separate `symbols` field, not a mutation of it.)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import requests

from top10.config import DATA_RAW, ET, et_now
from top10.storage import append_only_write

logger = logging.getLogger(__name__)

# --- S&P 500 movers feed (context only -- NOT the proxy-validation source) --

# Public, unauthenticated Robinhood S&P 500 movers endpoint. This is the
# same endpoint the `robin_stocks` SDK's `get_top_movers_sp500` hits, and
# it requires no auth token (verified by direct curl).
_SP500_MOVERS_URL = "https://api.robinhood.com/midlands/movers/sp500/"

# --- True "Top Movers" feed (the actual proxy-validation source) -----------

# Public, unauthenticated Robinhood curated "Top Movers" tag. Returns
# instrument URLs (verified by direct curl), not tickers -- see
# _resolve_instrument_symbols. This is the endpoint the `robin_stocks`
# SDK's `get_top_movers` hits internally.
_TOP_MOVERS_TAG_URL = "https://api.robinhood.com/midlands/tags/tag/top-movers/"

_INSTRUMENT_ID_RE = re.compile(r"/instruments/([0-9a-fA-F-]+)/?$")

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

_REQUEST_TIMEOUT_S = 15


def _http_get_json(url: str, *, params: dict[str, str] | None = None) -> Any:
    resp = requests.get(
        url,
        params=params,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        timeout=_REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


# --- instrument URL -> symbol resolution (cached; the mapping is stable) --


def _instrument_cache_path() -> Path:
    # Lives under DATA_RAW/rh_movers/ alongside the daily captures, but is
    # NOT itself a point-in-time capture -- it's a stable, mutable cache
    # and is intentionally exempt from append-only semantics.
    return DATA_RAW / "rh_movers" / "instrument_symbol_cache.json"


def _load_instrument_cache() -> dict[str, str]:
    path = _instrument_cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - a corrupt cache is not fatal
        logger.warning("_load_instrument_cache: failed to read %s: %s", path, exc)
        return {}


def _save_instrument_cache(cache: dict[str, str]) -> None:
    path = _instrument_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _resolve_instrument_symbols(instrument_urls: list[str]) -> list[str]:
    """Resolve instrument URLs to ticker symbols, preserving order.

    The instrument -> symbol mapping is stable (an instrument's symbol
    doesn't change), so resolutions are cached on disk at
    `DATA_RAW/rh_movers/instrument_symbol_cache.json` to avoid re-resolving
    the same instrument via an extra HTTP request every day.
    """
    cache = _load_instrument_cache()
    symbols: list[str] = []
    dirty = False

    for url in instrument_urls:
        inst_id = _instrument_id_from_url(url)
        cached = cache.get(inst_id) if inst_id else None
        if cached:
            symbols.append(cached)
            continue

        try:
            data = _http_get_json(url)
        except Exception as exc:  # noqa: BLE001 - skip unresolvable instruments
            logger.warning("_resolve_instrument_symbols: failed to resolve %s: %s", url, exc)
            continue

        symbol = data.get("symbol") if isinstance(data, dict) else None
        if not symbol:
            continue
        symbol = str(symbol).upper()
        symbols.append(symbol)
        if inst_id:
            cache[inst_id] = symbol
            dirty = True

    if dirty:
        _save_instrument_cache(cache)

    return symbols


def _instrument_id_from_url(url: str) -> str | None:
    match = _INSTRUMENT_ID_RE.search(url)
    return match.group(1) if match else None


# --- S&P 500 movers feed ----------------------------------------------------


def _fetch_sp500_via_robin_stocks(direction: str) -> Any | None:
    """Imported lazily so this module stays importable without the SDK
    installed -- it lives behind the optional `[collect]` extra."""
    from robin_stocks.robinhood import markets as rs_markets  # type: ignore[import-not-found]

    movers: Any = None
    try:
        movers = rs_markets.get_top_movers(direction)
    except TypeError:
        # Some robin_stocks releases don't accept a `direction` argument on
        # get_top_movers (only get_top_movers_sp500 does) -- fall through.
        movers = None

    if not movers or movers in ([None],):
        movers = rs_markets.get_top_movers_sp500(direction)

    if not movers or movers in ([None],):
        return None
    return movers


def _fetch_sp500_via_https(direction: str) -> dict[str, Any] | None:
    data = _http_get_json(_SP500_MOVERS_URL, params={"direction": direction})
    if not data or not data.get("results"):
        return None
    return data


def fetch_sp500_movers(direction: str = "up") -> dict[str, Any]:
    """Fetch Robinhood's S&P 500 movers feed for `direction` ("up" or
    "down"). Context only -- this is NOT the docs/LABEL_SPEC.md proxy-
    validation source; use `fetch_top_movers` for that.

    Returns `{"_source": "robinhood_sp500", "_fetch_path": "robin_stocks" |
    "https", "payload": <raw response, unedited>}`. Raises RuntimeError if
    both fetch paths fail.
    """
    try:
        payload = _fetch_sp500_via_robin_stocks(direction)
        if payload:
            logger.info("fetch_sp500_movers: robin_stocks path succeeded (direction=%s)", direction)
            return {"_source": "robinhood_sp500", "_fetch_path": "robin_stocks", "payload": payload}
        logger.warning("fetch_sp500_movers: robin_stocks path returned no data")
    except Exception as exc:  # noqa: BLE001 - any SDK failure must fall through
        logger.warning("fetch_sp500_movers: robin_stocks path raised %s: %s", type(exc).__name__, exc)

    try:
        payload = _fetch_sp500_via_https(direction)
        if payload:
            logger.info("fetch_sp500_movers: https fallback succeeded (direction=%s)", direction)
            return {"_source": "robinhood_sp500", "_fetch_path": "https", "payload": payload}
        logger.error("fetch_sp500_movers: https fallback returned no data")
    except Exception as exc:  # noqa: BLE001
        logger.error("fetch_sp500_movers: https fallback raised %s: %s", type(exc).__name__, exc)

    raise RuntimeError(
        f"fetch_sp500_movers: both robin_stocks and https fallback paths failed for direction={direction!r}"
    )


# --- True "Top Movers" feed (the proxy-validation source) ------------------


def _fetch_top_movers_via_robin_stocks() -> Any | None:
    from robin_stocks.robinhood import markets as rs_markets  # type: ignore[import-not-found]

    movers = rs_markets.get_top_movers()
    if not movers or movers in ([None],):
        return None
    return movers


def _symbols_from_rs_top_movers_payload(payload: Any) -> list[str]:
    """robin_stocks's `get_top_movers()` returns a list of dicts (each
    merged with quote data), keyed by `symbol`, in the tag's order."""
    symbols: list[str] = []
    for item in payload or []:
        if isinstance(item, dict):
            symbol = item.get("symbol")
            if symbol:
                symbols.append(str(symbol).upper())
    return symbols


def _fetch_top_movers_via_https() -> tuple[dict[str, Any], list[str]] | tuple[None, None]:
    data = _http_get_json(_TOP_MOVERS_TAG_URL)
    instrument_urls = (data or {}).get("instruments") or []
    if not instrument_urls:
        return None, None
    symbols = _resolve_instrument_symbols(instrument_urls)
    if not symbols:
        return None, None
    return data, symbols


def fetch_top_movers() -> dict[str, Any]:
    """Fetch Robinhood's TRUE curated "Top Movers" list -- the actual
    target of docs/LABEL_SPEC.md's proxy-validation gate. This is a single
    combined list (no up/down split exists at this endpoint -- verified
    live; `top-movers-up`/`top-movers-down`/etc. all 404).

    Returns `{"_source": "robinhood_top_movers", "_fetch_path":
    "robin_stocks" | "https", "payload": <raw response, unedited>,
    "symbols": [<ticker>, ...]}` (symbols resolved and order-preserved).
    Raises RuntimeError if both fetch paths fail.
    """
    try:
        payload = _fetch_top_movers_via_robin_stocks()
        symbols = _symbols_from_rs_top_movers_payload(payload) if payload else []
        if symbols:
            logger.info("fetch_top_movers: robin_stocks path succeeded")
            return {
                "_source": "robinhood_top_movers",
                "_fetch_path": "robin_stocks",
                "payload": payload,
                "symbols": symbols,
            }
        logger.warning("fetch_top_movers: robin_stocks path returned no data")
    except Exception as exc:  # noqa: BLE001 - any SDK failure must fall through
        logger.warning("fetch_top_movers: robin_stocks path raised %s: %s", type(exc).__name__, exc)

    try:
        raw, symbols = _fetch_top_movers_via_https()
        if symbols:
            logger.info("fetch_top_movers: https fallback succeeded")
            return {
                "_source": "robinhood_top_movers",
                "_fetch_path": "https",
                "payload": raw,
                "symbols": symbols,
            }
        logger.error("fetch_top_movers: https fallback returned no data")
    except Exception as exc:  # noqa: BLE001
        logger.error("fetch_top_movers: https fallback raised %s: %s", type(exc).__name__, exc)

    raise RuntimeError(
        "fetch_top_movers: both robin_stocks and https fallback paths failed for the TRUE top-movers feed"
    )


# --- capture -----------------------------------------------------------------


def capture(date: dt.date | None = None, direction: str = "up") -> Path:
    """Capture today's (or `date`'s) Robinhood movers feeds.

    Fetches BOTH the S&P 500 feed (context only) and the TRUE top-movers
    feed (the proxy-validation source), storing each raw and unedited
    under its own clearly separated key so they can never be conflated.

    Succeeds as long as at least one feed was captured. If the TRUE
    top-movers feed specifically could not be fetched, capture still
    succeeds on the S&P 500 feed alone but emits a loud warning and
    records `top_movers_available: false` -- the absence must never be
    papered over, since anything scoring against a missing true list must
    refuse rather than silently substitute the S&P 500 feed (see
    `overlap.py`). Raises RuntimeError only if BOTH feeds fail entirely.

    Writes append-only to `DATA_RAW/rh_movers/YYYY-MM-DD.json`. If that
    day's file already exists, the existing path is returned without
    re-fetching or overwriting -- temporal pre-commitment means a capture,
    once written, can never be edited.
    """
    trade_date = date or et_now().date()
    out_dir = DATA_RAW / "rh_movers"
    out_path = out_dir / f"{trade_date.isoformat()}.json"

    if out_path.exists():
        logger.warning(
            "capture: %s already exists; skipping (append-only, will not overwrite).",
            out_path,
        )
        return out_path

    sp500_result: dict[str, Any] | None = None
    sp500_error: Exception | None = None
    try:
        sp500_result = fetch_sp500_movers(direction)
    except Exception as exc:  # noqa: BLE001
        sp500_error = exc
        logger.error("capture: sp500 feed failed: %s: %s", type(exc).__name__, exc)

    top_movers_result: dict[str, Any] | None = None
    top_movers_error: Exception | None = None
    try:
        top_movers_result = fetch_top_movers()
    except Exception as exc:  # noqa: BLE001
        top_movers_error = exc
        logger.error("capture: top_movers feed failed: %s: %s", type(exc).__name__, exc)

    if sp500_result is None and top_movers_result is None:
        raise RuntimeError(
            f"capture: BOTH feeds failed for {trade_date.isoformat()} -- "
            f"sp500 error={sp500_error!r}, top_movers error={top_movers_error!r}. "
            "Nothing captured; this day's data point is PERMANENTLY LOST."
        )

    if top_movers_result is None:
        logger.warning(
            "capture: *** PRIMARY PROXY-VALIDATION SOURCE UNAVAILABLE for %s *** "
            "-- the TRUE Robinhood Top Movers feed could not be fetched (error=%r). "
            "Only the S&P 500 feed (context only, NOT usable for proxy validation) "
            "was captured this day. Per the plan's §1.3, the remaining fallback is "
            "a screenshot+OCR capture -- not implemented here.",
            trade_date.isoformat(),
            top_movers_error,
        )

    now_utc = dt.datetime.now(dt.timezone.utc)
    now_et = now_utc.astimezone(ET)

    envelope = {
        "captured_at_utc": now_utc.isoformat(),
        "captured_at_et": now_et.isoformat(),
        "trade_date": trade_date.isoformat(),
        "sp500": (
            {
                "_source": sp500_result["_source"],
                "_fetch_path": sp500_result["_fetch_path"],
                "available": True,
                "payload": sp500_result["payload"],
            }
            if sp500_result is not None
            else {"_source": "robinhood_sp500", "_fetch_path": None, "available": False, "payload": None}
        ),
        "top_movers": (
            {
                "_source": top_movers_result["_source"],
                "_fetch_path": top_movers_result["_fetch_path"],
                "available": True,
                "payload": top_movers_result["payload"],
                "symbols": top_movers_result["symbols"],
            }
            if top_movers_result is not None
            else {
                "_source": "robinhood_top_movers",
                "_fetch_path": None,
                "available": False,
                "payload": None,
                "symbols": None,
            }
        ),
        "top_movers_available": top_movers_result is not None,
    }

    append_only_write(envelope, out_path)
    logger.info(
        "capture: wrote %s (top_movers_available=%s)",
        out_path,
        envelope["top_movers_available"],
    )
    return out_path


def load_captured_movers(date: dt.date) -> dict[str, Any] | None:
    """Load the captured envelope for `date` from
    `DATA_RAW/rh_movers/YYYY-MM-DD.json`.

    Returns None (not an exception) when that date was never captured --
    gaps are expected and normal for a forward-only collection program.

    The returned dict always carries `top_movers_available` (bool) and a
    `top_movers_tickers` convenience key: the resolved TRUE top-movers
    ticker list (order-preserved) when available, or explicitly None when
    it isn't -- e.g. when only the S&P 500 context feed was captured that
    day. Callers must check `top_movers_available` /
    `top_movers_tickers is not None` before using it; this function never
    substitutes the S&P 500 feed for the true list.
    """
    path = DATA_RAW / "rh_movers" / f"{date.isoformat()}.json"
    if not path.exists():
        return None

    envelope = json.loads(path.read_text())
    envelope["top_movers_tickers"] = (
        parse_top_movers_tickers(envelope) if envelope.get("top_movers_available") else None
    )
    return envelope


# --- parsing -----------------------------------------------------------------


def _extract_symbols_from_raw(payload: Any) -> list[str]:
    """Tolerant, order-preserving symbol extraction from a raw feed
    payload shape (list-of-dicts, or dict with a `results` list-of-dicts)."""
    if isinstance(payload, dict):
        items = payload.get("results") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    symbols: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol") or item.get("ticker")
        if symbol:
            symbols.append(str(symbol).upper())
    return symbols


def parse_sp500_tickers(envelope: dict[str, Any]) -> list[str]:
    """Tolerant extraction of tickers from the S&P 500 section of a
    capture envelope (or a raw sp500 payload). Context data only -- never
    use this for proxy-validation overlap scoring; use
    `parse_top_movers_tickers` for that."""
    section = envelope.get("sp500", envelope) if isinstance(envelope, dict) else envelope
    payload = section.get("payload", section) if isinstance(section, dict) else section
    return _extract_symbols_from_raw(payload)


def parse_top_movers_tickers(envelope: dict[str, Any]) -> list[str]:
    """Order-preserving extraction of the TRUE top-movers ticker list from
    a capture envelope. Symbols are already resolved at capture time (see
    `fetch_top_movers` / `_resolve_instrument_symbols`) and stored under
    `envelope["top_movers"]["symbols"]`. Returns `[]` if unavailable --
    callers that need to distinguish "empty" from "never captured" should
    check `envelope.get("top_movers_available")` first."""
    section = envelope.get("top_movers", envelope) if isinstance(envelope, dict) else envelope
    if isinstance(section, dict) and section.get("symbols"):
        return [str(s).upper() for s in section["symbols"]]
    return []


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture Robinhood's movers feeds (true top-movers + S&P 500 context) for proxy-label validation."
    )
    parser.add_argument("--direction", choices=["up", "down"], default="up", help="Direction for the S&P 500 feed.")
    parser.add_argument(
        "--date",
        default=None,
        help="Trade date override, YYYY-MM-DD (default: today, ET).",
    )
    args = parser.parse_args(argv)

    _configure_logging()

    trade_date = dt.date.fromisoformat(args.date) if args.date else None

    try:
        path = capture(date=trade_date, direction=args.direction)
    except Exception as exc:  # noqa: BLE001 - must exit 1 so the cron alerts
        logger.error(
            "main: capture failed: %s: %s (all fetch paths for both feeds exhausted; "
            "see warnings above for per-path failure detail)",
            type(exc).__name__,
            exc,
        )
        return 1

    try:
        envelope = json.loads(path.read_text())
        top_movers_tickers = parse_top_movers_tickers(envelope) if envelope.get("top_movers_available") else []
        sp500_tickers = parse_sp500_tickers(envelope)
    except Exception as exc:  # noqa: BLE001 - reporting only, not fatal
        logger.warning("main: could not parse tickers from %s for logging: %s", path, exc)
        top_movers_tickers, sp500_tickers = [], []

    logger.info(
        "main: capture complete -> %s (top_movers: %d tickers, sp500: %d tickers)",
        path,
        len(top_movers_tickers),
        len(sp500_tickers),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
