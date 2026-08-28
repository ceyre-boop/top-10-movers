"""On-disk raw-JSON cache for vendor API responses.

Per docs/LABEL_SPEC.md ("raw dumps are untouched"), every raw vendor payload
is persisted verbatim to disk *before* it is parsed into a DataFrame. This
module is the single place that touches the filesystem for that purpose so
every adapter gets identical caching semantics.

Layout: ``DATA_RAW/<namespace>/<key>.json`` where ``namespace`` is expected
to be vendor-scoped (e.g. ``"polygon/daily_bars"``), giving the documented
``DATA_RAW/<vendor>/<namespace>/<key>.json`` shape.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from top10.config import DATA_RAW


def _cache_path(namespace: str, key: str) -> Path:
    safe_key = key.replace("/", "_").replace(":", "_")
    return Path(DATA_RAW) / namespace / f"{safe_key}.json"


def _is_empty_payload(payload: Any) -> bool:
    """True for None/[]/{} and vendor-style ``{"results": []}`` payloads.

    We never want to cache "nothing happened" as if it were a confirmed
    empty result, since that could mask a transient vendor/network failure.
    """
    if payload is None:
        return True
    if isinstance(payload, (list, dict)) and len(payload) == 0:
        return True
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list) and len(results) == 0:
            return True
    return False


def cached_call(
    namespace: str,
    key: str,
    fetch_fn: Callable[[], Any],
    *,
    ttl: float | None = None,
) -> Any:
    """Return raw JSON for ``(namespace, key)``, hitting the network only on a miss.

    - On a cache hit (file exists and, if ``ttl`` is set, is fresh enough),
      the payload is re-read from disk and ``fetch_fn`` is never called.
    - On a miss, ``fetch_fn`` is called once. Empty/error payloads are never
      written to disk (see :func:`_is_empty_payload`) so a bad response
      doesn't permanently masquerade as "no data".
    """
    path = _cache_path(namespace, key)
    if path.exists():
        if ttl is None or (time.time() - path.stat().st_mtime) <= ttl:
            with path.open("r") as f:
                return json.load(f)

    payload = fetch_fn()
    if _is_empty_payload(payload):
        return payload

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f)
    return payload
