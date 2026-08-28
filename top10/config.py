"""Project configuration: paths, API keys, and timezone helpers.

Loads `.env` (if present) via python-dotenv and resolves all data paths
relative to this file's location — never hardcode an absolute path.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load .env from the project root if it exists. Never raise if it's absent.
load_dotenv(override=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PIT = PROJECT_ROOT / "data" / "pit"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"
DATA_FEATURES = PROJECT_ROOT / "data" / "features"
DATA_PREDICTIONS = PROJECT_ROOT / "data" / "predictions"
DOCS = PROJECT_ROOT / "docs"
EXPERIMENTS = PROJECT_ROOT / "experiments"

_VENDOR_ENV_VARS = {
    "polygon": "POLYGON_API_KEY",
    "databento": "DATABENTO_API_KEY",
}


def get_api_key(vendor: str) -> str | None:
    """Return the API key for `vendor` ("polygon" | "databento"), or None.

    Never raises — imports and construction must not require a key.
    """
    env_var = _VENDOR_ENV_VARS.get(vendor.lower())
    if env_var is None:
        return None
    return os.environ.get(env_var) or None


ET = ZoneInfo("America/New_York")


def et_now() -> dt.datetime:
    """Current time as a timezone-aware datetime in America/New_York."""
    return dt.datetime.now(tz=ET)


def to_et_naive(ts: dt.datetime) -> dt.datetime:
    """Convert `ts` to America/New_York and strip tzinfo.

    If `ts` is naive, it is assumed to already be ET-naive and is returned
    unchanged.
    """
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(ET).replace(tzinfo=None)
