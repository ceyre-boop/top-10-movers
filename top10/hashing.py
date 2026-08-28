"""Deterministic hashing helpers for files and spec dicts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_file(path: Path | str) -> str:
    """Return the sha256 hex digest of the raw bytes of `path`."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_spec(obj: Any) -> str:
    """Return the sha256 hex digest of `obj`'s canonical JSON form.

    Canonical == sorted keys, no whitespace — so equal dicts (regardless of
    key insertion order) always hash the same.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_spec_hash(path: Path | str, expected_hash_path: Path | str) -> bool:
    """Return True iff `path`'s sha256 matches the hash recorded at
    `expected_hash_path`.

    The hash file may contain either a bare hex digest, or the standard
    `sha256sum` format `<hex>  <filename>`.
    """
    raw = Path(expected_hash_path).read_text().strip()
    expected = raw.split()[0] if raw else ""
    return hash_file(path) == expected
