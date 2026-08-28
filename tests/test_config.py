from __future__ import annotations

import datetime as dt

from top10 import config


def test_get_api_key_missing_returns_none(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    assert config.get_api_key("polygon") is None


def test_get_api_key_unknown_vendor_returns_none():
    assert config.get_api_key("nope") is None


def test_get_api_key_present(monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "secret")
    assert config.get_api_key("databento") == "secret"


def test_paths_relative_to_project_root():
    assert config.DATA_RAW == config.PROJECT_ROOT / "data" / "raw"
    assert config.DATA_PIT == config.PROJECT_ROOT / "data" / "pit"
    assert config.DATA_LABELS == config.PROJECT_ROOT / "data" / "labels"
    assert config.DATA_FEATURES == config.PROJECT_ROOT / "data" / "features"
    assert config.DATA_PREDICTIONS == config.PROJECT_ROOT / "data" / "predictions"
    assert config.DOCS == config.PROJECT_ROOT / "docs"
    assert config.EXPERIMENTS == config.PROJECT_ROOT / "experiments"


def test_to_et_naive_strips_tzinfo():
    aware = dt.datetime(2024, 1, 5, 16, 0, tzinfo=dt.timezone.utc)
    naive = config.to_et_naive(aware)
    assert naive.tzinfo is None


def test_to_et_naive_passthrough_for_naive_input():
    already_naive = dt.datetime(2024, 1, 5, 11, 0)
    assert config.to_et_naive(already_naive) == already_naive


def test_et_now_is_aware():
    now = config.et_now()
    assert now.tzinfo is not None
