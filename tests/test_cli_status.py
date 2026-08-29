"""`status` is the one command a human runs to check pipeline state, so a
wrong answer here is worse than no answer. These pin the four ways it lied:
vendor ignored MARKET_DATA_VENDOR, RH captures were counted in the wrong
directory (always 0), .gitkeep counted as data, and the holdout-seal
boolean was inverted.
"""

from __future__ import annotations

import json

import pytest

from top10 import cli


def _run_status(capsys, monkeypatch, tmp_path, **env):
    for k in ("MARKET_DATA_VENDOR", "POLYGON_API_KEY", "DATABENTO_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for name in ("DATA_RAW", "DATA_PIT", "DATA_LABELS", "DATA_FEATURES", "DATA_PREDICTIONS", "EXPERIMENTS"):
        d = tmp_path / name.lower()
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(f"top10.cli.{name}", d)
    assert cli.main(["status"]) == 0
    return capsys.readouterr().out


def test_status_honours_market_data_vendor(capsys, monkeypatch, tmp_path):
    out = _run_status(capsys, monkeypatch, tmp_path,
                      MARKET_DATA_VENDOR="composite", DATABENTO_API_KEY="x")
    assert "vendor: composite" in out


def test_status_never_prints_a_key_value(capsys, monkeypatch, tmp_path):
    secret = "db-SUPERSECRETVALUE123"
    out = _run_status(capsys, monkeypatch, tmp_path, DATABENTO_API_KEY=secret)
    assert secret not in out
    assert "databento key present: True" in out


def test_status_reports_holdout_as_sealed(capsys, monkeypatch, tmp_path):
    # assert_holdout_sealed RAISING on a holdout date means the seal WORKS.
    out = _run_status(capsys, monkeypatch, tmp_path)
    assert "holdout sealed: True" in out


def test_status_counts_rh_captures_in_the_real_directory(capsys, monkeypatch, tmp_path):
    raw = tmp_path / "data_raw"
    (raw / "rh_movers").mkdir(parents=True)
    (raw / "rh_movers" / "2026-08-28.json").write_text(
        json.dumps({"top_movers_available": True, "top_movers_tickers": ["AAA"]})
    )
    monkeypatch.delenv("MARKET_DATA_VENDOR", raising=False)
    for name in ("DATA_PIT", "DATA_LABELS", "DATA_FEATURES", "DATA_PREDICTIONS", "EXPERIMENTS"):
        d = tmp_path / name.lower(); d.mkdir(exist_ok=True)
        monkeypatch.setattr(f"top10.cli.{name}", d)
    monkeypatch.setattr("top10.cli.DATA_RAW", raw)

    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "captured RH days: 1" in out
    assert "2026-08-28" in out


def test_status_warns_when_no_captures_exist(capsys, monkeypatch, tmp_path):
    out = _run_status(capsys, monkeypatch, tmp_path)
    assert "captured RH days: 0" in out
    assert "permanent" in out  # P1 warning must be present


def test_gitkeep_is_not_counted_as_data(tmp_path):
    (tmp_path / ".gitkeep").write_text("")
    assert cli._count_files(tmp_path) == 0
    (tmp_path / "real.parquet").write_text("x")
    assert cli._count_files(tmp_path) == 1
