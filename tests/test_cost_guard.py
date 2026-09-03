"""Offline tests for top10.data.cost_guard. No network access, no API keys.

All Databento SDK calls are mocked -- no live paid calls are ever made.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from top10.data.cost_guard import (
    BudgetExceeded,
    ConfirmationRequired,
    CostGuard,
    estimate_cost,
)
from top10.data.databento import DatabentoSource, FIRST_AVAILABLE_DATE


# --- estimate_cost ------------------------------------------------------------


def test_estimate_cost_calls_metadata_get_cost_not_arithmetic():
    calls = []

    class _FakeMetadata:
        def get_cost(self, **kwargs):
            calls.append(kwargs)
            return 1.2345

    class _FakeClient:
        metadata = _FakeMetadata()

    result = estimate_cost(_FakeClient(), dataset="XNAS.ITCH", schema="ohlcv-1d")

    assert result == 1.2345
    assert calls == [{"dataset": "XNAS.ITCH", "schema": "ohlcv-1d"}]


# --- CostGuard.guarded_request ------------------------------------------------


def test_guarded_request_refuses_when_estimate_would_breach_ceiling(tmp_path):
    guard = CostGuard(
        ceiling_usd=10.0, ledger_path=tmp_path / "ledger.json", confirm_threshold_usd=100.0
    )

    def _fetch():
        raise AssertionError("fetch_fn must never run when the budget is refused")

    with pytest.raises(BudgetExceeded, match=r"\$15\.00"):
        guard.guarded_request(_fetch, cost_estimate=15.0, description="too big")

    assert guard.spent == 0.0
    assert not (tmp_path / "ledger.json").exists()


def test_guarded_request_allows_and_records_spend_within_budget(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    guard = CostGuard(
        ceiling_usd=10.0, ledger_path=ledger_path, confirm_threshold_usd=100.0
    )

    fetch_calls = {"n": 0}

    def _fetch():
        fetch_calls["n"] += 1
        return {"ok": True}

    result = guard.guarded_request(_fetch, cost_estimate=3.0, description="small pull")

    assert result == {"ok": True}
    assert fetch_calls["n"] == 1
    assert guard.spent == 3.0
    assert ledger_path.exists()


def test_guarded_request_refuses_second_request_that_would_breach_cumulative_ceiling(
    tmp_path,
):
    guard = CostGuard(
        ceiling_usd=10.0, ledger_path=tmp_path / "ledger.json", confirm_threshold_usd=100.0
    )
    guard.guarded_request(lambda: None, cost_estimate=8.0, description="first")
    assert guard.spent == 8.0

    with pytest.raises(BudgetExceeded):
        guard.guarded_request(lambda: None, cost_estimate=5.0, description="second")

    # Refused request must not have been recorded.
    assert guard.spent == 8.0


# --- ledger persistence across process restarts -------------------------------


def test_ledger_persists_and_reloads_across_instances(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    guard1 = CostGuard(
        ceiling_usd=100.0, ledger_path=ledger_path, confirm_threshold_usd=100.0
    )
    guard1.guarded_request(lambda: "a", cost_estimate=4.0, description="req-1")
    guard1.guarded_request(lambda: "b", cost_estimate=6.0, description="req-2")
    assert guard1.spent == 10.0

    # Simulate a fresh process: new CostGuard instance, same ledger path.
    guard2 = CostGuard(
        ceiling_usd=100.0, ledger_path=ledger_path, confirm_threshold_usd=100.0
    )
    assert guard2.spent == 10.0

    # And the running total is still enforced across the "restart".
    with pytest.raises(BudgetExceeded):
        guard2.guarded_request(lambda: None, cost_estimate=95.0, description="req-3")

    payload = json.loads(ledger_path.read_text())
    assert len(payload["entries"]) == 2
    descriptions = {e["description"] for e in payload["entries"]}
    assert descriptions == {"req-1", "req-2"}


def test_cost_guard_default_ceiling_is_below_the_promotional_credit(monkeypatch):
    monkeypatch.delenv("DATABENTO_BUDGET_USD", raising=False)
    guard = CostGuard()
    assert guard.ceiling_usd < 125.0


def test_cost_guard_reads_ceiling_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "42.5")
    guard = CostGuard(ledger_path=tmp_path / "ledger.json")
    assert guard.ceiling_usd == 42.5


# --- require_confirmation_above -----------------------------------------------


def test_request_above_confirmation_threshold_raises_without_confirm(tmp_path):
    guard = CostGuard(ceiling_usd=100.0, ledger_path=tmp_path / "ledger.json")

    with pytest.raises(ConfirmationRequired):
        guard.guarded_request(lambda: None, cost_estimate=6.0, description="pricey")

    assert guard.spent == 0.0


def test_request_above_confirmation_threshold_succeeds_with_confirm_true(tmp_path):
    guard = CostGuard(ceiling_usd=100.0, ledger_path=tmp_path / "ledger.json")

    result = guard.guarded_request(
        lambda: "done", cost_estimate=6.0, description="pricey", confirm=True
    )

    assert result == "done"
    assert guard.spent == 6.0


# --- DatabentoSource wiring: no unguarded route to a paid request -----------


class _FakeMetadata:
    def __init__(self, cost=0.5, record_count=100):
        self.cost = cost
        self.record_count = record_count
        self.get_cost_calls: list[dict] = []
        self.get_record_count_calls: list[dict] = []

    def get_cost(self, **kwargs):
        self.get_cost_calls.append(kwargs)
        return self.cost

    def get_record_count(self, **kwargs):
        self.get_record_count_calls.append(kwargs)
        return self.record_count


class _FakeStore:
    def __init__(self, df):
        self._df = df

    def to_df(self):
        return self._df


class _FakeTimeseries:
    def __init__(self, df):
        self._df = df
        self.get_range_calls: list[dict] = []

    def get_range(self, **kwargs):
        self.get_range_calls.append(kwargs)
        return _FakeStore(self._df)


class _FakeClient:
    def __init__(self, df=None, cost=0.5, record_count=1):
        self.metadata = _FakeMetadata(cost=cost, record_count=record_count)
        self.timeseries = _FakeTimeseries(df if df is not None else pd.DataFrame())


def _source_with_fake_client(monkeypatch, tmp_path, client):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()
    monkeypatch.setattr(source, "_get_client", lambda: client)
    guard = CostGuard(
        ceiling_usd=100.0, ledger_path=tmp_path / "ledger.json", confirm_threshold_usd=100000.0
    )
    monkeypatch.setattr(source, "_cost_guard", lambda: guard)
    return source, guard


def test_fetch_bars_estimates_cost_before_requesting_data(monkeypatch, tmp_path):
    client = _FakeClient(cost=0.1)
    source, guard = _source_with_fake_client(monkeypatch, tmp_path, client)

    source._fetch_bars("XNAS.ITCH", "ohlcv-1d", "2024-01-05", "2024-01-05")

    assert len(client.metadata.get_cost_calls) == 1
    assert len(client.timeseries.get_range_calls) == 1
    assert guard.spent == 0.1


def test_fetch_bars_refuses_when_estimate_breaches_ceiling(monkeypatch, tmp_path):
    client = _FakeClient(cost=1000.0)
    source, guard = _source_with_fake_client(monkeypatch, tmp_path, client)

    with pytest.raises(BudgetExceeded):
        source._fetch_bars("XNAS.ITCH", "ohlcv-1d", "2024-01-05", "2024-01-05")

    # The estimate was made, but the paid `get_range` call must never fire.
    assert len(client.metadata.get_cost_calls) == 1
    assert len(client.timeseries.get_range_calls) == 0


# --- May-2018 history-start constraint -----------------------------------------


def test_daily_bars_rejects_start_before_may_2018(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    with pytest.raises(ValueError, match="2018-05-01"):
        source.daily_bars(dt.date(2018, 1, 1), dt.date(2018, 6, 1))


def test_premarket_bars_rejects_trade_date_before_may_2018(monkeypatch, tmp_path):
    monkeypatch.setattr("top10.data.cache.DATA_RAW", tmp_path)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    with pytest.raises(ValueError, match="2018-05-01"):
        source.premarket_bars(dt.date(2017, 12, 1), ["AAPL"])


def test_first_available_date_constant_is_may_2018():
    assert FIRST_AVAILABLE_DATE == dt.date(2018, 5, 1)


# --- dry_run performs no data download -----------------------------------------


def test_daily_bars_dry_run_returns_estimate_without_downloading(monkeypatch, tmp_path):
    client = _FakeClient(cost=7.5, record_count=12345)
    source, guard = _source_with_fake_client(monkeypatch, tmp_path, client)

    result = source.daily_bars(dt.date(2024, 1, 5), dt.date(2024, 1, 5), dry_run=True)

    # daily_bars unions the three LISTING venues (XNAS/XNYS/XASE), because
    # no consolidated US equities dataset reaches back before 2023. So a
    # dry run reports the SUM across venues, not one dataset's figure.
    assert result["dry_run"] is True
    assert result["start"] == "2024-01-05"
    assert result["end"] == "2024-01-05"
    assert result["cost_usd"] == pytest.approx(7.5 * 3)
    assert result["record_count"] == 12345 * 3
    assert len(result["venues"]) == 3
    assert {v["dataset"] for v in result["venues"]} == {
        "XNAS.ITCH", "XNYS.PILLAR", "XASE.PILLAR"
    }
    # schema/symbols are per-venue on the aggregate result, not top-level.
    assert all(v["schema"] == "ohlcv-1d" for v in result["venues"])
    assert all(v["symbols"] == "ALL_SYMBOLS" for v in result["venues"])
    _unused = {
    }
    # No `timeseries.get_range` call -- nothing was downloaded.
    assert len(client.timeseries.get_range_calls) == 0
    # No spend recorded for a dry run.
    assert guard.spent == 0.0


def test_premarket_bars_dry_run_returns_estimate_without_downloading(monkeypatch, tmp_path):
    client = _FakeClient(cost=2.0, record_count=42)
    source, guard = _source_with_fake_client(monkeypatch, tmp_path, client)

    result = source.premarket_bars(dt.date(2024, 1, 5), ["AAPL", "MSFT"], dry_run=True)

    assert result["dry_run"] is True
    assert result["cost_usd"] == 2.0
    assert result["record_count"] == 42
    assert len(client.timeseries.get_range_calls) == 0
    assert guard.spent == 0.0


# --- symbol resolution: clear NotImplementedError, not silent survivors ------


def test_resolve_symbols_raises_not_implemented_naming_the_gap(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    source = DatabentoSource()

    with pytest.raises(NotImplementedError, match="symbol resolution"):
        source.resolve_symbols(["AAPL"], dt.date(2024, 1, 5))


# --- Defect 2: the cost guard was not tracking real spend --------------------
#
# The $35.47 actually spent went to an ad-hoc
# data/raw/databento/preholdout/_spend.json (keyed venue/start/end/cost/rows,
# no `actual_usd`) written by a script that bypassed CostGuard entirely. The
# real, reconciled ledger lives at data/raw/databento/_spend_ledger.json.


def test_cost_guard_reads_reconciled_spend_from_the_real_ledger():
    from top10.config import PROJECT_ROOT

    guard = CostGuard(ledger_path=PROJECT_ROOT / "data" / "raw" / "databento" / "_spend_ledger.json")
    assert guard.spent == pytest.approx(35.468129664658)


def test_cost_guard_real_ledger_has_exactly_one_reconciliation_entry():
    from top10.config import PROJECT_ROOT

    guard = CostGuard(ledger_path=PROJECT_ROOT / "data" / "raw" / "databento" / "_spend_ledger.json")
    assert len(guard._entries) == 1


def test_load_raises_clear_error_on_entry_missing_actual_usd(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "entries": [
                    {"venue": "XNAS.ITCH", "start": "2018-05-01", "end": "2019-01-01", "cost": 1.92, "rows": 123}
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="actual_usd"):
        CostGuard(ledger_path=ledger_path)


def test_default_ceiling_leaves_credit_aware_headroom(monkeypatch):
    monkeypatch.delenv("DATABENTO_BUDGET_USD", raising=False)
    guard = CostGuard(ledger_path=None)
    assert guard.ceiling_usd == pytest.approx(110.0)


def test_default_ceiling_still_overridden_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "42.5")
    guard = CostGuard(ledger_path=tmp_path / "ledger.json")
    assert guard.ceiling_usd == 42.5


def test_request_refused_when_it_would_exceed_the_credit_not_just_a_looser_ceiling(
    tmp_path,
):
    # A ceiling deliberately set ABOVE the $125 credit (simulating a
    # misconfigured override) must still be refused once the request would
    # push cumulative spend past the credit itself -- the guard's job is to
    # keep real billing from ever starting, not merely to respect whatever
    # ceiling was passed in.
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps({"entries": [{"timestamp": "t", "description": "prior spend", "actual_usd": 120.0}]})
    )
    guard = CostGuard(ceiling_usd=200.0, ledger_path=ledger_path, confirm_threshold_usd=100000.0)

    assert guard.spent == 120.0
    with pytest.raises(BudgetExceeded):
        guard.guarded_request(lambda: None, cost_estimate=10.0, description="would cross the $125 credit")
