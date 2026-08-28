"""Structural spend guard for Databento paid requests.

The user has a $125 promotional Databento credit that **auto-bills real
money once exhausted**, and Databento does not publish per-GB rates for
equities OHLCV -- the only reliable way to know what a request costs is to
ask Databento itself, before the request is made. This module exists so
that overspend is structurally impossible rather than merely discouraged:

- :func:`estimate_cost` always calls Databento's own
  ``metadata.get_cost()`` -- never record-size arithmetic, which is a
  guess and can guess wrong in the expensive direction.
- :class:`CostGuard` holds a hard ceiling (default $100, deliberately
  below the $125 credit so there is headroom before real billing starts)
  and a running spend ledger persisted to disk so the credit's state
  survives process restarts, same as the credit itself does.
- :meth:`CostGuard.guarded_request` is the ONLY sanctioned way to combine
  "check the estimate against the ledger" and "actually make the paid
  call" -- there must be no code path in this project that makes a paid
  Databento request without going through it.
- :meth:`CostGuard.require_confirmation_above` makes any single expensive
  request a human decision (``confirm=True``), never a default.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

from top10.config import DATA_RAW

# Deliberately below the $125 promotional credit: the whole point of a
# ceiling is to leave headroom before the credit runs out and real money
# starts billing, not to spend right up to the edge of it.
_DEFAULT_BUDGET_USD = 100.0

# Above this per-request estimate, spending requires an explicit human
# `confirm=True` -- see `require_confirmation_above`.
_DEFAULT_CONFIRM_THRESHOLD_USD = 5.00

T = TypeVar("T")


class BudgetExceeded(RuntimeError):
    """Raised when a request's estimated cost would push cumulative spend
    past the configured ceiling. The request is never made."""


class ConfirmationRequired(RuntimeError):
    """Raised when a single request's estimated cost exceeds the
    confirmation threshold and the caller did not pass ``confirm=True``.
    This is meant to be a human decision, never a silent default."""


def estimate_cost(client: Any, **request_params: Any) -> float:
    """Return the USD cost Databento reports for a request, BEFORE it runs.

    Always defers to Databento's own ``client.metadata.get_cost(...)`` --
    never estimated from record-count/size arithmetic. ``request_params``
    are passed straight through (``dataset``, ``schema``, ``symbols``,
    ``stype_in``, ``start``, ``end``, ...) and must match what will
    actually be requested via ``client.timeseries.get_range(...)``, or the
    estimate is meaningless.
    """
    return float(client.metadata.get_cost(**request_params))


def _default_ledger_path() -> Path:
    return Path(DATA_RAW) / "databento" / "_spend_ledger.json"


class CostGuard:
    """Enforces a hard USD ceiling on Databento spend across restarts.

    The ledger is read on construction and rewritten after every
    successful guarded request, so a new process (or a crashed-and-
    restarted one) picks up exactly where the last one left off -- the
    promotional credit does not reset just because the process did.
    """

    def __init__(
        self,
        ceiling_usd: float | None = None,
        ledger_path: Path | None = None,
        confirm_threshold_usd: float = _DEFAULT_CONFIRM_THRESHOLD_USD,
    ) -> None:
        if ceiling_usd is None:
            ceiling_usd = float(
                os.environ.get("DATABENTO_BUDGET_USD", _DEFAULT_BUDGET_USD)
            )
        self.ceiling_usd = ceiling_usd
        self.confirm_threshold_usd = confirm_threshold_usd
        self.ledger_path = Path(ledger_path) if ledger_path else _default_ledger_path()
        self._entries: list[dict[str, Any]] = self._load()

    # -- ledger persistence ---------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        if self.ledger_path.exists():
            with self.ledger_path.open("r") as f:
                data = json.load(f)
            return list(data.get("entries", []))
        return []

    def _save(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("w") as f:
            json.dump({"entries": self._entries}, f, indent=2, default=str)

    @property
    def spent(self) -> float:
        """Cumulative USD spent so far, per the persisted ledger."""
        return sum(float(e["actual_usd"]) for e in self._entries)

    @property
    def remaining(self) -> float:
        return self.ceiling_usd - self.spent

    # -- guard ------------------------------------------------------------------

    def require_confirmation_above(self, usd: float, *, confirm: bool = False) -> None:
        """Raise :class:`ConfirmationRequired` for an expensive request
        unless the caller explicitly passed ``confirm=True``."""
        if usd > self.confirm_threshold_usd and not confirm:
            raise ConfirmationRequired(
                f"request estimated at ${usd:.2f}, above the "
                f"${self.confirm_threshold_usd:.2f} confirmation threshold -- "
                "pass confirm=True to proceed. This must be a human decision, "
                "not a default."
            )

    def guarded_request(
        self,
        fetch_fn: Callable[[], T],
        cost_estimate: float,
        description: str,
        *,
        confirm: bool = False,
    ) -> T:
        """The only sanctioned way to make a paid Databento request.

        Refuses (raising :class:`BudgetExceeded`) when
        ``spent + cost_estimate > ceiling_usd``, and refuses (raising
        :class:`ConfirmationRequired`) when ``cost_estimate`` exceeds the
        confirmation threshold and ``confirm`` was not passed. Only after
        both checks pass does ``fetch_fn`` actually run; on success, the
        estimate is appended to the persisted ledger as actual spend.
        """
        self.require_confirmation_above(cost_estimate, confirm=confirm)

        projected = self.spent + cost_estimate
        if projected > self.ceiling_usd:
            raise BudgetExceeded(
                f"refusing request {description!r}: estimated ${cost_estimate:.2f} "
                f"would bring cumulative spend to ${projected:.2f}, over the "
                f"${self.ceiling_usd:.2f} ceiling (already spent ${self.spent:.2f} "
                f"per {self.ledger_path})."
            )

        result = fetch_fn()
        self._record(description, cost_estimate)
        return result

    def _record(self, description: str, estimate_usd: float) -> None:
        # Databento's `metadata.get_cost()` is the deterministic price for a
        # fully-specified request (same dataset/schema/symbols/date-range
        # that is then requested), so the estimate IS what gets billed --
        # there is no separate post-hoc "actual" figure the SDK returns
        # from `get_range()` itself. We ledger the estimate as actual spend
        # and document that here so nobody mistakes this for a reconciled
        # invoice line.
        entry = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "description": description,
            "estimate_usd": estimate_usd,
            "actual_usd": estimate_usd,
        }
        self._entries.append(entry)
        self._save()
