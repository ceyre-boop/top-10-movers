"""Live prediction, scoring, and monitoring -- plan §7, temporal
pre-commitment.

Three entry points, each meant to be invoked by a scheduled job:

- `predict_for_date` -- 09:25 ET, T2 decision time. Computes T2 features,
  predicts, and writes+commits `data/predictions/YYYY-MM-DD.json` via
  `top10.storage.append_only_write` (refuses to overwrite -- past-you
  binds present-you). The `git commit` IS the pre-commitment: it proves
  the prediction existed before the outcome was knowable, so it must be
  explicit and must fail loudly if it fails.
- `score_prior_day` -- 16:05 ET. Reads the day's prediction, the captured
  Robinhood top-movers list, and the proxy label; computes hits against
  both and writes a scoring record.
- `rolling_monitor` -- rolling precision@10 vs B4 vs the holdout
  expectation. Implements the §7 stop rule (drift > 1.5 hits below the
  holdout expectation for 20 consecutive days) and the §10 kill criterion
  (live < B4 for 30 consecutive days). Both verdicts are made loud and
  unambiguous.

`top10.data`, `top10.collect`, and `top10.features.t2` are imported
LAZILY inside functions -- they may be mid-flight in another module.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from top10.config import DATA_PREDICTIONS, PROJECT_ROOT, et_now
from top10.features.spec import T2_SPEC, FeatureSpec, validate_frame
from top10.leakage import assert_decision_time_safe
from top10.storage import append_only_write

logger = logging.getLogger(__name__)

SCORES_DIR = PROJECT_ROOT / "data" / "scores"

# §7 stop rule: drift strictly greater than this many hits, sustained for
# `STOP_WINDOW_DAYS` consecutive days.
STOP_DRIFT_HITS = 1.5
STOP_WINDOW_DAYS = 20

# §10 kill criterion: live below B4 for this many consecutive days.
KILL_WINDOW_DAYS = 30


class GitCommitError(RuntimeError):
    """Raised when the pre-commitment `git add` / `git commit` fails."""


# --- predict_for_date ------------------------------------------------------


def _run_git(args: list[str], *, repo_root: Path) -> None:
    result = subprocess.run(
        ["git", *args], cwd=str(repo_root), capture_output=True, text=True
    )
    if result.returncode != 0:
        raise GitCommitError(
            f"predict_for_date: `git {' '.join(args)}` failed (exit {result.returncode}) "
            f"in {repo_root}: {result.stderr.strip()}"
        )


def _git_commit_prediction(path: Path, *, repo_root: Path, date_str: str) -> None:
    """Explicit `git add` + `git commit` of the prediction file. The
    commit IS the pre-commitment -- fail loudly (raise) rather than
    swallow a failure, since a silently-uncommitted prediction has no
    temporal proof behind it at all."""
    rel_path = path.relative_to(repo_root) if path.is_absolute() else path
    _run_git(["add", str(rel_path)], repo_root=repo_root)
    _run_git(
        ["commit", "-m", f"predict: {date_str} T2 prediction (pre-commitment)"],
        repo_root=repo_root,
    )
    logger.info("predict_for_date: committed %s", rel_path)


def predict_for_date(
    date: dt.date,
    model_path: str | Path,
    *,
    features: pd.DataFrame | None = None,
    feature_spec: FeatureSpec = T2_SPEC,
    predictions_dir: Path = DATA_PREDICTIONS,
    repo_root: Path = PROJECT_ROOT,
    k: int = 10,
    git_commit: bool = True,
) -> Path:
    """Compute T2 features at 09:25 ET for `date`, predict with the model
    at `model_path`, and write+commit `data/predictions/<date>.json`.

    `features`, if given, is used as-is (validated against `feature_spec`)
    -- this is the injection point tests use to avoid touching the
    network/vendor data at all. When omitted, callers are expected to have
    already built the day's T2 features via `top10.pipeline` /
    `top10.features.t2` and pass them in; this function does not itself
    orchestrate a live feature build (that lives in the pipeline / a
    dedicated live-feature script), since doing so here would require
    importing the still-mid-flight vendor adapters at call time in a way
    that can't be reliably injected in tests.

    Raises `FileExistsError` (via `append_only_write`) if a prediction for
    `date` already exists -- past-you binds present-you.
    """
    if features is None:
        raise ValueError(
            "predict_for_date: `features` must be supplied (T2 features for `date`). "
            "Build them first via top10.pipeline.build_features_step('T2', ...) or "
            "top10.features.t2.build_t2_features."
        )

    date_ts = pd.Timestamp(date)
    validate_frame(features, feature_spec)

    # Defect 5: this is the one place a leak becomes a permanent,
    # signed (git-committed) artifact -- and it previously had no PIT
    # assertion at all, only a column-name/order check. Determine this
    # spec's own decision time for `date` and assert every feature row was
    # actually knowable at it before a prediction is ever computed from it.
    if feature_spec.task == "T1":
        from top10.features.t1 import decision_time_t1  # lazy: may be mid-flight

        decision_time = decision_time_t1(date_ts)
    else:
        from top10.features.t2 import decision_time_t2  # lazy: may be mid-flight

        decision_time = decision_time_t2(date_ts)
    assert_decision_time_safe(features, decision_time)

    from top10.model import Top10Ranker  # lazy: lightgbm-backed, avoid import cost/dep leakage

    model = Top10Ranker.load(model_path, feature_spec=feature_spec)
    predictions = model.rank_top_k(features, k=k, feature_spec=feature_spec)

    payload = {
        "trade_date": date_ts.date().isoformat(),
        "generated_at_et": et_now().isoformat(),
        "model_path": str(model_path),
        "feature_spec_hash": feature_spec.spec_hash,
        "k": k,
        "predictions": [
            {"ticker": row["ticker"], "score": float(row["score"])}
            for _, row in predictions.sort_values("score", ascending=False).iterrows()
        ],
    }

    out_path = Path(predictions_dir) / f"{date_ts.date().isoformat()}.json"
    append_only_write(payload, out_path)
    logger.info("predict_for_date: wrote %s", out_path)

    if git_commit:
        _git_commit_prediction(out_path, repo_root=repo_root, date_str=date_ts.date().isoformat())

    return out_path


# --- score_prior_day ------------------------------------------------------


def _read_prediction(date_ts: pd.Timestamp, predictions_dir: Path) -> dict:
    path = Path(predictions_dir) / f"{date_ts.date().isoformat()}.json"
    if not path.exists():
        raise FileNotFoundError(f"score_prior_day: no prediction file at {path}")
    return json.loads(path.read_text())


def _load_captured_rh(date_ts: pd.Timestamp) -> list[str]:
    """Lazily load the 16:05 ET captured Robinhood top-movers list for
    `date_ts` via `top10.collect` -- imported lazily since it may be
    mid-flight in another module.

    Defect 5: `load_captured_movers` returns an ENVELOPE (`dict | None`),
    not a bare list -- `list(envelope)` silently yielded the dict's KEYS
    instead of tickers, and `list(None)` raised a confusing `TypeError` on
    any uncaptured day. This reads the envelope properly: refuses (raises)
    when the day was never captured at all, and refuses when only the S&P
    500 context feed is available (`top_movers_available=False`) --
    scoring against the S&P 500 feed instead of the true top-movers list
    would silently corrupt the entire live forward test.
    """
    from top10.collect import load_captured_movers  # lazy

    envelope = load_captured_movers(date_ts.date())
    if envelope is None:
        raise FileNotFoundError(
            f"_load_captured_rh: no Robinhood capture found for {date_ts.date()}; "
            "cannot score against a missing capture."
        )
    if not envelope.get("top_movers_available"):
        raise ValueError(
            f"_load_captured_rh: {date_ts.date()} only captured the S&P 500 context "
            "feed (top_movers_available=False) -- refusing to score against it. The "
            "S&P 500 feed is NOT the docs/LABEL_SPEC.md proxy-validation source; "
            "scoring against it would silently corrupt the entire live forward test."
        )

    tickers = envelope.get("top_movers_tickers")
    if not tickers:
        raise ValueError(
            f"_load_captured_rh: {date_ts.date()} envelope claims "
            "top_movers_available=True but has no top_movers_tickers -- refusing to "
            "score against an empty/invalid capture."
        )
    return list(tickers)


def _load_proxy_labels(date_ts: pd.Timestamp) -> pd.DataFrame:
    """Lazily load the day's proxy labels from `data/labels/`."""
    from top10.config import DATA_LABELS, DOCS
    from top10.hashing import hash_file

    spec_hash = hash_file(DOCS / "LABEL_SPEC.md")
    path = Path(DATA_LABELS) / spec_hash / f"{date_ts.date().isoformat()}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["trade_date", "ticker", "rank", "return_t", "label", "label_spec_version", "as_of"])
    from top10.storage import read_parquet

    return read_parquet(path)


def score_prior_day(
    date: dt.date,
    *,
    predictions_dir: Path = DATA_PREDICTIONS,
    captured_rh: Sequence[str] | None = None,
    proxy_labels: pd.DataFrame | None = None,
    scores_dir: Path = SCORES_DIR,
) -> dict:
    """At 16:05 ET, read `date`'s prediction, the captured Robinhood list,
    and the proxy label; compute hits against both; write a scoring
    record to `data/scores/<date>.json` (append-only, same
    pre-commitment discipline as the prediction itself).

    `captured_rh` / `proxy_labels` are injectable for tests; when omitted
    they're loaded lazily from disk (`top10.collect` / `data/labels/`).
    """
    date_ts = pd.Timestamp(date)
    payload = _read_prediction(date_ts, Path(predictions_dir))
    predicted_tickers = [row["ticker"] for row in payload["predictions"]]

    if captured_rh is None:
        captured_rh = _load_captured_rh(date_ts)
    captured_rh = list(captured_rh)

    if proxy_labels is None:
        proxy_labels = _load_proxy_labels(date_ts)
    proxy_positive = set(
        proxy_labels[proxy_labels["label"] == 1]["ticker"]
    ) if not proxy_labels.empty else set()

    rh_hits = len(set(predicted_tickers) & set(captured_rh))
    proxy_hits = len(set(predicted_tickers) & proxy_positive)

    record = {
        "trade_date": date_ts.date().isoformat(),
        "scored_at_et": et_now().isoformat(),
        "predicted_tickers": predicted_tickers,
        "captured_rh": captured_rh,
        "rh_hits": rh_hits,
        "proxy_hits": proxy_hits,
        "n_predicted": len(predicted_tickers),
    }

    out_path = Path(scores_dir) / f"{date_ts.date().isoformat()}.json"
    append_only_write(record, out_path)
    logger.info("score_prior_day: wrote %s (rh_hits=%d, proxy_hits=%d)", out_path, rh_hits, proxy_hits)
    return record


# --- rolling_monitor ------------------------------------------------------


@dataclass
class MonitorVerdict:
    stop: bool
    kill: bool
    reasons: list[str] = field(default_factory=list)
    window_hits: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.stop and not self.kill:
            return f"[OK] rolling_monitor: no stop/kill condition triggered ({len(self.window_hits)} day window)."
        banner = []
        if self.stop:
            banner.append("*** STOP ***")
        if self.kill:
            banner.append("*** KILL ***")
        return " ".join(banner) + " " + " | ".join(self.reasons)


def _load_recent_scores(scores_dir: Path, n: int) -> list[dict]:
    scores_dir = Path(scores_dir)
    if not scores_dir.exists():
        return []
    files = sorted(scores_dir.glob("*.json"))
    records = [json.loads(p.read_text()) for p in files[-n:]]
    return records


def rolling_monitor(
    window: int = 20,
    *,
    live_hits: Sequence[float] | None = None,
    b4_hits: Sequence[float] | None = None,
    holdout_expectation: float | None = None,
    scores_dir: Path = SCORES_DIR,
    hits_field: str = "proxy_hits",
) -> MonitorVerdict:
    """Rolling precision@10 (as raw hit counts) monitor.

    - §7 stop rule: STOP if, for the most recent `STOP_WINDOW_DAYS` (20)
      consecutive days, every day's live hits are more than
      `STOP_DRIFT_HITS` (1.5) below `holdout_expectation`. Requires at
      least 20 days of history -- 19 days can never trigger STOP.
    - §10 kill criterion: KILL if, for the most recent `KILL_WINDOW_DAYS`
      (30) consecutive days, live hits are strictly below the
      corresponding B4 hits every single day.

    `live_hits` / `b4_hits` are injectable (aligned, same-length,
    chronologically ordered sequences) for tests; when `live_hits` is
    omitted, it's loaded from `data/scores/*.json` via `hits_field`.
    """
    if live_hits is None:
        records = _load_recent_scores(scores_dir, max(window, STOP_WINDOW_DAYS, KILL_WINDOW_DAYS))
        live_hits = [float(r.get(hits_field, 0)) for r in records]
    live_hits = list(live_hits)

    reasons: list[str] = []
    stop = False
    kill = False

    if holdout_expectation is not None and len(live_hits) >= STOP_WINDOW_DAYS:
        last_n = live_hits[-STOP_WINDOW_DAYS:]
        drift = [holdout_expectation - h for h in last_n]
        if all(d > STOP_DRIFT_HITS for d in drift):
            stop = True
            reasons.append(
                f"STOP (§7): live hits have drifted > {STOP_DRIFT_HITS} hits below the "
                f"holdout expectation ({holdout_expectation}) for {STOP_WINDOW_DAYS} "
                "consecutive days."
            )

    if b4_hits is not None:
        b4_hits = list(b4_hits)
        if len(live_hits) >= KILL_WINDOW_DAYS and len(b4_hits) >= KILL_WINDOW_DAYS:
            recent_live = live_hits[-KILL_WINDOW_DAYS:]
            recent_b4 = b4_hits[-KILL_WINDOW_DAYS:]
            if all(l < b for l, b in zip(recent_live, recent_b4)):
                kill = True
                reasons.append(
                    f"KILL (§10): live hits have been strictly below B4 for "
                    f"{KILL_WINDOW_DAYS} consecutive days."
                )

    verdict = MonitorVerdict(stop=stop, kill=kill, reasons=reasons, window_hits=live_hits[-window:])
    if stop or kill:
        logger.warning("rolling_monitor: %s", verdict)
    else:
        logger.info("rolling_monitor: %s", verdict)
    return verdict
