"""TOP10 command-line entry point.

Usage: `python -m top10.cli <command> [args...]`

Subcommands: ingest, labels, features, sanity, baselines, walkforward,
predict, score, monitor, ceiling, status.

Every command exits non-zero on failure (either an explicit `return 1` on
a handled failure, or an uncaught exception propagating out of `main`,
which is the correct non-zero-exit behavior for a CLI).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import logging
import sys
from pathlib import Path

from top10.config import (
    DATA_FEATURES,
    DATA_LABELS,
    DATA_PIT,
    DATA_PREDICTIONS,
    DATA_RAW,
    EXPERIMENTS,
    get_api_key,
)
from top10.experiment import assert_holdout_sealed

logger = logging.getLogger(__name__)

_VENDORS = ("polygon", "databento")


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from top10.pipeline import ingest

    # Route through the same guarded chokepoint every other holdout-reaching
    # entry point uses (Plan §6 / P12) -- a bare `--start/--end` was
    # previously enough to build holdout-dated raw data with no token.
    assert_holdout_sealed([args.start, args.end], unseal_token=args.unseal_token)
    ingest(args.vendor, args.start, args.end)
    print(f"ingest: {args.vendor} {args.start}..{args.end} done.")
    return 0


def _cmd_labels(args: argparse.Namespace) -> int:
    from top10.pipeline import build_labels_step, ingest

    assert_holdout_sealed([args.start, args.end], unseal_token=args.unseal_token)
    frames = ingest(args.vendor, args.start, args.end)
    labels = build_labels_step(frames, args.start, args.end)
    print(f"labels: {len(labels)} row(s) built for {args.start}..{args.end}.")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    from top10.pipeline import build_features_step, ingest

    assert_holdout_sealed([args.start, args.end], unseal_token=args.unseal_token)
    frames = ingest(args.vendor, args.start, args.end)
    features = build_features_step(
        args.task,
        args.start,
        args.end,
        daily_bars=frames.get("daily_bars"),
        ticker_meta=frames.get("ticker_meta"),
        earnings=frames.get("earnings"),
    )
    print(f"features[{args.task}]: {len(features)} row(s) built for {args.start}..{args.end}.")
    return 0


def _cmd_sanity(args: argparse.Namespace) -> int:
    from top10.sanity import _main as sanity_main

    argv = ["--labels", args.labels]
    if args.corporate_actions:
        argv += ["--corporate-actions", args.corporate_actions]
    return sanity_main(argv)


def _cmd_baselines(args: argparse.Namespace) -> int:
    print(
        "baselines: this command requires pre-built universe/labels/bars/earnings/"
        "premarket frames; invoke top10.pipeline.run_baselines_step directly from a "
        "script with those frames loaded. No default on-disk wiring exists yet."
    )
    return 0


def _cmd_walkforward(args: argparse.Namespace) -> int:
    print(
        "walkforward: this command requires a model_factory and pre-built features/"
        "labels; invoke top10.pipeline.run_walkforward_step directly from a script. "
        "No default on-disk wiring exists yet."
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from top10.predict_live import predict_for_date

    predict_for_date(args.date, args.model_path, git_commit=not args.no_commit)
    print(f"predict: wrote+committed prediction for {args.date}.")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from top10.predict_live import score_prior_day

    record = score_prior_day(args.date)
    print(f"score: {record}")
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    from top10.predict_live import rolling_monitor

    verdict = rolling_monitor(
        window=args.window,
        holdout_expectation=args.holdout_expectation,
    )
    print(verdict)
    return 1 if (verdict.stop or verdict.kill) else 0


def _cmd_ceiling(args: argparse.Namespace) -> int:
    print(
        "ceiling: this command requires a labels frame and an injected classifier_fn "
        "(human or LLM); invoke top10.ceiling.sample_positives / classify / "
        "estimate_ceiling / render_ceiling_md directly from a script."
    )
    return 0


def _count_files(directory: Path, pattern: str = "**/*") -> int:
    directory = Path(directory)
    if not directory.exists():
        return 0
    return sum(1 for p in directory.glob(pattern) if p.is_file())


def _cmd_status(args: argparse.Namespace) -> int:
    from top10.experiment import HOLDOUT_START, UNSEAL_TOKEN, assert_holdout_sealed
    from top10.storage import LeakageError

    lines: list[str] = []

    vendor = args.vendor or next((v for v in _VENDORS if get_api_key(v)), None) or "(none configured)"
    lines.append(f"vendor: {vendor}")

    for v in _VENDORS:
        key_present = get_api_key(v) is not None
        lines.append(f"  {v} key present: {key_present}")

    lines.append(f"data/raw rows (files): {_count_files(DATA_RAW)}")
    lines.append(f"data/pit rows (files): {_count_files(DATA_PIT)}")
    lines.append(f"data/labels rows (files): {_count_files(DATA_LABELS)}")
    lines.append(f"data/features rows (files): {_count_files(DATA_FEATURES)}")
    lines.append(f"data/predictions rows (files): {_count_files(DATA_PREDICTIONS)}")

    label_files = sorted(glob.glob(str(Path(DATA_LABELS) / "**" / "*.parquet"), recursive=True))
    latest_label_date = Path(label_files[-1]).stem if label_files else "(none)"
    lines.append(f"latest label date: {latest_label_date}")

    captured_rh_days = _count_files(Path(DATA_RAW) / "robinhood") if (Path(DATA_RAW) / "robinhood").exists() else 0
    lines.append(f"captured RH days: {captured_rh_days}")

    experiments_dir = Path(EXPERIMENTS)
    logged_experiment_count = len(list(experiments_dir.glob("EXP-*.md"))) if experiments_dir.exists() else 0
    lines.append(f"logged experiment count: {logged_experiment_count}")

    try:
        assert_holdout_sealed([HOLDOUT_START])
        sealed = True
    except LeakageError:
        sealed = False
    lines.append(f"holdout sealed: {sealed} (unseal requires token={UNSEAL_TOKEN!r})")

    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m top10.cli",
        description="TOP10 daily top-movers predictor -- pipeline, live prediction, and ceiling CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Pull and persist point-in-time raw data.")
    p_ingest.add_argument("--vendor", required=True, choices=_VENDORS)
    p_ingest.add_argument("--start", required=True, type=_parse_date)
    p_ingest.add_argument("--end", required=True, type=_parse_date)
    p_ingest.add_argument("--unseal-token", required=False, default=None, dest="unseal_token")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_labels = sub.add_parser("labels", help="Build (or resume) labels.")
    p_labels.add_argument("--vendor", required=True, choices=_VENDORS)
    p_labels.add_argument("--start", required=True, type=_parse_date)
    p_labels.add_argument("--end", required=True, type=_parse_date)
    p_labels.add_argument("--unseal-token", required=False, default=None, dest="unseal_token")
    p_labels.set_defaults(func=_cmd_labels)

    p_features = sub.add_parser("features", help="Build (or resume) T1/T2 features.")
    p_features.add_argument("--vendor", required=True, choices=_VENDORS)
    p_features.add_argument("--task", required=True, choices=("T1", "T2"))
    p_features.add_argument("--start", required=True, type=_parse_date)
    p_features.add_argument("--end", required=True, type=_parse_date)
    p_features.add_argument("--unseal-token", required=False, default=None, dest="unseal_token")
    p_features.set_defaults(func=_cmd_features)

    p_sanity = sub.add_parser("sanity", help="Run sanity checks against built labels.")
    p_sanity.add_argument("--labels", required=True)
    p_sanity.add_argument("--corporate-actions", required=False, default=None)
    p_sanity.set_defaults(func=_cmd_sanity)

    p_baselines = sub.add_parser("baselines", help="Compute B0-B4 baselines.")
    p_baselines.set_defaults(func=_cmd_baselines)

    p_walkforward = sub.add_parser("walkforward", help="Run walk-forward evaluation.")
    p_walkforward.set_defaults(func=_cmd_walkforward)

    p_predict = sub.add_parser("predict", help="Predict and pre-commit for a trade date (09:25 ET).")
    p_predict.add_argument("--date", required=True, type=_parse_date)
    p_predict.add_argument("--model-path", required=True, dest="model_path")
    p_predict.add_argument("--no-commit", action="store_true")
    p_predict.set_defaults(func=_cmd_predict)

    p_score = sub.add_parser("score", help="Score the prior day's prediction (16:05 ET).")
    p_score.add_argument("--date", required=True, type=_parse_date)
    p_score.set_defaults(func=_cmd_score)

    p_monitor = sub.add_parser("monitor", help="Run the rolling stop/kill monitor.")
    p_monitor.add_argument("--window", type=int, default=20)
    p_monitor.add_argument("--holdout-expectation", type=float, default=None, dest="holdout_expectation")
    p_monitor.set_defaults(func=_cmd_monitor)

    p_ceiling = sub.add_parser("ceiling", help="Ceiling estimation helpers.")
    p_ceiling.set_defaults(func=_cmd_ceiling)

    p_status = sub.add_parser("status", help="Print pipeline/config status.")
    p_status.add_argument("--vendor", required=False, default=None, choices=_VENDORS)
    p_status.set_defaults(func=_cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary: report and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
