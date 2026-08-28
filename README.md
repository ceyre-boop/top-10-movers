# TOP10 — Daily Top-Movers Predictor

A research vehicle that ranks the 10 tickers most likely to appear on the day's
top-gainers list, before the market opens. This is a pre-registered,
sealed-holdout research project, not a trading system: the point is to find
out honestly whether this is predictable at all, and to make it structurally
hard to fool ourselves if it isn't.

## The task: T1 / T2

Two decision-time variants, per `docs/LABEL_SPEC.md` / `docs/PREREG_TOP10.md`:

- **T1** — predict using only information available as of the **prior day's
  16:00 ET close**.
- **T2** — predict at **09:25 ET** on the trade date itself, using T1's
  information plus that morning's premarket data.

Both tasks share the same proxy label: rank the day's eligible universe by
unadjusted close-over-close return and label the top 10 `label=1`. See
`docs/LABEL_SPEC.md` for the full universe/label/corporate-action rules.

## Two constraints that shape everything here

- **P1 — no historical Robinhood top-movers archive.** Robinhood's actual
  "top movers" list isn't archived anywhere we can pull historically, so the
  label is a *proxy* (top-10 by return within a filtered universe), validated
  going forward by capturing the real list daily at 16:05 ET starting on live
  collection day and comparing overlap (`docs/LABEL_SPEC.md` "Proxy
  validation"). The proxy is not assumed correct — it's tuned against reality
  until median 30-day overlap is `>= 8/10`, then frozen.
- **P2 — survivorship.** The universe must include names that are later
  delisted; excluding them retroactively removes the specific stocks a
  top-movers predictor most needs to catch (huge, often terminal, one-day
  moves). `top10.labels.build_universe` enforces this explicitly.

## Quickstart

```bash
pip install -e '.[dev,collect]'          # add '.[model]' too if you're training
# or, for a specific vendor:
pip install -e '.[dev,collect,polygon]'  # or '.[dev,collect,databento]'

cp .env.example .env                     # then fill in your vendor key locally
python -m top10.cli status               # sanity-check config without touching data
```

## Phase → command map

| Phase | Command |
|---|---|
| Pull + persist point-in-time raw data | `python -m top10.cli ingest --vendor polygon --start 2015-01-01 --end 2022-12-31` |
| Build (resumable) labels | `python -m top10.cli labels --vendor polygon --start ... --end ...` |
| Build (resumable) T1/T2 features | `python -m top10.cli features --vendor polygon --task T1 --start ... --end ...` |
| Run sanity checks (aborts the pipeline on failure) | `python -m top10.cli sanity --labels data/labels/<hash>` |
| Compute B0–B4 baselines | `python -m top10.cli baselines` (invoke `top10.pipeline.run_baselines_step` from a script with loaded frames) |
| Walk-forward evaluation | `python -m top10.cli walkforward` (invoke `top10.pipeline.run_walkforward_step` from a script) |
| Live predict (09:25 ET, pre-commits via git) | `python -m top10.cli predict --date 2026-08-27 --model-path models/current` |
| Score prior day (16:05 ET) | `python -m top10.cli score --date 2026-08-27` |
| Rolling stop/kill monitor | `python -m top10.cli monitor` |
| Ceiling estimation helpers | `python -m top10.cli ceiling` (invoke `top10.ceiling.*` from a script with an injected classifier) |
| Config / row-count / holdout status | `python -m top10.cli status` |

All of the above also runs as `python -m top10.cli <command>` (the
`top10/cli.py` module is directly executable via `if __name__ == "__main__"`).

## Storage layout

```
data/raw/<namespace>/<key>.json        # untouched raw vendor payloads (§ caching)
data/pit/<vendor>/<dataset>/<range>.parquet   # point-in-time ingested tables
data/labels/<label-spec-hash>/<date>.parquet  # labels, keyed by frozen label-spec hash
data/features/<feature-spec-hash>/<date>.parquet  # T1/T2 features, keyed by feature-spec hash
data/predictions/<date>.json           # append-only, pre-committed live predictions
data/scores/<date>.json                # append-only daily scoring records
experiments/EXP-###.md                 # one file per logged experiment
docs/RESULT_TOP10.md                   # reserved for the single sealed holdout result
```

Every row-bearing table carries an `as_of` column and every read/write in
`top10.storage` enforces the point-in-time invariant: nothing whose `as_of`
is after the decision time is ever usable to make that decision
(`docs/LABEL_SPEC.md` "Point-in-time invariants").

## Research discipline — read before running anything against 2023+

This project's entire value depends on not fooling ourselves. The rules are
enforced in code, not just written down:

- **The holdout (`2023-01-01` through data end) is sealed until
  `docs/PREREG_TOP10.md` is frozen and committed.** Every code path that
  could touch it — walk-forward splits, tuning windows, manual scripts —
  routes through `top10.experiment.assert_holdout_sealed`, which refuses
  unless called with the literal unseal token `"PREREG_FROZEN"`.
- **One holdout run only.** Once unsealed, the holdout is evaluated exactly
  once. There is no "one more look."
- **Every run is logged to `experiments/` or it doesn't count.**
  `top10.experiment.log_experiment` writes `experiments/EXP-###.md`; only
  experiments logged there — and only the ones flagged
  "counts toward family-wise correction" — feed the Holm correction
  (`top10.metrics.family_wise_correction`) that the final claim must be
  corrected against. An unlogged run is not evidence.
- **Ceiling estimation never touches the holdout either** —
  `top10.ceiling.sample_positives` samples only from `trade_date < 2023-01-01`
  by default.

## Kill criteria

Two independent stop conditions, both implemented to fail loudly rather than
degrade silently:

1. **Walk-forward gate** (`top10.walkforward.beats_baseline_gate`) — if the
   model does not beat baseline B4 (premarket gap %) in at least 5 of 7
   evaluated years, the result does not support a primary success claim.
2. **Live monitor** (`top10.predict_live.rolling_monitor`) —
   - **STOP**: live precision drifts more than 1.5 hits/day below the
     holdout expectation for 20 consecutive days.
   - **KILL**: live hits fall strictly below baseline B4 for 30 consecutive
     days.

   The scheduled `.github/workflows/predict-live.yml` job runs `monitor`
   after every day's scoring and fails the CI run (non-zero exit) on either
   verdict, so a failed run *is* the alert.

## Secure configuration

Vendor credentials must be supplied through a secure environment variable or
repository secret — `POLYGON_API_KEY` for Polygon, `DATABENTO_API_KEY` for
Databento (see `top10/config.py`'s `_VENDOR_ENV_VARS`).

- **Local development**: create a local `.env` file (never committed —
  `.gitignore` already excludes `.env*` except `.env.example`).
- **GitHub Actions / CI**: add a repository or environment secret named
  `POLYGON_API_KEY` and/or `DATABENTO_API_KEY`; the workflow references them
  via `secrets.*` and never inlines a key.

The actual API key is never stored anywhere in this repository, and
`python -m top10.cli status` reports only whether a key is *present*, never
its value.

## Current repository contents

- `docs/LABEL_SPEC.md` — frozen proxy-label definition
- `docs/LABEL_SPEC.sha256` — hash for the committed label spec
- `docs/PREREG_TOP10.md` — pre-registration draft for the sealed holdout
- `docs/CEILING.md` — ceiling-estimation protocol
- `docs/RESULT_TOP10.md` — reserved for the one-time sealed holdout result
- `top10/` — pipeline, model, features, labels, sanity, leakage, and live
  prediction/monitoring code
- `data/` — raw, point-in-time, labels, features, predictions, and scores
- `experiments/` — one file per logged experiment
- `.github/workflows/predict-live.yml` — scheduled predict (09:25 ET) and
  score+monitor (16:10 ET) jobs
