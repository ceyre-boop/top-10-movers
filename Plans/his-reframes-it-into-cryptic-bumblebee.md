# EXP-005-TFM — a fair, cheap test of TimesFM on the top-10 task

## Context

The ask: use Google's TimesFM to predict the daily top-10 % gainers.

TimesFM is a genuinely strong forecaster. But there is already decisive
local evidence about it on this exact class of data, and it should frame
everything below.

**`/Users/taboost/quant/data/research/hyp120/VERDICT.md`** — sealed and
adjudicated 2026-09-16, TimesFM 3.0 MLX, 50,182 daily windows over 19
series, 2016–2026:

- **Direction: FAIL.** Sign hit rate 51.45% (equity drift). Per-series
  Spearman(forecast, realized) = **−0.0088, CI [−0.0156, −0.0026]** —
  negative, CI excludes zero.
- **Magnitude: FAIL.** Quantile-derived sigma ties EWMA(0.94) exactly
  (0.6808 vs 0.6807) and over-forecasts vol by ~18%.

Combined with EXP-004, where `xs_vol_rank` carried **7× the gain of the
next feature**, the picture is: TimesFM's one usable output is a
volatility estimator that ties a two-line EWMA, offered to a model
already dominated by a volatility ranker.

**Honest prior: ~10–15%** that this produces a per-year-consistent
improvement over EXP-003.

It is still worth running, for three reasons that do not depend on it
working: it closes the "a foundation model will find it" question with
local evidence; it produces a reusable forecast cache; and it forces
payment of a reproducibility debt that currently makes EXP-003 and
EXP-004 uncitable.

### The one hypothesis with a real chance

Ranking by point forecast is the *worst* use — HYP-120 shows its rank
correlation with outcomes is negative, and existing momentum features
already own that channel. `q90 − q10` is not new either; it is EWMA vol
with a 1.3 GB wrapper.

The only quantity TimesFM emits that the incumbent cannot reconstruct is
**conditional asymmetry** — whether the right tail is fatter than the
left, given recent history. Nothing in the 15 T1 features measures
predictive skew. That is the hypothesis. Everything else is a control.

## Prerequisite: the reproducibility debt

EXP-003 and EXP-004 ran from scratch scripts no longer on disk.
`xs_vol_rank`, `xs_ret_rank`, `top90`, `logpx`, `dist52`, `vol20`,
`vol5`, `relvol_prev` appear nowhere in the repo except EXP-004's prose.
`validate_frame` / `write_features` were bypassed, `data/features/` is
empty, `FEATURE_SPEC_VERSION` is still `"1"`.

**Both logged variants rest on definitions that no longer exist**, so the
TimesFM delta has nothing honest to measure against. This is step one,
not a side quest.

## Decisions taken

**Checkpoint: TimesFM 2.5, Apache-2.0.** 3.0 weights are
`timesfm-non-commercial-license-v1.0` — non-commercial, non-production.
Any path to live trading needs 2.5. Consequence: 2.5 runs through
`src/timesfm` (torch backend, 200M params), not the MLX 3.0 fast path,
so expect materially lower throughput than 3.0's 666 series/s. Budget
wall-clock accordingly; compute is still free and local.

**Target: log returns, never raw close.** Two reasons. It removes the
only plausible memorization key (see probes), and it stops split-day
price jumps entering the context as fake ±50% returns.

**Context: 256 sessions, min 128, `padding_mode="none"`.** History starts
2018-05-22, so a 512 context would destroy 2019 and 2020 as evaluation
years. Pre-declare 256 and never tune it — each context length tried is
a variant.

## Contamination probes

TimesFM's pretraining corpus is undocumented — an exhaustive grep of the
checkout, its bundled skill, and the cached `config.json` found no
statement of what it trained on, no date range, no cutoff. This matters
because every guard in this repo checks `as_of <= decision_time` on rows
fed *in*; a pretrained model carries information in its weights, which
those guards structurally cannot inspect.

One structural fact makes this tractable: in
`timesfm3/mlx/timesfm3_forecaster.py`, **`ts_ids` is never passed to the
model** — it only labels the output. The model cannot see ticker
identity except through the numbers. So memorization needs a *numeric*
key, and the only identifying one is the absolute price level.

**Probe A — level key (sharpest, ~2 min).** Forecast ~10k windows three
ways: (A) log returns, (B) raw close, (C) raw close × a per-window
random constant — identical shape, destroyed level. Compare B vs C by
CRPS converted to return space and normalized by trailing 20d realized
vol (without that normalization the comparison is mechanically
meaningless). **If B beats C with CI excluding zero and >2% effect, the
price level carries information the shape does not — that is
memorization. STOP.** Expected: no difference, and the primary encoding
is returns anyway, which closes this channel by construction.

**Probe B — skill floor.** Rank by `tfm_q50` on a 2021 cache; hits/day
vs B0 0.084 / B1 0.433 / B2 0.603, plus daily cross-sectional Spearman
with date-block bootstrap. If there is no directional skill,
contamination-on-direction is moot. **If cross-sectional IC > +0.05 with
CI excluding zero, stop and investigate** — that number does not exist
in nature on 1,256 Nasdaq names.

**Probe C — exposure tiers.** Mega-caps (likely in any corpus) vs
delisted microcaps (unlikely), coarsened-exact-matched on vol quintile ×
price tercile × month. Only the *directional* leg is diagnostic — a
calibration advantage for mega-caps is expected and is not evidence of
contamination. Pre-declare that so the probe cannot be over-read later.

**Residual, stated in the EXP file:** none of this inspects the weights.
The probes bound the risk behaviourally; feeding returns rather than
prices is the actual structural mitigation.

## Gate R — the cheap kill, and the point of the design

On the 2021 cache, in rank space with per-day cross-sectional demeaning:
Spearman of each TFM feature against each incumbent feature, and
residual R² after projecting out `{vol5, vol20, xs_vol_rank,
relvol_prev, dist52, logpx}`.

> **If every TFM feature has |Spearman| ≥ 0.90 with `xs_vol_rank` or
> residual variance < 10% after projection → STOP. Write EXP-005 as a
> null. No full cache, no model fit.**

~30 minutes, zero model fits, and *more* informative than a fit would
be, because it says **why**: the model is re-parameterizing volatility
rather than adding information. This is the modal outcome.

## Pre-declared features (locked by spec hash before any fit)

| Feature | Definition | Role |
|---|---|---|
| `tfm_q50` | median h=1 forecast, return space | direction — expected null |
| `tfm_q90` | 90th pct h=1 | right-tail level |
| `tfm_spread` | q90 − q10 | **redundancy control** — expect ≈ vol |
| `tfm_skew_norm` | ((q90−q50) − (q50−q10)) / (q90−q10) | **the hypothesis** |
| `tfm_upside_ratio` | (q90−q50) / (q50−q10) | hypothesis, alt scaling |
| `tfm_q90_xs_rank` | cross-sectional pct of `tfm_q90` that day | task is cross-sectional |
| `tfm_skew_xs_rank` | cross-sectional pct of `tfm_skew_norm` | same |

Seven features is not multiple testing **provided there is exactly one
model fit**. Choosing three after seeing precision@10 is. The spec hash
is what makes that enforceable.

## Implementation

**`top10/features/bars_t1.py`** *(new)* — rebuild the 15 EXP-003
features from `universe_liquidity.parquet` alone, pure pandas, names
matching the EXP-004 importance table exactly so both experiments become
citable. Stamp `as_of` = 16:00 ET on `prev_date`.

**`top10/features/tfm.py`** *(new)* — the 7 features above from a
quantile frame. Never imports `timesfm`. Unit-testable with synthetic
quantiles.

**`top10/forecast/timesfm_cache.py`** *(new)* — the ONLY module touching
TimesFM, lazily imported inside the function (mirror the `lightgbm`
pattern in `top10/model.py`). Reuse the quantile/sigma conventions in
`/Users/taboost/quant/research/tfm/forecast.py` rather than reinventing.
Writes `data/raw/timesfm/<model_tag>/<year>.parquet`, resumable per
year. **Internal assertion:** the context's last element must correspond
to `prev_date`, and `ctx_len == min(256, sessions strictly before
trade_date)`.

**`top10/features/spec.py`** *(edit)* — add `T1B_SPEC` and
`T1B_TFM_SPEC`; add both to the `write_features` decision-time ladder;
bump `FEATURE_SPEC_VERSION` to `"2"`. **Also fix a latent defect found
during design:** the task dispatch falls through to `decision_time =
None` for an unknown task, and `assert_decision_time_safe` is then
skipped — so registering any new task name silently disables the leakage
gate on write. The `else` must raise. This bites immediately, because
this plan adds two task names.

**`top10/runner.py`** *(new)* — the committed walk-forward runner
EXP-003/004 lacked. Loads parquet → builds features through
`write_features` so the leakage gate fires on every write →
`expanding_window_splits(retrain="yearly")` → `run_walkforward` →
`compare_to_baseline` → `log_experiment`. **No `unseal_token` parameter
anywhere in this module** — the holdout stays sealed structurally.

**`top10/cli.py`** *(edit)* — extend the existing `walkforward`
subparser with `--variant {t1b, t1b_tfm}`.

**Environment — two processes, deliberately.** The top-10 venv is Python
3.14; TimesFM must not be installed into it. The cache builder runs
under `/Users/taboost/quant/.venv313` and writes parquet; everything in
`top10/` reads that parquet and never imports TimesFM. This keeps a
non-commercially-licensed dependency out of the project graph, makes the
expensive step resumable, and means a null costs nothing to re-verify.
Add `data/raw/timesfm/` to `.gitignore`.

## Family-wise correction

`count_corrected_variants()` currently returns 2. **The TimesFM arm
counts as exactly one variant.** Probes and Gate R produce no
precision@10 claim and do not count — state that in the EXP file.
Volume-as-covariate and an `|r|` target are pre-declared as the *only*
conditional follow-ups, runnable solely if the primary is non-null; if
it is null they are not run and the count stays at 1. Expected
denominator: 3.

## Verification

```bash
cd /Users/taboost/top-10-movers && .venv/bin/python -m pytest -q

# STEP 1 GATE — the repro must land first
.venv/bin/python -m top10.cli walkforward --variant t1b
#   EXPECT mean 0.660 ± 0.02 hits/day; per-year vs B2 +0.100 / +0.076 / +0.045
#   If it does not reproduce, STOP — there is no incumbent to measure against

.venv/bin/python -c "from top10.features.spec import T1B_SPEC, FEATURE_SPEC_VERSION; \
  print(FEATURE_SPEC_VERSION, T1B_SPEC.spec_hash)"
#   EXPECT "2" + a hash pasted into the EXP file BEFORE any fit

# Cache (different interpreter, on purpose)
/Users/taboost/quant/.venv313/bin/python -m top10.forecast.timesfm_cache --year 2021 --ctx 256
#   EXPECT q10<=...<=q90 monotone every row; ctx_len==min(256, prior sessions)
#   Determinism: re-forecast 100 fixed windows, EXPECT max|delta| < 1e-6

# Probes — EXPECT (per HYP-120): hits/day <= B1, IC CI straddling 0,
#   no level-key effect, no Tier-H direction gap

# GATE R — the cheap kill
#   EXPECT (modal): every TFM feature |Spearman| >= 0.90 vs xs_vol_rank
#                   or residual variance < 10%  ->  NULL, stop here

# Only if Gate R passes
.venv/bin/python -m top10.cli walkforward --variant t1b_tfm
#   Read the PER-YEAR TREND, not the mean. EXP-004's precedent binds:
#   +0.100 -> +0.004 -> -0.025 was correctly called null despite a +0.026 mean.
```

**Pass/fail signal:** a margin over EXP-003 that *holds or grows* across
2020 → 2021 → 2022. A shrinking margin is noise and gets reported as
null, exactly as EXP-004 was.

## What this cannot do

It cannot evaluate the pre-registered claim — that needs B4, which needs
premarket bars, still unpulled. It cannot see news, and the plan's own
P8 estimates 40–60% of top movers are driven by unscheduled news that is
absent from price history by definition. A positive result here would be
an internal improvement over EXP-003, nothing more.
