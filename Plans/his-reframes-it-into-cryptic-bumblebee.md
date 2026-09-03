# Structural features: short interest and days-to-cover

## Context

EXP-003 fitted the first real model on 4.6 years of Nasdaq data. It beat the
best baseline (B2, 5-day realized volatility) in 3 of 3 years — but by only
0.06 hits/day, and **the margin shrank as training data grew** (+0.100 →
+0.076 → +0.045). That trajectory is what noise looks like, not signal.

The reason is visible in the feature list: every T1 feature is derived from
price and volume. The model has no information about *why* a stock is
primed to move — no float, no short interest, no crowding, no catalyst. It
is being asked to predict explosive moves from the shape of past returns
alone.

The obvious fix is the "loaded spring" family of structural preconditions:
small float, high short interest, high days-to-cover, thin liquidity. This
plan adds the subset of those that can be built **correctly and for free**,
and documents precisely why the rest cannot.

It deliberately defers premarket/T2 work. That is the larger prize — B4 and
the pre-registered claim both depend on it — but structural features are
cheaper, and if they do not move T1 at all, that is important evidence
about the ceiling before any money is spent.

## What is actually buildable — verified against the live APIs

| Feature | Status | Evidence |
|---|---|---|
| Short interest | **BUILD** | Polygon `/stocks/v1/short-interest` works on the free key, bi-monthly, back to 2017-12-29 |
| Days-to-cover | **BUILD** | Returned precomputed by the same endpoint, using Polygon's *consolidated* `avg_daily_volume` |
| Short interest % float | **PARTIAL** | Needs float; see below. Use short-interest-to-ADV instead |
| Float / market cap | **REJECT** | Not point-in-time — `date=2020-06-01` returns `None`; only current values exist |
| Premarket RVOL | **DEFER** | Requires premarket bars; deferred by decision |
| Options activity / gamma | **REJECT** | No vendor in `docs/DATA_SOURCES.md`, no scaffolding, zero repo references |
| Social mention velocity | **REJECT (forward-only)** | Pushshift is dead; no historical source. Collectable going forward only |
| Scheduled catalysts | **REJECT for backtest** | Finnhub free tier is ~1 month lookback; unusable over 2018–2022 |

### Why float is rejected rather than merely unavailable

Polygon returns `weighted_shares_outstanding` only as a **current** value.
Applying today's share count to a 2020 row imports every subsequent
dilution, buyback and offering. For the microcaps that dominate top-gainer
lists, dilution is enormous *and directly correlated with the squeeze-and-
collapse events being predicted*. It would inflate a backtest convincingly
and fail live — the exact P3 failure mode.

LABEL_SPEC (frozen, hashed) binds this explicitly: "Every upstream row used
to create labels **or later features** must satisfy `as_of <= decision_time`."

Substitute **short-interest-to-ADV** (short shares ÷ consolidated average
daily volume) which is fully point-in-time and captures the same crowding
intuition without the float denominator.

## Blocking defects — fix before any new features

These are not cleanup. Each one silently corrupts the result of the work
that follows.

### 1. The T1 path cannot run on real adapter output
`top10/features/t1.py:339-361` raises `KeyError` for missing
`short_interest_pct_float` / `days_to_cover`, which it reads from
`ticker_meta`. But `TICKER_META_COLUMNS` (`top10/data/base.py:45-66`)
deliberately excludes them — they live in `SHORT_INTEREST_COLUMNS`, fetched
by a separate `short_interest()` method, and **nothing joins the two**.
`pipeline.ingest()` fetches only daily_bars / corporate_actions /
ticker_meta / earnings.

Worse, the guard is gated `if not ticker_meta.empty:` — an empty frame
skips it and silently NaNs every metadata feature.

This is exactly the feature being added, so it must be fixed first.

### 2. The family-wise correction denominator is silently zero
`top10/experiment.py::count_corrected_variants()` returns **0**. Its regex
expects the literal template line `**Counts toward family-wise correction?
(y/n)**: y`; EXP-003 was hand-written as `**Counts toward family-wise
correction?** **YES**`. The true count is 1.

Adding feature families is precisely what multiplies tested variants, so
the Holm denominator must be correct *before* more variants exist. A
denominator of 0 or 1 when the truth is 5 turns a null result into a
"discovery".

### 3. The cost guard is not tracking real spend
`top10/data/cost_guard.py` reads `data/raw/databento/_spend_ledger.json`,
which **does not exist**. The $35.47 actually spent was written to an
ad-hoc `preholdout/_spend.json` by a script that bypassed `CostGuard`. The
guard believes $0 is spent against a $100 ceiling, so it would authorize
$135.47 cumulative — past the $125 credit into real billing.

Not needed for this plan (no spend), but it must be fixed before the
deferred premarket pull.

## Implementation

### Step 1 — Fix the three defects
- `top10/experiment.py`: widen `_COUNTS_LINE_RE` to accept both the
  template form and the prose form, **and** rewrite EXP-003's line to the
  canonical template form. Add a test asserting `count_corrected_variants()
  == 1` against the real `experiments/` directory.
- `top10/data/cost_guard.py`: reconcile the ledger — seed it from
  `preholdout/_spend.json` so `spent` reads $35.47. Add a test that the
  guard refuses a request that would exceed the *credit*, not just the
  ceiling.
- `top10/features/t1.py`: accept short-interest data as its **own frame
  parameter** rather than expecting it merged into `ticker_meta`. Remove
  the `if not ticker_meta.empty` gate so an empty frame raises instead of
  silently NaN-ing.

### Step 2 — Add a short-interest ingest path
`top10/pipeline.py`: add `short_interest` to the `_PIT_DATASETS` tuple and
to `ingest()`'s fetcher dict, so it is persisted to `data/pit/` like every
other source. Reuse the existing `PolygonSource.short_interest()`
(`top10/data/polygon.py:412-489`) — it is implemented, paginates correctly,
and already stamps `as_of` as the publish date (or `settlement_date + 14d`
as a deliberately conservative fallback). Do not reimplement it.

### Step 3 — Feature engineering
New columns appended to `T1_COLUMNS` in `top10/features/spec.py`, bumping
`FEATURE_SPEC_VERSION` from `"1"` to `"2"`:

- `short_interest_shares` — raw, log-scaled
- `short_interest_to_adv` — short shares ÷ consolidated ADV (the float-free
  crowding measure)
- `days_to_cover` — Polygon's precomputed figure
- `short_interest_chg_1p` — change vs the prior bi-monthly reading (the
  *rate of change* is often more informative than the level)
- `short_interest_staleness_days` — days since the reading became knowable;
  bi-monthly data is up to ~3 weeks stale and the model should be able to
  discount accordingly

Reuse `_latest_pit_row` (`top10/features/t1.py:117-132`) for the
as-of-gated lookup — it is the established forward-fill-from-publish-date
pattern and must not be re-invented. Values forward-fill from `as_of` only,
never from `settlement_date`.

### Step 4 — Re-run the walk-forward
Same protocol as EXP-003 so the comparison is clean: expanding window,
yearly retrain, test years 2020/2021/2022, LightGBM binary with
auto-computed `scale_pos_weight`. Report precision@10 and hits/day against
B0/B1/B2, and against **the EXP-003 model itself** — that delta is the
actual question this plan asks.

Log as **EXP-004, counting toward family-wise correction** (variant 2).

## What this plan explicitly does not claim

It does not evaluate the pre-registered success criterion. That requires
B4, which requires premarket bars, which are deferred. A result here is
"structural features improve (or fail to improve) T1 over EXP-003" — a
useful internal comparison and nothing more.

If short interest moves T1 materially, premarket becomes worth its cost. If
it does not, that is strong evidence the T1 ceiling is low, and the
sensible next move is Alpaca's free IEX premarket feed rather than paid
data.

## Verification

```bash
# 1. Defects fixed
./.venv/bin/python -m pytest tests/ -q                    # expect all pass
./.venv/bin/python -c "from top10.experiment import count_corrected_variants; \
  print(count_corrected_variants())"                      # expect 1, not 0
./.venv/bin/python -c "from top10.data.cost_guard import CostGuard; \
  g=CostGuard(); print(g.spent)"                          # expect 35.47, not 0.0

# 2. Short interest ingested and point-in-time
#    every as_of must be >= settlement_date (publish lag), never equal
./.venv/bin/python -c "import pandas as pd; \
  d=pd.read_parquet('data/pit/.../short_interest...'); \
  print((d.as_of >= d.settlement_date).all())"            # expect True

# 3. Features build without KeyError on real adapter output
python -m top10.cli features --task T1 --start 2018-05-01 --end 2022-12-31

# 4. Anti-leakage guards still hold on the new columns
#    assert_self_exclusion must not flag them; shuffle_label_test must fail
./.venv/bin/python -m pytest tests/test_leakage.py -q

# 5. The comparison that matters
./.venv/bin/python <walk-forward script>
#    expect: hits/day vs EXP-003's 0.660, and vs B2's 0.603
```

**Pass/fail signal:** the run either beats EXP-003's 0.660 hits/day by a
margin that holds or grows across 2020 → 2021 → 2022, or it does not.
A margin that shrinks year-over-year — as EXP-003's did — is noise, and
should be reported as such rather than as an improvement.
