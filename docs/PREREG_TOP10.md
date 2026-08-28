# PREREG_TOP10 — Sealed Holdout Plan

Status: draft until Phase 6 freeze  
Holdout period: `2023-01-01` through data end  
Primary metric: `precision@10`  
Secondary metric: `MAP@10`

## Task definitions

- **T1**: predict before market open using information available as of prior close
- **T2**: predict at 09:25 ET using prior-close information plus premarket features

## Frozen label reference

- Label spec: `docs/LABEL_SPEC.md`
- Label spec hash: `6ed1fa78ee1e1b4c5c8aaac3d9030a96698b1ba667972502eb8b7b039b2a2fa3`

## Baselines to beat

- B0 Random
- B1 Yesterday's top 10 repeated
- B2 Highest 5-day realized volatility
- B3 Earnings-today intersected with highest 20-day volatility
- B4 Top 10 by premarket gap percent with premarket dollar volume threshold for T2

## Model family

- LightGBM binary classifier as baseline model
- Optional LambdaRank variant
- Walk-forward expanding-window evaluation only

## Anti-leakage requirements

- Every feature row carries `as_of <= decision_time`
- No future-adjusted prices
- Self-exclusion invariant enforced
- Shuffle-label test must collapse to near-random performance

## Success criterion

Primary success claim requires beating B4 by at least 1.0 average hits/day on holdout with corrected significance for the tested model family.

## Data coverage constraint (added before freeze)

Primary vendor: **Databento** (`MARKET_DATA_VENDOR=databento`).

Databento US equities history begins **2018-05-01**. The original plan assumed a
2015 start; that history is not available at this budget. Actual usable spans:

- Pre-holdout (train + validation): `2018-05-01` -> `2022-12-31`
- Sealed holdout: `2023-01-01` -> data end

This yields four full pre-holdout calendar years (2019, 2020, 2021, 2022) plus a
partial 2018.

### Amended per-year gate

The original criterion — "beats B4 in at least 5 of 7 years" — is unsatisfiable
under this coverage and is replaced, before any holdout run, by:

> The model must beat B4 in **at least 4 of the 5** pre-holdout calendar years
> (2018, 2019, 2020, 2021, 2022). A calendar year counts only if it contains at
> least 100 trading days; 2018 is therefore included only if its post-May-01
> span qualifies, and is otherwise excluded, reducing the requirement to 3 of 4.

Rationale: the original gate demanded a ~71% year-win rate. The amended gate
holds that proportion rather than relaxing it, so shorter history costs
statistical power but does not lower the bar.

### Consequences for interpretation

- Fewer regime-distinct years. 2020-2021 (COVID + meme era) is a large share of
  the remaining sample, so per-year variance is expected to be high and any
  aggregate result is more regime-dependent than the original design intended.
- The family-wise correction in `top10/metrics.py` is unchanged and still applies
  to the count of model variants logged in `experiments/`.

### Premarket data-source caveat (affects B4)

B4 thresholds on premarket dollar volume. If premarket bars are sourced from an
IEX-only free feed (Alpaca/Tiingo), that covers roughly 2.5% of consolidated
volume, so a `$500k` IEX threshold is NOT equivalent to a `$500k` SIP threshold.
The premarket source used for the holdout run must be recorded in
`docs/RESULT_TOP10.md`, and B4 and the model must use the SAME source. Comparing
a SIP-fed model against an IEX-fed B4 would invalidate the primary claim.

## One-time holdout rule

This document must be finalized and committed before the holdout is run. `docs/RESULT_TOP10.md` is reserved for the single recorded holdout outcome.
