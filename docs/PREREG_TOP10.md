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

## One-time holdout rule

This document must be finalized and committed before the holdout is run. `docs/RESULT_TOP10.md` is reserved for the single recorded holdout outcome.
