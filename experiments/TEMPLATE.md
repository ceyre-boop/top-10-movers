# Experiment Template — docs/PREREG_TOP10.md §5.3

Copy this file to `experiments/EXP-###.md` for every model run you want to
count toward the final claim. **An unlogged run does not count** -- if it
isn't filed here before the holdout is read, it cannot be cited in
`docs/RESULT_TOP10.md` or in the family-wise correction in
`top10/metrics.family_wise_correction`.

## Identity

- **Experiment ID**: EXP-###
- **Date**: YYYY-MM-DD
- **Author**:

## Hypothesis

<!-- One or two sentences: what feature/model change is being tested and
why it should move precision@10 / MAP@10 relative to the prior best run. -->

## Spec hashes

- **Feature spec hash**: `<sha256 via top10.hashing.hash_spec on the feature spec dict>`
- **Label spec hash**: `<sha256, must match docs/PREREG_TOP10.md's frozen label spec hash>`

## Model configuration

- **Task**: T1 | T2
- **Model family**: LightGBM binary classifier | LambdaRank | other
- **Hyperparameters**: <link to config file or inline dict>
- **Feature set**: <list or link to feature spec>

## Split / walk-forward window

- **Train window**: start - end
- **Validation window (walk-forward, expanding)**: start - end
- **Holdout window** (only if this run is the sealed holdout evaluation): start - end

## Results — per-year table

| Year | precision@10 | MAP@10 | mean hits | median hits | n_days |
|------|--------------|--------|-----------|-------------|--------|
|      |              |        |           |             |        |

## Comparison vs B4

- **Mean hits/day delta vs B4**:
- **Years won vs B4** (out of years evaluated):
- **Paired t-test**: statistic = , p-value =
- **Wilcoxon signed-rank test**: statistic = , p-value =

## Family-wise correction

- **Counts toward family-wise correction? (y/n)**:
- If yes, record this experiment's raw p-value here so it can be included
  in the `top10.metrics.family_wise_correction` call over all counted
  variants: p-value =

## Notes / caveats

<!-- Anything that would make a reviewer distrust this number: data gaps,
survivorship concerns, corporate-action artifacts, etc. -->
