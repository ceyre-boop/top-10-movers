# LABEL_SPEC — TOP10 Daily Top-Movers Predictor

Status: frozen v1  
Task scope: proxy-label definition only  
Primary task split: T1 (prior close), T2 (09:25 ET with premarket)

## Universe

For each trading day `t`, define the candidate universe using information available as of the prior close:

- US common stocks and ADRs listed on NYSE, NASDAQ, or AMEX
- Prior close price `>= 1.00`
- 20-day average daily dollar volume `>= 1,000,000`
- Include names that are later delisted
- Exclude OTC securities
- Flag and evaluate, rather than automatically keep, ETFs, warrants, rights, units, and pre-merger SPACs

## Label

For each ticker in the day's universe:

`return_t = (close_t / close_t-1) - 1`

`y = 1` when the ticker ranks in the top 10 by `return_t` within the same day-universe; otherwise `y = 0`.

Persist:

- `trade_date`
- `ticker`
- `rank`
- `return_t`
- `label`
- `label_spec_version`

## Corporate-action exclusions

Before ranking:

- Join corporate actions point-in-time
- Exclude split and reverse-split days from labels
- Track ticker changes and delisting boundaries point-in-time

If the daily top-10 median move is implausibly large because of a corporate action artifact, the label set is invalid and must be rebuilt.

## Proxy validation

Starting on live collection day:

- At 16:05 ET, capture the real Robinhood top movers list
- Store the raw response, timestamped, without editing
- Weekly, compute overlap between the proxy top 10 and the captured Robinhood list
- Tune only universe filters until median overlap across 30 days is at least `8/10`
- Once the overlap target is met, freeze the filters and log the freeze date

## Point-in-time invariants

Every upstream row used to create labels or later features must satisfy:

- `as_of <= decision_time`
- Delisted-later symbols remain eligible before delisting
- No adjusted-price artifact may leak future corporate actions into historical rows

## Data dependencies

Minimum required inputs:

- Daily OHLCV including delisted symbols
- Corporate actions: splits, dividends, ticker changes
- Ticker metadata with point-in-time active ranges

## Output path

Label files are written under:

`data/labels/<label-spec-hash>/`
