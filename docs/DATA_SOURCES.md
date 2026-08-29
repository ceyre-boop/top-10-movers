# DATA_SOURCES — vendor coverage decision table

Status: living doc, updated when a new adapter is wired up (`top10/data/*.py`).

## VERIFIED-AGAINST-LIVE-DATA CORRECTION (read this first)

A LIVE test run against real Databento data found the adapter's PREVIOUS
model of Databento equities was wrong in ways that silently produced
garbage -- a median top-10-gainer return of **+8702%** and rows where two
unrelated tickers shared an identical close. Everything below is verified
against real API responses, not theorized, and the fix is now live in
`top10/data/databento.py` / `top10/data/symbology.py`.

1. **`instrument_id` is REASSIGNED DAILY, not a stable per-company id.**
   LULU's `instrument_id` across seven consecutive sessions: `6844, 6843,
   6839, 6840, 6841, 6837, 6837`. MRNA's: `7345, 7343, 7339, 7340, 7342,
   7338, 7340`. (AAPL happened to stay `27` throughout, which is why
   spot-checking one liquid name hid this bug.) A range-wide
   `symbology.resolve(start_date=..., end_date=...)` -- one interval
   spanning many days -- is therefore WRONG: it was labeling id `6844` as
   "LULU" on every day in range, when the very next day `6844` could be a
   different company. **Fix:** `SymbolResolver.resolve_day(day,
   instrument_ids)` resolves EXACTLY one trading day at a time
   (`start_date=D`, `end_date=D+1`), batched in groups of 500 ids, cached
   to disk per `(dataset, day)` and never re-resolved. The old
   `resolve_range` method (persisted multi-day intervals) has been
   DELETED, not deprecated, so it cannot be reached by accident.

2. **No consolidated US equities dataset exists before 2023.** Verified via
   `metadata.get_dataset_range`: the three LISTING-VENUE feeds --
   `XNAS.ITCH` (Nasdaq), `XNYS.PILLAR` (NYSE), `XASE.PILLAR` (AMEX) --
   all have `ohlcv-1d` from **2018-05-01**. `EQUS.SUMMARY` starts
   2024-07-01; `DBEQ.BASIC`/`EQUS.MINI`/`IEXG.TOPS`/`XCIS`/`XCHI` start
   2023-03-28 -- every consolidated feed starts INSIDE this project's
   2023-01-01 holdout, useless for training. **Fix:**
   `DatabentoSource.daily_bars` now UNIONS the three listing-venue
   datasets (`LISTING_VENUE_DATASETS`), never a single dataset.

3. **Per-venue feeds are partial -- volume and price need different
   treatment.** Each listing-venue dataset contains only trades EXECUTED
   ON that one exchange.
   - **Volume**: SUMMED across the three fetched venues for a
     `(date, ticker)`. This still UNDERSTATES true consolidated (SIP)
     volume, since ARCX/IEXG/MEMX/etc. venues are not fetched at all --
     this directly reduces the effective $1M ADV universe filter in
     LABEL_SPEC relative to a SIP-based filter.
   - **Close**: taken from the LISTING venue's bar ONLY (the official
     closing auction runs on the listing venue), from the `definition`
     schema's `exchange` field -- NEVER averaged or "last one wins"
     across venues.

4. **`ohlcv-1d` `ts_event` is UTC midnight OF the trade date.** Converting
   it to America/New_York shifts every bar back one day and manufactures
   phantom Sunday sessions (verified: a "2022-06-05" row appeared from a
   Monday bar). **Fix:** the UTC calendar date is used directly, never
   tz-converted; `as_of` is then stamped at 16:00 ET of that date per the
   normal P4 contract.

5. **`to_df(map_symbols=True)` does NOT populate `symbol` for
   `ALL_SYMBOLS` requests** (verified: all 30,379 rows null in a live
   pull). The adapter never reads `record["symbol"]` for daily bars --
   ticker resolution goes exclusively through `SymbolResolver.resolve_day`.

6. **`definition` schema gives listing venue + security type.**
   `raw_symbol`, `exchange` (listing venue), `security_type`/
   `instrument_class`. Verified values: `C` (common stock), `E` (ETF),
   `W` (warrant), `P` (preferred), `U` (unit), plus `Q`/`O`/`I`/`A`/`H`/
   `M` (flagged, not silently treated as common stock). Definition pulls
   cost ~$0.065/venue/day, so they are pulled for ONE representative day
   per calendar month per venue and carried forward for that whole month
   -- a documented staleness tradeoff (a mid-month listing-venue change
   misclassifies the close venue for the rest of that month), not a bug.

7. **Databento flags degraded days** (verified live: a pull warned
   `2022-07-18 (degraded), 2022-07-25 (degraded)`). `daily_bars` now
   calls the free `metadata.get_dataset_condition` endpoint and EXCLUDES
   degraded/pending days from its output by default (a degraded day
   silently produces wrong returns), recording which dates were dropped
   on `DatabentoSource.last_degraded_dates`.

### Verified Databento costs (`ohlcv-1d`, `ALL_SYMBOLS`, all three
listing-venue datasets unioned)

| Range | Cost |
|---|---|
| One day | $0.037 |
| 2018-05-01 -> 2023-01-01 (pre-holdout training window) | $35.47 |
| 2018-05-01 -> 2026-08-29 (full history to date) | $68.61 |

Comfortably inside both the $125 promotional credit and `CostGuard`'s
default $100 ceiling.

This is the decision table for "which vendor(s) do I need to run T1 / T1+T2".
Cross-reference `docs/LABEL_SPEC.md`'s P2 survivorship-bias requirement
("Include names that are later delisted") and `top10/data/base.py`'s frozen
`MarketDataSource` contract before picking a combination.

## Requirement x source matrix

| Requirement | CRSP (WRDS) | Databento | Polygon | Composite | Alpaca | Finnhub | Tiingo | EODHD |
|---|---|---|---|---|---|---|---|---|
| Daily OHLCV, incl. delisted (P2) | Yes — decades, delisted names + delisting returns via `dsedelist` | Yes — `ohlcv-1d`, ALL_SYMBOLS incl. delisted, but only from **2018-05-01** | Yes — grouped-daily incl. delisted, but vendor's "latest known" ticker snapshot, not truly point-in-time | Yes — routed to Databento | No bulk delisted-history feed | No | Partial — survivor-biased free tier; paid tier has some delisted coverage | Partial — paid tier only |
| Corporate actions (splits/div/ticker chg) | Yes — `dsedist`, `dsenames` transitions, `dsedelist` | No | Yes — `/v3/reference/splits`, `/v3/reference/dividends`; no bulk ticker-change feed | Yes — routed to Polygon, free tier; point-in-time `instrument_id` cross-check via Databento symbology | No | No | Limited | Yes (paid) |
| Ticker metadata, point-in-time | Yes — `dsenames` `namedt`/`nameendt` are genuinely point-in-time | No | Best-effort — vendor's LATEST classification only, not true point-in-time history | Yes — routed to Polygon (same "latest known" limitation), cross-checked against Databento symbology | No | No | No | Limited |
| Premarket minute bars (04:00–09:25 ET) | **No** — standard CRSP has no intraday data; separate CRSP intraday product often not in a university subscription | Yes — `ohlcv-1m`, paid, from 2018-05-01 | Yes — paid | Yes — routed to Databento | Yes — **free**, IEX feed, 2016+ | No | No | No |
| Earnings calendar | **No** | No | Yes — best-effort, plan-dependent | Yes — routed to Finnhub (free tier, ~1 month lookback in practice) | No | Yes — free tier, but in practice only ~last 1 month of history regardless of requested range (see `top10/data/free_tier.py`) | No | Limited |
| Short interest (FINRA) | **No** | No | Yes — `/stocks/v1/short-interest`, plan-dependent | Routed to Polygon; raises `NotImplementedError` if the account's plan doesn't include it | No | No | No | No |

## `CompositeSource` (`MARKET_DATA_VENDOR=composite`) — the recommended default

**This is the recommended vendor for this project.** No single free source
covers everything this project needs (see matrix above); `top10.data.
composite.CompositeSource` routes each `MarketDataSource` method to
whichever already-wired-up vendor actually covers it, per the explicit
`ROUTING` table in `top10/data/composite.py` (also readable via
`CompositeSource.describe_routing()`):

| Method | Routed to | Why |
|---|---|---|
| `daily_bars` | Databento | P2-safe full daily cross-section (`ALL_SYMBOLS`, incl. delisted names), 2018-05-01+ |
| `premarket_bars` | Databento | full-tape 04:00–09:25 ET minute bars, 2018-05-01+ |
| `corporate_actions` | Polygon | `/v3/reference/splits` + `/v3/reference/dividends` — small reference endpoints, free tier |
| `ticker_meta` | Polygon | free bulk reference ticker list, cross-checked against Databento symbology for `instrument_id` alignment |
| `earnings` | Finnhub | `FinnhubEarnings`, free tier |
| `short_interest` | Polygon | plan-dependent; raises `NotImplementedError` if unavailable |

**`MARKET_DATA_VENDOR=composite` requires BOTH `DATABENTO_API_KEY` and
`POLYGON_API_KEY`** (Polygon's free tier needs no card) to be set — a
missing key breaks only the capability routed to that vendor (e.g. a
missing `FINNHUB_API_KEY` breaks `earnings` but not `daily_bars`), never
silently degrades to an empty frame. **The Polygon free key is required
for correct labels: without `corporate_actions`, `top10.labels.
build_labels` cannot exclude split days (docs/LABEL_SPEC.md
"Corporate-action exclusions"), and every label set built without it is
contaminated by split-day price artifacts (e.g. a 1:20 reverse split
reading as a +1900% return).**

Cross-vendor risk this closes: Databento identifies instruments by
`instrument_id`; Polygon identifies them by ticker string. A ticker string
IS reused across unrelated issuers over time — joining a Polygon split
onto Databento bars by ticker string alone, with no regard for reuse,
would apply the wrong split to the wrong company. `CompositeSource`
resolves each `corporate_actions`/`ticker_meta` row's `instrument_id`
point-in-time (AT that row's own date, via `top10.data.symbology.
SymbolResolver`), and `CompositeSource.alignment_report(start, end)`
surfaces every ticker that could NOT be confidently resolved — those rows
are never silently dropped, only flagged.

Every delegate call goes through the existing on-disk `cached_call`
(`top10/data/cache.py`), same as every other adapter here: a second
identical `(capability, start, end)` request performs zero network calls.
The Polygon delegate is constructed at `calls_per_min=5` to match Polygon's
actual free-tier ceiling.

**What `composite` still does NOT solve** (same gaps as the underlying
vendors — see "What Databento's P2 fix does NOT cover" below): no
authoritative delisting reason/return, no float-shares figure. If any of
these prove load-bearing, **EODHD at $19.99/mo** remains the cheapest
paid supplement (see below).

## What solves P2 (survivorship bias) at research grade

**The user running this project has NO WRDS/CRSP access.** CRSP is the only
source in the matrix above that solves P2 at research grade in the
traditional sense (decades of history, authoritative `dlstcd`/`dlret`
delisting reason + return via `crsp.dsedelist`) — but it is not available
here, so it cannot be the answer to the plan's P2 kill criterion for this
user. **Databento is what actually solves P2 for this project**, starting
2018-05-01, via a different mechanism than CRSP's authoritative delisting
table:

- `DatabentoSource.daily_bars` pulls the FULL daily cross-section for every
  requested trading day (`symbols="ALL_SYMBOLS"`), never a per-symbol pull
  over a pre-resolved ticker list. Survivorship bias enters when a backtest
  is built from TODAY's ticker list and history is pulled for those names
  only; it does NOT enter when each day's universe is built from THAT
  DAY's true cross-section, because every symbol that traded that day is
  present in that day's data, delisted-by-today names included. This is
  the core P2 defense and is marked as such directly in
  `top10/data/databento.py`.
- `top10/data/symbology.py`'s `SymbolResolver.resolve_day` performs
  PER-TRADING-DAY `instrument_id -> raw_symbol` resolution (see
  "VERIFIED-AGAINST-LIVE-DATA CORRECTION" above — Databento's
  `instrument_id` is REASSIGNED DAILY, so it is informational-only, NOT
  the stable cross-day identifier CRSP's `permno` is). `daily_bars`
  therefore resolves `ticker` fresh for every single trading day, batched
  in groups of 500 ids, and it is `ticker` — not `instrument_id` — that is
  now the safe, stable identity to `groupby`/join on across days in a
  pulled frame (see `infer_delistings`, which is keyed on `ticker` for
  exactly this reason). `instrument_id` is still carried through
  `daily_bars` as an extra column beyond the frozen contract, but purely
  as informational per-row metadata.
- `infer_delistings()` (in `top10/data/databento.py`) derives delisting
  events from `daily_bars` itself, since Databento has no CRSP-style
  delisting-event table. This is explicitly INFERRED, not authoritative —
  see its docstring for the halt-vs-delisting distinction and the
  `confidence` column (never a bare boolean).
- `verify_no_survivorship()` is the loud, runnable check that this is
  actually working on a given pulled frame: it asserts that tickers
  present early in the window are absent by the end of it, and treats
  "zero disappearances" as a FAIL — the survivorship-bias signature is
  otherwise completely invisible (plausible prices, plausible volumes,
  plausible date range; only the ticker cast fails to turn over).

Databento and Polygon both technically include delisted names in their bulk
daily-bars pulls, but:
- **Databento's history starts 2018-05-01** (`FIRST_AVAILABLE_DATE` in
  `top10/data/databento.py`) — it cannot backfill anything earlier at all.
- **Polygon's ticker metadata is NOT truly point-in-time** — its bulk
  ticker-list endpoint reflects each ticker's LATEST known classification,
  not what applied on every historical trade date (see the "KNOWN
  LIMITATION" note in `PolygonSource.ticker_meta`). `active_from`/
  `active_to` are correct; `name`/`security_type`/`exchange`/`market_cap`
  are not point-in-time.

CRSP's `dsenames` table, by contrast, updates `namedt`/`nameendt` exactly
when a name/ticker/exchange/share-code change takes effect — genuinely
point-in-time, not a "latest known" snapshot. If this project ever gets
WRDS access, CRSP remains the strictly stronger long-run answer to P2
(decades vs. 2018-05-01+, authoritative delisting reason/return vs.
inferred-with-confidence) and should replace the Databento-based pipeline
described above rather than run alongside it.

### What Databento's P2 fix does NOT cover

Even with the pieces above wired up, several things the plan may still need
are simply absent from Databento and are NOT solved by this fix:
- **No authoritative delisting reason or delisting return** — `dlstcd`/
  `dlret`'s CRSP analogs do not exist; `infer_delistings()`'s `confidence`
  column is a coarse, undocumented-probability triage signal, not a
  research-grade substitute.
- **No float-shares figure** (same gap CRSP itself has).
- **No earnings-calendar feed** (`DatabentoSource.earnings` still raises).
- **No FINRA short-interest feed** (`DatabentoSource.short_interest` still
  raises).
- **No splits/dividends/ticker-change feed** (`DatabentoSource.
  corporate_actions` still raises for everything except the inferred-
  delisting rows described above).

**If the free/already-budgeted path above proves insufficient** (e.g. the
survivorship-verification check fails, or one of the gaps above turns out
to be load-bearing), **EODHD at $19.99/mo** is the cheapest paid
supplemental option in this matrix: per the requirement table, its paid
tier adds delisted-name daily-bars coverage, splits/dividends/ticker-change
corporate actions, and market-cap/float ticker metadata that Databento does
not carry at all — at the cost of being a "limited"/paid-tier feed rather
than a full-tape, research-grade one. It would slot in as a
`ticker_meta`/`corporate_actions` supplement alongside Databento's
`daily_bars`, not a replacement for it.

## What is needed for T1-only vs. T1+T2

- **This user has no WRDS/CRSP access — the recommended path is
  `MARKET_DATA_VENDOR=composite`.** It covers T1 (daily bars, corporate
  actions, ticker metadata) AND T2 (premarket bars) in one vendor-agnostic
  adapter, per the routing table above, requiring only `DATABENTO_API_KEY`
  and `POLYGON_API_KEY` (free, no card). See "`CompositeSource`" above for
  what it does and does not solve.
- **T1-only (prior-close decisions)** can also run entirely on **CRSP**,
  if/when WRDS access exists: daily bars, corporate actions, and
  point-in-time ticker metadata are all covered by `crsp.dsf` /
  `crsp.dsedist` / `crsp.dsenames` / `crsp.dsedelist`. `top10.data.crsp.
  CRSPSource.premarket_bars` raises `NotImplementedError` — this is
  expected and does not block T1.
- **T1+T2 (adds 09:25 ET premarket decisions) requires a second source**
  for premarket minute bars, since CRSP's standard equity database has no
  intraday data and the separate CRSP intraday product is frequently not
  part of a university subscription. Two options:
  - **Alpaca** (free, IEX feed, 2016+) — cheapest path, but see the IEX
    caveat below.
  - **Databento** (`ohlcv-1m`, paid, 2018-05-01+) — full-tape coverage,
    but no history before 2018-05-01 and adds a cost-guarded paid feed to
    the pipeline (`top10/data/cost_guard.py`). This is what `composite`
    routes `premarket_bars` to.
- Earnings calendar and short interest are **not** covered by CRSP or
  Databento at all; Polygon (`PolygonSource.earnings`,
  `PolygonSource.short_interest`) or Finnhub (`FinnhubEarnings`, free tier,
  ~1 month lookback in practice) are the only wired-up sources for those.

**Recommended combination:**
- No WRDS access (this project, today): `MARKET_DATA_VENDOR=composite`
  (Databento + Polygon free tier + Finnhub free tier, all one adapter).
- Once WRDS/CRSP access is confirmed: CRSP alone for T1-only; for T1+T2,
  CRSP (bars/corp-actions/ticker-meta) + Alpaca or Databento (premarket
  bars) + Polygon or Finnhub (earnings, short interest).

## IEX ~2.5%-of-SIP caveat (premarket dollar-volume features)

Alpaca's free tier is backed by the **IEX** feed, which historically
represents roughly **2.5% of consolidated (SIP) volume**. Any premarket
dollar-volume feature built on Alpaca/IEX bars is therefore a small, noisy
sample of true premarket activity, not the full tape — this matters
specifically for features that rank/threshold on premarket dollar volume
(the T2 decision path), where IEX's low share of SIP volume can make a
thinly-traded name look artificially quiet, or a single large IEX-routed
order look artificially dominant. Databento's `ohlcv-1m` (full-tape) does
not have this caveat, at the cost of history starting 2018-05-01 and being
a paid, cost-guarded feed.

## History span summary

| Source | Daily bars history | Premarket bars history |
|---|---|---|
| CRSP | Decades (WRDS subscription start varies) | Not available (standard file) |
| Databento | **2018-05-01+** (`FIRST_AVAILABLE_DATE`) | 2018-05-01+ |
| Polygon | Vendor-dependent, generally deep | Vendor-dependent |
| Alpaca | N/A (not used for daily bars here) | 2016+ (IEX feed) |
