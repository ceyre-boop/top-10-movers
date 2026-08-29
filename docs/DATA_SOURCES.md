# DATA_SOURCES — vendor coverage decision table

Status: living doc, updated when a new adapter is wired up (`top10/data/*.py`).

This is the decision table for "which vendor(s) do I need to run T1 / T1+T2".
Cross-reference `docs/LABEL_SPEC.md`'s P2 survivorship-bias requirement
("Include names that are later delisted") and `top10/data/base.py`'s frozen
`MarketDataSource` contract before picking a combination.

## Requirement x source matrix

| Requirement | CRSP (WRDS) | Databento | Polygon | Alpaca | Finnhub | Tiingo | EODHD |
|---|---|---|---|---|---|---|---|
| Daily OHLCV, incl. delisted (P2) | Yes — decades, delisted names + delisting returns via `dsedelist` | Yes — `ohlcv-1d`, ALL_SYMBOLS incl. delisted, but only from **2018-05-01** | Yes — grouped-daily incl. delisted, but vendor's "latest known" ticker snapshot, not truly point-in-time | No bulk delisted-history feed | No | Partial — survivor-biased free tier; paid tier has some delisted coverage | Partial — paid tier only |
| Corporate actions (splits/div/ticker chg) | Yes — `dsedist`, `dsenames` transitions, `dsedelist` | No | Yes — `/v3/reference/splits`, `/v3/reference/dividends`; no bulk ticker-change feed | No | No | Limited | Yes (paid) |
| Ticker metadata, point-in-time | Yes — `dsenames` `namedt`/`nameendt` are genuinely point-in-time | No | Best-effort — vendor's LATEST classification only, not true point-in-time history | No | No | No | Limited |
| Premarket minute bars (04:00–09:25 ET) | **No** — standard CRSP has no intraday data; separate CRSP intraday product often not in a university subscription | Yes — `ohlcv-1m`, paid, from 2018-05-01 | Yes — paid | Yes — **free**, IEX feed, 2016+ | No | No | No |
| Earnings calendar | **No** | No | Yes — best-effort, plan-dependent | No | Yes — free tier, but in practice only ~last 1 month of history regardless of requested range (see `top10/data/free_tier.py`) | No | Limited |
| Short interest (FINRA) | **No** | No | Yes — `/stocks/v1/short-interest`, plan-dependent | No | No | No | No |

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
- `top10/data/symbology.py`'s `SymbolResolver` adds point-in-time
  raw_symbol <-> `instrument_id` resolution (Databento's stable identifier,
  carried through `daily_bars` as an extra column beyond the frozen
  contract, same pattern as CRSP's `permno`) — this closes the symbol-reuse
  gap that survivorship-fixed-but-ticker-keyed data would otherwise still
  have: a raw ticker string IS reused across unrelated issuers over time,
  and `detect_reuse()` makes every such collision visible rather than
  silently merging two companies' histories into one series.
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

- **T1-only (prior-close decisions)** can run entirely on **CRSP**: daily
  bars, corporate actions, and point-in-time ticker metadata are all
  covered by `crsp.dsf` / `crsp.dsedist` / `crsp.dsenames` /
  `crsp.dsedelist`. `top10.data.crsp.CRSPSource.premarket_bars` raises
  `NotImplementedError` — this is expected and does not block T1.
- **T1+T2 (adds 09:25 ET premarket decisions) requires a second source**
  for premarket minute bars, since CRSP's standard equity database has no
  intraday data and the separate CRSP intraday product is frequently not
  part of a university subscription. Two options:
  - **Alpaca** (free, IEX feed, 2016+) — cheapest path, but see the IEX
    caveat below.
  - **Databento** (`ohlcv-1m`, paid, 2018-05-01+) — full-tape coverage,
    but no history before 2018-05-01 and adds a cost-guarded paid feed to
    the pipeline (`top10/data/cost_guard.py`).
- Earnings calendar and short interest are **not** covered by CRSP or
  Databento at all; Polygon (`PolygonSource.earnings`,
  `PolygonSource.short_interest`) or Finnhub (`FinnhubEarnings`, free tier,
  ~1 month lookback in practice) are the only wired-up sources for those.

**Recommended combination once WRDS/CRSP access is confirmed:**
- T1-only: CRSP alone.
- T1+T2: CRSP (bars/corp-actions/ticker-meta) + Alpaca or Databento
  (premarket bars) + Polygon or Finnhub (earnings, short interest).

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
