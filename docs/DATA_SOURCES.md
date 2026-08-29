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

**CRSP is the only source here that solves P2 at research grade.** It carries
delisted securities with proper delisting-return handling (`crsp.dsedelist`:
`dlstdt`, `dlstcd`, `dlret`) going back decades — this is exactly what the
LABEL_SPEC universe rule ("Include names that are later delisted") requires
for a research-grade backtest, not just a forward-looking pilot.

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
point-in-time, not a "latest known" snapshot.

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
