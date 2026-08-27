# TOP10 — Daily Top-Movers Predictor

This repository contains the research scaffolding for a daily top-movers predictor that ranks the 10 tickers most likely to appear on the day's top-gainers list before the market opens.

## Current repository contents

- `docs/LABEL_SPEC.md` — frozen proxy-label definition
- `docs/LABEL_SPEC.sha256` — hash for the committed label spec
- `docs/PREREG_TOP10.md` — pre-registration draft for the sealed holdout
- `docs/CEILING.md` — ceiling-estimation protocol
- `docs/RESULT_TOP10.md` — reserved for the one-time sealed holdout result
- `data/` — repository layout for raw, point-in-time, labels, features, and predictions
- `experiments/` — reserved for experiment logs

## Secure configuration

Databento credentials must be supplied through a secure environment variable or repository secret named `DATABENTO_API_KEY`.

- Local development: create a local `.env` file that is not committed
- GitHub Actions / CI: add a repository or environment secret named `DATABENTO_API_KEY`

The actual API key is intentionally not stored anywhere in this repository.