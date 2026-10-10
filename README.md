# AI Trader Starter

A lightweight Python trading bot template designed for market research and paper trading. It combines technical feature engineering with a simple probabilistic signal model.

## What it does

- Loads market data for a symbol such as XAUUSD
- Creates technical features such as RSI, MACD, ATR, volatility, range, and trend strength
- Trains a logistic-regression signal model from historical data
- Runs a simple backtest with risk controls and drawdown monitoring
- Provides a paper-trading entry point for a live-friendly workflow

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Reproducible canonical datasets

Canonical raw data is resolved from `TRADING_DATA_ROOT`, which must point to
the directory containing `dukascopy/` and
`external_secondary/octa_mt4/XAU_15m_data.csv`. Without the variable, the
default is this repository's `data/raw` directory. Relative overrides are
resolved from the repository root. For example, in PowerShell:

```powershell
$env:TRADING_DATA_ROOT = 'C:\TWS\New - test\data\raw'
python scripts\verify.py
```

The verifier streams the canonical files, checks the pinned manifest at
`data/dataset_manifest.json`, runs the full test suite, and checks
`git diff --check`. The manifest aggregate is SHA-256 over UTF-8 JSON with
sorted keys, compact separators, and no `aggregate_sha256` field; its file
entries are sorted by relative path and contain no machine-specific paths.
Raw datasets remain ignored by Git. Dataset-dependent tests skip with an
explicit diagnostic only when data is not configured; a missing file under a
configured data root fails the test.

## Demo mode

By default, the app generates synthetic OHLCV data so it can run without external subscriptions or local CSV files.

## Important disclaimer

This project is for research and education. It is not financial advice and should not be used for live deployment without extensive validation and risk review.

## Project structure

- `main.py` - entry point
- `src/data_loader.py` - data sourcing and synthetic generation
- `src/features.py` - technical indicator construction
- `src/models.py` - signal model training and scoring
- `src/backtest.py` - risk-aware backtesting loop
- `src/paper_trader.py` - execution and monitoring wrapper
- `config.yaml` - configuration

## Customization ideas

- Replace the synthetic feed with broker or exchange data
- Add more features and ensemble models
- Add execution rules and order management
- Add regression tests and validation pipelines
- Connect to a broker or API layer
