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
