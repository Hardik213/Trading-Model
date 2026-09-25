from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def _resolve_mt5_path() -> str:
    return os.getenv("MT5_PATH") or "C:/Program Files/MetaTrader 5/terminal64.exe"


def _mt5_timeframe(timeframe: str) -> int:
    mapping = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 16385,
        "4h": 16388,
        "1d": 16390,
        "1w": 16391,
    }
    key = str(timeframe).lower().replace(" ", "")
    if key in mapping:
        return mapping[key]
    return 16385


def get_live_mt5_data(symbol: str, start_date: str, end_date: str, timeframe: str = "1h") -> pd.DataFrame | None:
    try:
        import MetaTrader5 as mt5
    except Exception:  # pragma: no cover
        return None

    try:
        path = _resolve_mt5_path()
        initialized = mt5.initialize(path=path)
        if not initialized:
            login = os.getenv("MT5_LOGIN")
            password = os.getenv("MT5_PASSWORD")
            server = os.getenv("MT5_SERVER")
            if login:
                initialized = mt5.initialize(path=path, login=int(login), password=password or "", server=server or "")
        if not initialized:
            return None

        mt5.symbol_select(symbol, True)
        start_dt = pd.Timestamp(start_date).to_pydatetime()
        end_dt = pd.Timestamp(end_date).to_pydatetime()
        rates = mt5.copy_rates_range(symbol, _mt5_timeframe(timeframe), start_dt, end_dt)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        if "time" not in df.columns:
            return None
        df["Datetime"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        })
        df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].set_index("Datetime")
        df.index.name = "Datetime"
        df["Symbol"] = symbol.upper()
        return df
    except Exception:  # pragma: no cover
        return None
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def generate_synthetic_ohlcv(symbol: str, periods: int = 600, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start, periods=periods, freq="h")
    base_price = 2100.0
    drift = 0.0005
    volatility = 0.009

    returns = rng.normal(drift, volatility, periods)
    closes = [base_price]
    opens = [base_price]
    highs = [base_price]
    lows = [base_price]
    volumes = [1000]

    for i in range(1, periods):
        prev_close = closes[-1]
        close = max(prev_close * (1.0 + returns[i]), 1.0)
        open_price = opens[-1]
        high = max(open_price, prev_close, close) * (1.0 + abs(rng.normal(0, volatility)))
        low = min(open_price, prev_close, close) * (1.0 - abs(rng.normal(0, volatility)))
        volume = max(100, int(1000 + rng.normal(0, 250)))
        closes.append(float(close))
        opens.append(float(open_price))
        highs.append(float(high))
        lows.append(float(low))
        volumes.append(volume)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
    df.columns = [c for c in df.columns]
    df.index.name = "Datetime"
    df["Symbol"] = symbol
    return df


def get_data_path(symbol: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / f"{symbol.lower()}.csv"


def load_market_data(symbol: str, start_date: str, end_date: str, use_live_data: bool = False, timeframe: str = "1h", allow_synthetic_fallback: bool = False) -> pd.DataFrame:
    symbol = symbol.upper()
    file_path = get_data_path(symbol)
    df = None

    if use_live_data:
        df = get_live_mt5_data(symbol, start_date, end_date, timeframe=timeframe)
        if df is not None:
            df.attrs["market_source"] = "live_mt5"
    elif file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.attrs["market_source"] = "csv"

    if df is None and not use_live_data:
        df = get_live_mt5_data(symbol, start_date, end_date, timeframe=timeframe)
        if df is not None:
            df.attrs["market_source"] = "live_mt5"

    if df is None:
        if allow_synthetic_fallback:
            df = generate_synthetic_ohlcv(symbol, periods=600, start=start_date)
            df.attrs["market_source"] = "synthetic"
        else:
            raise RuntimeError(f"No real market data available for {symbol}. MT5 is not connected or reachable, and synthetic fallback is disabled.")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    df = df.sort_index().copy()
    df = df.loc[~df.index.duplicated(keep="last")]
    return df
