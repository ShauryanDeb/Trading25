"""Fetch 5-minute OHLCV bars from Alpaca historical data API with parquet cache."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(__file__).parent.parent / "data" / "intraday_cache"


def _get_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        api_key=os.environ.get("APCA_API_KEY_ID_INTRADAY"),
        secret_key=os.environ.get("APCA_API_SECRET_KEY_INTRADAY"),
    )


def fetch_5min(
    symbol: str,
    days_back: int = 55,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return 5-minute OHLCV bars for *symbol* via yfinance (up to 60 days).

    Returns DataFrame with columns Open/High/Low/Close/Volume, DatetimeIndex (UTC).
    """
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_5min_{days_back}d.parquet"

    # Use cache if less than 4 hours old
    if cache_path.exists() and not force_refresh:
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if age < 14400:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index, utc=True)
            return df

    days_back = min(days_back, 55)  # yfinance 5-min limit is 60 days
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days_back}d", interval="5m", auto_adjust=True)

    if df.empty:
        raise ValueError(f"No 5-min data returned for {symbol}")

    df.index = df.index.tz_convert("UTC")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "timestamp"

    df.to_parquet(cache_path)
    return df


def fetch_latest_bars(symbol: str, n: int = 50) -> pd.DataFrame:
    """Fetch the most recent *n* 5-minute bars for live signal generation via yfinance."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1d", interval="5m", auto_adjust=True)

    if df.empty:
        return pd.DataFrame()

    df.index = df.index.tz_convert("UTC")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    return df.iloc[-n:]
