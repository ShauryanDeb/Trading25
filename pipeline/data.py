"""Fetch OHLCV data from yfinance with parquet cache."""
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _cache_path(symbol: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{start}_{end}.parquet"


def fetch(
    symbol: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV for *symbol*, reading from parquet cache when available.

    Returns a DataFrame with columns Open/High/Low/Close/Volume, DatetimeIndex (tz-naive).
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = _cache_path(symbol, start, end)

    if cache.exists() and not force_refresh:
        df = pd.read_parquet(cache)
        df.index = df.index.astype("datetime64[ns]")
        return df

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.index = df.index.astype("datetime64[ns]")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    df.to_parquet(cache)
    return df
