"""5-minute bar features for intraday ML model."""
from __future__ import annotations

import numpy as np
import pandas as pd

INTRADAY_FEATURE_COLS = [
    # Momentum / returns
    "Return_1bar",
    "Return_3bar",
    "Return_6bar",
    # RSI on 5-min
    "RSI_14",
    # MACD on 5-min
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    # VWAP deviation
    "VWAP_Dev",
    # Bollinger band position
    "BB_Position",
    "BB_Width",
    # Volume
    "Volume_Ratio",
    "Volume_Spike",
    # Volatility
    "ATR_14",
    "Volatility_10bar",
    # Trend
    "EMA_9",
    "EMA_21",
    "Price_to_EMA9",
    "EMA_Cross",
    # Rate of change
    "ROC_3",
    "ROC_6",
]

INTRADAY_LABEL_COL = "Target"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday features from a 5-min OHLCV(+VWAP) DataFrame.

    Target: 1 if close of next bar > current close, else 0.
    """
    out = df.copy()

    # Returns
    out["Return_1bar"] = out["Close"].pct_change(1)
    out["Return_3bar"] = out["Close"].pct_change(3)
    out["Return_6bar"] = out["Close"].pct_change(6)

    # RSI
    out["RSI_14"] = _rsi(out["Close"])

    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    # VWAP deviation (uses Vwap column if available, else rolling proxy)
    if "Vwap" in out.columns:
        out["VWAP_Dev"] = (out["Close"] - out["Vwap"]) / out["Vwap"]
    else:
        typical = (out["High"] + out["Low"] + out["Close"]) / 3
        cum_vol = out["Volume"].cumsum()
        cum_tpv = (typical * out["Volume"]).cumsum()
        vwap = cum_tpv / cum_vol.replace(0, np.nan)
        out["VWAP_Dev"] = (out["Close"] - vwap) / vwap

    # Bollinger bands
    mid = out["Close"].rolling(20).mean()
    std = out["Close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    out["BB_Position"] = (out["Close"] - lower) / (upper - lower).replace(0, np.nan)
    out["BB_Width"] = (2 * std) / mid.replace(0, np.nan)

    # Volume
    vol_ma = out["Volume"].rolling(20).mean()
    out["Volume_Ratio"] = out["Volume"] / vol_ma.replace(0, np.nan)
    out["Volume_Spike"] = (out["Volume"] > vol_ma * 2).astype(float)

    # ATR & volatility
    out["ATR_14"] = _atr(out)
    out["Volatility_10bar"] = out["Return_1bar"].rolling(10).std()

    # EMAs & trend
    out["EMA_9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["Price_to_EMA9"] = out["Close"] / out["EMA_9"]
    out["EMA_Cross"] = (out["EMA_9"] > out["EMA_21"]).astype(float)

    # Rate of change
    out["ROC_3"] = out["Close"].pct_change(3)
    out["ROC_6"] = out["Close"].pct_change(6)

    # Target: 1 if next bar close > current close
    out[INTRADAY_LABEL_COL] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    out = out[INTRADAY_FEATURE_COLS + [INTRADAY_LABEL_COL]].dropna()
    return out


def build_intraday_features_for_symbol(
    symbol: str,
    days_back: int = 90,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch 5-min bars and return feature matrix."""
    from pipeline.intraday_data import fetch_5min
    df = fetch_5min(symbol, days_back=days_back, force_refresh=force_refresh)
    return build_intraday_features(df)
