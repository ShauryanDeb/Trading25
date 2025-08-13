import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add several common technical indicators to the dataframe."""
    df = df.copy()

    # Simple moving averages
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # Exponential moving averages
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Bollinger Bands
    mid = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = mid + 2 * std
    df["BB_Lower"] = mid - 2 * std

    # MACD and signal line
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["RSI_14"] = compute_rsi(df["Close"], window=14)

    # Stochastic Oscillator
    stoch_k, stoch_d = compute_stochastic(df, window=14)
    df["Stoch_%K"] = stoch_k
    df["Stoch_%D"] = stoch_d

    # Average True Range
    df["ATR_14"] = compute_atr(df, window=14)

    # Commodity Channel Index
    df["CCI_20"] = compute_cci(df, window=20)

    # On-Balance Volume
    df["OBV"] = compute_obv(df)

    return df


OPTION_FEATURES = [
    "CallVolume", "PutVolume", "CallOI", "PutOI", "CallIV", "PutIV"
]


def add_option_features(df: pd.DataFrame, opt_df: pd.DataFrame) -> pd.DataFrame:
    """Merge option-based features into the price dataframe."""
    opt_df = opt_df.copy()
    opt_df.index = pd.to_datetime(opt_df.index)
    df = df.join(opt_df[OPTION_FEATURES], how="left")
    df[OPTION_FEATURES] = df[OPTION_FEATURES].fillna(method="ffill")
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_stochastic(df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series]:
    """Return %K and %D stochastic oscillator series."""
    low_min = df["Low"].rolling(window=window).min()
    high_max = df["High"].rolling(window=window).max()
    percent_k = (df["Close"] - low_min) / (high_max - low_min) * 100
    percent_d = percent_k.rolling(window=3).mean()
    return percent_k, percent_d


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def compute_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci


def compute_obv(df: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

