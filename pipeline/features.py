"""OHLCV → technical indicators + macro features → feature matrix."""
import numpy as np
import pandas as pd
from pipeline.data import fetch

# ---------------------------------------------------------------------------
# Feature column list (used by model.py and tests)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Price-derived
    "Return_1d",
    "Return_5d",
    "Return_20d",
    # Moving averages
    "MA_20",
    "MA_50",
    "EMA_20",
    "EMA_50",
    "Price_to_MA20",
    # Volatility
    "ATR_14",
    "BB_Width",
    "Volatility_20d",
    # Momentum
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Stoch_K",
    "Stoch_D",
    "CCI_20",
    # Volume
    "Volume_Ratio",
    "OBV_Change",
    # Macro
    "VIX",
    "TNX",
]

LABEL_COL = "Target"


# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------

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


def _obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff().fillna(0))
    return (sign * df["Volume"]).cumsum()


# ---------------------------------------------------------------------------
# Macro fetch
# ---------------------------------------------------------------------------

def _fetch_macro(start: str, end: str) -> pd.DataFrame:
    """Return daily VIX and 10-yr yield (TNX), forward-filled."""
    vix = fetch("^VIX", start=start, end=end)[["Close"]].rename(columns={"Close": "VIX"})
    tnx = fetch("^TNX", start=start, end=end)[["Close"]].rename(columns={"Close": "TNX"})
    macro = vix.join(tnx, how="outer").ffill()
    return macro


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all features from an OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        macro: Optional pre-fetched macro DataFrame (VIX, TNX columns).
               If None, macro columns will be NaN.

    Returns:
        DataFrame with FEATURE_COLS + Target column, NaN rows dropped.
    """
    out = df.copy()

    # Returns
    out["Return_1d"] = out["Close"].pct_change(1)
    out["Return_5d"] = out["Close"].pct_change(5)
    out["Return_20d"] = out["Close"].pct_change(20)

    # Moving averages
    out["MA_20"] = out["Close"].rolling(20).mean()
    out["MA_50"] = out["Close"].rolling(50).mean()
    out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["Price_to_MA20"] = out["Close"] / out["MA_20"]

    # Volatility
    out["ATR_14"] = _atr(out)
    mid = out["Close"].rolling(20).mean()
    std = out["Close"].rolling(20).std()
    out["BB_Width"] = (2 * std) / mid
    out["Volatility_20d"] = out["Return_1d"].rolling(20).std()

    # Momentum
    out["RSI_14"] = _rsi(out["Close"])
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    low_min = out["Low"].rolling(14).min()
    high_max = out["High"].rolling(14).max()
    out["Stoch_K"] = (out["Close"] - low_min) / (high_max - low_min) * 100
    out["Stoch_D"] = out["Stoch_K"].rolling(3).mean()

    tp = (out["High"] + out["Low"] + out["Close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out["CCI_20"] = (tp - sma_tp) / (0.015 * mad)

    # Volume
    vol_ma = out["Volume"].rolling(20).mean()
    out["Volume_Ratio"] = out["Volume"] / vol_ma.replace(0, np.nan)
    obv = _obv(out)
    out["OBV_Change"] = obv.pct_change(5)

    # Macro
    if macro is not None:
        out = out.join(macro[["VIX", "TNX"]], how="left")
        out[["VIX", "TNX"]] = out[["VIX", "TNX"]].ffill()
    else:
        out["VIX"] = np.nan
        out["TNX"] = np.nan

    # Target: 1 if next-day return > 0 else 0
    out[LABEL_COL] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    out = out[FEATURE_COLS + [LABEL_COL]].dropna()
    return out


def build_features_for_symbol(
    symbol: str,
    start: str = "2015-01-01",
    end: str | None = None,
    include_macro: bool = True,
) -> pd.DataFrame:
    """Convenience: fetch OHLCV + macro and return feature matrix."""
    from pipeline.data import fetch as _fetch

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    ohlcv = _fetch(symbol, start=start, end=end)

    macro = None
    if include_macro:
        try:
            macro = _fetch_macro(start=start, end=end)
        except Exception:
            pass  # macro optional; columns will be NaN

    return build_features(ohlcv, macro=macro)
