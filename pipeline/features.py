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
    # SPY-relative (market context + alpha signal)
    "SPY_Return_1d",
    "SPY_Return_5d",
    "SPY_Return_20d",
    "Rel_Return_1d",
    "Rel_Return_5d",
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
    """Return daily VIX, 10-yr yield (TNX), and SPY returns, forward-filled."""
    vix = fetch("^VIX", start=start, end=end)[["Close"]].rename(columns={"Close": "VIX"})
    tnx = fetch("^TNX", start=start, end=end)[["Close"]].rename(columns={"Close": "TNX"})
    spy = fetch("SPY", start=start, end=end)[["Close"]]
    spy["SPY_Return_1d"] = spy["Close"].pct_change(1, fill_method=None)
    spy["SPY_Return_5d"] = spy["Close"].pct_change(5, fill_method=None)
    spy["SPY_Return_20d"] = spy["Close"].pct_change(20, fill_method=None)
    spy = spy[["SPY_Return_1d", "SPY_Return_5d", "SPY_Return_20d"]]
    macro = vix.join(tnx, how="outer").join(spy, how="outer").ffill()
    return macro


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    target_mode: str = "rel_5d",
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """Compute all features from an OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        macro: Optional pre-fetched macro DataFrame (VIX, TNX columns).
               If None, macro columns will be NaN.
        target_mode: Label definition —
            "rel_5d": 1 if stock beats SPY by >0.3% over the next 5 days.
                      Default: walk-forward 2021-2026 validated this label
                      (+35.6% OOS, PF 1.18) over the abs_3d alternative.
            "abs_3d": 1 if absolute return over the next 3 trading days
                      exceeds +0.5%. Tested 2026-07: LOST the walk-forward
                      head-to-head (-14.1% OOS) — predicting absolute
                      direction is mostly predicting market beta, which
                      this feature set cannot do. Kept for experiments.
        drop_unlabeled: If True (training), drop rows whose forward-return
            window extends past the data end — they have no label. If False
            (inference), keep them with Target=NaN so the latest row is
            genuinely the latest bar, not label-horizon days stale.

    Returns:
        DataFrame with FEATURE_COLS + Target column; rows with NaN features
        always dropped.
    """
    out = df.copy()

    # Returns
    out["Return_1d"] = out["Close"].pct_change(1, fill_method=None)
    out["Return_5d"] = out["Close"].pct_change(5, fill_method=None)
    out["Return_20d"] = out["Close"].pct_change(20, fill_method=None)

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

    # Macro + SPY context
    macro_cols = ["VIX", "TNX", "SPY_Return_1d", "SPY_Return_5d", "SPY_Return_20d"]
    if macro is not None:
        available = [c for c in macro_cols if c in macro.columns]
        out = out.join(macro[available], how="left")
        out[available] = out[available].ffill()
    else:
        for c in macro_cols:
            out[c] = np.nan

    # SPY-relative momentum features (capture alpha vs market)
    out["Rel_Return_1d"] = out["Return_1d"] - out["SPY_Return_1d"]
    out["Rel_Return_5d"] = out["Return_5d"] - out["SPY_Return_5d"]

    if target_mode == "rel_5d":
        stock_5d = out["Close"].shift(-5) / out["Close"] - 1
        spy_5d = out["SPY_Return_5d"].shift(-5)
        out[LABEL_COL] = ((stock_5d - spy_5d) > 0.003).astype(float)
        # Rows whose forward window extends past the data end have no label
        out.loc[out["Close"].shift(-5).isna(), LABEL_COL] = np.nan
    elif target_mode == "abs_3d":
        stock_3d = out["Close"].shift(-3) / out["Close"] - 1
        out[LABEL_COL] = (stock_3d > 0.005).astype(float)
        out.loc[out["Close"].shift(-3).isna(), LABEL_COL] = np.nan
    else:
        raise ValueError(f"Unknown target_mode: {target_mode!r}")

    out = out[FEATURE_COLS + [LABEL_COL]].dropna(subset=FEATURE_COLS)
    if drop_unlabeled:
        out = out.dropna(subset=[LABEL_COL])
        out[LABEL_COL] = out[LABEL_COL].astype(int)
    return out


def build_features_for_symbol(
    symbol: str,
    start: str = "2015-01-01",
    end: str | None = None,
    include_macro: bool = True,
    target_mode: str = "rel_5d",
    drop_unlabeled: bool = True,
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

    return build_features(ohlcv, macro=macro, target_mode=target_mode,
                          drop_unlabeled=drop_unlabeled)
