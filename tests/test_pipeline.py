"""Smoke tests for the trading pipeline."""
import numpy as np
import pandas as pd
import pytest

from pipeline.data import fetch
from pipeline.features import (
    FEATURE_COLS,
    LABEL_COL,
    _fetch_macro,
    build_features,
    build_features_for_symbol,
)
from pipeline.model import StockEnsemble


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ohlcv():
    return fetch("AAPL", start="2020-01-01", end="2022-12-31")


@pytest.fixture(scope="module")
def macro():
    return _fetch_macro(start="2020-01-01", end="2022-12-31")


@pytest.fixture(scope="module")
def features(ohlcv, macro):
    return build_features(ohlcv, macro=macro)


# ---------------------------------------------------------------------------
# Data tests
# ---------------------------------------------------------------------------

def test_fetch_returns_ohlcv(ohlcv):
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(ohlcv.columns)
    assert len(ohlcv) > 200


def test_fetch_cache_hit(tmp_path, monkeypatch):
    """Second call returns cached parquet without network hit."""
    import pipeline.data as data_mod
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path)
    df1 = data_mod.fetch("MSFT", start="2021-01-01", end="2021-12-31")
    # Re-fetch — if cache works, we don't need to mock yfinance
    df2 = data_mod.fetch("MSFT", start="2021-01-01", end="2021-12-31")
    pd.testing.assert_frame_equal(df1, df2)


def test_fetch_no_data_raises():
    with pytest.raises(ValueError, match="No data"):
        fetch("INVALID_TICKER_XYZ_NOPE", start="2020-01-01", end="2020-06-01")


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------

def test_feature_cols_present(features):
    assert set(FEATURE_COLS).issubset(features.columns)


def test_no_nan_in_features(features):
    assert features[FEATURE_COLS].isna().sum().sum() == 0


def test_target_is_binary(features):
    assert set(features[LABEL_COL].unique()).issubset({0, 1})


def test_macro_vix_present(features):
    assert "VIX" in features.columns
    assert features["VIX"].notna().all()


def test_macro_tnx_present(features):
    assert "TNX" in features.columns
    assert features["TNX"].notna().all()


def test_macro_ffill(ohlcv):
    """Macro forward-fill: no NaN after join even on sparse macro dates."""
    macro = _fetch_macro(start="2020-01-01", end="2022-12-31")
    feats = build_features(ohlcv, macro=macro)
    assert feats["VIX"].isna().sum() == 0
    assert feats["TNX"].isna().sum() == 0


def test_build_features_without_macro(ohlcv):
    """build_features still works when macro=None (columns will be NaN before dropna)."""
    feats = build_features(ohlcv, macro=None)
    # VIX/TNX rows become NaN → all dropped by dropna
    assert "VIX" in feats.columns


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

def test_ensemble_fit_predict(features):
    X = features[FEATURE_COLS]
    y = features[LABEL_COL]
    model = StockEnsemble()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_ensemble_save_load(tmp_path, features):
    X = features[FEATURE_COLS]
    y = features[LABEL_COL]
    model = StockEnsemble()
    model.fit(X, y)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = StockEnsemble.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


# ---------------------------------------------------------------------------
# Synthetic-data tests: SPY-relative features and label engineering
# ---------------------------------------------------------------------------

@pytest.fixture()
def _rising_ohlcv():
    """200 business days: stock has strong upward drift (+0.15%/day mean) with noise.

    Noise ensures RSI, Stoch, and other indicators see both up and down days,
    preventing division-by-zero or all-NaN edge cases in ratio computations.
    """
    n = 200
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=n)
    # Daily returns: strong upward drift + small noise so some days go down
    daily_ret = 0.0015 + rng.normal(0, 0.005, n)
    close = 100.0 * np.cumprod(1 + daily_ret)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame(
        {"Open": close * 0.999, "High": high,
         "Low": low, "Close": close,
         "Volume": np.full(n, 1_000_000, dtype=float)},
        index=dates,
    )


@pytest.fixture()
def _flat_macro(_rising_ohlcv):
    """Macro with SPY returning 0% at every horizon — market is flat."""
    idx = _rising_ohlcv.index
    return pd.DataFrame(
        {"VIX": 15.0, "TNX": 4.0,
         "SPY_Return_1d": 0.0, "SPY_Return_5d": 0.0, "SPY_Return_20d": 0.0},
        index=idx,
    )


def test_spy_feature_cols_in_feature_cols():
    """All five SPY/relative columns must appear in FEATURE_COLS."""
    expected = {"SPY_Return_1d", "SPY_Return_5d", "SPY_Return_20d",
                "Rel_Return_1d", "Rel_Return_5d"}
    assert expected.issubset(set(FEATURE_COLS))


def test_spy_relative_features_present(_rising_ohlcv, _flat_macro):
    """build_features populates all SPY-relative columns."""
    feats = build_features(_rising_ohlcv, macro=_flat_macro)
    for col in ("Rel_Return_1d", "Rel_Return_5d", "SPY_Return_1d",
                "SPY_Return_5d", "SPY_Return_20d"):
        assert col in feats.columns, f"{col} missing"


def test_rel_return_equals_stock_return_when_spy_flat(_rising_ohlcv, _flat_macro):
    """With SPY return = 0, Rel_Return equals the raw stock return."""
    feats = build_features(_rising_ohlcv, macro=_flat_macro)
    np.testing.assert_allclose(
        feats["Rel_Return_1d"].values, feats["Return_1d"].values, rtol=1e-9,
        err_msg="Rel_Return_1d != Return_1d when SPY is flat",
    )
    np.testing.assert_allclose(
        feats["Rel_Return_5d"].values, feats["Return_5d"].values, rtol=1e-9,
        err_msg="Rel_Return_5d != Return_5d when SPY is flat",
    )


def test_label_mostly_one_when_stock_outperforms(_rising_ohlcv, _flat_macro):
    """Stock +0.15%/day mean drift with 0% SPY → 5-day alpha >> 0.3% → label mostly 1."""
    feats = build_features(_rising_ohlcv, macro=_flat_macro)
    assert len(feats) > 0, "build_features returned empty DataFrame on synthetic data"
    label_mean = feats[LABEL_COL].mean()
    assert label_mean > 0.5, f"Expected mostly 1s given strong upward drift, got mean={label_mean:.2f}"


def test_label_mostly_zero_when_stock_underperforms():
    """Stock drifts down vs SPY +1% per 5d → underperforms → label mostly 0."""
    n = 200
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2022-01-03", periods=n)
    # Stock drifts slightly DOWN while SPY goes up → persistent underperformance
    daily_ret = -0.001 + rng.normal(0, 0.005, n)
    close = 100.0 * np.cumprod(1 + daily_ret)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    ohlcv = pd.DataFrame(
        {"Open": close * 0.999, "High": high, "Low": low,
         "Close": close, "Volume": np.full(n, 1_000_000, dtype=float)},
        index=dates,
    )
    macro = pd.DataFrame(
        {"VIX": 15.0, "TNX": 4.0,
         "SPY_Return_1d": 0.002, "SPY_Return_5d": 0.01, "SPY_Return_20d": 0.04},
        index=dates,
    )
    feats = build_features(ohlcv, macro=macro)
    assert len(feats) > 0, "build_features returned empty DataFrame"
    label_mean = feats[LABEL_COL].mean()
    assert label_mean < 0.4, f"Expected mostly 0s, got mean={label_mean:.2f}"


def test_synthetic_model_fit_predict(_rising_ohlcv, _flat_macro):
    """StockEnsemble trains and predicts on synthetic OHLCV+macro without error."""
    feats = build_features(_rising_ohlcv, macro=_flat_macro)
    X = feats[FEATURE_COLS]
    y = feats[LABEL_COL]
    model = StockEnsemble()
    model.fit(X, y)
    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert np.all(probas >= 0) and np.all(probas <= 1)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)
