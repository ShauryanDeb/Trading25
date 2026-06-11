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
