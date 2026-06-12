"""Intraday XGBoost + RandomForest ensemble for 5-min bar signals."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline.intraday_features import INTRADAY_FEATURE_COLS


class IntradayEnsemble:
    """Soft-voting ensemble tuned for intraday 5-min bar classification."""

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None

    def _build(self) -> Pipeline:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        )
        voter = VotingClassifier(
            estimators=[("xgb", xgb), ("rf", rf)],
            voting="soft",
        )
        return Pipeline([("scaler", StandardScaler()), ("model", voter)])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "IntradayEnsemble":
        self._pipeline = self._build()
        self._pipeline.fit(X[INTRADAY_FEATURE_COLS].fillna(0), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(X[INTRADAY_FEATURE_COLS].fillna(0))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(X[INTRADAY_FEATURE_COLS].fillna(0))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "IntradayEnsemble":
        obj = cls()
        obj._pipeline = joblib.load(path)
        return obj
