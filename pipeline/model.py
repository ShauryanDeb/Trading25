"""XGBoost + RandomForest soft-voting ensemble."""
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

from pipeline.features import FEATURE_COLS


class StockEnsemble:
    """Soft-voting ensemble of XGBoost + RandomForest."""

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    def _build(self) -> Pipeline:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )
        voter = VotingClassifier(
            estimators=[("xgb", xgb), ("rf", rf)],
            voting="soft",
        )
        return Pipeline([("scaler", StandardScaler()), ("model", voter)])

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockEnsemble":
        self._pipeline = self._build()
        self._pipeline.fit(X[FEATURE_COLS].fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(X[FEATURE_COLS].fillna(0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(X[FEATURE_COLS].fillna(0))

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "StockEnsemble":
        obj = cls()
        obj._pipeline = joblib.load(path)
        return obj
