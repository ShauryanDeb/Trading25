"""Expanding walk-forward backtester."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from pipeline.features import FEATURE_COLS, LABEL_COL
from pipeline.model import StockEnsemble


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    accuracy: float
    sharpe: float
    max_drawdown: float
    total_return: float
    n_trades: int
    equity_curve: pd.Series = field(repr=False)


def _sharpe(returns: pd.Series, periods: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def walk_forward(
    df: pd.DataFrame,
    n_folds: int = 5,
    min_train_years: int = 2,
) -> List[FoldResult]:
    """Expanding walk-forward on a feature DataFrame.

    Args:
        df: Output of ``build_features_for_symbol`` — must have FEATURE_COLS + Target.
        n_folds: Number of test folds.
        min_train_years: Minimum trading days in first training window (approx).

    Returns:
        List of FoldResult, one per fold.
    """
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).sort_index()
    n = len(df)
    min_train = min_train_years * 252
    test_size = max(1, (n - min_train) // n_folds)

    results = []
    for fold in range(n_folds):
        train_end_idx = min_train + fold * test_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_size, n)

        if test_start_idx >= n:
            break

        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[test_start_idx:test_end_idx]

        model = StockEnsemble()
        model.fit(train_df[FEATURE_COLS], train_df[LABEL_COL])

        preds = model.predict(test_df[FEATURE_COLS])
        accuracy = float((preds == test_df[LABEL_COL].values).mean())

        # Strategy: long when model predicts 1, flat otherwise
        # Approximate 1-day forward return for the test period
        fwd_returns = test_df.index.to_series().map(
            lambda d: np.nan
        )
        # We need raw close prices — stored in LABEL_COL as direction only,
        # so we approximate using Return_1d shifted by -1.
        strat_returns = test_df["Return_1d"].shift(-1).fillna(0) * preds
        strat_returns = strat_returns.dropna()

        equity = (1 + strat_returns).cumprod()
        sharpe = _sharpe(strat_returns)
        mdd = _max_drawdown(equity)
        total_ret = float(equity.iloc[-1] - 1) if len(equity) else 0.0
        n_trades = int(preds.sum())

        results.append(
            FoldResult(
                fold=fold + 1,
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()),
                accuracy=accuracy,
                sharpe=sharpe,
                max_drawdown=mdd,
                total_return=total_ret,
                n_trades=n_trades,
                equity_curve=equity,
            )
        )

    return results


def summary(results: List[FoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Fold": r.fold,
                "Train": f"{r.train_start} → {r.train_end}",
                "Test": f"{r.test_start} → {r.test_end}",
                "Accuracy": f"{r.accuracy:.2%}",
                "Sharpe": f"{r.sharpe:.2f}",
                "MaxDD": f"{r.max_drawdown:.2%}",
                "TotalReturn": f"{r.total_return:.2%}",
                "Trades": r.n_trades,
            }
        )
    return pd.DataFrame(rows)
