import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib
from features import add_technical_indicators

FEATURE_COLS = [
    'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
    'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
    'RSI_14', 'Stoch_%K', 'Stoch_%D', 'ATR_14',
    'CCI_20', 'OBV'
]

def load_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    X = df[FEATURE_COLS]
    y = df['Target']
    return X.values, y.values

def load_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data['X'], data['y']

def tune(dataset: str, save_path: str | None = None) -> None:
    if dataset.lower().endswith('.npz'):
        X, y = load_npz(dataset)
    else:
        X, y = load_csv(dataset)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    model = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(model, param_grid, cv=tscv, n_jobs=-1)
    gs.fit(X, y)
    print(f"Best params: {gs.best_params_}")
    print(f"Best score: {gs.best_score_:.4f}")

    if save_path:
        joblib.dump(gs.best_estimator_, save_path)
        print(f"Saved best model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Tune RandomForest hyperparameters")
    parser.add_argument('dataset', help='CSV or NPZ dataset')
    parser.add_argument('--model-out', help='Optional path to save best model')
    args = parser.parse_args()
    tune(args.dataset, args.model_out)


if __name__ == '__main__':
    main()
