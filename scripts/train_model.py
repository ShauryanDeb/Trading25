import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from features import add_technical_indicators, add_option_features, OPTION_FEATURES

FEATURE_COLS = [
    'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
    'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
    'RSI_14', 'Stoch_%K', 'Stoch_%D', 'ATR_14',
    'CCI_20', 'OBV',
] + OPTION_FEATURES


def prepare_dataset(csv_path: str, options_path: str | None = None) -> pd.DataFrame:
    """Load price data from csv, optional options data, and add features and target."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    if options_path:
        opt_df = pd.read_csv(options_path, index_col=0, parse_dates=True)
        df = add_option_features(df, opt_df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


def load_npz_dataset(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data['X'], data['y']


def train(df: pd.DataFrame) -> RandomForestClassifier:
    X = df[[c for c in FEATURE_COLS if c in df.columns]]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model


def train_arrays(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train trading model")
    parser.add_argument('data', help='CSV or NPZ dataset')
    parser.add_argument('--options', help='CSV file with option features')
    parser.add_argument('--model-out', default='model.pkl', help='File to save trained model')
    args = parser.parse_args()

    if args.data.lower().endswith('.npz'):
        X, y = load_npz_dataset(args.data)
        model = train_arrays(X, y)
    else:
        df = prepare_dataset(args.data, args.options)
        model = train(df)

    joblib.dump(model, args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
