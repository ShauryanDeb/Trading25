import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from features import add_technical_indicators


def prepare_dataset(csv_path: str) -> pd.DataFrame:
    """Load price data from csv and add features and target."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


def train(df: pd.DataFrame):
    feature_cols = [
        'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
        'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
        'RSI_14', 'Stoch_%K', 'Stoch_%D', 'ATR_14',
        'CCI_20', 'OBV'
    ]
    X = df[feature_cols]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train trading model")
    parser.add_argument('csv', help='CSV file with price data')
    args = parser.parse_args()

    df = prepare_dataset(args.csv)
    train(df)


if __name__ == "__main__":
    main()
