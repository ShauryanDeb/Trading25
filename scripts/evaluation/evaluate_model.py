import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from features import add_technical_indicators, add_option_features, OPTION_FEATURES

FEATURE_COLS = [
    'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
    'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
    'RSI_14', 'Stoch_%K', 'Stoch_%D', 'ATR_14',
    'CCI_20', 'OBV',
] + OPTION_FEATURES

def load_dataset(csv_path: str, options_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    if options_path:
        opt_df = pd.read_csv(options_path, index_col=0, parse_dates=True)
        df = add_option_features(df, opt_df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


def evaluate(model_path: str, df: pd.DataFrame) -> float:
    model = joblib.load(model_path)
    X = df[FEATURE_COLS]
    y = df['Target']
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(classification_report(y, preds))
    print(f"Accuracy: {acc:.2%}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on historical data")
    parser.add_argument('model', help='Path to saved model')
    parser.add_argument('csv', help='CSV file with historical price data')
    parser.add_argument('--options', help='CSV with option features')
    args = parser.parse_args()

    df = load_dataset(args.csv, args.options)
    evaluate(args.model, df)


if __name__ == '__main__':
    main()
