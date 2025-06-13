import argparse
import pandas as pd
import numpy as np
from features import add_technical_indicators

FEATURE_COLS = [
    'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
    'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
    'RSI_14', 'Stoch_%K', 'Stoch_%D', 'ATR_14',
    'CCI_20', 'OBV'
]


def vectorize(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    X = df[FEATURE_COLS].astype(np.float32).values
    y = df['Target'].astype(np.int8).values
    np.savez_compressed(output_path, X=X, y=y)
    print(f"Saved {len(df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Vectorize price data to npz")
    parser.add_argument('csv', help='CSV file with price data')
    parser.add_argument('--output', default='data.npz', help='Output npz file')
    args = parser.parse_args()
    vectorize(args.csv, args.output)


if __name__ == '__main__':
    main()
