import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from features import add_technical_indicators


def backtest(csv_path: str) -> float:
    """Run a simple walk-forward backtest and return cumulative return."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    prices = df['Close']
    predictions = []
    test_indices = []

    for i in range(50, len(df) - 1):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+1]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        feature_cols = [
            'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
            'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
            'RSI_14',
        ]
        model.fit(train_df[feature_cols], train_df['Target'])
        pred = model.predict(test_df[feature_cols])[0]
        predictions.append(pred)
        test_indices.append(test_df.index[0])

    preds = pd.Series(predictions, index=test_indices)
    returns = prices.pct_change().shift(-1)
    strategy_returns = returns.loc[preds.index] * preds
    cumulative_return = (1 + strategy_returns).prod() - 1
    buy_hold_return = (prices.iloc[len(preds)] / prices.iloc[0]) - 1

    print(f"Strategy cumulative return: {cumulative_return:.2%}")
    print(f"Buy and hold return: {buy_hold_return:.2%}")
    return cumulative_return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest trading strategy")
    parser.add_argument('csv', help='CSV file with price data')
    args = parser.parse_args()

    backtest(args.csv)
