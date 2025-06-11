import argparse
import time
import joblib
import pandas as pd
import yfinance as yf
from features import add_technical_indicators

FEATURE_COLS = [
    'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
    'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
    'RSI_14', 'Stoch_%K', 'Stoch_%D', 'ATR_14',
    'CCI_20', 'OBV'
]

def fetch_recent_data(ticker: str, period: str = "60d", interval: str = "1m") -> pd.DataFrame:
    """Fetch recent intraday data for the given ticker."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df

def run(model_path: str, ticker: str, poll_interval: int) -> None:
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}. Polling {ticker} every {poll_interval}s...")
    while True:
        df = fetch_recent_data(ticker)
        df = add_technical_indicators(df)
        df.dropna(inplace=True)
        latest = df.iloc[-1:]
        features = latest[FEATURE_COLS]
        pred = model.predict(features)[0]
        direction = "UP" if pred == 1 else "DOWN"
        timestamp = latest.index[-1]
        print(f"{timestamp} prediction: {direction}")
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Run real-time predictions")
    parser.add_argument("model", help="Trained model file")
    parser.add_argument("ticker", help="Ticker symbol to monitor")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    args = parser.parse_args()
    run(args.model, args.ticker, args.interval)


if __name__ == "__main__":
    main()
