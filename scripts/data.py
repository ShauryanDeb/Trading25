import yfinance as yf
import pandas as pd


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data for a given ticker."""
    data = yf.download(ticker, start=start, end=end)
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download historical data")
    parser.add_argument("ticker", help="Ticker symbol to download")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="data.csv", help="File to save data")
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df.to_csv(args.output)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
