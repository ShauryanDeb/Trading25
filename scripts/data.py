import yfinance as yf
import pandas as pd


def download_data(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Download historical OHLCV data for a given ticker."""
    if start is None:
        data = yf.download(ticker, period="max")
    else:
        data = yf.download(ticker, start=start, end=end)
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download historical data")
    parser.add_argument("ticker", help="Ticker symbol to download")
    parser.add_argument("--start", default="1980-01-01", help="Start date YYYY-MM-DD, use none for full history")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="data.csv", help="File to save data")
    args = parser.parse_args()

    start = None if args.start and args.start.lower() == 'none' else args.start
    end = None if args.end and args.end.lower() == 'none' else args.end
    df = download_data(args.ticker, start, end)
    df.to_csv(args.output)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
