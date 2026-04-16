import argparse
import pandas as pd
import yfinance as yf

OPTION_COLS = [
    "CallVolume", "PutVolume", "CallOI", "PutOI", "CallIV", "PutIV"
]


def get_nearest_expiration(ticker: str) -> str:
    """Return the nearest expiration date available for the ticker."""
    t = yf.Ticker(ticker)
    expirations = t.options
    if not expirations:
        raise ValueError(f"No option expirations found for {ticker}")
    return expirations[0]

def fetch_options_summary(ticker: str, expiration: str | None = None) -> pd.DataFrame:
    """Fetch option chain summary for a single expiration date."""
    if expiration is None:
        expiration = get_nearest_expiration(ticker)
    t = yf.Ticker(ticker)
    chain = t.option_chain(expiration)
    calls = chain.calls
    puts = chain.puts
    data = {
        "CallVolume": calls["volume"].sum(),
        "PutVolume": puts["volume"].sum(),
        "CallOI": calls["openInterest"].sum(),
        "PutOI": puts["openInterest"].sum(),
        "CallIV": calls["impliedVolatility"].mean(),
        "PutIV": puts["impliedVolatility"].mean(),
    }
    df = pd.DataFrame([data])
    df.index = [pd.to_datetime("today").normalize()]
    return df

def main():
    parser = argparse.ArgumentParser(description="Download option chain summary")
    parser.add_argument("ticker", help="Underlying ticker")
    parser.add_argument("--expiration", help="Expiration date YYYY-MM-DD (default nearest)")
    parser.add_argument("--output", default="options.csv", help="Output CSV file")
    args = parser.parse_args()

    df = fetch_options_summary(args.ticker, args.expiration)
    df.to_csv(args.output)
    print(f"Saved option summary to {args.output}")

if __name__ == "__main__":
    main()
