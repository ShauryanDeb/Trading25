"""Trade log analyzer.

Reads all reports/trades_YYYYMMDD.csv files, fetches next-day closes from
yfinance to compute actual win rate, and prints a summary of key metrics.

Usage:
    python reports/analyze.py              # analyze all trade logs
    python reports/analyze.py --since 2026-06-01   # filter by date
    python reports/analyze.py --symbol AAPL        # filter by symbol
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPORTS_DIR = _ROOT / "reports"


# ---------------------------------------------------------------------------
# Load & enrich
# ---------------------------------------------------------------------------

def load_trades(since: str | None = None, symbol: str | None = None) -> pd.DataFrame:
    csvs = sorted(REPORTS_DIR.glob("trades_*.csv"))
    if not csvs:
        print("No trade logs found in reports/")
        sys.exit(0)

    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"])
            frames.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    trades = pd.concat(frames, ignore_index=True)
    trades["date"] = trades["timestamp"].dt.date.astype(str)

    if since:
        trades = trades[trades["date"] >= since]
    if symbol:
        trades = trades[trades["symbol"] == symbol.upper()]

    # Exclude dry-run entries from live PnL calc but keep for signal analysis
    trades["is_live"] = ~trades["side"].str.contains("dry_run", na=False)
    return trades


def fetch_next_day_returns(trades: pd.DataFrame) -> pd.DataFrame:
    """Add next_day_close and win columns by fetching prices from yfinance."""
    import yfinance as yf

    symbols = trades["symbol"].unique().tolist()
    print(f"Fetching next-day prices for {len(symbols)} symbols...")

    price_cache: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period="6mo")["Close"]
            hist.index = hist.index.tz_localize(None).normalize()
            price_cache[sym] = hist
        except Exception:
            pass

    results = []
    for _, row in trades.iterrows():
        sym = row["symbol"]
        entry_date = pd.Timestamp(row["date"])
        entry_price = float(row["price"])
        series = price_cache.get(sym)

        next_close = np.nan
        if series is not None:
            future = series[series.index > entry_date]
            if not future.empty:
                next_close = float(future.iloc[0])

        pnl_pct = (next_close - entry_price) / entry_price if not np.isnan(next_close) else np.nan
        results.append({**row.to_dict(), "next_close": next_close, "pnl_pct": pnl_pct})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    buys = df[df["side"].isin(["buy", "buy_dry_run"])].copy()
    live_buys = buys[buys["is_live"]].copy()

    print("\n" + "=" * 60)
    print("  TRADE LOG ANALYSIS")
    print("=" * 60)

    # --- Overview ---
    print(f"\n{'Total signals logged:':<35} {len(buys)}")
    print(f"{'Live trades (non-dry-run):':<35} {len(live_buys)}")
    if not buys.empty:
        print(f"{'Date range:':<35} {buys['date'].min()}  to  {buys['date'].max()}")
        print(f"{'Unique symbols traded:':<35} {buys['symbol'].nunique()}")

    # --- Win rate (requires next_close) ---
    if "pnl_pct" in buys.columns:
        scored = buys.dropna(subset=["pnl_pct"])
        if not scored.empty:
            win_rate = (scored["pnl_pct"] > 0).mean()
            avg_win = scored.loc[scored["pnl_pct"] > 0, "pnl_pct"].mean()
            avg_loss = scored.loc[scored["pnl_pct"] < 0, "pnl_pct"].mean()
            print(f"\n{'--- Signal Quality ---':}")
            print(f"{'Scored trades:':<35} {len(scored)}")
            print(f"{'Win rate:':<35} {win_rate:.1%}  (want > 52%)")
            print(f"{'Avg win:':<35} {avg_win:.2%}" if not np.isnan(avg_win) else f"{'Avg win:':<35} n/a")
            print(f"{'Avg loss:':<35} {avg_loss:.2%}" if not np.isnan(avg_loss) else f"{'Avg loss:':<35} n/a")
            if not np.isnan(avg_win) and not np.isnan(avg_loss) and avg_loss != 0:
                profit_factor = abs(avg_win / avg_loss) * win_rate / (1 - win_rate)
                print(f"{'Profit factor:':<35} {profit_factor:.2f}  (want > 1.0)")

    # --- Signal strength ---
    print(f"\n{'--- Signal Strength ---':}")
    print(f"{'Avg proba_up (buy signals):':<35} {buys['signal_proba'].mean():.3f}")
    print(f"{'Signals > 0.60:':<35} {(buys['signal_proba'] > 0.60).sum()}")
    print(f"{'Signals > 0.65:':<35} {(buys['signal_proba'] > 0.65).sum()}")

    # --- Per-symbol breakdown ---
    print(f"\n{'--- Per-Symbol Summary ---':}")
    by_sym = buys.groupby("symbol").agg(
        signals=("signal_proba", "count"),
        avg_proba=("signal_proba", "mean"),
    )
    if "pnl_pct" in buys.columns:
        scored_sym = buys.dropna(subset=["pnl_pct"]).groupby("symbol").agg(
            win_rate=("pnl_pct", lambda x: (x > 0).mean()),
            avg_pnl=("pnl_pct", "mean"),
        )
        by_sym = by_sym.join(scored_sym, how="left")

    by_sym = by_sym.sort_values("avg_proba", ascending=False)
    print(by_sym.to_string(float_format="{:.3f}".format))

    # --- Turnover ---
    print(f"\n{'--- Turnover ---':}")
    by_date = buys.groupby("date")["symbol"].count()
    print(f"{'Avg signals per day:':<35} {by_date.mean():.1f}")
    print(f"{'Max signals in one day:':<35} {by_date.max()}")
    print(f"{'Days with any signal:':<35} {len(by_date)}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trade logs")
    parser.add_argument("--since", default=None, help="Only include trades on or after YYYY-MM-DD")
    parser.add_argument("--symbol", default=None, help="Filter to a single symbol")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip yfinance fetch (no win rate, faster)")
    args = parser.parse_args()

    trades = load_trades(since=args.since, symbol=args.symbol)
    print(f"Loaded {len(trades)} trade rows from {len(list(REPORTS_DIR.glob('trades_*.csv')))} file(s)")

    if not args.no_fetch:
        trades = fetch_next_day_returns(trades)

    print_summary(trades)


if __name__ == "__main__":
    main()
