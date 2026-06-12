"""CLI: fetch / train / backtest commands."""
import argparse
import sys
from pathlib import Path


def cmd_fetch(args):
    from pipeline.data import fetch
    df = fetch(args.symbol, start=args.start, end=args.end, force_refresh=args.refresh)
    print(f"Fetched {len(df)} rows for {args.symbol}  ({df.index[0].date()} to {df.index[-1].date()})")


UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "SNPS",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA",
    "COF", "USB", "PNC", "TFC", "CME", "ICE", "CB", "PGR", "MET", "AIG",
    # Healthcare
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "ISRG", "VRTX", "REGN", "ZTS", "SYK", "BSX", "MDT", "ELV", "CVS",
    # Consumer
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW", "TJX", "BKNG",
    "MAR", "HLT", "YUM", "DPZ", "CMG", "ORLY", "AZO", "TSCO", "DG", "DLTR",
    # Industrials & Energy
    "CAT", "DE", "HON", "GE", "RTX", "LMT", "BA", "UPS", "FDX", "NSC",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "HES",
    # Other large-caps
    "PEP", "KO", "PG", "MO", "PM", "DIS", "NFLX", "T", "VZ", "CMCSA",
]


def cmd_train(args):
    import pandas as pd
    from pipeline.features import build_features_for_symbol
    from pipeline.model import StockEnsemble

    symbols = UNIVERSE if args.universe else [args.symbol]
    frames = []
    for sym in symbols:
        print(f"Building features for {sym}...")
        try:
            feats = build_features_for_symbol(sym, start=args.start)
            frames.append(feats)
            print(f"  {len(feats)} samples")
        except Exception as e:
            print(f"  SKIP {sym}: {e}")

    if not frames:
        print("No data — aborting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    X = combined.drop(columns=["Target"])
    y = combined["Target"]
    print(f"\nTotal: {len(combined)} samples from {len(frames)} symbols, {X.shape[1]} features")

    model = StockEnsemble()
    model.fit(X, y)
    out = Path(args.output)
    model.save(out)
    print(f"Model saved -> {out}")


def cmd_backtest(args):
    from pipeline.features import build_features_for_symbol
    from backtest.walk_forward import summary, walk_forward

    print(f"Running walk-forward backtest for {args.symbol}...")
    feats = build_features_for_symbol(args.symbol, start=args.start)
    results = walk_forward(feats, n_folds=args.folds)
    tbl = summary(results)
    if getattr(args, "verbose", False):
        for r in results:
            print(f"Fold {r.fold}: Sharpe={r.sharpe:.3f}  MaxDD={r.max_drawdown:.2%}  "
                  f"Return={r.total_return:.2%}  Trades={r.n_trades}")
        print()
    print(tbl.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Trading ML pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Download and cache OHLCV data")
    p_fetch.add_argument("symbol")
    p_fetch.add_argument("--start", default="2015-01-01")
    p_fetch.add_argument("--end", default=None)
    p_fetch.add_argument("--refresh", action="store_true")

    # train
    p_train = sub.add_parser("train", help="Train ensemble model")
    p_train.add_argument("symbol", nargs="?", default="AAPL",
                         help="Single symbol to train on (ignored if --universe is set)")
    p_train.add_argument("--universe", action="store_true",
                         help="Train on all 20 universe symbols (recommended)")
    p_train.add_argument("--start", default="2015-01-01")
    p_train.add_argument("--output", default="models/model.pkl")

    # backtest
    p_bt = sub.add_parser("backtest", help="Walk-forward backtest")
    p_bt.add_argument("symbol")
    p_bt.add_argument("--start", default="2015-01-01")
    p_bt.add_argument("--folds", type=int, default=5)
    p_bt.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    {"fetch": cmd_fetch, "train": cmd_train, "backtest": cmd_backtest}[args.command](args)


if __name__ == "__main__":
    main()
