"""CLI: fetch / train / backtest commands."""
import argparse
import sys
from pathlib import Path


def cmd_fetch(args):
    from pipeline.data import fetch
    df = fetch(args.symbol, start=args.start, end=args.end, force_refresh=args.refresh)
    print(f"Fetched {len(df)} rows for {args.symbol}  ({df.index[0].date()} to {df.index[-1].date()})")


def cmd_train(args):
    from pipeline.features import build_features_for_symbol
    from pipeline.model import StockEnsemble

    print(f"Building features for {args.symbol}...")
    feats = build_features_for_symbol(args.symbol, start=args.start)
    X = feats.drop(columns=["Target"])
    y = feats["Target"]
    print(f"  {len(feats)} samples, {X.shape[1]} features")

    model = StockEnsemble()
    model.fit(X, y)
    out = Path(args.output)
    model.save(out)
    print(f"Model saved → {out}")


def cmd_backtest(args):
    from pipeline.features import build_features_for_symbol
    from backtest.walk_forward import summary, walk_forward

    print(f"Running walk-forward backtest for {args.symbol}...")
    feats = build_features_for_symbol(args.symbol, start=args.start)
    results = walk_forward(feats, n_folds=args.folds)
    tbl = summary(results)
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
    p_train.add_argument("symbol")
    p_train.add_argument("--start", default="2015-01-01")
    p_train.add_argument("--output", default="models/model.pkl")

    # backtest
    p_bt = sub.add_parser("backtest", help="Walk-forward backtest")
    p_bt.add_argument("symbol")
    p_bt.add_argument("--start", default="2015-01-01")
    p_bt.add_argument("--folds", type=int, default=5)

    args = parser.parse_args()
    {"fetch": cmd_fetch, "train": cmd_train, "backtest": cmd_backtest}[args.command](args)


if __name__ == "__main__":
    main()
