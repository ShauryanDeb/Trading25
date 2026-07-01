"""Walk-forward evaluation — the only honest way to measure this strategy.

For each test segment (default: calendar years), trains a fresh model using
ONLY data that precedes the segment, then simulates trading on the segment
with capital chained forward. No model ever sees its own test period.

The label's forward-return window (3 or 5 days) is dropped from the end of
each training set by build_features (rows without a complete forward window
have no label), which acts as the purge/embargo between train and test.

Usage:
    from pipeline.walkforward import walk_forward
    result = walk_forward(start_year=2021)          # defaults: abs_3d + rank
    result = walk_forward(target_mode="rel_5d", entry_mode="threshold")
"""

from __future__ import annotations

import pandas as pd

from pipeline.backtest import DEFAULT_UNIVERSE, run_swing_backtest, _compute_metrics
from pipeline.data import fetch
from pipeline.features import _fetch_macro, build_features


def _segments(start_year: int, end_date: str) -> list[tuple[str, str, str]]:
    """Return (train_end, test_start, test_end) triples for yearly segments."""
    end_ts = pd.Timestamp(end_date)
    segs = []
    for year in range(start_year, end_ts.year + 1):
        test_start = f"{year}-01-01"
        test_end = min(pd.Timestamp(f"{year}-12-31"), end_ts).strftime("%Y-%m-%d")
        train_end = f"{year - 1}-12-31"
        if pd.Timestamp(test_start) > end_ts:
            break
        segs.append((train_end, test_start, test_end))
    return segs


def _train_model(symbols: list[str], train_start: str, train_end: str,
                 target_mode: str, verbose: bool = False):
    from pipeline.model import StockEnsemble

    macro = None
    try:
        macro = _fetch_macro(train_start, train_end)
    except Exception:
        pass

    frames = []
    for sym in symbols:
        try:
            ohlcv = fetch(sym, start=train_start, end=train_end)
            frames.append(build_features(ohlcv, macro=macro, target_mode=target_mode))
        except Exception as e:
            if verbose:
                print(f"    train SKIP {sym}: {e}")
    combined = pd.concat(frames).sort_index().reset_index(drop=True)
    X = combined.drop(columns=["Target"])
    y = combined["Target"]
    if verbose:
        print(f"    {len(combined)} training rows through {train_end}")
    model = StockEnsemble()
    model.fit(X, y)
    return model


def walk_forward(
    symbols: list[str] | None = None,
    start_year: int = 2021,
    end_date: str | None = None,
    train_start: str = "2015-01-01",
    initial_capital: float = 50_000.0,
    target_mode: str = "abs_3d",
    entry_mode: str = "rank",
    cost_bps: float = 5.0,
    verbose: bool = False,
    **strategy_kwargs,
) -> dict:
    """Run yearly walk-forward: retrain before each test year, chain capital.

    Returns {"segments": DataFrame, "trades": DataFrame, "metrics": dict}.
    Every metric in the output is out-of-sample.
    """
    symbols = symbols or DEFAULT_UNIVERSE
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    seg_rows = []
    all_trades = []
    all_portfolio = []
    capital = initial_capital

    for train_end, test_start, test_end in _segments(start_year, end_date):
        print(f"  [{test_start[:4]}] training through {train_end}, "
              f"testing {test_start}..{test_end}  (capital ${capital:,.0f})")
        model = _train_model(symbols, train_start, train_end, target_mode, verbose)

        res = run_swing_backtest(
            model=model,
            symbols=symbols,
            start=test_start,
            end=test_end,
            initial_capital=capital,
            target_mode=target_mode,
            entry_mode=entry_mode,
            cost_bps=cost_bps,
            verbose=verbose,
            **strategy_kwargs,
        )
        m = res["metrics"]
        seg_rows.append({
            "year": test_start[:4],
            "start_capital": round(capital, 2),
            "end_capital": m["final_value"],
            "return_pct": m["total_return_pct"],
            "win_rate_pct": m["win_rate_pct"],
            "profit_factor": m["profit_factor"],
            "sharpe": m["sharpe"],
            "max_dd_pct": m["max_drawdown_pct"],
            "n_trades": m["n_trades"],
            "spy_return_pct": m["spy_return_pct"],
        })
        if not res["trades"].empty:
            all_trades.append(res["trades"])
        all_portfolio.append(res["portfolio"])
        capital = m["final_value"]

    segments_df = pd.DataFrame(seg_rows)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    portfolio_df = pd.concat(all_portfolio, ignore_index=True)

    spy = fetch("SPY", start=f"{start_year}-01-01", end=end_date)
    metrics = _compute_metrics(trades_df, portfolio_df, initial_capital,
                               f"{start_year}-01-01", end_date, spy)
    return {"segments": segments_df, "trades": trades_df,
            "portfolio": portfolio_df, "metrics": metrics}


def print_walk_forward(result: dict, label: str = "") -> None:
    m = result["metrics"]
    print(f"\n{'=' * 66}")
    print(f"WALK-FORWARD (all out-of-sample){f' — {label}' if label else ''}")
    print("=" * 66)
    print(result["segments"].to_string(index=False))
    print("-" * 66)
    print(f"  OVERALL: return={m['total_return_pct']:+.2f}%  CAGR={m['cagr_pct']:+.2f}%  "
          f"Sharpe={m['sharpe']:.2f}")
    print(f"           win_rate={m['win_rate_pct']:.1f}%  profit_factor={m['profit_factor']:.3f}  "
          f"max_dd={m['max_drawdown_pct']:.1f}%  trades={m['n_trades']}")
    print(f"           SPY buy-and-hold={m['spy_return_pct']:+.2f}%  "
          f"alpha={m['alpha_pct']:+.2f}%")
