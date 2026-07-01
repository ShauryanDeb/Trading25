"""Hold-horizon experiment — does aligning the max-hold with the rel_5d label
horizon beat the honest walk-forward baseline (+35.6% / PF 1.18, 2021-2026H1)?

PRE-REGISTERED 2026-07-01, before any variant result was observed:

  Hypothesis: the model predicts 5-TRADING-DAY relative strength (rel_5d label,
  the finding-5 winner), but max_hold_days=3 calendar days force-exits trades
  after ~1-3 trading days — before the predicted window completes — and the
  resulting churn (2,357 trades x 2 x 5 bps on ~5% position notional) is a
  material cost drag. Holding to the label horizon should harvest more of the
  predicted move per round trip and mechanically cut trade count.

  Primary:   max_hold_days=7 calendar days. 7 calendar days = exactly 5
             trading days for any weekday entry (mod holidays) = the label
             horizon. Chosen by prior, NOT by scanning.
  Neighbors: 5 and 10 — robustness check only. They must be directionally
             consistent with the primary (smooth response surface); they are
             NOT candidates to "pick the best of".
  Control:   3 (the baseline) — must reproduce `python run.py walkforward`
             exactly (models are deterministic: fixed seeds, cached data).

  Accept iff ALL of:
    (a) primary total return > +35.6% AND profit factor > 1.18
    (b) primary improves the yearly return in >= 4 of 6 segments
    (c) neighbors 5 and 10 also directionally improve on the baseline
    (d) trade count drops materially (the claimed mechanism must show up)
  Everything else (thresholds, sizing, universe, costs, label, entry mode)
  stays untouched. One parameter changes.

Training is invariant to the hold horizon, so each segment's model is trained
once and reused across variants; simulation goes through the pristine
run_swing_backtest with capital chained per variant, exactly as
pipeline.walkforward.walk_forward does.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.backtest import DEFAULT_UNIVERSE, run_swing_backtest, _compute_metrics
from pipeline.data import fetch
from pipeline.walkforward import _segments, _train_model, print_walk_forward

VARIANTS = [3, 5, 7, 10]  # 3 = control; 7 = pre-registered primary


def main() -> None:
    start_year = 2021
    train_start = "2015-01-01"
    initial_capital = 50_000.0
    target_mode, entry_mode, cost_bps = "rel_5d", "threshold", 5.0
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    symbols = DEFAULT_UNIVERSE

    capital = {v: initial_capital for v in VARIANTS}
    seg_rows: dict[int, list] = {v: [] for v in VARIANTS}
    trades: dict[int, list] = {v: [] for v in VARIANTS}
    portfolio: dict[int, list] = {v: [] for v in VARIANTS}

    for train_end, test_start, test_end in _segments(start_year, end_date):
        print(f"\n[{test_start[:4]}] training through {train_end} ...", flush=True)
        model = _train_model(symbols, train_start, train_end, target_mode)
        for v in VARIANTS:
            res = run_swing_backtest(
                model=model,
                symbols=symbols,
                start=test_start,
                end=test_end,
                initial_capital=capital[v],
                target_mode=target_mode,
                entry_mode=entry_mode,
                cost_bps=cost_bps,
                max_hold_days=v,
            )
            m = res["metrics"]
            seg_rows[v].append({
                "year": test_start[:4],
                "start_capital": round(capital[v], 2),
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
                trades[v].append(res["trades"])
            portfolio[v].append(res["portfolio"])
            capital[v] = m["final_value"]
            print(f"    hold={v:2d}: {m['total_return_pct']:+7.2f}%  "
                  f"PF={m['profit_factor']:.3f}  trades={m['n_trades']}  "
                  f"avg_hold={m['avg_hold_days']:.1f}d", flush=True)

    spy = fetch("SPY", start=f"{start_year}-01-01", end=end_date)
    for v in VARIANTS:
        trades_df = pd.concat(trades[v], ignore_index=True) if trades[v] else pd.DataFrame()
        portfolio_df = pd.concat(portfolio[v], ignore_index=True)
        metrics = _compute_metrics(trades_df, portfolio_df, initial_capital,
                                   f"{start_year}-01-01", end_date, spy)
        result = {"segments": pd.DataFrame(seg_rows[v]), "trades": trades_df,
                  "portfolio": portfolio_df, "metrics": metrics}
        print_walk_forward(result, label=f"max_hold_days={v}")
        pd.DataFrame(seg_rows[v]).to_csv(
            f"reports/hold_experiment_segments_h{v}.csv", index=False)
        if not trades_df.empty:
            trades_df.to_csv(f"reports/hold_experiment_trades_h{v}.csv", index=False)


if __name__ == "__main__":
    main()
