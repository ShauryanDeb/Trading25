"""Swing strategy portfolio backtester.

Simulates the full swing-bot strategy (long + short, stops, sector caps,
max-hold) over a historical date range using the trained ensemble model.
No look-ahead: signals at date T use only features available on date T,
and fills are executed at the *next* day's Open.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data import fetch
from pipeline.features import FEATURE_COLS, _fetch_macro, build_features

# ──────────────────────────────────────────────────────────────────────────────
# Sector map  (mirrors alpaca_bot.py)
# ──────────────────────────────────────────────────────────────────────────────
_SYM_TO_SECTOR: dict[str, str] = {
    **{s: "tech" for s in [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE",
        "CRM", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "SNPS"]},
    **{s: "finance" for s in [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA",
        "COF", "USB", "PNC", "TFC", "CME", "ICE", "CB", "PGR", "MET", "AIG"]},
    **{s: "health" for s in [
        "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
        "GILD", "ISRG", "VRTX", "REGN", "ZTS", "SYK", "BSX", "MDT", "ELV", "CVS"]},
    **{s: "consumer" for s in [
        "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW", "TJX", "BKNG",
        "MAR", "HLT", "YUM", "DPZ", "CMG", "ORLY", "AZO", "TSCO", "DG", "DLTR"]},
    **{s: "industrial" for s in [
        "CAT", "DE", "HON", "GE", "RTX", "LMT", "BA", "UPS", "FDX", "NSC"]},
    **{s: "energy" for s in [
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "DVN"]},
    **{s: "other" for s in [
        "PEP", "KO", "PG", "MO", "PM", "DIS", "NFLX", "T", "VZ", "CMCSA"]},
}

# Default backtesting universe — liquid names across all sectors
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "BAC", "GS", "V", "MA",
    "UNH", "LLY", "JNJ", "ABBV", "MRK",
    "WMT", "HD", "COST",
    "XOM", "CVX", "COP", "SLB", "DVN",
    "AVGO", "AMD", "QCOM", "CRM", "ADBE",
]

SCENARIOS: dict[str, tuple[str, str]] = {
    "2020-crash":    ("2020-02-01", "2020-04-30"),
    "2022-bear":     ("2022-01-01", "2022-12-31"),
    "2023-recovery": ("2023-01-01", "2023-12-31"),
    "2024-bull":     ("2024-01-01", "2024-12-31"),
    "ytd":           ("2026-01-01", "2026-06-29"),
    "3yr":           ("2023-01-01", "2026-06-29"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _next_open(ohlcv: pd.DataFrame, from_date: pd.Timestamp) -> float | None:
    """Return the Open price on or after *from_date*."""
    future = ohlcv.loc[ohlcv.index >= from_date]
    return float(future.iloc[0]["Open"]) if not future.empty else None


def _sector(sym: str) -> str:
    return _SYM_TO_SECTOR.get(sym, "other")


# ──────────────────────────────────────────────────────────────────────────────
# Core simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_swing_backtest(
    model_path: str = "models/model.pkl",
    model=None,
    symbols: list[str] | None = None,
    start: str = "2022-01-01",
    end: str = "2022-12-31",
    initial_capital: float = 50_000.0,
    buy_threshold: float = 0.48,
    short_threshold: float = 0.30,
    sell_threshold: float = 0.42,
    cover_threshold: float = 0.48,
    stop_loss_pct: float = 0.07,
    max_hold_days: int = 3,
    max_position_pct: float = 0.05,
    max_per_sector: int = 3,
    max_positions: int = 10,
    max_short_positions: int = 5,
    entry_mode: str = "threshold",
    rank_long_k: int = 5,
    rank_short_k: int = 5,
    cost_bps: float = 5.0,
    target_mode: str = "abs_3d",
    verbose: bool = False,
) -> dict:
    """Run a full portfolio backtest and return trades + portfolio DataFrames + metrics.

    Args:
        model: Pre-loaded StockEnsemble. If None, loads from model_path.
        entry_mode:
            "threshold" — long when proba > buy_threshold, short when
                          proba < short_threshold (fixed cutoffs).
            "rank"      — long the top rank_long_k of the day's cross-sectional
                          proba ranking, short the bottom rank_short_k. Adapts
                          to compressed proba distributions; longs still
                          require proba above the day's median, shorts below.
        cost_bps: One-way slippage+spread cost in basis points applied to
                  every fill (5 = 0.05%). Buys/covers fill higher, sells/
                  shorts fill lower.
        target_mode: Passed to build_features — must match the label the
                     model was trained on.
    """
    from pipeline.model import StockEnsemble

    symbols = symbols or DEFAULT_UNIVERSE
    # Extra warmup so indicators (MA50, RSI etc.) are stable at start date
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")

    if model is None:
        print(f"  Loading model from {model_path} ...")
        model = StockEnsemble.load(Path(model_path))

    cost = cost_bps / 10_000.0

    # ── Phase 1: Fetch data & build signal series per symbol ──────────────────
    print(f"  Fetching data for {len(symbols)} symbols  ({start} to {end}) ...")
    try:
        macro = _fetch_macro(warmup_start, end)
    except Exception as exc:
        print(f"  [WARN] macro fetch failed ({exc}), continuing without macro features")
        macro = None

    signal_matrix: dict[str, pd.Series] = {}   # sym → Series(date → proba)
    price_matrix: dict[str, pd.DataFrame] = {}  # sym → full OHLCV

    for sym in symbols:
        try:
            ohlcv = fetch(sym, start=warmup_start, end=end)
            feats = build_features(ohlcv, macro=macro, target_mode=target_mode)
            feat_rows = feats[FEATURE_COLS].dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probas = model.predict_proba(feat_rows)[:, 1]
            signal_matrix[sym] = pd.Series(
                probas.astype(float), index=feat_rows.index, name=sym
            )
            price_matrix[sym] = ohlcv
            if verbose:
                print(f"    {sym}: {len(feat_rows)} signal rows")
        except Exception as exc:
            if verbose:
                print(f"    SKIP {sym}: {exc}")

    if not signal_matrix:
        raise RuntimeError("No symbols loaded — check model path and data.")

    # ── Phase 2: Trading calendar ─────────────────────────────────────────────
    spy_ohlcv = fetch("SPY", start=start, end=end)
    trading_dates = spy_ohlcv.index[spy_ohlcv.index >= pd.Timestamp(start)]
    spy_start_price = float(spy_ohlcv.iloc[0]["Close"])

    # ── Phase 3: Day-by-day simulation ────────────────────────────────────────
    cash: float = initial_capital
    positions: dict[str, dict] = {}
    portfolio_history: list[dict] = []
    all_trades: list[dict] = []

    for i, date in enumerate(trading_dates):
        next_date = trading_dates[i + 1] if i + 1 < len(trading_dates) else None

        # Signals available today
        signals: dict[str, float] = {
            sym: float(s.loc[date])
            for sym, s in signal_matrix.items()
            if date in s.index
        }

        # ── Exit checks ───────────────────────────────────────────────────────
        for sym in list(positions.keys()):
            pos = positions[sym]
            ohlcv_sym = price_matrix.get(sym, pd.DataFrame())
            if ohlcv_sym.empty or date not in ohlcv_sym.index:
                continue

            close_price = float(ohlcv_sym.loc[date, "Close"])
            is_short = pos["side"] == "short"
            pnl_pct = (
                (pos["entry_price"] - close_price) / pos["entry_price"]
                if is_short
                else (close_price - pos["entry_price"]) / pos["entry_price"]
            )
            hold_days = (date - pos["entry_date"]).days
            proba = signals.get(sym, 0.5)

            exit_reason: str | None = None
            if pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif hold_days >= max_hold_days:
                exit_reason = "max_hold"
            elif not is_short and proba < sell_threshold:
                exit_reason = "signal_fade"
            elif is_short and proba >= cover_threshold:
                exit_reason = "signal_recovered"

            if exit_reason:
                # Fill at next Open if available, else current Close
                exit_price = close_price
                if next_date is not None:
                    p = _next_open(ohlcv_sym, next_date)
                    if p is not None:
                        exit_price = p
                # Slippage: covers fill above the print, sells below it
                exit_price *= (1 + cost) if is_short else (1 - cost)

                if is_short:
                    actual_pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
                    pnl_dollar = pos["qty"] * pos["entry_price"] * actual_pnl_pct
                    cash += pos["cost"] + pnl_dollar
                else:
                    actual_pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                    pnl_dollar = pos["qty"] * (exit_price - pos["entry_price"])
                    cash += pos["qty"] * exit_price

                all_trades.append({
                    "symbol": sym,
                    "side": pos["side"],
                    "entry_date": pos["entry_date"].date(),
                    "exit_date": date.date(),
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "qty": pos["qty"],
                    "pnl_pct": round(actual_pnl_pct, 5),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "exit_reason": exit_reason,
                    "hold_days": hold_days,
                })
                del positions[sym]

        # ── Mark portfolio to market ──────────────────────────────────────────
        mark = 0.0
        for sym, pos in positions.items():
            ohlcv_sym = price_matrix.get(sym, pd.DataFrame())
            if not ohlcv_sym.empty and date in ohlcv_sym.index:
                cp = float(ohlcv_sym.loc[date, "Close"])
                if pos["side"] == "short":
                    mark += pos["cost"] + pos["qty"] * (pos["entry_price"] - cp)
                else:
                    mark += pos["qty"] * cp
            else:
                mark += pos["cost"]  # no data: carry at cost

        portfolio_value = cash + mark
        spy_close = float(spy_ohlcv.loc[date, "Close"]) if date in spy_ohlcv.index else None
        spy_bnh = spy_close / spy_start_price * initial_capital if spy_close else None

        portfolio_history.append({
            "date": date.date(),
            "portfolio_value": round(portfolio_value, 2),
            "cash": round(cash, 2),
            "n_positions": len(positions),
            "spy_bnh": round(spy_bnh, 2) if spy_bnh else None,
        })

        if next_date is None:
            break

        # ── Entry logic ───────────────────────────────────────────────────────
        max_alloc = portfolio_value * max_position_pct

        # Count current sector exposures
        long_sector: dict[str, int] = {}
        short_sector: dict[str, int] = {}
        n_longs = n_shorts = 0
        for sym, pos in positions.items():
            sec = _sector(sym)
            if pos["side"] == "long":
                long_sector[sec] = long_sector.get(sec, 0) + 1
                n_longs += 1
            else:
                short_sector[sec] = short_sector.get(sec, 0) + 1
                n_shorts += 1

        # Candidate selection — threshold (fixed cutoffs) or rank (cross-sectional)
        if entry_mode == "rank":
            ranked = sorted(signals.items(), key=lambda x: -x[1])
            day_median = float(np.median([p for _, p in ranked])) if ranked else 0.5
            long_candidates = [(s, p) for s, p in ranked[:rank_long_k] if p > day_median]
            short_candidates = [(s, p) for s, p in ranked[-rank_short_k:] if p < day_median]
            short_candidates.sort(key=lambda x: x[1])
        else:
            long_candidates = sorted(
                [(s, p) for s, p in signals.items() if p >= buy_threshold],
                key=lambda x: -x[1],
            )
            short_candidates = sorted(
                [(s, p) for s, p in signals.items() if p < short_threshold],
                key=lambda x: x[1],
            )

        for sym, _ in long_candidates:
            if n_longs >= max_positions:
                break
            if sym in positions or long_sector.get(_sector(sym), 0) >= max_per_sector:
                continue
            ohlcv_sym = price_matrix.get(sym, pd.DataFrame())
            entry_price = _next_open(ohlcv_sym, next_date)
            if not entry_price or entry_price <= 0:
                continue
            entry_price *= (1 + cost)  # buys fill above the print
            qty = max(1, int(max_alloc / entry_price))
            outlay = qty * entry_price
            if outlay > cash * 0.95:
                continue
            cash -= outlay
            positions[sym] = {
                "side": "long",
                "entry_price": entry_price,
                "entry_date": next_date,
                "qty": qty,
                "cost": outlay,
            }
            long_sector[_sector(sym)] = long_sector.get(_sector(sym), 0) + 1
            n_longs += 1

        for sym, _ in short_candidates:
            if n_shorts >= max_short_positions:
                break
            if sym in positions or short_sector.get(_sector(sym), 0) >= max_per_sector:
                continue
            ohlcv_sym = price_matrix.get(sym, pd.DataFrame())
            entry_price = _next_open(ohlcv_sym, next_date)
            if not entry_price or entry_price <= 0:
                continue
            entry_price *= (1 - cost)  # shorts fill below the print
            qty = max(1, int(max_alloc / entry_price))
            outlay = qty * entry_price  # margin collateral
            if outlay > cash * 0.95:
                continue
            cash -= outlay
            positions[sym] = {
                "side": "short",
                "entry_price": entry_price,
                "entry_date": next_date,
                "qty": qty,
                "cost": outlay,
            }
            short_sector[_sector(sym)] = short_sector.get(_sector(sym), 0) + 1
            n_shorts += 1

    # ── Force-close all remaining positions at period end ─────────────────────
    last_date = trading_dates[-1]
    for sym, pos in list(positions.items()):
        ohlcv_sym = price_matrix.get(sym, pd.DataFrame())
        if ohlcv_sym.empty or last_date not in ohlcv_sym.index:
            continue
        cp = float(ohlcv_sym.loc[last_date, "Close"])
        is_short = pos["side"] == "short"
        cp *= (1 + cost) if is_short else (1 - cost)
        if is_short:
            pnl_pct = (pos["entry_price"] - cp) / pos["entry_price"]
            pnl_dollar = pos["qty"] * pos["entry_price"] * pnl_pct
            cash += pos["cost"] + pnl_dollar
        else:
            pnl_pct = (cp - pos["entry_price"]) / pos["entry_price"]
            pnl_dollar = pos["qty"] * (cp - pos["entry_price"])
            cash += pos["qty"] * cp

        entry_dt = pos["entry_date"]
        all_trades.append({
            "symbol": sym,
            "side": pos["side"],
            "entry_date": entry_dt.date() if hasattr(entry_dt, "date") else entry_dt,
            "exit_date": last_date.date(),
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(cp, 4),
            "qty": pos["qty"],
            "pnl_pct": round(pnl_pct, 5),
            "pnl_dollar": round(pnl_dollar, 2),
            "exit_reason": "end_of_period",
            "hold_days": (last_date - entry_dt).days,
        })

    # ── Compute metrics & return ──────────────────────────────────────────────
    trades_df = pd.DataFrame(all_trades)
    portfolio_df = pd.DataFrame(portfolio_history)

    # Update final row with actual cash
    if not portfolio_df.empty:
        portfolio_df.iloc[-1, portfolio_df.columns.get_loc("portfolio_value")] = round(cash, 2)
        portfolio_df.iloc[-1, portfolio_df.columns.get_loc("cash")] = round(cash, 2)

    metrics = _compute_metrics(trades_df, portfolio_df, initial_capital, start, end, spy_ohlcv)
    return {"trades": trades_df, "portfolio": portfolio_df, "metrics": metrics}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    trades_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    initial_capital: float,
    start: str,
    end: str,
    spy_ohlcv: pd.DataFrame,
) -> dict:
    if portfolio_df.empty:
        return {}

    final_value = float(portfolio_df.iloc[-1]["portfolio_value"])
    total_return = (final_value - initial_capital) / initial_capital

    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.01)
    cagr = (final_value / initial_capital) ** (1.0 / years) - 1

    vals = portfolio_df["portfolio_value"].values.astype(float)
    peak = np.maximum.accumulate(vals)
    drawdowns = (vals - peak) / np.where(peak == 0, 1, peak)
    max_dd = float(drawdowns.min())

    daily_rets = pd.Series(vals).pct_change().dropna()
    sharpe = (
        float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))
        if len(daily_rets) > 1 and daily_rets.std() > 0
        else 0.0
    )

    # Trade stats — exclude end_of_period "forced" closes for cleaner win rate
    if not trades_df.empty:
        closed = trades_df[trades_df["exit_reason"] != "end_of_period"]
        n_trades = len(closed)
        if n_trades > 0:
            win_rate = float((closed["pnl_dollar"] > 0).mean())
            wins = closed.loc[closed["pnl_dollar"] > 0, "pnl_dollar"].sum()
            losses = closed.loc[closed["pnl_dollar"] < 0, "pnl_dollar"].abs().sum()
            profit_factor = float(wins / losses) if losses > 0 else float("inf")
            avg_hold = float(closed["hold_days"].mean())
            n_longs = int((closed["side"] == "long").sum())
            n_shorts = int((closed["side"] == "short").sum())
            longs_c = closed[closed["side"] == "long"]
            shorts_c = closed[closed["side"] == "short"]
            long_wr = float((longs_c["pnl_dollar"] > 0).mean()) if n_longs > 0 else 0.0
            short_wr = float((shorts_c["pnl_dollar"] > 0).mean()) if n_shorts > 0 else 0.0
        else:
            win_rate = profit_factor = avg_hold = long_wr = short_wr = 0.0
            n_longs = n_shorts = 0
    else:
        n_trades = n_longs = n_shorts = 0
        win_rate = profit_factor = avg_hold = long_wr = short_wr = 0.0

    spy_return = (float(spy_ohlcv.iloc[-1]["Close"]) / float(spy_ohlcv.iloc[0]["Close"])) - 1
    spy_cagr = (1 + spy_return) ** (1.0 / years) - 1

    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(min(profit_factor, 99.9), 3),
        "avg_hold_days": round(avg_hold, 1),
        "n_trades": n_trades,
        "n_longs": n_longs,
        "n_shorts": n_shorts,
        "long_win_rate_pct": round(long_wr * 100, 1),
        "short_win_rate_pct": round(short_wr * 100, 1),
        "spy_return_pct": round(spy_return * 100, 2),
        "spy_cagr_pct": round(spy_cagr * 100, 2),
        "alpha_pct": round((total_return - spy_return) * 100, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

def print_report(result: dict, scenario_name: str = "") -> None:
    m = result["metrics"]
    trades = result["trades"]
    portfolio = result["portfolio"]

    W = 62
    title = f" BACKTEST{f': {scenario_name}' if scenario_name else ''} "
    print("\n" + "=" * W)
    print(title.center(W))
    print("=" * W)

    print(f"\n  PERFORMANCE")
    print(f"    Initial capital:   ${m['initial_capital']:>10,.0f}")
    print(f"    Final value:       ${m['final_value']:>10,.2f}")
    print(f"    Total return:      {m['total_return_pct']:>+9.2f}%")
    print(f"    CAGR:              {m['cagr_pct']:>+9.2f}%")
    print(f"    Max drawdown:      {m['max_drawdown_pct']:>9.2f}%")
    print(f"    Sharpe ratio:      {m['sharpe']:>9.3f}")

    print(f"\n  vs SPY BUY-AND-HOLD")
    print(f"    SPY return:        {m['spy_return_pct']:>+9.2f}%")
    print(f"    SPY CAGR:          {m['spy_cagr_pct']:>+9.2f}%")
    print(f"    Alpha:             {m['alpha_pct']:>+9.2f}%")

    print(f"\n  TRADE STATISTICS")
    print(f"    Closed trades:     {m['n_trades']:>9d}")
    print(f"    Longs:             {m['n_longs']:>9d}  win {m['long_win_rate_pct']:.1f}%")
    print(f"    Shorts:            {m['n_shorts']:>9d}  win {m['short_win_rate_pct']:.1f}%")
    print(f"    Overall win rate:  {m['win_rate_pct']:>9.1f}%")
    print(f"    Profit factor:     {m['profit_factor']:>9.3f}")
    print(f"    Avg hold days:     {m['avg_hold_days']:>9.1f}")

    if not trades.empty:
        def _fmt_row(r):
            return (
                f"    {r['symbol']:6s} {r['side']:5s}  "
                f"{str(r['entry_date']):10s} to {str(r['exit_date']):10s}  "
                f"{r['pnl_pct']*100:+.2f}%  ${r['pnl_dollar']:>+8.2f}"
                f"  [{r['exit_reason']}]"
            )

        print(f"\n  TOP 5 WINNERS")
        for _, r in trades.nlargest(5, "pnl_dollar").iterrows():
            print(_fmt_row(r))

        print(f"\n  TOP 5 LOSERS")
        for _, r in trades.nsmallest(5, "pnl_dollar").iterrows():
            print(_fmt_row(r))

        # Exit-reason breakdown
        print(f"\n  EXIT BREAKDOWN")
        for reason, grp in trades.groupby("exit_reason"):
            wr = (grp["pnl_dollar"] > 0).mean() * 100
            avg = grp["pnl_dollar"].mean()
            print(f"    {reason:20s}  n={len(grp):3d}  win={wr:4.0f}%  avg=${avg:+.2f}")

    # Monthly breakdown
    if not portfolio.empty:
        port = portfolio.copy()
        port["date"] = pd.to_datetime(port["date"])
        port["month"] = port["date"].dt.to_period("M")
        monthly_strat = port.groupby("month")["portfolio_value"].agg(["first", "last"])
        monthly_strat["ret"] = (monthly_strat["last"] / monthly_strat["first"] - 1) * 100

        spy_col = port.copy()
        spy_col["spy_bnh"] = pd.to_numeric(spy_col["spy_bnh"], errors="coerce")
        monthly_spy = spy_col.groupby("month")["spy_bnh"].agg(["first", "last"])
        monthly_spy["ret"] = (monthly_spy["last"] / monthly_spy["first"] - 1) * 100

        print(f"\n  MONTHLY  (strat vs SPY)")
        print(f"    {'Month':<10}  {'Strat':>7}  {'SPY':>7}")
        for period in monthly_strat.index:
            sr = monthly_strat.loc[period, "ret"]
            sp = monthly_spy.loc[period, "ret"] if period in monthly_spy.index else float("nan")
            spy_str = f"{sp:>+6.1f}%" if not np.isnan(sp) else "    N/A"
            print(f"    {str(period):<10}  {sr:>+6.1f}%  {spy_str}")

    print("\n" + "=" * W)
