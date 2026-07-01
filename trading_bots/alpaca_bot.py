"""Alpaca paper-trading bot using alpaca-py SDK.

Loads a trained StockEnsemble model, generates signals via predict_proba,
and rebalances at market open — long-only, 5% max position, -7% stop-loss.

Environment (.env):
    APCA_API_KEY_ID     -- Alpaca paper key
    APCA_API_SECRET_KEY -- Alpaca paper secret
    MODEL_PATH          -- path to saved StockEnsemble .pkl  (default: models/model.pkl)

Flags:
    --dry-run   Connect to Alpaca, print positions/buying power, run one signal
                cycle but do NOT place any orders.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run as a script from any directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
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
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "DVN",
    # Other large-caps
    "PEP", "KO", "PG", "MO", "PM", "DIS", "NFLX", "T", "VZ", "CMCSA",
]

MAX_POSITION_PCT = 0.05   # 5% of portfolio per symbol
TOP_N = 10                # max positions held at once — matches backtested optimum
MAX_PER_SECTOR = 3        # max concurrent positions within one GICS sector
MAX_HOLD_DAYS = 3         # force-exit after 3 calendar days — limits exposure to
                          # correlated sector drawdowns that the signal doesn't catch
BUY_THRESHOLD = 0.48      # predict_proba > this => BUY candidate
SELL_THRESHOLD = 0.42     # predict_proba < this => exit long / SELL/SKIP
SHORT_THRESHOLD = 0.30    # predict_proba < this => SHORT candidate
COVER_THRESHOLD = 0.48    # predict_proba > this => cover short (signal recovered)
STOP_LOSS_PCT = -0.07     # -7% from entry => stop-loss (long); +7% => stop (short)

SECTOR_MAP: dict[str, list[str]] = {
    "Tech":        ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","ORCL","ADBE",
                    "CRM","AMD","INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","SNPS"],
    "Financials":  ["JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","V","MA",
                    "COF","USB","PNC","TFC","CME","ICE","CB","PGR","MET","AIG"],
    "Healthcare":  ["UNH","LLY","JNJ","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
                    "GILD","ISRG","VRTX","REGN","ZTS","SYK","BSX","MDT","ELV","CVS"],
    "Consumer":    ["WMT","COST","HD","MCD","SBUX","NKE","TGT","LOW","TJX","BKNG",
                    "MAR","HLT","YUM","DPZ","CMG","ORLY","AZO","TSCO","DG","DLTR",
                    "PEP","KO","PG","MO","PM"],
    "Industrials": ["CAT","DE","HON","GE","RTX","LMT","BA","UPS","FDX","NSC"],
    "Energy":      ["XOM","CVX","COP","SLB","EOG","OXY","MPC","VLO","PSX","DVN"],
    "Comms":       ["DIS","NFLX","T","VZ","CMCSA"],
}
# Reverse lookup: symbol -> sector
_SYM_TO_SECTOR: dict[str, str] = {
    sym: sector for sector, syms in SECTOR_MAP.items() for sym in syms
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client():
    from alpaca.trading.client import TradingClient
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise EnvironmentError(
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env"
        )
    return TradingClient(key, secret, paper=True)


def _portfolio_value(client) -> float:
    return float(client.get_account().portfolio_value)


def _buying_power(client) -> float:
    return float(client.get_account().buying_power)


def _current_positions(client) -> dict[str, dict]:
    """Return {symbol: {market_value, avg_entry_price, qty, side}}."""
    result = {}
    for p in client.get_all_positions():
        side = str(p.side).lower()
        side = "short" if "short" in side else "long"
        result[p.symbol] = {
            "market_value": float(p.market_value),
            "avg_entry_price": float(p.avg_entry_price),
            "qty": float(p.qty),
            "side": side,
        }
    return result


def _latest_price(symbol: str) -> float | None:
    """Fetch latest close from yfinance (fallback when market closed)."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="2d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def _entry_dates() -> tuple[dict[str, pd.Timestamp], dict[str, pd.Timestamp]]:
    """Scan trade CSVs to find the most recent entry date for longs and shorts.

    Returns (long_dates, short_dates) — each {symbol: timestamp}.
    """
    long_dates: dict[str, pd.Timestamp] = {}
    short_dates: dict[str, pd.Timestamp] = {}
    report_dir = Path(__file__).resolve().parent.parent / "reports"
    for csv_path in sorted(report_dir.glob("trades_*.csv")):
        try:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            for _, row in df.iterrows():
                side = str(row.get("side", ""))
                sym = row["symbol"]
                ts = pd.Timestamp(row["timestamp"]).tz_localize(None)
                if side.startswith("buy") and not side.startswith("buy_to_cover"):
                    if sym not in long_dates or ts > long_dates[sym]:
                        long_dates[sym] = ts
                elif side == "sell_short":
                    if sym not in short_dates or ts > short_dates[sym]:
                        short_dates[sym] = ts
        except Exception:
            pass
    return long_dates, short_dates


def _generate_signals(model, symbols: list[str]) -> dict[str, float]:
    """Return {symbol: proba_up} for each symbol using predict_proba."""
    from pipeline.features import build_features_for_symbol

    signals: dict[str, float] = {}
    for sym in symbols:
        try:
            # drop_unlabeled=False keeps the label-horizon tail rows so
            # iloc[-1] is today's bar, not a 5-day-stale one
            feats = build_features_for_symbol(sym, drop_unlabeled=False)
            if feats.empty:
                log.warning("  %s: empty features, skipping", sym)
                continue
            latest = feats.iloc[[-1]]
            proba = float(model.predict_proba(latest)[0][1])
            signals[sym] = proba
            log.info("  %s  proba_up=%.3f", sym, proba)
        except Exception as e:
            log.warning("  %s signal failed: %s", sym, e)
    return signals


# ---------------------------------------------------------------------------
# Rebalance logic
# ---------------------------------------------------------------------------

def rebalance(client, model, dry_run: bool = False) -> list[dict]:
    """Run one rebalance cycle. Returns list of trade dicts for logging."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    portfolio_value = _portfolio_value(client)
    buying_power = _buying_power(client)
    log.info("Portfolio value: $%.2f  |  Buying power: $%.2f", portfolio_value, buying_power)

    current_positions = _current_positions(client)
    log.info("Open positions: %s", list(current_positions.keys()) or "none")

    signals = _generate_signals(model, UNIVERSE)
    long_entry_dates, short_entry_dates = _entry_dates()
    today = pd.Timestamp.now().normalize()
    trades = []
    max_alloc = portfolio_value * MAX_POSITION_PCT

    def _close(sym, pos, reason, close_side):
        price = _latest_price(sym) or pos["avg_entry_price"]
        entry = pos["avg_entry_price"]
        is_short = pos["side"] == "short"
        pnl_pct = (entry - price) / entry if is_short else (price - entry) / entry
        log.info("Close %s [%s] PnL=%.2f%% — %s", sym, pos["side"], pnl_pct * 100, reason)
        if not dry_run:
            try:
                client.close_position(sym)
                trades.append({
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "symbol": sym, "side": close_side, "qty": pos["qty"],
                    "price": price, "signal_proba": signals.get(sym, 0.0),
                    "pnl_pct": f"{pnl_pct:.4f}", "exit_reason": reason,
                })
            except Exception as e:
                log.warning("  Close failed for %s: %s", sym, e)
        else:
            log.info("  [DRY-RUN] Would close %s (%s)", sym, reason)

    # ---- Exit existing positions ----
    for sym, pos in current_positions.items():
        price = _latest_price(sym)
        if not price or pos["avg_entry_price"] <= 0:
            continue
        entry = pos["avg_entry_price"]
        is_short = pos["side"] == "short"
        pnl_pct = (entry - price) / entry if is_short else (price - entry) / entry
        proba = signals.get(sym, 0.0)

        if is_short:
            entry_ts = short_entry_dates.get(sym)
            days_held = (today - entry_ts.normalize()).days if entry_ts else 0
            # Stop-loss: price rose above entry by STOP_LOSS_PCT (adverse for short)
            if pnl_pct <= STOP_LOSS_PCT:
                _close(sym, pos, f"stop_loss (PnL={pnl_pct:.2%})", "buy_to_cover")
            elif days_held >= MAX_HOLD_DAYS:
                _close(sym, pos, f"max_hold ({days_held}d)", "buy_to_cover")
            elif proba >= COVER_THRESHOLD:
                _close(sym, pos, f"signal_recovered (proba={proba:.3f})", "buy_to_cover")
        else:
            entry_ts = long_entry_dates.get(sym)
            days_held = (today - entry_ts.normalize()).days if entry_ts else 0
            if pnl_pct <= STOP_LOSS_PCT:
                _close(sym, pos, f"stop_loss (PnL={pnl_pct:.2%})", "sell")
            elif days_held >= MAX_HOLD_DAYS:
                _close(sym, pos, f"max_hold ({days_held}d)", "sell")
            elif proba < SELL_THRESHOLD:
                _close(sym, pos, f"signal_fade (proba={proba:.3f})", "sell")

    # Refresh positions after exits
    current_positions = _current_positions(client)

    # ---- Long entries ----
    all_long = sorted(
        [(sym, p) for sym, p in signals.items() if p > BUY_THRESHOLD],
        key=lambda x: x[1], reverse=True,
    )
    sector_counts: dict[str, int] = {}
    long_candidates: list[tuple[str, float]] = []
    for sym, p in all_long:
        if len(long_candidates) >= TOP_N:
            break
        sector = _SYM_TO_SECTOR.get(sym, "Other")
        if sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
            continue
        long_candidates.append((sym, p))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    log.info("Long candidates: %d", len(long_candidates))
    for sym, proba in long_candidates:
        if sym in current_positions:
            log.info("  %s already held — skipping", sym)
            continue
        price = _latest_price(sym)
        if not price or price <= 0:
            continue
        qty = int(max_alloc / price)
        if qty < 1:
            continue
        log.info("BUY %d x %s @ ~$%.2f (proba=%.3f)", qty, sym, price, proba)
        if not dry_run:
            try:
                client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=qty,
                    side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                ))
                trades.append({"timestamp": datetime.now(tz=timezone.utc).isoformat(),
                                "symbol": sym, "side": "buy", "qty": qty,
                                "price": price, "signal_proba": proba,
                                "pnl_pct": "", "exit_reason": ""})
            except Exception as e:
                log.warning("  Buy failed for %s: %s", sym, e)
        else:
            log.info("  [DRY-RUN] Would buy %d x %s", qty, sym)
            trades.append({"timestamp": datetime.now(tz=timezone.utc).isoformat(),
                           "symbol": sym, "side": "buy_dry_run", "qty": qty,
                           "price": price, "signal_proba": proba,
                           "pnl_pct": "", "exit_reason": ""})

    # ---- Short entries ----
    all_short = sorted(
        [(sym, p) for sym, p in signals.items() if p < SHORT_THRESHOLD],
        key=lambda x: x[1],  # lowest proba first = highest conviction
    )
    short_sector_counts: dict[str, int] = {}
    short_candidates: list[tuple[str, float]] = []
    for sym, p in all_short:
        if len(short_candidates) >= TOP_N:
            break
        sector = _SYM_TO_SECTOR.get(sym, "Other")
        if short_sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
            continue
        short_candidates.append((sym, p))
        short_sector_counts[sector] = short_sector_counts.get(sector, 0) + 1

    log.info("Short candidates: %d", len(short_candidates))
    for sym, proba in short_candidates:
        if sym in current_positions:
            log.info("  %s already held — skipping short", sym)
            continue
        price = _latest_price(sym)
        if not price or price <= 0:
            continue
        qty = int(max_alloc / price)
        if qty < 1:
            continue
        log.info("SHORT %d x %s @ ~$%.2f (proba=%.3f)", qty, sym, price, proba)
        if not dry_run:
            try:
                client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=qty,
                    side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                ))
                trades.append({"timestamp": datetime.now(tz=timezone.utc).isoformat(),
                                "symbol": sym, "side": "sell_short", "qty": qty,
                                "price": price, "signal_proba": proba,
                                "pnl_pct": "", "exit_reason": ""})
            except Exception as e:
                log.warning("  Short failed for %s: %s", sym, e)
        else:
            log.info("  [DRY-RUN] Would short %d x %s", qty, sym)
            trades.append({"timestamp": datetime.now(tz=timezone.utc).isoformat(),
                           "symbol": sym, "side": "sell_short_dry_run", "qty": qty,
                           "price": price, "signal_proba": proba,
                           "pnl_pct": "", "exit_reason": ""})

    return trades


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca paper-trading bot")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Connect to Alpaca, print state, run signals but skip order submission",
    )
    args = parser.parse_args()

    model_path = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python run.py train <symbol>` first."
        )

    from pipeline.model import StockEnsemble
    log.info("Loading model from %s", model_path)
    model = StockEnsemble.load(model_path)

    client = _get_client()
    log.info("Connected to Alpaca paper API")

    if args.dry_run:
        log.info("=== DRY-RUN: one signal cycle, no orders placed ===")
        rebalance(client, model, dry_run=True)
        log.info("=== DRY-RUN complete ===")
    else:
        import time
        log.info("Starting daily rebalance loop (Ctrl-C to stop)...")
        while True:
            now = datetime.now(tz=timezone.utc)
            if now.weekday() < 5 and now.hour == 14 and now.minute == 35:
                try:
                    clock = client.get_clock()
                except Exception as e:
                    log.warning("Could not fetch market clock: %s — skipping rebalance", e)
                    time.sleep(60)
                    continue
                if not clock.is_open:
                    log.info("Market closed on %s (holiday?) — skipping rebalance", now.date())
                    time.sleep(60)
                    continue
                log.info("--- 9:35 ET rebalance %s ---", now.date())
                try:
                    rebalance(client, model, dry_run=False)
                except Exception as e:
                    log.error("Rebalance error: %s", e)
                time.sleep(60)
            else:
                time.sleep(30)


if __name__ == "__main__":
    main()
