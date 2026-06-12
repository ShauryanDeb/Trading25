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
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "HES",
    # Other large-caps
    "PEP", "KO", "PG", "MO", "PM", "DIS", "NFLX", "T", "VZ", "CMCSA",
]

MAX_POSITION_PCT = 0.05   # 5% of portfolio per symbol
TOP_N = 20                # max positions held at once (rank-and-cap)
BUY_THRESHOLD = 0.55      # predict_proba > this => BUY candidate
SELL_THRESHOLD = 0.45     # predict_proba < this => SELL/SKIP
STOP_LOSS_PCT = -0.07     # -7% from entry => stop-loss


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
    """Return {symbol: {market_value, avg_entry_price, qty}}."""
    result = {}
    for p in client.get_all_positions():
        result[p.symbol] = {
            "market_value": float(p.market_value),
            "avg_entry_price": float(p.avg_entry_price),
            "qty": float(p.qty),
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


def _generate_signals(model, symbols: list[str]) -> dict[str, float]:
    """Return {symbol: proba_up} for each symbol using predict_proba."""
    from pipeline.features import build_features_for_symbol

    signals: dict[str, float] = {}
    for sym in symbols:
        try:
            feats = build_features_for_symbol(sym)
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
    trades = []

    # Check stop-losses on existing positions
    for sym, pos in current_positions.items():
        price = _latest_price(sym)
        if price and pos["avg_entry_price"] > 0:
            pnl_pct = (price - pos["avg_entry_price"]) / pos["avg_entry_price"]
            if pnl_pct <= STOP_LOSS_PCT:
                log.info("STOP-LOSS triggered for %s (PnL=%.2f%%)", sym, pnl_pct * 100)
                if not dry_run:
                    try:
                        client.close_position(sym)
                        trades.append({
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                            "symbol": sym, "side": "sell", "qty": pos["qty"],
                            "price": price, "signal_proba": signals.get(sym, 0.0),
                        })
                    except Exception as e:
                        log.warning("  Stop-loss close failed for %s: %s", sym, e)
                else:
                    log.info("  [DRY-RUN] Would close %s (stop-loss)", sym)

    # Close positions where signal dropped below sell threshold
    for sym, pos in current_positions.items():
        proba = signals.get(sym, 0.0)
        if proba < SELL_THRESHOLD:
            log.info("Closing %s (proba=%.3f < %.2f)", sym, proba, SELL_THRESHOLD)
            if not dry_run:
                try:
                    client.close_position(sym)
                    price = _latest_price(sym) or pos["avg_entry_price"]
                    trades.append({
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        "symbol": sym, "side": "sell", "qty": pos["qty"],
                        "price": price, "signal_proba": proba,
                    })
                except Exception as e:
                    log.warning("  Failed to close %s: %s", sym, e)
            else:
                log.info("  [DRY-RUN] Would close %s", sym)

    # Rank buy candidates by proba, cap at TOP_N
    buy_candidates = sorted(
        [(sym, p) for sym, p in signals.items() if p > BUY_THRESHOLD],
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_N]
    log.info(
        "Buy candidates: %d above threshold, keeping top %d",
        len([p for p in signals.values() if p > BUY_THRESHOLD]),
        len(buy_candidates),
    )

    # Open positions for ranked buy signals
    max_alloc = portfolio_value * MAX_POSITION_PCT
    for sym, proba in buy_candidates:
        current_val = current_positions.get(sym, {}).get("market_value", 0.0)
        if current_val >= max_alloc * 0.95:
            log.info("  %s already fully allocated", sym)
            continue
        price = _latest_price(sym)
        if not price or price <= 0:
            log.warning("  No price for %s, skipping", sym)
            continue
        target_qty = int(max_alloc / price)
        if target_qty < 1:
            continue
        log.info("BUY %d x %s @ ~$%.2f (proba=%.3f)", target_qty, sym, price, proba)
        if not dry_run:
            try:
                order = MarketOrderRequest(
                    symbol=sym,
                    qty=target_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                client.submit_order(order)
                trades.append({
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "symbol": sym, "side": "buy", "qty": target_qty,
                    "price": price, "signal_proba": proba,
                })
            except Exception as e:
                log.warning("  Order failed for %s: %s", sym, e)
        else:
            log.info("  [DRY-RUN] Would buy %d x %s", target_qty, sym)
            trades.append({
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "symbol": sym, "side": "buy_dry_run", "qty": target_qty,
                "price": price, "signal_proba": proba,
            })

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
