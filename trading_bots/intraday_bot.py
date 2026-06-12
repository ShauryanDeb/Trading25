"""Intraday momentum scalping bot — 5-minute bars, separate Alpaca paper account.

Strategy:
  - Re-scores all universe symbols every 5 minutes during market hours
  - BUY when proba_up > 0.60 and not already in position
  - EXIT when: +0.75% profit target, -0.35% stop-loss, or 3:55 PM ET hard close
  - Max 10 concurrent positions at 3% each (long-only, no overnight holds)

Environment (.env):
  APCA_API_KEY_ID_INTRADAY     -- second paper account key
  APCA_API_SECRET_KEY_INTRADAY -- second paper account secret
  MODEL_PATH_INTRADAY          -- path to intraday model pkl (default: models/intraday_model.pkl)

Usage:
  python trading_bots/intraday_bot.py              # run live during market hours
  python trading_bots/intraday_bot.py --dry-run    # signals only, no orders
  python trading_bots/intraday_bot.py --train      # train/retrain intraday model then exit
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
REPORTS_DIR = _ROOT / "reports"
MODEL_PATH = Path(os.getenv("MODEL_PATH_INTRADAY", str(_ROOT / "models" / "intraday_model.pkl")))

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "SNPS",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA",
    "COF", "USB", "PNC", "TFC", "CME", "ICE", "CB", "PGR", "MET", "AIG",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "ISRG", "VRTX", "REGN", "ZTS", "SYK", "BSX", "MDT", "ELV", "CVS",
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW", "TJX", "BKNG",
    "MAR", "HLT", "YUM", "DPZ", "CMG", "ORLY", "AZO", "TSCO", "DG", "DLTR",
    "CAT", "DE", "HON", "GE", "RTX", "LMT", "BA", "UPS", "FDX", "NSC",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "DVN",
]

BUY_THRESHOLD = 0.60      # tighter than daily — intraday is noisier
PROFIT_TARGET = 0.0075    # +0.75% exit
STOP_LOSS = -0.0035       # -0.35% exit
MAX_POSITIONS = 10        # max concurrent holdings
POSITION_PCT = 0.03       # 3% of portfolio per position
HARD_CLOSE_HOUR = 15      # 3 PM ET
HARD_CLOSE_MINUTE = 55    # 3:55 PM ET


# ---------------------------------------------------------------------------
# Alpaca client
# ---------------------------------------------------------------------------

def _get_client():
    from alpaca.trading.client import TradingClient
    key = os.environ.get("APCA_API_KEY_ID_INTRADAY")
    secret = os.environ.get("APCA_API_SECRET_KEY_INTRADAY")
    if not key or not secret:
        raise EnvironmentError("APCA_API_KEY_ID_INTRADAY / APCA_API_SECRET_KEY_INTRADAY not set in .env")
    return TradingClient(key, secret, paper=True)


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

def _log_trade(trade: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(tz=ET).strftime("%Y%m%d")
    out_path = REPORTS_DIR / f"intraday_trades_{date_str}.csv"
    fieldnames = ["timestamp", "symbol", "side", "qty", "price", "signal_proba", "pnl_pct", "exit_reason"]
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: trade.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _score_symbol(model, symbol: str) -> float | None:
    """Return proba_up for a single symbol using latest 5-min bars."""
    from pipeline.intraday_data import fetch_latest_bars
    from pipeline.intraday_features import build_intraday_features

    try:
        bars = fetch_latest_bars(symbol, n=60)
        if len(bars) < 30:
            return None
        feats = build_intraday_features(bars)
        if feats.empty:
            return None
        latest = feats.iloc[[-1]]
        return float(model.predict_proba(latest)[0][1])
    except Exception as e:
        log.debug("  %s score failed: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# Position monitoring
# ---------------------------------------------------------------------------

def _get_positions(client) -> dict[str, dict]:
    result = {}
    for p in client.get_all_positions():
        result[p.symbol] = {
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "current_price": float(p.current_price),
        }
    return result


def _check_exits(client, positions: dict, dry_run: bool) -> None:
    """Check profit target and stop-loss on all open positions."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    for sym, pos in positions.items():
        entry = pos["avg_entry_price"]
        price = pos["current_price"]
        pnl_pct = (price - entry) / entry

        reason = None
        if pnl_pct >= PROFIT_TARGET:
            reason = "profit_target"
        elif pnl_pct <= STOP_LOSS:
            reason = "stop_loss"

        if reason:
            log.info("EXIT %s @ $%.2f  PnL=%.2f%%  reason=%s", sym, price, pnl_pct * 100, reason)
            if not dry_run:
                try:
                    client.close_position(sym)
                    _log_trade({
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        "symbol": sym, "side": "sell", "qty": pos["qty"],
                        "price": price, "signal_proba": "", "pnl_pct": f"{pnl_pct:.4f}",
                        "exit_reason": reason,
                    })
                except Exception as e:
                    log.warning("  Close failed for %s: %s", sym, e)
            else:
                log.info("  [DRY-RUN] Would close %s (%s)", sym, reason)


def _hard_close_all(client, positions: dict, dry_run: bool) -> None:
    """Force-close all positions at 3:55 PM ET."""
    log.info("HARD CLOSE: closing all %d positions before market close", len(positions))
    for sym, pos in positions.items():
        price = pos["current_price"]
        entry = pos["avg_entry_price"]
        pnl_pct = (price - entry) / entry
        log.info("  Close %s @ $%.2f  PnL=%.2f%%", sym, price, pnl_pct * 100)
        if not dry_run:
            try:
                client.close_position(sym)
                _log_trade({
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "symbol": sym, "side": "sell", "qty": pos["qty"],
                    "price": price, "signal_proba": "", "pnl_pct": f"{pnl_pct:.4f}",
                    "exit_reason": "hard_close",
                })
            except Exception as e:
                log.warning("  Hard close failed for %s: %s", sym, e)
        else:
            log.info("  [DRY-RUN] Would hard-close %s", sym)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_session(dry_run: bool = False) -> None:
    """Run one full intraday session (9:35 AM to 3:55 PM ET)."""
    from pipeline.intraday_model import IntradayEnsemble
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    log.info("Loading intraday model from %s", MODEL_PATH)
    model = IntradayEnsemble.load(MODEL_PATH)
    client = _get_client()

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    log.info("Intraday account: $%.2f portfolio  |  $%.2f buying power",
             portfolio_value, float(account.buying_power))

    max_alloc = portfolio_value * POSITION_PCT

    while True:
        now = datetime.now(tz=ET)

        # Hard close at 3:55 PM ET
        if now.hour > HARD_CLOSE_HOUR or (now.hour == HARD_CLOSE_HOUR and now.minute >= HARD_CLOSE_MINUTE):
            positions = _get_positions(client)
            if positions:
                _hard_close_all(client, positions, dry_run)
            log.info("Session complete for %s", now.date())
            break

        # Only trade during market hours
        if now.hour < 9 or (now.hour == 9 and now.minute < 35):
            log.info("Waiting for market open (9:35 AM ET)...")
            time.sleep(60)
            continue

        # --- Every 5-minute tick ---
        log.info("--- Tick %s ---", now.strftime("%H:%M"))

        # Check exits on open positions first
        positions = _get_positions(client)
        if positions:
            _check_exits(client, positions, dry_run)
            positions = _get_positions(client)  # refresh after exits

        # Score universe for new entries
        if len(positions) < MAX_POSITIONS:
            slots = MAX_POSITIONS - len(positions)
            candidates = []
            for sym in UNIVERSE:
                if sym in positions:
                    continue
                proba = _score_symbol(model, sym)
                if proba is not None:
                    log.debug("  %s  proba=%.3f", sym, proba)
                    if proba > BUY_THRESHOLD:
                        candidates.append((sym, proba))

            # Rank by proba, take top available slots
            candidates.sort(key=lambda x: x[1], reverse=True)
            for sym, proba in candidates[:slots]:
                try:
                    from pipeline.intraday_data import fetch_latest_bars
                    bars = fetch_latest_bars(sym, n=2)
                    price = float(bars["Close"].iloc[-1]) if not bars.empty else None
                except Exception:
                    price = None

                if not price or price <= 0:
                    continue

                qty = int(max_alloc / price)
                if qty < 1:
                    continue

                log.info("BUY %d x %s @ ~$%.2f  proba=%.3f", qty, sym, price, proba)
                if not dry_run:
                    try:
                        order = MarketOrderRequest(
                            symbol=sym,
                            qty=qty,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                        )
                        client.submit_order(order)
                        _log_trade({
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                            "symbol": sym, "side": "buy", "qty": qty,
                            "price": price, "signal_proba": proba,
                            "pnl_pct": "", "exit_reason": "",
                        })
                    except Exception as e:
                        log.warning("  Order failed for %s: %s", sym, e)
                else:
                    log.info("  [DRY-RUN] Would buy %d x %s", qty, sym)
        else:
            log.info("Max positions (%d) reached — monitoring exits only", MAX_POSITIONS)

        # Sleep until next 5-min bar
        time.sleep(300)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_model(symbols: list[str] | None = None, days_back: int = 90) -> None:
    """Train IntradayEnsemble on 5-min bars from universe and save to MODEL_PATH."""
    import pandas as pd
    from pipeline.intraday_features import build_intraday_features_for_symbol
    from pipeline.intraday_model import IntradayEnsemble

    symbols = symbols or UNIVERSE[:20]  # default to first 20 for speed
    frames = []
    for sym in symbols:
        log.info("Fetching 5-min bars for %s...", sym)
        try:
            feats = build_intraday_features_for_symbol(sym, days_back=days_back)
            log.info("  %d samples", len(feats))
            frames.append(feats)
        except Exception as e:
            log.warning("  SKIP %s: %s", sym, e)

    if not frames:
        log.error("No data fetched — aborting training")
        return

    combined = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
    X = combined.drop(columns=["Target"])
    y = combined["Target"]
    log.info("Training on %d samples from %d symbols, %d features", len(combined), len(frames), X.shape[1])

    model = IntradayEnsemble()
    model.fit(X, y)
    model.save(MODEL_PATH)
    log.info("Intraday model saved -> %s", MODEL_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday momentum scalping bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run signals and log, but do not place any orders")
    parser.add_argument("--train", action="store_true",
                        help="Train the intraday model on all universe symbols then exit")
    parser.add_argument("--train-symbols", nargs="+", default=None,
                        help="Subset of symbols to train on (default: full universe)")
    parser.add_argument("--days-back", type=int, default=90,
                        help="Days of 5-min history to use for training")
    args = parser.parse_args()

    if args.train:
        train_model(symbols=args.train_symbols, days_back=args.days_back)
        return

    if not MODEL_PATH.exists():
        log.error("No intraday model found at %s. Run with --train first.", MODEL_PATH)
        sys.exit(1)

    now = datetime.now(tz=ET)
    if now.weekday() >= 5:
        log.warning("Today is a weekend — no market session. Exiting.")
        sys.exit(0)

    log.info("Starting intraday session  dry_run=%s", args.dry_run)
    run_session(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
