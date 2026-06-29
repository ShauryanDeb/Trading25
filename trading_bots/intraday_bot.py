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

BUY_THRESHOLD = 0.55      # raised from 0.30 — only high-conviction longs
SHORT_THRESHOLD = 0.25    # proba below this => high-conviction short
# ATR-based exits: profit target = ATR_MULT_TARGET * ATR / price,
#                  stop-loss    = ATR_MULT_STOP    * ATR / price
ATR_MULT_TARGET = 2.0     # take profit at 2x the 14-bar ATR
ATR_MULT_STOP = 1.0       # cut loss at 1x the 14-bar ATR
# Floor stops prevent getting clipped by bid-ask spread on low-volatility large-caps.
MIN_STOP_PCT = 0.006      # never stop out on less than 0.6% adverse move
MIN_TARGET_PCT = 0.012    # never take profit on less than 1.2% gain (maintains 2:1 R/R)
VIX_MAX = 25.0            # suspend trading when VIX exceeds this level
MIN_HOLD_BARS = 2         # don't check exits until 2 ticks (~10 min) after entry
ENTRY_START_HOUR = 10     # only open new positions after 10:00 AM ET
ENTRY_END_HOUR = 15       # stop opening new positions at 3:00 PM ET
MAX_POSITIONS = 10        # max concurrent long holdings
MAX_SHORT_POSITIONS = 5   # max concurrent short holdings
POSITION_PCT = 0.03       # 3% of portfolio per position
HARD_CLOSE_HOUR = 15      # 3 PM ET
HARD_CLOSE_MINUTE = 55    # 3:55 PM ET
SPY_BARS_DOWN = 3         # consecutive negative SPY bars that block new long entries


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


def _get_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    key = os.environ.get("APCA_API_KEY_ID_INTRADAY")
    secret = os.environ.get("APCA_API_SECRET_KEY_INTRADAY")
    return StockHistoricalDataClient(key, secret)


def _current_vix() -> float:
    """Return latest VIX close via yfinance."""
    try:
        import yfinance as yf
        hist = yf.Ticker("^VIX").history(period="2d")
        return float(hist["Close"].iloc[-1])
    except Exception:
        return 0.0


def _live_ask(data_client, symbol: str) -> float | None:
    """Return current ask price from Alpaca real-time quote."""
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_stock_latest_quote(req)
        ask = float(quote[symbol].ask_price)
        return ask if ask > 0 else None
    except Exception:
        return None


def _spy_trending_down() -> bool:
    """Return True if the last SPY_BARS_DOWN 5-min bars all closed lower than the prior bar.

    A streak of consecutive down bars means the broad market is in an intraday
    downtrend. Opening new long positions into that tape results in systematic
    stop-outs — blocking entries during these streaks avoids buying falling knives.
    Returns False (allow entries) on any data error so the bot degrades gracefully.
    """
    try:
        from pipeline.intraday_data import fetch_latest_bars
        bars = fetch_latest_bars("SPY", n=SPY_BARS_DOWN + 2)
        if len(bars) < SPY_BARS_DOWN + 1:
            return False
        closes = bars["Close"].iloc[-(SPY_BARS_DOWN + 1):]
        changes = closes.diff().dropna()
        down_streak = all(c < 0 for c in changes.iloc[-SPY_BARS_DOWN:])
        if down_streak:
            log.info("SPY down %d consecutive bars — blocking new entries", SPY_BARS_DOWN)
        return down_streak
    except Exception as e:
        log.debug("SPY direction check failed: %s", e)
        return False


def _spy_trending_up() -> bool:
    """Return True if the last SPY_BARS_DOWN 5-min bars all closed higher.

    Used to block new short entries when the broad market is rallying —
    shorting into an up-tape produces systematic stop-outs.
    Returns False (allow short entries) on any data error.
    """
    try:
        from pipeline.intraday_data import fetch_latest_bars
        bars = fetch_latest_bars("SPY", n=SPY_BARS_DOWN + 2)
        if len(bars) < SPY_BARS_DOWN + 1:
            return False
        closes = bars["Close"].iloc[-(SPY_BARS_DOWN + 1):]
        changes = closes.diff().dropna()
        up_streak = all(c > 0 for c in changes.iloc[-SPY_BARS_DOWN:])
        if up_streak:
            log.info("SPY up %d consecutive bars — blocking new short entries", SPY_BARS_DOWN)
        return up_streak
    except Exception as e:
        log.debug("SPY direction check failed: %s", e)
        return False


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

def _score_symbol(model, symbol: str) -> tuple[float, float] | None:
    """Return (proba_up, atr) for a single symbol using latest 5-min bars.

    ATR is the raw 14-bar average true range in dollars, used for exit sizing.
    Returns None if data is insufficient.
    """
    from pipeline.intraday_data import fetch_latest_bars
    from pipeline.intraday_features import build_intraday_features, _atr

    try:
        bars = fetch_latest_bars(symbol, n=60)
        if len(bars) < 30:
            return None
        feats = build_intraday_features(bars)
        if feats.empty:
            return None
        latest = feats.iloc[[-1]]
        proba = float(model.predict_proba(latest)[0][1])
        atr = float(latest["ATR_14"].iloc[0]) if "ATR_14" in latest.columns else 0.0
        return proba, atr
    except Exception as e:
        log.debug("  %s score failed: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# Position monitoring
# ---------------------------------------------------------------------------

def _get_positions(client) -> dict[str, dict]:
    result = {}
    for p in client.get_all_positions():
        side = str(p.side).lower()
        if "short" in side:
            side = "short"
        else:
            side = "long"
        result[p.symbol] = {
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "current_price": float(p.current_price),
            "side": side,
        }
    return result


def _check_exits(client, positions: dict, dry_run: bool,
                 stopped_out: set | None = None) -> set:
    """Check ATR-scaled profit target and stop-loss on eligible positions.

    Handles both long and short positions — PnL direction is inverted for shorts.
    Returns set of symbols that hit stop-loss (to add to session cooldown).
    """
    newly_stopped: set[str] = set()
    for sym, pos in positions.items():
        entry = pos["avg_entry_price"]
        price = pos["current_price"]
        is_short = pos.get("side") == "short"

        # For shorts, profit = price falling below entry; loss = price rising above entry
        pnl_pct = (entry - price) / entry if is_short else (price - entry) / entry

        atr = pos.get("entry_atr", 0.0)
        if atr > 0 and entry > 0:
            profit_target = max(ATR_MULT_TARGET * atr / entry, MIN_TARGET_PCT)
            stop_loss = -max(ATR_MULT_STOP * atr / entry, MIN_STOP_PCT)
        else:
            profit_target = MIN_TARGET_PCT
            stop_loss = -MIN_STOP_PCT

        reason = None
        if pnl_pct >= profit_target:
            reason = "profit_target"
        elif pnl_pct <= stop_loss:
            reason = "stop_loss"

        if reason:
            side_label = "short" if is_short else "long"
            log.info("EXIT %s [%s] @ $%.2f  PnL=%.2f%%  reason=%s",
                     sym, side_label, price, pnl_pct * 100, reason)
            if not dry_run:
                try:
                    client.close_position(sym)
                    close_side = "buy_to_cover" if is_short else "sell"
                    _log_trade({
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        "symbol": sym, "side": close_side, "qty": pos["qty"],
                        "price": price, "signal_proba": "", "pnl_pct": f"{pnl_pct:.4f}",
                        "exit_reason": reason,
                    })
                    if reason == "stop_loss":
                        newly_stopped.add(sym)
                        log.info("  %s cooled off for rest of session", sym)
                except Exception as e:
                    log.warning("  Close failed for %s: %s", sym, e)
            else:
                log.info("  [DRY-RUN] Would close %s (%s)", sym, reason)
    return newly_stopped


def _hard_close_all(client, positions: dict, dry_run: bool) -> None:
    """Force-close all positions at 3:55 PM ET."""
    log.info("HARD CLOSE: closing all %d positions before market close", len(positions))
    for sym, pos in positions.items():
        price = pos["current_price"]
        entry = pos["avg_entry_price"]
        is_short = pos.get("side") == "short"
        pnl_pct = (entry - price) / entry if is_short else (price - entry) / entry
        log.info("  Close %s [%s] @ $%.2f  PnL=%.2f%%",
                 sym, "short" if is_short else "long", price, pnl_pct * 100)
        if not dry_run:
            try:
                client.close_position(sym)
                close_side = "buy_to_cover" if is_short else "sell"
                _log_trade({
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "symbol": sym, "side": close_side, "qty": pos["qty"],
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

def _market_is_open(client) -> bool:
    """Return True only when Alpaca's clock says the market is currently open."""
    try:
        clock = client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        log.warning("Could not fetch market clock: %s — assuming closed", e)
        return False


def run_session(dry_run: bool = False) -> None:
    """Run one full intraday session (9:35 AM to 3:55 PM ET)."""
    from pipeline.intraday_model import IntradayEnsemble
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    log.info("Loading intraday model from %s", MODEL_PATH)
    model = IntradayEnsemble.load(MODEL_PATH)
    client = _get_client()
    data_client = _get_data_client()

    # Market-open guard — skip entirely on holidays and weekends
    if not _market_is_open(client):
        log.info("Market is closed today — session cancelled")
        return

    # VIX regime check — suspend session if volatility is too high
    vix = _current_vix()
    if vix > VIX_MAX:
        log.warning("VIX=%.1f > %.0f — high-volatility regime, session suspended", vix, VIX_MAX)
        return
    log.info("VIX=%.1f — within normal range, proceeding", vix)

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    log.info("Intraday account: $%.2f portfolio  |  $%.2f buying power",
             portfolio_value, float(account.buying_power))

    max_alloc = portfolio_value * POSITION_PCT
    stopped_out: set[str] = set()         # cooled-off after long stop-loss
    short_stopped_out: set[str] = set()   # cooled-off after short stop-loss
    entry_tick: dict[str, int] = {}       # symbol -> tick at long entry
    entry_atr: dict[str, float] = {}      # symbol -> ATR at long entry
    short_entry_tick: dict[str, int] = {} # symbol -> tick at short entry
    short_entry_atr: dict[str, float] = {}# symbol -> ATR at short entry
    tick = 0

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
        tick += 1

        # Check exits — attach stored ATR, split long/short for min-hold tracking
        positions = _get_positions(client)
        for sym, pos in positions.items():
            if pos["side"] == "short":
                pos["entry_atr"] = short_entry_atr.get(sym, 0.0)
                eligible_tick = short_entry_tick.get(sym, 0)
            else:
                pos["entry_atr"] = entry_atr.get(sym, 0.0)
                eligible_tick = entry_tick.get(sym, 0)
            pos["_eligible"] = (tick - eligible_tick) >= MIN_HOLD_BARS

        if positions:
            eligible = {s: p for s, p in positions.items() if p.get("_eligible")}
            if eligible:
                exited = _check_exits(client, eligible, dry_run)
                # Route stop-outs to the correct cooldown set by side
                for sym in exited:
                    if positions[sym]["side"] == "short":
                        short_stopped_out.add(sym)
                    else:
                        stopped_out.add(sym)
                    log.info("  %s cooled off for rest of session", sym)
            positions = _get_positions(client)  # refresh after exits

        long_positions = {s: p for s, p in positions.items() if p["side"] == "long"}
        short_positions = {s: p for s, p in positions.items() if p["side"] == "short"}

        entry_allowed = ENTRY_START_HOUR <= now.hour < ENTRY_END_HOUR
        if not entry_allowed:
            log.info("Outside entry window (%d:00–%d:00 ET) — monitoring exits only",
                     ENTRY_START_HOUR, ENTRY_END_HOUR)
            time.sleep(300)
            continue

        spy_down = _spy_trending_down()
        spy_up = _spy_trending_up()

        # ---- Long entries ----
        if len(long_positions) < MAX_POSITIONS and not spy_down:
            slots = MAX_POSITIONS - len(long_positions)
            long_candidates = []
            for sym in UNIVERSE:
                if sym in positions or sym in stopped_out:
                    continue
                result = _score_symbol(model, sym)
                if result is not None:
                    proba, atr = result
                    if proba > BUY_THRESHOLD:
                        long_candidates.append((sym, proba, atr))

            long_candidates.sort(key=lambda x: x[1], reverse=True)
            for sym, proba, atr in long_candidates[:slots]:
                price = _live_ask(data_client, sym)
                if not price or price <= 0:
                    continue
                qty = int(max_alloc / price)
                if qty < 1:
                    continue
                log.info("BUY %d x %s @ $%.2f  proba=%.3f", qty, sym, price, proba)
                if not dry_run:
                    try:
                        client.submit_order(MarketOrderRequest(
                            symbol=sym, qty=qty,
                            side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                        ))
                        entry_tick[sym] = tick
                        entry_atr[sym] = atr
                        _log_trade({"timestamp": datetime.now(tz=timezone.utc).isoformat(),
                                    "symbol": sym, "side": "buy", "qty": qty,
                                    "price": price, "signal_proba": proba,
                                    "pnl_pct": "", "exit_reason": ""})
                    except Exception as e:
                        log.warning("  Long order failed for %s: %s", sym, e)
                else:
                    log.info("  [DRY-RUN] Would buy %d x %s", qty, sym)
        elif spy_down:
            log.info("SPY downtrend — skipping new long entries this tick")

        # ---- Short entries ----
        if len(short_positions) < MAX_SHORT_POSITIONS and not spy_up:
            short_slots = MAX_SHORT_POSITIONS - len(short_positions)
            short_candidates = []
            for sym in UNIVERSE:
                if sym in positions or sym in short_stopped_out:
                    continue
                result = _score_symbol(model, sym)
                if result is not None:
                    proba, atr = result
                    if proba < SHORT_THRESHOLD:
                        short_candidates.append((sym, proba, atr))

            # Rank by lowest proba first (highest conviction shorts)
            short_candidates.sort(key=lambda x: x[1])
            for sym, proba, atr in short_candidates[:short_slots]:
                price = _live_ask(data_client, sym)
                if not price or price <= 0:
                    continue
                qty = int(max_alloc / price)
                if qty < 1:
                    continue
                log.info("SHORT %d x %s @ $%.2f  proba=%.3f", qty, sym, price, proba)
                if not dry_run:
                    try:
                        client.submit_order(MarketOrderRequest(
                            symbol=sym, qty=qty,
                            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                        ))
                        short_entry_tick[sym] = tick
                        short_entry_atr[sym] = atr
                        _log_trade({"timestamp": datetime.now(tz=timezone.utc).isoformat(),
                                    "symbol": sym, "side": "sell_short", "qty": qty,
                                    "price": price, "signal_proba": proba,
                                    "pnl_pct": "", "exit_reason": ""})
                    except Exception as e:
                        log.warning("  Short order failed for %s: %s", sym, e)
                else:
                    log.info("  [DRY-RUN] Would short %d x %s", qty, sym)
        elif spy_up:
            log.info("SPY uptrend — skipping new short entries this tick")

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

    combined = pd.concat(frames).sort_index().reset_index(drop=True)
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
