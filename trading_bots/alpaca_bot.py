"""Alpaca paper-trading bot using alpaca-py SDK.

Loads a trained StockEnsemble model, runs daily signals for the top-20
S&P 500 names, and rebalances at market open — long-only, 5% max position.

Environment (.env):
    APCA_API_KEY_ID     — Alpaca paper key
    APCA_API_SECRET_KEY — Alpaca paper secret
    MODEL_PATH          — path to saved StockEnsemble .pkl  (default: models/model.pkl)
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from pipeline.features import build_features_for_symbol
from pipeline.model import StockEnsemble

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
TOP20_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK-B", "JPM", "UNH",
    "V", "XOM", "LLY", "JNJ", "AVGO",
    "MA", "HD", "PG", "MRK", "COST",
]

MAX_POSITION_PCT = 0.05  # 5% of portfolio per symbol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client() -> TradingClient:
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    return TradingClient(key, secret, paper=True)


def _portfolio_value(client: TradingClient) -> float:
    account = client.get_account()
    return float(account.portfolio_value)


def _current_positions(client: TradingClient) -> dict[str, float]:
    """Return {symbol: market_value} for open positions."""
    positions = client.get_all_positions()
    return {p.symbol: float(p.market_value) for p in positions}


def _latest_price(symbol: str) -> float | None:
    """Pull yesterday's close from yfinance as a proxy for current price."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def _generate_signals(model: StockEnsemble, symbols: list[str]) -> dict[str, int]:
    """Return {symbol: signal} where signal ∈ {0, 1}."""
    signals: dict[str, int] = {}
    for sym in symbols:
        try:
            feats = build_features_for_symbol(sym)
            if feats.empty:
                continue
            latest = feats.iloc[[-1]]
            signal = int(model.predict(latest)[0])
            signals[sym] = signal
            log.info("  %s → signal=%d", sym, signal)
        except Exception as e:
            log.warning("  %s signal failed: %s", sym, e)
    return signals


# ---------------------------------------------------------------------------
# Rebalance logic
# ---------------------------------------------------------------------------

def rebalance(client: TradingClient, model: StockEnsemble) -> None:
    portfolio_value = _portfolio_value(client)
    log.info("Portfolio value: $%.2f", portfolio_value)

    current_positions = _current_positions(client)
    signals = _generate_signals(model, TOP20_SP500)

    # Symbols to be long (signal=1)
    long_symbols = [s for s, sig in signals.items() if sig == 1]
    max_alloc = portfolio_value * MAX_POSITION_PCT

    # Close positions for symbols with signal=0
    for sym, mkt_val in current_positions.items():
        if sym not in signals or signals[sym] == 0:
            log.info("Closing position: %s ($%.2f)", sym, mkt_val)
            try:
                client.close_position(sym)
            except Exception as e:
                log.warning("  Failed to close %s: %s", sym, e)

    # Open / top-up positions for long signals
    for sym in long_symbols:
        current_val = current_positions.get(sym, 0.0)
        if current_val >= max_alloc * 0.95:  # already fully allocated
            continue
        price = _latest_price(sym)
        if price is None or price <= 0:
            log.warning("  No price for %s, skipping", sym)
            continue
        target_qty = int(max_alloc / price)
        if target_qty < 1:
            continue
        log.info("Buying %d shares of %s @ ~$%.2f", target_qty, sym, price)
        try:
            order = MarketOrderRequest(
                symbol=sym,
                qty=target_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(order)
        except Exception as e:
            log.warning("  Order failed for %s: %s", sym, e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    model_path = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python run.py train <symbol>` first."
        )

    log.info("Loading model from %s", model_path)
    model = StockEnsemble.load(model_path)
    client = _get_client()

    log.info("Starting daily rebalance loop (Ctrl-C to stop)...")
    while True:
        now = datetime.now(tz=timezone.utc)
        # Run once at ~09:31 ET (13:31 UTC) on weekdays
        if now.weekday() < 5 and now.hour == 13 and now.minute == 31:
            log.info("--- Market open rebalance %s ---", now.date())
            try:
                rebalance(client, model)
            except Exception as e:
                log.error("Rebalance error: %s", e)
            time.sleep(60)  # avoid double-fire within the same minute
        else:
            time.sleep(30)


if __name__ == "__main__":
    main()
