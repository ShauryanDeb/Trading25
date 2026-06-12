"""Daily paper-trading scheduler.

Runs the rebalance cycle at 9:35 AM ET on weekdays via APScheduler.
Writes a trade log to reports/trades_YYYYMMDD.csv after each cycle.

Usage:
    python trading_bots/scheduler.py              # start scheduler (blocks)
    python trading_bots/scheduler.py --test-now   # run one cycle immediately, then exit
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root on path
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
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_ROOT / "models" / "model.pkl")))


# ---------------------------------------------------------------------------
# Trade log writer
# ---------------------------------------------------------------------------

def _write_trade_log(trades: list[dict]) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(tz=ET).strftime("%Y%m%d")
    out_path = REPORTS_DIR / f"trades_{date_str}.csv"
    fieldnames = ["timestamp", "symbol", "side", "qty", "price", "signal_proba"]
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for trade in trades:
            writer.writerow({k: trade.get(k, "") for k in fieldnames})
    log.info("Trade log written -> %s  (%d rows)", out_path, len(trades))
    return out_path


# ---------------------------------------------------------------------------
# Portfolio state logger
# ---------------------------------------------------------------------------

def _log_portfolio(client) -> None:
    account = client.get_account()
    log.info(
        "Portfolio state: value=$%.2f  cash=$%.2f  buying_power=$%.2f",
        float(account.portfolio_value),
        float(account.cash),
        float(account.buying_power),
    )
    positions = client.get_all_positions()
    if positions:
        for p in positions:
            log.info(
                "  Position: %s  qty=%s  mkt_val=$%.2f  unrealized_pl=$%.2f",
                p.symbol, p.qty, float(p.market_value), float(p.unrealized_pl),
            )
    else:
        log.info("  No open positions")


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------

def run_cycle(dry_run: bool = False) -> None:
    """One full rebalance + log cycle."""
    from pipeline.model import StockEnsemble
    from trading_bots.alpaca_bot import _get_client, rebalance

    log.info("=== Rebalance cycle start (%s) ===", datetime.now(tz=ET).strftime("%Y-%m-%d %H:%M ET"))

    if not MODEL_PATH.exists():
        log.error("Model not found at %s — skipping cycle", MODEL_PATH)
        return

    model = StockEnsemble.load(MODEL_PATH)
    client = _get_client()

    trades = rebalance(client, model, dry_run=dry_run)
    _write_trade_log(trades)
    _log_portfolio(client)

    log.info("=== Rebalance cycle complete ===")


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-trading scheduler")
    parser.add_argument(
        "--test-now", action="store_true",
        help="Run one cycle immediately in dry-run mode (no orders placed) then exit",
    )
    parser.add_argument(
        "--run-now", action="store_true",
        help="Run one live cycle immediately (real paper orders) then exit",
    )
    args = parser.parse_args()

    if args.test_now:
        log.info("Scheduler running (test-now dry-run mode)")
        run_cycle(dry_run=True)
        return

    if args.run_now:
        log.info("Scheduler running (run-now LIVE mode)")
        run_cycle(dry_run=False)
        return

    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone=ET)
    scheduler.add_job(
        run_cycle,
        trigger="cron",
        day_of_week="mon-fri",
        hour=9,
        minute=35,
        kwargs={"dry_run": False},
        id="daily_rebalance",
        name="Daily 9:35 ET rebalance",
    )

    log.info("Scheduler running — daily rebalance job scheduled for 09:35 ET (Mon-Fri)")
    log.info("Press Ctrl-C to stop")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")


if __name__ == "__main__":
    main()
