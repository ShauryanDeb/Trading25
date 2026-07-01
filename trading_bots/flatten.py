"""Close all positions and cancel all open orders on one or both paper accounts.

Usage:
    python trading_bots/flatten.py            # flatten both accounts
    python trading_bots/flatten.py --swing    # swing account only
    python trading_bots/flatten.py --intraday # intraday account only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def flatten(client, label: str) -> None:
    positions = client.get_all_positions()
    account = client.get_account()
    log.info("[%s] portfolio=$%.2f  open positions: %d",
             label, float(account.portfolio_value), len(positions))
    for p in positions:
        log.info("  %s  qty=%s  side=%s  unrealized_pl=$%.2f",
                 p.symbol, p.qty, p.side, float(p.unrealized_pl))
    if not positions:
        log.info("[%s] nothing to close", label)
        return
    # cancel_orders=True also clears any queued DAY orders
    results = client.close_all_positions(cancel_orders=True)
    log.info("[%s] close_all_positions submitted (%d orders)", label, len(results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten paper accounts")
    parser.add_argument("--swing", action="store_true", help="swing account only")
    parser.add_argument("--intraday", action="store_true", help="intraday account only")
    args = parser.parse_args()

    do_swing = args.swing or not args.intraday
    do_intraday = args.intraday or not args.swing

    if do_swing:
        from trading_bots.alpaca_bot import _get_client as swing_client
        flatten(swing_client(), "SWING")
    if do_intraday:
        from trading_bots.intraday_bot import _get_client as intraday_client
        flatten(intraday_client(), "INTRADAY")


if __name__ == "__main__":
    main()
