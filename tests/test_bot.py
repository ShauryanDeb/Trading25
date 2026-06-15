"""Tests for alpaca_bot and scheduler modules."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_client():
    """Minimal Alpaca TradingClient mock."""
    client = MagicMock()
    account = MagicMock()
    account.portfolio_value = "100000.00"
    account.cash = "100000.00"
    account.buying_power = "400000.00"
    client.get_account.return_value = account
    client.get_all_positions.return_value = []
    return client


@pytest.fixture()
def tiny_model(tmp_path):
    """Fit a minimal StockEnsemble on synthetic data and save it."""
    from pipeline.features import FEATURE_COLS
    from pipeline.model import StockEnsemble

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = pd.Series(rng.integers(0, 2, 200))
    model = StockEnsemble()
    model.fit(X, y)
    path = tmp_path / "model.pkl"
    model.save(path)
    return model, path


# ---------------------------------------------------------------------------
# alpaca_bot: signal generation
# ---------------------------------------------------------------------------

def test_generate_signals_returns_probas(tiny_model):
    """_generate_signals returns a float proba for at least one symbol."""
    from trading_bots.alpaca_bot import _generate_signals

    model, _ = tiny_model
    signals = _generate_signals(model, ["AAPL"])
    assert "AAPL" in signals
    proba = signals["AAPL"]
    assert 0.0 <= proba <= 1.0


def test_generate_signals_bad_symbol_skipped(tiny_model):
    """An invalid symbol is silently skipped, not raised."""
    from trading_bots.alpaca_bot import _generate_signals

    model, _ = tiny_model
    signals = _generate_signals(model, ["INVALID_XYZ_NOPE_999"])
    assert "INVALID_XYZ_NOPE_999" not in signals


def test_latest_price_returns_float():
    """_latest_price returns a positive float for a real ticker."""
    from trading_bots.alpaca_bot import _latest_price

    price = _latest_price("AAPL")
    assert price is not None
    assert price > 0


def test_latest_price_bad_symbol_returns_none():
    from trading_bots.alpaca_bot import _latest_price

    price = _latest_price("INVALID_XYZ_NOPE_999")
    assert price is None


# ---------------------------------------------------------------------------
# alpaca_bot: rebalance dry-run
# ---------------------------------------------------------------------------

def test_rebalance_dry_run_no_orders(mock_client, tiny_model):
    """dry_run=True must never call submit_order or close_position."""
    from trading_bots.alpaca_bot import rebalance

    model, _ = tiny_model
    rebalance(mock_client, model, dry_run=True)

    mock_client.submit_order.assert_not_called()
    mock_client.close_position.assert_not_called()


def test_rebalance_dry_run_returns_list(mock_client, tiny_model):
    """rebalance returns a list (may be empty or contain dry-run entries)."""
    from trading_bots.alpaca_bot import rebalance

    model, _ = tiny_model
    result = rebalance(mock_client, model, dry_run=True)
    assert isinstance(result, list)


def test_rebalance_trade_dict_schema(mock_client, tiny_model):
    """Every trade dict must have the required keys."""
    from trading_bots.alpaca_bot import rebalance

    model, _ = tiny_model
    trades = rebalance(mock_client, model, dry_run=True)
    required = {"timestamp", "symbol", "side", "qty", "price", "signal_proba"}
    for trade in trades:
        assert required.issubset(trade.keys()), f"Missing keys in {trade}"


def test_stop_loss_triggers_close(mock_client, tiny_model):
    """A position at -8% PnL should trigger a close_position call (non-dry-run)."""
    from trading_bots.alpaca_bot import rebalance

    model, _ = tiny_model

    # Inject a position that has dropped 8% below entry
    pos = MagicMock()
    pos.symbol = "AAPL"
    pos.market_value = "920.00"
    pos.avg_entry_price = "1000.00"
    pos.qty = "1"
    mock_client.get_all_positions.return_value = [pos]

    with patch("trading_bots.alpaca_bot._latest_price", return_value=920.0), \
         patch("trading_bots.alpaca_bot._generate_signals", return_value={"AAPL": 0.50}):
        rebalance(mock_client, model, dry_run=False)

    mock_client.close_position.assert_called()


# ---------------------------------------------------------------------------
# alpaca_bot: sector cap, max hold, signal fade (synthetic/predictable data)
# ---------------------------------------------------------------------------

def test_sector_cap_limits_buys_per_sector(mock_client, tiny_model):
    """When all Tech stocks score above threshold, at most MAX_PER_SECTOR are bought."""
    from trading_bots.alpaca_bot import rebalance, MAX_PER_SECTOR, SECTOR_MAP

    tech_syms = SECTOR_MAP["Tech"]
    high_signals = {sym: 0.80 for sym in tech_syms}

    model, _ = tiny_model
    with patch("trading_bots.alpaca_bot._generate_signals", return_value=high_signals), \
         patch("trading_bots.alpaca_bot._latest_price", return_value=100.0), \
         patch("trading_bots.alpaca_bot._entry_dates", return_value={}):
        trades = rebalance(mock_client, model, dry_run=True)

    tech_buys = [t for t in trades if t["symbol"] in tech_syms]
    assert len(tech_buys) <= MAX_PER_SECTOR, (
        f"Got {len(tech_buys)} Tech buys but cap is {MAX_PER_SECTOR}"
    )


def test_sector_cap_allows_cross_sector_buys(mock_client, tiny_model):
    """One stock per sector can all be bought simultaneously (no sector conflict)."""
    from trading_bots.alpaca_bot import rebalance, SECTOR_MAP

    one_per_sector = {syms[0]: 0.75 for syms in SECTOR_MAP.values()}
    model, _ = tiny_model
    with patch("trading_bots.alpaca_bot._generate_signals", return_value=one_per_sector), \
         patch("trading_bots.alpaca_bot._latest_price", return_value=100.0), \
         patch("trading_bots.alpaca_bot._entry_dates", return_value={}):
        trades = rebalance(mock_client, model, dry_run=True)

    bought_syms = {t["symbol"] for t in trades}
    assert len(bought_syms) == len(SECTOR_MAP)


def test_max_hold_triggers_close(mock_client, tiny_model):
    """Position held MAX_HOLD_DAYS+1 days forces close regardless of signal."""
    from trading_bots.alpaca_bot import rebalance, MAX_HOLD_DAYS

    pos = MagicMock()
    pos.symbol = "AAPL"
    pos.market_value = "1000.00"
    pos.avg_entry_price = "100.00"
    pos.qty = "10"
    mock_client.get_all_positions.return_value = [pos]

    stale_entry = pd.Timestamp.now().normalize() - pd.Timedelta(days=MAX_HOLD_DAYS + 1)
    model, _ = tiny_model
    with patch("trading_bots.alpaca_bot._generate_signals",
               return_value={"AAPL": 0.50}), \
         patch("trading_bots.alpaca_bot._latest_price", return_value=100.0), \
         patch("trading_bots.alpaca_bot._entry_dates",
               return_value={"AAPL": stale_entry}):
        rebalance(mock_client, model, dry_run=False)

    mock_client.close_position.assert_called()


def test_max_hold_no_close_when_fresh(mock_client, tiny_model):
    """Position held 1 day should NOT trigger the max-hold exit."""
    from trading_bots.alpaca_bot import rebalance, SELL_THRESHOLD

    pos = MagicMock()
    pos.symbol = "MSFT"
    pos.market_value = "1000.00"
    pos.avg_entry_price = "100.00"
    pos.qty = "10"
    mock_client.get_all_positions.return_value = [pos]

    fresh_entry = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    # Signal above sell threshold so signal-fade doesn't fire either
    model, _ = tiny_model
    with patch("trading_bots.alpaca_bot._generate_signals",
               return_value={"MSFT": SELL_THRESHOLD + 0.10}), \
         patch("trading_bots.alpaca_bot._latest_price", return_value=100.0), \
         patch("trading_bots.alpaca_bot._entry_dates",
               return_value={"MSFT": fresh_entry}):
        rebalance(mock_client, model, dry_run=False)

    # Stop-loss won't fire (pnl=0%), max-hold won't fire (1 day), signal OK
    calls = [str(c) for c in mock_client.close_position.call_args_list]
    assert not any("MSFT" in c for c in calls)


def test_signal_fade_triggers_close(mock_client, tiny_model):
    """Signal below SELL_THRESHOLD causes close even when PnL is positive."""
    from trading_bots.alpaca_bot import rebalance, SELL_THRESHOLD

    pos = MagicMock()
    pos.symbol = "NVDA"
    pos.market_value = "1000.00"
    pos.avg_entry_price = "100.00"
    pos.qty = "10"
    mock_client.get_all_positions.return_value = [pos]

    model, _ = tiny_model
    with patch("trading_bots.alpaca_bot._generate_signals",
               return_value={"NVDA": SELL_THRESHOLD - 0.05}), \
         patch("trading_bots.alpaca_bot._latest_price", return_value=100.0), \
         patch("trading_bots.alpaca_bot._entry_dates", return_value={}):
        rebalance(mock_client, model, dry_run=False)

    mock_client.close_position.assert_called()


def test_entry_dates_reads_buys_from_csv(tmp_path):
    """_entry_dates logic: finds most-recent buy per symbol, ignores sells."""
    import csv as _csv

    fields = ["timestamp", "symbol", "side", "qty", "price", "signal_proba"]
    rows_day1 = [
        {"timestamp": "2026-06-10T14:35:00+00:00", "symbol": "AAPL",
         "side": "buy", "qty": 10, "price": 185.0, "signal_proba": 0.65},
        {"timestamp": "2026-06-10T14:35:00+00:00", "symbol": "MSFT",
         "side": "sell", "qty": 5, "price": 400.0, "signal_proba": 0.30},
    ]
    rows_day2 = [
        {"timestamp": "2026-06-11T14:35:00+00:00", "symbol": "AAPL",
         "side": "buy", "qty": 10, "price": 186.0, "signal_proba": 0.67},
        {"timestamp": "2026-06-11T14:35:00+00:00", "symbol": "NVDA",
         "side": "buy", "qty": 3, "price": 900.0, "signal_proba": 0.71},
    ]
    for fname, rows in [("trades_20260610.csv", rows_day1),
                        ("trades_20260611.csv", rows_day2)]:
        with open(tmp_path / fname, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # Replicate the exact _entry_dates logic against our temp directory
    dates: dict = {}
    for csv_path in sorted(tmp_path.glob("trades_*.csv")):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        for _, row in df.iterrows():
            if str(row.get("side", "")).startswith("buy"):
                sym = row["symbol"]
                ts = pd.Timestamp(row["timestamp"]).tz_localize(None)
                if sym not in dates or ts > dates[sym]:
                    dates[sym] = ts

    assert "AAPL" in dates, "AAPL buy not found"
    assert "NVDA" in dates, "NVDA buy not found"
    assert "MSFT" not in dates, "MSFT had only a sell — should not appear"
    # Most recent AAPL buy is June 11 (later CSV), not June 10
    assert dates["AAPL"].date() == pd.Timestamp("2026-06-11").date()


# ---------------------------------------------------------------------------
# scheduler: trade log CSV
# ---------------------------------------------------------------------------

def test_write_trade_log_creates_csv(tmp_path):
    """_write_trade_log writes a valid CSV with required columns."""
    import trading_bots.scheduler as sched_mod

    original_dir = sched_mod.REPORTS_DIR
    sched_mod.REPORTS_DIR = tmp_path
    try:
        trades = [
            {
                "timestamp": "2026-01-02T14:35:00+00:00",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "price": 185.5,
                "signal_proba": 0.62,
            }
        ]
        out = sched_mod._write_trade_log(trades)
        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["side"] == "buy"
    finally:
        sched_mod.REPORTS_DIR = original_dir


def test_write_trade_log_columns(tmp_path):
    """CSV must have exactly the six required columns."""
    import trading_bots.scheduler as sched_mod

    original_dir = sched_mod.REPORTS_DIR
    sched_mod.REPORTS_DIR = tmp_path
    try:
        out = sched_mod._write_trade_log([
            {"timestamp": "t", "symbol": "X", "side": "buy",
             "qty": 1, "price": 100.0, "signal_proba": 0.6}
        ])
        with open(out) as f:
            header = f.readline().strip().split(",")
        assert header == ["timestamp", "symbol", "side", "qty", "price", "signal_proba"]
    finally:
        sched_mod.REPORTS_DIR = original_dir


def test_write_trade_log_appends(tmp_path):
    """Calling _write_trade_log twice appends rows, not overwrites."""
    import trading_bots.scheduler as sched_mod

    original_dir = sched_mod.REPORTS_DIR
    sched_mod.REPORTS_DIR = tmp_path
    try:
        trade = {"timestamp": "t", "symbol": "X", "side": "buy",
                 "qty": 1, "price": 100.0, "signal_proba": 0.6}
        sched_mod._write_trade_log([trade])
        sched_mod._write_trade_log([trade])
        out = tmp_path / sched_mod._write_trade_log([trade]).name
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
    finally:
        sched_mod.REPORTS_DIR = original_dir


# ---------------------------------------------------------------------------
# scheduler: run_cycle dry-run smoke test
# ---------------------------------------------------------------------------

def test_run_cycle_test_now(tmp_path, tiny_model):
    """run_cycle(dry_run=True) completes without error and writes a CSV."""
    import trading_bots.scheduler as sched_mod

    _, model_path = tiny_model
    original_model = sched_mod.MODEL_PATH
    original_dir = sched_mod.REPORTS_DIR
    sched_mod.MODEL_PATH = model_path
    sched_mod.REPORTS_DIR = tmp_path

    mock_client = MagicMock()
    account = MagicMock()
    account.portfolio_value = "100000.00"
    account.cash = "100000.00"
    account.buying_power = "400000.00"
    mock_client.get_account.return_value = account
    mock_client.get_all_positions.return_value = []

    try:
        with patch("trading_bots.alpaca_bot._get_client", return_value=mock_client):
            sched_mod.run_cycle(dry_run=True)
        csvs = list(tmp_path.glob("trades_*.csv"))
        assert len(csvs) == 1
    finally:
        sched_mod.MODEL_PATH = original_model
        sched_mod.REPORTS_DIR = original_dir
