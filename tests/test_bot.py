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
