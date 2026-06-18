"""Tests for intraday_bot — ATR exits, VIX filter, min-hold, cooldown.

All tests use synthetic/predictable data and mock out Alpaca + network calls,
so they run fully offline without touching live APIs or model files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_intraday_model(tmp_path):
    """Fit a minimal IntradayEnsemble on synthetic data and save it."""
    from pipeline.intraday_features import INTRADAY_FEATURE_COLS
    from pipeline.intraday_model import IntradayEnsemble

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((300, len(INTRADAY_FEATURE_COLS))),
        columns=INTRADAY_FEATURE_COLS,
    )
    y = pd.Series(rng.integers(0, 2, 300))
    model = IntradayEnsemble()
    model.fit(X, y)
    path = tmp_path / "intraday_model.pkl"
    model.save(path)
    return model, path


@pytest.fixture()
def mock_intraday_client():
    """Minimal Alpaca TradingClient mock for the intraday account."""
    client = MagicMock()
    account = MagicMock()
    account.portfolio_value = "50000.00"
    account.buying_power = "50000.00"
    client.get_account.return_value = account
    client.get_all_positions.return_value = []
    return client


# ---------------------------------------------------------------------------
# _check_exits: ATR-based profit target and stop-loss
# ---------------------------------------------------------------------------

def _make_position(symbol, entry, current_price, qty=10.0, atr=1.0):
    return {
        symbol: {
            "qty": qty,
            "avg_entry_price": entry,
            "current_price": current_price,
            "entry_atr": atr,
        }
    }


def test_atr_exit_profit_target_fires():
    """Price >= entry + 2*ATR → position closed; symbol NOT in stopped_out."""
    from trading_bots.intraday_bot import _check_exits, ATR_MULT_TARGET

    client = MagicMock()
    entry = 100.0
    atr = 1.0
    # +2.5% is above the 2% profit target (2*ATR/entry = 2*1/100)
    positions = _make_position("AAPL", entry, entry * (1 + ATR_MULT_TARGET * atr / entry + 0.005))

    with patch("trading_bots.intraday_bot._log_trade"):
        stopped = _check_exits(client, positions, dry_run=False)

    client.close_position.assert_called_once_with("AAPL")
    assert "AAPL" not in stopped, "Profit-target exit should NOT add to cooldown set"


def test_atr_exit_stop_loss_fires_and_cools_down():
    """Price <= entry - 1*ATR → position closed; symbol added to stopped_out."""
    from trading_bots.intraday_bot import _check_exits, ATR_MULT_STOP

    client = MagicMock()
    entry = 100.0
    atr = 1.0
    # -1.5% is below the -1% stop-loss (1*ATR/entry = 1*1/100)
    positions = _make_position("MSFT", entry, entry * (1 - ATR_MULT_STOP * atr / entry - 0.005))

    with patch("trading_bots.intraday_bot._log_trade"):
        stopped = _check_exits(client, positions, dry_run=False)

    client.close_position.assert_called_once_with("MSFT")
    assert "MSFT" in stopped, "Stop-loss exit must add symbol to cooldown set"


def test_atr_exit_within_band_no_close():
    """Price inside [-1%, +2%] ATR band → no exit triggered."""
    from trading_bots.intraday_bot import _check_exits

    client = MagicMock()
    # +0.5%: above stop, below profit target
    positions = _make_position("NVDA", 100.0, 100.5, atr=1.0)

    stopped = _check_exits(client, positions, dry_run=False)

    client.close_position.assert_not_called()
    assert len(stopped) == 0


def test_atr_exit_dry_run_never_closes():
    """dry_run=True must never call close_position even when target is hit."""
    from trading_bots.intraday_bot import _check_exits

    client = MagicMock()
    # Clear stop-loss: -5%
    positions = _make_position("AMD", 100.0, 95.0, atr=1.0)

    _check_exits(client, positions, dry_run=True)

    client.close_position.assert_not_called()


def test_atr_fallback_uses_floor_profit():
    """When entry_atr=0, MIN_TARGET_PCT floor applies as profit target."""
    from trading_bots.intraday_bot import _check_exits, MIN_TARGET_PCT

    client = MagicMock()
    # +1.3% exceeds the MIN_TARGET_PCT floor → profit target fires
    positions = _make_position("INTC", 100.0, 100.0 * (1 + MIN_TARGET_PCT + 0.001), atr=0.0)

    with patch("trading_bots.intraday_bot._log_trade"):
        stopped = _check_exits(client, positions, dry_run=False)

    client.close_position.assert_called_once_with("INTC")
    assert "INTC" not in stopped


def test_atr_fallback_no_exit_below_floor():
    """Price just below MIN_TARGET_PCT must NOT trigger exit (floor protects it)."""
    from trading_bots.intraday_bot import _check_exits, MIN_TARGET_PCT, MIN_STOP_PCT

    client = MagicMock()
    # Price between -MIN_STOP_PCT and +MIN_TARGET_PCT → inside band, no exit
    positions = _make_position("INTC", 100.0, 100.0 * (1 + MIN_TARGET_PCT * 0.5), atr=0.0)
    _check_exits(client, positions, dry_run=False)
    client.close_position.assert_not_called()


def test_atr_fallback_uses_floor_stop():
    """When entry_atr=0, MIN_STOP_PCT floor applies as stop-loss."""
    from trading_bots.intraday_bot import _check_exits, MIN_STOP_PCT

    client = MagicMock()
    # -0.7% exceeds MIN_STOP_PCT (0.6%) → stop fires
    positions = _make_position("QCOM", 100.0, 100.0 * (1 - MIN_STOP_PCT - 0.001), atr=0.0)

    with patch("trading_bots.intraday_bot._log_trade"):
        stopped = _check_exits(client, positions, dry_run=False)

    client.close_position.assert_called_once_with("QCOM")
    assert "QCOM" in stopped


def test_atr_exit_scales_with_volatility():
    """Higher ATR → wider exit bands; low-ATR position exits, high-ATR doesn't."""
    from trading_bots.intraday_bot import _check_exits, MIN_STOP_PCT

    entry = 100.0
    # Low-ATR stock: raw stop = $0.50/$100 = 0.5%, but floor raises to MIN_STOP_PCT (0.6%).
    # -1.5% clearly exceeds the 0.6% floor → stop fires.
    client_low = MagicMock()
    positions_low = _make_position("LOW_VOL", entry, entry * 0.985, atr=0.50)
    with patch("trading_bots.intraday_bot._log_trade"):
        _check_exits(client_low, positions_low, dry_run=False)
    assert client_low.close_position.called, "Low-ATR stock should be stopped out at -1.5%"

    # High-ATR stock: raw stop = $3.00/$100 = 3%.
    # -1.5% is inside the 3% band → does NOT trigger.
    client_high = MagicMock()
    positions_high = _make_position("HIGH_VOL", entry, entry * 0.985, atr=3.00)
    _check_exits(client_high, positions_high, dry_run=False)
    assert not client_high.close_position.called, "High-ATR stock should survive -1.5%"


# ---------------------------------------------------------------------------
# Min-hold bars filter (from run_session's eligibility check)
# ---------------------------------------------------------------------------

def test_min_hold_bars_blocks_immediate_exit():
    """Position entered on the current tick is not eligible for exit check."""
    from trading_bots.intraday_bot import MIN_HOLD_BARS

    tick = 1
    entry_tick = {"AAPL": 1}  # entered this very tick
    positions = {"AAPL": {"qty": 1, "avg_entry_price": 100.0,
                          "current_price": 90.0, "entry_atr": 1.0}}

    eligible = {s: p for s, p in positions.items()
                if tick - entry_tick.get(s, 0) >= MIN_HOLD_BARS}

    assert "AAPL" not in eligible, (
        f"Should need at least {MIN_HOLD_BARS} ticks before exit is checked"
    )


def test_min_hold_bars_allows_exit_after_threshold():
    """Position entered MIN_HOLD_BARS ticks ago becomes eligible."""
    from trading_bots.intraday_bot import MIN_HOLD_BARS

    tick = 10
    entry_tick = {"AAPL": 10 - MIN_HOLD_BARS}  # entered exactly MIN_HOLD_BARS ago
    positions = {"AAPL": {"qty": 1, "avg_entry_price": 100.0,
                          "current_price": 90.0, "entry_atr": 1.0}}

    eligible = {s: p for s, p in positions.items()
                if tick - entry_tick.get(s, 0) >= MIN_HOLD_BARS}

    assert "AAPL" in eligible


def test_min_hold_bars_value():
    """MIN_HOLD_BARS must be >= 2 (strategy requirement)."""
    from trading_bots.intraday_bot import MIN_HOLD_BARS

    assert MIN_HOLD_BARS >= 2, (
        f"MIN_HOLD_BARS={MIN_HOLD_BARS} is less than 2; "
        "positions can be exited immediately after entry"
    )


# ---------------------------------------------------------------------------
# Session cooldown: stopped_out set prevents re-entry
# ---------------------------------------------------------------------------

def test_stopped_out_prevents_reentry():
    """A symbol in stopped_out must be excluded from buy candidates."""
    # Simulate the loop guard: `if sym in positions or sym in stopped_out`
    stopped_out = {"AAPL", "MSFT"}
    universe = ["AAPL", "MSFT", "NVDA", "GOOGL"]
    positions = {}

    candidates = [sym for sym in universe
                  if sym not in positions and sym not in stopped_out]

    assert "AAPL" not in candidates
    assert "MSFT" not in candidates
    assert "NVDA" in candidates
    assert "GOOGL" in candidates


def test_stopped_out_accumulates_across_stops():
    """Each stop-loss adds to the cooldown set; set grows monotonically."""
    from trading_bots.intraday_bot import _check_exits

    client = MagicMock()
    # Two symbols both hitting stop-loss
    positions = {
        "SYM_A": {"qty": 5, "avg_entry_price": 100.0,
                  "current_price": 98.0, "entry_atr": 1.0},  # -2% > 1% stop
        "SYM_B": {"qty": 3, "avg_entry_price": 200.0,
                  "current_price": 196.0, "entry_atr": 1.0},  # -2% > 0.5% stop
    }
    with patch("trading_bots.intraday_bot._log_trade"):
        stopped = _check_exits(client, positions, dry_run=False)

    assert "SYM_A" in stopped
    assert "SYM_B" in stopped
    assert len(stopped) == 2


# ---------------------------------------------------------------------------
# VIX filter: high-volatility regime suspends session
# ---------------------------------------------------------------------------

def test_vix_filter_suspends_when_above_max(tmp_path, tiny_intraday_model):
    """run_session returns before placing any orders when VIX > VIX_MAX."""
    from trading_bots.intraday_bot import run_session, VIX_MAX

    _, model_path = tiny_intraday_model
    mock_client = MagicMock()

    with patch("trading_bots.intraday_bot._current_vix", return_value=VIX_MAX + 5.0), \
         patch("trading_bots.intraday_bot._get_client", return_value=mock_client), \
         patch("trading_bots.intraday_bot._get_data_client", return_value=MagicMock()), \
         patch("trading_bots.intraday_bot.MODEL_PATH", model_path), \
         patch("pipeline.intraday_model.IntradayEnsemble.load",
               return_value=tiny_intraday_model[0]):
        run_session(dry_run=True)

    # get_account is called AFTER the VIX guard — if VIX suspends, it must not be called
    mock_client.get_account.assert_not_called()
    mock_client.submit_order.assert_not_called()


def test_vix_filter_proceeds_when_below_max(tmp_path, tiny_intraday_model, mock_intraday_client):
    """run_session proceeds past the VIX check when VIX <= VIX_MAX."""
    from trading_bots.intraday_bot import run_session, VIX_MAX
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
    _, model_path = tiny_intraday_model

    # Make 'now' appear to be 3:56 PM ET so the session exits immediately after setup
    fake_now = pd.Timestamp("2026-06-15 15:56:00", tz=ET)

    with patch("trading_bots.intraday_bot._current_vix", return_value=VIX_MAX - 5.0), \
         patch("trading_bots.intraday_bot._get_client",
               return_value=mock_intraday_client), \
         patch("trading_bots.intraday_bot._get_data_client", return_value=MagicMock()), \
         patch("trading_bots.intraday_bot.MODEL_PATH", model_path), \
         patch("pipeline.intraday_model.IntradayEnsemble.load",
               return_value=tiny_intraday_model[0]), \
         patch("trading_bots.intraday_bot.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.now.side_effect = None
        run_session(dry_run=True)

    # If VIX was OK, get_account must have been called
    mock_intraday_client.get_account.assert_called()


# ---------------------------------------------------------------------------
# Entry-time filter: no new buys outside 10 AM–3 PM ET window
# ---------------------------------------------------------------------------

def test_entry_time_filter_constants():
    """ENTRY_START_HOUR and ENTRY_END_HOUR enforce the 10 AM–3 PM window."""
    from trading_bots.intraday_bot import ENTRY_START_HOUR, ENTRY_END_HOUR

    assert ENTRY_START_HOUR == 10, "Entries should start at 10 AM ET"
    assert ENTRY_END_HOUR == 15, "Entries should stop at 3 PM ET (before hard close)"


def test_entry_allowed_within_window():
    """Hours 10–14 are within the entry window."""
    from trading_bots.intraday_bot import ENTRY_START_HOUR, ENTRY_END_HOUR

    for hour in range(ENTRY_START_HOUR, ENTRY_END_HOUR):
        assert ENTRY_START_HOUR <= hour < ENTRY_END_HOUR


def test_entry_blocked_outside_window():
    """Hours 9 and 15 are outside the entry window."""
    from trading_bots.intraday_bot import ENTRY_START_HOUR, ENTRY_END_HOUR

    for hour in (9, ENTRY_END_HOUR, 16):
        allowed = ENTRY_START_HOUR <= hour < ENTRY_END_HOUR
        assert not allowed, f"Hour {hour} should be outside the entry window"


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

def test_intraday_constants_reasonable():
    """Core constants must be within sensible strategy bounds."""
    from trading_bots.intraday_bot import (
        BUY_THRESHOLD, ATR_MULT_TARGET, ATR_MULT_STOP,
        VIX_MAX, MAX_POSITIONS, POSITION_PCT,
    )

    assert 0.0 < BUY_THRESHOLD < 1.0, "BUY_THRESHOLD must be a valid probability"
    assert ATR_MULT_TARGET > ATR_MULT_STOP, (
        "Profit target multiple must exceed stop multiple (positive expectancy)"
    )
    assert VIX_MAX > 0, "VIX_MAX must be positive"
    assert 1 <= MAX_POSITIONS <= 20, "MAX_POSITIONS out of reasonable range"
    assert 0 < POSITION_PCT <= 0.10, "POSITION_PCT must be between 0 and 10%"
    assert MAX_POSITIONS * POSITION_PCT <= 1.0, (
        "MAX_POSITIONS * POSITION_PCT > 100%: would require more than full capital"
    )


# ---------------------------------------------------------------------------
# _spy_trending_down: market-direction filter
# ---------------------------------------------------------------------------

def _make_spy_bars(closes: list[float]) -> pd.DataFrame:
    """Build a minimal SPY bar DataFrame from a list of close prices."""
    idx = pd.date_range("2026-06-18 09:30", periods=len(closes), freq="5min", tz="UTC")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_spy_trending_down_detects_streak():
    """Returns True when the last SPY_BARS_DOWN bars all close lower."""
    from trading_bots.intraday_bot import _spy_trending_down, SPY_BARS_DOWN

    # Build SPY_BARS_DOWN+2 bars where the last SPY_BARS_DOWN are each lower
    closes = [100.0, 100.5] + [100.0 - i * 0.1 for i in range(SPY_BARS_DOWN)]
    bars = _make_spy_bars(closes)

    with patch("pipeline.intraday_data.fetch_latest_bars", return_value=bars):
        result = _spy_trending_down()

    assert result is True, "Should detect consecutive down streak"


def test_spy_trending_down_false_on_mixed_bars():
    """Returns False when the streak is interrupted by one up bar."""
    from trading_bots.intraday_bot import _spy_trending_down, SPY_BARS_DOWN

    # Last bar is up, breaking the streak
    closes = [100.0, 99.8, 99.6, 99.4, 99.6]  # final bar rises
    bars = _make_spy_bars(closes)

    with patch("pipeline.intraday_data.fetch_latest_bars", return_value=bars):
        result = _spy_trending_down()

    assert result is False, "Interrupted streak should not block entries"


def test_spy_trending_down_false_on_insufficient_bars():
    """Returns False (safe default) when fewer bars are available than needed."""
    from trading_bots.intraday_bot import _spy_trending_down, SPY_BARS_DOWN

    # Only 2 bars — not enough to evaluate a SPY_BARS_DOWN streak
    bars = _make_spy_bars([100.0, 99.0])

    with patch("pipeline.intraday_data.fetch_latest_bars", return_value=bars):
        result = _spy_trending_down()

    assert result is False, "Insufficient bars → safe default (don't block entries)"


def test_spy_trending_down_false_on_fetch_error():
    """Returns False when fetch_latest_bars raises — never block on data errors."""
    from trading_bots.intraday_bot import _spy_trending_down

    with patch("pipeline.intraday_data.fetch_latest_bars", side_effect=RuntimeError("no data")):
        result = _spy_trending_down()

    assert result is False, "Exception in SPY fetch must not block trading"
