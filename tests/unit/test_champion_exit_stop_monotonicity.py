import time
from unittest.mock import patch, MagicMock
import pytest

import main
from exit_policy_engine import ExitPolicyEngine
from order_state_machine import StopStateMachine


def test_evaluate_exit_monotonic_be_stop_long():
    """Verify evaluate_exit never returns a widened new_stop_loss on a long position with trailed SL."""
    engine = ExitPolicyEngine()
    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "entry_price": 100000.0,
        "stop_loss": 100400.0,  # Trailed into profit (entry + 0.8 ATR)
        "take_profit": 103000.0,
        "atr_dollars": 500.0,
        "leverage": 5.0,
        "break_even_triggered": False,
        "half_closed": False,
    }
    # Price reaches 100750 (entry + 1.5 ATR)
    current_price = 100750.0

    reason, updates, trace = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=current_price,
        current_time=time.time(),
        regime="TRENDING",
        adx_val=30.0,
        swing_price=100300.0
    )

    # updates should mark break_even_triggered=True, but MUST NOT include a widened new_stop_loss (e.g. 100058)
    if "new_stop_loss" in updates:
        assert updates["new_stop_loss"] >= active_trade["stop_loss"], (
            f"Widened stop returned for Long! Proposed: {updates['new_stop_loss']} < Current: {active_trade['stop_loss']}"
        )


def test_evaluate_exit_monotonic_be_stop_short():
    """Verify evaluate_exit never returns a widened new_stop_loss on a short position with trailed SL."""
    engine = ExitPolicyEngine()
    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bearish",
        "entry_price": 100000.0,
        "stop_loss": 99600.0,  # Trailed into profit (entry - 0.8 ATR)
        "take_profit": 97000.0,
        "atr_dollars": 500.0,
        "leverage": 5.0,
        "break_even_triggered": False,
        "half_closed": False,
    }
    # Price drops to 99250 (entry - 1.5 ATR)
    current_price = 99250.0

    reason, updates, trace = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=current_price,
        current_time=time.time(),
        regime="TRENDING",
        adx_val=30.0,
        swing_price=99700.0
    )

    if "new_stop_loss" in updates:
        assert updates["new_stop_loss"] <= active_trade["stop_loss"], (
            f"Widened stop returned for Short! Proposed: {updates['new_stop_loss']} > Current: {active_trade['stop_loss']}"
        )


def test_update_bybit_stop_loss_rejects_widening_via_snapshot():
    """Verify update_bybit_stop_loss rejects a worse SL when guarded by immutable snapshot."""
    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "qty": 0.01,
        "stop_loss": 100400.0,
        "stop_state": "TRAILING"
    }

    # Attempt to widen SL to 100058 while current snapshot is 100400.0
    with patch("main.bot_state", {"live_price_BTCUSDT": 100750.0}), \
         patch("main.bybit_post_request") as mock_post:

        result = main.update_bybit_stop_loss(
            symbol="BTCUSDT",
            sl_price=100058.0,
            active_trade=active_trade,
            current_sl_snapshot=100400.0
        )
        assert result is False
        mock_post.assert_not_called()


def test_update_bybit_stop_loss_accepts_tighter_sl():
    """Verify update_bybit_stop_loss allows tightening SL."""
    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "qty": 0.01,
        "stop_loss": 100400.0,
        "stop_state": "TRAILING",
        "target_stop_state": "TRAILING"
    }

    with patch("main.bot_state", {"live_price_BTCUSDT": 101000.0}), \
         patch("main.bybit_post_request", return_value={"retCode": 0}) as mock_post:

        result = main.update_bybit_stop_loss(
            symbol="BTCUSDT",
            sl_price=100600.0,
            active_trade=active_trade,
            current_sl_snapshot=100400.0
        )
        assert result is True
        mock_post.assert_called_once()
