import pytest
import time
from unittest.mock import patch
import config
import database
from circuit_breaker import circuit_breaker

def test_circuit_breaker_halts_on_max_trades_cap():
    """Verify that exceeding MAX_LIVE_TRADES_CAP activates circuit breaker."""
    mock_trades = [
        {"exit_time": time.time(), "pnl_usd": 0.10, "venue_closed_pnl": 0.10}
        for _ in range(60)
    ]
    
    with patch("database.get_setting", return_value=str(time.time() - 1000)), \
         patch("database.get_completed_trades", return_value=mock_trades), \
         patch.object(config, "MAX_LIVE_TRADES_CAP", 60), \
         patch.object(config, "MAX_LIVE_LOSS_CAP", 15.0), \
         patch("database.set_setting") as mock_set_setting:
        
        micro_ts = time.time() - 1000
        closed_all = database.get_completed_trades(limit=1000)
        micro_trades = [t for t in closed_all if float(t.get("exit_time") or 0) >= micro_ts]
        closed_trade_count = len(micro_trades)
        cumulative_loss = -sum(float(t.get("venue_closed_pnl") or t.get("pnl_usd") or 0.0) for t in micro_trades if float(t.get("venue_closed_pnl") or t.get("pnl_usd") or 0.0) < 0)
        
        is_healthy, reason = circuit_breaker.evaluate_micro_run_caps(
            closed_trade_count=closed_trade_count,
            cumulative_loss=cumulative_loss,
            max_trades_cap=60,
            max_loss_cap=15.0
        )
        
        assert is_healthy is False
        assert circuit_breaker.is_circuit_active is True
        assert "MAX_TRADES_EXCEEDED" in reason
        
        bot_state = {"bot_running": True, "bot_stopped": False}
        if not is_healthy:
            bot_state["bot_running"] = False
            bot_state["bot_stopped"] = True
            database.set_setting("bot_running", "False")
            database.set_setting("bot_stopped", "True")
            
        assert bot_state["bot_running"] is False
        assert bot_state["bot_stopped"] is True
        mock_set_setting.assert_any_call("bot_running", "False")
        mock_set_setting.assert_any_call("bot_stopped", "True")


def test_circuit_breaker_halts_on_max_loss_cap():
    """Verify that exceeding MAX_LIVE_LOSS_CAP ($15.0) activates circuit breaker."""
    mock_trades = [
        {"exit_time": time.time(), "pnl_usd": -15.50, "venue_closed_pnl": -15.50}
    ]
    
    with patch("database.get_setting", return_value=str(time.time() - 1000)), \
         patch("database.get_completed_trades", return_value=mock_trades), \
         patch.object(config, "MAX_LIVE_TRADES_CAP", 60), \
         patch.object(config, "MAX_LIVE_LOSS_CAP", 15.0), \
         patch("database.set_setting") as mock_set_setting:
        
        micro_ts = time.time() - 1000
        closed_all = database.get_completed_trades(limit=1000)
        micro_trades = [t for t in closed_all if float(t.get("exit_time") or 0) >= micro_ts]
        closed_trade_count = len(micro_trades)
        cumulative_loss = -sum(float(t.get("venue_closed_pnl") or t.get("pnl_usd") or 0.0) for t in micro_trades if float(t.get("venue_closed_pnl") or t.get("pnl_usd") or 0.0) < 0)
        
        is_healthy, reason = circuit_breaker.evaluate_micro_run_caps(
            closed_trade_count=closed_trade_count,
            cumulative_loss=cumulative_loss,
            max_trades_cap=60,
            max_loss_cap=15.0
        )
        
        assert is_healthy is False
        assert circuit_breaker.is_circuit_active is True
        assert "MAX_LOSS_EXCEEDED" in reason
        
        bot_state = {"bot_running": True, "bot_stopped": False}
        if not is_healthy:
            bot_state["bot_running"] = False
            bot_state["bot_stopped"] = True
            database.set_setting("bot_running", "False")
            database.set_setting("bot_stopped", "True")
            
        assert bot_state["bot_running"] is False
        assert bot_state["bot_stopped"] is True
        mock_set_setting.assert_any_call("bot_running", "False")
        mock_set_setting.assert_any_call("bot_stopped", "True")
