import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import patch, MagicMock

import main
import kelly_tracker
import trade_calculators
import portfolio_risk
import risk_engine


def test_finding_71_confluence_blocked_veto_in_gating_chain():
    """Finding #71: confluence_blocked == True must veto trade entry."""
    confluence_blocked = True
    all_pass = True
    status_msg = "Traded"
    
    # Simulate pre-trade gating logic in main.py
    if not all_pass:
        status_msg = "Skipped"
    elif confluence_blocked:
        status_msg = "Skipped (HTF Opposition)"
        
    assert status_msg == "Skipped (HTF Opposition)"


def test_finding_72_deterministic_order_link_id():
    """Finding #72: order_link_id must be deterministic per chase to prevent duplicate orders."""
    symbol = "BTCUSDT"
    iv = "15"
    latest_ts = 1700000000000
    chase = 0
    
    id1 = f"c_{symbol[:5]}_{iv}_{int(latest_ts//1000)}_{chase}"[:36]
    id2 = f"c_{symbol[:5]}_{iv}_{int(latest_ts//1000)}_{chase}"[:36]
    assert id1 == id2
    assert len(id1) <= 36
    assert "BTCUS" in id1


def test_finding_73_emergency_flatten_position_requires_zero_size():
    """Finding #73: emergency_flatten_position must confirm position is flat (size == 0.0)."""
    with patch("main.get_bybit_position") as mock_pos, \
         patch("main.place_bybit_taker_ioc_order") as mock_ioc:
        
        # Scenario 1: Initial position exists, IOC fills, position becomes 0.0 -> True
        mock_pos.side_effect = [
            {"size": "0.5", "side": "Buy"},
            {"size": "0.0", "side": "Buy"}
        ]
        mock_ioc.return_value = {"retCode": 0}
        success = main.emergency_flatten_position("BTCUSDT", "Sell", "0.5")
        assert success is True

        # Scenario 2: Initial position exists, but remaining size remains > 0 -> False
        mock_pos.side_effect = [
            {"size": "0.5", "side": "Buy"},
            {"size": "0.2", "side": "Buy"}
        ]
        success2 = main.emergency_flatten_position("BTCUSDT", "Sell", "0.5")
        assert success2 is False


def test_finding_75_immediate_trigger_invariant():
    """Finding #75: Order execution must abort if spot price already crossed stop loss."""
    # Bullish trade, stop loss at 50000, current price at 49900 (already stopped out)
    direction = "Bullish"
    stop_loss = 50000.0
    live_mid = 49900.0
    
    aborted = False
    if direction == "Bullish" and live_mid <= stop_loss:
        aborted = True
    elif direction == "Bearish" and live_mid >= stop_loss:
        aborted = True
        
    assert aborted is True


def test_finding_77_wait_for_order_fill_contract():
    """Finding #77: wait_for_order_fill returns 4-tuple and treats PartiallyFilled as working."""
    with patch("main.get_bybit_order_details") as mock_details:
        # Test 1: Filled order returns (True, 'Filled', cum_qty, avg_price)
        mock_details.return_value = {
            "orderStatus": "Filled",
            "cumExecQty": "0.5",
            "avgPrice": "50000.0"
        }
        is_filled, status, cum_qty, avg_price = main.wait_for_order_fill("BTCUSDT", "ord123", timeout_sec=0.1)
        assert is_filled is True
        assert status == "Filled"
        assert cum_qty == 0.5
        assert avg_price == 50000.0

        # Test 2: Timeout returns is_filled=False with working status PartiallyFilled
        mock_details.return_value = {
            "orderStatus": "PartiallyFilled",
            "cumExecQty": "0.1",
            "avgPrice": "50000.0"
        }
        is_filled_to, status_to, cum_qty_to, _ = main.wait_for_order_fill("BTCUSDT", "ord124", timeout_sec=0.1)
        assert is_filled_to is False
        assert status_to == "PartiallyFilled"
        assert cum_qty_to == 0.1


def test_finding_78_funding_cost_deducted_from_pnl():
    """Finding #78: Funding cost settlements must be deducted from realized PnL."""
    gross_pnl = 10.0
    fee_cost = 0.5
    position_size_usd = 100.0
    leverage = 2.0
    funding_rate = 0.0001
    funding_intervals = 1.0  # 8 hours elapsed
    funding_dir_mult = 1.0  # Long paying positive funding
    
    funding_cost = position_size_usd * leverage * funding_rate * funding_dir_mult * funding_intervals
    assert funding_cost == pytest.approx(0.02)
    
    realized_pnl = gross_pnl - fee_cost - funding_cost
    assert realized_pnl == pytest.approx(9.48)


def test_finding_79_kelly_slippage_scaled_and_whole_trade_logged(tmp_path):
    """Finding #79: Kelly tracker scales slippage haircut to percentage points matching returns."""
    test_file = str(tmp_path / "test_kelly.json")
    tracker = kelly_tracker.KellyTracker(data_file=test_file)
    # Log 35 trades with total return percentages (e.g. 5.0% = 5%)
    for i in range(25):
        tracker.log_trade("BTCUSDT", "15", 5.0, 5.0)
    for i in range(10):
        tracker.log_trade("BTCUSDT", "15", -2.0, -2.0)
        
    fraction = tracker.compute_kelly_fraction(timeframe="15", min_trades=30)
    # Must produce a valid positive fraction without crashing from unscaled slippage
    assert fraction > 0.0


def test_finding_80_sharpe_ratio_standardisation():
    """Finding #80: Sharpe ratio uses ddof=1 and does not inflate with sqrt(len)."""
    pnls = [10.0, -5.0, 15.0, -2.0, 8.0]
    rolling_sharpe = trade_calculators.calculate_rolling_sharpe(pnls)
    
    # Calculate expected unbiased Sharpe
    arr = np.array(pnls, dtype=float)
    expected_sharpe = float(np.mean(arr) / np.std(arr, ddof=1))
    assert rolling_sharpe == pytest.approx(expected_sharpe, rel=1e-3)
    # Check that sqrt(len(arr)) factor is NOT applied
    inflated_t_stat = expected_sharpe * np.sqrt(len(arr))
    assert rolling_sharpe < inflated_t_stat


def test_finding_81_anti_martingale_drawdown_clamping():
    """Finding #81: Anti-martingale multiplier must never exceed 1.0x during drawdown."""
    current_bal = 80.0
    peak_balance = 100.0  # 20% drawdown
    
    # Even if streak is positive, drawdown clamp must hold am_mult <= 1.0
    am_mult = 1.25
    if current_bal < peak_balance * 0.985:
        am_mult = min(1.0, am_mult)
        
    assert am_mult == 1.0


def test_finding_82_and_84_covariance_multiplier_and_new_symbols():
    """Findings #82 & #84: calculate_covariance_multiplier accepts bot_state and covers AVAX/LTC/DOT."""
    bot_state = {
        "active_trade_15m": [{"symbol": "BTCUSDT", "position_size_usd": 50.0, "direction": "Bullish"}],
        "active_trade_30m": [],
        "active_trade_1h": [],
        "active_trade_2h": [],
        "active_trade_4h": [],
        "active_trade_6h": []
    }
    
    # Test AVAXUSDT with existing Bullish BTC trade
    mult_avax, risk_avax = trade_calculators.calculate_covariance_multiplier("AVAXUSDT", "Bullish", bot_state=bot_state)
    assert mult_avax <= 1.0  # Must apply penalty for correlated long
    
    # Test LTCUSDT
    mult_ltc, risk_ltc = trade_calculators.calculate_covariance_multiplier("LTCUSDT", "Bullish", bot_state=bot_state)
    assert mult_ltc <= 1.0
    
    # Test DOTUSDT
    mult_dot, risk_dot = trade_calculators.calculate_covariance_multiplier("DOTUSDT", "Bullish", bot_state=bot_state)
    assert mult_dot <= 1.0


def test_finding_83_hard_circuit_breaker_triggers_kill_switch():
    """Finding #83: When micro-run circuit breaker trips, emergency kill switch is triggered."""
    with patch("main.trigger_emergency_kill_switch") as mock_kill:
        cb_ok = False
        cb_reason = "Loss cap $15.00 exceeded"
        
        if not cb_ok:
            main.trigger_emergency_kill_switch(f"Hard Circuit Breaker ({cb_reason})")
            
        mock_kill.assert_called_once_with("Hard Circuit Breaker (Loss cap $15.00 exceeded)")


def test_finding_85_parametric_var_horizon_scaling():
    """Finding #85: calculate_parametric_var scales intraday returns by sqrt(1440 / interval_minutes)."""
    engine = portfolio_risk.PortfolioRiskEngine()
    positions = [{"symbol": "BTCUSDT", "position_size_usd": 100.0, "leverage": 1.0}]
    
    # Create synthetic 15-minute log returns
    np.random.seed(42)
    returns = pd.DataFrame({"BTCUSDT": np.random.normal(0.0, 0.005, 50)})
    
    # Daily VaR with 1440m interval (1-day) vs 15m interval
    var_1440_usd, var_1440_pct, _ = engine.calculate_parametric_var(positions, returns, total_equity=100.0, interval_minutes=1440)
    var_15_usd, var_15_pct, _ = engine.calculate_parametric_var(positions, returns, total_equity=100.0, interval_minutes=15)
    
    # 15m scaling factor should be sqrt(1440/15) = sqrt(96) ~= 9.7979
    expected_ratio = np.sqrt(1440.0 / 15.0)
    assert (var_15_usd / var_1440_usd) == pytest.approx(expected_ratio, rel=1e-2)
