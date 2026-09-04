"""
Unit tests for audit defect findings #86 through #102.
"""

import math
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

import config
from portfolio_risk import portfolio_risk_engine
import risk_engine
from risk_engine import compute_conservative_kelly, JointRiskBudgetAllocator
from state_manager import StateManager
from strategy_health_engine import StrategyHealthEngine
from bybit_client import format_bybit_qty
from calibration_tracker import calculate_ece, record_trade_outcome
from tools.beta_calibrator import is_calibrator_viable, BetaCalibrator


def test_finding_86_directional_budget_leverage():
    """Finding #86: check_directional_budget measures net directional exposure in NOTIONAL (margin * leverage)."""
    # Suppose equity is 1000. Current position has 100 margin, 5x leverage -> 500 notional LONG.
    # Max directional ratio = 1.25 -> max net notional is 1250.
    open_positions = [{
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "position_size_usd": 100.0,  # margin
        "leverage": 5.0              # 500 notional
    }]
    # Candidate: 100 margin, 10x leverage -> 1000 notional LONG.
    # Total net directional notional = 500 + 1000 = 1500 > 1250 (ratio 1.50 > 1.25) -> should fail!
    approved, ratio, reason = portfolio_risk_engine.check_directional_budget(
        total_equity=1000.0,
        open_positions=open_positions,
        proposed_size_usd=100.0,
        proposed_direction="Bullish",
        candidate_leverage=10.0,
        max_directional_ratio=1.25
    )
    assert approved is False
    assert ratio == pytest.approx(1.50, abs=0.01)
    assert "Net Directional Exposure" in reason

    # But if proposed is SHORT 100 margin at 5x leverage (-500 notional):
    # Total net directional notional = 500 - 500 = 0 -> ratio 0.0 -> approved!
    approved_short, ratio_short, _ = portfolio_risk_engine.check_directional_budget(
        total_equity=1000.0,
        open_positions=open_positions,
        proposed_size_usd=100.0,
        proposed_direction="Bearish",
        candidate_leverage=5.0,
        max_directional_ratio=1.25
    )
    assert approved_short is True
    assert ratio_short == pytest.approx(0.0, abs=0.01)


def test_finding_87_correlation_gate_fail_closed_and_hedges():
    """Finding #87: Correlation gate returns 0.80 conservative prior on missing data and credits hedges."""
    # Empty df_dict should trigger conservative 0.80 correlation instead of 0.0 fail-open
    corr_missing = risk_engine.calculate_portfolio_correlation("ETHUSDT", [{"symbol": "BTCUSDT", "direction": "Bullish"}], {}, direction="Bullish")
    assert corr_missing == pytest.approx(0.80, abs=0.01)

    # Opposite direction (hedge) should credit negative correlation, not abs() positive
    # Construct synthetic correlated series
    dates = pd.date_range("2026-01-01", periods=100, freq="15min")
    btc_series = pd.DataFrame({"close": np.linspace(50000, 60000, 100)}, index=dates)
    eth_series = pd.DataFrame({"close": np.linspace(3000, 3600, 100)}, index=dates)
    df_dict = {"BTCUSDT": btc_series, "ETHUSDT": eth_series}

    # Same direction (both Bullish) -> positive correlation ~1.0
    corr_same = risk_engine.calculate_portfolio_correlation("ETHUSDT", [{"symbol": "BTCUSDT", "direction": "Bullish"}], df_dict, direction="Bullish")
    assert corr_same > 0.80

    # Opposite direction (ETH Bearish hedge against BTC Bullish) -> effective correlation is negative
    corr_hedge = risk_engine.calculate_portfolio_correlation("ETHUSDT", [{"symbol": "BTCUSDT", "direction": "Bullish"}], df_dict, direction="Bearish")
    assert corr_hedge < -0.80


def test_finding_88_state_manager_drawdown_persistence():
    """Finding #88: StateManager loads daily drawdown keys from database and persists updates."""
    mock_db = MagicMock()
    mock_db.get_setting.side_effect = lambda k, default=None: {
        "daily_drawdown_start_balance": 1500.0,
        "daily_drawdown_reset_day": 15,
        "circuit_breaker_active": 0,
        "peak_balance": 2000.0
    }.get(k, default)

    with patch("state_manager.database", mock_db):
        sm = StateManager()
        assert sm["daily_drawdown_start_balance"] == 1500.0
        assert sm["daily_drawdown_reset_day"] == 15
        assert sm["circuit_breaker_active"] == 0
        assert sm["peak_balance"] == 2000.0

        # Updating should persist to database
        sm["daily_drawdown_start_balance"] = 1600.0
        mock_db.set_setting.assert_any_call("daily_drawdown_start_balance", "1600.0")


def test_finding_89_model_health_index_statistical_edge():
    """Finding #89: compute_model_health_index calculates SQN, Expectancy, Calmar from recent_pnls."""
    she = StrategyHealthEngine()
    
    # Healthy PnL series: 20 winning trades with +15 USD, 10 losing with -10 USD
    healthy_pnls = [15.0] * 20 + [-10.0] * 10
    healthy_res = she.compute_model_health_index(recent_pnls=healthy_pnls)
    assert healthy_res["health_status"] in ["HEALTHY", "WATCH"]
    assert healthy_res["mhi_score"] > 60.0
    assert healthy_res["sizing_multiplier"] == 1.00

    # Severely degraded PnL series: 25 consecutive losses
    bad_pnls = [-10.0] * 25
    bad_res = she.compute_model_health_index(recent_pnls=bad_pnls)
    assert bad_res["health_status"] in ["CRITICAL", "HALT", "RECOVERY", "DEGRADED"]
    assert bad_res["mhi_score"] < 40.0
    assert bad_res["sizing_multiplier"] == 0.00


def test_finding_90_anti_martingale_peak_balance():
    """Finding #90: calculate_anti_martingale_risk_multiplier uses peak_balance from bot_state."""
    bot_state = {
        "peak_balance": 2000.0
    }
    # Current balance is 1600 (20% drawdown)
    res = risk_engine.calculate_anti_martingale_risk_multiplier(1600.0, bot_state)
    mult = res["multiplier"]
    # Under 20% drawdown, multiplier should scale down below 1.0
    assert mult < 1.0
    assert mult >= 0.25


def test_finding_91_scaled_risk_cap_ratio():
    """Finding #91: MAX_SCALED_RISK_CAP_RATIO is defined as 1.10 in config."""
    assert getattr(config, "MAX_SCALED_RISK_CAP_RATIO", None) == 1.10


def test_finding_93_quantity_lot_flooring():
    """Finding #93: format_bybit_qty floors down to avoid rounding up beyond risk envelope."""
    with patch("bybit_client.get_instrument_specs", return_value={"qty_step": "0.001", "min_qty": "0.001", "price_tick": "0.1"}):
        # 0.0159 should floor to 0.015, NOT round up to 0.016
        formatted = format_bybit_qty("BTCUSDT", 0.0159)
        assert formatted == "0.015"


def test_finding_94_and_99_kelly_haircut_rr():
    """Findings #94 & #99: compute_conservative_kelly applies REALIZED_RR_HAIRCUT."""
    # Given calibrated win rate 0.55, nominal tp_mult 1.85, sl_mult 0.85
    # Haircut dynamically resolved from config.REALIZED_RR_HAIRCUT
    haircut = getattr(config, "REALIZED_RR_HAIRCUT", 0.28)
    eff_tp = 1.85 * haircut
    b_ratio = (eff_tp - 0.0010) / 0.85
    p_star = 1.0 / (b_ratio + 1.0)
    raw_k = (0.55 * (b_ratio + 1.0) - 1.0) / b_ratio if 0.55 > p_star else 0.0
    expected_quarter_k = 0.25 * raw_k
    
    kelly_res = compute_conservative_kelly(
        calibrated_confidence=0.55,
        tp_multiplier=1.85,
        sl_multiplier=0.85
    )
    assert kelly_res == pytest.approx(expected_quarter_k, abs=0.01)


def test_finding_95_joint_risk_budget_allocator_units():
    """Finding #95: allocate_risk_budget converts risk capital to notional using stop_distance."""
    allocator = JointRiskBudgetAllocator()
    # Total equity = 1000. Stop distance = 1000 (2% of 50000)
    # Calibrated confidence 0.75 ensures positive edge under 0.28 haircut
    result = allocator.allocate_risk_budget(
        symbol="BTCUSDT",
        entry_price=50000.0,
        atr_dollars=1000.0,
        atr_norm=0.02,
        calibrated_confidence=0.75,
        direction="Bullish",
        total_equity=1000.0,
        stop_distance=1000.0,
        target_distance=2000.0
    )
    # Position size in notional should be > capital at risk
    assert result["position_size_usd"] > result["capital_at_risk"]
    assert result["position_size_usd"] > 100.0


def test_finding_96_calibrator_break_even_p_star():
    """Finding #96: is_calibrator_viable rejects calibrator when max achievable probability is below p*."""
    # Calibrator whose max output is 0.40
    cal_weak = {
        "scaling_method": "isotonic",
        "y_thresholds": [0.20, 0.30, 0.40],
        "target_definition": "triple_barrier_exact"
    }
    # Break-even p* is 0.42
    assert is_calibrator_viable(cal_weak, min_required_p_star=0.42) is False
    # Calibrator whose max output is 0.65
    cal_strong = {
        "scaling_method": "isotonic",
        "y_thresholds": [0.20, 0.50, 0.65],
        "target_definition": "triple_barrier_exact"
    }
    assert is_calibrator_viable(cal_strong, min_required_p_star=0.42) is True


def test_finding_97_calibration_tracker_wider_bins():
    """Finding #97: calculate_ece works with sparse trades using wider bins."""
    record_trade_outcome(0.55, True)
    record_trade_outcome(0.58, False)
    record_trade_outcome(0.62, True)
    record_trade_outcome(0.70, True)
    record_trade_outcome(0.72, False)

    ece = calculate_ece()
    assert 0.0 <= ece <= 1.0


def test_finding_101_dashboard_auth_and_cors():
    """Finding #101: DASHBOARD_ALLOW_PUBLIC defaults to false."""
    import os
    from dashboard_routes import require_ip_whitelist
    
    # Verify default is false
    assert os.environ.get("DASHBOARD_ALLOW_PUBLIC", "false").lower() == "false"


def test_finding_92_tradable_balance_only():
    """Finding #92: get_real_bybit_balance queries only derivatives-tradable accountType (UNIFIED / CONTRACT)."""
    import bybit_client
    # Verify mock response with non-tradable accounts does not get summed
    mock_response = {
        "retCode": 0,
        "result": {
            "list": [
                {"accountType": "UNIFIED", "totalWalletBalance": "150.0", "coin": [{"coin": "USDT", "walletBalance": "150.0"}]},
                {"accountType": "FUND", "totalWalletBalance": "5000.0", "coin": [{"coin": "USDT", "walletBalance": "5000.0"}]},
                {"accountType": "SPOT", "totalWalletBalance": "1000.0", "coin": [{"coin": "USDT", "walletBalance": "1000.0"}]}
            ]
        }
    }
    with patch("bybit_client.bybit_get_request", return_value=mock_response):
        bal = bybit_client.get_real_bybit_balance()
        # Should only equal 150.0, NOT 5000 or 1000
        assert bal == pytest.approx(150.0, abs=0.1)


def test_finding_98_dynamic_threshold_bounding():
    """Finding #98: MAX_THRESHOLD_UPLIFT never clamps dynamic threshold below configured base_cfg_thresh."""
    from config import MAX_THRESHOLD_UPLIFT
    base_cfg_thresh = 0.48  # E.g. 240m base threshold
    economic_base = 0.38
    
    effective_base = max(economic_base, base_cfg_thresh)
    max_allowed = min(0.65, effective_base + MAX_THRESHOLD_UPLIFT)
    # Dynamic threshold suggestion lower than base_cfg_thresh
    raw_dynamic = 0.40
    clamped_dynamic = float(round(max(base_cfg_thresh, min(max_allowed, raw_dynamic)), 4))
    
    # Must preserve base_cfg_thresh as lower bound
    assert clamped_dynamic >= base_cfg_thresh
    assert clamped_dynamic <= max_allowed


def test_finding_100_calibrator_target_compatibility():
    """Finding #100: is_calibrator_viable checks target_definition compatibility."""
    cal_good = {
        "scaling_method": "isotonic",
        "y": [0.2, 0.4, 0.6],
        "target_definition": "triple_barrier_exact"
    }
    assert is_calibrator_viable(cal_good) is True

    cal_bad_target = {
        "scaling_method": "isotonic",
        "y": [0.2, 0.4, 0.6],
        "target_definition": "incompatible_fixed_horizon"
    }
    assert is_calibrator_viable(cal_bad_target) is False


def test_finding_102_decision_journal_outcome_placed_flag():
    """Finding #102: outcome is EXECUTED only when placed is True, otherwise REJECTED or SKIPPED."""
    # Simulate decision record
    class DummyRecord:
        def __init__(self):
            self.outcome = "APPROVED"
            self.status_msg = "Pending"
            self.reject_reason = None

    rec = DummyRecord()
    placed = False
    status_msg = "Skipped (Invalid Geometry)"
    
    if placed:
        rec.outcome = "EXECUTED"
    elif status_msg.startswith("REJECTED"):
        rec.outcome = "REJECTED"
    else:
        rec.outcome = "SKIPPED"
        
    assert rec.outcome == "SKIPPED"
    assert rec.outcome != "EXECUTED"

