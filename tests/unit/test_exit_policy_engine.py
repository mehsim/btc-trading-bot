import pytest
import os
import json
from exit_policy_engine import ExitPolicyEngine, compute_file_sha256, ENGINE_VERSION

def test_sha256_computation(tmp_path):
    p = tmp_path / "test_policy.json"
    p.write_text('{"key": "value"}')
    h = compute_file_sha256(str(p))
    assert len(h) == 64
    assert h != "UNKNOWN_HASH"

def test_engine_version_validation(tmp_path):
    engine = ExitPolicyEngine()
    
    # Test valid policy
    valid_p = tmp_path / "valid_policy.json"
    valid_p.write_text(json.dumps({"min_engine_version": "3.0", "parameters": {}}))
    res = engine._load_and_validate_policy(str(valid_p))
    assert res["min_engine_version"] == "3.0"
    
    # Test incompatible policy
    invalid_p = tmp_path / "invalid_policy.json"
    invalid_p.write_text(json.dumps({"min_engine_version": "4.0", "parameters": {}}))
    with pytest.raises(ValueError, match="Incompatible policy"):
        engine._load_and_validate_policy(str(invalid_p))

def test_compute_be_buffer():
    engine = ExitPolicyEngine()
    buf = engine.compute_be_buffer(
        symbol="BTCUSDT",
        leverage=10.0,
        entry_price=60000.0,
        atr_dollars=600.0,
        safety_margin_atr=0.10
    )
    # (0.0011 fee + 0.0005 slippage) * 10 lev = 0.016 fraction of price (or 0.0016 * 60000 = 96) + 60 safety = 69.6
    assert buf > 50.0
    assert buf < 200.0

def test_evaluate_stagnation_gate():
    engine = ExitPolicyEngine()
    
    # All 5 criteria satisfied
    is_stag, reason = engine.evaluate_stagnation_gate(
        pnl_usd=-10.0,
        current_atr=400.0,
        entry_atr=600.0,
        current_volume=50.0,
        avg_volume=100.0,
        trade_age_hours=7.0,
        stagnation_age_hours=6.0,
        price_dev=100.0,
        adx_val=15.0,
        regime="RANGING"
    )
    assert is_stag is True
    assert "5-FACTOR STAGNATION" in reason

    # Positive PnL should prevent stagnation exit
    is_stag2, _ = engine.evaluate_stagnation_gate(
        pnl_usd=+10.0,
        current_atr=400.0,
        entry_atr=600.0,
        current_volume=50.0,
        avg_volume=100.0,
        trade_age_hours=7.0,
        stagnation_age_hours=6.0,
        price_dev=100.0,
        adx_val=15.0,
        regime="RANGING"
    )
    assert is_stag2 is False

def test_hybrid_trailing_stop():
    engine = ExitPolicyEngine()
    sl_long = engine.evaluate_hybrid_trailing_stop(
        direction="Bullish",
        current_price=64000.0,
        entry_price=60000.0,
        stop_loss=59000.0,
        swing_price=62500.0,
        atr_dollars=500.0
    )
    # Swing low (62500) - 125 = 62375
    assert sl_long >= 62000.0

def test_evaluate_exit_trace():
    engine = ExitPolicyEngine()
    active_trade = {
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "entry_price": 60000.0,
        "stop_loss": 59000.0,
        "take_profit": 63000.0,
        "half_closed": False,
        "position_size_usd": 15.0,
        "leverage": 10.0,
        "atr_dollars": 500.0,
        "entry_time": (1783800000 * 1000)
    }
    
    # Evaluate at normal price
    exit_reason, updates, trace = engine.evaluate_exit(
        active_trade=active_trade,
        current_price=60500.0,
        current_time=1783810000,
        regime="RANGING"
    )
    assert exit_reason is None
    assert "policy_hash" in trace
    assert "exit_efficiency" in trace
    assert trace["policy_id"] == engine.active_champion_id


def test_mhi_scaling_soft_limit_differs():
    engine = ExitPolicyEngine()
    eval_healthy = engine.evaluate_10_level_exit_hierarchy(
        symbol="BTCUSDT", interval="15", current_price=64000.0, entry_price=64000.0,
        stop_loss=63000.0, take_profit=66000.0, direction="Bullish", candles_elapsed=5,
        mhi_status=100.0
    )
    eval_degraded = engine.evaluate_10_level_exit_hierarchy(
        symbol="BTCUSDT", interval="15", current_price=64000.0, entry_price=64000.0,
        stop_loss=63000.0, take_profit=66000.0, direction="Bullish", candles_elapsed=5,
        mhi_status=40.0
    )
    assert eval_healthy["soft_limit"] > eval_degraded["soft_limit"]
    assert eval_degraded["soft_limit"] == 8  # 15 * 0.50 floor round


def test_continuous_atr_adj_and_exit_scoring():
    engine = ExitPolicyEngine()
    eval_vol_low = engine.evaluate_10_level_exit_hierarchy(
        symbol="BTCUSDT", interval="15", current_price=64000.0, entry_price=64000.0,
        stop_loss=63000.0, take_profit=66000.0, direction="Bullish", candles_elapsed=5,
        garch_vol=0.005, rolling_vol_20th_pct=0.010, atr_ratio=0.9
    )
    eval_vol_high = engine.evaluate_10_level_exit_hierarchy(
        symbol="BTCUSDT", interval="15", current_price=64000.0, entry_price=64000.0,
        stop_loss=63000.0, take_profit=66000.0, direction="Bullish", candles_elapsed=5,
        garch_vol=0.015, rolling_vol_20th_pct=0.010, atr_ratio=1.1
    )
    assert eval_vol_high["exit_score"] > eval_vol_low["exit_score"]
    assert eval_vol_high["soft_limit"] >= eval_vol_low["soft_limit"]

