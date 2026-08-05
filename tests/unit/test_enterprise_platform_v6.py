import pytest
from statistical_validation import StatisticalValidation, statistical_validation
from data_quality_engine import DataQualityEngine, SystemHealthEngine

def test_bootstrap_confidence_interval():
    returns = [0.02, 0.05, -0.01, 0.03, 0.04, -0.02, 0.06, 0.01, 0.03, 0.02]
    mean_val, low_ci, high_ci = statistical_validation.compute_bootstrap_ci(returns, num_samples=500)
    assert mean_val > 0.0
    assert low_ci < mean_val < high_ci

def test_8_production_release_gates_pass():
    res = statistical_validation.evaluate_8_release_gates(
        walk_forward_pass=True,
        out_of_sample_pass=True,
        ece_calibration_pct=3.5,
        psi_drift_score=0.04,
        shadow_trades_count=120,
        research_notebook_approved=True,
        rollback_plan_defined=True,
        live_reality_check_pass=True,
        pf_baseline=1.50,
        pf_candidate=1.65,
        p_value=0.003
    )
    assert res["approved_for_production"] is True
    assert res["pf_gain"] == pytest.approx(0.15, 0.001)

def test_8_production_release_gates_fail_reality_check():
    res = statistical_validation.evaluate_8_release_gates(
        walk_forward_pass=True,
        out_of_sample_pass=True,
        ece_calibration_pct=3.5,
        psi_drift_score=0.04,
        shadow_trades_count=120,
        research_notebook_approved=True,
        rollback_plan_defined=True,
        live_reality_check_pass=False,  # Fails reality check (slippage/latency)
        pf_baseline=1.50,
        pf_candidate=1.65,
        p_value=0.01
    )
    assert res["approved_for_production"] is False

def test_data_quality_severity_levels():
    dq = DataQualityEngine()
    
    # 1. Healthy data
    res_low = dq.evaluate_data_quality()
    assert res_low["severity"] == "LOW"

    # 2. Corrupt timestamp -> CRITICAL
    res_crit = dq.evaluate_data_quality(corrupt_timestamps_count=1)
    assert res_crit["severity"] == "CRITICAL"
    assert res_crit["action"] == "STOP_TRADING_IMMEDIATELY"

    # 3. Data gap -> HIGH
    res_high = dq.evaluate_data_quality(stale_feed_seconds=400.0)
    assert res_high["severity"] == "HIGH"
    assert res_high["action"] == "DISABLE_NEW_ENTRIES"

def test_system_health_decoupling():
    sh = SystemHealthEngine()
    
    # Healthy system
    res = sh.evaluate_system_health(exchange_connected=True, db_connected=True)
    assert res["severity"] == "LOW"

    # Database disconnect -> CRITICAL
    res_db = sh.evaluate_system_health(db_connected=False)
    assert res_db["severity"] == "CRITICAL"
    assert res_db["action"] == "INFRASTRUCTURE_FAILOVER"
