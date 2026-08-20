import pytest
from statistical_validation import StatisticalValidation, statistical_validation
from data_quality_engine import DataQualityEngine, SystemHealthEngine

def test_bootstrap_confidence_interval():
    returns = [0.02, 0.05, -0.01, 0.03, 0.04, -0.02, 0.06, 0.01, 0.03, 0.02]
    mean_val, low_ci, high_ci = statistical_validation.compute_bootstrap_ci(returns, num_samples=500, block_len=2)
    assert mean_val > 0.0
    assert low_ci < mean_val < high_ci

def test_benjamini_hochberg_vector_fdr():
    p_vals = [0.001, 0.01, 0.04, 0.20, 0.80]
    q_vals = statistical_validation.benjamini_hochberg(p_vals, alpha=0.05)
    assert len(q_vals) == len(p_vals)
    # Monotonicity check on sorted ranks
    assert q_vals[0] <= q_vals[1] <= q_vals[2]
    assert q_vals[0] < 0.05

def test_dynamic_sample_power_calculation():
    # Large effect -> high power
    pow_large = statistical_validation.calculate_dynamic_sample_power(effect_size_d=0.50, completed_trades=100)
    assert pow_large["statistical_power"] > 0.80
    assert pow_large["is_statistically_sufficient"] is True

    # Tiny effect -> low power
    pow_small = statistical_validation.calculate_dynamic_sample_power(effect_size_d=0.02, completed_trades=100)
    assert pow_small["statistical_power"] < 0.20
    assert pow_small["is_statistically_sufficient"] is False

def test_sprt_sequential_test_preregistered():
    # Pre-registered d_target = 0.20, positive standardized drift -> ACCEPT_H1_PROMOTE
    res_pos = statistical_validation.run_sprt_sequential_test(n_samples=200, diff_mean=0.05, std_dev=0.10, d_target_registered=0.20)
    assert res_pos["sprt_decision"] == "ACCEPT_H1_PROMOTE"

    # Negative standardized drift -> REJECT_H1_ABORT
    res_neg = statistical_validation.run_sprt_sequential_test(n_samples=200, diff_mean=-0.05, std_dev=0.10, d_target_registered=0.20)
    assert res_neg["sprt_decision"] == "REJECT_H1_ABORT"

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
        p_value=0.003,
        num_trials=12
    )
    assert res["approved_for_production"] is True
    assert res["pf_gain"] == pytest.approx(0.15, 0.001)

def test_8_production_release_gates_not_evaluable_pf():
    res = statistical_validation.evaluate_8_release_gates(
        walk_forward_pass=True,
        out_of_sample_pass=True,
        ece_calibration_pct=3.5,
        psi_drift_score=0.04,
        shadow_trades_count=120,
        research_notebook_approved=True,
        rollback_plan_defined=True,
        live_reality_check_pass=True,
        pf_baseline=None,
        pf_candidate=None,
        p_value=0.001,
        num_trials=12
    )
    assert res["approved_for_production"] is False
    assert res["gate_details"]["Dual-Significance (PF Gain >= 0.05)"] == "NOT_EVALUABLE"

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
        p_value=0.01,
        num_trials=12
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
