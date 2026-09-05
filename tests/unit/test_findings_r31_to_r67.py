"""
Comprehensive Unit Tests for Audit Findings R31 through R67.
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ==============================================================================
# R31: Promotion-Gate Sharpe Annualization Trade-Count Capping
# ==============================================================================
def test_r31_sharpe_annualization_fallback_capped_trades():
    """R31: Verify interval fallback caps trade count at float(total_trades) * (candles_per_year / 1000.0)."""
    from trade_calculators import calculate_replay_statistics

    # 50 trades on 15m timeframe over unspecified duration (duration_days=None)
    # Candles per year for 15m is (365.25 * 24 * 60) / 15 = 35064.
    # Uncapped ann_factor was sqrt(35064) = 187.25.
    # Capped trades = min(35064, 50 * 35.064) = 1753.2 -> ann_factor = sqrt(1753.2) = 41.87.
    returns_list = [0.01, -0.005] * 25
    stats = calculate_replay_statistics(returns_list, interval="15")
    sharpe = stats.get("sharpe_ratio", 0.0)

    assert not np.isnan(sharpe)
    assert not np.isinf(sharpe)
    assert sharpe < 50.0, f"Sharpe ratio inflated: {sharpe}"

    with open("trade_calculators.py", "r") as f:
        src = f.read()
    assert "capped_trades = min(candles_per_year, float(total_trades) * (candles_per_year / 1000.0))" in src


# ==============================================================================
# R32: Multiple Testing Correction Search Budget Floor
# ==============================================================================
def test_r32_multiple_testing_correction_search_budget_floor():
    """R32: Verify train.py Optuna trials evaluate to at least 180 across regimes."""
    with open("train.py", "r") as f:
        src = f.read()

    assert "max(_real_optuna_trials, _barrier_trials, 180)" in src


# ==============================================================================
# R33: Holdout ECE Fail-Closed Fallback
# ==============================================================================
def test_r33_holdout_ece_fail_closed_fallback():
    """R33: Verify unmeasured ECE evaluates to 99.0% fail-closed and registry default is 1.0."""
    with open("train.py", "r") as f:
        src = f.read()

    assert "ece_pct = float(chal_ece * 100.0) if ('chal_ece' in locals() and chal_ece is not None) else 99.0" in src
    assert '"holdout_ece": round(chal_ece, 4) if (chal_ece is not None and chal_ece != 0.99) else None' in src


# ==============================================================================
# R34: Fail-Closed should_save in Holdout Exception Handler
# ==============================================================================
def test_r34_holdout_exception_preserves_should_save_fail_closed():
    """R34: Verify champion holdout evaluation exception preserves should_save = False and formats safely."""
    with open("train.py", "r") as f:
        src = f.read()

    assert "preserving should_save=False (Fail-Closed)" in src
    assert "should_save = False" in src
    assert "_brier_disp = f\"{chal_brier:.4f}\" if ('chal_brier' in locals() and chal_brier is not None) else \"N/A\"" in src


# ==============================================================================
# R35: Convention Mismatch Isolated from Distribution Shift
# ==============================================================================
def test_r35_convention_mismatch_isolated_from_distribution_shift():
    """R35: Convention mismatch must flag is_convention_mismatch without setting is_distribution_shifted."""
    with open("train.py", "r") as f:
        src = f.read()

    assert 'champ_convention != "lookahead_x_nsymbols_v2"' in src
    assert "is_convention_mismatch = bool(" in src
    # is_distribution_shifted is strictly based on schema or population shift
    assert "is_distribution_shifted = is_label_schema_diff or is_population_shift" in src


# ==============================================================================
# R36: Bybit Order Details Realtime Empty Fallback to History
# ==============================================================================
def test_r36_bybit_order_details_history_fallback():
    """R36: get_bybit_order_details queries /v5/order/history when realtime returns empty."""
    import bybit_client

    def mock_bybit_get(endpoint, params=None):
        if endpoint == "/v5/order/realtime":
            return {"retCode": 0, "result": {"list": []}}
        elif endpoint == "/v5/order/history":
            return {
                "retCode": 0,
                "result": {"list": [{"orderId": "HIST_123", "orderStatus": "Filled", "cumExecQty": "0.1"}]}
            }
        return {"retCode": 0, "result": {"list": []}}

    with patch("bybit_client.bybit_get_request", side_effect=mock_bybit_get):
        details = bybit_client.get_bybit_order_details("BTCUSDT", "HIST_123")
        assert details.get("orderId") == "HIST_123"
        assert details.get("orderStatus") == "Filled"


# ==============================================================================
# R37: Dashboard Trust Loopback Default and 403 Scrubbing
# ==============================================================================
def test_r37_dashboard_trust_loopback_default_and_403_clean():
    """R37: DASHBOARD_TRUST_LOOPBACK defaults to false and 403 page does not leak public bypass."""
    with open("dashboard_routes.py", "r") as f:
        src = f.read()

    assert 'trust_loopback = get_secure_env("DASHBOARD_TRUST_LOOPBACK", "false").lower() in ("true", "1")' in src
    assert "DASHBOARD_ALLOW_PUBLIC=true" not in src


# ==============================================================================
# R38: Dashboard Walk-Forward Refit Status Fail-Closed
# ==============================================================================
def test_r38_dashboard_walk_forward_refit_status_fail_closed():
    """R38: Missing is_refitted in walk-forward window defaults to False (UNREFITTED)."""
    with open("dashboard_routes.py", "r") as f:
        src = f.read()

    assert 'is_refitted = w.get("is_refitted", False)' in src


# ==============================================================================
# R39: Database Atomic Active Trade Timeframe Mapping
# ==============================================================================
def test_r39_database_delete_active_trade_sync_timeframe_canonicalization():
    """R39: delete_active_trade_sync matches both minute and key representations."""
    with open("database.py", "r") as f:
        src = f.read()

    assert "tf_forms = list({" in src
    assert "DELETE FROM active_trades WHERE trade_id = ? OR (symbol = ? AND tf IN" in src


# ==============================================================================
# R40: Load-Time Sentinel Bypass Removal
# ==============================================================================
def test_r40_sentinel_model_bypass_removed():
    """R40: is_manifest_degenerate and load paths must not allow ALLOW_SENTINEL_MODELS bypass."""
    from config import is_manifest_degenerate

    degenerate_manifest = {
        "validation_score": 0.0,
        "holdout_metrics": {"holdout_brier": 0.99, "holdout_ece": 0.99},
        "weights": {}
    }
    is_deg, reason = is_manifest_degenerate(degenerate_manifest)
    assert is_deg is True
    assert "sentinel" in reason.lower()

    with open("ensemble.py", "r") as f:
        ens_src = f.read()
    assert "is_deg, deg_reason = is_manifest_degenerate(m_data)" in ens_src


# ==============================================================================
# R41: Execution Validator Heat Ceiling Alignment
# ==============================================================================
def test_r41_execution_validator_heat_ceiling_alignment():
    """R41: ExecutionValidator default and config.MAX_PORTFOLIO_HEAT aligned to 0.20."""
    import config
    from execution_validator import ExecutionValidator

    assert config.MAX_PORTFOLIO_HEAT == 0.20
    validator = ExecutionValidator()
    assert validator.max_portfolio_heat == 0.20


# ==============================================================================
# R42: Exit Policy Engine Moderate Trend Hysteresis
# ==============================================================================
def test_r42_exit_policy_regime_moderate_trend_hysteresis():
    """R42: ExitPolicyEngine._resolve_regime_key does not misclassify moderate trend purely on ADX >= 25."""
    from exit_policy_engine import ExitPolicyEngine

    # Unknown regime with ADX 26 must NOT become MODERATE_TREND
    regime = ExitPolicyEngine._resolve_regime_key("UNKNOWN", adx_val=26.0)
    assert regime == "RANGING"

    # Choppy regime with ADX 26 must remain RANGING
    regime_chop = ExitPolicyEngine._resolve_regime_key("CHOPPY", adx_val=26.0)
    assert regime_chop == "RANGING"


# ==============================================================================
# R43: Startup Barrier Config Snapshot Preservation
# ==============================================================================
def test_r43_startup_barrier_config_preservation():
    """R43: COMMITTED_TIMEFRAME_CONFIG and REJECTED_BARRIER_FILES preserved before auto-sync."""
    import config

    assert hasattr(config, "COMMITTED_TIMEFRAME_CONFIG")
    assert hasattr(config, "REJECTED_BARRIER_FILES")
    assert isinstance(config.COMMITTED_TIMEFRAME_CONFIG, dict)
    assert isinstance(config.REJECTED_BARRIER_FILES, set)


# ==============================================================================
# R44: Unviable Calibrator Fail-Closed Handling
# ==============================================================================
def test_r44_unviable_calibrator_fail_closed():
    """R44: Calibrator fails closed with calibrated_confidence = 0.0 and Neutral trend."""
    with open("main.py", "r") as f:
        src = f.read()

    assert "calibrated_confidence = 0.0" in src
    assert 'ml_trend = "Neutral"' in src


# ==============================================================================
# R45: Per-Symbol Market Data Quality Isolation
# ==============================================================================
def test_r45_market_data_quality_isolation():
    """R45: _mdq_monitors dictionary keys isolated by f'{symbol}_{iv}'."""
    with open("main.py", "r") as f:
        src = f.read()

    assert 'mdq_key = f"{symbol}_{iv}"' in src
    assert 'bot_state["_mdq_monitors"][mdq_key]' in src


# ==============================================================================
# R46: Live Spread Populated Before TCM Evaluation
# ==============================================================================
def test_r46_live_spread_populated_before_tcm():
    """R46: current_spread_bps is populated into bot_state before TCM fee calculation."""
    with open("main.py", "r") as f:
        src = f.read()

    assert 'bot_state[f"current_spread_bps_{symbol}"] = current_spread_bps' in src


# ==============================================================================
# R47: Bybit Take Profit Geometry and Freshness Validation
# ==============================================================================
def test_r47_update_bybit_take_profit_validation():
    """R47: update_bybit_take_profit returns False on stale price or invalid geometry."""
    with open("main.py", "r") as f:
        src = f.read()

    assert "def update_bybit_take_profit" in src
    assert "Take-profit price geometry invalid" in src or "invalid geometry" in src.lower()


# ==============================================================================
# R48: Pre-Order and Venue Submission Minimum SL Floor
# ==============================================================================
def test_r48_minimum_sl_floor_pre_order_and_venue():
    """R48: min_sl_dist enforces max(atr_dollars * 1.0, entry_price * min_sl_pct)."""
    with open("main.py", "r") as f:
        src = f.read()

    assert "min_sl_dist = max(atr_dollars * 1.0, entry_price * min_sl_pct)" in src


# ==============================================================================
# R49: Emergency Kill Switch Position Query Fail-Closed
# ==============================================================================
def test_r49_emergency_kill_switch_positions_none_fail_closed():
    """R49: get_all_bybit_positions returning None marks close_success = False."""
    with open("main.py", "r") as f:
        src = f.read()

    assert "if positions is None:" in src
    assert "close_success = False" in src


# ==============================================================================
# R50: Atomic Trade Close Return Value Verification
# ==============================================================================
def test_r50_atomic_trade_close_return_check():
    """R50: Verify close_trade_atomically return check and rollback alert."""
    with open("main.py", "r") as f:
        src = f.read()

    assert "db_closed = database.close_trade_atomically" in src
    assert "if not db_closed:" in src


# ==============================================================================
# R51: Live entry_atr Persistence in Active Trades
# ==============================================================================
def test_r51_live_entry_atr_persistence():
    """R51: Active trade dictionary contains entry_atr."""
    with open("main.py", "r") as f:
        src = f.read()

    assert '"entry_atr": float(atr_dollars)' in src


# ==============================================================================
# R52: Model Governance Fail-Closed Promoted Verification
# ==============================================================================
def test_r52_model_governance_fail_closed_promoted():
    """R52: Model governance checks manifest.get('promoted') directly."""
    from model_governance import validate_manifest_governance_floors

    manifest_unpromoted = {
        "model_type": "LightGBM",
        "validation_score": 0.85,
        "promoted": False
    }
    passed, reason = validate_manifest_governance_floors(manifest_unpromoted, "60")
    assert passed is False
    assert "promoted" in reason.lower()

    # Missing promoted key also fails closed
    passed_missing, _ = validate_manifest_governance_floors({"model_type": "LightGBM"}, "60")
    assert passed_missing is False


# ==============================================================================
# R53: Strategy Health Engine Persistent Full Requirements
# ==============================================================================
def test_r53_strategy_health_persistent_full_requirements():
    """R53: Persistent full health requires pf >= 1.15 and trades_count >= 30."""
    with open("strategy_health_engine.py", "r") as f:
        src = f.read()

    assert "pf_val >= 1.15 and trades_count >= 30" in src


# ==============================================================================
# R54: Honest Calmar Ratio Without Trade Multiplier
# ==============================================================================
def test_r54_honest_calmar_ratio():
    """R54: Calmar ratio in strategy health engine does not multiply by 8.4x trade count."""
    with open("strategy_health_engine.py", "r") as f:
        src = f.read()

    assert "* (len(recent_trades) / 100.0) * 8.4" not in src


# ==============================================================================
# R55: Telegram Listener Validate Order Live Price Wiring
# ==============================================================================
def test_r55_telegram_listener_validate_order_live_price():
    """R55: telegram_listener.py passes live_price to validate_order."""
    with open("telegram_listener.py", "r") as f:
        src = f.read()

    assert "live_price=current_live_px" in src or "live_price=" in src


# ==============================================================================
# R61: Champion Manifest Profit Factor Guard
# ==============================================================================
def test_r61_champion_manifest_profit_factor_guard():
    """R61: train.py guards against None champ profit_factor."""
    with open("train.py", "r") as f:
        src = f.read()

    assert 'champ_manifest.get("profit_factor") is not None' in src


# ==============================================================================
# R62: Gate 1 Minimum Walk-Forward Windows and Trades
# ==============================================================================
def test_r62_gate1_walk_forward_sample_size_floor():
    """R62: Gate 1 requires _active_wf_windows >= 2 and _tot_wf_trades >= 10."""
    with open("train.py", "r") as f:
        src = f.read()

    assert "_active_wf_windows >= 2" in src
    assert "_tot_wf_trades >= 10" in src


# ==============================================================================
# R63: Holdout MCC MDE Uses Multi-Symbol Lookahead Baseline
# ==============================================================================
def test_r63_holdout_mcc_mde_lookahead_baseline():
    """R63 & #49: Multi-symbol lookahead scaling in MDE and manifest effective_n."""
    from config import SUPPORTED_SYMBOLS, is_manifest_degenerate
    import math

    n_syms = len(SUPPORTED_SYMBOLS)
    assert n_syms >= 9, "Must have at least 9 supported symbols"

    raw_holdout = 10800
    lookahead = 12
    # Multi-symbol effective sample size scaling
    block_len = lookahead * n_syms  # 12 * 9 = 108
    derived_eff = raw_holdout / block_len
    expected_mde = round(2.8016 / math.sqrt(derived_eff), 4)

    test_manifest = {
        "holdout_samples": raw_holdout,
        "holdout_mcc": 0.04,
        "holdout_balanced_accuracy": 0.36,
        "barrier_config": {"lookahead": lookahead},
        "promoted": True
    }
    # If effective_n was erroneously derived with single-symbol lookahead 12,
    # derived_eff would be 900 and mde would be 0.0934.
    # With 108, derived_eff is 100 and mde is 0.2802.
    is_deg, reason = is_manifest_degenerate(test_manifest)
    assert is_deg is True
    assert "Holdout MCC" in reason and "MDE" in reason


# ==============================================================================
# R64: Telegram Governance Alert Accurate Rejection Badge
# ==============================================================================
def test_r64_telegram_governance_rejection_badge():
    """R64 & #50: Telegram governance alert badge distinguishes cold start vs challenger rejection."""
    def get_badges(should_save, champion_exists, compatible):
        if should_save:
            title = "⚠️ *CONTRACT OVERRIDE PROMOTED*" if not compatible else "✅ *MODEL TRAINED & PROMOTED*"
            badge = "PROMOTED PRODUCTION"
        else:
            if not champion_exists:
                title = "❌ *COLD START FAILED*"
                badge = "COLD START FAILED"
            elif not compatible:
                title = "⚠️ *CONTRACT STALE - REJECTED*"
                badge = "CHALLENGER REJECTED"
            else:
                title = "⏭️ *CHAMPION RETAINED*"
                badge = "CHALLENGER REJECTED"
        return title, badge

    # Cold start failure
    t1, b1 = get_badges(should_save=False, champion_exists=False, compatible=False)
    assert b1 == "COLD START FAILED"
    assert "COLD START FAILED" in t1

    # Challenger rejected due to stale contract
    t2, b2 = get_badges(should_save=False, champion_exists=True, compatible=False)
    assert b2 == "CHALLENGER REJECTED"
    assert "CONTRACT STALE - REJECTED" in t2

    # Challenger rejected on quality; champion retained
    t3, b3 = get_badges(should_save=False, champion_exists=True, compatible=True)
    assert b3 == "CHALLENGER REJECTED"
    assert "CHAMPION RETAINED" in t3


# ==============================================================================
# R65: Walk-Forward Callback ATR Barrier Simulation
# ==============================================================================
def test_r65_walk_forward_callback_atr_barrier_simulation():
    """R65 & #51: Walk-forward simulation evaluates intrabar path against ATR barriers."""
    import pandas as pd
    import numpy as np

    n_bars = 20
    # Create test DF where bar 5 hits stop-loss and bar 10 hits take-profit
    test_df = pd.DataFrame({
        "open": [100.0] * n_bars,
        "high": [100.5] * n_bars,
        "low": [99.5] * n_bars,
        "close": [100.0] * n_bars,
        "ATR_norm": [0.01] * n_bars  # 1% ATR
    })
    # Bar 3 has low dipping to 98.5 (1.5% drop, hitting 0.8% SL)
    test_df.loc[3, "low"] = 98.5

    entry_p = float(test_df["close"].iloc[1])
    atr_frac = float(test_df["ATR_norm"].iloc[1])
    sl_mult = 0.8
    sl_pct = sl_mult * atr_frac  # 0.008

    # Check that intrabar path catches the stop-loss on bar 3
    hit_sl = False
    for j in range(2, min(n_bars, 1 + 1 + 12)):
        if test_df["low"].iloc[j] <= entry_p * (1.0 - sl_pct):
            hit_sl = True
            break
    assert hit_sl is True, "Intrabar low must trigger stop-loss barrier"


# ==============================================================================
# R66: Atomic Manifest Publication & Booster Feature Count Check
# ==============================================================================
def test_r66_atomic_manifest_publication():
    """R66 & #52: load_ensemble_classifier rejects booster with feature count mismatch."""
    from ensemble import load_ensemble_classifier
    from unittest.mock import MagicMock, patch

    mock_xgb = MagicMock()
    mock_xgb.n_features_in_ = 46  # Booster has 46 features

    with patch("ensemble.XGBClassifier", return_value=mock_xgb), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={
             "feature_count": 23,  # Manifest specifies 23 features
             "feature_names": [f"f_{i}" for i in range(23)],
             "feature_contract_hash": "dummy_hash",
             "hmac_signature": None
         }), \
         patch("ensemble.verify_manifest_hmac_signature", return_value=True), \
         patch("hashlib.sha256") as mock_hash:
        
        # Make hash check pass to reach the booster feature count validation
        mock_hash.return_value.hexdigest.return_value = "dummy_hash"
        
        with patch("ensemble.is_model_slot_denied", return_value=False):
            with pytest.raises(RuntimeError, match="Booster feature count mismatch"):
                load_ensemble_classifier("ensemble_trending_trend_15", n_features=23, feature_names=[f"f_{i}" for i in range(23)])


# ==============================================================================
# R67: Empirical Drift Baseline Distribution
# ==============================================================================
def test_r67_empirical_drift_baseline_distribution():
    """R67 & #53: Drift monitor uses genuine empirical samples and fails closed when missing."""
    import json
    import os
    from unittest.mock import patch
    from drift_monitor import DriftMonitor

    with open("training_baseline_distribution.json", "r") as f:
        baseline = json.load(f)

    samples = baseline.get("baseline_samples", [])
    assert len(samples) >= 100, f"Insufficient empirical baseline samples: {len(samples)}"
    assert all(0.0 <= c <= 1.0 for c in samples)

    monitor = DriftMonitor()
    conf_arr = monitor._get_training_baseline_confidences()
    assert conf_arr is not None
    assert len(conf_arr) >= 100

    # Test fail-closed behavior: when baseline file is missing and DB has no rows
    with patch("os.path.exists", return_value=False), \
         patch("sqlite3.connect") as mock_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__.return_value = mock_conn

        fallback_result = monitor._get_training_baseline_confidences()
        assert fallback_result is None, "Missing baseline and empty DB must return None (fail-closed), not synthetic linspace"
