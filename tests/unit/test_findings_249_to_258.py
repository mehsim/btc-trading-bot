import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import config


# --- Finding #81: MDE & Effective-N Governance ---
def test_finding_81_mde_and_effective_n_governance():
    # 1. Test dynamic derivation in config.is_manifest_degenerate
    manifest_without_mde = {
        "validation_metrics": {"mcc": 0.05, "accuracy": 0.55},
        "holdout_metrics": {"mcc": 0.04, "accuracy": 0.52},
        "n_holdout_samples": 864,
        "barrier_config": {"lookahead": 12},
        "features": ["f1", "f2"],
        "model_type": "ensemble"
    }

    # Effective N = 864 / 12 = 72.0; MDE = 2.8016 / sqrt(72.0) = 0.33017
    # holdout_mcc = 0.04 < 0.33017 -> should be degenerate
    is_degen, reason = config.is_manifest_degenerate(manifest_without_mde)
    assert is_degen is True
    assert "80% MDE" in reason or "holdout_mcc_below_mde_80pct" in reason

    # With holdout_mcc >= MDE
    manifest_sufficient_mde = {
        "validation_metrics": {"mcc": 0.40, "accuracy": 0.65},
        "holdout_metrics": {"mcc": 0.35, "accuracy": 0.62},
        "n_holdout_samples": 864,
        "barrier_config": {"lookahead": 12},
        "features": ["f1", "f2"],
        "model_type": "ensemble"
    }
    is_degen_ok, _ = config.is_manifest_degenerate(manifest_sufficient_mde)
    assert is_degen_ok is False

    # 2. Check dashboard_routes.py exposes holdout_effective_n and holdout_mcc_mde_80pct
    with open("dashboard_routes.py", "r") as f:
        dash_src = f.read()
    assert "holdout_effective_n" in dash_src
    assert "holdout_mcc_mde_80pct" in dash_src


# --- Finding #82: Effective-Sample Convention Mismatch ---
def test_finding_82_effective_sample_convention_mismatch():
    with open("train.py", "r") as f:
        train_src = f.read()

    # Legacy untagged champion defaults to legacy_single_symbol_v1
    assert "legacy_single_symbol_v1" in train_src
    assert 'champ_convention != "lookahead_x_nsymbols_v2"' in train_src
    assert "champ_mcc_val = None" in train_src
    assert "is_distribution_shifted = True" in train_src


# --- Finding #83: Unified Round-Trip Transaction Cost Modeling ---
def test_finding_83_unified_round_trip_transaction_cost_modeling():
    import trade_calculators
    from trade_calculators import get_canonical_round_trip_cost_bp
    from transaction_cost_model import TransactionCostModel

    # Verify default round trip fraction covers maker + taker + spread + impact
    assert abs(trade_calculators.DEFAULT_ROUND_TRIP_COST_FRAC - 0.00125) < 1e-6
    assert trade_calculators.DEFAULT_ROUND_TRIP_COST_FRAC >= config.MAKER_FEE_PCT + config.TAKER_FEE_PCT

    # Verify break-even stop uses config.TAKER_FEE_PCT
    be_stop = trade_calculators.calculate_break_even_stop(
        direction="Buy",
        entry_price=50000.0,
        spread_pct=0.0003,
        slippage_pct=0.0002
    )
    assert be_stop > 50000.0  # Must be above entry for Long to cover maker + taker + buffer

    # Verify TransactionCostModel round-trip computation with live spread & volatility
    tcm = TransactionCostModel()
    cost_res = tcm.estimate_transaction_cost(
        symbol="BTCUSDT",
        order_size_usd=50000.0,
        is_maker=True,
        round_trip=True,
        bid_ask_spread_bp=4.0,
        garch_sigma=0.015
    )
    assert cost_res["total_cost_usd"] > 0.0
    assert cost_res["fee_bp"] > 0.0
    assert cost_res["half_spread_bp"] > 0.0
    assert cost_res["market_impact_bp"] > 0.0

    # Helper function check
    bps = get_canonical_round_trip_cost_bp(bid_ask_spread_bp=3.0)
    assert bps > 5.0  # maker (2) + taker (5.5) + half spread (1.5) = 9.0 bps


# --- Finding #84: Denylist & Serve-Time Manifest Degeneracy Enforcement ---
def test_finding_84_denylist_and_manifest_degeneracy_enforcement():
    with open("ensemble.py", "r") as f:
        ens_src = f.read()

    # Crash sentinel check >= 0.99
    assert "float(h_brier) >= 0.99" in ens_src
    assert "float(h_ece) >= 0.99" in ens_src
    # Manifest degeneracy check at load time
    assert "is_manifest_degenerate" in ens_src

    with open("signal_evaluator.py", "r") as f:
        sig_src = f.read()

    # SignalEvaluator invokes is_manifest_degenerate on every candidate
    assert "is_manifest_degenerate(m_data)" in sig_src
    assert 'm_data.get("promoted", False)' in sig_src

    with open("main.py", "r") as f:
        main_src = f.read()

    # Reset candidate loop variables at candidate loop start
    assert "_mdata = None" in main_src
    assert "manifest_load_error = None" in main_src
    assert "is_promoted_flag = None" in main_src
    assert 'is_promoted_flag = _mdata.get("promoted", False)' in main_src
    assert "abstain_reason = None" in main_src


# --- Finding #85: Model Registry Multi-Slot Mapping & Historical Bonferroni ---
def test_finding_85_model_registry_multi_slot_mapping_and_historical_bonferroni():
    from mlops_engine import ModelRegistry
    import tempfile
    import os

    # Test auto-migration of single-record Production to slot mapping
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_file = os.path.join(tmpdir, "model_registry.json")
        legacy_data = {
            "Production": {
                "model_name": "legacy_trending_model",
                "tag": "prod",
                "features": ["f1"]
            },
            "Staging": [],
            "Archived": [
                {"model_name": "old_1", "slot_name": "ensemble_trending_15"},
                {"model_name": "old_2", "slot_name": "ensemble_trending_15"}
            ]
        }
        with open(reg_file, "w") as f:
            json.dump(legacy_data, f)

        registry = ModelRegistry(registry_file=reg_file)
        # Production should now be a dictionary keyed by slot
        assert isinstance(registry.models["Production"], dict)
        assert "ensemble_trending_15" in registry.models["Production"] or "legacy_trending_model" in registry.models["Production"]

    # Verify on-disk model_registry.json format
    with open("model_registry.json", "r") as f:
        disk_reg = json.load(f)
    assert isinstance(disk_reg.get("Production"), dict)

    # Verify historical Bonferroni correction and rich metrics in train.py
    with open("train.py", "r") as f:
        train_src = f.read()
    assert 'model_registry.models.get("Archived"' in train_src
    assert 'reg_slot_name = f"ensemble_{name}_{interval}"' in train_src
    assert "holdout_effective_n" in train_src


# --- Finding #86: Backtest Overlapping Win Rate CI Decision Refusal ---
def test_finding_86_backtest_overlapping_ci_decision_refusal():
    with open("backtest.py", "r") as f:
        bt_src = f.read()

    assert "[STATISTICALLY_INDISTINGUISHABLE]" in bt_src
    assert "wilson_score_interval" in bt_src or "wilson" in bt_src.lower()

    with open("backtest_results.json", "r") as f:
        bt_results = json.load(f)

    scenarios = bt_results.get("scenarios", [])
    assert len(scenarios) > 0
    # Verify scenarios do not claim distinct superiority if indistinguishable
    has_ci = False
    has_indistinguishable = False
    for s in scenarios:
        wr_str = s.get("Pessimistic WinRate", "")
        if "[" in wr_str and "%" in wr_str:
            has_ci = True
        if "[STATISTICALLY_INDISTINGUISHABLE]" in wr_str:
            has_indistinguishable = True
    assert has_ci is True
    assert has_indistinguishable is True


# --- Finding #87: SignalEvaluator Fail-Closed Holdout Gate & Default Floor Parity ---
def test_finding_87_signal_evaluator_fail_closed_holdout_gate():
    # 1. Config timeframe floors have 120, 240, and default
    assert "120" in config.TIMEFRAME_MIN_HOLDOUT_MCC
    assert "240" in config.TIMEFRAME_MIN_HOLDOUT_MCC
    assert "default" in config.TIMEFRAME_MIN_HOLDOUT_MCC
    assert config.TIMEFRAME_MIN_HOLDOUT_MCC["default"] == 0.035

    assert "120" in config.TIMEFRAME_MIN_HOLDOUT_BAL_ACC
    assert "240" in config.TIMEFRAME_MIN_HOLDOUT_BAL_ACC
    assert "default" in config.TIMEFRAME_MIN_HOLDOUT_BAL_ACC

    # 2. Verify signal_evaluator fail-closed holdout gate logic
    with open("signal_evaluator.py", "r") as f:
        sig_src = f.read()

    # Must fail-closed on None or below floor
    assert "m_hmcc is None or float(m_hmcc) < min_h_mcc" in sig_src
    assert "m_hbal is None or float(m_hbal) < min_h_bal" in sig_src


# --- Finding #88: Hierarchical 4H Macro Bias Technical Fallback ---
def test_finding_88_hierarchical_4h_macro_bias_technical_fallback():
    import signal_evaluator

    # Test 1: When ML macro model is Neutral, fall back to technical trend in bot_state
    bot_state_tech = {
        "macro_trend_BTCUSDT_4h": "Bullish",
        "latest_prediction_bg_BTCUSDT_4h": {"direction": "Neutral", "confidence": 0.50}
    }
    bias_tech = signal_evaluator.get_hierarchical_macro_bias(bot_state_tech, symbol="BTCUSDT")
    assert bias_tech["direction"] == "Bullish"

    # Test 2: Fall back to price vs EMA200 when trend string is missing
    bot_state_ema = {
        "close_BTCUSDT_4h": 52000.0,
        "ema200_BTCUSDT_4h": 50000.0,
        "latest_prediction_bg_BTCUSDT_4h": {"direction": "Neutral", "confidence": 0.50}
    }
    bias_ema = signal_evaluator.get_hierarchical_macro_bias(bot_state_ema, symbol="BTCUSDT")
    assert bias_ema["direction"] == "Bullish"

    # Verify main.py stores macro_trend_{symbol}_4h
    with open("main.py", "r") as f:
        main_src = f.read()
    assert "macro_trend_{symbol}_4h" in main_src


# --- Finding #89: Predictor Key Namespacing & Adaptive Stop Parity ---
def test_finding_89_predictor_key_namespacing_and_adaptive_stop_parity():
    with open("signal_evaluator.py", "r") as f:
        sig_src = f.read()

    # Background writes are namespaced under latest_prediction_bg_* and evaluator_prediction_*
    assert "latest_prediction_bg_" in sig_src
    assert "evaluator_prediction_" in sig_src

    # Confirm that raw authoritative writes do not occur in signal_evaluator
    assert 'self.bot_state[f"latest_prediction_{symbol}_{tf_key}"] =' not in sig_src
    assert 'self.bot_state[f"latest_prediction_{tf_key}"] =' not in sig_src

    # Adaptive stop uses structural adaptive stop rather than hardcoded 0.015
    assert "actual_sl_m =" in sig_src

    # Arbitrary 0.55 cap removed from eval_threshold
    assert "eval_threshold = min(0.55" not in sig_src


# --- Finding #90: Label Horizon Invariant & Exit Timer Floor ---
def test_finding_90_label_horizon_invariant_and_exit_timer_floor():
    from exit_policy_engine import ExitPolicyEngine

    engine = ExitPolicyEngine()
    # 4h interval lookahead is 12 bars; even with compressed ATR and low MHI, soft_limit must be >= 12
    res = engine.evaluate_10_level_exit_hierarchy(
        symbol="BTCUSDT",
        interval="240",
        current_price=50100.0,
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=53000.0,
        direction="Buy",
        candles_elapsed=2,
        expected_r=1.5,
        entry_regime="Trending",
        current_regime="Trending",
        garch_vol=0.02,
        rolling_vol_20th_pct=0.01,
        atr_ratio=0.1,    # extreme compression
        mhi_status=10.0   # low MHI
    )
    assert res["soft_limit_candles"] >= 12

    with open("main.py", "r") as f:
        main_src = f.read()

    # Verify end_time label horizon expiry in exit evaluation
    assert 'exit_reason = "HORIZON_EXPIRY [LABEL_TIMEOUT]"' in main_src
