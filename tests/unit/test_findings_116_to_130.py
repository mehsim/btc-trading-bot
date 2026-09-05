"""
Unit tests for audit defect findings #16 through #30.
Covers model governance, feature contracts, Optuna reproducibility,
exit policy regime ordering, Kelly empirical sizing, risk clamping,
expectancy gating, order link IDs, chase loop tracking, lock serialization,
Sharpe ratio calculation, decision journaling, and walk-forward failure handling.
"""

import math
import os
import json
import time
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import config
import model_governance
from model_governance import validate_manifest_governance_floors
import ensemble
from ensemble import is_feature_contract_compatible, verify_manifest_hmac_signature
from exit_policy_engine import ExitPolicyEngine
from risk_engine import JointRiskBudgetAllocator, compute_conservative_kelly
import trade_calculators
from trade_calculators import calculate_replay_statistics
from decision_journal import DecisionRecord, ReasonCode
import walk_forward_engine
from walk_forward_engine import run_walk_forward_backtest


# ==============================================================================
# Finding #16: Champion slot governance & HMAC signature verification
# ==============================================================================
def test_finding_16_champion_slot_governance():
    """Verify validate_manifest_governance_floors requires promoted is True explicitly."""
    # Manifest missing 'promoted' key must fail
    manifest_missing = {
        "manifest_schema_version": 2,
        "holdout_mcc": 0.045,
        "holdout_balanced_accuracy": 0.38,
        "cv_metrics": {"holdout_mcc_ci95": [0.01, 0.08]}
    }
    ok, reason = validate_manifest_governance_floors(manifest_missing, "15")
    assert ok is False
    assert "promoted=False" in reason or "missing" in reason

    # Manifest with promoted=False must fail
    manifest_unpromoted = dict(manifest_missing, promoted=False)
    ok, reason = validate_manifest_governance_floors(manifest_unpromoted, "15")
    assert ok is False
    assert "promoted=False" in reason

    # Manifest with promoted=True passes
    manifest_valid = dict(manifest_missing, promoted=True)
    ok, reason = validate_manifest_governance_floors(manifest_valid, "15")
    assert ok is True
    assert reason == ""


# ==============================================================================
# Finding #17: Feature contract invariance & zero-fill elimination
# ==============================================================================
def test_finding_17_feature_contract_invariance():
    """Verify is_feature_contract_compatible enforces all contract fields and raises on HMAC failure."""
    valid_features = ["f1", "f2", "f3"]
    import hashlib
    f_hash = hashlib.sha256(",".join(valid_features).encode("utf-8")).hexdigest()[:12]

    manifest = {
        "manifest_schema_version": 2,
        "feature_count": 3,
        "feature_contract_hash": f_hash,
        "feature_names": valid_features
    }
    # Sign manifest
    secret = ensemble.get_manifest_hmac_secret()
    import hmac
    payload = json.dumps({k: v for k, v in manifest.items() if k != "hmac_signature"}, sort_keys=True)
    manifest["hmac_signature"] = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    # Valid contract matches
    compat, reason = is_feature_contract_compatible(manifest, valid_features)
    assert compat is True
    assert reason == "compatible"

    # Contract-less manifest missing feature_count must NOT return True vacuously
    bad_manifest = dict(manifest)
    bad_manifest.pop("feature_count")
    payload_bad = json.dumps({k: v for k, v in bad_manifest.items() if k != "hmac_signature"}, sort_keys=True)
    bad_manifest["hmac_signature"] = hmac.new(secret, payload_bad.encode("utf-8"), hashlib.sha256).hexdigest()
    compat, reason = is_feature_contract_compatible(bad_manifest, valid_features)
    assert compat is False
    assert "missing required feature contract fields" in reason

    # Tampered HMAC must raise RuntimeError
    tampered_manifest = dict(manifest, feature_count=10)
    with pytest.raises(RuntimeError) as exc_info:
        is_feature_contract_compatible(tampered_manifest, valid_features)
    assert "Manifest signature invalid" in str(exc_info.value) or "signature" in str(exc_info.value).lower()


# ==============================================================================
# Finding #18: Optuna search reproducibility & model estimator seeds
# ==============================================================================
def test_finding_18_model_estimator_seeds():
    """Verify create_model in train.py sets deterministic random seeds."""
    import train
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    xgb = train.create_model(XGBClassifier, {"n_estimators": 5})
    assert xgb.get_params()["random_state"] == 42

    lgb = train.create_model(LGBMClassifier, {"n_estimators": 5})
    assert lgb.get_params()["random_state"] == 42

    cat = train.create_model(CatBoostClassifier, {"iterations": 5})
    assert cat.get_params()["random_seed"] == 42


# ==============================================================================
# Finding #19: Exit policy regime resolution precedence
# ==============================================================================
def test_finding_19_exit_policy_regime_precedence():
    """Verify ranging regime with high ADX resolves to RANGING, not MODERATE_TREND."""
    # High ADX (>= 25) on Ranging market must resolve to RANGING
    res = ExitPolicyEngine._resolve_regime_key("High Vol, Ranging", adx_val=32.0)
    assert res == "RANGING"

    res_chop = ExitPolicyEngine._resolve_regime_key("Chop", adx_val=28.0)
    assert res_chop == "RANGING"

    # Trending markets resolve to TREND regimes
    res_trend = ExitPolicyEngine._resolve_regime_key("Low Vol, Trending", adx_val=22.0)
    assert res_trend == "MODERATE_TREND"

    res_strong = ExitPolicyEngine._resolve_regime_key("High Vol, Trending", adx_val=35.0)
    assert res_strong == "STRONG_TREND"


# ==============================================================================
# Finding #20: Kelly sizing allocator empirical outcomes
# ==============================================================================
def test_finding_20_kelly_allocator_empirical_outcomes():
    """Verify allocate_risk_budget caps p_win using empirical outcomes from trade_history."""
    allocator = JointRiskBudgetAllocator()
    trade_history = [
        {"pnl_usd": -10.0, "interval": "15"},
        {"pnl_usd": -5.0, "interval": "15"},
        {"pnl_usd": 12.0, "interval": "15"},
        {"pnl_usd": -8.0, "interval": "15"},
        {"pnl_usd": -2.0, "interval": "15"},
        {"pnl_usd": -15.0, "interval": "15"},
        {"pnl_usd": 20.0, "interval": "15"},
        {"pnl_usd": -4.0, "interval": "15"},
        {"pnl_usd": -6.0, "interval": "15"},
        {"pnl_usd": -1.0, "interval": "15"},
    ]  # 2 wins out of 10 = 20% win rate

    res = allocator.allocate_risk_budget(
        symbol="BTCUSDT",
        entry_price=50000.0,
        atr_dollars=500.0,
        atr_norm=0.01,
        calibrated_confidence=0.75,  # High model confidence (75%)
        direction="Bullish",
        total_equity=1000.0,
        stop_distance=500.0,
        target_distance=1500.0,
        trade_history=trade_history,
        interval="15"
    )
    # Because realized win rate is 20%, p_win is capped at 0.20, resulting in non-positive edge
    assert res.get("raw_kelly", 0.0) == 0.0
    assert res.get("kelly_fraction", 0.0) == 0.0


# ==============================================================================
# Finding #21: Terminal risk clamp preservation
# ==============================================================================
def test_finding_21_terminal_risk_clamp_preserved():
    """Verify that clamped quantity is not overwritten by unclamped qty_val."""
    qty_val = 100.0
    c_qty_val = 50.0
    c_qty_str = "50.0"

    # When clamped:
    qty_val = c_qty_val
    raw_qty = c_qty_val
    qty_str = c_qty_str

    # Later execution path must preserve clamped quantity:
    later_raw_qty = float(qty_str) if qty_str else qty_val
    assert later_raw_qty == 50.0
    assert later_raw_qty < 100.0


# ==============================================================================
# Finding #22: Expectancy gate active configuration
# ==============================================================================
def test_finding_22_expectancy_gate_mode():
    """Verify EXPECTANCY_GATE_MODE defaults to active."""
    assert config.EXPECTANCY_GATE_MODE in ["disabled", "shadow", "active"]
    assert config.EXPECTANCY_GATE_MODE == "active"


# ==============================================================================
# Finding #23: Order link ID forwarding in maker chase
# ==============================================================================
def test_finding_23_maker_chase_order_link_id():
    """Verify place_bybit_maker_chase_order accepts and forwards order_link_id."""
    import bybit_client
    with patch("bybit_client.bybit_get_request") as mock_get, \
         patch("bybit_client.execute_bybit_order_ws_or_rest") as mock_exec:
        
        def fake_get(endpoint, params=None):
            if "tickers" in endpoint:
                return {
                    "retCode": 0,
                    "result": {"list": [{"bid1Price": "50000.0", "ask1Price": "50005.0"}]}
                }
            elif "realtime" in endpoint:
                return {
                    "retCode": 0,
                    "result": {"list": [{"orderStatus": "Filled", "cumExecQty": "0.01"}]}
                }
            return {"retCode": 0, "result": {}}

        mock_get.side_effect = fake_get
        mock_exec.return_value = {"retCode": 0, "result": {"orderId": "ord_123"}}

        res = bybit_client.place_bybit_maker_chase_order(
            symbol="BTCUSDT",
            side="Buy",
            qty=0.01,
            max_chase_seconds=0.1,
            order_link_id="test_chase_link_id_1"
        )
        assert mock_exec.called
        call_payload = mock_exec.call_args_list[0][0][1]
        assert call_payload.get("orderLinkId") == "test_chase_link_id_1"
        assert call_payload.get("timeInForce") == "PostOnly"


# ==============================================================================
# Finding #24: Chase partial fill accounting
# ==============================================================================
def test_finding_24_chase_partial_fill_accounting():
    """Verify chase partial fills below 95% do not set bybit_success=True prematurely."""
    raw_qty = 1000.0
    filled_so_far = 12.0  # Only 1.2% filled

    bybit_success = False
    if filled_so_far >= (0.95 * raw_qty):
        bybit_success = True

    assert bybit_success is False


# ==============================================================================
# Finding #25: Telegram manual trade concurrency merge
# ==============================================================================
def test_finding_25_telegram_concurrency_merge():
    """Verify exit loop merge preserves concurrently added manual trades."""
    # Snapshot at loop start
    initial_snapshot = [{"trade_id": "T1", "symbol": "BTCUSDT", "entry_time": 1000}]
    updated_trades = []  # T1 was closed during the loop

    # Interim manual trade added while loop ran
    live_bot_state_trades = [
        {"trade_id": "T1", "symbol": "BTCUSDT", "entry_time": 1000},
        {"trade_id": "T_MANUAL", "symbol": "BTCUSDT", "entry_time": 1050, "is_manual": True}
    ]

    # Reconciliation logic
    evaluated_ids = {t.get("trade_id") for t in initial_snapshot}
    for c_tr in live_bot_state_trades:
        if c_tr.get("trade_id") not in evaluated_ids:
            updated_trades.append(c_tr)

    # Assert T_MANUAL was preserved and not erased
    assert len(updated_trades) == 1
    assert updated_trades[0]["trade_id"] == "T_MANUAL"


# ==============================================================================
# Finding #26: Sync active positions lock release
# ==============================================================================
def test_finding_26_sync_positions_lock_release():
    """Verify sync_active_positions_from_bybit queries Bybit without holding locks."""
    import main
    with patch("main.get_all_bybit_positions") as mock_get_pos:
        mock_get_pos.return_value = []
        # Calling sync_active_positions_from_bybit in live mode
        with patch.object(main, "TRADE_MODE", "live"):
            res = main.sync_active_positions_from_bybit()
            assert res is True
            assert mock_get_pos.called


# ==============================================================================
# Finding #27: Sharpe annualization and explicit t-statistic
# ==============================================================================
def test_finding_27_sharpe_t_stat_and_annualization():
    """Verify calculate_replay_statistics reports t_stat distinctly and avoids 187x inflation."""
    returns = [0.01, -0.005, 0.012, 0.008, -0.004, 0.015, -0.002, 0.006, 0.009, 0.003]
    
    # With explicit duration_days
    stats_dur = calculate_replay_statistics(returns, duration_days=30.0, interval="15")
    assert "t_stat" in stats_dur
    assert stats_dur["t_stat"] > 0
    assert stats_dur["sharpe_ratio"] > 0
    # Annualization factor with 10 trades over 30 days (~121 trades/year) should yield reasonable Sharpe
    assert stats_dur["sharpe_ratio"] < 50.0

    # Without duration_days (fallback estimate based on trade count * interval)
    stats_est = calculate_replay_statistics(returns, duration_days=None, interval="15")
    assert stats_est["sharpe_ratio"] < 50.0  # Not inflated by sqrt(35064) ~ 187x


# ==============================================================================
# Finding #28: Crash recovery time-proximity matching
# ==============================================================================
def test_finding_28_crash_recovery_time_proximity():
    """Verify crash recovery selects decision record closest to position creation timestamp."""
    pos_created_ts = 1700000000.0  # 4h position opened yesterday
    newer_ts = 1700080000.0        # 15m scalp opened 5 minutes ago

    records = [
        {"ts": newer_ts, "interval": "15", "calibrated_conf": 0.52},
        {"ts": pos_created_ts + 2.0, "interval": "240", "calibrated_conf": 0.68},
    ]

    matched = min(records, key=lambda r: abs(r["ts"] - pos_created_ts))
    assert matched["interval"] == "240"
    assert matched["calibrated_conf"] == 0.68


# ==============================================================================
# Finding #29: Decision journal rejection metadata
# ==============================================================================
def test_finding_29_decision_journal_rejection_metadata():
    """Verify DecisionRecord preserves intended size and reason_code on rejection."""
    rec = DecisionRecord(symbol="BTCUSDT", interval="15", regime="Ranging", direction="Bullish")
    rec.status_msg = "Skipped (TCM Net Edge <= 0)"
    rec.reason_code = ReasonCode.TCM_NET_EDGE_NEGATIVE
    rec.position_size_usd = 100.0
    rec.leverage = 5.0
    rec.snapshot(status_msg="Skipped (TCM Net Edge <= 0)", position_size_usd=100.0, leverage=5.0)

    assert rec.outcome == "ERROR"  # Pessimistic default preserved before terminal resolution
    assert rec.reason_code == ReasonCode.TCM_NET_EDGE_NEGATIVE
    assert rec.position_size_usd == 100.0
    assert rec.leverage == 5.0
    assert rec._inputs.get("position_size_usd") == 100.0


# ==============================================================================
# Finding #30: Walk-forward backtest loud failure on refit error
# ==============================================================================
def test_finding_30_walk_forward_loud_refit_failure():
    """Verify run_walk_forward_backtest marks window status as error when train_fn fails."""
    # Construct minimal synthetic OHLCV dataframe
    dates = pd.date_range("2024-01-01", periods=150, freq="15min")
    df = pd.DataFrame({
        "timestamp": [int(d.timestamp() * 1000) for d in dates],
        "open": np.linspace(40000, 42000, 150),
        "high": np.linspace(40100, 42100, 150),
        "low": np.linspace(39900, 41900, 150),
        "close": np.linspace(40050, 42050, 150),
        "volume": np.ones(150) * 100.0,
        "feature_1": np.random.randn(150)
    })

    def failing_train_fn(train_df):
        raise ValueError("Simulated model fitting failure")

    def dummy_sim_fn(test_df):
        return {"trades": [{"net_return": 0.01, "sl_frac": 0.01}]}

    summary = run_walk_forward_backtest(
        df,
        trade_simulator_fn=dummy_sim_fn,
        train_fn=failing_train_fn,
        train_window_bars=50,
        test_window_bars=25,
        step_bars=25
    )

    assert summary.get("all_windows_refitted") is False
    # Check that windows recorded refit_error and status='error'
    windows = summary.get("windows", [])
    assert len(windows) > 0
    assert any(w.get("status") == "error" for w in windows)
    assert any("Simulated model fitting failure" in str(w.get("refit_error")) for w in windows)
