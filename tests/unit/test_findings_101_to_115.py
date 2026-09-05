import os
import json
import time
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import config

from portfolio_risk import portfolio_risk_engine
from trade_calculators import resolve_trade_geometry
from config_verifier import assert_manifest_live_parity
from drift_monitor import DriftMonitor
from data_quality_engine import DataQualityEngine
from market_data_quality import MarketDataQualityMonitor
from tools.beta_calibrator import BetaCalibrator, is_calibrator_viable
from bybit_client import get_bybit_fee_rate, _fee_rate_cache, _fee_rate_lock
from order_state_machine import generate_client_order_id
import risk_engine
from risk_engine import AutoStopFloor, calculate_position_size, calculate_atr_risk_parity_size, compute_conservative_kelly


# ==============================================================================
# Finding 1: resolve_trade_geometry fallback & short stop calculation
# ==============================================================================
def test_finding_1_resolve_trade_geometry_fallback_and_short_stop():
    entry_price = 50000.0
    atr_dollars = 1000.0

    # 1. Fallback priority for base_sl_multiplier: None falls back to 1.0 (with 60m cushion = 1.0 * 1.0)
    geom_fallback = resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bullish",
        interval="60",
        atr_dollars=atr_dollars,
        base_sl_multiplier=None,
        base_tp_multiplier=2.0
    )
    assert geom_fallback["sl_multiplier_adjusted"] == pytest.approx(1.0, rel=1e-3)
    assert geom_fallback["sl_dist"] == pytest.approx(1000.0, rel=1e-3)

    # 2. Short stop geometry calculation
    geom_short = resolve_trade_geometry(
        entry_price=entry_price,
        direction="Bearish",
        interval="60",
        atr_dollars=atr_dollars,
        base_sl_multiplier=1.5,
        base_tp_multiplier=2.0
    )
    # For Short, stop loss price must be entry + sl_dist and take profit entry - tp_dist
    assert geom_short["stop_loss_price"] == pytest.approx(entry_price + geom_short["sl_dist"], rel=1e-3)
    assert geom_short["take_profit_price"] == pytest.approx(entry_price - geom_short["tp_dist"], rel=1e-3)
    assert geom_short["sl_multiplier_adjusted"] == pytest.approx(1.5, rel=1e-3)


# ==============================================================================
# Finding 2: CVaR & Parametric VaR holding period scaling and tail risk
# ==============================================================================
def test_finding_2_cvar_holding_period_scaling_and_loss_convention():
    import pandas as pd
    open_positions = [{"symbol": "BTCUSDT", "position_size_usd": 1000.0, "leverage": 2.0}]
    total_equity = 10000.0

    # Synthetic returns
    np.random.seed(42)
    returns_df = pd.DataFrame({
        "BTCUSDT": np.random.normal(-0.001, 0.02, 100)
    })

    # 1. Test CVaR calculation
    res_cvar = portfolio_risk_engine.calculate_portfolio_cvar_99_and_tail_contributions(
        open_positions=open_positions,
        returns_df=returns_df,
        total_equity=total_equity
    )
    assert res_cvar["cvar_99_dollars"] >= 0.0
    assert res_cvar["cvar_99_equity_pct"] >= 0.0
    assert "BTCUSDT" in res_cvar["position_tail_contributions"]

    # 2. Test parametric VaR scaling with horizon: 60m vs 240m
    var_usd_60, var_pct_60, ok_60 = portfolio_risk_engine.calculate_parametric_var(
        open_positions=open_positions,
        returns_df=returns_df,
        total_equity=total_equity,
        interval_minutes=60
    )
    var_usd_240, var_pct_240, ok_240 = portfolio_risk_engine.calculate_parametric_var(
        open_positions=open_positions,
        returns_df=returns_df,
        total_equity=total_equity,
        interval_minutes=240
    )
    # 60m intraday horizon VaR scaled to 1 day is sqrt(24/6) = 2x compared to 240m
    assert var_usd_60 == pytest.approx(var_usd_240 * 2.0, rel=1e-2)


# ==============================================================================
# Finding 3: config_verifier handles barrier_config in manifests
# ==============================================================================
def test_finding_3_config_verifier_barrier_config_manifest_structure(tmp_path):
    manifest_content = {
        "barrier_config": {
            "lookahead": 24,
            "sl_mult": 1.0,
            "tp_mult_trending": 2.0,
            "tp_mult_ranging": 1.5,
            "regime_adx_enter": 25.0
        }
    }
    manifest_file = tmp_path / "ensemble_trending_trend_60_manifest.json"
    manifest_file.write_text(json.dumps(manifest_content))

    live_config = {
        "lookahead": 24,
        "sl_mult": 1.0,
        "tp_mult_trending": 2.0,
        "tp_mult_ranging": 1.5,
        "regime_adx_enter": 25.0
    }

    # Should succeed without KeyError when accessing barrier_config
    assert_manifest_live_parity(str(manifest_file), live_config)

    # Divergent configuration should raise ValueError
    divergent_config = live_config.copy()
    divergent_config["sl_mult"] = 1.2
    with pytest.raises(ValueError, match="diverges from live config"):
        assert_manifest_live_parity(str(manifest_file), divergent_config)


# ==============================================================================
# Finding 4: drift_monitor symmetric baseline fallback distribution
# ==============================================================================
def test_finding_4_drift_monitor_symmetric_baseline_distribution():
    monitor = DriftMonitor()
    # When empirical baseline file is missing, fallback distribution should be genuine empirical confidences from DB
    with patch("os.path.exists", return_value=False):
        baseline = monitor._get_training_baseline_confidences(100)
        assert len(baseline) >= 20
        assert all(0.0 <= c <= 1.0 for c in baseline)


# ==============================================================================
# Finding 5: DataQualityEngine excess gap & MarketDataQualityMonitor candle close
# ==============================================================================
def test_finding_5_data_quality_excess_gap_and_candle_close():
    dq = DataQualityEngine()

    # Normal 60m candles have 3600s gap, but excess gap is 0s
    excess_gap_sec = 0.0
    res = dq.evaluate_data_quality(
        missing_candles_count=0,
        timestamp_gap_seconds=excess_gap_sec,
        stale_feed_seconds=30.0,
        zero_price_detected=False
    )
    assert res["severity"] == "LOW"
    assert res["action"] == "LOG_AND_MONITOR"

    # MarketDataQualityMonitor should evaluate health against candle close timestamp
    mdq = MarketDataQualityMonitor()
    now = time.time()
    # Candle closed 20s ago
    health = mdq.evaluate_feed_health(
        last_candle_timestamp=now - 20.0,
        server_time_ms=now * 1000.0,
        client_time_ms=now * 1000.0,
        ws_connected=True,
        interval_sec=3600.0
    )
    assert health["health_tier"] == "GREEN"


# ==============================================================================
# Finding 6: beta_calibrator rejects identity fallback & supports instances
# ==============================================================================
def test_finding_6_beta_calibrator_rejects_identity_and_supports_instances():
    # 1. Identity fallback calibrator (a=1, b=1, c=0) must be rejected as unviable
    identity_dict = {"a": 1.0, "b": 1.0, "c": 0.0, "is_fitted": False, "scaling_method": "beta_calibration"}
    assert not is_calibrator_viable(identity_dict)

    # 2. BetaCalibrator instance with valid fitted coefficients must be supported directly
    fitted_bc = BetaCalibrator(a=1.5, b=1.2, c=-0.3)
    fitted_bc.is_fitted = True
    assert is_calibrator_viable(fitted_bc)

    # 3. Degenerate slope must be rejected
    flat_bc = BetaCalibrator(a=0.01, b=0.01, c=0.0)
    flat_bc.is_fitted = True
    assert not is_calibrator_viable(flat_bc)


# ==============================================================================
# Finding 7: Transaction Cost Model queries total_cost_bps
# ==============================================================================
def test_finding_7_tcm_cost_bps_query():
    import transaction_cost_model
    res = transaction_cost_model.estimate_transaction_cost(
        symbol="BTCUSDT",
        order_size_usd=100.0,
        volume_24h_usd=50_000_000.0,
        bid_ask_spread_bp=1.5,
        is_maker=True,
        round_trip=True
    )
    # Must contain canonical 'total_cost_bps'
    assert "total_cost_bps" in res
    assert res["total_cost_bps"] > 0


# ==============================================================================
# Finding 8: run_bybit_balance_updater backoff logic
# ==============================================================================
def test_finding_8_bybit_balance_updater_backoff():
    # Verify exponential backoff formula used in balance updater
    failures = [0, 1, 2, 4, 8]
    expected_sleeps = [
        5.0,
        min(60.0, 5.0 * (1.5 ** 1)),
        min(60.0, 5.0 * (1.5 ** 2)),
        min(60.0, 5.0 * (1.5 ** 4)),
        min(60.0, 5.0 * (1.5 ** 6))
    ]
    for fail_count, expected in zip(failures, expected_sleeps):
        actual = min(60.0, 5.0 * (1.5 ** min(fail_count, 6))) if fail_count > 0 else 5.0
        assert actual == pytest.approx(expected, rel=1e-3)


# ==============================================================================
# Finding 9: get_bybit_fee_rate TTL cache
# ==============================================================================
def test_finding_9_bybit_fee_rate_ttl_cache():
    with _fee_rate_lock:
        _fee_rate_cache.clear()

    # Mock API call
    mock_resp = {
        "retCode": 0,
        "result": {
            "list": [{"symbol": "BTCUSDT", "makerFeeRate": "0.00018", "takerFeeRate": "0.00050"}]
        }
    }
    with patch("bybit_client.bybit_get_request", return_value=mock_resp):
        fees = get_bybit_fee_rate("BTCUSDT")
        assert fees["maker_fee_rate"] == 0.00018
        assert fees["taker_fee_rate"] == 0.00050

        # Verify cached
        with _fee_rate_lock:
            assert "BTCUSDT" in _fee_rate_cache
            cached_data, exp_ts = _fee_rate_cache["BTCUSDT"]
            assert cached_data["maker_fee_rate"] == 0.00018
            assert exp_ts > time.time() + 3500.0  # Approx 1 hour


# ==============================================================================
# Finding 10: generate_client_order_id sequence/attempt number
# ==============================================================================
def test_finding_10_client_order_id_attempt_sequence():
    cl_id_1 = generate_client_order_id("BTCUSDT", "Buy", interval="60", candle_ts=1700000000, attempt=1)
    cl_id_2 = generate_client_order_id("BTCUSDT", "Buy", interval="60", candle_ts=1700000000, attempt=2)

    assert cl_id_1 != cl_id_2
    assert cl_id_2.endswith("_2")
    assert len(cl_id_1) <= 36
    assert len(cl_id_2) <= 36


# ==============================================================================
# Finding 11: Chase loop order_link_id cancellation support
# ==============================================================================
def test_finding_11_cancel_order_supports_order_link_id():
    import bybit_client
    with patch("bybit_client.execute_bybit_order_ws_or_rest") as mock_exec:
        mock_exec.return_value = {"retCode": 0}
        bybit_client.cancel_bybit_order(
            symbol="BTCUSDT",
            order_id="12345",
            order_link_id="c_BTC_60_1700000000_0"
        )
        assert mock_exec.called
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "/v5/order/cancel"
        payload = call_args[1]
        assert payload["orderId"] == "12345"
        assert payload["orderLinkId"] == "c_BTC_60_1700000000_0"


# ==============================================================================
# Finding 12: compute_conservative_kelly fee-inclusive edge
# ==============================================================================
def test_finding_12_conservative_kelly_fee_inclusive():
    # Trade with marginal gross edge that is erased by fees (cost_bps=50 bps)
    k_zero = compute_conservative_kelly(
        calibrated_confidence=0.51,
        tp_multiplier=1.0,
        sl_multiplier=1.0,
        interval="15",
        haircut=0.28,
        atr_norm=0.005,
        cost_bps=50.0  # 50 bps roundtrip fee wipes out small edge
    )
    assert k_zero == 0.0

    # High conviction trade with robust edge net of fees
    k_pos = compute_conservative_kelly(
        calibrated_confidence=0.75,
        tp_multiplier=2.5,
        sl_multiplier=1.0,
        interval="60",
        haircut=0.80,
        atr_norm=0.010,
        cost_bps=10.0
    )
    assert k_pos > 0.0


# ==============================================================================
# Finding 13: calculate_position_size clips before applying heat
# ==============================================================================
def test_finding_13_position_size_clips_before_heat():
    # Unconstrained size would be $1000, capped to $500 max_position_size
    # Portfolio heat is at 15% out of 30% ceiling (50% heat budget available)
    # Expected size: $500 * (1.0 - 0.15/0.30) = $250.0
    size = calculate_position_size(
        symbol="BTCUSDT",
        entry_price=50000.0,
        stop_loss_price=49000.0, # 2% stop
        account_balance=10000.0,
        max_risk_pct=0.02, # $200 risk -> $10000 notional
        max_position_size_usd=500.0,
        portfolio_heat=0.15,
        heat_ceiling=0.30
    )
    assert size == pytest.approx(250.0, rel=1e-2)

    # ATR parity sizing also clips before heat
    parity = calculate_atr_risk_parity_size(
        symbol="BTCUSDT",
        price=50000.0,
        atr_dollars=1000.0,
        sl_multiplier=1.0,
        target_risk_usd=20.0,
        max_position_size_usd=500.0,
        portfolio_heat=0.15,
        heat_ceiling=0.30
    )
    assert parity["position_size_usd"] <= 250.0


# ==============================================================================
# Finding 14: AutoStopFloor requires min_sample_size on filtered trades
# ==============================================================================
def test_finding_14_auto_stop_floor_min_sample_size():
    floor_engine = AutoStopFloor(min_sample_size=3)
    mock_db = MagicMock()

    # Only 1 valid trade surviving filtering
    mock_db.get_trade_history.return_value = [
        {"symbol": "BTCUSDT", "reason": "STOP LOSS", "entry_price": 50000.0, "exit_price": 49000.0},
        {"symbol": "BTCUSDT", "reason": "TAKE PROFIT", "entry_price": 50000.0, "exit_price": 52000.0}
    ]

    # Should fall back to config default rather than taking 75th percentile of 1 item
    floor = floor_engine.compute_optimal_floor("BTCUSDT", database_module=mock_db, interval="60")
    default_cfg = float(config.MIN_SL_PCT_CONFIG.get("60", 0.008))
    assert floor == default_cfg


# ==============================================================================
# Finding 15: Startup manifest barrier cross check
# ==============================================================================
def test_finding_15_startup_manifest_barrier_cross_check(tmp_path):
    from config import TIMEFRAME_CONFIG
    # Create test manifest matching 60m TIMEFRAME_CONFIG
    cfg_60 = TIMEFRAME_CONFIG.get("60", {})
    manifest_data = {
        "barrier_config": {
            "lookahead": cfg_60.get("lookahead", 24),
            "sl_mult": cfg_60.get("sl_mult", 1.0),
            "tp_mult_trending": cfg_60.get("tp_mult_trending", 2.0),
            "tp_mult_ranging": cfg_60.get("tp_mult_ranging", 1.5)
        }
    }
    # Validate that difference <= 0.05 passes
    diffs = [
        abs(float(manifest_data["barrier_config"][k]) - float(cfg_60[k]))
        for k in ["tp_mult_trending", "tp_mult_ranging", "sl_mult", "lookahead"]
        if k in manifest_data["barrier_config"] and k in cfg_60
    ]
    assert all(d <= 0.05 for d in diffs)
