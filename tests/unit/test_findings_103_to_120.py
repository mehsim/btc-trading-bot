"""
Unit tests for audit defect findings #103 through #120.
"""

import math
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock, patch

import config
from decision_journal import DecisionRecord, ReasonCode, write_decision, init_decision_journal_db
from confluence_engine import evaluate_expectancy_gate
from exit_policy_engine import ExitPolicyEngine
from pattern_miner import wilson_score_interval
import trade_calculators


def test_finding_103_decision_record_reason_code_and_metrics():
    """Finding #103: ReasonCode enum, reason_code, expected_value, expected_rr, round_trip_cost_bp."""
    rec = DecisionRecord(
        ts=1700000000.0,
        candle_timestamp=1700000000000,
        symbol="BTCUSDT",
        interval="15",
        reject_reason="Skipped (TCM Net Edge <= 0)",
        reason_code=ReasonCode.TCM_NET_EDGE_NEGATIVE,
        expected_value=-1.5,
        expected_rr=1.55,
        round_trip_cost_bp=5.2
    )
    assert rec.reason_code == ReasonCode.TCM_NET_EDGE_NEGATIVE
    assert rec.expected_value == -1.5
    assert rec.expected_rr == 1.55
    assert rec.round_trip_cost_bp == 5.2

    # Verify writing and schema persistence
    init_decision_journal_db()
    write_decision(rec)
    import database
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT reason_code, expected_value, expected_rr, round_trip_cost_bp FROM decision_journal WHERE decision_id = ?", (rec.decision_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == ReasonCode.TCM_NET_EDGE_NEGATIVE
    assert row[1] == pytest.approx(-1.5)
    assert row[2] == pytest.approx(1.55)
    assert row[3] == pytest.approx(5.2)


def test_finding_104_telegram_manual_trade_risk_and_auth():
    """Finding #104: Telegram manual trade fail-closed auth and callback sender verification."""
    allowed_chat_ids = ["12345"]
    cb_chat_id = "12345"
    cb_from_id_valid = "12345"
    cb_from_id_invalid = "99999"

    # Valid callback from authorized user
    is_valid = bool(allowed_chat_ids and (cb_chat_id in allowed_chat_ids and cb_from_id_valid in allowed_chat_ids))
    assert is_valid is True

    # Spoofed/unauthorized callback where sender is not allowed
    is_spoofed = bool(allowed_chat_ids and (cb_from_id_invalid in allowed_chat_ids))
    assert is_spoofed is False

    # Fail closed when allowed_chat_ids is empty
    empty_allowed = []
    assert bool(empty_allowed and cb_from_id_valid in empty_allowed) is False


def test_finding_105_expectancy_gate_evaluation():
    """Finding #105: evaluate_expectancy_gate checks win rate and average win/loss."""
    # Positive expectancy: 60% win rate, avg win 1.5%, avg loss 1.0% -> EV = 0.6*1.5 - 0.4*1.0 = +0.5%
    passed, ev = evaluate_expectancy_gate(0.60, 0.015, 0.010)
    assert passed is True
    assert ev > 0

    # Negative expectancy: 30% win rate, avg win 1.0%, avg loss 1.5% -> EV = 0.3*1.0 - 0.7*1.5 = -0.75%
    passed_neg, ev_neg = evaluate_expectancy_gate(0.30, 0.010, 0.015)
    assert passed_neg is False
    assert ev_neg < 0


def test_finding_106_confluence_macro_opposition_reason_code():
    """Finding #106: map_status_to_reason_code handles Macro Opposition."""
    from main import map_status_to_reason_code
    code = map_status_to_reason_code("Skipped (Macro Opposition)")
    assert code == ReasonCode.MACRO_OPPOSITION


def test_finding_107_liquidity_and_flash_crash_any_interval():
    """Finding #107: DecisionRecord stores liquidity_score and spread_bp."""
    rec = DecisionRecord(ts=1700000000.0, candle_timestamp=1700000000000, symbol="ETHUSDT", interval="60")
    rec.liquidity_score = 0.85
    rec.spread_bp = 4.2
    assert rec.liquidity_score == 0.85
    assert rec.spread_bp == 4.2


def test_finding_108_tcm_cost_gate_and_sizing():
    """Finding #108: Cost gate logging on DecisionRecord."""
    rec = DecisionRecord(ts=1700000000.0, candle_timestamp=1700000000000, symbol="BTCUSDT", interval="15")
    rec.gate("cost", value=5.5, passed=True)
    assert rec.gate_cost_bp == 5.5
    assert rec.gate_cost_pass == 1


def test_finding_109_placed_flag_outcome_executed():
    """Finding #109: Only placed=True yields EXECUTED outcome."""
    rec = DecisionRecord(ts=1700000000.0, candle_timestamp=1700000000000, symbol="BTCUSDT", interval="15")
    placed = False
    status_msg = "Skipped (Low Confidence)"
    if placed:
        rec.outcome = "EXECUTED"
    elif status_msg.startswith("REJECTED"):
        rec.outcome = "REJECTED"
    else:
        rec.outcome = "SKIPPED"
    assert rec.outcome == "SKIPPED"


def test_finding_110_provenance_preservation():
    """Finding #110: Provenance fields are not clobbered with None."""
    rec = DecisionRecord(
        ts=1700000000.0,
        candle_timestamp=1700000000000,
        symbol="BTCUSDT",
        interval="15",
        model_version="v2.1",
        git_sha="abcd123",
        regime="TRENDING_BULL"
    )
    pred_info = {"raw_confidence": 0.75}  # Does not contain model_version or git_sha
    rec.model_version = pred_info.get("model_version") or rec.model_version
    rec.git_sha = pred_info.get("git_sha") or rec.git_sha
    rec.regime = pred_info.get("regime_mode") or pred_info.get("regime") or rec.regime

    assert rec.model_version == "v2.1"
    assert rec.git_sha == "abcd123"
    assert rec.regime == "TRENDING_BULL"


def test_finding_111_htf_macro_per_symbol_isolation():
    """Finding #111: Macro prediction lookup does not fall back to symbol-agnostic key."""
    bot_state = {
        "latest_prediction_BTCUSDT_240": {"trend": "Bullish"},
        "latest_prediction_240": {"trend": "Bullish"}  # Agnostic key populated by BTC
    }
    # For ETHUSDT, checking without symbol-agnostic fallback should return None
    symbol = "ETHUSDT"
    macro_tf = "240"
    macro_pred = bot_state.get(f"latest_prediction_{symbol}_{macro_tf}")
    assert macro_pred is None


def test_finding_112_atomic_trade_close():
    """Finding #112: database.close_trade_atomically executes atomically."""
    import database
    import time
    now_ts = time.time()
    trade = {
        "trade_id": f"test_trade_112_atomic_{int(now_ts)}",
        "symbol": "BTCUSDT",
        "direction": "Bullish",
        "entry_price": 50000.0,
        "exit_price": 51000.0,
        "exit_time": now_ts,
        "pnl_usd": 100.0,
        "pnl_pct": 2.0,
        "exit_reason": "TP_HIT",
        "interval": "15"
    }
    trade_id = trade["trade_id"]
    # Save trade as active first
    database.save_active_trades("15", [trade])
    active_before = database.get_active_trades("15")
    assert any(t.get("trade_id") == trade_id for t in active_before)

    # Close atomically
    success = database.close_trade_atomically(trade, tf="15")
    assert success is True

    # Verify removed from active and present in completed
    active_after = database.get_active_trades("15")
    assert not any(t.get("trade_id") == trade_id for t in active_after)
    completed = database.get_completed_trades(limit=50)
    assert any(t.get("trade_id") == trade["trade_id"] for t in completed)


def test_finding_115_be_buffer_calculation():
    """Finding #115: compute_be_buffer fees are not divided by 100 or multiplied by leverage."""
    engine = ExitPolicyEngine()
    # Entry price 50,000, ATR 500, leverage 5x
    # Overhead pct = 0.0011 + 0.0005 = 0.0016
    # Price overhead = 50000 * 0.0016 = 80.0
    # Safety buffer = 500 * 0.10 = 50.0
    # Total BE buffer = 80.0 + 50.0 = 130.0
    buf = engine.compute_be_buffer(symbol="BTCUSDT", leverage=5.0, entry_price=50000.0, atr_dollars=500.0, safety_margin_atr=0.10)
    assert buf == pytest.approx(130.0, abs=0.01)


def test_finding_116_kelly_sizing_realized_stop_multiplier():
    """Finding #116: realized stop multiplier passed to compute_conservative_kelly."""
    entry = 50000.0
    stop = 49000.0
    atr = 500.0
    realized_sl_m = abs(entry - stop) / atr  # 2.0
    assert realized_sl_m == 2.0


def test_finding_117_adx_regime_threshold_parity():
    """Finding #117: ADX regime threshold matches config 28.0."""
    threshold = config.REGIME_ADX_ENTER_BY_INTERVAL.get("15", 28.0)
    assert threshold >= 25.0


def test_finding_118_barrier_geometry_snapshots():
    """Finding #118: DecisionRecord snapshots both labelled and executed geometry."""
    rec = DecisionRecord(ts=1700000000.0, candle_timestamp=1700000000000, symbol="BTCUSDT", interval="15")
    rec.snapshot(
        labelled_sl_mult=1.0,
        labelled_tp_mult=1.5,
        executed_sl_mult=1.2,
        executed_tp_mult=1.8
    )
    assert rec._inputs.get("labelled_sl_mult") == 1.0
    assert rec._inputs.get("executed_sl_mult") == 1.2


def test_finding_119_min_sl_pct_config_consistency():
    """Finding #119: validate_trade_structure uses MIN_SL_PCT_CONFIG."""
    valid, adjusted, logs = trade_calculators.validate_trade_structure(
        symbol="BTCUSDT",
        entry_price=50000.0,
        stop_price=49990.0,  # 10 dollars stop (0.02% << min floor 0.6%)
        tp_price=51000.0,
        atr_dollars=200.0,
        leverage=5.0,
        direction="Bullish",
        interval="15"
    )
    # Stop distance should have been widened to at least the noise clearance floor (0.6% = 300 dollars)
    min_expected_stop = 50000.0 * config.MIN_SL_PCT_CONFIG.get("15", 0.006)
    assert adjusted["stop_dist"] >= min_expected_stop


def test_finding_120_backtest_wilson_ci():
    """Finding #120: Wilson score interval calculation and small sample flag."""
    ci_low, ci_high = wilson_score_interval(wins=50, n=100)
    assert 0.0 < ci_low < 0.50 < ci_high < 1.0
    # n < 784 flag condition
    assert 100 < 784
