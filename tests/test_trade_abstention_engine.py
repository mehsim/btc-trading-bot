"""
Unit tests for Institutional Trade Abstention Engine & Replay Dataset Logger.
"""

import pytest
import os
from trade_abstention_engine import trade_abstention_engine
from abstention_dataset import abstention_dataset_logger

def test_execute_high_utility_trade():
    decision, score, reasons, metrics = trade_abstention_engine.evaluate_abstention(
        symbol="BTCUSDT", direction="BUY", expected_return_pct=0.020,
        calibrated_confidence=0.85, roundtrip_fee_pct=0.0010, spread_pct=0.0002
    )
    assert decision == "EXECUTE"
    assert score >= 70.0
    assert "APPROVED" in reasons[0]


def test_abstain_negative_net_return():
    # Fees and spread exceed expected return
    decision, score, reasons, metrics = trade_abstention_engine.evaluate_abstention(
        symbol="BTCUSDT", direction="BUY", expected_return_pct=0.0010,
        calibrated_confidence=0.55, roundtrip_fee_pct=0.0015, spread_pct=0.0008
    )
    assert decision == "ABSTAIN"
    assert score < 55.0
    assert any("<=" in r for r in reasons)


def test_wait_temporary_high_spread():
    decision, score, reasons, metrics = trade_abstention_engine.evaluate_abstention(
        symbol="BTCUSDT", direction="BUY", expected_return_pct=0.025,
        calibrated_confidence=0.80, spread_pct=0.0010
    )
    assert decision == "WAIT"


def test_execute_reduced_moderate_heat():
    decision, score, reasons, metrics = trade_abstention_engine.evaluate_abstention(
        symbol="BTCUSDT", direction="BUY", expected_return_pct=0.015,
        calibrated_confidence=0.72, portfolio_heat=0.14
    )
    assert decision in ("EXECUTE_REDUCED", "ABSTAIN")


def test_abstain_high_portfolio_heat_and_opportunity_cost():
    decision, score, reasons, metrics = trade_abstention_engine.evaluate_abstention(
        symbol="BTCUSDT", direction="BUY", expected_return_pct=0.012,
        calibrated_confidence=0.70, opportunity_cost_r=0.95, portfolio_heat=0.19
    )
    assert decision == "ABSTAIN"
    assert any("Opportunity Cost" in r or "Heat" in r for r in reasons)


def test_abstention_dataset_logger():
    abstention_dataset_logger.log_abstention_event(
        trade_id="BTC_TEST_123", symbol="BTCUSDT", direction="BUY",
        decision="ABSTAIN", score=42.5, reasons=["Expected Return <= 0"], metrics={"test": 1}
    )
    summary = abstention_dataset_logger.get_summary()
    assert summary["total_events"] > 0
    assert summary["abstain_count"] > 0
