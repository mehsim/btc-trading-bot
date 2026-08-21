import pytest
from unittest.mock import patch
from data import get_orderbook_imbalance
from trade_calculators import get_liquidity_score, SYMBOL_LIQUIDITY_BENCHMARKS


def test_import_get_orderbook_imbalance_from_data():
    """Verify get_orderbook_imbalance is exported and directly importable from data.py."""
    assert callable(get_orderbook_imbalance)


def test_get_liquidity_score_fail_closed_on_invalid_or_missing_symbol():
    """Verify that an invalid/illiquid/failing symbol fails closed to score 0.0, NOT 1.0."""
    with patch("data.get_orderbook_imbalance", return_value={"total_depth": 0.0, "spread": 0.001}):
        score = get_liquidity_score("SOMEILLIQUIDCOIN")
        assert score == 0.0, f"Expected 0.0 for illiquid symbol, got {score}"


def test_get_liquidity_score_on_network_error():
    """Verify get_liquidity_score returns 0.0 on exception rather than fake perfect 1.0."""
    with patch("data.get_orderbook_imbalance", side_effect=RuntimeError("API Failure")):
        score = get_liquidity_score("BTCUSDT")
        assert score == 0.0, f"Expected fail-closed 0.0 on exception, got {score}"


def test_get_liquidity_score_dynamic_turnover_benchmark():
    """Verify dynamic benchmark calculation scaling from 24h turnover."""
    # $100M 24h turnover -> benchmark = $50,000 depth. Total depth $25k -> 0.50 score
    with patch("data.get_orderbook_imbalance", return_value={"total_depth": 25_000.0, "spread": 0.0005}):
        score = get_liquidity_score("SMALLCOIN", turnover_24h=100_000_000.0)
        assert score == pytest.approx(0.50)


def test_get_liquidity_score_dynamic_symbol_tier_benchmarks():
    """Verify per-symbol tier benchmark resolution."""
    # BTCUSDT benchmark $400k. $200k depth -> 0.50 score
    with patch("data.get_orderbook_imbalance", return_value={"total_depth": 200_000.0, "spread": 0.0005}):
        btc_score = get_liquidity_score("BTCUSDT")
        assert btc_score == pytest.approx(0.50)

    # DOTUSDT benchmark $50k. $100k depth -> 1.0 max score
    with patch("data.get_orderbook_imbalance", return_value={"total_depth": 100_000.0, "spread": 0.0005}):
        dot_score = get_liquidity_score("DOTUSDT")
        assert dot_score == pytest.approx(1.0)


def test_get_liquidity_score_dynamic_spread_penalty():
    """Verify spread penalty degrades liquidity score when spread is wide."""
    # Good depth ($400k BTC benchmark), but wide spread (0.325% spread) -> 50% spread multiplier penalty
    with patch("data.get_orderbook_imbalance", return_value={"total_depth": 400_000.0, "spread": 0.00325}):
        score = get_liquidity_score("BTCUSDT")
        assert score == pytest.approx(0.50)

    # Extreme spread (> 0.50%) -> forced 0.0 liquidity score
    with patch("data.get_orderbook_imbalance", return_value={"total_depth": 5_000_000.0, "spread": 0.0060}):
        score = get_liquidity_score("BTCUSDT")
        assert score == 0.0
