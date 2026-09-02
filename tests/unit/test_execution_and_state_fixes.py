import pytest
from state_manager import default_json_serializer, StateManager
from trade_calculators import get_realized_rr_haircut, estimate_empirical_realized_rr
from production_regime_engine import production_regime_engine


def test_default_json_serializer_fallback_str():
    """Verify default_json_serializer falls back to str(obj) instead of raising or returning None."""
    class CustomObject:
        def __str__(self):
            return "custom_str_repr"
    obj = CustomObject()
    res = default_json_serializer(obj)
    assert res == "custom_str_repr"


def test_state_manager_bool_truthiness():
    """Verify StateManager truthiness is True without executing Redis KEYS scans."""
    sm = StateManager()
    assert bool(sm) is True


def test_realized_rr_haircut_capped_at_point_six():
    """Verify get_realized_rr_haircut cannot exceed 0.60 even when empirical R:R is high."""
    closed_trades = (
        [{"interval": "15", "regime": "trending", "pnl_usd": 300.0}] * 10 +
        [{"interval": "15", "regime": "trending", "pnl_usd": -50.0}] * 10
    )
    # empirical_rr = 300 / 50 = 6.0, nominal_rr = 1.5 -> ratio = 4.0 -> capped at 0.60
    haircut = get_realized_rr_haircut(
        interval="15", regime="trending", nominal_rr=1.5, closed_trades=closed_trades
    )
    assert haircut == 0.60


def test_estimate_empirical_realized_rr_pnl_usd_and_regime():
    """Verify estimate_empirical_realized_rr recognizes pnl_usd and normalizes regime string."""
    closed_trades = (
        [{"interval": "15", "regime": "Trending_Bullish", "pnl_usd": 200.0}] * 5 +
        [{"interval": "15", "regime": "STRONG_TREND", "pnl_usd": 200.0}] * 5 +
        [{"interval": "15", "regime": "TRENDING", "pnl_usd": -100.0}] * 10
    )
    rr = estimate_empirical_realized_rr(closed_trades=closed_trades, interval="15", regime="trending")
    assert rr is not None
    assert round(rr, 2) == 2.0


def test_high_vol_ranging_normalizes_to_ranging():
    """Verify High Vol, Ranging normalizes to RANGING rather than CHOPPY."""
    reg = production_regime_engine._normalize_regime("High Vol, Ranging")
    assert reg == "RANGING"
