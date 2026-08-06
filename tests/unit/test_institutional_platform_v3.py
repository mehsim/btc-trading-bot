import pytest
import config
from strategy_health_engine import StrategyHealthEngine, strategy_health_engine
from trade_repository import SQLiteTradeRepository

def test_3state_feature_flags_and_weights():
    assert config.EXIT_QUALITY_MODE in ["disabled", "shadow", "active"]
    assert config.EXPECTANCY_GATE_MODE in ["disabled", "shadow", "active"]
    assert "structure" in config.EQS_WEIGHTS
    assert config.EQS_WEIGHTS["structure"] == 20.0

def test_strategy_health_score_healthy():
    score, multiplier, rec = strategy_health_engine.evaluate_health(
        calibration_error_pct=0.03,
        psi_drift_score=0.05,
        rolling_profit_factor=1.90,
        current_drawdown_pct=3.0,
        win_rate_variance_pct=2.0,
        order_latency_ms=80.0
    )
    assert score >= 85.0
    assert multiplier == 1.00
    assert "NORMAL" in rec

def test_strategy_health_score_degraded():
    engine = StrategyHealthEngine()
    # High drawdown (12%), high drift (0.22), low PF (1.15)
    score, multiplier, rec = engine.evaluate_health(
        calibration_error_pct=0.12,
        psi_drift_score=0.22,
        rolling_profit_factor=1.15,
        current_drawdown_pct=12.0,
        win_rate_variance_pct=8.0,
        order_latency_ms=250.0
    )
    assert score < 85.0
    assert multiplier < 1.00

def test_trade_repository_abstraction(tmp_path):
    db_file = str(tmp_path / "test_trading_bot.db")
    repo = SQLiteTradeRepository(db_path=db_file)
    
    trade = {
        "symbol": "BTCUSDT",
        "interval": "15",
        "direction": "Bullish",
        "entry_price": 65000.0,
        "exit_price": 66000.0,
        "pnl_usd": 1.50,
        "change_pct": 1.54,
        "confidence": 0.92,
        "reason": "TP1 HIT",
        "exit_time": 1785361212.0
    }
    
    saved = repo.save_completed_trade(trade)
    assert saved is True

    trades = repo.get_completed_trades(limit=10)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTCUSDT"
    assert trades[0]["pnl_usd"] == 1.50
