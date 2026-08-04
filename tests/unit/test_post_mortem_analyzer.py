import pytest
from post_mortem_analyzer import EmpiricalPostMortemAnalyzer

def test_post_mortem_analyzer_metrics():
    analyzer = EmpiricalPostMortemAnalyzer()
    trade_record = {
        "symbol": "XRPUSDT",
        "trade_id": "XRPUSDT_1785767300",
        "interval": "240",
        "direction": "Bearish",
        "entry_price": 1.0682,
        "exit_price": 1.0797,
        "take_profit": 1.0473,
        "stop_loss": 1.0797,
        "position_size_usd": 14.63,
        "balance": 59.50,
        "leverage": 10.0,
        "confidence": 0.97,
        "pnl_usd": -1.74,
        "change_pct": -11.89,
        "atr_dollars": 0.0111,
        "success": False
    }
    ohlc = [{"open": 1.0746, "high": 1.0828, "low": 1.0721, "close": 1.0818, "volume": 5999556.9}]

    res = analyzer.analyze_trade(trade_record, ohlc)
    assert res["symbol"] == "XRPUSDT"
    assert res["expected_rr_ratio"] == 1.82
    assert res["realized_r"] == -1.10
    assert res["individual_brier_loss"] == 0.9409
    assert res["wallet_margin_pct"] == 24.59
    assert res["mae_pct"] == 1.37
    assert len(res["contributing_factors"]) >= 2
