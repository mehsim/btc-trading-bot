import time
from unittest.mock import patch
import pandas as pd


def test_chase_loop_deducts_partial_fills():
    """Verify chase loop deducts cancelled order partial fills so subsequent chases only order remaining qty."""
    from main import _execute_bybit_trade_async_inner

    symbol = "BTCUSDT"
    iv = "15"
    tf = "15m"
    ml_trend = "Bullish"
    leverage_val = 10
    raw_qty = 0.010
    qty_str = "0.010"
    entry_price = 50000.0
    stop_loss_price = 49500.0
    take_profit_price = 51000.0
    position_size_usd = 500.0
    calibrated_confidence = 0.65
    ml_confidence = 0.65
    dynamic_conf_threshold = 0.55
    latest_completed_ts = int(time.time() * 1000)
    latest_candle = {"ATR_norm": 0.01}
    df_completed = pd.DataFrame({"ATR_norm": [0.01] * 30})

    # Track limit orders placed
    placed_orders = []

    def mock_place_limit(sym, side, q_str, px, sl=None, tp=None, post_only=False, reduce_only=False, **kwargs):
        placed_orders.append({"qty": q_str, "price": px, "sl": sl, "tp": tp, "reduce_only": reduce_only})
        return {"retCode": 0, "result": {"orderId": f"order_{len(placed_orders)}"}}

    # Mock order details:
    # Order 1: Partially filled (0.004 of 0.010) before cancel
    # Order 2: Fully filled (0.006 of 0.006)
    def mock_order_details(sym, oid):
        if oid == "order_1":
            return {"orderStatus": "Cancelled", "cumExecQty": "0.004", "avgPrice": "50000.0"}
        elif oid == "order_2":
            return {"orderStatus": "Filled", "cumExecQty": "0.006", "avgPrice": "50010.0"}
        return None

    with patch("main.get_all_bybit_positions", return_value=[]), \
         patch("main.set_bybit_leverage", return_value=True), \
         patch("main.get_chase_limit_price", return_value=50000.0), \
         patch("main.place_bybit_limit_order", side_effect=mock_place_limit), \
         patch("main.wait_for_order_fill", side_effect=[False, True]), \
         patch("main.cancel_bybit_order", return_value={"retCode": 0}), \
         patch("main.get_bybit_order_details", side_effect=mock_order_details), \
         patch("main.update_bybit_stop_loss", return_value=True), \
         patch("main.update_bybit_take_profit", return_value=True), \
         patch("main.send_telegram_alert"), \
         patch("main.sync_active_positions_from_bybit"):

        _execute_bybit_trade_async_inner(
            symbol=symbol, iv=iv, tf=tf, ml_trend=ml_trend, leverage_val=leverage_val,
            qty_str=qty_str, raw_qty=raw_qty, entry_price=entry_price,
            stop_loss_price=stop_loss_price, take_profit_price=take_profit_price,
            position_size_usd=position_size_usd, kelly_fraction=0.1,
            calibrated_confidence=calibrated_confidence, ml_confidence=ml_confidence,
            dynamic_conf_threshold=dynamic_conf_threshold,
            latest_completed_ts=latest_completed_ts, latest_candle=latest_candle,
            pred_change=100.0, predicted_price=50100.0, atr_dollars=200.0,
            tp_multiplier_adjusted=2.0, sl_multiplier_adjusted=1.0,
            df_completed=df_completed, trade_uuid="test-uuid-1", duration_seconds=900,
            active_trade_key="BTCUSDT_15m"
        )

        chase_orders = [o for o in placed_orders if not o.get("reduce_only")]
        # Verify that Chase 1 requested 0.010 and Chase 2 only requested 0.006
        assert len(chase_orders) == 2
        assert float(chase_orders[0]["qty"]) == 0.010
        assert float(chase_orders[1]["qty"]) == 0.006
