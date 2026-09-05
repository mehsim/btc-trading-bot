import time
from unittest.mock import patch
import pandas as pd
import main


def test_maker_chase_attaches_sl_and_tp_to_entry_order():
    """Verify that maker chase Limit orders attach stopLoss and takeProfit directly on order placement."""
    placed_limit_orders = []

    def mock_place_limit(symbol, side, qty, price, sl=None, tp=None, reduce_only=False, post_only=False, **kwargs):
        placed_limit_orders.append({
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "sl": sl,
            "tp": tp,
            "post_only": post_only
        })
        return {"retCode": 0, "result": {"orderId": f"limit_{len(placed_limit_orders)}"}}

    df_completed = pd.DataFrame({"ATR_norm": [0.01] * 30})

    mock_details = {"orderStatus": "Filled", "cumExecQty": "0.010", "avgPrice": "60000.0"}
    with patch("main.get_all_bybit_positions", return_value=[]), \
         patch("main.set_bybit_leverage", return_value=True), \
         patch("main.get_bybit_bid_ask", return_value=(60000.0, 60000.0, 60000.0)), \
         patch("main.get_chase_limit_price", return_value=60000.0), \
         patch("main.place_bybit_limit_order", side_effect=mock_place_limit), \
         patch("main.wait_for_order_fill", return_value=True), \
         patch("main.cancel_bybit_order", return_value={"retCode": 0}), \
         patch("main.get_bybit_order_details", return_value=mock_details), \
         patch("main.update_bybit_stop_loss", return_value=True), \
         patch("main.update_bybit_take_profit", return_value=True), \
         patch("main.send_telegram_alert"), \
         patch("main.sync_active_positions_from_bybit"), \
         patch("bybit_client.bybit_get_request", return_value={"retCode": 0, "result": {}}):

        main._execute_bybit_trade_async_inner(
            symbol="BTCUSDT",
            iv="15",
            tf="15m",
            ml_trend="Bullish",
            leverage_val=10,
            qty_str="0.010",
            raw_qty=0.010,
            entry_price=60000.0,
            stop_loss_price=59640.0,
            take_profit_price=61000.0,
            position_size_usd=600.0,
            kelly_fraction=0.1,
            calibrated_confidence=0.85,
            ml_confidence=0.85,
            dynamic_conf_threshold=0.55,
            latest_completed_ts=int(time.time() * 1000),
            latest_candle={"ATR_norm": 0.01},
            pred_change=200.0,
            predicted_price=60200.0,
            atr_dollars=300.0,
            tp_multiplier_adjusted=1.66,
            sl_multiplier_adjusted=1.0,
            df_completed=df_completed,
            trade_uuid="test-uuid-maker-sltp",
            duration_seconds=900,
            active_trade_key="BTCUSDT_15m"
        )

    assert len(placed_limit_orders) >= 1
    first_order = placed_limit_orders[0]
    assert first_order["symbol"] == "BTCUSDT"
    assert first_order["sl"] == 59640.0
    assert first_order["tp"] == 61000.0
    assert first_order["post_only"] is True


def test_place_bybit_limit_order_payload_contains_sl_and_tp():
    """Verify that place_bybit_limit_order translates sl and tp arguments into Bybit payload fields."""
    with patch("bybit_client.execute_bybit_order_ws_or_rest") as mock_exec:
        mock_exec.return_value = {"retCode": 0}

        main.place_bybit_limit_order(
            symbol="BTCUSDT",
            side="Buy",
            qty="0.05",
            price=65000.0,
            sl=64000.0,
            tp=67000.0,
            post_only=True
        )

        assert mock_exec.called
        endpoint, payload = mock_exec.call_args[0]
        assert endpoint == "/v5/order/create"
        assert payload["stopLoss"] in ["64000.00", "64000.0", "64000"]
        assert payload["takeProfit"] in ["67000.00", "67000.0", "67000"]
        assert payload["timeInForce"] == "PostOnly"
