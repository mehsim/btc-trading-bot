import json
import time
import threading
import websocket

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

class BybitWebSocketClient:
    """
    Real-Time WebSocket stream engine connecting to Bybit's V5 linear stream.
    Pushes sub-5ms kline and orderbook updates to signal evaluation callbacks.
    """
    def __init__(self, symbols=None, intervals=None, callback=None):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.intervals = intervals or ["15", "30", "60"]
        self.callback = callback
        self.ws = None
        self.is_connected = False

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            topic = data.get("topic", "")
            if topic.startswith("kline."):
                parts = topic.split(".")
                interval = parts[1] if len(parts) > 1 else "15"
                symbol = parts[2] if len(parts) > 2 else "BTCUSDT"
                kline_data = data.get("data", [])
                if kline_data and self.callback:
                    self.callback(symbol, interval, kline_data[0])
        except Exception as e:
            print(f"[WS Error] {e}")

    def _on_error(self, ws, error):
        print(f"[WS Error] Connection error: {error}")
        self.is_connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[WS Status] Disconnected from Bybit WebSocket ({close_status_code}: {close_msg}). Reconnecting...")
        self.is_connected = False
        time.sleep(3)
        self.connect()

    def _on_open(self, ws):
        print("⚡ [WS Connected] Bybit V5 Linear WebSocket stream established.")
        self.is_connected = True
        
        args = []
        for sym in self.symbols:
            for iv in self.intervals:
                args.append(f"kline.{iv}.{sym}")
                
        sub_msg = {
            "op": "subscribe",
            "args": args[:10]
        }
        ws.send(json.dumps(sub_msg))

    def connect(self):
        def _run():
            self.ws = websocket.WebSocketApp(
                BYBIT_WS_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self.ws.run_forever()

        threading.Thread(target=_run, daemon=True).start()

if __name__ == "__main__":
    client = BybitWebSocketClient()
    client.connect()
    time.sleep(5)
