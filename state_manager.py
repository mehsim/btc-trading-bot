import threading
import database

class ObservedList(list):
    def __init__(self, items, on_mutate_callback):
        super().__init__(items)
        self._callback = on_mutate_callback
        
    def append(self, item):
        super().append(item)
        self._callback(self, item)
        
    def extend(self, items):
        super().extend(items)
        for item in items:
            self._callback(self, item)

class StateManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._cache = {
            "live_price": None,
            "live_price_BTCUSDT": None,
            "live_price_ETHUSDT": None,
            "live_price_SOLUSDT": None,
            "live_price_BNBUSDT": None,
            "live_price_ADAUSDT": None,
            "live_price_XRPUSDT": None,
            "live_price_AVAXUSDT": None,
            "live_price_NEARUSDT": None,
            "live_price_LINKUSDT": None,
            "live_price_LTCUSDT": None,
            "live_price_DOGEUSDT": None,
            "last_update": 0.0,
            
            "latest_prediction_1h": None,
            "latest_prediction_2h": None,
            "latest_prediction_4h": None,
            "latest_prediction_6h": None,
            
            "confluence_results_1h": None,
            "confluence_results_2h": None,
            "confluence_results_4h": None,
            "confluence_results_6h": None,
            
            "regime_1h": "Unknown",
            "regime_2h": "Unknown",
            "regime_4h": "Unknown",
            "regime_6h": "Unknown",
            
            "adx_1h": 0.0,
            "adx_2h": 0.0,
            "adx_4h": 0.0,
            "adx_6h": 0.0,
            
            "status": "Initializing",
            "retraining_status": "Idle",
            
            "calibration_1h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
            "calibration_2h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
            "calibration_4h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
            "calibration_6h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
            
            "daily_drawdown_start_balance": 80.0,
            "daily_drawdown_reset_day": -1,
            "circuit_breaker_active": False,
            "win_rate_by_tf": {"60": None, "120": None, "240": None, "360": None}
        }
        database.init_db()
        
        # Load settings from db or use default
        self._cache["simulated_balance"] = float(database.get_setting("simulated_balance", 80.0))
        self._cache["bot_running"] = database.get_setting("bot_running", "True") == "True"
        self._cache["fresh_reset_v3"] = database.get_setting("fresh_reset_v3", "False") == "True"
        
        # Load trade history and predictions into cache
        self._cache["trade_history"] = database.get_trade_history()
        self._cache["prediction_history"] = database.get_prediction_history()
        
        for tf in ["1h", "2h", "4h", "6h"]:
            self._cache[f"active_trade_{tf}"] = database.get_active_trades(tf)

    def __getitem__(self, key):
        with self._lock:
            val = self._cache.get(key)
            if key == "trade_history" and isinstance(val, list) and not isinstance(val, ObservedList):
                val = ObservedList(val, lambda lst, item: database.save_completed_trade(item))
                self._cache[key] = val
            elif key == "prediction_history" and isinstance(val, list) and not isinstance(val, ObservedList):
                val = ObservedList(val, lambda lst, item: database.save_prediction(item))
                self._cache[key] = val
            return val

    def __setitem__(self, key, value):
        with self._lock:
            if key == "trade_history" and isinstance(value, list) and not isinstance(value, ObservedList):
                value = ObservedList(value, lambda lst, item: database.save_completed_trade(item))
            elif key == "prediction_history" and isinstance(value, list) and not isinstance(value, ObservedList):
                value = ObservedList(value, lambda lst, item: database.save_prediction(item))
                
            self._cache[key] = value
            
            # Persist to database if it's one of the persistent keys
            if key in ["active_trade_1h", "active_trade_2h", "active_trade_4h", "active_trade_6h"]:
                tf = key.split("_")[-1]
                database.save_active_trades(tf, value)
            elif key in ["simulated_balance", "bot_running", "fresh_reset_v3"]:
                database.set_setting(key, str(value))

    def get(self, key, default=None):
        with self._lock:
            val = self._cache.get(key)
            if val is None:
                return default
            if key == "trade_history" and isinstance(val, list) and not isinstance(val, ObservedList):
                val = ObservedList(val, lambda lst, item: database.save_completed_trade(item))
                self._cache[key] = val
            elif key == "prediction_history" and isinstance(val, list) and not isinstance(val, ObservedList):
                val = ObservedList(val, lambda lst, item: database.save_prediction(item))
                self._cache[key] = val
            return val

    def __contains__(self, key):
        with self._lock:
            return key in self._cache

    def copy(self):
        with self._lock:
            res = {}
            for k, v in self._cache.items():
                if isinstance(v, list):
                    res[k] = list(v)
                elif isinstance(v, dict):
                    res[k] = dict(v)
                else:
                    res[k] = v
            return res

    def items(self):
        with self._lock:
            return list(self._cache.items())

    def update(self, other):
        with self._lock:
            for k, v in other.items():
                self[k] = v

    def save_prediction(self, pred):
        with self._lock:
            database.save_prediction(pred)
