import threading
import json
try:
    import redis
except ImportError:
    redis = None
import numpy as np
import database
from logger import log_event

class ObservedList(list):
    def __init__(self, items, on_mutate_callback):
        super().__init__(items)
        self._callback = on_mutate_callback
        
    def append(self, item):
        super().append(item)
        self._callback(self, item)
        
    def extend(self, items):
        super().extend(items)
        for it in items:
            self._callback(self, it)


def default_json_serializer(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


class StateManager:
    def __init__(self):
        import os
        self._lock = threading.RLock()
        self._redis = None
        try:
            redis_pass = os.environ.get("REDIS_PASSWORD", None)
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            self._redis = redis.Redis(host=redis_host, port=redis_port, db=0, password=redis_pass, decode_responses=True, socket_timeout=3.0, socket_connect_timeout=3.0)
            self._redis.ping()
            print("[StateManager] Connected to Redis server with authentication.")
        except Exception as e:
            print(f"[StateManager Warning] Could not connect to Redis: {e}. Falling back to memory-only state.")
            self._redis = None


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
            
            "latest_prediction_15m": None,
            "latest_prediction_30m": None,
            "latest_prediction_1h": None,
            "latest_prediction_2h": None,
            
            "confluence_results_15m": None,
            "confluence_results_30m": None,
            "confluence_results_1h": None,
            "confluence_results_2h": None,
            
            "regime_15m": "Unknown",
            "regime_30m": "Unknown",
            "regime_1h": "Unknown",
            "regime_2h": "Unknown",
            
            "adx_15m": 0.0,
            "adx_30m": 0.0,
            "adx_1h": 0.0,
            "adx_2h": 0.0,
            
            "status": "Initializing",
            "retraining_status": "Idle",
            
            "calibration_15m": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
            "calibration_30m": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
            "calibration_1h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
            "calibration_2h": {"p95": 0.55, "max_conf": 0.75, "mean": 54.81},
            
            "daily_drawdown_start_balance": 80.0,
            "daily_drawdown_reset_day": -1,
            "circuit_breaker_active": False,
            "win_rate_by_tf": {"15": None, "30": None, "60": None, "120": None}
        }
        database.init_db()
        import sys
        sys.stderr.write("[StateManager Debug] database.init_db() done.\n")
        sys.stderr.flush()
        
        # Load settings from db or use default
        self._cache["simulated_balance"] = float(database.get_setting("simulated_balance", 80.0))
        self._cache["bot_running"] = database.get_setting("bot_running", "True") == "True"
        self._cache["bot_stopped"] = database.get_setting("bot_stopped", "False") == "True"
        self._cache["fresh_reset_v3"] = database.get_setting("fresh_reset_v3", "False") == "True"
        sys.stderr.write("[StateManager Debug] Settings loaded from DB.\n")
        sys.stderr.flush()
        
        # Load trade history and predictions into cache
        raw_trades = database.get_trade_history()
        raw_preds = database.get_prediction_history()
        sys.stderr.write(f"[StateManager Debug] raw_trades ({len(raw_trades)}), raw_preds ({len(raw_preds)}) loaded.\n")
        sys.stderr.flush()
        self._cache["trade_history"] = ObservedList(raw_trades, lambda lst, item: self._on_mutate_trade(item))
        self._cache["prediction_history"] = ObservedList(raw_preds, lambda lst, item: self._on_mutate_prediction(item))
        sys.stderr.write("[StateManager Debug] ObservedList wrappers attached.\n")
        sys.stderr.flush()
        
        for tf in ["5m", "15m", "30m", "1h", "2h", "4h", "6h"]:
            trades = database.get_active_trades(tf)
            self._cache[f"active_trade_{tf}"] = trades
            # Re-seed Redis from SQLite so stale Redis values don't shadow loaded DB state
            if self._redis:
                try:
                    self._redis.set(f"bot_state:active_trade_{tf}", json.dumps(trades, default=default_json_serializer))
                except Exception as ex_redis_seed:
                    print(f"[StateManager] Failed to seed Redis active_trade_{tf}: {ex_redis_seed}")
        if self._redis:
            try:
                self._redis.set("bot_state:trade_history", json.dumps(raw_trades, default=default_json_serializer))
                self._redis.set("bot_state:prediction_history", json.dumps(raw_preds, default=default_json_serializer))
                self._redis.set("bot_state:bot_running", json.dumps(self._cache["bot_running"]))
                self._redis.set("bot_state:bot_stopped", json.dumps(self._cache["bot_stopped"]))
            except Exception as ex_redis_seed:
                print(f"[StateManager] Failed to seed Redis bot_running/bot_stopped/history: {ex_redis_seed}")
        sys.stderr.write("[StateManager Debug] Active trades and runtime settings loaded.\n")
        sys.stderr.flush()

        # Initialize defaults to Redis
        if self._redis:
            print("[StateManager Debug] Starting Redis default keys sync...", flush=True)
            try:
                with self._redis.pipeline() as pipe:
                    for k, v in self._cache.items():
                        if not self._redis.exists(f"bot_state:{k}"):
                            raw_v = list(v) if isinstance(v, ObservedList) else v
                            pipe.set(f"bot_state:{k}", json.dumps(raw_v))
                    pipe.execute()
            except Exception as e:
                print(f"[StateManager Redis Init Error] Failed to batch write Redis: {e}", flush=True)
            print("[StateManager Debug] Redis default keys sync completed.", flush=True)

    def _on_mutate_trade(self, *args):
        item = args[-1] if args else None
        if isinstance(item, list):
            for t in item:
                if isinstance(t, dict):
                    database.save_completed_trade(t)
        elif isinstance(item, dict):
            database.save_completed_trade(item)

    def _on_mutate_prediction(self, *args):
        item = args[-1] if args else None
        if isinstance(item, list):
            for p in item:
                if isinstance(p, dict):
                    database.save_prediction(p)
        elif isinstance(item, dict):
            database.save_prediction(item)

            
        if self._redis:
            try:
                raw_list = list(self._cache.get("prediction_history", []))
                self._redis.set("bot_state:prediction_history", json.dumps(raw_list))
            except Exception as e:
                log_event("WARNING", f"[StateManager Redis Error] Failed to sync prediction_history mutation: {e}")


    def __getitem__(self, key):
        with self._lock:
            val = None
            if key in self._cache:
                val = self._cache[key]
            elif self._redis and key not in getattr(self, "_dirty_redis_keys", set()):
                try:
                    val_str = self._redis.get(f"bot_state:{key}")
                    if val_str is not None:
                        val = json.loads(val_str)
                        self._cache[key] = val
                except Exception as e:
                    log_event("WARNING", f"[StateManager Redis Error] Failed to get key {key}: {e}")
                
            if key == "trade_history" and isinstance(val, list) and not isinstance(val, ObservedList):
                val = ObservedList(val, lambda lst, item: self._on_mutate_trade(item))
                self._cache[key] = val
            elif key == "prediction_history" and isinstance(val, list) and not isinstance(val, ObservedList):
                val = ObservedList(val, lambda lst, item: self._on_mutate_prediction(item))
                self._cache[key] = val
            return val

    def __setitem__(self, key, value):
        with self._lock:
            raw_value = value
            if isinstance(value, ObservedList):
                raw_value = list(value)
            elif key == "trade_history" and isinstance(value, list) and not isinstance(value, ObservedList):
                value = ObservedList(value, lambda lst, item: self._on_mutate_trade(item))
                raw_value = list(value)
            elif key == "prediction_history" and isinstance(value, list) and not isinstance(value, ObservedList):
                value = ObservedList(value, lambda lst, item: self._on_mutate_prediction(item))
                raw_value = list(value)
                
            if not hasattr(self, "_dirty_redis_keys"):
                self._dirty_redis_keys = set()

            if self._redis:
                try:
                    self._redis.set(f"bot_state:{key}", json.dumps(raw_value, default=default_json_serializer))
                    self._dirty_redis_keys.discard(key)
                except Exception as e:
                    log_event("WARNING", f"[StateManager Redis Error] Failed to set {key}: {e}")
                    self._dirty_redis_keys.add(key)
                    
            self._cache[key] = value
            
            # Persist to database if it's one of the persistent keys
            if key.startswith("active_trade_"):
                tf = key.replace("active_trade_", "")
                database.save_active_trades(tf, raw_value)
            elif key in ["simulated_balance", "bot_running", "fresh_reset_v3"]:
                database.set_setting(key, str(raw_value))

    def get(self, key, default=None):
        with self._lock:
            val = None
            if key in self._cache:
                val = self._cache[key]
            elif self._redis and key not in getattr(self, "_dirty_redis_keys", set()):
                try:
                    val_str = self._redis.get(f"bot_state:{key}")
                    if val_str is not None:
                        val = json.loads(val_str)
                        self._cache[key] = val
                except Exception as e:
                    log_event("WARNING", f"[StateManager Redis Error] Failed to get key {key}: {e}")
            
            if val is None:
                return default
            
            if key == "trade_history" and isinstance(val, list) and not isinstance(val, ObservedList):
                val = ObservedList(val, lambda lst, item: self._on_mutate_trade(item))
                self._cache[key] = val
            elif key == "prediction_history" and isinstance(val, list) and not isinstance(val, ObservedList):
                val = ObservedList(val, lambda lst, item: self._on_mutate_prediction(item))
                self._cache[key] = val
            return val

    def __contains__(self, key):
        with self._lock:
            if key in self._cache:
                return True
            if self._redis:
                try:
                    return bool(self._redis.exists(f"bot_state:{key}"))
                except Exception as e:
                    log_event("WARNING", f"[StateManager Redis Error] Failed to check exists for {key}: {e}")
            return False

    def copy(self):
        with self._lock:
            res = {}
            for k in self._cache.keys():
                res[k] = self[k]
            return res

    def items(self):
        with self._lock:
            res = []
            for k in self._cache.keys():
                res.append((k, self[k]))
            return res

    def update(self, other):
        with self._lock:
            for k, v in other.items():
                self[k] = v

    def save_prediction(self, pred):
        with self._lock:
            database.save_prediction(pred)

state_manager = StateManager()

