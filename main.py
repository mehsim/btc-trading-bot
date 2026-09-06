from typing import Optional, Any, Union
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import time
import json
import re
import math
from logger import log_event

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, Exception) as ex:
        log_event("WARNING", f"sys.stdout reconfigure exception: {ex}")
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, Exception) as ex:
        log_event("WARNING", f"sys.stderr reconfigure exception: {ex}")

import numpy as np
from datetime import datetime, timezone, timedelta
from kelly_tracker import global_kelly_tracker
from volatility_clusterer import volatility_clusterer
from gmm_trail import gmm_trailing_engine
from garch_monitor import garch_vol_monitor
from news_monitor import news_monitor
from decay_calibrator import decay_calibrator
import database
import trade_calculators
import exit_manager
from trade_calculators import (
    transaction_cost_model, calculate_break_even_stop, UnifiedTargetGenerator, calculate_probabilistic_utility_bootstrap,
    compute_be_trigger_distance, validate_trade_structure, AdaptiveVolumeGate, MFEBreakEvenTrigger,
    adaptive_volume_gate, mfe_be_trigger, choppiness_index, check_flash_crash, estimate_liquidation_pool,
    calculate_covariance_multiplier
)
from decision_outcome_db import decision_outcome_db
from meta_learning_engine import meta_learning_engine
from causal_attribution_engine import causal_attribution_engine
from counterfactual_replay_engine import counterfactual_replay_engine
from model_governance import extract_metric, validate_manifest_governance_floors
from probabilistic_policy_selector import probabilistic_policy_selector
from hierarchical_bayesian_engine import hierarchical_bayesian_engine
from drift_attribution_engine import drift_attribution_engine
from automatic_research_reporter import automatic_research_reporter
from exit_policy_engine import exit_policy_engine, PortfolioUtilityOptimizer, generate_continuous_policy_vector, log_checksummed_exit_decision
from order_state_machine import StopState, StopStateMachine
from secret_manager import get_secure_env
from circuit_breaker import circuit_breaker

from bybit_client import (
    bybit_get_request,
    bybit_post_request,
    get_real_bybit_balance_cached,
    get_all_bybit_positions,
    place_bybit_order,
    place_bybit_limit_order,
    place_bybit_taker_ioc_order,
    cancel_bybit_order,
    get_bybit_order_details,
    format_bybit_price,
    get_instrument_specs,
    get_bybit_time_offset,
    get_symbol_order_lock,
    execute_bybit_order_ws_or_rest,
    get_orderbook_imbalance as bybit_get_orderbook_imbalance,
    get_bybit_fee_rate
)
from confluence_engine import check_pre_trade_confluence
from telegram_bot import send_telegram_alert, execute_telegram_api_call
from telegram_listener import run_manual_confluence_report, start_telegram_command_listener
from dashboard_routes import dashboard_bp
from risk_limits import assert_risk_governance_invariants
from config_verifier import assert_shared_constants_aligned
from decision_journal import DecisionRecord, write_decision, ReasonCode

def format_bybit_qty(symbol: str, qty: float) -> str:
    """Finding #145: Local wrapper for format_bybit_qty that resolves get_instrument_specs
    through main's namespace so tests can patch 'main.get_instrument_specs' cleanly."""
    import math
    from decimal import Decimal
    q_val = max(0.0, float(qty))
    try:
        specs = get_instrument_specs(symbol)
        lot_str = str(specs.get("lotSize") or specs.get("qty_step") or specs.get("qtyStep") or "0.01")
        p = len(lot_str.split(".")[1]) if "." in lot_str else 0
        if p == 0:
            return f"{math.floor(q_val)}"
        lot_dec = Decimal(str(lot_str))
        q_dec = Decimal(str(q_val))
        floored_dec = (q_dec // lot_dec) * lot_dec
        return f"{floored_dec:.{p}f}"
    except Exception as exc:
        log_event("DEBUG", f"format_bybit_qty fallback: {exc}")
        from bybit_client import format_bybit_qty as _bc_fmt_qty
        return _bc_fmt_qty(symbol, qty)


def filter_unprocessed_active_trades(active_trades_list: list) -> list:
    """Filter out active trades that have been closed and already processed for exits."""
    if not isinstance(active_trades_list, list):
        active_trades_list = [] if active_trades_list is None else [active_trades_list]
    return [
        t for t in active_trades_list
        if isinstance(t, dict)
        and not (t.get("bybit_closed") and t.get("exit_processed", False))
        and not (t.get("closed") and t.get("exit_processed", False))
    ]


CANDLE_CLOCK_SKEW_TOLERANCE_SEC: float = -30.0


def compute_max_allowed_candle_age(iv: Union[str, int]) -> float:
    """Production formula for maximum candle freshness tolerance in seconds."""
    return min(900.0, max(300.0, int(iv) * 60 * 0.25))


def is_candle_fresh(latest_completed_ts: float, iv: Union[str, int], now_ms: Optional[float] = None) -> tuple[bool, float, float]:
    """
    Evaluates completed candle freshness against maximum allowed age and future clock skew tolerance.
    Returns (is_fresh, candle_age_sec, max_allowed_age_sec).
    """
    if now_ms is None:
        now_ms = time.time() * 1000.0
    interval_ms = int(iv) * 60 * 1000
    candle_close_ms = latest_completed_ts + interval_ms
    candle_age_sec = (now_ms - candle_close_ms) / 1000.0
    max_allowed_age_sec = compute_max_allowed_candle_age(iv)
    is_fresh = not (candle_age_sec > max_allowed_age_sec or candle_age_sec < CANDLE_CLOCK_SKEW_TOLERANCE_SEC)
    return is_fresh, candle_age_sec, max_allowed_age_sec


def check_live_feature_integrity(latest_candle_weighted, features_to_use):
    """
    Finding #80 & #162: Disallow blind 0.0 zero-filling across all feature branches.
    Returns (X_live_full, missing_model_features).
    """
    if isinstance(latest_candle_weighted, pd.Series):
        X_live_full = latest_candle_weighted.to_frame().T.reindex(columns=features_to_use)
    else:
        X_live_full = latest_candle_weighted.reindex(columns=features_to_use)

    X_live_full = X_live_full.apply(pd.to_numeric, errors="coerce")
    missing_model_features = [col for col in features_to_use if col not in latest_candle_weighted.index or pd.isna(X_live_full[col].iloc[0])]
    return X_live_full, missing_model_features


def map_status_to_reason_code(msg: str) -> Optional[str]:
    """Maps human-readable skip/reject status string to canonical ReasonCode enum."""
    if not msg:
        return None
    m = msg.upper()
    if "TCM NET EDGE" in m: return ReasonCode.TCM_NET_EDGE_NEGATIVE
    if "EXPECTED R" in m or "MIN R:R" in m or "ECON FAIL" in m: return ReasonCode.RR_BELOW_FLOOR
    if "HISTORICAL EV" in m or "EXPECTANCY" in m: return ReasonCode.EXPECTANCY_NEGATIVE
    if "MACRO OPPOSITION" in m or "HTF OPPOSITION" in m: return ReasonCode.MACRO_OPPOSITION
    if "FLASH CRASH" in m: return ReasonCode.FLASH_CRASH_ACTIVE
    if "LOW LIQUIDITY" in m: return ReasonCode.LOW_LIQUIDITY
    if "SPREAD WIDENING" in m: return ReasonCode.SPREAD_WIDENING
    if "KELLY EDGE" in m: return ReasonCode.KELLY_EDGE_NON_POSITIVE
    if "RISK CHECKLIST" in m: return ReasonCode.RISK_CHECKLIST_BLOCKED
    if "PREDICTION ERROR" in m: return ReasonCode.PREDICTION_ERROR
    if "LOW CONFIDENCE" in m: return ReasonCode.CONFIDENCE_BELOW_DYNAMIC_THRESHOLD
    if "CIRCUIT BREAKER" in m: return ReasonCode.CIRCUIT_BREAKER_ACTIVE
    if "BOT STOPPED" in m: return ReasonCode.BOT_STOPPED
    if "MARGIN" in m or "EXCEEDS WALLET" in m or "FREE MARGIN" in m or "EXCEEDS RISK CAP" in m or "BELOW RISK ALLOCATION FLOOR" in m: return ReasonCode.MARGIN_GUARD_EXCEEDED
    if "GEOMETRY" in m: return ReasonCode.GEOMETRY_INVALID
    if "CONCURRENT POSITIONS" in m: return ReasonCode.MAX_CONCURRENT_POSITIONS
    if "COOL-OFF" in m: return ReasonCode.COOL_OFF_ACTIVE
    if "FUNDING BLOCK" in m: return ReasonCode.FUNDING_BLOCK
    if "CONFORMAL UNCERTAINTY" in m or "UNCERTAINTY U" in m: return ReasonCode.HIGH_CONFORMAL_UNCERTAINTY
    if "VOLUME" in m: return ReasonCode.VOLUME_COMPRESSION
    if "ATR SPIKE" in m: return ReasonCode.ATR_SPIKE
    if "NEWS BLOCK" in m: return ReasonCode.NEWS_BLOCK
    if "CONTRADICTION" in m: return ReasonCode.CONTRADICTION
    if "NEUTRAL" in m: return ReasonCode.NEUTRAL
    if "CLUSTER LIMIT" in m: return ReasonCode.CLUSTER_LIMIT
    if "ALREADY ACTIVE" in m: return ReasonCode.ALREADY_ACTIVE
    if "CALIBRATOR" in m: return ReasonCode.CALIBRATOR_NON_VIABLE
    if "ADX" in m: return ReasonCode.ADX_BELOW_FLOOR
    return None

# F-09 Governance Startup Lock: Assert hard safety bounds before trading initialization
assert_risk_governance_invariants()
assert_shared_constants_aligned()

import collections
ACTIVE_TRADE_TF_KEYS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h"]
_recent_runtime_argmax = collections.defaultdict(lambda: collections.deque(maxlen=100))


# === FIX 1, 2, 3: TRADE STRUCTURE & RISK SANITIZATION GATE ===
MAX_RR_RATIO = {
    "5m": 3.0,
    "15": 4.0, "15m": 4.0,
    "30": 5.0, "30m": 5.0,
    "60": 6.0, "1h": 6.0,
    "120": 8.0, "2h": 8.0,
    "240": 8.0, "4h": 8.0,
    "360": 8.0, "6h": 8.0
}

MIN_RR_RATIO = {
    "5m": 1.2,
    "15": 1.3, "15m": 1.3,
    "30": 1.4, "30m": 1.4,
    "60": 1.5, "1h": 1.5,
    "120": 1.5, "2h": 1.5,
    "240": 1.5, "4h": 1.5,
    "360": 1.5, "6h": 1.5
}



class CircularLogBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.logs_list = []
        self.original_stdout = sys.__stdout__
        
    def write(self, message):
        try:
            self.original_stdout.write(message)
            self.original_stdout.flush()
        except Exception:
            try:
                if hasattr(self.original_stdout, "buffer"):
                    msg_str = str(message) if not isinstance(message, str) else message
                    self.original_stdout.buffer.write(msg_str.encode("utf-8", errors="replace"))
                    self.original_stdout.buffer.flush()
                else:
                    enc = getattr(self.original_stdout, "encoding", "ascii") or "ascii"
                    msg_str = str(message) if not isinstance(message, str) else message
                    safe_str = msg_str.encode(enc, errors="replace").decode(enc, errors="replace")
                    self.original_stdout.write(safe_str)
                    self.original_stdout.flush()
            except Exception:
                pass

        try:
            str_msg = message.decode("utf-8", errors="replace") if isinstance(message, bytes) else str(message)
            if str_msg.strip():
                clean_msg = re.sub(r'\x1b\[[0-9;]*[mK]', '', str_msg.strip())
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted = f"[{timestamp}] {clean_msg}" if not clean_msg.startswith("[") else clean_msg
                self.logs_list.append(formatted)
                if len(self.logs_list) > self.capacity:
                    self.logs_list.pop(0)
                try:
                    from logger import add_to_live_log
                    add_to_live_log(clean_msg)
                except (ImportError, AttributeError, ValueError):
                    pass
        except (ValueError, TypeError, UnicodeError):
            pass
                
    def flush(self):
        try:
            self.original_stdout.flush()
        except (IOError, OSError, AttributeError):
            pass

    def __getattr__(self, name):
        return getattr(self.original_stdout, name)

log_buffer = CircularLogBuffer(capacity=100)
sys.stdout = log_buffer

import os
print("[System Debug] os imported.")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
print("[System Debug] Thread env vars set.")
import time
startup_time = time.time()

from dotenv import load_dotenv
load_dotenv()
import features as features_module
import state_manager
import risk_engine
import mlops_engine
import retrain_pipeline
import feature_pipeline
import meta_model_engine

TRADE_MODE = os.environ.get("TRADE_MODE", "simulation").lower()
BYBIT_BASE_URL = "https://api-testnet.bybit.com" if TRADE_MODE == "testnet" else "https://api.bybit.com"
BYBIT_WS_URL = "wss://stream-testnet.bybit.com/v5/public/linear" if TRADE_MODE == "testnet" else "wss://stream.bybit.com/v5/public/linear"
BYBIT_PRIVATE_WS_URL = "wss://stream-testnet.bybit.com/v5/private" if TRADE_MODE == "testnet" else "wss://stream.bybit.com/v5/private"
private_ws_connected = False
private_ws_retry_delay = 5

active_public_ws = None
active_private_ws = None
last_private_ws_update_time = 0.0

# ==========================================
# TIMING & API/PROXY HIT INTERVALS
# ==========================================
CANDLE_CHECK_WINDOW_MINS = int(os.environ.get("CANDLE_CHECK_WINDOW_MINS", "4"))
CANDLE_CHECK_INTERVAL_SECS = int(os.environ.get("CANDLE_CHECK_INTERVAL_SECS", "20"))
BALANCE_UPDATE_INTERVAL_SECS = int(os.environ.get("BALANCE_UPDATE_INTERVAL_SECS", "120"))
POSITION_SYNC_INTERVAL_SECS = float(os.environ.get("POSITION_SYNC_INTERVAL_SECS", "30.0"))
POSITION_SYNC_IDLE_INTERVAL_SECS = float(os.environ.get("POSITION_SYNC_IDLE_INTERVAL_SECS", "120.0"))

import config
from config import TIMEFRAME_CONFIG
from strategy_health_engine import strategy_health_engine


print("[System Debug] Importing websocket...")
import websocket
print("[System Debug] websocket imported.")
import json
import requests
import asyncio
import aiohttp
import threading
import time

# Dedicated background thread and loop for async HTTP calls
_async_loop = None
_aiohttp_session = None

def _ensure_async_loop():
    global _async_loop, _aiohttp_session
    if _async_loop is None or not _async_loop.is_running():
        _async_loop = asyncio.new_event_loop()
        def _run():
            asyncio.set_event_loop(_async_loop)
            _async_loop.run_forever()
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        while not _async_loop.is_running():
            time.sleep(0.01)
        async def _init_session():
            global _aiohttp_session
            import ssl
            try:
                import certifi
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception as e_ssl:
                log_event("DEBUG", f"[Aiohttp SSL] Default context used: {e_ssl}")
                ssl_ctx = ssl.create_default_context()
            connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=100, keepalive_timeout=30)
            _aiohttp_session = aiohttp.ClientSession(connector=connector)
        asyncio.run_coroutine_threadsafe(_init_session(), _async_loop).result()

print("[System Debug] Importing pandas/numpy/joblib...")
import pandas as pd
import numpy as np
import joblib
print("[System Debug] pandas/numpy/joblib imported.")
import threading
import time
from datetime import datetime, timedelta, timezone

def get_pkt_time():
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=5)




def get_funding_adjustment(symbol: str, direction: str, funding_rate: float) -> float:
    """Bias confidence toward funded side (+0.03 boost) and penalize expensive side (-0.05)"""
    if funding_rate < -0.001:  # -0.1% funding: shorts get paid yield
        return +0.03 if direction == "Bearish" else -0.05
    elif funding_rate > 0.001: # +0.1% funding: longs get paid yield
        return +0.03 if direction == "Bullish" else -0.05
    return 0.0

def get_liquidity_score(symbol: str, orderbook_depth: int = 10) -> float:
    """Delegate to trade_calculators.get_liquidity_score (dynamic per-symbol benchmark)."""
    try:
        from trade_calculators import get_liquidity_score as _tc_liq
        return _tc_liq(symbol)
    except Exception:
        return 0.0

def send_daily_summary(chat_id=None):
    """Run at 00:00 UTC daily (or on-demand via Telegram) to summarize 24h performance & health"""
    try:
        now_ts = time.time()
        sec_24h = 24 * 3600.0
        trade_history = bot_state.get("trade_history", [])
        
        trades_24h = []
        for t in trade_history:
            try:
                exit_t = float(t.get("exit_time", 0.0))
                if (now_ts - exit_t) <= sec_24h:
                    trades_24h.append(t)
            except Exception:
                pass
        
        # Summarize across all active timeframes
        tf_summaries = []
        total_pnl_24h = 0.0
        total_trades_24h = len(trades_24h)
        
        for iv in ["15", "30", "60", "120", "240", "360"]:
            iv_trades = [t for t in trades_24h if str(t.get("interval")) == str(iv)]
            if not iv_trades:
                continue
            wins = sum(1 for t in iv_trades if t.get("success") is True or float(t.get("pnl_usd", 0.0)) > 0)
            wr = (wins / len(iv_trades) * 100.0) if iv_trades else 0.0
            pnl = sum(float(t.get("pnl_usd", 0.0)) for t in iv_trades)
            total_pnl_24h += pnl
            tf_summaries.append(f"  • *{iv}M*: {len(iv_trades)} trades | Win Rate: {wr:.1f}% | P&L: ${pnl:+.2f}")
            
        tf_text = "\n".join(tf_summaries) if tf_summaries else "  • No closed trades in last 24h"
        
        has_30m = os.path.exists("ensemble_trending_trend_30_xgb.json")
        drift_res = mlops_engine.check_model_drift("15", trade_history, window=100)
        drift_status = drift_res.get("status", "HEALTHY")
        
        current_eq = float(bot_state.get("live_balance", bot_state.get("wallet_balance", bot_state.get("simulated_balance", 80.0))))
        if current_eq <= 0:
            current_eq = float(bot_state.get("simulated_balance", 80.0))
            
        peak_eq = float(bot_state.get("peak_balance", current_eq))
        if current_eq > peak_eq:
            peak_eq = current_eq
            bot_state["peak_balance"] = peak_eq
            
        dd_pct = ((peak_eq - current_eq) / peak_eq * 100.0) if peak_eq > 0 else 0.0
        filter_stats = bot_state.get("filter_stats", {})
        
        summary_msg = (
            f"📊 *BTC TRADING BOT — DAILY SUMMARY* ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})\n\n"
            f"💰 *Overall 24h Result*: *${total_pnl_24h:+.2f}* ({total_trades_24h} trades)\n\n"
            f"⏱️ *Performance by Timeframe*:\n"
            f"{tf_text}\n\n"
            f"🏥 *System Health & Equity*:\n"
            f"  • Live Equity: *${current_eq:.2f}* (Peak: ${peak_eq:.2f})\n"
            f"  • Current Drawdown: {dd_pct:.1f}%\n"
            f"  • Model Drift Status: {drift_status}\n"
            f"  • 30M Model: {'✅ Dedicated' if has_30m else '⚠️ Fallback to 15M'}\n\n"
            f"🛡️ *Active Protection Filters (24h)*:\n"
            f"  • Choppiness blocks: {filter_stats.get('chop_blocks', 0)}\n"
            f"  • News blackouts: {filter_stats.get('news_blocks', 0)}\n"
            f"  • Flash crash saves: {filter_stats.get('flash_saves', 0)}\n"
            f"  • Liquidity skips: {filter_stats.get('liquidity_skips', 0)}\n"
        )
        
        target_chat_id = chat_id if chat_id else os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if target_chat_id:
            execute_telegram_api_call("sendMessage", {
                "chat_id": target_chat_id,
                "text": summary_msg,
                "parse_mode": "Markdown"
            })
        
        # If sending for daily midnight reset, reset filter stats
        if not chat_id:
            filter_stats["chop_blocks"] = 0
            filter_stats["news_blocks"] = 0
            filter_stats["flash_saves"] = 0
            filter_stats["liquidity_skips"] = 0
    except Exception as err:
        print(f"[Daily Summary Error] Failed to generate daily summary: {err}")

class HTFTrendCache:
    def __init__(self):
        self._cache = {}
        self.ttl = {"60": 300, "240": 900}
        self._lock = threading.Lock()

    def get_trend(self, symbol: str, interval: str) -> tuple:
        key = (symbol, str(interval))
        now = datetime.now(timezone.utc)
        with self._lock:
            if key in self._cache:
                cached = self._cache[key]
                age = (now - cached["timestamp"]).total_seconds()
                if age < self.ttl.get(str(interval), 300):
                    return cached["ema9"], cached["ema21"]
        
        try:
            df = get_history(symbol=symbol, interval=str(interval), limit=100, fail_if_stale=True)
            if df is not None and not df.empty and df.attrs.get("fetch_ok", True) and len(df) >= 21:
                df_completed = df.iloc[:-1].copy()
                ema9 = float(EMAIndicator(df_completed["close"], window=9).ema_indicator().iloc[-1])
                ema21 = float(EMAIndicator(df_completed["close"], window=21).ema_indicator().iloc[-1])
                with self._lock:
                    self._cache[key] = {"ema9": ema9, "ema21": ema21, "timestamp": now}
                return ema9, ema21
            else:
                log_event("WARNING", f"[HTFTrendCache] Data stale or insufficient for {symbol} {interval}m, returning neutral (0.0, 0.0)")
        except Exception as e:
            print(f"[HTFTrendCache Error] Failed to fetch {symbol} {interval}m: {e}")
            
        return 0.0, 0.0

    def invalidate(self, symbol: Optional[str] = None):
        with self._lock:
            if symbol:
                for iv in ["60", "240"]:
                    self._cache.pop((symbol, iv), None)
            else:
                self._cache.clear()

global_htf_trend_cache = HTFTrendCache()

def execute_telegram_api_call(method: str, payload: dict) -> dict:
    """Helper to send POST requests to Telegram API using proxy routing."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return {}
        
    custom_url = os.environ.get("TELEGRAM_API_URL")
    if custom_url:
        if not custom_url.endswith("/"):
            custom_url += "/"
        url = f"{custom_url}bot{token}/{method}"
    else:
        url = f"https://api.telegram.org/bot{token}/{method}"
        
    tg_proxy = os.environ.get("TELEGRAM_PROXY") or os.environ.get("BYBIT_PROXY")
    proxies_dict = None
    if tg_proxy:
        if custom_url:
            # If using a custom Cloudflare worker URL, route using standard HTTP proxy CONNECT
            proxies_dict = {"http": tg_proxy, "https": tg_proxy}
        else:
            # If connecting directly to api.telegram.org, use socks5h to delegate DNS resolution
            if "://" in tg_proxy:
                tg_proxy_clean = tg_proxy.split("://", 1)[1]
            else:
                tg_proxy_clean = tg_proxy
            socks_proxy = f"socks5h://{tg_proxy_clean}"
            proxies_dict = {"http": socks_proxy, "https": socks_proxy}
            
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Connection": "close"
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=20, proxies=proxies_dict, verify=True)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 400 and payload.get("parse_mode"):
                # Fallback to plain text if Markdown entity parsing failed
                plain_payload = dict(payload)
                plain_payload.pop("parse_mode", None)
                try:
                    resp_plain = requests.post(url, json=plain_payload, headers=headers, timeout=20, proxies=proxies_dict, verify=True)
                    if resp_plain.status_code == 200:
                        return resp_plain.json()
                except Exception:
                    pass
            print(f"[Telegram API Error] method={method} status={resp.status_code}: {resp.text}")
            return {}
        except Exception as e:
            err_str = str(e).lower()
            is_eof = "eof occurred" in err_str or "unexpected eof" in err_str or "connection aborted" in err_str
            
            if custom_url:
                try:
                    fallback_url = f"https://api.telegram.org/bot{token}/{method}"
                    fallback_proxies = None
                    if tg_proxy:
                        tg_clean = tg_proxy.split("://", 1)[-1]
                        fallback_proxies = {"http": f"socks5h://{tg_clean}", "https": f"socks5h://{tg_clean}"}
                    resp2 = requests.post(fallback_url, json=payload, headers=headers, timeout=20, proxies=fallback_proxies)
                    if resp2.status_code == 200:
                        return resp2.json()
                except Exception:
                    pass
                    
            if is_eof and attempt < 2:
                time.sleep(0.5)
                continue
                
            is_silent = method == "getUpdates" and any(x in err_str for x in ["timed out", "eof", "ssl", "connection"])
            if not is_silent:
                print(f"[Telegram API Exception] method={method}: {e}")
            return {}

def send_telegram_alert(message: str):
    def _post():
        try:
            import telegram_bot
            if hasattr(telegram_bot, "send_telegram_alert"):
                telegram_bot.send_telegram_alert(message)
                return
        except Exception:
            pass
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        raw_ids = os.environ.get("TELEGRAM_CHAT_IDS") or os.environ.get("TELEGRAM_CHAT_ID") or ""
        chat_ids = [cid.strip() for cid in raw_ids.split(",") if cid.strip()]
        if not token or not chat_ids:
            return
        for cid in chat_ids:
            payload = {
                "chat_id": cid,
                "text": message,
                "parse_mode": "Markdown"
            }
            execute_telegram_api_call("sendMessage", payload)
        
    threading.Thread(target=_post, daemon=True).start()
        

_dispatched_exit_alerts = set()


# Telegram listener extracted to telegram_listener.py


print("[System Debug] Importing ta...")
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
print("[System Debug] ta imported.")
print("[System Debug] Importing data.py...")
from data import (get_history, merge_derivatives_sentiment_features, classify_market_regime,
                  _merge_cached_derivatives, get_bybit_oi_history, get_bybit_funding_history,
                  get_fear_and_greed_history)
print("[System Debug] Importing Flask...")
from flask import Flask, jsonify, render_template, request, make_response

# ==========================================
# WEB DASHBOARD CONFIGURATION & STATE
# ==========================================
app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.register_blueprint(dashboard_bp)

@app.after_request
def add_cors_headers(response):
    cors_origin = os.environ.get("DASHBOARD_CORS_ORIGIN", "").strip()
    if cors_origin:
        response.headers["Access-Control-Allow-Origin"] = cors_origin
    elif os.environ.get("DASHBOARD_ALLOW_WILDCARD_CORS", "false").lower() in ("true", "1"):
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-KEY, X-ADMIN-KEY, Authorization"
    return response

bot_logs = []
logs_lock = threading.Lock()
retraining_lock = threading.Lock()
active_trades_lock = threading.RLock()

from state_manager import state_manager as bot_state

def trigger_emergency_kill_switch(reason: str = "Manual Trigger"):
    print(f"[EMERGENCY KILL SWITCH] Triggered! Reason: {reason}")
    bot_state["bot_running"] = False
    database.set_setting("bot_running", "False")
    
    cancel_success = True
    close_success = True
    errors = []

    try:
        from bybit_client import bybit_post_request, get_all_bybit_positions, place_bybit_taker_ioc_order, TRADE_MODE
        if TRADE_MODE != "simulation":
            res_cancel = bybit_post_request("/v5/order/cancel-all", {"category": "linear", "settleCoin": "USDT"})
            if isinstance(res_cancel, dict) and res_cancel.get("retCode") != 0:
                cancel_success = False
                errors.append(f"Cancel failed: {res_cancel.get('retMsg', 'Unknown error')}")
            
            positions = get_all_bybit_positions()
            if positions is None:
                close_success = False
                errors.append("Failed to query open positions from Bybit API (API error or timeout)")
            else:
                for p in positions:
                    sym = p.get("symbol")
                    sz = float(p.get("size", "0"))
                    side = p.get("side")
                    if sz > 0 and sym:
                        close_side = "Sell" if side == "Buy" else "Buy"
                        res_close = place_bybit_taker_ioc_order(
                            symbol=sym,
                            side=close_side,
                            qty=sz,
                            reduce_only=True,
                            order_link_id=f"kill_{sym}_{int(time.time()*1000)}"[:36]
                        )
                        if isinstance(res_close, dict) and res_close.get("retCode") != 0:
                            close_success = False
                            errors.append(f"Close {sym} failed: {res_close.get('retMsg')}")
    except Exception as err:
        errors.append(str(err))
        cancel_success = False
        close_success = False
        print(f"[Kill Switch Error] Failed executing emergency close: {err}")

    status_msg = f"🚨 *EMERGENCY KILL SWITCH ACTIVATED* 🚨\n• Reason: `{reason}`\n• Action: Bot halted."
    if errors:
        status_msg += f"\n• Errors encountered: {'; '.join(errors)}"
    else:
        status_msg += "\n• Working orders cancelled and open positions closed at market."

    send_telegram_alert(status_msg)
    return cancel_success and close_success, errors

from functools import wraps
import hmac




cached_news_sentiment = "Neutral"
cached_news_titles = []
news_sentiment_lock = threading.Lock()

# Thread-safe real-time Order Flow (CVD & OFI)
from config import SUPPORTED_SYMBOLS
order_flow_lock = threading.Lock()
order_flow_data = {s: {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0} for s in SUPPORTED_SYMBOLS}

# Thread-safe active background order execution guard
active_execution_lock = threading.RLock()
active_execution_symbols = set()
active_execution_notional = {}
active_execution_margins = {}
active_execution_risks = {}

economic_calendar_cache = None
last_calendar_fetch = 0.0
economic_calendar_lock = threading.Lock()

# Re-entrant lock for thread-safe access to bot_state and file IO
bot_state_lock = threading.RLock()

HISTORY_FILE = "/data/dashboard_history.json" if os.path.exists("/data") and os.access("/data", os.W_OK) else "dashboard_history.json"

def save_history():
    with bot_state_lock:
        # O(N) Trade Deduplication with Key Hashing (M-10)
        trades = list(bot_state.get("trade_history", []))
        if trades:
            seen_keys = set()
            deduped = []
            for t in reversed(trades):
                t_exit = float(t.get("exit_time") or 0.0)
                t_entry_p = round(float(t.get("entry_price") or 0.0), 4)
                t_exit_p = round(float(t.get("exit_price") or 0.0), 4)
                t_sym = str(t.get("symbol", ""))
                t_iv = str(t.get("interval", ""))
                t_dir = str(t.get("direction", "")).lower()
                t_window = int(t_exit // 43200) if t_exit > 0 else 0
                key = (t_sym, t_iv, t_dir, t_entry_p, t_exit_p, t_window)
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append(t)
            deduped.reverse()
            bot_state["trade_history"] = deduped[-1000:]


        # Filter out temporary Abstain/NaN entries from prediction_history
        bot_state["prediction_history"] = [
            p for p in bot_state.get("prediction_history", [])
            if p.get("status") != "Abstain" and p.get("calibrated_confidence") is not None and str(p.get("direction")) != "None"
        ]

        # Cap prediction history at 500 entries
        if len(bot_state["prediction_history"]) > 500:
            bot_state["prediction_history"] = bot_state["prediction_history"][-500:]
        # Recompute win rate by TF from trade history
        for tf_key in ["60", "120", "240", "360"]:
            tf_trades = [t for t in bot_state["trade_history"] if str(t.get("interval")) == tf_key]
            if tf_trades:
                wins = sum(1 for t in tf_trades if t.get("success"))
                bot_state["win_rate_by_tf"][tf_key] = round(wins / len(tf_trades) * 100, 1)
            else:
                bot_state["win_rate_by_tf"][tf_key] = None
        data = {
            "simulated_balance": bot_state["simulated_balance"],
            "trade_history": bot_state["trade_history"],
            "prediction_history": bot_state["prediction_history"],
            "active_trade_15m": bot_state.get("active_trade_15m", []),
            "active_trade_30m": bot_state.get("active_trade_30m", []),
            "active_trade_1h": bot_state.get("active_trade_1h", []),
            "active_trade_2h": bot_state.get("active_trade_2h", []),
            "active_trade_4h": bot_state.get("active_trade_4h", []),
            "active_trade_6h": bot_state.get("active_trade_6h", []),
            "bot_running": bot_state.get("bot_running", True),
            "fresh_reset_v3": bot_state.get("fresh_reset_v3", False)
        }
        try:
            dir_name = os.path.dirname(HISTORY_FILE)
            temp_file = os.path.join(dir_name, "dashboard_history_temp.json") if dir_name else "dashboard_history_temp.json"
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, HISTORY_FILE)
            
            # Persist active trades to SQLite database
            try:
                import database
                for tf_key in ACTIVE_TRADE_TF_KEYS:
                    database.save_active_trades(tf_key, bot_state.get(f"active_trade_{tf_key}", []))
            except Exception as db_sync_err:
                print(f"[Database Sync Warning] Failed to persist active trades to SQLite: {db_sync_err}")
                
            # If running on Hugging Face and write token is available, backup to HF Dataset
            token = os.environ.get("HF_TOKEN") or os.environ.get("token")
            space_id = os.environ.get("SPACE_ID")
            if token and space_id:
                def _hf_backup():
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        dataset_id = f"{space_id}-history"
                        api.create_repo(repo_id=dataset_id, repo_type="dataset", exist_ok=True, token=token)
                        api.upload_file(
                            path_or_fileobj=HISTORY_FILE,
                            path_in_repo="dashboard_history.json",
                            repo_id=dataset_id,
                            repo_type="dataset",
                            token=token
                        )
                    except Exception as hf_err:
                        print(f"HF Space Sync: Failed to backup history to Dataset: {hf_err}")
                threading.Thread(target=_hf_backup, daemon=True).start()
        except Exception as e:
            print(f"Error saving history to disk: {e}")

def migrate_active_trades(active_trades_list):
    if not isinstance(active_trades_list, list):
        return
    for t in active_trades_list:
        if "confidence" not in t:
            # Estimate confidence based on original sizing thresholds
            orig_size = t.get("original_size", t.get("position_size_usd", 9.5))
            if orig_size >= 11.0:
                t["confidence"] = 0.785
            elif orig_size >= 9.5:
                t["confidence"] = 0.685
            else:
                t["confidence"] = 0.585

def heal_completed_trades_bybit_order_ids():
    predictions_by_key = {}
    for p in bot_state.get("prediction_history", []):
        if p.get("status") == "Traded" and p.get("bybit_order_id"):
            key = (p.get("symbol"), str(p.get("interval", "60")), p.get("direction"))
            predictions_by_key.setdefault(key, []).append(p)

    healed_count = 0
    for t in bot_state.get("trade_history", []):
        if not t.get("bybit_order_id") or t.get("bybit_order_id") == "N/A":
            key = (t.get("symbol"), str(t.get("interval", "60")), t.get("direction"))
            candidates = predictions_by_key.get(key, [])
            if candidates:
                best_p = None
                min_diff = float("inf")
                exit_ts = t.get("exit_time", 0.0)
                for p in candidates:
                    pred_ts = p.get("timestamp", 0.0)
                    if pred_ts < exit_ts:
                        diff = exit_ts - pred_ts
                        if diff < min_diff:
                            min_diff = diff
                            best_p = p
                if best_p and min_diff < 86400 * 5:
                    t["bybit_order_id"] = best_p["bybit_order_id"]
                    if best_p.get("bybit_scale_out_order_id"):
                        t["bybit_scale_out_order_id"] = best_p["bybit_scale_out_order_id"]
                    healed_count += 1

    if healed_count > 0:
        print(f"[Heal] Successfully recovered missing bybit_order_id for {healed_count} completed trades.")
        save_history()

def deduplicate_completed_trades():
    trades = bot_state.get("trade_history", [])
    if not trades:
        return
        
    sorted_trades = sorted(trades, key=lambda x: float(x.get("exit_time", 0.0)), reverse=True)
    seen_keys = set()
    deduped = []
    for t in sorted_trades:
        t_exit = float(t.get("exit_time", 0.0))
        t_entry_p = round(float(t.get("entry_price", 0.0)), 4)
        t_exit_p = round(float(t.get("exit_price", 0.0)), 4)
        t_sym = str(t.get("symbol"))
        t_iv = str(t.get("interval"))
        t_dir = str(t.get("direction"))
        t_window = int(t_exit // 43200) if t_exit > 0 else 0
        key = (t_sym, t_iv, t_dir, t_entry_p, t_exit_p, t_window)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(t)
            
    deduped.reverse()
    if len(deduped) != len(trades):
        print(f"[Heal] Deduplicated trade history: removed {len(trades) - len(deduped)} duplicate records within 12h window.")
        bot_state["trade_history"] = deduped
        save_history()

def load_history():
    token = os.environ.get("HF_TOKEN") or os.environ.get("token")
    space_id = os.environ.get("SPACE_ID")
    
    # 1. If running on HF and token is available, restore history from HF Dataset
    if space_id and token:
        try:
            from huggingface_hub import hf_hub_download
            dataset_id = f"{space_id}-history"
            print(f"[Sync] Attempting to download history from Dataset {dataset_id}...")
            downloaded_path = hf_hub_download(  # nosec B615 — repo_id is derived from env-controlled space_id; no user input
                repo_id=dataset_id,
                filename="dashboard_history.json",
                repo_type="dataset",
                token=token,
                revision="main"
            )
            import shutil
            shutil.copy(downloaded_path, HISTORY_FILE)
            print("[Sync] Successfully restored history from Hugging Face Dataset.")
        except Exception as hf_err:
            print(f"[Sync] Could not restore history from HF Dataset (normal if first run): {hf_err}")

    # 2. Sync from AWS Server API if explicitly configured with remote target (not localhost)
    elif not space_id:
        target_server = os.environ.get("TARGET_AWS_SERVER") or os.environ.get("SYNC_SERVER_URL")
        if target_server and not any(h in target_server for h in ["127.0.0.1", "localhost", "0.0.0.0", "::1"]):
            try:
                aws_host = target_server
                if not aws_host.startswith("http://") and not aws_host.startswith("https://"):
                    aws_host = f"http://{aws_host}"
                if ":" not in aws_host.replace("http://", "").replace("https://", ""):
                    sync_port = os.environ.get("PORT", "5001")
                    aws_host = f"{aws_host}:{sync_port}"
                sync_url = f"{aws_host.rstrip('/')}/api/status"
                print(f"Syncing: Attempting to pull latest history from Remote Server API ({sync_url})...")
                headers = {}
                api_k = get_secure_env("DASHBOARD_API_KEY", "").strip()
                if api_k:
                    headers["X-API-KEY"] = api_k
                resp = requests.get(sync_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    remote_trades = data.get("trade_history", [])
                    remote_predictions = data.get("prediction_history", [])
                    remote_balance = data.get("simulated_balance", 80.0)
                    
                    remote_trades = [t for t in remote_trades if str(t.get("interval", "60")) != "5"]
                    remote_predictions = [p for p in remote_predictions if str(p.get("interval", "60")) != "5"]
                    
                    if len(remote_trades) > 0 or len(remote_predictions) > 0:
                        bot_state["simulated_balance"] = remote_balance
                        bot_state["trade_history"] = remote_trades
                        bot_state["prediction_history"] = remote_predictions
                        
                        for tf_key in ACTIVE_TRADE_TF_KEYS:
                            migrate_active_trades(bot_state[f"active_trade_{tf_key}"])
                            
                        local_halt_setting = database.get_setting("bot_running", "True")
                        if local_halt_setting == "False":
                            bot_state["bot_running"] = False
                        else:
                            bot_state["bot_running"] = data.get("bot_running", True)
                        bot_state["fresh_reset_v3"] = data.get("fresh_reset_v3", False)
                        print(f"Sync Success: Loaded {len(remote_trades)} trades and {len(remote_predictions)} predictions from Remote Server ({sync_url}).")
                        
                        save_history()
                        return
            except Exception as e:
                print(f"[Remote Sync] Could not fetch state from Remote Server ({sync_url}): {e}")


    # 2. Local/Persistent history fallback load
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                bot_state["simulated_balance"] = data.get("simulated_balance", 80.0)
                bot_state["trade_history"] = [t for t in data.get("trade_history", []) if str(t.get("interval", "60")) != "5"]
                for t in bot_state["trade_history"]:
                    if "interval" not in t:
                        t["interval"] = "60"
                bot_state["prediction_history"] = [
                    p for p in data.get("prediction_history", [])
                    if str(p.get("interval", "60")) != "5" and p.get("status") != "Abstain" and p.get("calibrated_confidence") is not None and str(p.get("direction")) != "None"
                ]
                for p in bot_state["prediction_history"]:
                    if "interval" not in p:
                        p["interval"] = "60"
                
                # Migrate legacy active trades loaded from SQLite
                for tf_key in ACTIVE_TRADE_TF_KEYS:
                    migrate_active_trades(bot_state.get(f"active_trade_{tf_key}", []))
                    
                sqlite_running = database.get_setting("bot_running")
                if sqlite_running is not None:
                    bot_state["bot_running"] = sqlite_running == "True"
                else:
                    bot_state["bot_running"] = data.get("bot_running", True)
                sqlite_stopped = database.get_setting("bot_stopped")
                if sqlite_stopped is not None:
                    bot_state["bot_stopped"] = sqlite_stopped == "True"
                bot_state["fresh_reset_v3"] = data.get("fresh_reset_v3", False)
                
                active_margin = sum(t.get("position_size_usd", 0.0) for tf_key in ACTIVE_TRADE_TF_KEYS for t in bot_state.get(f"active_trade_{tf_key}", []))
                print(f"[Local Sync Balance Audit] Cash Balance: ${bot_state['simulated_balance']:.2f} | Active Position Margin: ${active_margin:.2f} | Total Account Value: ${bot_state['simulated_balance'] + active_margin:.2f}")
                return
        except Exception as e:
            print(f"Error loading local history: {e}")

    # Fallback default initial state
    bot_state["simulated_balance"] = 80.0
    bot_state["trade_history"] = []
    bot_state["prediction_history"] = []

    # Force auto-reset if it's the first time running this updated version
    if not bot_state.get("fresh_reset_v3", False):
        print("[System Reset] Migrating history to fresh reset v3. Setting balance to 80.0 and clearing all old trades.")
        bot_state["simulated_balance"] = 80.0
        bot_state["daily_drawdown_start_balance"] = 80.0
        bot_state["trade_history"] = []
        bot_state["prediction_history"] = []
        bot_state["fresh_reset_v3"] = True
        save_history()
        
    deduplicate_completed_trades()
    heal_completed_trades_bybit_order_ids()
        
    bot_state["retraining_status"] = "Idle"

# Thread-safe print wrapper to redirect logs to dashboard log panel
_print = print
def print(*args, **kwargs):
    _print(*args, **kwargs)
    if "file" not in kwargs or kwargs["file"] is None:
        msg = " ".join(map(str, args))
        timestamp = get_pkt_time().strftime("%H:%M:%S")
        lines = msg.split('\n')
        with logs_lock:
            for line in lines:
                if line.strip(): # ignore empty lines in console
                    bot_logs.append(f"[{timestamp}] {line}")
            # Keep history to 200 lines
            if len(bot_logs) > 200:
                bot_logs[:] = bot_logs[-200:]

_last_balance_fetch = 0.0
_cached_balance = None
_balance_lock = threading.Lock()

def get_bybit_proxies():
    import os
    # If running on Hugging Face and no explicit BYBIT_PROXY is set, bypass internal HF proxy
    if os.environ.get("SPACE_ID") and not os.environ.get("BYBIT_PROXY"):
        return None

    proxy = (
        os.environ.get("BYBIT_PROXY") or
        os.environ.get("HTTPS_PROXY") or
        os.environ.get("HTTP_PROXY") or
        os.environ.get("https_proxy") or
        os.environ.get("http_proxy")
    )
    if proxy:
        if "://" not in proxy:
            proxy = "http://" + proxy
        return {
            "http": proxy,
            "https": proxy
        }
    return None

_session = None
_session_lock = threading.Lock()

def get_shared_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                proxies = get_bybit_proxies()
                if proxies:
                    _session.proxies.update(proxies)
    return _session

def parse_proxy_url(proxy_url):
    """Parse proxy URL into components for websocket-client."""
    from urllib.parse import urlparse
    if "://" not in proxy_url:
        proxy_url = "http://" + proxy_url
    parsed = urlparse(proxy_url)
    host = parsed.hostname
    port = parsed.port
    proxy_type = parsed.scheme or "http"
    if proxy_type == "socks":
        proxy_type = "socks5"
    auth = None
    if parsed.username and parsed.password:
        auth = (parsed.username, parsed.password)
    return host, port, auth, proxy_type

_cached_time_offset = 0.0
_last_time_sync = 0.0
_time_offset_lock = threading.Lock()

def get_bybit_time_offset():
    global _cached_time_offset, _last_time_sync
    with _time_offset_lock:
        if _cached_time_offset is not None and (time.time() - _last_time_sync) < 7200:
            return _cached_time_offset
    
    async def do_time_sync():
        proxy_dict = get_bybit_proxies()
        proxy_url = proxy_dict.get("https") or proxy_dict.get("http") if proxy_dict else None
        timeout = aiohttp.ClientTimeout(total=5)
        async with _aiohttp_session.get(f"{BYBIT_BASE_URL}/v5/market/time", proxy=proxy_url, timeout=timeout) as resp:
            status = resp.status
            data = await resp.json()
            return status, data

    for attempt in range(3):
        try:
            _ensure_async_loop()
            future = asyncio.run_coroutine_threadsafe(do_time_sync(), _async_loop)
            status, res = future.result(timeout=7)
            if status == 200:
                server_time = int(res["result"]["timeNano"]) // 1000000
                local_time = int(time.time() * 1000)
                offset = server_time - local_time
                print(f"[Bybit API] Successfully synced time offset: {offset}ms")
                with _time_offset_lock:
                    _cached_time_offset = float(offset)
                    _last_time_sync = time.time()
                return offset
        except Exception as e:
            if attempt == 2:
                print(f"[Bybit API Error] Failed to sync time after 3 attempts: {e}")
            time.sleep(1)
    with _time_offset_lock:
        if _cached_time_offset is None:
            _cached_time_offset = 0.0
    return 0.0

def bybit_post_request(endpoint, payload):
    import bybit_client
    return bybit_client.bybit_post_request(endpoint, payload)

def set_bybit_leverage(symbol, leverage):
    payload = {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    }
    res = bybit_post_request("/v5/position/set-leverage", payload)
    # 110043 means leverage is already set to this value, which is safe to ignore
    if res.get("retCode") in [0, 110043]:
        print(f"[Bybit API] Leverage set to {leverage}x for {symbol} successfully.")
        return True
    else:
        print(f"[Bybit API Error] Failed to set leverage for {symbol}: {res.get('retMsg')} (code: {res.get('retCode')})")
        return False

def format_bybit_price(symbol, price):
    try:
        from bybit_client import quantize_bybit_price
        return quantize_bybit_price(symbol, price)
    except Exception:
        price_precisions = {
            "BTCUSDT": 2,
            "ETHUSDT": 2,
            "SOLUSDT": 3,
            "BNBUSDT": 2,
            "AVAXUSDT": 3,
            "NEARUSDT": 3,
            "LINKUSDT": 3,
            "LTCUSDT": 2,
            "ADAUSDT": 4,
            "XRPUSDT": 4,
            "DOGEUSDT": 5,
            "DOTUSDT": 3,
            "SUIUSDT": 4,
            "APTUSDT": 3
        }
        p = price_precisions.get(symbol, 2)
        return str(round(price, p))

_instrument_info_cache = {}
_instrument_info_cache_lock = threading.Lock()

def get_bybit_min_qty_step(symbol):
    now_t = time.time()
    with _instrument_info_cache_lock:
        if symbol in _instrument_info_cache:
            cached_step, exp_t = _instrument_info_cache[symbol]
            if now_t < exp_t:
                return cached_step

    # Try fetching dynamically from Bybit REST API
    try:
        url = f"{BYBIT_BASE_URL}/v5/market/instruments-info"
        res = requests.get(url, params={"category": "linear", "symbol": symbol}, proxies=get_bybit_proxies(), timeout=5)
        if res.status_code == 200:
            data = res.json()
            items = data.get("result", {}).get("list", [])
            if items:
                lot_info = items[0].get("lotSizeFilter", {})
                qty_step = float(lot_info.get("qtyStep", 0.0) or lot_info.get("minOrderQty", 0.0))
                if qty_step > 0:
                    with _instrument_info_cache_lock:
                        _instrument_info_cache[symbol] = (qty_step, now_t + 3600.0)
                    return qty_step
    except Exception as e:
        log_event("DEBUG", f"Dynamic instrument fetch fallback for {symbol}: {e}")

    # Static Fallback
    min_limits = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.01,
        "SOLUSDT": 0.1,
        "BNBUSDT": 0.1,
        "AVAXUSDT": 0.1,
        "NEARUSDT": 0.1,
        "LINKUSDT": 0.1,
        "LTCUSDT": 0.1,
        "ADAUSDT": 1.0,
        "XRPUSDT": 1.0,
        "DOGEUSDT": 1.0,
        "DOTUSDT": 1.0,
        "SUIUSDT": 1.0,
        "APTUSDT": 0.1
    }
    fallback_step = min_limits.get(symbol, 0.1)
    with _instrument_info_cache_lock:
        _instrument_info_cache[symbol] = (fallback_step, now_t + 300.0)
    return fallback_step






def wait_for_order_fill(symbol, order_id, timeout_sec=2.0):
    """
    Polls Bybit order status until filled, rejected/cancelled, or timeout expires.
    Treats PartiallyFilled as working (keeps waiting until Filled or timeout).
    Returns (is_filled, status, cum_exec_qty, avg_price).
    """
    start_t = time.time()
    last_status = "Unknown"
    cum_qty = 0.0
    avg_price = 0.0
    while time.time() - start_t < timeout_sec:
        details = get_bybit_order_details(symbol, order_id)
        if details:
            last_status = details.get("orderStatus", "Unknown")
            cum_qty = float(details.get("cumExecQty", 0.0))
            avg_price = float(details.get("avgPrice", 0.0))
            if last_status == "Filled":
                return True, "Filled", cum_qty, avg_price
            elif last_status in ["Cancelled", "Rejected"]:
                return False, last_status, cum_qty, avg_price
        time.sleep(0.2)
    return False, last_status, cum_qty, avg_price



def bybit_get_request(endpoint, query_params):
    import bybit_client
    return bybit_client.bybit_get_request(endpoint, query_params)

def get_bybit_position(symbol):
    res = bybit_get_request("/v5/position/list", {"category": "linear", "symbol": symbol})
    if res.get("retCode") == 0:
        pos_list = res.get("result", {}).get("list", [])
        for pos in pos_list:
            if pos.get("symbol") == symbol:
                return pos
    return None

def get_bybit_closed_pnl(symbol, limit=1):
    """Fetch the most recent closed PnL record from Bybit for exact settled realized PnL."""
    res = bybit_get_request("/v5/position/closed-pnl", {"category": "linear", "symbol": symbol, "limit": str(limit)})
    if res.get("retCode") == 0:
        pnl_list = res.get("result", {}).get("list", [])
        if pnl_list:
            return pnl_list[0]
    return None

def get_bybit_accumulated_closed_pnl(symbol, entry_time_ms, expected_total_qty=None):
    """Retrieve all closed PnL records for a symbol from entry_time_ms to now and sum them up."""
    if not entry_time_ms or float(entry_time_ms) <= 0:
        entry_time_ms = int((time.time() - 86400) * 1000)
    res = bybit_get_request("/v5/position/closed-pnl", {"category": "linear", "symbol": symbol, "limit": "50"})

    if res.get("retCode") == 0:
        pnl_list = res.get("result", {}).get("list", [])
        total_pnl = 0.0
        exit_prices = []
        exit_quantities = []
        entry_values = []
        
        matched_count = 0
        for item in pnl_list:
            updated_time = int(item.get("updatedTime", 0))
            if updated_time >= entry_time_ms:
                matched_count += 1
                pnl_val = float(item.get("closedPnl", 0.0))
                total_pnl += pnl_val
                
                ep = float(item.get("avgExitPrice", 0.0))
                eq = float(item.get("closedSize", 0.0))
                ev = float(item.get("cumEntryValue", 0.0))
                if ep > 0 and eq > 0:
                    exit_prices.append(ep)
                    exit_quantities.append(eq)
                if ev > 0:
                    entry_values.append(ev)
                    
        if matched_count == 0:
            return None

        # If trade previously scaled out, ensure all legs (cumulative closed quantity) are published
        if expected_total_qty is not None and expected_total_qty > 0:
            total_closed_qty = sum(exit_quantities)
            if total_closed_qty < 0.90 * expected_total_qty:
                # Missing final leg from venue lag — signal retry loop to wait for venue publication
                return None

        avg_exit_price = None
        if exit_prices and exit_quantities:
            total_qty = sum(exit_quantities)
            if total_qty > 0:
                avg_exit_price = sum(p * q for p, q in zip(exit_prices, exit_quantities)) / total_qty
                
        total_entry_val = sum(entry_values)
        
        return {
            "total_pnl": total_pnl,
            "avg_exit_price": avg_exit_price,
            "total_entry_value": total_entry_val if total_entry_val > 0 else None
        }
    return None

def update_bybit_stop_loss(symbol, sl_price, active_trade=None, current_sl_snapshot=None):
    if active_trade:
        qty_val = float(active_trade.get("qty", 0.0))
        side = "Buy" if active_trade.get("direction") in ["Bullish", "BUY", "LONG", "UP"] else "Sell"
    else:
        pos = get_bybit_position(symbol)
        if not pos:
            print(f"[Bybit API] Stop Loss update skipped for {symbol}: No active position found.")
            return False
        qty_val = float(pos.get("size", "0"))
        side = pos.get("side", "Buy")  # "Buy" for Long, "Sell" for Short
        
    if qty_val == 0:
        print(f"[Bybit API] Stop Loss update skipped for {symbol}: Position size is 0.")
        return False
        
    now_t = time.time()
    live_price = bot_state.get(f"live_price_{symbol}")
    price_ts = bot_state.get(f"live_price_ts_{symbol}", 0.0)
    if live_price is None or price_ts <= 0.0 or (now_t - price_ts > 30.0):
        fresh_price = get_fallback_price(symbol)
        if fresh_price is not None:
            live_price = fresh_price
            bot_state[f"live_price_{symbol}"] = fresh_price
            bot_state[f"live_price_ts_{symbol}"] = now_t
            price_ts = now_t

    if live_price is None or price_ts <= 0.0 or (now_t - price_ts > 30.0):
        print(f"[Bybit API] Stop Loss update skipped for {symbol}: Live price unavailable or stale (age {now_t - price_ts:.1f}s) and fallback failed.")
        return False
        
    if active_trade:
        current_sl = float(current_sl_snapshot) if current_sl_snapshot is not None else float(active_trade.get("stop_loss", 0.0))
        direction_val = active_trade.get("direction", "Bullish")
        curr_state_str = active_trade.get("stop_state", "INITIAL")
        target_state_str = active_trade.get("target_stop_state", curr_state_str)
        # Institutional Monotonic Stop Loss State Machine Invariant
        if current_sl > 0:
            is_valid_transition, reason_msg = StopStateMachine.validate_monotonic_stop_update(
                direction=direction_val,
                current_sl=current_sl,
                proposed_sl=sl_price,
                current_state_str=curr_state_str,
                target_state_str=target_state_str
            )
            if not is_valid_transition:
                print(f"[Bybit API Monotonic Guard] Rejected SL update: {reason_msg}.")
                return False
            active_trade["stop_state"] = target_state_str

    if live_price is not None:
        if side == "Buy" or side == "Long":  # Long position: Stop Loss must be < current price
            if sl_price >= live_price:
                print(f"[Bybit API] Stop Loss update skipped for Long {symbol}: Proposed SL {sl_price:.4f} is >= current price {live_price:.4f}.")
                return False
        else:  # Short position: Stop Loss must be > current price
            if sl_price <= live_price:
                print(f"[Bybit API] Stop Loss update skipped for Short {symbol}: Proposed SL {sl_price:.4f} is <= current price {live_price:.4f}.")
                return False

    payload = {
        "category": "linear",
        "symbol": symbol,
        "stopLoss": format_bybit_price(symbol, sl_price),
        "positionIdx": 0
    }
    res = bybit_post_request("/v5/position/trading-stop", payload)
    if res.get("retCode") == 0:
        print(f"[Bybit API] Successfully updated Stop Loss on Bybit to {sl_price:.4f} for {symbol}.")
        return True
    elif "not modified" in str(res.get("retMsg", "")).lower() or res.get("retCode") == 130089:
        print(f"[Bybit API] Stop Loss for {symbol} is already set to {sl_price:.4f} (not modified).")
        return True
    else:
        print(f"[Bybit API Error] Failed to update Stop Loss for {symbol}: {res.get('retMsg')}")
        return False

def update_bybit_take_profit(symbol, tp_price, active_trade=None):
    """Sync the Take Profit on the Bybit server."""
    if active_trade:
        qty_val = float(active_trade.get("qty", 0.0))
        side = "Buy" if active_trade.get("direction") == "Bullish" else "Sell"
    else:
        pos = get_bybit_position(symbol)
        if not pos:
            print(f"[Bybit API] Take Profit update skipped for {symbol}: No active position found.")
            return False
        qty_val = float(pos.get("size", "0"))
        side = pos.get("side", "Buy")  # "Buy" for Long, "Sell" for Short
        
    if qty_val == 0:
        print(f"[Bybit API] Take Profit update skipped for {symbol}: Position size is 0.")
        return False
        
    now_t = time.time()
    live_price = bot_state.get(f"live_price_{symbol}")
    price_ts = bot_state.get(f"live_price_ts_{symbol}", 0.0)
    if live_price is None or price_ts <= 0.0 or (now_t - price_ts > 30.0):
        fresh_price = get_fallback_price(symbol)
        if fresh_price is not None:
            live_price = fresh_price
            bot_state[f"live_price_{symbol}"] = fresh_price
            bot_state[f"live_price_ts_{symbol}"] = now_t
            price_ts = now_t

    if live_price is None or price_ts <= 0.0 or (now_t - price_ts > 30.0):
        print(f"[Bybit API] Take Profit update skipped for {symbol}: Live price unavailable or stale (age {now_t - price_ts:.1f}s) and fallback failed.")
        return False
        
    if live_price is not None:
        if side == "Buy" or side == "Long":  # Long position: Take Profit must be > current price
            if tp_price <= live_price:
                log_event("WARNING", f"[{symbol}] TP {tp_price} <= price {live_price} on a long — skipped")
                return False
        else:  # Short position: Take Profit must be < current price
            if tp_price >= live_price:
                log_event("WARNING", f"[{symbol}] TP {tp_price} >= price {live_price} on a short — skipped")
                return False
    payload = {
        "category": "linear",
        "symbol": symbol,
        "takeProfit": format_bybit_price(symbol, tp_price),
        "positionIdx": 0
    }
    res = bybit_post_request("/v5/position/trading-stop", payload)
    if res.get("retCode") == 0:
        print(f"[Bybit API] Take Profit for {symbol} updated to {tp_price:.4f} successfully.")
        return True
    elif "not modified" in str(res.get("retMsg", "")).lower() or res.get("retCode") == 130089:
        print(f"[Bybit API] Take Profit for {symbol} is already set to {tp_price:.4f} (not modified).")
        return True
    else:
        print(f"[Bybit API Error] Failed to update Take Profit for {symbol}: {res.get('retMsg')}")
        return False



def emergency_flatten_position(symbol, opp_side, qty_str, max_retries=3):
    """
    Executes a verified emergency flatten order with reduce_only=True and retries.
    Strictly verifies that live position size is zero before returning True.
    """
    for attempt in range(1, max_retries + 1):
        try:
            pos = get_bybit_position(symbol)
            if pos is not None:
                live_sz = float(pos.get("size", 0.0))
                if live_sz == 0.0:
                    log_event("INFO", f"[{symbol} Emergency Flatten] Position already confirmed flat on Bybit (size == 0.0).")
                    return True
                submit_qty = format_bybit_qty(symbol, live_sz)
            else:
                log_event("WARNING", f"[{symbol} Emergency Flatten] Position query returned None (API error/timeout on attempt {attempt}). Proceeding with fallback qty {qty_str}.")
                submit_qty = qty_str
                live_sz = 0.0

            res = place_bybit_taker_ioc_order(symbol, opp_side, submit_qty, reduce_only=True)
            ret_code = res.get("retCode", -1) if isinstance(res, dict) else -1
            time.sleep(0.4)
            pos_after = get_bybit_position(symbol)
            if pos_after is not None and float(pos_after.get("size", 0.0)) == 0.0:
                log_event("INFO", f"[{symbol} Emergency Flatten] Position strictly confirmed flat on Bybit (Attempt {attempt}, status code: {ret_code}).")
                return True
            else:
                rem = float(pos_after.get("size", 0.0)) if pos_after else live_sz
                log_event("WARNING", f"[{symbol} Emergency Flatten] Order execution left unconfirmed or residual size {rem} (Attempt {attempt}).")
        except Exception as ex:
            log_event("ERROR", f"[{symbol} Emergency Flatten] Exception on attempt {attempt}: {ex}")
        time.sleep(0.5)
    return False

def get_bybit_bid_ask(symbol):
    res = bybit_get_request("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if res.get("retCode") == 0:
        lst = res.get("result", {}).get("list", [])
        if lst:
            tick = lst[0]
            bid = float(tick.get("bid1Price", 0.0))
            ask = float(tick.get("ask1Price", 0.0))
            last = float(tick.get("lastPrice", 0.0))
            return bid, ask, last
    return None, None, None

def get_chase_limit_price(symbol, side, chase, entry_price):
    """Calculates dynamic Limit Maker price using Orderbook Imbalance (OBI) and real-time orderbook depth."""
    try:
        from bybit_client import calculate_optimal_maker_price, get_orderbook_imbalance
        obi_data = get_orderbook_imbalance(symbol, depth=10)
        if obi_data and obi_data.get("status") == "OK":
            base_maker = calculate_optimal_maker_price(symbol, side, obi_data=obi_data, reference_price=entry_price)
            bid = float(obi_data.get("best_bid", 0.0))
            ask = float(obi_data.get("best_ask", 0.0))
            if bid > 0 and ask > 0 and ask > bid:
                spread = ask - bid
                step = spread * 0.1 * min(chase, 8)
                if side in ["Buy", "Bullish", "LONG", "BUY"]:
                    return min(ask - (spread * 0.05), base_maker + step)
                else:
                    return max(bid + (spread * 0.05), base_maker - step)
            return float(base_maker)
    except Exception:
        pass

    bid, ask, last = get_bybit_bid_ask(symbol)
    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask > bid:
        spread = ask - bid
        step = spread * 0.1 * min(chase, 8)
        if side in ["Buy", "Bullish", "LONG", "BUY"]:
            return min(ask - (spread * 0.05), bid + step)
        else:
            return max(bid + (spread * 0.05), ask - step)
    return float(entry_price)

def get_bybit_last_execution(symbol, order_id=None):
    params = {"category": "linear", "symbol": symbol, "limit": 1}
    if order_id:
        params["orderId"] = str(order_id)
    res = bybit_get_request("/v5/execution/list", params)
    if res.get("retCode") == 0:
        exec_list = res.get("result", {}).get("list", [])
        if exec_list:
            return exec_list[0]
    return None

def get_bybit_order_executions(symbol, order_id=None, order_link_id=None):
    params = {"category": "linear", "symbol": symbol, "limit": 10}
    if order_id:
        params["orderId"] = str(order_id)
    if order_link_id:
        params["orderLinkId"] = str(order_link_id)
    res = bybit_get_request("/v5/execution/list", params)
    if res.get("retCode") == 0:
        return res.get("result", {}).get("list", [])
    return []

def get_real_bybit_balance():
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    
    if not api_key or not api_secret:
        return "API_KEYS_MISSING"
        
    geo_blocked_encountered = False
    
    # Finding #92: Query strictly derivatives-tradable balance (UNIFIED or CONTRACT).
    # Never query FUND or SPOT which would oversize risk against unusable/non-tradable spot capital.
    for account_type in ["UNIFIED", "CONTRACT"]:
        res = bybit_get_request("/v5/account/wallet-balance", {"accountType": account_type})
            
        ret_code = res.get("retCode")
        if ret_code == 0:
            list_data = res.get("result", {}).get("list", [])
            if list_data:
                # Prefer available balance for derivatives, fallback to wallet balance / total equity
                derivatives_balance = (
                    list_data[0].get("totalAvailableBalance")
                    or list_data[0].get("totalWalletBalance")
                    or list_data[0].get("totalEquity")
                    or "0"
                )
                bal_val = float(derivatives_balance)
                if bal_val > 0.0:
                    return bal_val
        else:
            ret_msg = res.get("retMsg", "")
            if isinstance(ret_code, int) and (400 <= ret_code <= 599):
                print(f"[Bybit Balance] HTTP {ret_code} for {account_type}: {ret_msg}")
                if ret_code == 403 and ("cloudfront" in ret_msg.lower() or "block" in ret_msg.lower()):
                    geo_blocked_encountered = True
            else:
                if not (ret_code in [10001, 10003] and account_type == "CONTRACT"):
                    print(f"[Bybit Balance] Query error for {account_type}: Code {ret_code} - {ret_msg}")
                    
    if geo_blocked_encountered:
        return "GEO_BLOCKED"
    return 0.0

def get_real_bybit_balance_cached(force=False):
    import bybit_client
    return bybit_client.get_real_bybit_balance_cached(force=force)

def run_bybit_balance_updater():
    import bybit_client
    return bybit_client.run_bybit_balance_updater(bot_state=bot_state, bot_state_lock=bot_state_lock)



# HTTP API routes extracted to dashboard_routes.py



def retrain_models_thread(is_manual=False):
    """
    Worker function to retrain all models (5m, 15m, 60m) inside a background thread.
    Uses retraining_lock to prevent concurrent retraining.
    """
    if not retraining_lock.acquire(blocking=False):
        print("[Retraining] Retraining is already in progress. Skipping request.")
        return False
    
    def run_training():
        global bot_state
        try:
            # Sync latest predictions and trade history from Hugging Face Space first
            load_history()
            bot_state["retraining_status"] = "Optimizing..."
            print(f"[Retraining] Starting {'manual ' if is_manual else 'scheduled '}rolling retraining of models for 1h, 2h, 4h, and 6h intervals...")
            
            import sys
            import subprocess
            
            # Configure environment variables to restrict multi-threading inside the child process
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["VECLIB_MAXIMUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            
            # Retrain for all intervals sequentially using nice -n 19 to yield CPU to active trading bot
            for iv in ["15", "30", "60", "120", "240"]:
                print(f"[Retraining] Spawning throttled subprocess for interval {iv}m...")
                cmd = ["nice", "-n", "19", sys.executable, "train.py", "--interval", iv, "--pages", "10", "--live-feedback"]
                p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = p.communicate()
                if p.returncode == 0:
                    print(f"[Retraining] Retraining for interval {iv}m finished successfully.")
                else:
                    print(f"[Retraining Error] Retraining for interval {iv}m failed: {stderr}")
                    
            print("[Retraining] Rolling retraining completed successfully. Model files updated on disk.")
            send_telegram_alert("🔄 *MODEL RETRAINING COMPLETE* 🔄\n• Rolling model retraining finished successfully.\n• Ensemble and meta-classifiers updated and re-loaded on disk.")
        except Exception as e:
            print(f"[Retraining] Error during retraining process: {e}")
        finally:
            bot_state["retraining_status"] = "Idle"
            save_history()
            retraining_lock.release()

    threading.Thread(target=run_training, daemon=True).start()
    return True

JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_journal.csv")
JOURNAL_HEADER = ["timestamp", "symbol", "interval", "direction", "entry_price", "exit_price", "pnl_usd", "change_pct", "success", "reason", "balance", "leverage", "confidence", "pnl_source"]

def log_trade_journal(trade: dict):
    """Append a closed trade to trade_journal.csv."""
    import csv
    write_header = not os.path.exists(JOURNAL_PATH)
    if os.path.exists(JOURNAL_PATH):
        try:
            with open(JOURNAL_PATH, "r") as f_r:
                first_line = f_r.readline().strip().split(",")
            if "pnl_source" not in first_line:
                with open(JOURNAL_PATH, "r") as f_r:
                    lines = f_r.readlines()
                if lines:
                    lines[0] = ",".join(JOURNAL_HEADER) + "\n"
                    for idx in range(1, len(lines)):
                        line_parts = lines[idx].strip().split(",")
                        if len(line_parts) == 13:
                            lines[idx] = lines[idx].strip() + ",ESTIMATED\n"
                    with open(JOURNAL_PATH, "w") as f_w:
                        f_w.writelines(lines)
                write_header = False
        except Exception as e_mig:
            print(f"[Journal Migration] Warning: {e_mig}")
    try:
        with open(JOURNAL_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_HEADER, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(trade.get("exit_time", time.time()))),
                "symbol": trade.get("symbol"),
                "interval": trade.get("interval"),
                "direction": trade.get("direction"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "pnl_usd": trade.get("pnl_usd"),
                "change_pct": trade.get("change_pct"),
                "success": trade.get("success"),
                "reason": trade.get("reason"),
                "balance": trade.get("balance"),
                "leverage": trade.get("leverage"),
                "confidence": trade.get("confidence"),
                "pnl_source": trade.get("pnl_source", "ESTIMATED"),
            })
    except Exception as e:
        print(f"[Journal] Failed to write journal: {e}")
    try:
        import database
        database.save_completed_trade(trade)
    except Exception as db_err:
        print(f"[Journal DB Sync Warning] {db_err}")

def send_daily_journal_digest():
    """Send a Telegram daily summary of yesterday's closed trades."""
    import csv
    target_day = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    if not os.path.exists(JOURNAL_PATH):
        send_telegram_alert(f"📓 *Daily Trade Journal — {target_day}*\nNo trades recorded today.")
        return
    try:
        wins, losses, total_pnl, rows = 0, 0, 0.0, []
        with open(JOURNAL_PATH, "r") as f:
            for row in csv.DictReader(f):
                if row["timestamp"].startswith(target_day):
                    rows.append(row)
                    pnl = float(row.get("pnl_usd", 0))
                    total_pnl += pnl
                    is_win = str(row.get("success", "")).lower() in ["true", "1"] or row.get("success") is True
                    if is_win:
                        wins += 1
                    else:
                        losses += 1

        total = wins + losses
        wr = f"{wins/total*100:.1f}%" if total > 0 else "N/A"
        lines = [f"📓 *Daily Trade Journal — {target_day}*",
                 f"• Trades: {total} | ✅ {wins} / ❌ {losses} | WR: {wr}",
                 f"• Total PnL: *${total_pnl:+.2f}*", ""]
        for r in rows[-10:]:
            r_win = str(r.get("success", "")).lower() in ["true", "1"] or r.get("success") is True
            emoji = "✅" if r_win else "❌"
            lines.append(f"{emoji} {r['symbol']} {r['direction']} {r['interval']}m | ${float(r['pnl_usd']):+.2f} | {r['reason']}")
        send_telegram_alert("\n".join(lines))
    except Exception as e:
        print(f"[Journal Digest] Error: {e}")

def run_daily_journal_scheduler():
    """Send daily Telegram digest at 00:00 UTC every day."""
    print("[Journal Scheduler] Daily digest scheduler started.")
    while True:
        now = time.gmtime()
        # Sleep until next midnight UTC
        seconds_to_midnight = 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
        time.sleep(seconds_to_midnight)
        try:
            send_daily_journal_digest()
        except Exception as e:
            print(f"[Journal Scheduler] Error: {e}")

def run_daily_backup_scheduler():
    """
    Background scheduler that runs daily at 00:00 UTC.
    Calculates time to UTC midnight, sleeps, then creates a compressed zip file
    of trading_bot.db and trade_journal.csv, uploading to AWS S3 if credentials are set.
    """
    print("[Backup Scheduler] Daily database backup scheduler started.")
    import zipfile
    import shutil
    from database import DB_FILE
    
    while True:
        now = time.gmtime()
        # Sleep until next midnight UTC
        seconds_to_midnight = 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
        time.sleep(max(1, seconds_to_midnight))
        
        try:
            print("[Backup Scheduler] Triggering daily backup...")
            backup_dir = "backups"
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            zip_filename = os.path.join(backup_dir, f"backup_{timestamp_str}.zip")
            
            current_db = database.get_db_path()
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(current_db):
                    zipf.write(current_db, os.path.basename(current_db))
                if os.path.exists(current_db + "-wal"):
                    zipf.write(current_db + "-wal", os.path.basename(current_db + "-wal"))
                if os.path.exists(current_db + "-shm"):
                    zipf.write(current_db + "-shm", os.path.basename(current_db + "-shm"))
                if os.path.exists(JOURNAL_PATH):
                    zipf.write(JOURNAL_PATH, os.path.basename(JOURNAL_PATH))
                    
            print(f"[Backup Scheduler] Created local compressed backup: {zip_filename}")
            
            # Prune local backups older than 7 days
            try:
                now_t = time.time()
                for fname in os.listdir(backup_dir):
                    fpath = os.path.join(backup_dir, fname)
                    if fname.startswith("backup_") and fname.endswith(".zip"):
                        if os.path.isfile(fpath) and (now_t - os.path.getmtime(fpath)) > 7 * 86400:
                            os.remove(fpath)
                            print(f"[Backup Scheduler] Pruned old backup: {fname}")
            except Exception as prune_err:
                print(f"[Backup Scheduler Warning] Failed pruning old backups: {prune_err}")
            
            # Optional S3 upload
            s3_bucket = os.environ.get("AWS_S3_BUCKET")
            if s3_bucket:
                try:
                    import boto3
                    s3_client = boto3.client('s3')
                    s3_key = f"backups/{os.path.basename(zip_filename)}"
                    s3_client.upload_file(zip_filename, s3_bucket, s3_key)
                    print(f"[Backup Scheduler] Successfully uploaded backup to S3: s3://{s3_bucket}/{s3_key}")
                except Exception as s3_err:
                    print(f"[Backup Scheduler Warning] S3 upload failed (boto3 or credentials missing): {s3_err}")
        except Exception as e:
            print(f"[Backup Scheduler Error] Daily backup failed: {e}")

def run_pain_feedback_verifier():
    """
    Background worker that runs hourly to verify whether closed pain trades hit TP within 24h post-exit.
    """
    print("[Pain Feedback Verifier] Hourly 24h post-exit verification scheduler started.")
    while True:
        try:
            time.sleep(3600)  # Check hourly
            from data import get_history
            import pain_feedback
            if hasattr(pain_feedback, 'verify_pending_pain_trades'):
                pain_feedback.verify_pending_pain_trades(database_module=database, fetch_kline_func=get_history)
            elif hasattr(pain_feedback, 'pain_feedback') and hasattr(pain_feedback.pain_feedback, 'verify_pending_pain_trades'):
                pain_feedback.pain_feedback.verify_pending_pain_trades(database_module=database, fetch_kline_func=get_history)
        except Exception as e:
            print(f"[Pain Feedback Verifier Error] Exception in verification loop: {e}")


def run_daily_summary_scheduler():

    """
    Background scheduler that guarantees daily 00:00:00 UTC report execution.
    Calculates time to UTC midnight, sleeps, and sends Telegram daily digest.
    """
    print("[Daily Summary Scheduler] Dedicated 00:00 UTC summary report scheduler started.")
    while True:
        try:
            now_gm = time.gmtime()
            # Sleep until next midnight UTC
            seconds_to_midnight = 86400 - (now_gm.tm_hour * 3600 + now_gm.tm_min * 60 + now_gm.tm_sec)
            time.sleep(max(1, seconds_to_midnight))
            
            # Re-read time upon waking up
            wake_gm = time.gmtime()
            today_date_str = time.strftime("%Y-%m-%d", wake_gm)
            should_send = False
            with bot_state_lock:
                last_date = bot_state.get("last_daily_summary_date", "")
                if last_date != today_date_str:
                    bot_state["last_daily_summary_date"] = today_date_str
                    should_send = True
            
            if should_send:
                print(f"[Daily Summary Scheduler] Midnight UTC detected ({today_date_str}). Sending daily summary...")
                send_daily_summary()
        except Exception as e:
            print(f"[Daily Summary Scheduler Error] Exception in scheduler loop: {e}")
            time.sleep(60)

def run_rolling_retrain_scheduler():
    """
    Background scheduler that runs weekly on Sundays at 00:00 UTC.
    Checks time every 15 minutes.
    """
    log_event("INFO", "[Scheduler] Automated weekly Sunday retraining scheduler started.")
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            # Sunday is weekday 6. Check if it is Sunday between 00:00 and 00:20 UTC.
            if now_utc.weekday() == 6 and now_utc.hour == 0 and now_utc.minute < 20:
                log_event("INFO", f"[Scheduler] Sunday 00:00 UTC detected. Triggering weekly model retraining...")
                retrain_models_thread(is_manual=False)
                # Sleep 45 minutes to prevent double trigger within the same hour
                time.sleep(2700)
        except Exception as e:
            log_event("ERROR", f"[Scheduler] Error in weekly retraining scheduler: {e}")
        
        # Sleep for 15 minutes before checking time again
        time.sleep(900)

def check_champion_models_staleness(max_age_days: float = 14.0):
    """
    Checks age of champion model files across active intervals (Finding #40).
    Warns if any active champion model is older than max_age_days (14 days).
    """
    stale_models = []
    now = time.time()
    for iv in ["15", "30", "60", "120", "240"]:
        candidates = [
            f"ensemble_trending_trend_{iv}_xgb.json",
            f"ensemble_trending_price_{iv}_xgb.json",
            f"ensemble_ranging_trend_{iv}_xgb.json",
            f"ensemble_ranging_price_{iv}_xgb.json",
            f"ensemble_trending_trend_{iv}_manifest.json"
        ]
        for f in candidates:
            if os.path.exists(f):
                mtime = os.path.getmtime(f)
                age_days = (now - mtime) / 86400.0
                if age_days > max_age_days:
                    stale_models.append((f, iv, age_days))
                    break
    if stale_models:
        for f, iv, age_days in stale_models:
            msg = f"[Model Governance WARNING] Champion model ({f}) for {iv}m is {age_days:.1f} days old (> {max_age_days}d). Weekly retrain is scheduled for Sunday 00:00 UTC."
            log_event("WARNING", msg)
        if TRADE_MODE != "simulation":
            try:
                stale_summary = ", ".join([f"{iv}m ({age:.0f}d)" for _, iv, age in stale_models])
                send_telegram_alert(f"⚠️ *MODEL GOVERNANCE NOTICE*: Champion models older than {max_age_days:.0f}d detected: {stale_summary}. Automatic weekly retrain runs Sunday 00:00 UTC.")
            except Exception as ex_tg_stale:
                log_event("DEBUG", f"[Model Governance Notice] Could not send staleness alert: {ex_tg_stale}")


import logging
import wsgiref.simple_server
from werkzeug.serving import run_simple

def run_flask():
    import sys
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    port = int(os.environ.get("PORT", 5001))
    allow_pub = os.environ.get("DASHBOARD_ALLOW_PUBLIC", "false").lower() in ("true", "1")
    default_host = "0.0.0.0" if allow_pub else "127.0.0.1"  # nosec B104
    flask_host = os.environ.get("FLASK_HOST", default_host)
    sys.stderr.write(f"[Flask] Starting server on {flask_host}:{port}...\n")
    sys.stderr.flush()
    try:
        run_simple(flask_host, port, app, use_reloader=False, threaded=True)
    except Exception as e:
        sys.stderr.write(f"[Flask Error] Failed starting Flask server: {e}\n")
        sys.stderr.flush()



# =========================
# CONFIGURATION
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "60"
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT"]

# =========================
from xgboost import XGBClassifier, XGBRegressor
import joblib
from ensemble import load_ensemble_classifier, load_ensemble_regressor, _slice_model_input

# Startup Barrier Drift Assertion: Compare TIMEFRAME_CONFIG against optimized_barriers_*.json and active manifests
from config import TIMEFRAME_CONFIG, COMMITTED_TIMEFRAME_CONFIG, REJECTED_BARRIER_FILES
for _iv in ["15", "30", "60", "120", "240"]:
    _opt_path = f"optimized_barriers_{_iv}.json"
    if os.path.exists(_opt_path):
        if _opt_path in REJECTED_BARRIER_FILES:
            log_event("WARNING", f"[Startup Barrier Audit] {_opt_path} was rejected by config geometric/bound validation; preserving committed config.")
            continue
        try:
            with open(_opt_path, "r") as _of:
                _ob = json.load(_of)
            _committed = COMMITTED_TIMEFRAME_CONFIG.get(_iv, {})
            _cfg = TIMEFRAME_CONFIG.get(_iv, {})
            for _k in ["tp_mult_trending", "tp_mult_ranging", "sl_mult", "lookahead"]:
                if _k in _ob and _k in _committed:
                    _drift = abs(float(_ob[_k]) - float(_committed[_k]))
                    if _drift > 1e-9:
                        log_event("INFO", f"[Startup Barrier Override] {_iv}m {_k} overridden by {_opt_path}: {_committed[_k]} -> {_ob[_k]}")
        except Exception as _e:
            log_event("WARNING", f"[Startup Barrier Audit] Could not verify {_opt_path}: {_e}")

    # Finding #15: Cross-check against active manifest barrier_config (non-tautological)
    _mf_path = f"ensemble_trending_trend_{_iv}_manifest.json"
    if os.path.exists(_mf_path):
        try:
            with open(_mf_path, "r") as _mf_file:
                _mf_data = json.load(_mf_file)
            _barriers = _mf_data.get("barrier_config", {})
            _cfg = TIMEFRAME_CONFIG.get(_iv, {})
            is_denylisted = f"trending_{_iv}" in getattr(config, "MODEL_SLOT_DENYLIST", []) or _mf_data.get("promoted") is False
            for _k in ["tp_mult_trending", "tp_mult_ranging", "sl_mult", "lookahead"]:
                if _k in _barriers and _k in _cfg:
                    _diff = abs(float(_barriers[_k]) - float(_cfg[_k]))
                    if _diff > 0.05:
                        if is_denylisted:
                            log_event("WARNING", f"[Startup Barrier Audit] Divergence in denylisted/unpromoted slot {_mf_path} ({_k}: manifest {_barriers[_k]} vs live {_cfg[_k]}).")
                            break
                        raise RuntimeError(
                            f"[Startup Drift Error] TIMEFRAME_CONFIG['{_iv}']['{_k}'] ({_cfg[_k]}) "
                            f"diverges from manifest {_mf_path} ({_barriers[_k]}) by {_diff:.4f} > 0.05. Boot aborted."
                        )
        except Exception as _e:
            if isinstance(_e, RuntimeError):
                raise
            log_event("WARNING", f"[Startup Barrier Audit] Could not verify {_mf_path}: {_e}")

models_by_interval = {}
model_files_mtime = {}

for iv in ["15", "30", "60", "120", "240"]:
    models_by_interval[iv] = {
        "trending": {
            "trend": None,
            "price": None,
            "meta": None,
            "calibrator": None
        },
        "ranging": {
            "trend": None,
            "price": None,
            "meta": None,
            "calibrator": None
        },
        "selected_features": None
    }

def load_model_weights(iv):
    from config import MODEL_SLOT_DENYLIST
    if f"trending_{iv}" in MODEL_SLOT_DENYLIST and f"ranging_{iv}" in MODEL_SLOT_DENYLIST:
        models_by_interval.setdefault(iv, {})["_fully_denied"] = True
        log_event("INFO", f"[{iv}m] Both trending and ranging models denied by governance policy — interval offline.")
        return

    load_iv = iv
    if iv == "30" and not os.path.exists(f"ensemble_trending_trend_{iv}_xgb.json"):
        load_iv = "15"

    if load_iv != iv and load_iv in models_by_interval and models_by_interval[load_iv].get("trending", {}).get("trend") is not None:
        models_by_interval[iv] = models_by_interval[load_iv]
        return

    prefixes = {
        "trending_trend": f"ensemble_trending_trend_{load_iv}",
        "trending_price": f"ensemble_trending_price_{load_iv}",
        "ranging_trend": f"ensemble_ranging_trend_{load_iv}",
        "ranging_price": f"ensemble_ranging_price_{load_iv}",
        "trending_meta": f"meta_trending_trend_{load_iv}.json",
        "ranging_meta": f"meta_ranging_trend_{load_iv}.json"
    }
    
    # Update modification times
    for key, filename in prefixes.items():
        if os.path.exists(filename):
            model_files_mtime[f"{iv}_{key}"] = os.path.getmtime(filename)
        elif os.path.exists(f"{filename}_xgb.json"):
            model_files_mtime[f"{iv}_{key}"] = os.path.getmtime(f"{filename}_xgb.json")
            
    # Load
    try:
        # Load selected features independently from feature contract definition files
        feat_trending = None
        feat_ranging = None
        
        for f_name in [f"selected_features_{iv}_trending.json", f"selected_features_{iv}.json"]:
            if os.path.exists(f_name):
                try:
                    with open(f_name, "r") as f:
                        feat_trending = json.load(f)
                        if feat_trending:
                            break
                except Exception as e:
                    log_event("WARNING", f"Could not load {f_name}: {e}")

        for f_name in [f"selected_features_{iv}_ranging.json", f"selected_features_{iv}.json"]:
            if os.path.exists(f_name):
                try:
                    with open(f_name, "r") as f:
                        feat_ranging = json.load(f)
                        if feat_ranging:
                            break
                except Exception as e:
                    log_event("WARNING", f"Could not load {f_name}: {e}")

        if not feat_trending:
            from core import features as master_features
            feat_trending = master_features
        if not feat_ranging:
            from core import features as master_features
            feat_ranging = master_features
                
        from train import NON_STATIONARY_EXCLUDE
        if feat_trending:
            feat_trending = [f for f in feat_trending if f not in NON_STATIONARY_EXCLUDE]
        if feat_ranging:
            feat_ranging = [f for f in feat_ranging if f not in NON_STATIONARY_EXCLUDE]

        if feat_trending is None or feat_ranging is None:
            print(f"[Model Load Warning] selected_features for interval {iv} missing! Disabling model loading.")
            models_by_interval[iv]["selected_features"] = None
            return
            
        models_by_interval[iv]["selected_features_trending"] = feat_trending
        models_by_interval[iv]["selected_features_ranging"] = feat_ranging
        models_by_interval[iv]["selected_features"] = feat_trending
        
        n_features_trending = len(feat_trending)
        n_features_ranging = len(feat_ranging)
        print(f"Loaded feature counts - Trending: {n_features_trending}, Ranging: {n_features_ranging} for interval {iv}")
        
        from model_governance import extract_metric, validate_manifest_governance_floors
        manifest_by_prefix = {}

        def check_startup_manifest_health(prefix: str) -> bool:
            manifest_path = f"{prefix}_manifest.json"
            if not os.path.exists(manifest_path):
                msg = f"Model manifest missing for {prefix} ({iv}m)."
                log_event("CRITICAL", f"[Model Manifest Error] {msg}")
                send_telegram_alert(f"🚨 *MODEL MANIFEST MISSING* 🚨\n• *Model*: {prefix}\n• *Interval*: {iv}m\n• *Action*: Engaging Rule-Based Fallback")
                return False
            try:
                with open(manifest_path, "r") as f:
                    m = json.load(f)
                ok, reason = validate_manifest_governance_floors(m, str(iv))
                if not ok:
                    log_event("CRITICAL", f"[Model Governance Rejection] {reason} for {prefix} ({iv}m).")
                    send_telegram_alert(f"🚨 *MODEL GOVERNANCE REJECTION* 🚨\n• *Model*: {prefix}\n• *Interval*: {iv}m\n• *Reason*: {reason}")
                    return False

                from ensemble import verify_manifest_hmac_signature
                if not verify_manifest_hmac_signature(m):
                    sig_reason = "HMAC signature verification failed (tampered manifest)"
                    log_event("CRITICAL", f"[Model Manifest Security] {sig_reason} for {prefix} ({iv}m).")
                    send_telegram_alert(f"🚨 *TAMPERED MANIFEST DETECTED* 🚨\n• *Model*: {prefix}\n• *Interval*: {iv}m\n• *Reason*: {sig_reason}")
                    return False

                manifest_by_prefix[prefix] = m
                from model_governance import model_governance_engine
                model_governance_engine.log_barrier_manifest_audit("BTCUSDT", str(iv), m)
                return True
            except Exception as e:
                msg = f"Corrupted model manifest for {prefix} ({iv}m): {e}"
                log_event("CRITICAL", f"[Model Manifest Error] {msg}")
                send_telegram_alert(f"🚨 *CORRUPTED MANIFEST* 🚨\n• *Model*: {prefix}\n• *Interval*: {iv}m\n• *Error*: {str(e)}")
                return False

        from mlops_engine import load_production_model_from_registry

        # 1. Trending Classifier
        models_by_interval[iv]["trending"]["trend"] = None
        from config import MODEL_SLOT_DENYLIST
        if f"trending_{iv}" in MODEL_SLOT_DENYLIST:
            log_event("WARNING", f"[Model Slot Denylist] Skipping load for denied model slot 'trending_{iv}' (Fail-Closed Abstain).")
        else:
            try:
                reg_model_trending, ver_trending = load_production_model_from_registry(interval=str(iv), regime="trending", live_features=feat_trending)
                if reg_model_trending is not None:
                    if check_startup_manifest_health(prefixes['trending_trend']):
                        models_by_interval[iv]["trending"]["trend"] = reg_model_trending
                        models_by_interval[iv]["trending"]["model_version"] = ver_trending
                    else:
                        log_event("CRITICAL", f"[Model Governance Rejection] MLflow production model {ver_trending} failed startup manifest health check.")
                elif os.path.exists(f"{prefixes['trending_trend']}_xgb.json") and check_startup_manifest_health(prefixes['trending_trend']):
                    models_by_interval[iv]["trending"]["trend"] = load_ensemble_classifier(prefixes["trending_trend"], n_features_trending, feature_names=feat_trending)
                    m_tr = manifest_by_prefix.get(prefixes['trending_trend'], {})
                    models_by_interval[iv]["trending"]["manifest"] = m_tr
                    models_by_interval[iv]["trending"]["model_version"] = m_tr.get("model_version") or f"btc_{iv}m_trending_clf:v1.0"
                    models_by_interval[iv]["trending"]["git_sha"] = m_tr.get("git_sha")
                    models_by_interval[iv]["trending"]["manifest_schema_version"] = m_tr.get("manifest_schema_version")
                    models_by_interval[iv]["trending"]["feature_contract_hash"] = m_tr.get("feature_contract_hash") or m_tr.get("feature_hash")
                    models_by_interval[iv]["trending"]["calibrator_version"] = m_tr.get("calibrator_version")
                    models_by_interval[iv]["trending"]["calibrator_ece"] = m_tr.get("calibrator_ece")
            except Exception as e:
                log_event("CRITICAL", f"[Model Load Error] Refused/failed to load {prefixes['trending_trend']} for {iv}m: {e}")
                send_telegram_alert(f"🚨 *MODEL GOVERNANCE LOAD FAILURE* 🚨\n• *Model*: {prefixes['trending_trend']}\n• *Interval*: {iv}m\n• *Reason*: {str(e)}")

            # 2. Trending Regressor
            try:
                if os.path.exists(f"{prefixes['trending_price']}_xgb.json") and check_startup_manifest_health(prefixes['trending_price']):
                    models_by_interval[iv]["trending"]["price"] = load_ensemble_regressor(prefixes["trending_price"], n_features_trending, feature_names=feat_trending)
            except Exception as e:
                log_event("CRITICAL", f"[Model Load Error] Refused/failed to load {prefixes['trending_price']} for {iv}m: {e}")
                send_telegram_alert(f"🚨 *MODEL GOVERNANCE LOAD FAILURE* 🚨\n• *Model*: {prefixes['trending_price']}\n• *Interval*: {iv}m\n• *Reason*: {str(e)}")

            # 3. Trending Meta Classifier
            try:
                if os.path.exists(prefixes["trending_meta"]):
                    if check_startup_manifest_health(prefixes['trending_trend']):
                        meta_clf = XGBClassifier()
                        meta_clf.load_model(prefixes["trending_meta"])
                        models_by_interval[iv]["trending"]["meta"] = meta_clf
                    else:
                        log_event("WARNING", f"[Model Load Warning] Skipped {prefixes['trending_meta']} due to failed manifest health check.")
            except Exception as e:
                log_event("WARNING", f"[Model Load Warning] Failed to load {prefixes['trending_meta']}: {e}")

        # 4. Ranging Classifier & Regressor (Loaded when dynamic regime routing is enabled)
        models_by_interval[iv]["ranging"]["trend"] = None
        from config import ENABLE_DYNAMIC_REGIME_ROUTING, DYNAMIC_REGIME_ROUTING_INTERVALS
        is_ranging_enabled_for_iv = ENABLE_DYNAMIC_REGIME_ROUTING or (str(iv) in DYNAMIC_REGIME_ROUTING_INTERVALS)
        if is_ranging_enabled_for_iv and f"ranging_{iv}" not in MODEL_SLOT_DENYLIST:
            try:
                reg_model_ranging, ver_ranging = load_production_model_from_registry(interval=str(iv), regime="ranging", live_features=feat_ranging)
                if reg_model_ranging is not None:
                    if check_startup_manifest_health(prefixes['ranging_trend']):
                        models_by_interval[iv]["ranging"]["trend"] = reg_model_ranging
                        models_by_interval[iv]["ranging"]["model_version"] = ver_ranging
                    else:
                        log_event("CRITICAL", f"[Model Governance Rejection] MLflow production model {ver_ranging} failed startup manifest health check.")
                elif os.path.exists(f"{prefixes['ranging_trend']}_xgb.json") and check_startup_manifest_health(prefixes['ranging_trend']):
                    models_by_interval[iv]["ranging"]["trend"] = load_ensemble_classifier(prefixes["ranging_trend"], n_features_ranging, feature_names=feat_ranging)
                    m_rg = manifest_by_prefix.get(prefixes['ranging_trend'], {})
                    models_by_interval[iv]["ranging"]["manifest"] = m_rg
                    models_by_interval[iv]["ranging"]["model_version"] = m_rg.get("model_version") or f"btc_{iv}m_ranging_clf:v1.0"
                    models_by_interval[iv]["ranging"]["git_sha"] = m_rg.get("git_sha")
                    models_by_interval[iv]["ranging"]["manifest_schema_version"] = m_rg.get("manifest_schema_version")
                    models_by_interval[iv]["ranging"]["feature_contract_hash"] = m_rg.get("feature_contract_hash") or m_rg.get("feature_hash")
                    models_by_interval[iv]["ranging"]["calibrator_version"] = m_rg.get("calibrator_version")
                    models_by_interval[iv]["ranging"]["calibrator_ece"] = m_rg.get("calibrator_ece")
            except Exception as e:
                log_event("CRITICAL", f"[Model Load Error] Refused/failed to load {prefixes['ranging_trend']} for {iv}m: {e}")
                send_telegram_alert(f"🚨 *MODEL GOVERNANCE LOAD FAILURE* 🚨\n• *Model*: {prefixes['ranging_trend']}\n• *Interval*: {iv}m\n• *Reason*: {str(e)}")

            # 5. Ranging Regressor
            try:
                if os.path.exists(f"{prefixes['ranging_price']}_xgb.json") and check_startup_manifest_health(prefixes['ranging_price']):
                    models_by_interval[iv]["ranging"]["price"] = load_ensemble_regressor(prefixes["ranging_price"], n_features_ranging, feature_names=feat_ranging)
            except Exception as e:
                log_event("CRITICAL", f"[Model Load Error] Refused/failed to load {prefixes['ranging_price']} for {iv}m: {e}")
                send_telegram_alert(f"🚨 *MODEL GOVERNANCE LOAD FAILURE* 🚨\n• *Model*: {prefixes['ranging_price']}\n• *Interval*: {iv}m\n• *Reason*: {str(e)}")

            # 6. Ranging Meta Classifier
            try:
                if os.path.exists(prefixes["ranging_meta"]):
                    if check_startup_manifest_health(prefixes['ranging_trend']):
                        meta_clf = XGBClassifier()
                        meta_clf.load_model(prefixes["ranging_meta"])
                        models_by_interval[iv]["ranging"]["meta"] = meta_clf
                    else:
                        log_event("WARNING", f"[Model Load Warning] Skipped {prefixes['ranging_meta']} due to failed manifest health check.")
            except Exception as e:
                log_event("WARNING", f"[Model Load Warning] Failed to load {prefixes['ranging_meta']}: {e}")

        # 7. Calibrators (Always isolated and loaded regardless of model load status)
        def verify_calibrator_barrier_geometry(cal_obj: dict, cal_file: str, regime: str) -> bool:
            if not isinstance(cal_obj, dict):
                return False
            b_geom = cal_obj.get("barrier_geometry")
            if not b_geom or not isinstance(b_geom, dict):
                man_path = f"{prefixes.get(f'{regime}_trend', '')}_manifest.json"
                if os.path.exists(man_path):
                    try:
                        with open(man_path, "r") as mf:
                            m_data = json.load(mf)
                        b_geom = m_data.get("barrier_config")
                    except Exception:
                        b_geom = None
            if not b_geom or not isinstance(b_geom, dict):
                msg = f"Calibrator '{cal_file}' and manifest missing barrier geometry."
                log_event("CRITICAL", f"[Calibrator Barrier Error] {msg} Slot set to None (Fail-Closed).")
                send_telegram_alert(f"🚨 *CALIBRATOR BARRIER MISSING* 🚨\n• File: `{cal_file}`\n• Interval: `{iv}m`\n• {msg}\n• Action: *Trading Disabled for this slot (Fail-Closed)*")
                return False
            from config import TIMEFRAME_CONFIG
            cfg_live = TIMEFRAME_CONFIG.get(str(iv), {})
            live_tp = float(cfg_live.get(f"tp_mult_{regime}", cfg_live.get("tp_mult_trending", 1.85)))
            live_sl = float(cfg_live.get("sl_mult", 0.85))
            live_lh = int(cfg_live.get("lookahead", 12))

            cal_tp = float(b_geom.get(f"tp_mult_{regime}", b_geom.get("tp_mult_trending", live_tp)))
            cal_sl = float(b_geom.get("sl_mult", live_sl))
            cal_lh = int(b_geom.get("lookahead", live_lh))

            # Finding #76: Tighten tolerances to ensure high-fidelity barrier calibration
            if abs(cal_tp - live_tp) > 0.10 or abs(cal_sl - live_sl) > 0.05 or abs(cal_lh - live_lh) > 1:
                msg = f"Calibrator '{cal_file}' barrier geometry (TP={cal_tp:.2f}, SL={cal_sl:.2f}, LH={cal_lh}) diverges from active TIMEFRAME_CONFIG (TP={live_tp:.2f}, SL={live_sl:.2f}, LH={live_lh})."
                log_event("CRITICAL", f"[Calibrator Barrier Divergence] {msg} Slot set to None (Fail-Closed).")
                send_telegram_alert(f"🚨 *CALIBRATOR BARRIER DIVERGENCE* 🚨\n• File: `{cal_file}`\n• Interval: `{iv}m`\n• {msg}\n• Action: *Trading Disabled for this slot (Fail-Closed)*")
                return False

            # Finding #96 & #100: Check economic viability against fee-inclusive break-even p* & target compatibility
            from tools.beta_calibrator import is_calibrator_viable
            haircut = getattr(config, "REALIZED_RR_HAIRCUT", 0.28)
            eff_tp = live_tp * haircut
            roundtrip_cost = 0.0010
            p_star = (live_sl + roundtrip_cost) / (eff_tp + live_sl)
            if not is_calibrator_viable(cal_obj, min_required_p_star=p_star, require_target_def=True):
                msg = f"Calibrator '{cal_file}' achievable probability ceiling cannot reach break-even p* ({p_star:.4f}) under live R:R or lacks required target_definition."
                log_event("CRITICAL", f"[Calibrator Non-Viable] {msg} Slot set to None (Fail-Closed).")
                send_telegram_alert(f"🚨 *CALIBRATOR NON-VIABLE* 🚨\n• File: `{cal_file}`\n• Interval: `{iv}m`\n• {msg}\n• Action: *Trading Disabled for this slot (Fail-Closed)*")
                return False

            target_def = cal_obj.get("target_definition")
            if not target_def or target_def not in ["triple_barrier_exact", "triple_barrier"]:
                msg = f"Calibrator '{cal_file}' target definition '{target_def}' is missing or incompatible with live execution engine."
                log_event("CRITICAL", f"[Calibrator Target Incompatible] {msg} Slot set to None (Fail-Closed).")
                return False

            return True

        try:
            trending_cal_file = f"calibrator_trending_{iv}.json"
            if os.path.exists(trending_cal_file):
                with open(trending_cal_file, "r") as f:
                    cal_data = json.load(f)
                if verify_calibrator_barrier_geometry(cal_data, trending_cal_file, "trending"):
                    models_by_interval[iv]["trending"]["calibrator"] = cal_data
                    print(f"Loaded Isotonic Regression calibrator: {trending_cal_file}")
                else:
                    models_by_interval[iv]["trending"]["calibrator"] = None
            else:
                models_by_interval[iv]["trending"]["calibrator"] = None
                log_event("CRITICAL", f"[Calibrator Missing] Trending calibrator '{trending_cal_file}' not found. Slot set to None (Fail-Closed).")
                send_telegram_alert(f"🚨 *MISSING CALIBRATOR ARTIFACT* 🚨\n• Interval: `{iv}m`\n• Regime: `Trending`\n• File: `{trending_cal_file}`\n• Action: *Trading Disabled for this slot (Fail-Closed)*")
        except Exception as e:
            models_by_interval[iv]["trending"]["calibrator"] = None
            log_event("CRITICAL", f"[Calibrator Error] Failed loading trending calibrator for {iv}m: {e}. Slot set to None (Fail-Closed).")
            send_telegram_alert(f"🚨 *CALIBRATOR LOAD ERROR* 🚨\n• Interval: `{iv}m`\n• Regime: `Trending`\n• Error: {e}\n• Action: *Trading Disabled for this slot (Fail-Closed)*")

        try:
            ranging_cal_file = f"calibrator_ranging_{iv}.json"
            if os.path.exists(ranging_cal_file):
                with open(ranging_cal_file, "r") as f:
                    cal_data = json.load(f)
                if verify_calibrator_barrier_geometry(cal_data, ranging_cal_file, "ranging"):
                    models_by_interval[iv]["ranging"]["calibrator"] = cal_data
                    print(f"Loaded Isotonic Regression calibrator: {ranging_cal_file}")
                else:
                    models_by_interval[iv]["ranging"]["calibrator"] = None
            else:
                models_by_interval[iv]["ranging"]["calibrator"] = None
                log_event("CRITICAL", f"[Calibrator Missing] Ranging calibrator '{ranging_cal_file}' not found. Slot set to None (Fail-Closed).")
                send_telegram_alert(f"🚨 *MISSING CALIBRATOR ARTIFACT* 🚨\n• Interval: `{iv}m`\n• Regime: `Ranging`\n• File: `{ranging_cal_file}`\n• Action: *Trading Disabled for this slot (Fail-Closed)*")
        except Exception as e:
            models_by_interval[iv]["ranging"]["calibrator"] = None
            log_event("CRITICAL", f"[Calibrator Error] Failed loading ranging calibrator for {iv}m: {e}. Slot set to None (Fail-Closed).")
            send_telegram_alert(f"🚨 *CALIBRATOR LOAD ERROR* 🚨\n• Interval: `{iv}m`\n• Regime: `Ranging`\n• Error: {e}\n• Action: *Trading Disabled for this slot (Fail-Closed)*")
            
        print(f"Successfully loaded ensemble and meta models for interval {iv}")
        try:
            import gc, ctypes
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (AttributeError, OSError):
                pass
        except (ImportError, AttributeError, OSError):
            pass
    except Exception as e:
        log_event("CRITICAL", f"[Model Load Fatal Error] Fatal unexpected error loading interval {iv}: {e}")
        send_telegram_alert(f"🚨 *FATAL MODEL LOADING ERROR* 🚨\n• *Interval*: {iv}m\n• *Error*: {str(e)}")



def check_and_hot_reload_models():
    reloaded = []
    for iv in ["15", "30", "60", "120", "240"]:
        filenames = {
            "trending_trend": f"ensemble_trending_trend_{iv}_xgb.json",
            "trending_price": f"ensemble_trending_price_{iv}_xgb.json",
            "trending_meta": f"meta_trending_trend_{iv}.json",
            "ranging_trend": f"ensemble_ranging_trend_{iv}_xgb.json",
            "ranging_price": f"ensemble_ranging_price_{iv}_xgb.json",
            "ranging_meta": f"meta_ranging_trend_{iv}.json"
        }
        
        changed = False
        for key, filename in filenames.items():
            if os.path.exists(filename):
                current_mtime = os.path.getmtime(filename)
                mtime_key = f"{iv}_{key}"
                if mtime_key not in model_files_mtime or current_mtime > model_files_mtime[mtime_key]:
                    changed = True
                    break
                    
        if changed:
            print(f"[Hot-Reload] Model update detected for {iv} on disk. Reloading in memory...")
            try:
                load_model_weights(iv)
                for key, filename in filenames.items():
                    if os.path.exists(filename):
                        model_files_mtime[f"{iv}_{key}"] = os.path.getmtime(filename)
                try:
                    p95, max_conf = calculate_historical_thresholds(models_by_interval[iv]["trending"]["trend"], iv)
                    tf_map_startup = {"15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h"}
                    tf_key = tf_map_startup[iv]
                    bot_state[f"calibration_{tf_key}"] = {
                        "p95": p95,
                        "max_conf": max_conf,
                        "mean": 54.81
                    }
                    log_event("INFO", f"[Hot-Reload] Recalculated calibration thresholds for {iv} (p95: {p95:.2f}, max_conf: {max_conf:.2f})")
                except Exception as e:
                    log_event("WARNING", f"[Hot-Reload] Warning: Could not recalculate thresholds for {iv}m: {e}")
                reloaded.append(iv)
            except Exception as exc:
                import logging
                logging.exception(f"[Hot-Reload] Exception reloading models for interval {iv}: {exc}")
    return reloaded

# =========================
# WEB SOCKET FOR LIVE PRICE
# =========================
live_price = None
last_ws_update_time = 0.0
ws_connected = False  # Track if WebSocket is currently connected
ws_retry_delay = 3  # Reconnection backoff delay (reset on successful connect)

def run_fallback_price_updater():
    """
    Periodic thread that queries Bybit REST API for spot prices.
    Acts as a failover if WebSocket is geoblocked or disconnected.
    Polls adaptively: every 10s when WS is down, every 5min when WS is active.
    """
    global live_price, last_ws_update_time
    print("[Price Fallback] Background updater thread started.")
    last_fallback_run = 0.0
    last_binance_run = 0.0
    while True:
        try:
            now = time.time()
            ws_active = ws_connected and (now - last_ws_update_time < 30)
            has_active_trades = any(len(bot_state.get(f"active_trade_{tf}", [])) > 0 for tf in ACTIVE_TRADE_TF_KEYS)

            # Adaptive interval: 20s if WS down with trades, 60s if WS down idle, 900s if WS up
            if not ws_active:
                poll_interval = 20 if has_active_trades else 60
                binance_interval = 60 if has_active_trades else 180
            else:
                poll_interval = 900  # Throttle to 15 minutes to minimize proxy bandwidth
                binance_interval = 900

            if now - last_fallback_run >= poll_interval:
                last_fallback_run = now
                url = f"{BYBIT_BASE_URL}/v5/market/tickers"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                resp = requests.get(url, params={"category": "linear"}, headers=headers, proxies=get_bybit_proxies(), timeout=8)
                found_symbols = set()
                if resp.status_code == 200:
                    data = resp.json()
                    ticker_list = data.get("result", {}).get("list", [])
                    for ticker in ticker_list:
                        sym = ticker.get("symbol")
                        if sym in SUPPORTED_SYMBOLS:
                            val_str = ticker.get("lastPrice")
                            if val_str:
                                val = float(val_str)
                                bot_state[f"live_price_{sym}"] = val
                                bot_state[f"live_price_ts_{sym}"] = now
                                found_symbols.add(sym)
                                if sym == "BTCUSDT":
                                    live_price = val
                                    bot_state["live_price"] = val
                                    bot_state["last_update"] = now
                                bot_state["last_rest_price_time"] = now

                # Try Bulk Bybit Live API fallback for missing symbols first (such as AVAX and LTC on testnet)
                missing = [s for s in SUPPORTED_SYMBOLS if s not in found_symbols]
                if missing and TRADE_MODE == "testnet":
                    try:
                        url = "https://api.bybit.com/v5/market/tickers"
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        }
                        bresp = requests.get(url, params={"category": "linear"}, headers=headers, proxies=get_bybit_proxies(), timeout=8)
                        if bresp.status_code == 200:
                            ticker_list = bresp.json().get("result", {}).get("list", [])
                            for ticker in ticker_list:
                                sym = ticker.get("symbol")
                                if sym in missing:
                                    val_str = ticker.get("lastPrice")
                                    if val_str:
                                        val = float(val_str)
                                        bot_state[f"live_price_{sym}"] = val
                                        bot_state[f"live_price_ts_{sym}"] = now
                                        found_symbols.add(sym)
                    except Exception as ble:
                        print(f"[Price Fallback] Bybit Live bulk fetch error: {ble}")

                # Bulk Binance fallback for missing symbols (throttled)
                missing = [s for s in SUPPORTED_SYMBOLS if s not in found_symbols]
                if missing and (now - last_binance_run >= binance_interval):
                    last_binance_run = now
                    try:
                        bresp = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=8)
                        if bresp.status_code == 200:
                            binance_prices = {t["symbol"]: float(t["price"]) for t in bresp.json()}
                            for sym in missing:
                                if sym in binance_prices:
                                    bot_state[f"live_price_{sym}"] = binance_prices[sym]
                                    bot_state[f"live_price_ts_{sym}"] = now
                                    if sym == "BTCUSDT":
                                        live_price = binance_prices[sym]
                                        bot_state["live_price"] = binance_prices[sym]
                                        bot_state["last_update"] = now
                                    bot_state["last_rest_price_time"] = now
                    except Exception as be:
                        print(f"[Price Fallback] Binance bulk fetch error: {be}")
        except Exception as e:
            print(f"[Price Fallback Exception] {e}")

        time.sleep(15)  # Gated sleep to conserve CPU cycles

def send_email_notification(subject, body):
    """
    Sends an email alert.
    Uses Resend HTTP API (port 443) if RESEND_API_KEY is configured.
    Otherwise, falls back to standard SMTP (port 587/465).
    """
    import requests
    import smtplib
    from email.mime.text import MIMEText
    
    email_to = os.getenv("EMAIL_TO", "mehsimleo@gmail.com")
    resend_api_key = os.getenv("RESEND_API_KEY", "")
    
    # 1. Attempt to send via Resend HTTPS API (firewall resistant)
    if resend_api_key:
        try:
            print("[Email Notification] Sending email via Resend HTTP API...")
            url = "https://api.resend.com/emails"
            headers = {
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json"
            }
            # Resend onboarding@resend.dev allows sending to your own email address (email_to)
            payload = {
                "from": "UBOTE Alerts <onboarding@resend.dev>",
                "to": email_to,
                "subject": subject,
                "html": body
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code in [200, 201, 202]:
                print(f"[Email Notification] Successfully sent email via Resend to {email_to}")
                return True
            else:
                print(f"[Email Notification] Resend API error ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"[Email Notification] Resend HTTP API failed: {e}")
            
    # 2. Fallback to SMTP
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        smtp_port = 587
        
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    
    if not smtp_user or not smtp_password:
        print("[Email Notification] Skipped SMTP fallback: SMTP_USER or SMTP_PASSWORD environment variables not set.")
        return False
        
    try:
        msg = MIMEText(body, "html")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = email_to
        
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            print(f"[Email Notification] Successfully sent email via SMTP to {email_to}")
            return True
    except Exception as e:
        print(f"[Email Notification] SMTP connection failed: {e}")
        return False

def on_message(ws, message):
    global live_price, last_ws_update_time
    last_ws_update_time = time.time()
    try:
        data = json.loads(message)
        if "op" in data or "success" in data:
            log_event("INFO", f"[WS] control message: {data}")
        topic = data.get("topic", "")
        
        # 1. Price Tickers Handler
        if topic.startswith("tickers."):
            ticker_data = data.get("data", {})
            sym = ticker_data.get("symbol")
            price_str = ticker_data.get("lastPrice")
            if price_str and sym:
                val = float(price_str)
                bot_state[f"live_price_{sym}"] = val
                bot_state[f"live_price_ts_{sym}"] = last_ws_update_time
                if sym == "BTCUSDT":
                    live_price = val
                    bot_state["live_price"] = val
                    bot_state["last_update"] = last_ws_update_time
                    
        # 2. Public Trade (CVD) Handler
        elif topic.startswith("publicTrade."):
            trade_list = data.get("data", [])
            for t in trade_list:
                sym = t.get("s")
                side = t.get("S")
                qty = float(t.get("v", 0.0))
                if sym:
                    with order_flow_lock:
                        if sym not in order_flow_data:
                            order_flow_data[sym] = {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0}
                        order_flow_data[sym]["cvd"] += qty if side == "Buy" else -qty
                        
        # 3. Order Book L2 (OFI & Depth Cache) Handler
        elif topic.startswith("orderbook.50."):
            ob_data = data.get("data", {})
            sym = ob_data.get("s")
            bids = ob_data.get("b", [])
            asks = ob_data.get("a", [])
            
            if sym:
                with order_flow_lock:
                    if sym not in order_flow_data:
                        order_flow_data[sym] = {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0, "last_ob_ts": 0.0}
                    
                    state = order_flow_data[sym]
                    state["last_ob_ts"] = time.time()
                    is_snapshot = (data.get("type") == "snapshot")
                    if is_snapshot:
                        state["latest_bids"] = bids[:25]
                        state["latest_asks"] = asks[:25]
                    else:
                        if bids:
                            cached_b = {float(p): float(s) for p, s in state["latest_bids"]}
                            for p, s in bids:
                                price, size = float(p), float(s)
                                if size == 0.0:
                                    cached_b.pop(price, None)
                                else:
                                    cached_b[price] = size
                            state["latest_bids"] = sorted([[str(p), str(s)] for p, s in cached_b.items()], key=lambda x: float(x[0]), reverse=True)[:25]
                        if asks:
                            cached_a = {float(p): float(s) for p, s in state["latest_asks"]}
                            for p, s in asks:
                                price, size = float(p), float(s)
                                if size == 0.0:
                                    cached_a.pop(price, None)
                                else:
                                    cached_a[price] = size
                            state["latest_asks"] = sorted([[str(p), str(s)] for p, s in cached_a.items()], key=lambda x: float(x[0]))[:25]

                    bids_cache = state["latest_bids"]
                    asks_cache = state["latest_asks"]
                    
                    # Compute actual L2 imbalance and spread
                    if bids_cache and asks_cache:
                        bid_L1 = float(bids_cache[0][0])
                        ask_L1 = float(asks_cache[0][0])
                        state["ob_spread_L2"] = (ask_L1 - bid_L1) / bid_L1 if bid_L1 > 0 else 0.0
                        
                        top_bids_size = sum(float(b[1]) for b in bids_cache[:10])
                        top_asks_size = sum(float(a[1]) for a in asks_cache[:10])
                        tot_size = top_bids_size + top_asks_size + 1e-8
                        state["ob_imbalance_L2"] = (top_bids_size - top_asks_size) / tot_size

                    # Compute L1 OFI
                    if bids_cache:
                        bid_p = float(bids_cache[0][0])
                        bid_q = float(bids_cache[0][1])
                        if bid_p > state["prev_bid_price"]:
                            db = bid_q
                        elif bid_p == state["prev_bid_price"]:
                            db = bid_q - state["prev_bid_size"]
                        else:
                            db = 0.0
                        state["prev_bid_price"] = bid_p
                        state["prev_bid_size"] = bid_q
                    else:
                        db = 0.0
                        
                    if asks_cache:
                        ask_p = float(asks_cache[0][0])
                        ask_q = float(asks_cache[0][1])
                        if ask_p > state["prev_ask_price"]:
                            da = 0.0
                        elif ask_p == state["prev_ask_price"]:
                            da = ask_q - state["prev_ask_size"]
                        else:
                            da = ask_q
                        state["prev_ask_price"] = ask_p
                        state["prev_ask_size"] = ask_q
                    else:
                        da = 0.0
                        
                    state["ofi"] += (db - da)

        # 4. Public Liquidation Feed Handler
        elif topic.startswith("liquidation."):
            liq_data = data.get("data", {})
            sym = liq_data.get("symbol")
            side = liq_data.get("side") # Buy/Sell
            qty = float(liq_data.get("size", 0.0))
            price = float(liq_data.get("price", 0.0))
            usd_val = qty * price
            if sym and usd_val > 0.0:
                with order_flow_lock:
                    if sym not in order_flow_data:
                        order_flow_data[sym] = {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0}
                    state = order_flow_data[sym]
                    if side == "Buy": # Short liquidation (market buy filled)
                        state["liq_short_1h"] += usd_val
                    else: # Long liquidation (market sell filled)
                        state["liq_long_1h"] += usd_val
    except Exception as e:
        print(f"[WebSocket msg exception] {e}")

def on_open(ws):
    global ws_connected, ws_retry_delay, active_public_ws, last_ws_update_time
    ws_connected = True
    active_public_ws = ws
    last_ws_update_time = time.time()
    ws_retry_delay = 3  # Reset backoff on successful connection
    print("Connected to Bybit WebSocket for multi-asset prices and order flow")
    
    # Subscribe to tickers, publicTrade, orderbook, and liquidation topics for all supported symbols
    args = []
    for s in SUPPORTED_SYMBOLS:
        args.append(f"tickers.{s}")
        args.append(f"publicTrade.{s}")
        args.append(f"orderbook.50.{s}")
        
    chunk_size = 10
    for i in range(0, len(args), chunk_size):
        chunk = args[i:i + chunk_size]
        ws.send(json.dumps({
            "op": "subscribe",
            "args": chunk
        }))
        time.sleep(0.2)
        
    # Heartbeat Daemon Thread to send custom text pings every 20 seconds
    def send_heartbeat():
        while ws_connected:
            try:
                ws.send(json.dumps({"op": "ping"}))
            except Exception:
                break
            time.sleep(20)
    threading.Thread(target=send_heartbeat, daemon=True).start()

def on_close(ws, close_status_code, close_msg):
    global ws_connected, active_public_ws
    ws_connected = False
    active_public_ws = None
    print(f"[WebSocket Closed] code={close_status_code}, msg={close_msg}")

def on_error(ws, error):
    print(f"[WebSocket Error] {error}")

def start_ws():
    global ws_connected, ws_retry_delay, active_public_ws
    url = BYBIT_WS_URL
    print(f"[WebSocket Connecting] url={url}")
    # Parse proxy settings from BYBIT_PROXY env var
    proxy_host, proxy_port, proxy_auth, proxy_type_str = None, None, None, None
    proxy_url = os.environ.get("BYBIT_PROXY")
    if proxy_url:
        proxy_host, proxy_port, proxy_auth, proxy_type_str = parse_proxy_url(proxy_url)
        print(f"[WebSocket] Using proxy: {proxy_host}:{proxy_port} (type={proxy_type_str})")
    while True:
        try:
            ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            active_public_ws = ws
            import ssl
            ssl_opts = {"cert_reqs": ssl.CERT_REQUIRED}
            try:
                import certifi
                ssl_opts["ca_certs"] = certifi.where()
            except Exception:
                pass
            ws.run_forever(
                ping_interval=0,
                http_proxy_host=proxy_host,
                http_proxy_port=proxy_port,
                http_proxy_auth=proxy_auth,
                proxy_type=proxy_type_str,
                sslopt=ssl_opts
            )

        except Exception as e:
            print(f"[WebSocket run_forever exception] {e}")
        ws_connected = False
        active_public_ws = None
        print(f"[WebSocket] Reconnecting in {ws_retry_delay}s...")
        time.sleep(ws_retry_delay)
        ws_retry_delay = min(ws_retry_delay * 2, 60)  # Backoff up to 60s

# WebSocket thread is started inside if __name__ == "__main__" block at the bottom

def on_private_open(ws):
    global private_ws_connected, active_private_ws, last_private_ws_update_time
    private_ws_connected = True
    active_private_ws = ws
    last_private_ws_update_time = time.time()
    print("[WebSocket Private] Connected. Authenticating...")
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print("[WebSocket Private] API Key or Secret missing. Cannot authenticate.")
        ws.close()
        return
    import time as t_module
    import hmac
    import hashlib
    import json
    expires = int((t_module.time() + 10) * 1000)
    signature_raw = f"GET/realtime{expires}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        signature_raw.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    auth_payload = {
        "op": "auth",
        "args": [api_key, expires, signature]
    }
    ws.send(json.dumps(auth_payload))
    
    # Heartbeat Daemon Thread for private WebSocket to send custom text pings every 20 seconds
    def send_private_heartbeat():
        import json as j_module
        import time as t_module
        while private_ws_connected:
            try:
                ws.send(j_module.dumps({"op": "ping"}))
            except Exception:
                break
            t_module.sleep(20)
    threading.Thread(target=send_private_heartbeat, daemon=True).start()

_ws_filled_orders = {}
_ws_filled_orders_lock = threading.Lock()
_last_ws_pos_sync_time = 0.0

_pos_sync_event = threading.Event()
_pos_sync_worker_started = False
_pos_sync_worker_lock = threading.Lock()

def _position_sync_worker_loop():
    while True:
        try:
            _pos_sync_event.wait()
            _pos_sync_event.clear()
            sync_active_positions_from_bybit()
            time.sleep(1.0)  # Rate-limit between consecutive background syncs
        except Exception as e:
            log_event("WARNING", f"[Position Sync Worker Error] {e}")
            time.sleep(2.0)

def start_position_sync_worker():
    global _pos_sync_worker_started
    with _pos_sync_worker_lock:
        if not _pos_sync_worker_started:
            _pos_sync_worker_started = True
            t = threading.Thread(target=_position_sync_worker_loop, daemon=True, name="PositionSyncWorker")
            t.start()

def request_position_sync():
    """Trigger debounced position sync via dedicated background worker."""
    start_position_sync_worker()
    _pos_sync_event.set()

def on_private_message(ws, message):
    import json
    import time
    global last_private_ws_update_time, _last_ws_pos_sync_time
    last_private_ws_update_time = time.time()
    try:
        data = json.loads(message)
        op = data.get("op")
        topic = data.get("topic")
        
        if op == "auth":
            if data.get("success") is True:
                print("[WebSocket Private] Authentication successful. Subscribing to topics...")
                sub_payload = {
                    "op": "subscribe",
                    "args": ["position", "wallet", "order"]
                }
                ws.send(json.dumps(sub_payload))
            else:
                print(f"[WebSocket Private] Authentication failed: {data.get('ret_msg')}")
                ws.close()
        elif topic == "wallet":
            wallet_data = data.get("data", [])
            if wallet_data:
                # Capture venue-reported margin so the wallet-utilisation ceiling has an
                # authoritative source. Reconstructing used margin from local trade records
                # misses any position opened outside the bot, so read it from the venue.
                try:
                    _acct = wallet_data[0]
                    _te = float(_acct.get("totalEquity") or _acct.get("totalWalletBalance") or 0.0)
                    _im = float(_acct.get("totalInitialMargin") or 0.0)
                    if _te > 0.0:
                        bot_state["wallet_margin_info"] = {
                            "total_equity": _te,
                            "used_margin": _im,
                            "ts": time.time(),
                        }
                except (TypeError, ValueError) as _mi_err:
                    # Leave the previous snapshot in place; staleness is enforced downstream by
                    # evaluate_pre_trade_checklist, which rejects rather than skipping the check.
                    log_event("WARNING", f"[Wallet Margin] Could not parse venue margin fields: {_mi_err}")

                total_equity = wallet_data[0].get("totalEquity") or wallet_data[0].get("totalWalletBalance")
                if total_equity:
                    val = float(total_equity)
                    with _balance_lock:
                        global _cached_balance, _last_balance_fetch
                        _cached_balance = val
                        _last_balance_fetch = time.time()
                    if TRADE_MODE != "simulation":
                        bot_state["simulated_balance"] = val
                    print(f"[WebSocket Private] Balance updated dynamically from wallet stream: {val}")
        elif topic in ("order", "position"):
            if topic == "order":
                order_list = data.get("data", [])
                for ord in order_list:
                    status = ord.get("orderStatus")
                    ord_id = ord.get("orderId")
                    if ord_id and status == "Filled":
                        with _ws_filled_orders_lock:
                            _ws_filled_orders[ord_id] = ord
                            if len(_ws_filled_orders) > 500:
                                _ws_filled_orders.clear()
            
            # Debounce position sync request to dedicated background worker
            now_t = time.time()
            if now_t - _last_ws_pos_sync_time >= 1.0:
                _last_ws_pos_sync_time = now_t
                request_position_sync()
    except Exception as e:
        print(f"[WebSocket Private Message Error] {e}")

def on_private_error(ws, error):
    print(f"[WebSocket Private Error] {error}")

def on_private_close(ws, close_status_code, close_msg):
    global private_ws_connected, active_private_ws
    private_ws_connected = False
    active_private_ws = None
    print(f"[WebSocket Private Closed] code={close_status_code}, msg={close_msg}")

def start_private_ws():
    global private_ws_connected, private_ws_retry_delay, active_private_ws
    url = BYBIT_PRIVATE_WS_URL
    print(f"[WebSocket Private Connecting] url={url}")
    proxy_host, proxy_port, proxy_auth, proxy_type_str = None, None, None, None
    proxy_url = os.environ.get("BYBIT_PROXY")
    if proxy_url:
        proxy_host, proxy_port, proxy_auth, proxy_type_str = parse_proxy_url(proxy_url)
    while True:
        try:
            ws = websocket.WebSocketApp(
                url,
                on_open=on_private_open,
                on_message=on_private_message,
                on_error=on_private_error,
                on_close=on_private_close
            )
            active_private_ws = ws
            import ssl
            ssl_opts_priv = {"cert_reqs": ssl.CERT_REQUIRED}
            try:
                import certifi
                ssl_opts_priv["ca_certs"] = certifi.where()
            except Exception:
                pass
            ws.run_forever(
                ping_interval=0,
                http_proxy_host=proxy_host,
                http_proxy_port=proxy_port,
                http_proxy_auth=proxy_auth,
                proxy_type=proxy_type_str,
                sslopt=ssl_opts_priv
            )

        except Exception as e:
            print(f"[WebSocket Private run_forever exception] {e}")
        private_ws_connected = False
        active_private_ws = None
        print(f"[WebSocket Private] Reconnecting in {private_ws_retry_delay}s...")
        time.sleep(private_ws_retry_delay)
        private_ws_retry_delay = min(private_ws_retry_delay * 2, 60)

def run_websocket_watchdog():
    global last_ws_update_time, last_private_ws_update_time
    global active_public_ws, active_private_ws
    global ws_connected, private_ws_connected
    
    print("[WebSocket Watchdog] Active keep-alive thread started.")
    last_ws_update_time = time.time()
    last_private_ws_update_time = time.time()
    
    while True:
        time.sleep(15)
        now = time.time()
        
        # Check Public WebSocket (Ticker/Prices)
        if ws_connected and active_public_ws:
            silent_duration = now - last_ws_update_time
            if silent_duration > 60:
                print(f"[WebSocket Watchdog] Public WebSocket silent for {silent_duration:.1f}s (>60s). Force closing to trigger reconnect...")
                try:
                    active_public_ws.close()
                except Exception as e:
                    print(f"[WebSocket Watchdog] Error closing public ws: {e}")
                    
        # Check Private WebSocket (Position/Orders/Wallet)
        if private_ws_connected and active_private_ws:
            silent_duration = now - last_private_ws_update_time
            if silent_duration > 60:
                print(f"[WebSocket Watchdog] Private WebSocket silent for {silent_duration:.1f}s (>60s). Force closing to trigger reconnect...")
                try:
                    active_private_ws.close()
                except Exception as e:
                    print(f"[WebSocket Watchdog] Error closing private ws: {e}")


# =========================
# FEATURE ENGINE
# =========================
features = [
    "RSI", "MACD_diff", "MFI", "ATR_norm",
    "close_to_EMA9", "close_to_EMA21", "close_to_EMA50", "close_to_EMA200", "EMA9_to_EMA21", 
    "BB_pct", "BB_width", "return_5m", "volatility_10m", "volume_ratio",
    "high_low_ratio", "open_close_ratio", "RSI_diff", "MACD_diff_diff", "ROC_5", "ROC_10",
    "ADX", "ADX_pos", "ADX_neg", "close_to_VWAP",
    "btc_return_5m", "btc_return_5m_lag1", "btc_return_5m_lag2", "btc_return_5m_lag3",
    "RSI_24", "ROC_24", "volatility_24h",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "RSI_z", "ADX_z", "close_to_Kalman"
]
for lag in [1, 2, 3, 4, 5]:
    features.append(f"return_5m_lag{lag}")
for lag in [1, 2, 3]:
    features.append(f"volume_ratio_lag{lag}")
for lag in [1, 2]:
    features.append(f"RSI_lag{lag}")
    features.append(f"MACD_diff_lag{lag}")
    features.append(f"BB_pct_lag{lag}")

# New features from Option 2
features.extend(["open_interest", "funding_rate", "fear_greed"])
for lag in [1, 2]:
    features.append(f"open_interest_lag{lag}")
    features.append(f"funding_rate_lag{lag}")
    features.append(f"fear_greed_lag{lag}")

# New microstructure and derivatives momentum features
features.extend([
    "open_interest_pct_change", "funding_rate_diff", 
    "CVD_true", "OFI_true"
])
for lag in [1, 2]:
    features.append(f"open_interest_pct_change_lag{lag}")
    features.append(f"funding_rate_diff_lag{lag}")
    features.append(f"CVD_true_lag{lag}")
    features.append(f"OFI_true_lag{lag}")

# New Wick Volume features (absorption/liquidation proxies)
features.extend([
    "upper_wick_volume_ratio", "lower_wick_volume_ratio"
])
for lag in [1, 2]:
    features.append(f"upper_wick_volume_ratio_lag{lag}")
    features.append(f"lower_wick_volume_ratio_lag{lag}")

features.append("hours_to_news")

# New Correlation and OI momentum features
features.extend(["oi_change_1h", "oi_change_4h", "btc_close", "btc_volume", "btc_rsi"])
for lag in [1, 2]:
    features.append(f"oi_change_1h_lag{lag}")
    features.append(f"oi_change_4h_lag{lag}")
    features.append(f"btc_close_lag{lag}")
    features.append(f"btc_volume_lag{lag}")
    features.append(f"btc_rsi_lag{lag}")

# Advanced Microstructure features
features.extend([
    "roll_spread", "leverage_divergence", "oi_velocity", "funding_acceleration", "bid_ask_imbalance_ohlc",
    "ob_imbalance_L2", "ob_spread_L2", "liq_long_1h", "liq_short_1h"
])
for lag in [1, 2]:
    features.append(f"roll_spread_lag{lag}")
    features.append(f"leverage_divergence_lag{lag}")
    features.append(f"oi_velocity_lag{lag}")
    features.append(f"funding_acceleration_lag{lag}")
    features.append(f"bid_ask_imbalance_ohlc_lag{lag}")
    features.append(f"close_to_Kalman_lag{lag}")
    features.append(f"ob_imbalance_L2_lag{lag}")
    features.append(f"ob_spread_L2_lag{lag}")
    features.append(f"liq_long_1h_lag{lag}")
    features.append(f"liq_short_1h_lag{lag}")

# Garman-Klass Volatility features
features.extend(["volatility_gk", "volatility_gk_lag1", "volatility_gk_lag2"])

# Cross-Asset Lead-Lag Correlation features
features.extend(["lead_lag_diff_5m", "lead_lag_diff_1h", "lead_lag_diff_4h", "volume_ratio_to_btc"])
for lag in [1, 2]:
    features.append(f"lead_lag_diff_5m_lag{lag}")
    features.append(f"lead_lag_diff_1h_lag{lag}")
    features.append(f"lead_lag_diff_4h_lag{lag}")
    features.append(f"volume_ratio_to_btc_lag{lag}")

# Initial model loading is deferred to main() to ensure Flask starts immediately on HF Spaces.


def add_features(df, symbol=None, interval=None):
    if df is not None:
        if symbol:
            df.attrs["symbol"] = symbol
            if "symbol" not in df.columns:
                df["symbol"] = symbol
        if interval:
            df.attrs["interval"] = str(interval)
            if "interval" not in df.columns:
                df["interval"] = str(interval)
    return features_module.add_features(df, fetch_calendar_callback=fetch_economic_calendar_cached, symbol=symbol, interval=interval)

def build_df(current_price):
    try:
        # Fetch target coin data (limit 300 to satisfy EMA 200)
        df_target = get_history(symbol=SYMBOL, interval=INTERVAL, limit=300)
        if df_target is not None and len(df_target) > 0:
            df_target.loc[df_target.index[-1], "close"] = current_price
            
            if SYMBOL == "BTCUSDT":
                df = df_target.copy()
                df["close_btc"] = df["close"]
            else:
                # Fetch BTCUSDT data
                df_btc = get_history(symbol="BTCUSDT", interval=INTERVAL, limit=300)
                if df_btc is not None and len(df_btc) > 0:
                    df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                    # Merge target coin and BTCUSDT on timestamp
                    df = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
                else:
                    return None
            
            if len(df) > 0:
                df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=INTERVAL)
                df = add_features(df, symbol=SYMBOL, interval=INTERVAL)
                return df
    except Exception as e:
        print(f"Error fetching candle data: {e}")
    return None

def get_local_time_str(t):
    # Pakistan timezone is UTC + 5 hours (18000 seconds)
    return datetime.fromtimestamp(t + 18000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def evaluate_predictions(df_completed, interval, symbol):
    if not bot_state["prediction_history"]:
        return

    # Create a map of timestamp to close price for quick lookup
    ts_map = {}
    for _, row in df_completed.iterrows():
        ts_map[int(row["timestamp"])] = float(row["close"])

    for pred in bot_state["prediction_history"]:
        eval_dict = pred.get("evaluation")
        if eval_dict is None:
            eval_dict = {
                "evaluated": False,
                "exit_price": None,
                "change": None,
                "change_pct": None,
                "success": None
            }
            pred["evaluation"] = eval_dict
            bot_state.save_prediction(pred)
            
        if pred.get("interval") == interval and pred.get("symbol", "BTCUSDT") == symbol and not eval_dict.get("evaluated"):
            interval_mins = int(interval)
            cfg = TIMEFRAME_CONFIG.get(str(interval), {"lookahead": 10})
            lookahead = cfg.get("lookahead", 10)
            target_ts = int(pred["candle_timestamp"]) + (interval_mins * 60 * 1000 * lookahead)
            if target_ts in ts_map:
                exit_price = ts_map[target_ts]
                ref_price = pred["ref_price"]
                change = exit_price - ref_price
                change_pct = (change / ref_price) * 100
                direction = pred["direction"]
                
                # Check success
                success = (change > 0 and direction == "Bullish") or (change < 0 and direction == "Bearish")
                
                pred["evaluation"] = {
                    "evaluated": True,
                    "exit_price": float(exit_price),
                    "change": float(change),
                    "change_pct": float(change_pct),
                    "success": bool(success)
                }
                bot_state.save_prediction(pred)
                
                # Print to log
                success_str = "SUCCESSFUL" if success else "UNSUCCESSFUL"
                print(f"[Prediction Tracker] Evaluated {interval}m Prediction from {get_local_time_str(pred['candle_timestamp']/1000)}: Direction: {direction} | Ref Price: {ref_price:.2f} | Exit Price: {exit_price:.2f} | Change: {change:+.2f} ({change_pct:+.3f}%) | Result: {success_str} | Status: {pred['status']}")

# =========================
# NEWS & SOCIAL SENTIMENT ANALYSIS
# =========================
sentiment_pipeline = None

from news_monitor import safe_parse_xml


def get_reddit_posts():
    """
    Fetches the top crypto/bitcoin post titles from Reddit RSS feeds.
    Does not require API keys, but does require a descriptive User-Agent.
    """
    subreddits = ["CryptoCurrency", "Bitcoin"]
    posts = []
    # Using the recommended Reddit API user agent format to prevent blocks
    headers = {"User-Agent": "btc-trading-bot:v1.0.0 (by /u/btc-trading-bot-user)"}
    
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot/.rss"
        try:
            res = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=10)
            if res.status_code == 200:
                xml_content = res.content.decode("utf-8")
                # Parse the Atom XML feed safely
                root = safe_parse_xml(xml_content)
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                sub_posts = []
                for entry in root.findall("atom:entry", namespace):
                    title_elem = entry.find("atom:title", namespace)
                    if title_elem is not None and title_elem.text:
                        sub_posts.append(title_elem.text.strip())
                # Limit to top 5 posts per subreddit to avoid skewing sentiment
                posts.extend(sub_posts[:5])
                print(f"[News/Sentiment] Fetched {len(sub_posts[:5])} posts from r/{sub} RSS.")
            else:
                if res.status_code != 429:
                    print(f"[News/Sentiment] Reddit r/{sub} feed returned status code {res.status_code}")
        except Exception as e:
            print(f"[News/Sentiment] Exception fetching Reddit r/{sub} feed: {e}")
    return posts

def get_cryptopanic_posts():
    """
    Fetches the top live crypto news from RSS feeds of Cointelegraph, CoinDesk, and Decrypt.
    Replaces the discontinued CryptoPanic developer API. No API key is needed.
    """
    feeds = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        "https://decrypt.co/feed"
    ]
    posts = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    for url in feeds:
        try:
            res = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=8)
            if res.status_code == 200:
                xml_content = res.content
                root = safe_parse_xml(xml_content)
                feed_posts = []
                for item in root.findall(".//item"):
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        feed_posts.append(title_elem.text.strip())
                posts.extend(feed_posts[:4])
                print(f"[News/Sentiment] Fetched {len(feed_posts[:4])} articles from RSS: {url}")
            else:
                print(f"[News/Sentiment] RSS feed {url} returned status code {res.status_code}")
        except Exception as e:
            print(f"[News/Sentiment] Exception fetching RSS feed {url}: {e}")
            
    return posts[:12]

def get_x_tweets():
    """
    Fetches recent tweets matching crypto/bitcoin search query.
    Requires an X Developer Bearer Token in .env (Basic or Pro subscription).
    """
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        return []
    
    query = os.environ.get("X_SEARCH_QUERY", "Bitcoin OR BTC lang:en -is:retweet")
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "v2RecentSearchPython"
    }
    params = {
        "query": query,
        "max_results": 10,
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            tweets = []
            for t in data.get("data", []):
                text = t.get("text")
                if text:
                    tweets.append(text.strip())
            print(f"[News/Sentiment] Fetched {len(tweets)} tweets from X API.")
            return tweets
        else:
            print(f"[News/Sentiment] X API returned status code {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[News/Sentiment] Exception fetching X tweets: {e}")
    return []

def get_news_sentiment():
    global sentiment_pipeline
    titles = []

    # 1. Fetch from Cointelegraph RSS (standard news)
    url = "https://cointelegraph.com/rss"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=10)
        if res.status_code == 200:
            xml_content = res.content.decode("utf-8")
            root = safe_parse_xml(xml_content)
            rss_titles = []
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                if title_elem is not None and title_elem.text:
                    rss_titles.append(title_elem.text.strip())
            titles.extend(rss_titles[:10])
            print(f"[News/Sentiment] Fetched {len(rss_titles[:10])} articles from Cointelegraph RSS.")
    except Exception as e:
        print(f"[News/Sentiment] Error fetching Cointelegraph RSS: {e}")

    # 1b. Fetch from CoinDesk RSS
    url_coindesk = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
    try:
        res = requests.get(url_coindesk, headers=headers, proxies=get_bybit_proxies(), timeout=10)
        if res.status_code == 200:
            xml_content = res.content.decode("utf-8")
            root = safe_parse_xml(xml_content)
            coindesk_titles = []
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                if title_elem is not None and title_elem.text:
                    coindesk_titles.append(title_elem.text.strip())
            coindesk_titles = []
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                if title_elem is not None and title_elem.text:
                    coindesk_titles.append(title_elem.text.strip())
            titles.extend(coindesk_titles[:10])
            print(f"[News/Sentiment] Fetched {len(coindesk_titles[:10])} articles from CoinDesk RSS.")
    except Exception as e:
        print(f"[News/Sentiment] Error fetching CoinDesk RSS: {e}")

    # 2. Fetch from Reddit RSS (free social sentiment fallback)
    reddit_posts = get_reddit_posts()
    titles.extend(reddit_posts)

    # 3. Fetch from CryptoPanic (optional aggregated news/social)
    cryptopanic_posts = get_cryptopanic_posts()
    titles.extend(cryptopanic_posts)

    # 4. Fetch from X / Twitter (optional premium social sentiment)
    x_tweets = get_x_tweets()
    titles.extend(x_tweets)

    # Clean up empty or duplicate titles
    seen = set()
    cleaned_titles = []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            cleaned_titles.append(t)
            
    # Limit combined sources to top 25 to avoid overloaded HuggingFace pipeline times
    cleaned_titles = cleaned_titles[:25]

    if not cleaned_titles:
        print("[News/Sentiment] No content found across any source. Sentiment defaults to Neutral.")
        return "Neutral", [], "NO_NEWS"

    try:
        token = os.environ.get("HF_TOKEN") or os.environ.get("token")
        if token:
            API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
            headers = {"Authorization": f"Bearer {token}"}
            
            total_score = 0.0
            processed = 0
            print(f"[News/Sentiment Serverless] Querying HF Inference API for {len(cleaned_titles)} inputs...")
            for text in cleaned_titles:
                try:
                    resp = requests.post(API_URL, headers=headers, json={"inputs": text[:300]}, timeout=5)
                    if resp.status_code == 200:
                        res = resp.json()
                        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                            scores = {item["label"].lower(): item["score"] for item in res[0]}
                            bullish = scores.get("positive", 0.0)
                            bearish = scores.get("negative", 0.0)
                            total_score += (bullish - bearish)
                            processed += 1
                except Exception as e:
                    pass
            
            if processed > 0:
                avg_score = total_score / processed
                sentiment = "Neutral"
                if avg_score > 0.05:
                    sentiment = "Bullish"
                elif avg_score < -0.05:
                    sentiment = "Bearish"
                print(f"[News/Sentiment Serverless] Analysis complete. Avg Score: {avg_score:.4f} | Aggregated: {sentiment}")
                return sentiment, cleaned_titles, "FINBERT_SERVERLESS"

        # Local pipeline fallback if HF_TOKEN is missing
        if sentiment_pipeline is None:
            try:
                from transformers import pipeline
                sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
            except ImportError:
                # Comprehensive expanded financial & crypto lexicon
                bullish_keywords = {
                    # Market Price Action
                    "bullish", "surge", "surges", "surging", "rally", "rallies", "rallying", "breakout", "skyrocket", "skyrockets", "skyrocketing",
                    "gain", "gains", "gaining", "all-time high", "ath", "soar", "soars", "soaring", "pump", "pumping", "pumps",
                    "rebound", "rebounds", "rebounding", "recovery", "recovers", "recovering", "uptrend", "outperform", "outperforms",
                    "bounce", "bouncing", "climb", "climbs", "climbing", "record high", "milestone", "moon", "soaring",
                    "bull run", "historic rally", "breaks above", "tops", "new high", "break above",
                    # Flows & Institutional
                    "inflow", "inflows", "accumulation", "accumulating", "accumulate", "adoption", "approval", "approved", "approves",
                    "institutional", "profit", "profits", "profitable", "upward", "bull", "bulls", "optimistic", "growth", "expanding",
                    "buy", "buying", "bought", "reserve", "treasury", "sec approval", "etf approval", "partnership", "mainnet", "upgrade",
                    "stimulus", "rate cut", "rate cuts", "dovish", "easing", "support level", "holder", "holders", "record inflow"
                }
                bearish_keywords = {
                    # Market Price Action & Losses
                    "bearish", "crash", "crashes", "crashing", "dump", "dumps", "dumping", "plunge", "plunges", "plunging",
                    "drop", "drops", "dropping", "fall", "falls", "falling", "slide", "slides", "sliding", "tumble", "tumbles", "tumbling",
                    "collapse", "collapses", "collapsing", "selloff", "sell-off", "selloffs", "downtrend", "slump", "slumps", "retreat",
                    "retreats", "bleeding", "capitulation", "correction", "wilt", "wilts", "wilted", "slips", "drops back", "stalled",
                    # Risk, Hacks & Failures
                    "hack", "hacked", "hacks", "exploit", "exploited", "exploits", "stolen", "theft", "drain", "drained", "scam", "rugpull",
                    "fraud", "bankruptcy", "bankrupt", "bankruptcies", "insolvent", "insolvency", "liquidation", "liquidations", "liquidated",
                    "outflow", "outflows", "loss", "losses", "bear", "bears", "pessimistic", "panic", "sell", "selling", "sold",
                    "cut headcount", "layoff", "layoffs", "cuts 1", "cuts 2", "job cuts", "halt", "halts",
                    # Regulatory & Macro Hardship
                    "ban", "banned", "banning", "bans", "lawsuit", "lawsuits", "sued", "suing", "sec", "crackdown", "probe", "investigation",
                    "subpoena", "fine", "fined", "penalty", "rate hike", "rate hikes", "hawkish", "inflation", "recession", "war", "restriction",
                    "anti-crypto"
                }
                
                total_score = 0.0
                for text in cleaned_titles:
                    text_lower = text.lower()
                    b_count = sum(1 for kw in bullish_keywords if kw in text_lower)
                    r_count = sum(1 for kw in bearish_keywords if kw in text_lower)
                    if b_count + r_count > 0:
                        score = (b_count - r_count) / (b_count + r_count)
                    else:
                        score = 0.0
                    total_score += score
                    
                avg_score = total_score / len(cleaned_titles)
                sentiment = "Neutral"
                if avg_score > 0.05:
                    sentiment = "Bullish"
                elif avg_score < -0.05:
                    sentiment = "Bearish"
                print(f"[News/Sentiment Lexicon Local] Analyzed {len(cleaned_titles)} titles via local financial lexicon. Avg Score: {avg_score:.4f} | Aggregated: {sentiment}")
                return sentiment, cleaned_titles, "KEYWORD"

        print(f"[News/Sentiment Local] Running local FinBERT pipeline on {len(cleaned_titles)} inputs...")
        results = sentiment_pipeline(cleaned_titles)
        
        total_score = 0.0
        for r in results:
            label = r["label"].lower()
            score = float(r["score"])
            if label == "positive":
                total_score += score
            elif label == "negative":
                total_score -= score
                
        avg_score = total_score / len(cleaned_titles)
        
        sentiment = "Neutral"
        if avg_score > 0.05:
            sentiment = "Bullish"
        elif avg_score < -0.05:
            sentiment = "Bearish"
            
        print(f"[News/Sentiment Local] Analysis complete. Avg Score: {avg_score:.4f} | Aggregated: {sentiment}")
        return sentiment, cleaned_titles, "FINBERT_LOCAL"
    except Exception as e:
        print(f"[News/Sentiment] Error executing FinBERT analysis: {e}")
    return "Neutral", [], "ERROR_FALLBACK"

def run_news_sentiment_updater():
    global cached_news_sentiment, cached_news_titles, cached_news_source
    print("[News/Sentiment] Background updater thread started.")
    try:
        sentiment, titles, source = get_news_sentiment()
        with news_sentiment_lock:
            cached_news_sentiment = sentiment
            cached_news_titles = titles
            cached_news_source = source
        print(f"[News/Sentiment] Startup background update success: {sentiment} ({source}, based on {len(titles)} inputs).")
    except Exception as e:
        print(f"[News/Sentiment] Startup background update error: {e}")
        
    while True:
        time.sleep(15 * 60)
        try:
            print("[News/Sentiment] Triggering periodic background news sentiment update...")
            sentiment, titles, source = get_news_sentiment()
            with news_sentiment_lock:
                cached_news_sentiment = sentiment
                cached_news_titles = titles
                cached_news_source = source
            print(f"[News/Sentiment] Background update success: {sentiment} ({source}, based on {len(titles)} inputs).")
        except Exception as e:
            print(f"[News/Sentiment] Error in background news sentiment update: {e}")

def fetch_economic_calendar_cached(start_ts_ms=None, end_ts_ms=None):
    global economic_calendar_cache, last_calendar_fetch
    now_ts = time.time()
    with economic_calendar_lock:
        if economic_calendar_cache is not None and (now_ts - last_calendar_fetch < 86400):
            return economic_calendar_cache
            
        try:
            finnhub_token = os.environ.get("FINNHUB_TOKEN", "free")
            from datetime import datetime, timedelta
            now = datetime.now(timezone.utc)
            if start_ts_ms:
                from_dt = datetime.fromtimestamp(start_ts_ms / 1000.0, tz=timezone.utc)
            else:
                from_dt = now - timedelta(days=60)
                
            if end_ts_ms:
                to_dt = datetime.fromtimestamp(end_ts_ms / 1000.0, tz=timezone.utc) + timedelta(days=2)
            else:
                to_dt = now + timedelta(days=7)
                
            from_str = from_dt.strftime("%Y-%m-%d")
            to_str = to_dt.strftime("%Y-%m-%d")
            
            print(f"[News/Sentiment] Fetching economic calendar from {from_str} to {to_str}...")
            session = get_shared_session()
            resp = session.get(
                "https://finnhub.io/api/v1/calendar/economic",
                params={"token": finnhub_token, "from": from_str, "to": to_str},
                timeout=10
            )
            if resp.status_code == 200:
                events = resp.json().get("economicCalendar", [])
                high_impact = ["CPI", "FOMC", "NFP", "Non-Farm", "Federal Reserve", "Interest Rate"]
                filtered_events = []
                for ev in events:
                    if any(kw.lower() in ev.get("event", "").lower() for kw in high_impact):
                        ev_time_str = ev.get("time", "")
                        try:
                            ev_time = datetime.strptime(ev_time_str, "%Y-%m-%d %H:%M:%S")
                            filtered_events.append(ev_time)
                        except Exception:
                            pass
                economic_calendar_cache = sorted(filtered_events)
                last_calendar_fetch = now_ts
                print(f"[News/Sentiment] Cached {len(economic_calendar_cache)} high-impact calendar events.")
                return economic_calendar_cache
            else:
                print(f"[News/Sentiment Warning] Finnhub returned status {resp.status_code}. Caching fallback.")
        except Exception as e:
            print(f"[News/Sentiment] Error caching economic calendar: {e}")
        
        economic_calendar_cache = []
        last_calendar_fetch = now_ts
        return economic_calendar_cache

def add_news_proximity_feature(df):
    if df.empty:
        df["hours_to_news"] = 72.0
        return df
        
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    events = fetch_economic_calendar_cached(start_ts, end_ts)
    
    if not events:
        df["hours_to_news"] = 72.0
        return df
        
    import bisect
    df_dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    hours_to_news_list = []
    events_utc = [pd.Timestamp(ev).tz_localize("UTC") for ev in events]
    
    for current_time in df_dt:
        idx = bisect.bisect_right(events_utc, current_time)
        if idx < len(events_utc):
            next_event = events_utc[idx]
            diff_hours = (next_event - current_time).total_seconds() / 3600.0
            hours_to_news_list.append(min(72.0, max(0.0, diff_hours)))
        else:
            hours_to_news_list.append(72.0)
            
    df["hours_to_news"] = hours_to_news_list
    return df

# =========================
# ORDER BOOK PRESSURE
# =========================
def get_orderbook_imbalance(symbol=None):
    if symbol is None:
        symbol = SYMBOL
        
    # WebSocket Cache Check (Instantly bypass HTTP request if cache is warm and fresh <= 5.0s)
    with order_flow_lock:
        cached = order_flow_data.get(symbol)
        now_ts = time.time()
        # Finding #151: Check orderbook timestamp freshness (stale if older than 5.0s)
        is_fresh = cached and (now_ts - cached.get("last_ob_ts", 0.0) <= 5.0)
        if is_fresh and cached.get("latest_bids") and cached.get("latest_asks"):
            bids = cached["latest_bids"]
            asks = cached["latest_asks"]
            try:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid_price = (best_bid + best_ask) / 2.0
                spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 0.0
                
                bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids)
                ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks)
                total_depth_usd = bid_depth_usd + ask_depth_usd

                bid_vol = 0.0
                ask_vol = 0.0
                for i in range(min(10, len(bids), len(asks))):
                    w = 1.0 / (i + 1.0)
                    bid_vol += float(bids[i][1]) * w
                    ask_vol += float(asks[i][1]) * w
                    
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                return {
                    "imbalance": float(imbalance),
                    "spread": float(spread),
                    "total_depth": float(total_depth_usd),
                    "bid_depth": float(bid_depth_usd),
                    "ask_depth": float(ask_depth_usd),
                    "timestamp": cached.get("last_ob_ts", now_ts)
                }
            except Exception:
                pass

    from data import get_orderbook_imbalance as data_get_ob
    return data_get_ob(symbol=symbol)


def get_orderbook_imbalance_and_spread(symbol=None):
    """Finding #151: WebSocket-cache-aware orderbook helper. Rejects stale cache older than 5.0s
    and falls back to a direct bybit_get_request call so it can be patched in tests."""
    if symbol is None:
        symbol = SYMBOL

    with order_flow_lock:
        cached = order_flow_data.get(symbol)
        now_ts = time.time()
        is_fresh = cached and (now_ts - cached.get("last_ob_ts", 0.0) <= 5.0)
        if is_fresh and cached.get("latest_bids") and cached.get("latest_asks"):
            bids = cached["latest_bids"]
            asks = cached["latest_asks"]
            try:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid_price = (best_bid + best_ask) / 2.0
                spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 0.0
                bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids)
                ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks)
                total_depth_usd = bid_depth_usd + ask_depth_usd
                bid_vol = 0.0
                ask_vol = 0.0
                for i in range(min(10, len(bids), len(asks))):
                    w = 1.0 / (i + 1.0)
                    bid_vol += float(bids[i][1]) * w
                    ask_vol += float(asks[i][1]) * w
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                return {
                    "imbalance": float(imbalance),
                    "spread": float(spread),
                    "total_depth": float(total_depth_usd),
                    "bid_depth": float(bid_depth_usd),
                    "ask_depth": float(ask_depth_usd),
                    "timestamp": cached.get("last_ob_ts", now_ts)
                }
            except Exception as exc:
                log_event("DEBUG", f"orderbook cache parse notice: {exc}")


    # Cache is stale or missing — fall back to direct API call (patchable in tests)
    try:
        res = bybit_get_request(
            "/v5/market/orderbook",
            params={"category": "linear", "symbol": symbol.upper(), "limit": 25}
        )
        if res and res.get("retCode") == 0:
            result = res.get("result", {})
            bids = result.get("b", [])
            asks = result.get("a", [])
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid_price = (best_bid + best_ask) / 2.0
                spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 0.0
                bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids)
                ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks)
                total_depth_usd = bid_depth_usd + ask_depth_usd
                bid_vol = 0.0
                ask_vol = 0.0
                for i in range(min(10, len(bids), len(asks))):
                    w = 1.0 / (i + 1.0)
                    bid_vol += float(bids[i][1]) * w
                    ask_vol += float(asks[i][1]) * w
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                return {
                    "imbalance": float(imbalance),
                    "spread": float(spread),
                    "total_depth": float(total_depth_usd),
                    "bid_depth": float(bid_depth_usd),
                    "ask_depth": float(ask_depth_usd)
                }
    except Exception as exc:
        log_event("DEBUG", f"orderbook direct request notice: {exc}")


    return {"imbalance": 0.0, "spread": 0.0, "total_depth": 0.0, "bid_depth": 0.0, "ask_depth": 0.0}


# ==========================================
# CONFIDENCE CALIBRATION & HISTORICAL STATS
# ==========================================
def calculate_historical_thresholds(model_trend, interval):
    if model_trend is None:
        return 0.55, 0.75
    print(f"Fetching historical data to calibrate confidence percentiles (last 1,000 candles for {SYMBOL} + BTCUSDT on {interval}m interval)...")
    try:
        df_target = get_history(symbol=SYMBOL, interval=interval, limit=1000, pages=1)
        df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=1000, pages=1)
        
        if df_target is not None and len(df_target) > 0 and df_btc is not None and len(df_btc) > 0:
            df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
            df = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
            if len(df) > 0:
                df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=interval)
                df = add_features(df, symbol=SYMBOL, interval=interval)
                
                selected_features_list = None
                if interval in models_by_interval:
                    selected_features_list = models_by_interval[interval].get("selected_features")
                if selected_features_list is None:
                    import json
                    selected_features_filename = f"selected_features_{interval}.json"
                    if os.path.exists(selected_features_filename):
                        with open(selected_features_filename, "r") as f:
                            selected_features_list = json.load(f)
                            
                if selected_features_list is not None:
                    X_hist = df[selected_features_list].values
                else:
                    model_n_feats = getattr(model_trend, "n_features_in_", None)
                    if model_n_feats is not None and model_n_feats < len(features):
                        X_hist = df[features[:model_n_feats]].values
                    else:
                        X_hist = df[features].values
                probs = model_trend.predict_proba(X_hist)
                confidences = np.max(probs, axis=1)
                
                p95 = float(np.percentile(confidences, 95))
                max_conf = float(np.max(confidences))
                mean_conf = float(np.mean(confidences))
                
                print(f"Confidence Calibration Done for {interval}m:")
                print(f"  - Historical Mean: {mean_conf*100:.2f}%")
                print(f"  - 95th Percentile Threshold (Maps to 80%): {p95*100:.2f}%")
                print(f"  - Maximum Confidence (Maps to 100%): {max_conf*100:.2f}%")
                return p95, max_conf
    except Exception as e:
        print(f"Error calculating calibration for {interval}m: {e}. Using defaults.")
    
    return 0.55, 0.75

def calibrate_confidence(raw_conf, eps=1e-3):
    """
    Preserves true calibrated probability output from ensemble classifier
    clipped away from 0.0 and 1.0 saturation boundary (EPS = 1e-3).
    """
    return float(np.clip(raw_conf, eps, 1.0 - eps))

_funding_rate_cache = {}
_funding_rate_cache_lock = threading.Lock()

def get_funding_rate(symbol=SYMBOL):
    now_t = time.time()
    with _funding_rate_cache_lock:
        if symbol in _funding_rate_cache:
            rate_val, exp_t = _funding_rate_cache[symbol]
            if now_t < exp_t:
                return rate_val
    try:
        url = f"{BYBIT_BASE_URL}/v5/market/tickers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "linear", "symbol": symbol}, headers=headers, proxies=get_bybit_proxies(), timeout=5)
        if response.status_code == 200:
            res = response.json()
            if "result" in res and "list" in res["result"] and len(res["result"]["list"]) > 0:
                rate_str = res["result"]["list"][0].get("fundingRate")
                if rate_str:
                    rate_val = float(rate_str)
                    with _funding_rate_cache_lock:
                        _funding_rate_cache[symbol] = (rate_val, now_t + 60.0)
                    return rate_val
    except Exception as e:
        log_event("WARNING", f"Error fetching funding rate: {e}")
    return 0.0

def get_rolling_correlation(symbol_a: str, symbol_b: str, interval: str = "60", window: int = 30) -> float:
    if symbol_a == symbol_b:
        return 1.0
    try:
        df_a = get_history(symbol=symbol_a, interval=interval, limit=window)
        df_b = get_history(symbol=symbol_b, interval=interval, limit=window)
        if df_a is not None and df_b is not None and len(df_a) >= 15 and len(df_b) >= 15:
            merged = pd.merge(df_a[["timestamp", "close"]], df_b[["timestamp", "close"]], on="timestamp", suffixes=("_a", "_b"))
            if len(merged) >= 10:
                corr = merged["close_a"].corr(merged["close_b"])
                if not np.isnan(corr):
                    return float(np.clip(corr, -1.0, 1.0))
    except Exception as e:
        log_event("DEBUG", f"Rolling correlation calculation fallback ({symbol_a}/{symbol_b}): {e}")
    return 0.70



def calculate_daily_pnl(trades=None, time_now_dt=None):
    """Finding #144: Safely calculates the sum of PnL from trades closed today.

    Accepts ``time_now_dt`` as a ``datetime`` object or ``None`` (defaults to
    ``datetime.now(timezone.utc)``).  Tolerates trade records whose
    ``closed_at`` field is a ``datetime`` object, a valid ISO-8601 string, or
    a malformed/missing value — invalid entries are silently skipped.
    """
    from datetime import datetime, timezone
    try:
        if time_now_dt is None or not isinstance(time_now_dt, datetime):
            time_now_dt = datetime.now(timezone.utc)
        # Normalise to UTC-aware
        if time_now_dt.tzinfo is None:
            time_now_dt = time_now_dt.replace(tzinfo=timezone.utc)
        today_date = time_now_dt.date()
    except Exception as exc:
        log_event("DEBUG", f"calculate_daily_pnl date handling notice: {exc}")
        today_date = None

    if trades is None:
        trades = bot_state.get("trade_history", [])

    total = 0.0
    for t in trades:
        try:
            closed_at = t.get("closed_at")
            if closed_at is None:
                continue
            if isinstance(closed_at, datetime):
                trade_dt = closed_at
            else:
                trade_dt = datetime.fromisoformat(str(closed_at))
            if trade_dt.tzinfo is None:
                trade_dt = trade_dt.replace(tzinfo=timezone.utc)
            if today_date is not None and trade_dt.date() != today_date:
                continue
            pnl_val = t.get("realized_pnl", t.get("pnl", 0.0))
            total += float(pnl_val) if pnl_val is not None else 0.0
        except Exception as exc:
            log_event("DEBUG", f"calculate_daily_pnl trade item notice: {exc}")
            continue
    return float(total)


def calculate_recent_performance_leverage_multiplier(days=7):
    """
    Calculates a leverage multiplier based on the rolling Sharpe ratio of completed trades.
    Reduces max leverage during drawdowns to manage risk.
    """
    try:
        trades = bot_state.get("trade_history", [])
        if len(trades) < 5:
            return 1.0
            
        import time as _time
        cutoff = _time.time() - days * 86400
        recent_trades = [t for t in trades if float(t.get("exit_time", 0.0)) >= cutoff]
        
        if len(recent_trades) < 3:
            return 1.0
            
        pnls = [float(t.get("pnl_usd", 0.0)) for t in recent_trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        
        if std_pnl < 1e-4:
            return 1.0
            
        sharpe = mean_pnl / std_pnl
        if sharpe >= 0:
            return 1.0
        else:
            # Scale down leverage. Clamp multiplier between 0.30 and 1.0
            multiplier = max(0.30, min(1.0, 1.0 - 0.3 * abs(sharpe)))
            print(f"[Sharpe-Adaptive Leverage] Sharpe={sharpe:.2f} -> Sizing down leverage by {multiplier:.2f}x")
            return float(multiplier)
    except Exception as e:
        print(f"[Sharpe-Adaptive Leverage Error] {e}")
        return 1.0

# Pre-trade confluence check is imported directly from confluence_engine.py

# =========================
# LIVE LOOP
# =========================
def get_fallback_price(symbol=SYMBOL):
    # 1. Try Bybit API
    try:
        url = f"{BYBIT_BASE_URL}/v5/market/tickers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, params={"category": "linear", "symbol": symbol}, headers=headers, proxies=get_bybit_proxies(), timeout=5)
        if response.status_code == 200:
            res = response.json()
            ticker_list = res.get("result", {}).get("list", [])
            if ticker_list:
                ticker = ticker_list[0]
                price_key = "lastPrice"
                return float(ticker.get("lastPrice"))
            else:
                print(f"Bybit price ticker list is empty for {symbol}")
        else:
            print(f"Bybit price ticker for {symbol} returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Bybit price fallback for {symbol}: {e}")

    # 1b. Try Bybit Live API as fallback for testnet missing symbols
    if TRADE_MODE == "testnet":
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, params={"category": "linear", "symbol": symbol}, headers=headers, proxies=get_bybit_proxies(), timeout=5)
            if response.status_code == 200:
                res = response.json()
                ticker_list = res.get("result", {}).get("list", [])
                if ticker_list:
                    ticker = ticker_list[0]
                    return float(ticker.get("lastPrice"))
        except Exception as e:
            print(f"Error fetching Bybit Live price fallback for {symbol}: {e}")

    # 2. Try Coinbase API (only for BTCUSDT)
    if symbol == "BTCUSDT":
        try:
            response = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=5)
            if response.status_code == 200:
                res = response.json()
                return float(res["data"]["amount"])
        except Exception:
            pass

    # 3. Try Binance API
    try:
        response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5)
        if response.status_code == 200:
            res = response.json()
            return float(res["price"])
        else:
            print(f"Binance price ticker for {symbol} returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Binance price fallback for {symbol}: {e}")

def load_initial_prices():
    global live_price, last_ws_update_time
    print("[Startup] Loading initial market prices for all assets...")
    try:
        url = f"{BYBIT_BASE_URL}/v5/market/tickers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, params={"category": "linear"}, headers=headers, proxies=get_bybit_proxies(), timeout=8)
        found_symbols = set()
        if resp.status_code == 200:
            ticker_list = resp.json().get("result", {}).get("list", [])
            for ticker in ticker_list:
                sym = ticker.get("symbol")
                if sym in SUPPORTED_SYMBOLS:
                    val_str = ticker.get("lastPrice")
                    if val_str:
                        val = float(val_str)
                        now_init = time.time()
                        bot_state[f"live_price_{sym}"] = val
                        bot_state[f"live_price_ts_{sym}"] = now_init
                        found_symbols.add(sym)
                        if sym == "BTCUSDT":
                            live_price = val
                            bot_state["live_price"] = val
                            bot_state["last_rest_price_time"] = now_init
                            bot_state["last_update"] = now_init
        
        # Fall back to external sources (Binance/Coinbase) for any missing symbols (e.g. LINKUSDT on testnet)
        for sym in SUPPORTED_SYMBOLS:
            if sym not in found_symbols:
                val = get_fallback_price(sym)
                if val is not None:
                    now_fb = time.time()
                    bot_state[f"live_price_{sym}"] = val
                    bot_state[f"live_price_ts_{sym}"] = now_fb
                    if sym == "BTCUSDT" and live_price is None:
                        live_price = val
                        bot_state["live_price"] = val
                        bot_state["last_rest_price_time"] = now_fb
                        bot_state["last_update"] = now_fb

        # Pre-warm instrument precision cache to ensure 0-latency live order paths
        for sym in SUPPORTED_SYMBOLS:
            try:
                get_bybit_min_qty_step(sym)
            except Exception as ex_spec:
                log_event("DEBUG", f"Pre-warm spec notice for {sym}: {ex_spec}")
    except Exception as e:
        print(f"[Initial Prices] Error loading prices at startup: {e}")


def get_bybit_entry_order_qty(symbol, side):
    """Query Bybit order history for the symbol to find the originally requested quantity of the entry order."""
    try:
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": 20
        }
        res = bybit_get_request("/v5/order/history", params)
        if res.get("retCode") == 0:
            orders = res.get("result", {}).get("list", [])
            for o in orders:
                o_side = o.get("side")
                o_cum = float(o.get("cumExecQty", 0.0))
                o_qty = float(o.get("qty", 0.0))
                if o_side == side and o_cum > 0.0:
                    return o_qty, o_cum, o.get("orderId")
    except Exception as e:
        print(f"[Entry Order Query Warning] Error for {symbol} {side}: {e}")
    return None, None, None

def get_bybit_active_limit_order_id(symbol, side):
    """Query Bybit active open orders to find if there is an active limit order on the specified side."""
    try:
        params = {
            "category": "linear",
            "symbol": symbol
        }
        res = bybit_get_request("/v5/order/realtime", params)
        if res.get("retCode") == 0:
            orders = res.get("result", {}).get("list", [])
            for o in orders:
                if o.get("orderType") == "Limit" and o.get("side") == side and o.get("orderStatus") in ["New", "PartiallyFilled"]:
                    return o.get("orderId")
    except Exception as e:
        print(f"[Active Limit Order Query Warning] Error for {symbol} {side}: {e}")
    return None

def sync_active_positions_from_bybit():
    """Real-time Sync: Sync all active trades from Bybit to keep bot_state completely aligned with testnet/live."""
    if TRADE_MODE == "simulation":
        return True
    
    try:
        # Finding #26: Release locks during blocking network I/O; acquire locks only for memory/DB mutations
        import contextlib
        with contextlib.nullcontext():
            with contextlib.nullcontext():
                pos_list = get_all_bybit_positions()
                if pos_list is None:
                    print("[Position Sync] Failed to fetch positions from Bybit. Skipping sync to prevent false exits.")
                    return False
                
                # Filter for positions with non-zero size
                open_positions = {}
                for pos in pos_list:
                    qty_val = float(pos.get("size", "0"))
                    if qty_val > 0:
                        open_positions[pos.get("symbol")] = pos

                matched_symbols = set()
                for tf_key in ACTIVE_TRADE_TF_KEYS:
                    with active_trades_lock:
                        current_trades = bot_state.get(f"active_trade_{tf_key}", [])
                        if not isinstance(current_trades, list):
                            current_trades = []
                        current_trades = list(current_trades)
                    
                    seen_symbols_in_tf = set()
                    updated_trades = []
                    for t in current_trades:
                        symbol = t.get("symbol")
                        if not symbol:
                            continue
                        if symbol in seen_symbols_in_tf:
                            print(f"[De-duplication] Discarded duplicate active trade for {symbol} in timeframe {tf_key}.")
                            continue
                        seen_symbols_in_tf.add(symbol)
                        if symbol in open_positions:
                            pos = open_positions[symbol]
                            t["bybit_closed"] = False
                            
                            conf_val = t.get("confidence", 0.0)
                            is_zero_conf = False
                            if conf_val is not None and conf_val != "MT":
                                try:
                                    is_zero_conf = (float(conf_val) == 0.0)
                                except (ValueError, TypeError):
                                    pass
                            
                            if is_zero_conf:
                                trade_dir = t.get("direction", "Bullish")
                                for p in reversed(bot_state.get("prediction_history", [])):
                                    if p.get("symbol") == symbol and p.get("direction") == trade_dir:
                                        if abs(p.get("timestamp", 0) - time.time()) < 86400 * 2:
                                            t["confidence"] = float(p.get("calibrated_confidence", p.get("confidence", 0.63)))
                                            print(f"[Sync Confidence Restore] Restored confidence for recovered {symbol} trade: {t['confidence']*100:.2f}%")
                                            save_history()
                                            break

                            # Side Mismatch Guard: verify Bybit position side aligns with trade direction
                            pos_side = pos.get("side") # "Buy" or "Sell"
                            trade_direction = t.get("direction") # "Bullish" or "Bearish"
                            mismatch = False
                            if trade_direction == "Bullish" and pos_side != "Buy":
                                mismatch = True
                            elif trade_direction == "Bearish" and pos_side != "Sell":
                                mismatch = True
                                
                            if mismatch:
                                print(f"[Side Mismatch Guard] Stale local record detected for {symbol} in {tf_key} (Direction: {trade_direction} vs Bybit Live: {pos_side}). Discarding stale local record to preserve live exchange trade.")
                                continue
        
                            t["entry_price"] = float(pos.get("avgPrice", t["entry_price"]))
                            t["liq_price"] = float(pos.get("liqPrice", 0.0)) if pos.get("liqPrice") else 0.0
                            t["mark_price"] = float(pos.get("markPrice", 0.0)) if pos.get("markPrice") else 0.0
                            t["qty"] = float(pos.get("size", t["qty"]))
                            t["leverage"] = float(pos.get("leverage", t["leverage"]))
                            
                            # Sync TP/SL from Bybit exchange if they exist and are non-zero
                            bybit_sl = float(pos.get("stopLoss", 0.0)) if pos.get("stopLoss") else 0.0
                            bybit_tp = float(pos.get("takeProfit", 0.0)) if pos.get("takeProfit") else 0.0
                            if bybit_sl > 0.0:
                                if t.get("break_even_triggered", False):
                                    if t.get("direction") == "Bullish":
                                        t["stop_loss"] = max(bybit_sl, t["entry_price"])
                                    else:
                                        t["stop_loss"] = min(bybit_sl, t["entry_price"])
                                else:
                                    t["stop_loss"] = bybit_sl
                            if bybit_tp > 0.0:
                                t["take_profit"] = bybit_tp
                            
                            pos_val = float(pos.get("positionValue", 0.0))
                            t["position_size_usd"] = pos_val / t["leverage"] if t["leverage"] > 0 else pos_val
                            t["qty"] = abs(float(pos.get("size", 0.0)))
                            
                            # Proportional Unrealized PnL calculation
                            try:
                                same_symbol_trades = []
                                for tf_check in ACTIVE_TRADE_TF_KEYS:
                                    for t_item in bot_state.get(f"active_trade_{tf_check}", []):
                                        if t_item.get("symbol") == symbol:
                                            same_symbol_trades.append(t_item)
                                total_lev_size = sum(float(t_item.get("position_size_usd", 0.0)) * float(t_item.get("leverage", 1.0)) for t_item in same_symbol_trades)
                                position_pnl = float(pos.get("unrealisedPnl", 0.0))
                                if total_lev_size > 0:
                                    this_lev_size = float(t.get("position_size_usd", 0.0)) * float(t.get("leverage", 1.0))
                                    t["bybit_unrealized_pnl"] = round(position_pnl * (this_lev_size / total_lev_size), 2)
                                else:
                                    t["bybit_unrealized_pnl"] = position_pnl
                            except Exception:
                                t["bybit_unrealized_pnl"] = float(pos.get("unrealisedPnl", 0.0))
                            
                            # Sanitize ATR, TP, and SL for active trades to prevent invalid/stuck parameters
                            avg_price = t["entry_price"]
                            mark_price = t["mark_price"]
                            liq_price = t["liq_price"]
                            direction = t.get("direction", "Bullish")
                            
                            # Fix ATR if it's unreasonably large or unset
                            current_atr = t.get("atr_dollars", 0.0)
                            if current_atr <= 0 or current_atr > 0.05 * avg_price:
                                current_atr = 0.015 * avg_price
                                t["atr_dollars"] = current_atr
                                
                            # Fix Take Profit if it is unset (0.0) or missing on venue
                            venue_tp = float(pos.get("takeProfit", 0.0) or 0.0)
                            local_tp = float(t.get("take_profit", 0.0))
                            if local_tp == 0.0:
                                if direction == "Bullish":
                                    local_tp = max(mark_price + 1.25 * current_atr, avg_price + 1.25 * current_atr)
                                else:
                                    local_tp = min(mark_price - 1.25 * current_atr, avg_price - 1.25 * current_atr)
                                t["take_profit"] = local_tp
                            
                            # If venue has no TP set or venue differs materially from local target TP, re-push
                            if TRADE_MODE != "simulation" and (venue_tp == 0.0 or abs(venue_tp - local_tp) > 1e-4):
                                update_bybit_take_profit(symbol, t["take_profit"], t)
                                    
                            # Fix Stop Loss if it is unset, below liquidation (for long), or above liquidation (for short)
                            sl_val = t.get("stop_loss", 0.0)
                            sl_updated = False
                            if direction == "Bullish":
                                if sl_val <= 0.0 or (liq_price > 0.0 and sl_val <= liq_price) or sl_val < avg_price - 3.0 * current_atr:
                                    sl_val = avg_price - 0.75 * current_atr
                                    if liq_price > 0.0 and sl_val <= liq_price:
                                        sl_val = liq_price + 0.2 * current_atr
                                    sl_updated = True
                            else:
                                if sl_val <= 0.0 or (liq_price > 0.0 and sl_val >= liq_price) or sl_val > avg_price + 3.0 * current_atr:
                                    sl_val = avg_price + 0.75 * current_atr
                                    if liq_price > 0.0 and sl_val >= liq_price:
                                        sl_val = liq_price - 0.2 * current_atr
                                    sl_updated = True
                                    
                            if sl_updated:
                                if TRADE_MODE != "simulation":
                                    success = update_bybit_stop_loss(symbol, sl_val, t)
                                    if success:
                                        t["stop_loss"] = sl_val
                                        t["sl_venue_synced"] = True
                                    else:
                                        # bybit_sl is 0.0 exactly when the venue has NO stop attached,
                                        # and this branch is reached precisely because the stop was
                                        # missing or invalid -- so adopting it recorded "no stop" and
                                        # discarded the only protective level that had been computed.
                                        # Retain the sanitized level for tracking and surface the fact
                                        # that the venue is unprotected. Alert once per transition so a
                                        # persistently failing push does not spam every sync pass.
                                        _was_synced = t.get("sl_venue_synced", True)
                                        t["stop_loss"] = sl_val
                                        t["sl_venue_synced"] = False
                                        if _was_synced is not False:
                                            log_event("CRITICAL", f"[{symbol}] Venue stop-loss push FAILED. Position is UNPROTECTED on the exchange; retaining local stop {sl_val:.6f} for tracking only.")
                                            send_telegram_alert(f"🚨 *UNPROTECTED POSITION* 🚨\n• *Symbol*: {symbol}\n• Venue stop-loss push failed\n• Exchange has NO stop attached\n• Local stop retained: `{sl_val:.6f}`")
                                else:
                                    t["stop_loss"] = sl_val
                            
                            # Dynamically reconstruct original size for recovered trades
                            if t.get("recovered", False) and t.get("original_size_reconstructed", False) is not True:
                                orig_qty, filled_qty, entry_ord_id = get_bybit_entry_order_qty(symbol, pos_side)
                                if orig_qty is not None and orig_qty > 0.0:
                                    t["original_size"] = (orig_qty * avg_price) / t["leverage"] if t["leverage"] > 0 else (orig_qty * avg_price)
                                    t["fill_pct"] = round((filled_qty / orig_qty) * 100.0, 2)
                                    t["original_size_reconstructed"] = True
                                    if entry_ord_id and not t.get("bybit_order_id"):
                                        t["bybit_order_id"] = entry_ord_id
                                    print(f"[Crash Recovery Sync] Reconstructed original size for {symbol}: target={t['original_size']:.2f}, filled={t['position_size_usd']:.2f} ({t['fill_pct']}%)")
                                    save_history()

                            # Recover missing/unset bybit_order_id
                            if not t.get("bybit_order_id") or t.get("bybit_order_id") == "N/A":
                                _, _, entry_ord_id = get_bybit_entry_order_qty(symbol, pos_side)
                                if entry_ord_id:
                                    t["bybit_order_id"] = entry_ord_id
                                    print(f"[Sync] Recovered missing bybit_order_id {entry_ord_id} for {symbol}")

                            # Recover missing/unset bybit_scale_out_order_id
                            if not t.get("bybit_scale_out_order_id") or t.get("bybit_scale_out_order_id") == "N/A":
                                limit_side = "Sell" if pos_side == "Buy" else "Buy"
                                limit_ord_id = get_bybit_active_limit_order_id(symbol, limit_side)
                                if limit_ord_id:
                                    t["bybit_scale_out_order_id"] = limit_ord_id
                                    print(f"[Sync] Recovered missing bybit_scale_out_order_id {limit_ord_id} for {symbol}")
                            
                            updated_trades.append(t)
                            matched_symbols.add(symbol)
                        else:
                            # Position is closed on Bybit: keep ONLY if exit has not yet been processed
                            if symbol in active_execution_symbols:
                                # Finding #22: DO NOT flag as closed while order placement/execution is in-flight!
                                updated_trades.append(t)
                                continue
                            if not t.get("exit_processed", False):
                                t["bybit_closed"] = True
                                updated_trades.append(t)
                            else:
                                print(f"[Sync Cleanup] Removed already processed closed trade for {symbol} ({tf_key}).")
                    
                    with active_trades_lock:
                        # Findings #25 & N5: Merge concurrent state updates and manual trades before overwriting active trades
                        curr_live = bot_state.get(f"active_trade_{tf_key}", [])
                        if isinstance(curr_live, list):
                            live_by_id = {
                                (c_tr.get("trade_id") or f"{c_tr.get('symbol')}_{c_tr.get('entry_time')}"): c_tr
                                for c_tr in curr_live
                            }
                            filtered_updated = []
                            for t in updated_trades:
                                t_id = t.get("trade_id") or f"{t.get('symbol')}_{t.get('entry_time')}"
                                live_tr = live_by_id.get(t_id)
                                if live_tr:
                                    # Preserve dynamic exit / scale-out / BE state from concurrent exit evaluations
                                    if live_tr.get("exit_processed", False):
                                        t["exit_processed"] = True
                                    if live_tr.get("break_even_triggered", False):
                                        t["break_even_triggered"] = True
                                    if live_tr.get("half_closed", False):
                                        t["half_closed"] = True
                                        if "scaled_out_margin" in live_tr:
                                            t["scaled_out_margin"] = live_tr["scaled_out_margin"]
                                        if "scaled_out_pnl" in live_tr:
                                            t["scaled_out_pnl"] = live_tr["scaled_out_pnl"]
                                        if "scale_out_price" in live_tr:
                                            t["scale_out_price"] = live_tr["scale_out_price"]
                                    c_sl = live_tr.get("stop_loss")
                                    if c_sl is not None and float(c_sl) > 0:
                                        t_sl = float(t.get("stop_loss") or 0.0)
                                        direction = t.get("direction", "Bullish")
                                        if direction == "Bullish":
                                            if float(c_sl) > t_sl:
                                                t["stop_loss"] = float(c_sl)
                                        else:
                                            if t_sl == 0.0 or float(c_sl) < t_sl:
                                                t["stop_loss"] = float(c_sl)
                                    t["highest_price"] = max(float(t.get("highest_price", 0.0) or 0.0), float(live_tr.get("highest_price", 0.0) or 0.0))
                                    t_lowest = float(t.get("lowest_price", float("inf")) or float("inf"))
                                    c_lowest = float(live_tr.get("lowest_price", float("inf")) or float("inf"))
                                    t["lowest_price"] = min(t_lowest, c_lowest)
                                
                                # Do not retain trade if exit was already processed and position not open on Bybit
                                if t.get("exit_processed", False) and t.get("symbol") not in open_positions:
                                    continue
                                filtered_updated.append(t)
                            
                            upd_ids = {t.get("trade_id") or f"{t.get('symbol')}_{t.get('entry_time')}" for t in filtered_updated}
                            for c_tr in curr_live:
                                c_id = c_tr.get("trade_id") or f"{c_tr.get('symbol')}_{c_tr.get('entry_time')}"
                                if c_id not in upd_ids and not c_tr.get("exit_processed", False):
                                    filtered_updated.append(c_tr)
                            updated_trades = filtered_updated
                        bot_state[f"active_trade_{tf_key}"] = updated_trades
                        try:
                            import database
                            database.save_active_trades(tf_key, updated_trades)
                        except Exception as ex_db_s:
                            log_event("WARNING", f"Error persisting active trades for {tf_key}: {ex_db_s}")
    
                # Reconstruct any open positions on Bybit that are NOT in bot_state (orphaned/manual positions)
                recovered = 0
                for symbol, pos in open_positions.items():
                    in_active_execution = (symbol in active_execution_symbols)
                    if in_active_execution:
                        log_event("INFO", f"[Crash Recovery] Skipped recovery scan for {symbol} - trade is currently being executed async.")
                        continue
                    # Re-check ALL active timeframes live in bot_state at this moment under lock (Finding N7)
                    with active_trades_lock:
                        currently_tracked = any(
                            any(t.get("symbol") == symbol for t in bot_state.get(f"active_trade_{k}", []))
                            for k in ACTIVE_TRADE_TF_KEYS
                        )
                    if currently_tracked:
                        log_event("INFO", f"[Crash Recovery] Skipped recovery for {symbol} - already tracked in current bot_state (live re-check).")
                        continue
                    if symbol not in matched_symbols:
                        avg_price = float(pos.get("avgPrice", "0"))
                        liq_price = float(pos.get("liqPrice", "0")) if pos.get("liqPrice") else 0.0
                        mark_price = float(pos.get("markPrice", "0")) if pos.get("markPrice") else 0.0
                        leverage_val = float(pos.get("leverage", "1"))
                        side_str = pos.get("side", "Buy")
                        direction = "Bullish" if side_str == "Buy" else "Bearish"
                        sl_price = float(pos.get("stopLoss", "0")) if pos.get("stopLoss") else 0.0
                        tp_price = float(pos.get("takeProfit", "0")) if pos.get("takeProfit") else 0.0
                        position_value = float(pos.get("positionValue", "0"))
                        position_size_usd = position_value / leverage_val if leverage_val > 0 else position_value
                        qty_val = float(pos.get("size", "0"))
                        
                        # Retrieve the original target qty of the entry order
                        orig_qty, filled_qty, entry_order_id = get_bybit_entry_order_qty(symbol, side_str)
                        if orig_qty is not None and orig_qty > 0.0:
                            original_size = (orig_qty * avg_price) / leverage_val if leverage_val > 0 else (orig_qty * avg_price)
                            fill_pct = round((filled_qty / orig_qty) * 100.0, 2)
                            original_size_reconstructed = True
                            print(f"[Crash Recovery] Discovered target size for {symbol}: target={original_size:.2f}, filled={position_size_usd:.2f} ({fill_pct}%)")
                        else:
                            original_size = position_size_usd
                            fill_pct = 100.0
                            original_size_reconstructed = False
                        
                        limit_side = "Sell" if side_str == "Buy" else "Buy"
                        scale_out_order_id = get_bybit_active_limit_order_id(symbol, limit_side)
                        
                        import uuid
                        trade_uuid = str(uuid.uuid4())
                        
                        # Accurate timeframe and confidence resolution from decision_journal execution records
                        matched_tf = "4h"  # fallback default
                        matched_confidence = 0.50
                        try:
                            import database
                            import sqlite3
                            con = database.get_db_connection()
                            cur = con.cursor()
                            expected_dir_alt = "Long" if direction == "Bullish" else "Short"
                            min_recov_ts = time.time() - (86400 * 2)
                            # Finding #28: Match closest entry timestamp to position creation time rather than latest row
                            ref_pos_ts = float(pos.get("createdTime", 0)) / 1000.0 if pos.get("createdTime") else (float(pos.get("updatedTime", 0)) / 1000.0 if pos.get("updatedTime") else time.time())
                            cur.execute("""
                                SELECT interval, calibrated_conf, direction, ts FROM decision_journal 
                                WHERE symbol = ? AND outcome = 'EXECUTED' 
                                  AND (direction = ? OR direction = ?)
                                  AND ts >= ?
                                ORDER BY ABS(ts - ?) ASC LIMIT 1
                            """, (symbol, direction, expected_dir_alt, min_recov_ts, ref_pos_ts))
                            exec_row = cur.fetchone()
                            con.close()
                            if exec_row:
                                matched_row_dir = str(exec_row[2])
                                if matched_row_dir in (direction, expected_dir_alt):
                                    tf_map_inv = {"5": "5m", "15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h", "360": "6h"}
                                    matched_tf = tf_map_inv.get(str(exec_row[0]), "4h")
                                    matched_confidence = float(exec_row[1] or 0.50)
                                    print(f"[Crash Recovery] Matched executed journal record for {symbol} ({matched_row_dir}): {matched_tf} ({exec_row[0]}m), conf={matched_confidence*100:.2f}%")
                        except Exception as e_rec:
                            log_event("WARNING", f"Failed to query decision_journal for recovery: {e_rec}")
                            for p in reversed(bot_state.get("prediction_history", [])):
                                if p.get("symbol") == symbol and p.get("direction") == direction:
                                    if abs(p.get("timestamp", 0) - time.time()) < 86400 * 2:
                                        matched_tf_interval = p.get("interval", "240")
                                        tf_map_inv = {"5": "5m", "15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h", "360": "6h"}
                                        matched_tf = tf_map_inv.get(str(matched_tf_interval), "4h")
                                        matched_confidence = float(p.get("calibrated_confidence", p.get("confidence", 0.50)))
                                        break

                        recov_sl_mult = risk_engine.get_timeframe_stop_multiplier(matched_tf)

                        # Calculate proper ATR on recovery: prefer measured ATR over inversion
                        calc_atr = None
                        try:
                            _df_r = get_history(symbol=symbol, interval="60", limit=50)
                            if _df_r is not None and len(_df_r) >= 15:
                                if "ATR" in _df_r.columns:
                                    _a = float(_df_r["ATR"].iloc[-1])
                                else:
                                    _high = _df_r["high"].astype(float)
                                    _low = _df_r["low"].astype(float)
                                    _cp = _df_r["close"].astype(float).shift(1)
                                    _tr = pd.concat([_high - _low, (_high - _cp).abs(), (_low - _cp).abs()], axis=1).max(axis=1)
                                    _a = float(_tr.ewm(alpha=1.0/14.0, adjust=False).mean().dropna().iloc[-1])
                                if _a > 0:
                                    calc_atr = _a
                        except Exception as _e:
                            log_event("DEBUG", f"Recovery ATR fetch failed for {symbol}: {_e}")

                        if calc_atr is None:
                            calc_atr = abs(avg_price - sl_price) / max(0.1, recov_sl_mult) if sl_price > 0 else 0.015 * avg_price
                            log_event("WARNING", f"[{symbol}] Recovery ATR inverted from stop ({recov_sl_mult:.2f}x TF multiplier) — may be inaccurate")

                        if calc_atr is None or calc_atr <= 0:
                            calc_atr = 0.015 * avg_price
                            log_event("WARNING", f"[{symbol}] ATR unavailable in recovery — using 1.5% fallback ({calc_atr:.4f})")
                        elif calc_atr > 0.05 * avg_price:
                            calc_atr = 0.05 * avg_price
                            log_event("INFO", f"[{symbol}] Recovery ATR clamped to 5% ceiling ({calc_atr:.4f})")
                        
                        # Sanitize TP and SL on recovery
                        if tp_price == 0.0:
                            if direction == "Bullish":
                                tp_price = max(mark_price + 1.25 * calc_atr, avg_price + 1.25 * calc_atr)
                            else:
                                tp_price = min(mark_price - 1.25 * calc_atr, avg_price - 1.25 * calc_atr)
                                
                        if sl_price == 0.0 or abs(avg_price - sl_price) > 3.0 * calc_atr:
                            if direction == "Bullish":
                                sl_price = avg_price - recov_sl_mult * calc_atr
                                if liq_price > 0.0 and sl_price <= liq_price:
                                    sl_price = min(avg_price * 0.999, liq_price + 0.2 * calc_atr)
                                if sl_price >= avg_price:
                                    log_event("ERROR", f"[{symbol}] Liquidation too close for a valid long stop — leaving exchange SL")
                                    send_telegram_alert(f"🚨 {symbol}: cannot place valid SL, liq {liq_price} too close to entry {avg_price}")
                                    sl_price = None
                            else:
                                sl_price = avg_price + recov_sl_mult * calc_atr
                                if liq_price > 0.0 and sl_price >= liq_price:
                                    sl_price = max(avg_price * 1.001, liq_price - 0.2 * calc_atr)
                                if sl_price <= avg_price:
                                    log_event("ERROR", f"[{symbol}] Liquidation too close for a valid short stop — leaving exchange SL")
                                    send_telegram_alert(f"🚨 {symbol}: cannot place valid SL, liq {liq_price} too close to entry {avg_price}")
                                    sl_price = None
                        # Push the recovered/sanitized TP & SL to Bybit
                        if TRADE_MODE != "simulation":
                            update_bybit_take_profit(symbol, tp_price)
                            if sl_price is not None:
                                success = update_bybit_stop_loss(symbol, sl_price)
                                if not success:
                                    sl_price = float(pos.get("stopLoss", "0")) if pos.get("stopLoss") else 0.0

                        _iv = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360}.get(matched_tf, 60)
                        _la = TIMEFRAME_CONFIG.get(str(_iv), {}).get("lookahead", 10)

                        recovered_trade = {
                            "trade_id": f"{symbol}_{trade_uuid}_recovered",
                            "interval": str(_iv),
                            "timeframe": str(matched_tf),
                            "bybit_order_id": entry_order_id,
                            "bybit_scale_out_order_id": scale_out_order_id,
                            "symbol": symbol,
                            "entry_price": avg_price,
                            "predicted_price": avg_price,
                            "stop_loss": sl_price,
                            "take_profit": tp_price,
                            "initial_stop_loss": sl_price,
                            "initial_take_profit": tp_price,
                            "initial_planned_rr": float(abs(tp_price - avg_price) / max(1e-9, abs(avg_price - sl_price))) if (sl_price and tp_price) else 1.5,
                            "direction": direction,
                            "end_time": float(time.time() + _iv * 60 * _la),
                            "entry_time": max(int(pos.get("createdTime") or 0), int(pos.get("updatedTime") or 0)) or int(time.time() * 1000),
                            "atr_dollars": calc_atr,
                            "entry_atr": calc_atr,
                            "highest_price": max(avg_price, mark_price) if direction == "Bullish" else avg_price,
                            "lowest_price": min(avg_price, mark_price) if direction == "Bearish" else avg_price,
                            "break_even_triggered": False,
                            "half_closed": False,
                            "original_size": original_size,
                            "position_size_usd": position_size_usd,
                            "fill_pct": fill_pct,
                            "original_size_reconstructed": original_size_reconstructed,
                            "scaled_out_pnl": 0.0,
                            "kelly_fraction": 0.0,
                            "leverage": leverage_val,
                            "confidence": matched_confidence,
                            "qty": qty_val,
                            "original_qty": orig_qty if (orig_qty is not None and orig_qty > 0.0) else qty_val,
                            "liq_price": liq_price,
                            "mark_price": mark_price,
                            "entry_regime": str(bot_state.get(f"regime_{symbol}_{_iv}", bot_state.get(f"regime_{_iv}", "Trending"))),
                            "regime": str(bot_state.get(f"regime_{symbol}_{_iv}", bot_state.get(f"regime_{_iv}", "Trending"))),
                            "entry_scale_mult": 1.20,
                            "recovered": True
                        }
                        
                        tf_key = matched_tf
                        with active_trades_lock:
                            # Finding N7: Re-verify under lock that symbol is not active in execution and not already in ANY timeframe
                            if symbol in active_execution_symbols:
                                log_event("INFO", f"[Crash Recovery] Skipped duplicate recovery for {symbol} - trade execution active under lock.")
                                continue
                            already_in_any_tf = any(
                                any(t.get("symbol") == symbol for t in bot_state.get(f"active_trade_{k}", []))
                                for k in ACTIVE_TRADE_TF_KEYS
                            )
                            if already_in_any_tf:
                                log_event("INFO", f"[Crash Recovery] Skipped duplicate recovery for {symbol} - already tracked in an active timeframe under lock.")
                                continue
                            active_trades_list = bot_state.get(f"active_trade_{tf_key}", [])
                            if not isinstance(active_trades_list, list):
                                active_trades_list = []
                            active_trades_list.append(recovered_trade)
                            bot_state[f"active_trade_{tf_key}"] = active_trades_list
                            try:
                                import database
                                database.save_active_trades(tf_key, active_trades_list)
                            except Exception as ex_db_save:
                                print(f"[Crash Recovery] Failed to persist recovered trade to DB: {ex_db_save}")
                        recovered += 1
                        print(f"[Crash Recovery] Discovered/Recovered open position on Bybit: {symbol} {direction}")
                        
                if recovered > 0:
                    save_history()
                return True
    except Exception as e:
        print(f"[Crash Recovery] Error checking Bybit: {e}")
        return False

def recover_missed_closed_trades():
    """Scan Bybit closed PnL records on startup to detect and alert on any trades closed while the bot was offline."""
    if TRADE_MODE == "simulation":
        return
    print("[Crash Recovery] Scanning recently closed trades on Bybit...")
    now_ms = int(time.time() * 1000)
    one_day_ms = 24 * 3600 * 1000
    
    for symbol in SUPPORTED_SYMBOLS:
        try:
            res = bybit_get_request("/v5/position/closed-pnl", {
                "category": "linear",
                "symbol": symbol,
                "limit": "10"
            })
            if res.get("retCode") == 0:
                pnl_list = res.get("result", {}).get("list", [])
                for item in pnl_list:
                    updated_time_ms = int(item.get("updatedTime", 0))
                    if now_ms - updated_time_ms < one_day_ms:
                        exit_price = float(item.get("avgExitPrice", 0.0))
                        entry_price = float(item.get("avgEntryPrice", 0.0))
                        closed_pnl = float(item.get("closedPnl", 0.0))
                        side = item.get("side")
                        direction = "Bullish" if side == "Sell" else "Bearish"
                        
                        exit_time_sec = updated_time_ms / 1000.0
                        already_logged = False
                        db_trades = database.get_completed_trades(limit=200)
                        all_known_trades = list(bot_state.get("trade_history", [])) + db_trades
                        for t in all_known_trades:
                            if t.get("symbol") == symbol and abs(float(t.get("exit_time", 0.0) or 0.0) - exit_time_sec) < 60.0:
                                already_logged = True
                                break
                                
                        if not already_logged:
                            print(f"[Crash Recovery] Discovered missed closed trade on Bybit: {symbol} at exit price {exit_price}")
                            new_bal = bot_state.get("simulated_balance", 0.0)
                            change_pct = ((exit_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
                            raw_return = change_pct if direction == "Bullish" else -change_pct
                            
                            trade_record = {
                                "symbol": symbol,
                                "entry_time": float(item.get("createdTime") or int(time.time() * 1000)),
                                "exit_time": exit_time_sec,
                                "interval": "Unknown",
                                "direction": direction,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "change_pct": raw_return,
                                "success": closed_pnl > 0,
                                "reason": "CLOSED WHILE OFFLINE / RECOVERY SCAN",
                                "position_size_usd": float(item.get("qty", 0.0)) * entry_price,
                                "original_size": float(item.get("qty", 0.0)) * entry_price,
                                "pnl_usd": closed_pnl,
                                "balance": new_bal,
                                "leverage": 1.0,
                                "confidence": 0.0,
                                "take_profit": 0.0,
                                "stop_loss": 0.0,
                                "atr_dollars": 0.0,
                                "fill_pct": 100.0,
                                "bybit_order_id": "RECOVERED",
                                "bybit_scale_out_order_id": "RECOVERED"
                            }
                            
                            bot_state["trade_history"].append(trade_record)
                            save_history()
                            log_trade_journal(trade_record)
                            
                            emoji = "🚀" if closed_pnl > 0 else "🔴"
                            status_str = "*TAKE PROFIT HIT (RECOVERED)*" if closed_pnl > 0 else "*STOP LOSS HIT (RECOVERED)*"
                            send_telegram_alert(
                                f"{emoji} {status_str} 💻\n"
                                f"• *Asset*: {symbol}\n"
                                f"• *Status*: Trade closed while bot container was offline/restarting.\n"
                                f"• *Direction*: {direction}\n"
                                f"• *Entry Price*: ${entry_price:.2f}\n"
                                f"• *Exit Price*: ${exit_price:.2f}\n"
                                f"• *Realized PnL*: *${closed_pnl:+.2f}*\n"
                                f"• *Exit Time*: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(exit_time_sec))}"
                            )
        except Exception as err:
            print(f"[Crash Recovery Scan Error] for {symbol}: {err}")

def execute_bybit_trade_async(*args, **kwargs):
    symbol = args[0] if args else "Unknown"
    journal_rec = kwargs.get("journal_rec")
    try:
        _execute_bybit_trade_async_inner(*args, **kwargs)
    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"[CRITICAL EXECUTION ERROR] Async trade execution failed for {symbol}: {err_msg}")
        traceback.print_exc()
        send_telegram_alert(f"🚨 *Async Trade Execution Error* 🚨\n• Symbol: `{symbol}`\n• Error: `{err_msg}`")
        if journal_rec is not None:
            try:
                journal_rec.outcome = "SKIPPED"
                journal_rec.reject_reason = f"Async execution exception: {err_msg}"
                journal_rec.trade_id = None
                write_decision(journal_rec)
            except Exception as _wj_err:
                log_event("WARNING", f"Failed to journal async execution exception: {_wj_err}")
    finally:
        with active_execution_lock:
            active_execution_symbols.discard(symbol)
            active_execution_margins.pop(symbol, None)
            active_execution_notional.pop(symbol, None)
            active_execution_risks.pop(symbol, None)
            bot_state["in_flight_risk_usd"] = sum(active_execution_risks.values())

def _execute_bybit_trade_async_inner(symbol, iv, tf, ml_trend, leverage_val, qty_str, raw_qty, entry_price, stop_loss_price, take_profit_price, position_size_usd, kelly_fraction, calibrated_confidence, ml_confidence, dynamic_conf_threshold, latest_completed_ts, latest_candle, pred_change, predicted_price, atr_dollars, tp_multiplier_adjusted, sl_multiplier_adjusted, df_completed, trade_uuid, duration_seconds, active_trade_key, is_oversized=False, intended_size_usd=None, decision_ts=None, journal_rec=None):

    if latest_candle is None:
        latest_candle = {}
    if df_completed is None:
        df_completed = pd.DataFrame()
        
    bybit_success = True
    bybit_order_id = None
    bybit_scale_out_order_id = None
    actual_qty = raw_qty
    order_res = {}

    def _abort_async(reason: str, reason_code: Optional[Any] = None):
        if journal_rec is not None:
            try:
                journal_rec.outcome = "SKIPPED"
                journal_rec.reject_reason = reason
                if reason_code is not None:
                    journal_rec.reason_code = reason_code.value if hasattr(reason_code, "value") else str(reason_code)
                journal_rec.trade_id = None
                write_decision(journal_rec)
                log_event("INFO", f"[{symbol} {iv}m] Journalled async skipped decision: {reason}")
            except Exception as _w_err:
                log_event("WARNING", f"Failed to journal async abort ({reason}): {_w_err}")
    
    # 0. Signal TTL Guard (Hard Abort if decision was made too long ago)
    signal_ttl_seconds = min(120.0, max(30.0, int(iv) * 60 * 0.10))
    if decision_ts is not None:
        elapsed_since_decision = time.time() - float(decision_ts)
        if elapsed_since_decision > signal_ttl_seconds:
            log_event("WARNING", f"[{symbol} {iv}m Signal TTL] Decision expired ({elapsed_since_decision:.1f}s > {signal_ttl_seconds:.1f}s). Aborting order submission.")
            send_telegram_alert(f"⚠️ [{symbol} {iv}m] Order aborted: Signal TTL expired ({elapsed_since_decision:.1f}s > {signal_ttl_seconds:.1f}s).")
            _abort_async(f"Signal TTL expired ({elapsed_since_decision:.1f}s > {signal_ttl_seconds:.1f}s)")
            return

    sl_source = "STRUCTURAL_SWING" if str(iv) in ["15", "30", "60"] else "ATR_DYNAMIC"
    sl_override_reason = "Dynamic Pivot Envelope" if str(iv) in ["15", "30", "60"] else f"{sl_multiplier_adjusted:.2f}x ATR Target"
    min_sl_cfg = getattr(config, "MIN_SL_PCT_CONFIG", {})
    min_sl_pct = float(min_sl_cfg.get(str(iv), min_sl_cfg.get("default", 0.008)))
    atr_sl_dist = atr_dollars * sl_multiplier_adjusted
    min_sl_dist = max(atr_dollars * 1.0, entry_price * min_sl_pct)
    raw_sl_dist = abs(entry_price - stop_loss_price) if stop_loss_price is not None else atr_sl_dist
    # Finding R48 & #16: Pre-order minimum-stop floor enforcement with R:R preservation
    effective_sl_dist = max(raw_sl_dist, min_sl_dist)
    if stop_loss_price is not None:
        stop_loss_price = (entry_price - effective_sl_dist) if ml_trend == "Bullish" else (entry_price + effective_sl_dist)

    # Invariant: Rescale take_profit_price if minimum stop floor widened the stop distance, preserving target R:R
    if effective_sl_dist > (raw_sl_dist + 1e-6) and raw_sl_dist > 0 and take_profit_price is not None:
        raw_tp_dist = abs(take_profit_price - entry_price)
        target_rr = raw_tp_dist / raw_sl_dist
        new_tp_dist = max(raw_tp_dist, effective_sl_dist * target_rr)
        take_profit_price = (entry_price + new_tp_dist) if ml_trend == "Bullish" else (entry_price - new_tp_dist)

    # Invariant: Rescale quantity if minimum stop floor widened the stop distance, preserving dollar risk
    if effective_sl_dist > (raw_sl_dist + 1e-6) and raw_sl_dist > 0:
        rescaled_raw_qty = raw_qty * (raw_sl_dist / effective_sl_dist)
        rescaled_qty_str = format_bybit_qty(symbol, rescaled_raw_qty)
        rescaled_val = float(rescaled_qty_str) if rescaled_qty_str else 0.0
        min_order_value = float(getattr(config, "MIN_ORDER_VALUE_USD", 5.0))
        if rescaled_val * entry_price < min_order_value:
            log_event("WARNING", f"[{symbol} {iv}m Pre-Order Floor] Widening SL ({raw_sl_dist:.2f} -> {effective_sl_dist:.2f}) rescales qty below min notional (${min_order_value:.2f}). Aborting entry.")
            _abort_async("Widened SL reduced order below min notional")
            return
        log_event("INFO", f"[{symbol} {iv}m Pre-Order Floor] Rescaled qty ({qty_str} -> {rescaled_qty_str}) for widened SL ({raw_sl_dist:.2f} -> {effective_sl_dist:.2f}) to maintain dollar risk invariant.")
        raw_qty = rescaled_val
        actual_qty = rescaled_val
        qty_str = rescaled_qty_str
        position_size_usd = (raw_qty * entry_price) / max(1.0, leverage_val)
    
    # 1. Live Exchange Position Guard
    try:
        pos_list = get_all_bybit_positions()
        if pos_list:
            existing_pos = next((p for p in pos_list if p.get("symbol") == symbol and float(p.get("size", "0")) > 0), None)
            if existing_pos:
                print(f"[{symbol} {iv}m API Block] Live order placement skipped: a live position already exists on Bybit.")
                sync_active_positions_from_bybit()
                _abort_async("Live position already exists on Bybit")
                return
    except Exception as pos_check_err:
        print(f"[{symbol} {iv}m API Warning] Live Position Guard check failed: {pos_check_err}")

    # Finding #154: Defensive validation for invalid entry_price / current_price to prevent ZeroDivisionError
    import math
    if entry_price is None or (isinstance(entry_price, float) and (math.isnan(entry_price) or entry_price <= 0)) or float(entry_price) <= 0:
        log_event("WARNING", f"[{symbol} {iv}m] Invalid entry price: {entry_price}. Aborting trade execution.")
        _abort_async(f"Invalid entry price: {entry_price}")
        return

    # 1. Pre-Flight Geometry Assertion (Hard Abort — Do NOT place order if invalid)
    try:
        trade_calculators.assert_valid_geometry(ml_trend, entry_price, stop_loss_price, take_profit_price, symbol=f"{symbol} {iv}m")
    except ValueError as geom_err:
        log_event("ERROR", str(geom_err))
        send_telegram_alert(f"🚨 *CRITICAL ORDER ABORT*: {symbol} {iv}m invalid geometry: SL={stop_loss_price}, Entry={entry_price}, TP={take_profit_price}")
        _abort_async(f"Invalid geometry: {geom_err}")
        return

    # 2. Pre-Flight Horizon Reachability Guard
    import math
    from config import TIMEFRAME_CONFIG
    cfg = TIMEFRAME_CONFIG.get(str(iv), {})
    lookahead = cfg.get("lookahead", 10)
    reach_factor = getattr(config, "HORIZON_REACHABILITY_FACTOR", 1.5)
    max_reachable = math.sqrt(lookahead) * atr_dollars * reach_factor
    preflight_tp_dist = abs(take_profit_price - entry_price)
    if preflight_tp_dist > max_reachable:
        log_event("WARNING", f"[{symbol} {iv}m Reachability Guard] Pre-flight TP distance (${preflight_tp_dist:.4f}) exceeds horizon reach (${max_reachable:.4f}). Aborting trade entry.")
        send_telegram_alert(f"⚠️ [{symbol} {iv}m] Trade aborted pre-flight — TP target (${preflight_tp_dist:.4f}) exceeds horizon reach (${max_reachable:.4f}).")
        _abort_async(f"TP target (${preflight_tp_dist:.4f}) exceeds horizon reach (${max_reachable:.4f})")
        return

    # 3. Pre-Flight Economic Gate (Realized R:R with haircut)
    from trade_calculators import passes_economic_gate, calculate_required_p
    if not passes_economic_gate(entry=entry_price, tp=take_profit_price, sl=stop_loss_price, conf=calibrated_confidence):
        _sl_dist = abs(entry_price - stop_loss_price)
        _required_p = calculate_required_p(entry=entry_price, tp=take_profit_price, sl=stop_loss_price)
        log_event("WARNING", f"[{symbol} {iv}m Pre-Flight Economic Gate] Realized R:R gate failed (nominal R:R {preflight_tp_dist/max(1e-9, _sl_dist):.2f} with haircut requires {_required_p:.3f}, have {calibrated_confidence:.3f}). Aborting.")
        send_telegram_alert(f"⚠️ [{symbol} {iv}m] Trade aborted pre-flight — Realized R:R requires {_required_p:.3f}, have {calibrated_confidence:.3f}")
        _abort_async(f"Realized R:R requires {_required_p:.3f}, have {calibrated_confidence:.3f}")
        return

    pre_entry_price = float(entry_price)
    pre_sl_dist = abs(pre_entry_price - stop_loss_price) if stop_loss_price is not None else (sl_multiplier_adjusted * atr_dollars)
    pre_tp_dist = abs(take_profit_price - pre_entry_price) if take_profit_price is not None else (tp_multiplier_adjusted * atr_dollars)

    # 4. Pre-Flight Adverse Price-Drift Check (Live Mid vs Stale Candle Close Entry)
    live_bid, live_ask, live_last = get_bybit_bid_ask(symbol)
    live_mid = (live_bid + live_ask) / 2.0 if (live_bid is not None and live_ask is not None and live_bid > 0 and live_ask > 0) else (live_last if live_last and live_last > 0 else None)
    if live_mid is None or live_mid <= 0:
        log_event("WARNING", f"[{symbol} {iv}m Adverse Drift Guard] Cannot retrieve live market price. Aborting order (Fail-Closed).")
        send_telegram_alert(f"⚠️ [{symbol} {iv}m] Order aborted: Market price unavailable for pre-flight drift check.")
        _abort_async("Market price unavailable for pre-flight drift check")
        return
    # 4. Immediate Trigger Invariant: Abort if live price has already breached Stop Loss
    if stop_loss_price is not None:
        if ml_trend == "Bullish" and live_mid <= stop_loss_price:
            log_event("WARNING", f"[{symbol} {iv}m Immediate Trigger Guard] Live mid (${live_mid:.4f}) already breached Long Stop Loss (${stop_loss_price:.4f}). Aborting order placement.")
            send_telegram_alert(f"⚠️ [{symbol} {iv}m] Order aborted: Live price (${live_mid:.4f}) already breached Stop Loss (${stop_loss_price:.4f}).")
            _abort_async(f"Live price (${live_mid:.4f}) already breached Long Stop Loss (${stop_loss_price:.4f})")
            return
        elif ml_trend == "Bearish" and live_mid >= stop_loss_price:
            log_event("WARNING", f"[{symbol} {iv}m Immediate Trigger Guard] Live mid (${live_mid:.4f}) already breached Short Stop Loss (${stop_loss_price:.4f}). Aborting order placement.")
            send_telegram_alert(f"⚠️ [{symbol} {iv}m] Order aborted: Live price (${live_mid:.4f}) already breached Stop Loss (${stop_loss_price:.4f}).")
            _abort_async(f"Live price (${live_mid:.4f}) already breached Short Stop Loss (${stop_loss_price:.4f})")
            return

    max_adverse_drift = max(0.25 * atr_dollars, pre_entry_price * 0.0025)

    if ml_trend == "Bullish" and (pre_entry_price - live_mid) > max_adverse_drift:
        adverse_pts = pre_entry_price - live_mid
        log_event("WARNING", f"[{symbol} {iv}m Adverse Drift Guard] Live price ({live_mid:.2f}) drifted {adverse_pts:.2f} below entry ({pre_entry_price:.2f}) > max allowed {max_adverse_drift:.2f} (0.25 ATR). Aborting order.")
        send_telegram_alert(f"⚠️ [{symbol} {iv}m] Order aborted: Adverse price drift ({adverse_pts:.2f} > {max_adverse_drift:.2f}).")
        _abort_async(f"Adverse price drift ({adverse_pts:.2f} > {max_adverse_drift:.2f})")
        return
    elif ml_trend == "Bearish" and (live_mid - pre_entry_price) > max_adverse_drift:
        adverse_pts = live_mid - pre_entry_price
        log_event("WARNING", f"[{symbol} {iv}m Adverse Drift Guard] Live price ({live_mid:.2f}) drifted {adverse_pts:.2f} above entry ({pre_entry_price:.2f}) > max allowed {max_adverse_drift:.2f} (0.25 ATR). Aborting order.")
        send_telegram_alert(f"⚠️ [{symbol} {iv}m] Order aborted: Adverse price drift ({adverse_pts:.2f} > {max_adverse_drift:.2f}).")
        _abort_async(f"Adverse price drift ({adverse_pts:.2f} > {max_adverse_drift:.2f})")
        return

    print(f"[{symbol} {iv}m API] Preparing to open live position on Bybit ({TRADE_MODE.upper()})...")
    leverage_ok = set_bybit_leverage(symbol, leverage_val)
    if leverage_ok:
        side = "Buy" if ml_trend == "Bullish" else "Sell"
        bybit_success = False
        
        rolling_atr = df_completed["ATR_norm"].tail(30)
        atr_mean = rolling_atr.mean()
        atr_std = rolling_atr.std()
        vol_z_score = (latest_candle["ATR_norm"] - atr_mean) / (atr_std + 1e-8) if atr_std > 0 else 0.0
        is_extreme_volatility = vol_z_score > 3.0
        
        if is_extreme_volatility:
            print(f"[{symbol} {iv}m API] Volatility Spiked! Z-score: {vol_z_score:.2f} > 3.0. Placing Taker IOC order immediately...")
            from order_state_machine import generate_client_order_id
            ioc_order_link_id = generate_client_order_id(symbol, side, interval=str(iv), candle_ts=int(latest_completed_ts))
            order_res = place_bybit_taker_ioc_order(symbol, side, qty_str, sl=stop_loss_price, tp=take_profit_price, order_link_id=ioc_order_link_id)
            if order_res.get("retCode") == 0:
                bybit_order_id = order_res.get("result", {}).get("orderId")
                bybit_success = True
                time.sleep(0.5)
                order_details = get_bybit_order_details(symbol, bybit_order_id)
                if order_details:
                    entry_price = float(order_details.get("avgPrice", entry_price))
                    actual_qty = float(order_details.get("cumExecQty", 0.0))
                else:
                    # Finding #77: Query execution list specifically for bybit_order_id on API read failure
                    execs = get_bybit_order_executions(symbol, order_id=bybit_order_id) if bybit_order_id else []
                    if execs:
                        fill_q = sum(float(x.get("execQty", 0.0)) for x in execs)
                        sum_val = sum(float(x.get("execQty", 0.0)) * float(x.get("execPrice", entry_price)) for x in execs)
                        actual_qty = fill_q
                        if fill_q > 0:
                            entry_price = sum_val / fill_q
                    else:
                        pos_res = get_bybit_position(symbol)
                        if pos_res and isinstance(pos_res, dict):
                            sym_pos = pos_res
                        else:
                            pos_list = get_all_bybit_positions()
                            sym_pos = next((p for p in pos_list if isinstance(p, dict) and p.get("symbol") == symbol), {}) if isinstance(pos_list, list) else {}
                        pos_qty = float(sym_pos.get("size", 0.0))
                        actual_qty = min(pos_qty, raw_qty) if pos_qty > 0 else 0.0
                if actual_qty <= 0.0:
                    bybit_success = False
            else:
                print(f"[{symbol} {iv}m API ERROR] Taker IOC order failed: {order_res.get('retMsg')}")
        else:
            # Normal Volatility: Place Limit Maker entry order with dynamic price chasing
            filled_so_far = 0.0
            weighted_sum_px = 0.0
            chase_order_ids = set()
            recorded_chase_exec_ids = set()
            credited_per_chase_order = {}
            
            for chase in range(5):
                if decision_ts is not None:
                    elapsed = time.time() - float(decision_ts)
                    if elapsed > signal_ttl_seconds:
                        print(f"[{symbol} {iv}m API] Signal TTL expired during chase ({elapsed:.1f}s > {signal_ttl_seconds:.1f}s). Aborting remaining chase iterations.")
                        if filled_so_far >= (0.95 * raw_qty):
                            bybit_success = True
                            break
                        elif filled_so_far > 0:
                            break
                        else:
                            log_event("WARNING", f"[{symbol} {iv}m API] Signal TTL expired during chase with 0 fills. Aborting order placement completely.")
                            send_telegram_alert(f"⚠️ [{symbol} {iv}m] Order aborted: Signal TTL expired during chase.")
                            _abort_async("Signal TTL expired during chase with 0 fills")
                            return

                if chase > 0:
                    c_bid, c_ask, c_last = get_bybit_bid_ask(symbol)
                    c_mid = (c_bid + c_ask) / 2.0 if (c_bid is not None and c_ask is not None and c_bid > 0 and c_ask > 0) else (c_last if c_last and c_last > 0 else 0.0)
                    if c_mid <= 0:
                        log_event("WARNING", f"[{symbol} {iv}m API] Live price unavailable before chase {chase+1}. Aborting remaining chase.")
                        if filled_so_far >= (0.95 * raw_qty):
                            bybit_success = True
                        break
                    # Immediate Trigger Invariant check (Finding #55)
                    if (ml_trend == "Bullish" and c_mid <= stop_loss_price) or (ml_trend == "Bearish" and c_mid >= stop_loss_price):
                        log_event("WARNING", f"[{symbol} {iv}m API] Live price ({c_mid:.2f}) breached stop ({stop_loss_price:.2f}) before chase {chase+1}. Aborting chase.")
                        if filled_so_far >= (0.95 * raw_qty):
                            bybit_success = True
                        break
                    # Adverse drift check (Finding #55)
                    if abs(c_mid - entry_price) > max(0.25 * atr_dollars, entry_price * 0.0025):
                        log_event("WARNING", f"[{symbol} {iv}m API] Live price drifted adversely ({c_mid:.2f} vs entry {entry_price:.2f}) before chase {chase+1}. Aborting chase.")
                        if filled_so_far >= (0.95 * raw_qty):
                            bybit_success = True
                        break

                remaining_qty = max(0.0, raw_qty - filled_so_far)
                if remaining_qty <= 0:
                    bybit_success = True
                    break
                    
                import math
                _step_res = get_bybit_min_qty_step(symbol)
                min_q, step_q = _step_res if isinstance(_step_res, (tuple, list)) else (float(_step_res), float(_step_res))
                floored_remaining = math.floor(remaining_qty / step_q + 1e-9) * step_q if step_q > 0 else remaining_qty
                if floored_remaining < min_q:
                    if filled_so_far >= (0.95 * raw_qty):
                        bybit_success = True
                    break
                chase_qty_str = format_bybit_qty(symbol, floored_remaining)

                limit_entry_price = get_chase_limit_price(symbol, side, chase, entry_price)
                print(f"[{symbol} {iv}m API] Chase {chase+1}/5: Placing Limit Maker order for {chase_qty_str} (remaining of {raw_qty}) at {limit_entry_price:.2f}...")
                chase_order_link_id = f"c_{symbol[:5]}_{iv}_{int(latest_completed_ts//1000)}_{chase}"[:36]
                order_res = place_bybit_limit_order(
                    symbol, side, chase_qty_str, limit_entry_price,
                    sl=stop_loss_price, tp=take_profit_price, post_only=True,
                    order_link_id=chase_order_link_id
                )
                if journal_rec is not None:
                    try:
                        journal_rec.order_payload_json = json.dumps({
                            "symbol": symbol, "side": side, "qty": str(chase_qty_str),
                            "price": limit_entry_price, "orderLinkId": chase_order_link_id
                        })
                        journal_rec.venue_response_json = json.dumps(order_res)
                    except Exception as _ex_pj:
                        log_event("WARNING", f"Error serializing order payload for journal: {_ex_pj}")
                
                if order_res.get("retCode") == 0:
                    bybit_order_id = order_res.get("result", {}).get("orderId")
                    if bybit_order_id:
                        chase_order_ids.add(bybit_order_id)
                    _w_res = wait_for_order_fill(symbol, bybit_order_id, timeout_sec=2.0)
                    if isinstance(_w_res, (tuple, list)) and len(_w_res) >= 4:
                        is_filled, f_status, f_cum, f_px = _w_res[:4]
                    else:
                        is_filled = bool(_w_res)
                        f_status = "Filled" if is_filled else "Unknown"
                        f_cum = floored_remaining if is_filled else 0.0
                        f_px = limit_entry_price
                    if is_filled or (raw_qty > 0 and f_cum >= (0.95 * raw_qty)):
                        bybit_success = True
                        if bybit_order_id:
                            cancel_bybit_order(symbol, bybit_order_id)
                        else:
                            cancel_bybit_order(symbol, order_link_id=chase_order_link_id)
                        time.sleep(0.2)
                        fill_px = f_px if f_px > 0 else limit_entry_price
                        fill_q = f_cum if f_cum > 0 else floored_remaining
                        filled_so_far += fill_q
                        weighted_sum_px += (fill_q * fill_px)
                        if bybit_order_id:
                            credited_per_chase_order[bybit_order_id] = credited_per_chase_order.get(bybit_order_id, 0.0) + fill_q
                        break
                    else:
                        print(f"[{symbol} {iv}m API] Order {bybit_order_id} not filled within 2.0s (Status: {f_status}, Fill: {f_cum}). Cancelling and recalculating price...")
                        if bybit_order_id:
                            cancel_res = cancel_bybit_order(symbol, bybit_order_id)
                        else:
                            cancel_res = cancel_bybit_order(symbol, order_link_id=chase_order_link_id)
                        time.sleep(0.3)
                        post_cancel_details = get_bybit_order_details(symbol, bybit_order_id) if bybit_order_id else get_bybit_order_details(symbol, order_link_id=chase_order_link_id)
                        if post_cancel_details:
                            status = post_cancel_details.get("orderStatus")
                            cum_qty = float(post_cancel_details.get("cumExecQty", 0.0))
                            avg_px = float(post_cancel_details.get("avgPrice", limit_entry_price))
                            
                            if cum_qty > 0:
                                filled_so_far += cum_qty
                                weighted_sum_px += (cum_qty * avg_px)
                                if bybit_order_id:
                                    credited_per_chase_order[bybit_order_id] = credited_per_chase_order.get(bybit_order_id, 0.0) + cum_qty
                                print(f"[{symbol} {iv}m API] Order {bybit_order_id} executed {cum_qty} (Total filled so far: {filled_so_far:.4f}/{raw_qty:.4f}).")

                            if status == "Filled" or (raw_qty > 0 and filled_so_far >= (0.95 * raw_qty)):
                                print(f"[{symbol} {iv}m API] Order {bybit_order_id} reached target fill threshold. Completing chase.")
                                bybit_success = True
                                break
                            elif status in ["New", "PartiallyFilled"]:
                                print(f"[{symbol} {iv}m API WARNING] Order {bybit_order_id} still {status} after cancel. Retrying cancellation...")
                                if bybit_order_id:
                                    cancel_bybit_order(symbol, bybit_order_id)
                                else:
                                    cancel_bybit_order(symbol, order_link_id=chase_order_link_id)
                                time.sleep(0.3)
                                recheck_details = get_bybit_order_details(symbol, bybit_order_id) if bybit_order_id else get_bybit_order_details(symbol, order_link_id=chase_order_link_id)
                                if recheck_details:
                                    r_status = recheck_details.get("orderStatus")
                                    r_cum = float(recheck_details.get("cumExecQty", 0.0))
                                    delta_cum = max(0.0, r_cum - cum_qty)
                                    if delta_cum > 0:
                                        filled_so_far += delta_cum
                                        weighted_sum_px += (delta_cum * float(recheck_details.get("avgPrice", limit_entry_price)))
                                        if bybit_order_id:
                                            credited_per_chase_order[bybit_order_id] = credited_per_chase_order.get(bybit_order_id, 0.0) + delta_cum
                                    if r_status in ["New", "PartiallyFilled"]:
                                        log_event("CRITICAL", f"[{symbol} {iv}m API] Order {bybit_order_id} remains active ({r_status}) after cancel retry! Aborting chase loop to prevent double-order position stacking.")
                                        send_telegram_alert(f"🚨 *CHASE ABORT: UNCONFIRMED CANCEL* 🚨\n• Symbol: `{symbol}`\n• Order: `{bybit_order_id}`\n• Status: `{r_status}`\n• Action: Aborted chase loop (Fail-Closed).")
                                        break
                                elif cancel_res and isinstance(cancel_res, dict) and cancel_res.get("retCode") != 0 and "Order not exists" not in str(cancel_res.get("retMsg")):
                                    log_event("CRITICAL", f"[{symbol} {iv}m API] Order {bybit_order_id} cancel confirmation failed! Aborting chase loop to prevent double-order position stacking.")
                                    break
                        else:
                            # Fallback: check execution history if order details returned None (rate limit / indexing lag)
                            try:
                                exec_res = bybit_get_request("/v5/execution/list", {"category": "linear", "symbol": symbol, "orderId": bybit_order_id, "limit": 5})
                                if exec_res and exec_res.get("retCode") == 0:
                                    e_list = exec_res.get("result", {}).get("list", [])
                                    if e_list:
                                        e_qty = sum(float(x.get("execQty", 0.0)) for x in e_list)
                                        e_px = sum(float(x.get("execQty", 0.0)) * float(x.get("execPrice", limit_entry_price)) for x in e_list) / max(1e-9, e_qty)
                                        if e_qty > 0:
                                            filled_so_far += e_qty
                                            weighted_sum_px += (e_qty * e_px)
                                            if bybit_order_id:
                                                credited_per_chase_order[bybit_order_id] = credited_per_chase_order.get(bybit_order_id, 0.0) + e_qty
                                            for x in e_list:
                                                if x.get("execId"):
                                                    recorded_chase_exec_ids.add(x.get("execId"))
                                            print(f"[{symbol} {iv}m API Fallback] Recovered execution fill {e_qty} for {bybit_order_id}.")
                            except Exception as ex_e_hist:
                                log_event("WARNING", f"[{symbol} {iv}m API Fallback] Execution history check error: {ex_e_hist}")

                            if cancel_res and isinstance(cancel_res, dict) and cancel_res.get("retCode") != 0 and "Order not exists" not in str(cancel_res.get("retMsg")):
                                log_event("CRITICAL", f"[{symbol} {iv}m API] Order {bybit_order_id} cancellation returned unconfirmed error ({cancel_res.get('retMsg')}) and details unavailable! Aborting chase to prevent stacking.")
                                break
                elif order_res.get("retCode") == 10006 or "PostOnly" in str(order_res.get("retMsg")):
                    print(f"[{symbol} {iv}m API] Maker PostOnly would cross spread. Retrying next chase tick...")
                else:
                    print(f"[{symbol} {iv}m API ERROR] Limit order placement failed: {order_res.get('retMsg')}")
                    # Reconcile: Query Bybit realtime orders by orderLinkId in case request was executed on exchange
                    try:
                        chk = bybit_get_request("/v5/order/realtime", {"category": "linear", "symbol": symbol, "orderLinkId": chase_order_link_id})
                        if chk.get("retCode") == 0 and chk.get("result", {}).get("list"):
                            o_item = chk["result"]["list"][0]
                            o_id = o_item.get("orderId")
                            if o_id:
                                chase_order_ids.add(o_id)
                                cancel_bybit_order(symbol, o_id)
                                time.sleep(0.3)
                    except Exception as chk_ex:
                        log_event("WARNING", f"[{symbol} {iv}m] orderLinkId reconciliation error: {chk_ex}")
                    break
                    
            if filled_so_far > 0:
                actual_qty = filled_so_far
                entry_price = weighted_sum_px / max(1e-9, filled_so_far)
                if filled_so_far >= (0.95 * raw_qty):
                    bybit_success = True

            # Fallback to Taker IOC if Limit Maker chase attempts did not reach completion
            if not bybit_success:
                remaining_qty = max(0.0, raw_qty - filled_so_far)
                _step_res = get_bybit_min_qty_step(symbol)
                min_q, step_q = _step_res if isinstance(_step_res, (tuple, list)) else (float(_step_res), float(_step_res))
                floored_ioc = math.floor(remaining_qty / step_q + 1e-9) * step_q if step_q > 0 else remaining_qty
                ioc_qty_str = format_bybit_qty(symbol, floored_ioc)
                try:
                    ioc_valid = float(ioc_qty_str) >= min_q
                except (ValueError, TypeError):
                    ioc_valid = False

                if not ioc_valid:
                    if filled_so_far > 0:
                        bybit_success = True
                else:
                    # Item 31: Query executions strictly filtered by chase order IDs to prevent undercounts and foreign fill contamination
                    for c_oid in list(chase_order_ids):
                        exec_records = get_bybit_order_executions(symbol, order_id=c_oid)
                        if not exec_records:
                            continue
                        order_tot_qty = 0.0
                        order_weighted_px = 0.0
                        for rec in exec_records:
                            rec_oid = rec.get("orderId")
                            if rec_oid and rec_oid != c_oid:
                                log_event("WARNING", f"Ignoring foreign fill: Foreign/unrelated execution {rec.get('execId')} belongs to {rec_oid} != {c_oid}")
                                continue
                            e_id = rec.get("execId")
                            e_q = float(rec.get("execQty", 0.0))
                            e_p = float(rec.get("execPrice", entry_price))
                            if e_id and e_id not in recorded_chase_exec_ids:
                                recorded_chase_exec_ids.add(e_id)
                            order_tot_qty += e_q
                            order_weighted_px += (e_q * e_p)
                        already_credited = credited_per_chase_order.get(c_oid, 0.0)
                        uncredited_delta = max(0.0, order_tot_qty - already_credited)
                        if uncredited_delta > 0:
                            avg_exec_px = (order_weighted_px / order_tot_qty) if order_tot_qty > 0 else entry_price
                            log_event("INFO", f"[{symbol} {iv}m API] Recovered {uncredited_delta:.4f} uncredited fill for chase order {c_oid} at {avg_exec_px:.2f} before IOC fallback.")
                            fill_q = min(uncredited_delta, max(0.0, raw_qty - filled_so_far))
                            if fill_q > 0:
                                filled_so_far += fill_q
                                weighted_sum_px += (fill_q * avg_exec_px)
                            credited_per_chase_order[c_oid] = already_credited + uncredited_delta

                    if filled_so_far > 0:
                        actual_qty = filled_so_far
                        entry_price = weighted_sum_px / max(1e-9, filled_so_far)
                        if filled_so_far >= (0.95 * raw_qty):
                            bybit_success = True

                    if not bybit_success:
                        # Recalculate remaining IOC quantity after execution checks
                        rem_ioc = max(0.0, raw_qty - filled_so_far)
                        floored_ioc = math.floor(rem_ioc / step_q + 1e-9) * step_q if step_q > 0 else rem_ioc
                        ioc_qty_str = format_bybit_qty(symbol, floored_ioc)
                        
                        # Finding #55: Re-verify fresh price and Immediate Trigger Invariant before IOC
                        ioc_bid, ioc_ask, ioc_last = get_bybit_bid_ask(symbol)
                        ioc_mid = (ioc_bid + ioc_ask) / 2.0 if (ioc_bid is not None and ioc_ask is not None and ioc_bid > 0 and ioc_ask > 0) else (ioc_last if ioc_last and ioc_last > 0 else 0.0)
                        ioc_aborted = False
                        if ioc_mid <= 0:
                            log_event("WARNING", f"[{symbol} {iv}m API] Live price unavailable before fallback IOC. Aborting IOC.")
                            ioc_aborted = True
                        elif (ml_trend == "Bullish" and ioc_mid <= stop_loss_price) or (ml_trend == "Bearish" and ioc_mid >= stop_loss_price):
                            log_event("WARNING", f"[{symbol} {iv}m API] Live price ({ioc_mid:.2f}) breached stop ({stop_loss_price:.2f}) before fallback IOC. Aborting IOC.")
                            ioc_aborted = True
                        elif abs(ioc_mid - entry_price) > max(0.25 * atr_dollars, entry_price * 0.0025):
                            log_event("WARNING", f"[{symbol} {iv}m API] Live price drifted adversely ({ioc_mid:.2f} vs entry {entry_price:.2f}) before fallback IOC. Aborting IOC.")
                            ioc_aborted = True

                        if ioc_aborted:
                            if filled_so_far > 0:
                                actual_qty = filled_so_far
                                entry_price = weighted_sum_px / max(1e-9, filled_so_far)
                                bybit_success = True
                            else:
                                bybit_success = False
                        else:
                            print(f"[{symbol} {iv}m API] Limit Maker chasing exhausted. Executing fallback Taker IOC for remaining {ioc_qty_str}...")
                            # Finding #54: Deterministic order_link_id for IOC
                            ioc_order_link_id = f"ioc_{symbol[:5]}_{iv}_{int(latest_completed_ts//1000)}"[:36]
                            order_res = place_bybit_taker_ioc_order(symbol, side, ioc_qty_str, sl=stop_loss_price, tp=take_profit_price, order_link_id=ioc_order_link_id)
                            if order_res.get("retCode") == 0:
                                bybit_order_id = order_res.get("result", {}).get("orderId")
                                bybit_success = True
                                time.sleep(0.5)
                                order_details = get_bybit_order_details(symbol, order_id=bybit_order_id, order_link_id=ioc_order_link_id)
                                if order_details:
                                    fill_px = float(order_details.get("avgPrice", entry_price))
                                    fill_q = float(order_details.get("cumExecQty", 0.0))
                                else:
                                    # Finding #77: Query execution list specifically for bybit_order_id on API read failure
                                    ioc_execs = get_bybit_order_executions(symbol, order_id=bybit_order_id, order_link_id=ioc_order_link_id)
                                    if ioc_execs:
                                        fill_q = sum(float(x.get("execQty", 0.0)) for x in ioc_execs)
                                        sum_v = sum(float(x.get("execQty", 0.0)) * float(x.get("execPrice", entry_price)) for x in ioc_execs)
                                        fill_px = (sum_v / fill_q) if fill_q > 0 else entry_price
                                    else:
                                        pos_res = get_bybit_position(symbol)
                                        if pos_res and isinstance(pos_res, dict):
                                            sym_pos = pos_res
                                        else:
                                            pos_list = get_all_bybit_positions()
                                            sym_pos = next((p for p in pos_list if isinstance(p, dict) and p.get("symbol") == symbol), {}) if isinstance(pos_list, list) else {}
                                        current_pos_size = float(sym_pos.get("size", 0.0))
                                        pos_delta = max(0.0, current_pos_size - filled_so_far)
                                        fill_q = min(pos_delta, floored_ioc)
                                        fill_px = entry_price
                                
                                filled_so_far += fill_q
                                weighted_sum_px += (fill_q * fill_px)
                                actual_qty = filled_so_far
                                entry_price = weighted_sum_px / max(1e-9, filled_so_far)
                                if filled_so_far <= 0.0:
                                    bybit_success = False
                            else:
                                print(f"[{symbol} {iv}m API ERROR] Fallback Taker IOC order failed: {order_res.get('retMsg')}")
                                # Finding #54: Realtime reconciliation via orderLinkId
                                try:
                                    time.sleep(0.3)
                                    recon_details = get_bybit_order_details(symbol, order_link_id=ioc_order_link_id)
                                    if recon_details:
                                        r_status = recon_details.get("orderStatus")
                                        r_cum = float(recon_details.get("cumExecQty", 0.0))
                                        r_px = float(recon_details.get("avgPrice", entry_price))
                                        r_oid = recon_details.get("orderId")
                                        if r_cum > 0:
                                            filled_so_far += r_cum
                                            weighted_sum_px += (r_cum * r_px)
                                            bybit_order_id = r_oid
                                            bybit_success = True
                                            log_event("INFO", f"[{symbol} {iv}m API] Reconciled IOC fill ({r_cum}) via orderLinkId {ioc_order_link_id}")
                                        if r_status in ["New", "PartiallyFilled"]:
                                            cancel_bybit_order(symbol, r_oid)
                                except Exception as _rec_err:
                                    log_event("WARNING", f"[{symbol} {iv}m API] IOC reconciliation error: {_rec_err}")
                                if filled_so_far >= (0.95 * raw_qty):
                                    actual_qty = filled_so_far
                                    entry_price = weighted_sum_px / max(1e-9, filled_so_far)
                                    bybit_success = True
                                elif filled_so_far > 0:
                                    actual_qty = filled_so_far
                                    entry_price = weighted_sum_px / max(1e-9, filled_so_far)
                                    bybit_success = False
                                else:
                                    bybit_success = False

            # Finding #24: Sweep all chase order IDs on exit to cancel any resting orders
            for chk_oid in list(chase_order_ids):
                try:
                    cancel_bybit_order(symbol, chk_oid)
                except Exception as _cl_ex:
                    log_event("WARNING", f"Error sweeping chase order {chk_oid}: {_cl_ex}")
                        
        min_fill_pct = getattr(config, "MIN_ACCEPTABLE_FILL_PCT", 0.60)
        if raw_qty > 0 and actual_qty > 0:
            fill_ratio = actual_qty / raw_qty
        elif bybit_success and actual_qty > 0:
            fill_ratio = 1.0
        else:
            fill_ratio = 0.0

        if bybit_success and (fill_ratio < min_fill_pct or actual_qty <= 0):
            if actual_qty <= 0:
                log_event("WARNING", f"[{symbol} {iv}m API] Zero executed quantity reported for order. Marking as unfilled.")
                bybit_success = False
            else:
                log_event("WARNING", f"[{symbol} {iv}m API] Fill ratio {fill_ratio*100:.1f}% below {min_fill_pct*100:.0f}% threshold. Reversing partial fill...")
                if getattr(config, "RESIDUAL_ACTION", "CLOSE") == "CLOSE":
                    opp_side = "Sell" if side == "Buy" else "Buy"
                    flatten_ok = emergency_flatten_position(symbol, opp_side, format_bybit_qty(symbol, actual_qty))
                    if not flatten_ok:
                        log_event("CRITICAL", f"[{symbol} {iv}m] Emergency flatten returned False — retaining position with active SL/TP protection.")
                        bybit_success = True
                    else:
                        bybit_success = False

        if bybit_success:
            sl_dist = pre_sl_dist if 'pre_sl_dist' in locals() and pre_sl_dist > 0 else (abs(entry_price - stop_loss_price) if stop_loss_price is not None else (sl_multiplier_adjusted * atr_dollars))
            raw_sl_dist_pre = sl_dist
            # Finding R48 & #16: Enforce minimum-stop floor before submitting to venue with R:R preservation
            if 'min_sl_dist' in locals() and min_sl_dist > 0:
                sl_dist = max(sl_dist, float(min_sl_dist))
            tp_dist = pre_tp_dist if 'pre_tp_dist' in locals() and pre_tp_dist > 0 else (abs(take_profit_price - entry_price) if take_profit_price is not None else (tp_multiplier_adjusted * atr_dollars))
            if sl_dist > (raw_sl_dist_pre + 1e-6) and raw_sl_dist_pre > 0:
                target_rr_venue = tp_dist / raw_sl_dist_pre
                tp_dist = max(tp_dist, sl_dist * target_rr_venue)

            # Finding N2: Enforce Terminal Risk-at-Stop Hard Boundary Assertion on venue SL
            from risk_limits import HARD_MAX_RISK_PER_TRADE_PCT
            current_bal_val = float(bot_state.get("live_balance", bot_state.get("wallet_balance", 80.0)))
            max_allowed_risk_usd = current_bal_val * HARD_MAX_RISK_PER_TRADE_PCT
            actual_risk_at_stop = actual_qty * sl_dist
            if actual_risk_at_stop > max_allowed_risk_usd + 1e-6:
                safe_sl_dist = max_allowed_risk_usd / max(1e-8, actual_qty)
                log_event("CRITICAL", f"[{symbol} {iv}m Terminal Risk Boundary] Venue stop distance (${sl_dist:.2f}) risk (${actual_risk_at_stop:.2f}) exceeds HARD_MAX_RISK_PER_TRADE_PCT (${max_allowed_risk_usd:.2f}). Clamping venue SL distance to ${safe_sl_dist:.2f}.")
                sl_dist = min(sl_dist, safe_sl_dist)

            stop_loss_price = (entry_price - sl_dist) if ml_trend == "Bullish" else (entry_price + sl_dist)
            take_profit_price = (entry_price + tp_dist) if ml_trend == "Bullish" else (entry_price - tp_dist)

            # Set SL/TP on active position on Bybit with return verification and fail-safe flatten
            temp_trade = {"qty": str(actual_qty), "direction": ml_trend}
            sl_ok = update_bybit_stop_loss(symbol, stop_loss_price, active_trade=temp_trade)
            if not sl_ok:
                time.sleep(0.5)
                sl_ok = update_bybit_stop_loss(symbol, stop_loss_price, active_trade=temp_trade)
                if not sl_ok:
                    log_event("CRITICAL", f"[{symbol} {iv}m] Failed to place Stop Loss on Bybit after fill! Emergency flattening position.")
                    send_telegram_alert(
                        f"🚨 *CRITICAL SL FAILURE - EMERGENCY FLATTEN* 🚨\n"
                        f"• *Asset*: {symbol}\n"
                        f"• *Detail*: Stop loss placement failed twice on Bybit after fill. Attempting emergency flatten..."
                    )
                    flatten_side = "Sell" if ml_trend == "Bullish" else "Buy"
                    flatten_ok = emergency_flatten_position(symbol, flatten_side, format_bybit_qty(symbol, actual_qty))
                    if flatten_ok:
                        log_event("INFO", f"[{symbol} {iv}m] Emergency flatten succeeded after SL failure.")
                        _abort_async("Emergency flatten succeeded after SL failure")
                        return
                    else:
                        log_event("CRITICAL", f"[{symbol} {iv}m] Emergency flatten FAILED after SL placement failure! Retaining trade in state with sl_failed=True.")
                        send_telegram_alert(
                            f"🚨 *CRITICAL: EMERGENCY FLATTEN FAILED* 🚨\n"
                            f"• *Asset*: {symbol}\n"
                            f"• *Detail*: Both SL placement and emergency flatten failed. Position retained in active_trades for exit loop recovery."
                        )

            tp_ok = update_bybit_take_profit(symbol, take_profit_price, active_trade=temp_trade)
            if not tp_ok:
                time.sleep(0.5)
                tp_ok = update_bybit_take_profit(symbol, take_profit_price, active_trade=temp_trade)
                if not tp_ok:
                    log_event("WARNING", f"[{symbol} {iv}m] Failed to place Take Profit on Bybit after fill (SL is active).")
            
            # Place scale-out limit order on Bybit using timeframe and trend-adaptive ATR target
            limit_side = "Sell" if ml_trend == "Bullish" else "Buy"
            entry_adx_val = float(latest_candle.get("ADX", 25.0)) if isinstance(latest_candle, dict) and "ADX" in latest_candle else 25.0
            if str(iv) in ["240", "360"]:
                entry_scale_mult = 1.60 if entry_adx_val >= 35.0 else (1.20 if entry_adx_val < 22.0 else 1.40)
            elif str(iv) in ["60", "120"]:
                entry_scale_mult = 1.40 if entry_adx_val >= 35.0 else (1.00 if entry_adx_val < 22.0 else 1.20)
            else:
                entry_scale_mult = 1.20 if entry_adx_val >= 35.0 else (0.80 if entry_adx_val < 22.0 else 1.00)
            limit_price = entry_price + entry_scale_mult * atr_dollars if ml_trend == "Bullish" else entry_price - entry_scale_mult * atr_dollars
            limit_qty_str = format_bybit_qty(symbol, actual_qty * 0.5)
            limit_qty_val = float(limit_qty_str)
            scale_out_val = limit_qty_val * limit_price
            
            if scale_out_val >= 5.0:
                print(f"[{symbol} {iv}m API] Placing scale-out limit order for {limit_qty_str} at ${limit_price:.4f}...")
                limit_res = place_bybit_limit_order(symbol, limit_side, limit_qty_str, limit_price, reduce_only=True)
                if limit_res.get("retCode") == 0:
                    bybit_scale_out_order_id = limit_res.get("result", {}).get("orderId")
                    print(f"[{symbol} {iv}m API] Scale-out limit order placed successfully. Order ID: {bybit_scale_out_order_id}")
                else:
                    print(f"[{symbol} {iv}m API WARNING] Failed to place scale-out limit order: {limit_res.get('retMsg')}")
    else:
        bybit_success = False
        send_telegram_alert(
            f"🔴 *BYBIT LEVERAGE SETTING ERROR* 🔴\n"
            f"• *Asset*: {symbol}\n"
            f"• *Interval*: {iv}m\n"
            f"• *Target Leverage*: {leverage_val}x\n"
            f"• *Detail*: Failed to configure leverage on Bybit."
        )
        _abort_async("Failed to configure leverage on Bybit")
        return
        
    if bybit_success:
        actual_notional_val = float(actual_qty * entry_price)
        actual_margin_usd = float(actual_notional_val / leverage_val) if leverage_val > 0 else float(position_size_usd)
        actual_size_usd = actual_margin_usd
        init_risk_dist = abs(entry_price - stop_loss_price)
        init_reward_dist = abs(take_profit_price - entry_price)
        init_planned_rr = (init_reward_dist / init_risk_dist) if init_risk_dist > 0 else 0.0

        active_trade = {
            "trade_id": f"{symbol}_{trade_uuid}",
            "interval": str(iv),
            "timeframe": str(tf),
            "bybit_order_id": bybit_order_id,
            "bybit_scale_out_order_id": bybit_scale_out_order_id,
            "symbol": symbol,
            "entry_price": float(entry_price),
            "predicted_price": float(predicted_price),
            "stop_loss": float(stop_loss_price),
            "take_profit": float(take_profit_price),
            "initial_stop_loss": float(stop_loss_price),
            "initial_take_profit": float(take_profit_price),
            "stop_state": "INITIAL",
            "stop_state_meta": {
                "stop_state": "INITIAL",
                "stop_version": "v3.2",
                "transition_reason": "InitialTradeOpened",
                "locked_r": 0.0,
                "expected_net_pnl": 0.0,
                "updated_at": time.time()
            },
            "sl_multiplier": float(sl_multiplier_adjusted),
            "tp_multiplier": float(tp_multiplier_adjusted),
            "sl_source": str(locals().get("sl_source", "STRUCTURE")),
            "min_sl_pct": float(locals().get("min_sl_pct", 0.005)),
            "atr_sl_dist": float(locals().get("atr_sl_dist", atr_dollars * sl_multiplier_adjusted)),
            "min_sl_dist": float(locals().get("min_sl_dist", atr_dollars * 0.60)),
            "initial_planned_rr": float(init_planned_rr),
            "entry_regime": str(bot_state.get(f"regime_{symbol}_{iv}", bot_state.get(f"regime_{iv}", "Trending"))),
            "regime": str(bot_state.get(f"regime_{symbol}_{iv}", bot_state.get(f"regime_{iv}", "Trending"))),
            "entry_scale_mult": float(entry_scale_mult) if "entry_scale_mult" in locals() and entry_scale_mult is not None else 1.20,
            "direction": str(ml_trend),
            "end_time": float(time.time() + duration_seconds),
            "entry_time": int(time.time() * 1000),
            "atr_dollars": float(atr_dollars),
            "entry_atr": float(atr_dollars),
            "highest_price": float(entry_price),
            "lowest_price": float(entry_price),
            "swing_low_3b": float(df_completed["low"].tail(3).min()) if (df_completed is not None and not df_completed.empty and "low" in df_completed.columns) else float(entry_price),
            "swing_high_3b": float(df_completed["high"].tail(3).max()) if (df_completed is not None and not df_completed.empty and "high" in df_completed.columns) else float(entry_price),
            "break_even_triggered": False,
            "half_closed": False,
            "original_size": float(position_size_usd),
            "intended_size_usd": float(intended_size_usd if intended_size_usd is not None else position_size_usd),
            "position_size_usd": actual_size_usd,
            "sl_failed": not bool(sl_ok),
            "tp_failed": not bool(tp_ok),
            "scaled_out_pnl": 0.0,
            "kelly_fraction": float(kelly_fraction),
            "leverage": float(leverage_val),
            "confidence": float(calibrated_confidence),
            "qty": float(actual_qty),
            "original_qty": float(actual_qty),
            "notional_usd": round(float(actual_qty) * float(entry_price), 2),
            "stop_loss_pct": round(abs(float(entry_price) - float(stop_loss_price)) / max(1.0, float(entry_price)), 6),
            "fill_pct": round((actual_qty / raw_qty) * 100.0, 2) if raw_qty > 0 else 100.0,
            "oversized": bool(is_oversized)
        }

        
        with active_trades_lock:
            current_trades = bot_state.get(active_trade_key, [])
            if not isinstance(current_trades, list):
                current_trades = []
            current_trades = list(current_trades)
            current_trades.append(active_trade)
            bot_state[active_trade_key] = current_trades
            
        sync_active_positions_from_bybit()

        if journal_rec is not None:
            try:
                journal_rec.outcome = "EXECUTED"
                journal_rec.trade_id = f"{symbol}_{trade_uuid}"
                journal_rec.reject_reason = None
                journal_rec.position_size_usd = float(actual_margin_usd)
                journal_rec.leverage = float(leverage_val)
                write_decision(journal_rec)
                log_event("INFO", f"[{symbol} {iv}m] Journalled async executed decision for {journal_rec.trade_id}")
            except Exception as _w_exec_err:
                log_event("WARNING", f"Failed to journal async executed decision: {_w_exec_err}")
        
        entry_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        send_telegram_alert(
            f"🟢 *POSITION OPENED (SUCCESSFUL SIGNAL)* 🟢\n"
            f"• *Asset*: {symbol}\n"
            f"• *Interval*: {iv}m\n"
            f"• *Direction*: {ml_trend}\n"
            f"• *Entry Price*: ${float(entry_price):.6f}\n"
            f"• *Entry Time*: {entry_time_str}\n"
            f"• *Take Profit*: ${float(take_profit_price):.6f}\n"
            f"• *Stop Loss*: ${float(stop_loss_price):.6f}\n"
            f"• *SL Source*: {sl_source} ({sl_override_reason})\n"
            f"• *Planned R:R*: {init_planned_rr:.2f}\n"
            f"• *Calibrated Confidence*: {calibrated_confidence * 100:.2f}%\n"
            f"• *Leverage*: {leverage_val:.1f}x\n"
            f"• *Position Size (Margin)*: ${actual_margin_usd:.2f} (Value: ${actual_notional_val:.2f})\n"
            f"• *Execution Mode*: {TRADE_MODE.upper()}"
        )
        print(f"[{symbol} {iv}m SL SOURCE DIAGNOSTIC]")
        print(f"  • ATR Stop Dist: ${atr_sl_dist:.6f}")
        print(f"  • Min Floor Dist: ${min_sl_dist:.6f} ({min_sl_pct:.2f}%)")
        print(f"  • Applied Stop Dist: ${raw_sl_dist:.6f}")
        print(f"  • SL Source: {sl_source} ({sl_override_reason})")
        print(f"[{symbol} {iv}m TRADE SANITY CHECK LOG]")
        print(f"  • ATR: {atr_dollars:.6f}")
        print(f"  • ATR (USD): ${atr_dollars:.6f}")
        print(f"  • sl_mult: {sl_multiplier_adjusted:.2f}")
        print(f"  • tp_mult: {tp_multiplier_adjusted:.2f}")
        print(f"  • Calculated SL (Initial): {stop_loss_price:.6f}")
        print(f"  • Calculated TP (Initial): {take_profit_price:.6f}")
        print(f"  • Initial Planned R:R: {init_planned_rr:.2f}")
        print(f"  • Current Trailing SL: {stop_loss_price:.6f}")
        print(f"  • Current Dashboard SL: {stop_loss_price:.6f}")
        print(f"[{symbol} {iv}m] Trade Opened: {ml_trend} at price {entry_price:.6f} (SL: {stop_loss_price:.6f}, TP: {take_profit_price:.6f}, SL Source: {sl_source}, Planned R:R: {init_planned_rr:.2f})")
    else:
        err_msg = order_res.get('retMsg', "Execution failed")
        err_code = order_res.get('retCode', "N/A")
        send_telegram_alert(
            f"🔴 *BYBIT API ORDER ERROR* 🔴\n"
            f"• *Asset*: {symbol}\n"
            f"• *Interval*: {iv}m\n"
            f"• *Direction*: {ml_trend}\n"
            f"• *Error Message*: {err_msg} (Code: {err_code})"
        )
        _abort_async(f"Bybit API order error: {err_msg} (Code: {err_code})")

def main():
    global live_price, last_ws_update_time
    from datetime import datetime, timezone, timedelta
    # Load model weights here (deferred from module level)
    for iv in ["15", "30", "60", "120", "240"]:
        load_model_weights(iv)
    load_history()
    print(f"{SYMBOL} LIVE BOT RUNNING...")
    send_telegram_alert(f"✅ *Bot started successfully on {TRADE_MODE.upper()} mode.*\n_Container boot detected — all systems nominal._")
    

    # Pre-load initial prices for all supported symbols
    load_initial_prices()

    # Crash Recovery: re-sync orphaned Bybit positions
    sync_active_positions_from_bybit()
    
    # Crash Recovery: scan for closed trades while offline
    recover_missed_closed_trades()

    print("Connecting to WebSocket and waiting for initial price...")

    startup_timeout = 5
    start_wait = time.time()
    while live_price is None:
        if time.time() - start_wait > startup_timeout:
            print("WebSocket connecting... Fetching ticker price from API fallback...")
            fallback = get_fallback_price()
            if fallback is not None:
                live_price = fallback
                bot_state["last_rest_price_time"] = time.time()
            else:
                time.sleep(2)
        time.sleep(0.5)

    print(f"Initial price loaded: {live_price:.2f}")
    print(f"\n==================================================")
    print(f"[+] Local Web Dashboard is running at http://localhost:5001")
    print(f"==================================================\n")
    bot_state["live_price"] = live_price

    # Calculate calibration boundaries at startup for each interval
    tf_map_startup = {"15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h"}
    for iv in ["15", "30", "60", "120", "240"]:
        if iv in models_by_interval:
            p95, max_conf = calculate_historical_thresholds(models_by_interval[iv]["trending"]["trend"], iv)
            tf_key = tf_map_startup[iv]
            bot_state[f"calibration_{tf_key}"] = {
                "p95": p95,
                "max_conf": max_conf,
                "mean": 54.81
            }

    print(f"Starting loop... Checking for new completed candle signals and exit monitoring...")
    bot_state["status"] = "Running"

    last_processed_timestamps = {
        "last_processed_15_ts": None,
        "last_processed_30_ts": None,
        "last_processed_60_ts": None,
        "last_processed_120_ts": None,
        "last_processed_240_ts": None
    }
    startup_check_done = False
    last_check_hour = -1
    last_position_sync_time = 0.0
    completed_this_hour = set()
    hour_check_complete = False
    last_candle_check_time = 0.0

    # Trigger immediate sync at main loop startup
    try:
        sync_active_positions_from_bybit()
    except Exception as ex_init_sync:
        print(f"[Main Loop Startup] Immediate position sync error: {ex_init_sync}")

    while True:
        current_time = time.time()
        
        # Sync active positions from Bybit periodically to save proxy bandwidth
        has_active_positions = any(len(bot_state.get(f"active_trade_{tf}", [])) > 0 for tf in ACTIVE_TRADE_TF_KEYS)
        sync_interval = POSITION_SYNC_INTERVAL_SECS if has_active_positions else POSITION_SYNC_IDLE_INTERVAL_SECS
        
        if (current_time - last_position_sync_time >= sync_interval):
            success = sync_active_positions_from_bybit()
            if success:
                last_position_sync_time = current_time
            else:
                # Retry in 5 seconds
                last_position_sync_time = current_time - sync_interval + 5.0
        
        # 1. Health check & current price update (Adaptive to save proxy bandwidth)
        # Rely on background run_fallback_price_updater. Only query directly if live_price is None
        # or has not been updated in over 10 minutes (600s) as a fail-safe.
        if live_price is None or (current_time - last_ws_update_time > 30.0 and current_time - bot_state.get("last_rest_price_time", 0.0) > 30.0):
            fallback_price = get_fallback_price()
            if fallback_price is not None:
                print(f"[{get_pkt_time().strftime('%H:%M:%S')}] WebSocket/Fallback price is stale or disconnected. Fetching price: {fallback_price:.2f}")
                live_price = fallback_price
                bot_state["last_rest_price_time"] = current_time
            
        current_price = live_price
        if current_price is None:
            print("Could not obtain price. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # 2. Check Exits for each timeframe if a trade is active
        tf_map = {"15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h"}
        active_trades_updated = False
        with active_trades_lock:
            # Institutional Portfolio Utility Optimizer rebalancing across open positions
            all_open_trades = {}
            for iv_k in ["15", "30", "60", "120", "240"]:
                tf_k = f"active_trade_{tf_map[iv_k]}"
                for tr in bot_state.get(tf_k, []) or []:
                    tr_id = tr.get("trade_id")
                    if tr_id:
                        all_open_trades[tr_id] = tr
        rebal_close_set = set()
        rebal_scale_set = set()
        if all_open_trades:
            # Filter out trades younger than 1 candle from portfolio rebalancing to prevent premature scale-outs
            mature_trades = {}
            for tid, tr in all_open_trades.items():
                et_ms = tr.get("entry_time", 0)
                if et_ms and et_ms > 1e12:  # valid ms timestamp
                    age_sec = time.time() - (et_ms / 1000.0)
                elif et_ms and et_ms > 1e9:  # valid sec timestamp
                    age_sec = time.time() - et_ms
                else:
                    age_sec = 0  # unknown age, skip rebalancing
                # Require at least 1 candle of age for the trade's timeframe before rebalancing
                tr_iv = int(tr.get("interval", 60) or 60)
                min_age = tr_iv * 60  # 1 candle in seconds
                if age_sec >= min_age:
                    mature_trades[tid] = tr
            if mature_trades:
                portfolio_rebal = PortfolioUtilityOptimizer.optimize_portfolio_capital(mature_trades)
            else:
                portfolio_rebal = {}
            if isinstance(portfolio_rebal, dict):
                rebal_close_set = set(portfolio_rebal.get("close_trades", []))
                rebal_scale_set = set(portfolio_rebal.get("scale_out_trades", []))

        for iv in ["15", "30", "60", "120", "240"]:
            tf = tf_map[iv]
            active_trade_key = f"active_trade_{tf}"
            with active_trades_lock:
                active_trades_list = bot_state.get(active_trade_key, [])
                if not isinstance(active_trades_list, list):
                    active_trades_list = [] if active_trades_list is None else [active_trades_list]
                    bot_state[active_trade_key] = active_trades_list
                else:
                    active_trades_list = list(active_trades_list)
            
            updated_trades = []
            for active_trade in active_trades_list:
                active_symbol = active_trade.get("symbol", "BTCUSDT")
                now_exit = time.time()
                symbol_price = bot_state.get(f"live_price_{active_symbol}")
                symbol_price_ts = bot_state.get(f"live_price_ts_{active_symbol}", 0.0)
                if symbol_price is None or symbol_price_ts <= 0.0 or (now_exit - symbol_price_ts > 30.0):
                    fresh_price = get_fallback_price(active_symbol)
                    if fresh_price is not None:
                        symbol_price = fresh_price
                        bot_state[f"live_price_{active_symbol}"] = fresh_price
                        bot_state[f"live_price_ts_{active_symbol}"] = now_exit
                        symbol_price_ts = now_exit
                    else:
                        log_event("WARNING", f"[{active_symbol}] Live price stale (age {now_exit - symbol_price_ts:.1f}s) and fallback unavailable. Abstaining from exit evaluation.")
                        updated_trades.append(active_trade)
                        continue
                if symbol_price is None or symbol_price_ts <= 0.0 or (now_exit - symbol_price_ts > 30.0):
                    updated_trades.append(active_trade)
                    continue
                current_price = symbol_price
                
                stop_loss = float(active_trade.get("stop_loss", 0.0))
                take_profit = float(active_trade.get("take_profit", 0.0))
                direction = str(active_trade.get("direction", "Bullish"))
                end_time = float(active_trade.get("end_time", time.time() + 3600))
                entry_price = float(active_trade.get("entry_price", current_price))
                predicted_price = float(active_trade.get("predicted_price", entry_price))

                # Bybit Live position query and state tracking
                bybit_closed = False
                bybit_scaled_out = False
                bybit_exit_price = None
                bybit_realized_pnl = None
                bybit_pnl_data = None
                pnl_source = "SIMULATION" if TRADE_MODE == "simulation" else "ESTIMATED"
                
                if TRADE_MODE != "simulation":
                    # Active recovery for sl_failed / unanchored stop loss
                    if active_trade.get("sl_failed", False):
                        retry_sl_ok = update_bybit_stop_loss(active_symbol, stop_loss, active_trade=active_trade)
                        if retry_sl_ok:
                            active_trade["sl_failed"] = False
                            active_trade["sl_failed_retries"] = 0
                            active_trades_updated = True
                            log_event("INFO", f"[{active_symbol}] Recovered missing Bybit Stop Loss at ${stop_loss:.2f}.")
                        else:
                            retries = active_trade.get("sl_failed_retries", 0) + 1
                            active_trade["sl_failed_retries"] = retries
                            if retries >= 3 and not bybit_closed:
                                log_event("CRITICAL", f"[{active_symbol}] Stop Loss recovery failed 3 times! Triggering emergency flatten to prevent unprotected exposure.")
                                send_telegram_alert(f"🚨 *CRITICAL SL FAILURE - EMERGENCY MARKET CLOSE* 🚨\n• Symbol: {active_symbol}\n• Detail: SL placement failed 3x in exit loop. Closing position via emergency market order.")
                                flatten_side = "Sell" if direction == "Bullish" else "Buy"
                                raw_qty = active_trade.get("qty", 0.0)
                                flatten_ok = emergency_flatten_position(active_symbol, flatten_side, format_bybit_qty(active_symbol, raw_qty))
                                if flatten_ok:
                                    exit_reason = "CRITICAL FAIL-SAFE: UNSTOPPED POSITION CLOSED VIA EMERGENCY FLATTEN"

                    # Active recovery for tp_failed / unanchored take profit
                    if active_trade.get("tp_failed", False):
                        retry_tp_ok = update_bybit_take_profit(active_symbol, take_profit, active_trade=active_trade)
                        if retry_tp_ok:
                            active_trade["tp_failed"] = False
                            active_trade["tp_failed_retries"] = 0
                            active_trades_updated = True
                            log_event("INFO", f"[{active_symbol}] Recovered missing Bybit Take Profit at ${take_profit:.2f}.")
                        else:
                            active_trade["tp_failed_retries"] = active_trade.get("tp_failed_retries", 0) + 1

                    if active_trade.get("bybit_closed", False):
                        bybit_closed = True
                    else:
                        # Detect scale-out fill from cached and stored qty values
                        original_qty = float(active_trade.get("original_qty", active_trade.get("qty", 0.0)))
                        current_qty = float(active_trade.get("qty", 0.0))
                        if original_qty > 0 and current_qty <= (original_qty * 0.6) and not active_trade.get("half_closed", False):
                            # Verify fill status of the scale-out limit order to prevent premature/false triggers
                            scale_out_order_id = active_trade.get("bybit_scale_out_order_id")
                            if scale_out_order_id:
                                order_details = get_bybit_order_details(active_symbol, scale_out_order_id)
                                if order_details and order_details.get("orderStatus") == "Filled":
                                    bybit_scaled_out = True
                                else:
                                    status_msg = order_details.get("orderStatus") if order_details else "Unknown"
                                    stuck_since = active_trade.get("scale_out_stuck_since", time.time())
                                    active_trade.setdefault("scale_out_stuck_since", stuck_since)
                                    
                                    # Timeframe-scaled scale-out timeout
                                    if str(iv) in ["240", "360"]:
                                        max_scale_out_wait = 43200  # 12 hours for 4H/6H swing runners
                                    elif str(iv) in ["60", "120"]:
                                        max_scale_out_wait = 7200   # 2 hours for 1H/2H trades
                                    elif str(iv) == "30":
                                        max_scale_out_wait = 1800   # 30 mins
                                    else:
                                        max_scale_out_wait = 600    # 10 mins for 15m scalps
                                        
                                    if time.time() - stuck_since > max_scale_out_wait:
                                        print(f"[{active_symbol} {iv}m] Scale-out limit order expired after {max_scale_out_wait//60} min ({status_msg}). Cancelling stale order.")
                                        cancel_bybit_order(active_symbol, scale_out_order_id)
                                        active_trade.pop("scale_out_stuck_since", None)
                                        active_trade["bybit_scale_out_order_id"] = None
                                    else:
                                        pass
                            else:
                                # Fallback only if price has actually reached scale-out profit target
                                atr_d = active_trade.get("atr_dollars", 0.015 * entry_price)
                                scale_mult_req = 1.40 if str(iv) in ["240", "360"] else (1.20 if str(iv) in ["60", "120"] else 0.80)
                                reached_scale_target = False
                                if direction == "Bullish" and current_price >= entry_price + scale_mult_req * atr_d:
                                    reached_scale_target = True
                                elif direction == "Bearish" and current_price <= entry_price - scale_mult_req * atr_d:
                                    reached_scale_target = True
                                    
                                if reached_scale_target:
                                    bybit_scaled_out = True
                                else:
                                    active_trade["original_qty"] = current_qty

                # Trailing stop and break-even variables
                atr_dollars = active_trade.get("atr_dollars", 50.0)
                highest_price = active_trade.get("highest_price", entry_price)
                lowest_price = active_trade.get("lowest_price", entry_price)
                break_even_triggered = active_trade.get("break_even_triggered", False)
                position_size_usd = active_trade.get("position_size_usd", 100.0)

                # Volatility-Scaled Trailing Stops & Break-Even via exit_manager module
                if active_trade.get("half_closed", False):
                    trailing_multiplier = 1.0
                else:
                    current_adx = bot_state.get(f"adx_{tf}", 20.0)
                    trailing_multiplier = exit_manager.compute_trailing_multiplier(active_trade, tf, current_adx)

                min_pct_floor = risk_engine.auto_stop_floor.get_floor(active_symbol, database_module=database, interval=str(iv)) if hasattr(risk_engine, 'auto_stop_floor') else volatility_clusterer.get_symbol_break_even_floor(active_symbol)
                be_mult = mfe_be_trigger.get_trigger_multiple(active_symbol, timeframe=str(iv))
                # Finding #19: Coordinate BE trigger distance with champion policy regime threshold
                regime_for_be = active_trade.get("entry_regime") or bot_state.get(f"regime_{active_symbol}_{iv}", bot_state.get(f"regime_{iv}", "RANGING"))
                reg_key_be = exit_policy_engine._resolve_regime_key(str(regime_for_be), float(bot_state.get(f"adx_{active_symbol}_{iv}", 20.0)))
                policy_params_be = exit_policy_engine.active_policy.get(reg_key_be, {}) if hasattr(exit_policy_engine, "active_policy") and exit_policy_engine.active_policy else {}
                policy_be_mult = float(policy_params_be.get("be_trigger_atr_mult", 0.0))
                if policy_be_mult > 0:
                    be_mult = max(be_mult, policy_be_mult)
                trade_leverage = float(active_trade.get("leverage", 1.0))
                required_be_dist = compute_be_trigger_distance(atr_dollars, trade_leverage, iv, be_mult, entry_price, min_pct_floor)

                exit_eval = exit_manager.evaluate_trailing_and_break_even(
                    active_symbol, iv, tf, direction, entry_price, current_price,
                    highest_price, lowest_price, stop_loss, break_even_triggered,
                    atr_dollars, position_size_usd, active_trade, required_be_dist,
                    trailing_multiplier, update_bybit_stop_loss, trade_mode=TRADE_MODE
                )
                highest_price = exit_eval["highest_price"]
                lowest_price = exit_eval["lowest_price"]
                stop_loss = exit_eval["stop_loss"]
                break_even_triggered = exit_eval["break_even_triggered"]
                if exit_eval["active_trades_updated"]:
                    active_trades_updated = True

                # Rule 10: ATR Fibonacci Step-Lock (38.2% -> lock 25%, 50% -> lock 40%, 61.8% -> lock 55%)
                take_profit_val = active_trade.get("take_profit", 0.0)
                total_tp_range = abs(take_profit_val - entry_price)
                if direction == "Bullish":
                    current_move = max(0.0, current_price - entry_price)
                else:
                    current_move = max(0.0, entry_price - current_price)
                progress_pct = (current_move / total_tp_range) if total_tp_range > 0 else 0.0
                
                raw_fib = getattr(config, "FIBONACCI_STEP_LOCKS", {0.618: 0.55, 0.50: 0.40, 0.382: 0.25})
                if isinstance(raw_fib, dict) and "levels" in raw_fib and "locks" in raw_fib:
                    fib_locks = {float(k): float(v) for k, v in zip(raw_fib["levels"], raw_fib["locks"])}
                elif isinstance(raw_fib, dict):
                    fib_locks = {float(k): float(v) for k, v in raw_fib.items() if not isinstance(k, str) or k.replace('.', '', 1).isdigit()}
                else:
                    fib_locks = {0.618: 0.55, 0.50: 0.40, 0.382: 0.25}

                locked_pct = 0.0
                for threshold in sorted(fib_locks.keys(), reverse=True):
                    if progress_pct >= threshold:
                        locked_pct = fib_locks[threshold]
                        break
                    
                if locked_pct > 0.0 and current_move > 0.0:
                    if direction == "Bullish":
                        fib_sl = max(stop_loss, entry_price + (current_price - entry_price) * locked_pct)
                        if fib_sl > stop_loss:
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, fib_sl, active_trade)
                                if success:
                                    stop_loss = fib_sl
                                    active_trade["stop_loss"] = stop_loss
                                    active_trades_updated = True
                                    print(f"[{iv}m Fib Step-Lock] Progress {progress_pct*100:.1f}%. Locked {locked_pct*100:.0f}% profit. SL: {stop_loss:.2f}")
                            else:
                                stop_loss = fib_sl
                                active_trade["stop_loss"] = stop_loss
                                active_trades_updated = True
                                print(f"[{iv}m Fib Step-Lock] Progress {progress_pct*100:.1f}%. Locked {locked_pct*100:.0f}% profit. SL: {stop_loss:.2f}")
                    else:
                        fib_sl = min(stop_loss, entry_price - (entry_price - current_price) * locked_pct)
                        if fib_sl < stop_loss:
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, fib_sl, active_trade)
                                if success:
                                    stop_loss = fib_sl
                                    active_trade["stop_loss"] = stop_loss
                                    active_trades_updated = True
                                    print(f"[{iv}m Fib Step-Lock] Progress {progress_pct*100:.1f}%. Locked {locked_pct*100:.0f}% profit. SL: {stop_loss:.2f}")
                            else:
                                stop_loss = fib_sl
                                active_trade["stop_loss"] = stop_loss
                                active_trades_updated = True
                                print(f"[{iv}m Fib Step-Lock] Progress {progress_pct*100:.1f}%. Locked {locked_pct*100:.0f}% profit. SL: {stop_loss:.2f}")

                # Scale-Out (50% partial profit taking at timeframe & trend-adaptive ATR multiple)
                from config import TAKER_FEE_PCT, SCALE_OUT_CONFIG
                scale_out_mult = float(active_trade.get("entry_scale_mult", 0.0))
                if scale_out_mult <= 0:
                    trade_adx = float(active_trade.get("adx", active_trade.get("entry_adx", 25.0)))
                    if str(iv) in ["240", "360"]:
                        scale_out_mult = 1.60 if trade_adx >= 35.0 else (1.20 if trade_adx < 22.0 else 1.40)
                    elif str(iv) in ["60", "120"]:
                        scale_out_mult = 1.40 if trade_adx >= 35.0 else (1.00 if trade_adx < 22.0 else 1.20)
                    else:
                        scale_out_mult = 1.20 if trade_adx >= 35.0 else (0.80 if trade_adx < 22.0 else 1.00)
                scale_out_portion = SCALE_OUT_CONFIG.get("position_portion", 0.50)
                half_closed = active_trade.get("half_closed", False)
                trigger_scale_out = False
                if not half_closed:
                    if TRADE_MODE != "simulation":
                        trigger_scale_out = bybit_scaled_out
                    else:
                        if active_trade.get("scale_out_triggered"):
                            trigger_scale_out = True
                        elif active_trade.get("trade_id") in rebal_scale_set:
                            trigger_scale_out = True
                            print(f"[{active_symbol} {iv}m Portfolio Rebalance] Scale-out triggered by Portfolio Utility Optimizer.")
                        elif direction == "Bullish" and current_price >= entry_price + scale_out_mult * atr_dollars:
                            trigger_scale_out = True
                        elif direction == "Bearish" and current_price <= entry_price - scale_out_mult * atr_dollars:
                            trigger_scale_out = True

                if trigger_scale_out and not half_closed:
                    if direction == "Bullish":
                        # Scale-Out Triggered for Long
                        half_closed = True
                        active_trade["half_closed"] = True
                        
                        # Close configured portion of the position (derived from original entry margin)
                        orig_margin = float(active_trade.get("original_size", active_trade.get("position_size_usd", position_size_usd)))
                        closed_size = round(orig_margin * scale_out_portion, 2)
                        remaining_size = round(orig_margin - closed_size, 2)
                        
                        # Calculate profit on closed portion (correct taker fee on leveraged size)
                        raw_return_pct = ((current_price - entry_price) / entry_price) * 100.0
                        lev = active_trade.get("leverage", 1.0)
                        gross_pnl = closed_size * (raw_return_pct * lev / 100.0)
                        taker_fee_cost = closed_size * lev * TAKER_FEE_PCT  # exit side only
                        from decimal import Decimal, ROUND_HALF_UP
                        def _q2(v):
                            return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
                        pnl_usd = _q2(gross_pnl - taker_fee_cost)
                        if pnl_usd < -closed_size:
                            pnl_usd = -closed_size
                            net_return_pct = -100.0
                        else:
                            net_return_pct = _q2((pnl_usd / closed_size) * 100.0) if closed_size > 0 else 0.0
                        
                        # Save scaled out pnl and execution metadata
                        active_trade["scaled_out_pnl"] = pnl_usd
                        active_trade["scale_out_price"] = current_price
                        active_trade["scaled_out_margin"] = closed_size
                        
                        # Refund closed size + PnL to wallet balance (only in simulation)
                        if TRADE_MODE == "simulation":
                            bot_state["simulated_balance"] = round(bot_state["simulated_balance"] + closed_size + pnl_usd, 2)
                        
                        # Update position details
                        position_size_usd = remaining_size
                        active_trade["position_size_usd"] = remaining_size
                        
                        # Move stop loss to timeframe-scaled break-even (monotonic non-widening)
                        be_sl = calculate_break_even_stop(direction, entry_price, current_price, atr_dollars, interval=str(iv))
                        target_sl = max(be_sl, stop_loss)
                        if target_sl > stop_loss + 1e-4:
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, target_sl, active_trade, current_sl_snapshot=stop_loss)
                                if success:
                                    stop_loss = target_sl
                                    active_trade["stop_loss"] = target_sl
                                    active_trade["break_even_triggered"] = True
                                    active_trades_updated = True
                                    print(f"[{active_symbol} {iv}m Scale-Out] {int(scale_out_portion*100)}% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to entry: {entry_price:.2f}")
                                    update_bybit_take_profit(active_symbol, take_profit, active_trade)
                                else:
                                    print(f"[{active_symbol} {iv}m Scale-Out ERROR] Failed to update Stop Loss to entry on Bybit. SL remains at {stop_loss:.2f}")
                            else:
                                stop_loss = target_sl
                                active_trade["stop_loss"] = target_sl
                                active_trade["break_even_triggered"] = True
                                active_trades_updated = True
                                print(f"[{active_symbol} {iv}m Scale-Out] {int(scale_out_portion*100)}% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to entry: {entry_price:.2f}")
                        else:
                            active_trade["break_even_triggered"] = True
                        
                    elif direction == "Bearish":
                        # Scale-Out Triggered for Short
                        half_closed = True
                        active_trade["half_closed"] = True
                        
                        # Close configured portion of position (derived from original entry margin)
                        orig_margin = float(active_trade.get("original_size", active_trade.get("position_size_usd", position_size_usd)))
                        closed_size = round(orig_margin * scale_out_portion, 2)
                        remaining_size = round(orig_margin - closed_size, 2)
                        
                        # Calculate profit on closed portion (correct taker fee on leveraged size)
                        raw_return_pct = ((entry_price - current_price) / entry_price) * 100.0
                        lev = active_trade.get("leverage", 1.0)
                        gross_pnl = closed_size * (raw_return_pct * lev / 100.0)
                        taker_fee_cost = closed_size * lev * TAKER_FEE_PCT  # exit side only
                        from decimal import Decimal, ROUND_HALF_UP
                        def _q2_short(v):
                            return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
                        pnl_usd = _q2_short(gross_pnl - taker_fee_cost)
                        if pnl_usd < -closed_size:
                            pnl_usd = -closed_size
                            net_return_pct = -100.0
                        else:
                            net_return_pct = _q2_short((pnl_usd / closed_size) * 100.0) if closed_size > 0 else 0.0
                        
                        # Save scaled out pnl and execution metadata
                        active_trade["scaled_out_pnl"] = pnl_usd
                        active_trade["scale_out_price"] = current_price
                        active_trade["scaled_out_margin"] = closed_size
                        
                        # Refund closed size + PnL to wallet balance (only in simulation)
                        if TRADE_MODE == "simulation":
                            bot_state["simulated_balance"] = round(bot_state["simulated_balance"] + closed_size + pnl_usd, 2)
                        
                        # Update position details
                        position_size_usd = remaining_size
                        active_trade["position_size_usd"] = remaining_size
                        
                        # Move stop loss to fee-adjusted break-even floor (monotonic non-widening)
                        be_sl = calculate_break_even_stop(direction, entry_price, current_price, atr_dollars, interval=str(iv))
                        target_sl = min(be_sl, stop_loss)
                        if target_sl < stop_loss - 1e-4:
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, target_sl, active_trade, current_sl_snapshot=stop_loss)
                                if success:
                                    stop_loss = target_sl
                                    active_trade["stop_loss"] = target_sl
                                    active_trade["break_even_triggered"] = True
                                    active_trades_updated = True
                                    print(f"[{active_symbol} {iv}m Scale-Out] {int(scale_out_portion*100)}% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to fee-adjusted entry: {stop_loss:.2f}")
                                    update_bybit_take_profit(active_symbol, take_profit, active_trade)
                                else:
                                    print(f"[{active_symbol} {iv}m Scale-Out ERROR] Failed to update Stop Loss to fee-adjusted entry on Bybit. SL remains at {stop_loss:.2f}")
                            else:
                                stop_loss = target_sl
                                active_trade["stop_loss"] = target_sl
                                active_trade["break_even_triggered"] = True
                                active_trades_updated = True
                                print(f"[{active_symbol} {iv}m Scale-Out] {int(scale_out_portion*100)}% Profit Locked! Closed: ${closed_size:.2f} at {current_price:.2f} (PnL: {pnl_usd:+.2f}). Remaining size: ${remaining_size:.2f}. SL moved to fee-adjusted entry: {stop_loss:.2f}")
                        else:
                            active_trade["break_even_triggered"] = True
                
                # Tier-2 Runner Profit Ratchet: When half-closed runner reaches >= +1.5x ATR, lock in +0.5x ATR guaranteed profit
                if half_closed and atr_dollars > 0:
                    tier2_trigger = 1.5 * atr_dollars
                    tier2_lock = 0.5 * atr_dollars
                    if direction == "Bullish" and current_price >= entry_price + tier2_trigger:
                        target_tier2_sl = entry_price + tier2_lock
                        if target_tier2_sl > stop_loss:
                            if TRADE_MODE != "simulation":
                                t2_ok = update_bybit_stop_loss(active_symbol, target_tier2_sl, active_trade)
                                if t2_ok:
                                    stop_loss = target_tier2_sl
                                    active_trade["stop_loss"] = target_tier2_sl
                                    active_trade["tier2_profit_locked"] = True
                                    active_trades_updated = True
                                    log_event("INFO", f"[{active_symbol} {iv}m Tier-2 Ratchet] Runner at +1.5x ATR! Guaranteed profit stop trailed to +0.5x ATR (${stop_loss:.2f})")
                            else:
                                stop_loss = target_tier2_sl
                                active_trade["stop_loss"] = target_tier2_sl
                                active_trade["tier2_profit_locked"] = True
                                active_trades_updated = True
                                log_event("INFO", f"[{active_symbol} {iv}m Tier-2 Ratchet] Runner at +1.5x ATR! Guaranteed profit stop trailed to +0.5x ATR (${stop_loss:.2f})")
                    elif direction == "Bearish" and current_price <= entry_price - tier2_trigger:
                        target_tier2_sl = entry_price - tier2_lock
                        if target_tier2_sl < stop_loss:
                            if TRADE_MODE != "simulation":
                                t2_ok = update_bybit_stop_loss(active_symbol, target_tier2_sl, active_trade)
                                if t2_ok:
                                    stop_loss = target_tier2_sl
                                    active_trade["stop_loss"] = target_tier2_sl
                                    active_trade["tier2_profit_locked"] = True
                                    active_trades_updated = True
                                    log_event("INFO", f"[{active_symbol} {iv}m Tier-2 Ratchet] Runner at -1.5x ATR! Guaranteed profit stop trailed to -0.5x ATR (${stop_loss:.2f})")
                            else:
                                stop_loss = target_tier2_sl
                                active_trade["stop_loss"] = target_tier2_sl
                                active_trade["tier2_profit_locked"] = True
                                active_trades_updated = True
                                log_event("INFO", f"[{active_symbol} {iv}m Tier-2 Ratchet] Runner at -1.5x ATR! Guaranteed profit stop trailed to -0.5x ATR (${stop_loss:.2f})")
                
                remaining_seconds = max(0, int(end_time - current_time))
                mins, secs = divmod(remaining_seconds, 60)
                countdown_str = f"{mins:02d}m {secs:02d}s"
                
                # Sanity check logging for active trade (e.g. BNB and other positions)
                atr_val = float(active_trade.get("atr_dollars", 0.0))
                sl_m = float(active_trade.get("sl_multiplier", 0.75))
                tp_m = float(active_trade.get("tp_multiplier", 1.65))
                reg_name = active_trade.get("regime", "Trending/Ranging")
                calc_sl = float(active_trade.get("initial_stop_loss", stop_loss))
                calc_tp = float(active_trade.get("initial_take_profit", take_profit))
                r_dist = abs(entry_price - calc_sl)
                w_dist = abs(calc_tp - entry_price)
                plan_rr = (w_dist / r_dist) if r_dist > 0 else 0.0

                print(f"[{active_symbol} {iv}m Active Trade SANITY CHECK]")
                print(f"  • ATR: {atr_val:.6f}")
                print(f"  • ATR (USD): ${atr_val:.6f}")
                print(f"  • sl_mult: {sl_m:.2f}")
                print(f"  • tp_mult: {tp_m:.2f}")
                print(f"  • Regime: {reg_name}")
                print(f"  • Calculated SL (Initial): {calc_sl:.6f}")
                print(f"  • Calculated TP (Initial): {calc_tp:.6f}")
                print(f"  • Initial Planned R:R: {plan_rr:.2f}")
                print(f"  • Current Trailing SL: {stop_loss:.6f}")
                print(f"  • Current Dashboard SL: {stop_loss:.6f}")

                print(f"[{active_symbol} {iv}m Active Trade] {direction} | Price: {current_price:.6f} (Entry: {entry_price:.6f}, SL: {stop_loss:.6f}, TP: {take_profit:.6f}) | Countdown: {countdown_str}")
                exit_reason = None
                half_closed = active_trade.get("half_closed", False)
                
                # 1 & 2. 10-Level Institutional Adaptive Exit Hierarchy Evaluation
                entry_time_ms = active_trade.get("entry_time") or 0
                # Guard against corrupted/missing entry_time (epoch 0 or unreasonable values)
                if entry_time_ms < 1e12:  # not a valid millisecond timestamp
                    entry_time_ms = int(time.time() * 1000)  # treat as just opened
                    active_trade["entry_time"] = entry_time_ms  # persist fix
                tf_mins = max(1, int(iv))
                candles_elapsed = max(0, int((time.time() - (entry_time_ms / 1000.0)) / (tf_mins * 60)))
                
                atr_dollars = active_trade.get("atr_dollars") or max(1e-6, entry_price * 0.01)
                highest_p = active_trade.get("highest_price", current_price)
                lowest_p = active_trade.get("lowest_price", current_price)
                pnl_dist_mfe = (highest_p - entry_price) if direction == "Bullish" else (entry_price - lowest_p)
                risk_dist_mfe = abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 1e-6 else atr_dollars
                mfe_r = round(pnl_dist_mfe / risk_dist_mfe, 2)
                
                curr_regime = bot_state.get(f"regime_{active_symbol}_{iv}") or bot_state.get(f"regime_{iv}", "Trending")
                entry_regime_val = str(active_trade.get("entry_regime", curr_regime))
                
                # Compute rolling volatility parameters for Levels 5, 6, 7 (M-14)
                garch_vol_val = float(atr_dollars / max(1e-6, current_price))
                rolling_vol_20th = float(garch_vol_val * 0.70)
                atr_ratio_val = 1.0
                df_recent_pos = get_history(symbol=active_symbol, interval=str(iv), limit=100)
                if df_recent_pos is not None and not df_recent_pos.empty and len(df_recent_pos) >= 20:
                    if "ATR_norm" not in df_recent_pos.columns or "ATR" not in df_recent_pos.columns:
                        high_s = df_recent_pos["high"].astype(float)
                        low_s = df_recent_pos["low"].astype(float)
                        close_s = df_recent_pos["close"].astype(float)
                        prev_close_s = close_s.shift(1)
                        tr_s = pd.concat([high_s - low_s, (high_s - prev_close_s).abs(), (low_s - prev_close_s).abs()], axis=1).max(axis=1)
                        atr_s = tr_s.ewm(alpha=1.0/14.0, adjust=False).mean()
                        df_recent_pos["ATR"] = atr_s
                        df_recent_pos["ATR_norm"] = atr_s / close_s.replace(0, np.nan)
                    df_norm_clean = df_recent_pos["ATR_norm"].dropna()
                    if len(df_norm_clean) >= 20:
                        rolling_vol_20th = float(df_norm_clean.tail(96).quantile(0.20))
                        mean_atr = float(df_norm_clean.tail(96).mean())
                        atr_ratio_val = float(df_norm_clean.iloc[-1] / max(1e-4, mean_atr))
                
                # Incoming signal opportunity cost & portfolio heat
                with active_execution_lock:
                    in_flight_margin_val = sum(active_execution_margins.values())
                total_active_val = sum(t.get("position_size_usd", 0.0) for tf_k in ["15m", "30m", "1h", "2h", "4h", "6h"] for t in bot_state.get(f"active_trade_{tf_k}", [])) + in_flight_margin_val
                current_equity = bot_state.get("simulated_balance", 80.0)
                port_heat = total_active_val / max(1.0, current_equity)
                is_heat_full = port_heat >= 0.80

                # Find best incoming predicted expected R across other symbols
                best_incoming_r = None
                for other_sym in SUPPORTED_SYMBOLS:
                    if other_sym != active_symbol:
                        pred_other = bot_state.get(f"latest_prediction_{other_sym}_{iv}", {})
                        if isinstance(pred_other, dict) and pred_other.get("direction") in ["Bullish", "Bearish"]:
                            other_ref_price = float(pred_other.get("ref_price") or pred_other.get("entry_price") or bot_state.get(f"live_price_{other_sym}") or 1.0)
                            other_change_pct = abs(float(pred_other.get("predicted_change", 0.0))) / max(1e-6, other_ref_price)
                            other_atr_pct = float(bot_state.get(f"atr_norm_{other_sym}_{iv}") or 0.015)
                            r_cand = other_change_pct / max(1e-4, other_atr_pct)
                            if best_incoming_r is None or r_cand > best_incoming_r:
                                best_incoming_r = r_cand

                hierarchy_eval = exit_policy_engine.evaluate_10_level_exit_hierarchy(
                    symbol=active_symbol,
                    interval=str(iv),
                    current_price=current_price,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    direction=direction,
                    candles_elapsed=candles_elapsed,
                    expected_r=float(active_trade.get("initial_planned_rr", 1.4)),
                    mfe_r=mfe_r,
                    entry_regime=entry_regime_val,
                    current_regime=str(curr_regime),
                    garch_vol=garch_vol_val,
                    rolling_vol_20th_pct=rolling_vol_20th,
                    atr_ratio=atr_ratio_val,
                    mhi_status=float(bot_state.get(f"mhi_{iv}", bot_state.get("mhi_score", 70.0))) if "bot_state" in globals() and hasattr(bot_state, "get") else 70.0,
                    incoming_signal_expected_r=best_incoming_r,
                    portfolio_heat_full=is_heat_full
                )
                
                if hierarchy_eval.get("should_exit"):
                    exit_reason = f"EXIT HIERARCHY LEVEL {hierarchy_eval.get('exit_level')}: {hierarchy_eval.get('exit_reason')}"
                    print(f"[{active_symbol} {iv}m Exit Hierarchy Triggered] Level {hierarchy_eval.get('exit_level')} -> {hierarchy_eval.get('exit_reason')} | Exit Score: {hierarchy_eval.get('exit_score')}")

                # Champion & Shadow Exit Policy Engine Evaluation (Finding #139, #11, #14)
                try:
                    curr_vol = float(df_recent_pos["volume"].iloc[-1]) if (df_recent_pos is not None and "volume" in df_recent_pos.columns and len(df_recent_pos) > 0) else 100.0
                    avg_vol = float(df_recent_pos["volume"].iloc[-20:].mean()) if (df_recent_pos is not None and "volume" in df_recent_pos.columns and len(df_recent_pos) >= 5) else 120.0
                    curr_atr_val = None
                    if df_recent_pos is not None and not df_recent_pos.empty:
                        if ("ATR" not in df_recent_pos.columns or "ATR_norm" not in df_recent_pos.columns) and len(df_recent_pos) >= 2:
                            high_s = df_recent_pos["high"].astype(float)
                            low_s = df_recent_pos["low"].astype(float)
                            close_s = df_recent_pos["close"].astype(float)
                            prev_close_s = close_s.shift(1)
                            tr_s = pd.concat([high_s - low_s, (high_s - prev_close_s).abs(), (low_s - prev_close_s).abs()], axis=1).max(axis=1)
                            df_recent_pos["ATR"] = tr_s.ewm(alpha=1.0/14.0, adjust=False).mean()
                            df_recent_pos["ATR_norm"] = df_recent_pos["ATR"] / close_s.replace(0, np.nan)
                        if "ATR" in df_recent_pos.columns and len(df_recent_pos["ATR"].dropna()) > 0:
                            curr_atr_val = float(df_recent_pos["ATR"].dropna().iloc[-1])
                        elif "ATR_norm" in df_recent_pos.columns and "close" in df_recent_pos.columns and len(df_recent_pos["ATR_norm"].dropna()) > 0:
                            curr_atr_val = float(df_recent_pos["ATR_norm"].dropna().iloc[-1] * float(df_recent_pos["close"].iloc[-1]))

                    champ_exit_reason, champ_updates, exit_trace = exit_policy_engine.evaluate_exit(
                        active_trade=active_trade,
                        current_price=current_price,
                        current_time=time.time(),
                        regime=str(curr_regime),
                        adx_val=float(bot_state.get(f"adx_{active_symbol}_{iv}", 20.0)),
                        current_volume=curr_vol,
                        avg_volume=avg_vol,
                        current_atr=curr_atr_val,
                        swing_price=float(active_trade.get("swing_low_3b", current_price)) if direction == "Bullish" else float(active_trade.get("swing_high_3b", current_price))
                    )
                    if champ_updates:
                        if "new_stop_loss" in champ_updates:
                            new_sl_val = float(champ_updates["new_stop_loss"])
                            is_long = direction in ["Bullish", "BUY", "LONG", "UP"]
                            is_tighter = (new_sl_val > stop_loss + 1e-4) if is_long else (new_sl_val < stop_loss - 1e-4)
                            if new_sl_val > 0 and is_tighter:
                                current_sl_snapshot = float(stop_loss)
                                if TRADE_MODE != "simulation":
                                    success = update_bybit_stop_loss(
                                        active_symbol,
                                        new_sl_val,
                                        active_trade=active_trade,
                                        current_sl_snapshot=current_sl_snapshot
                                    )
                                    if success:
                                        stop_loss = new_sl_val
                                        active_trade["stop_loss"] = stop_loss
                                        active_trades_updated = True
                                else:
                                    stop_loss = new_sl_val
                                    active_trade["stop_loss"] = stop_loss
                                    active_trades_updated = True
                        if champ_updates.get("break_even_triggered"):
                            active_trade["break_even_triggered"] = True
                        if champ_updates.get("trigger_scale_out") and not half_closed:
                            if TRADE_MODE == "simulation" or not active_trade.get("bybit_scale_out_order_id"):
                                if not active_trade.get("scale_out_triggered"):
                                    active_trade["scale_out_triggered"] = True
                                    active_trades_updated = True

                    # Exit Quality Score (Finding #17, #76)
                    eqs_mode = getattr(config, "EXIT_QUALITY_MODE", "shadow")
                    if eqs_mode != "disabled":
                        try:
                            from confluence_engine import calculate_exit_quality_score
                            eqs_score = calculate_exit_quality_score(
                                structure_pass=bool(not active_trade.get("sl_failed", False)),
                                liquidity_pass=bool(not is_heat_full),
                                expected_move_pct=abs(current_price - entry_price) / max(1e-6, entry_price),
                                spread_pct=float(bot_state.get(f"spread_bps_{active_symbol}", 3.0)) / 10000.0,
                                funding_rate=float(bot_state.get(f"funding_rate_{active_symbol}", 0.0001)),
                                atr_norm=float(atr_ratio_val * 0.004),
                                regime=str(curr_regime)
                            )
                            if exit_trace and isinstance(exit_trace, dict):
                                exit_trace["exit_quality_score"] = eqs_score
                            bot_state[f"latest_eqs_{active_symbol}_{iv}"] = eqs_score
                            if eqs_mode == "active" and eqs_score < 40.0 and not exit_reason:
                                exit_reason = f"EXIT_QUALITY_DEGRADED ({eqs_score:.1f} < 40.0)"
                                log_event("WARNING", f"[{active_symbol} {iv}m] EQS active trigger: {exit_reason}. Marking trade for exit.")
                        except Exception as ex_eqs:
                            log_event("DEBUG", f"[{active_symbol} {iv}m] EQS evaluation notice: {ex_eqs}")

                    if exit_trace:
                        bot_state["latest_exit_decision_trace"] = exit_trace
                    if champ_exit_reason and not exit_reason:
                        exit_reason = champ_exit_reason
                except Exception as ex_champ:
                    log_event("WARNING", f"[{active_symbol} {iv}m] evaluate_exit notice: {ex_champ}")

                if active_trade.get("trade_id") in rebal_close_set and not exit_reason:
                    exit_reason = "PORTFOLIO_UTILITY_REBALANCE_CLOSE"
                    print(f"[{active_symbol} {iv}m Portfolio Rebalance] Low-utility trade marked for closure to harvest margin.")

                
                # 3. SL/TP price checks (fallback for missing venue orders or simulation)
                if not exit_reason:
                    if direction == "Bullish":
                        if current_price <= stop_loss and (TRADE_MODE == "simulation" or active_trade.get("sl_failed")):
                            exit_reason = "TRAILING STOP HIT [SUCCESS]" if half_closed else "STOP LOSS HIT [FAIL]"
                        elif current_price >= take_profit:
                            exit_reason = "TAKE PROFIT HIT [SUCCESS]"
                    else:
                        if current_price >= stop_loss and (TRADE_MODE == "simulation" or active_trade.get("sl_failed")):
                            exit_reason = "TRAILING STOP HIT [SUCCESS]" if half_closed else "STOP LOSS HIT [FAIL]"
                        elif current_price <= take_profit:
                            exit_reason = "TAKE PROFIT HIT [SUCCESS]"

                # Finding #90: Enforce label horizon expiration (end_time) with Level 10 Runner Extension
                if not exit_reason and end_time > 0 and current_time >= end_time:
                    is_long_dir = direction in ["Bullish", "BUY", "LONG", "UP"]
                    pnl_dist = (current_price - entry_price) if is_long_dir else (entry_price - current_price)
                    risk_dist = abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 1e-6 else 1.0
                    curr_r_val = round(pnl_dist / risk_dist, 2)
                    decayed_exp_r = float(hierarchy_eval.get("decayed_expected_r", 0.0)) if isinstance(hierarchy_eval, dict) else 0.0
                    _cur_r = str(curr_regime).strip().lower()
                    _ent_r = str(entry_regime_val).strip().lower()
                    reg_intact = (
                        _cur_r == _ent_r or
                        ("trend" in _cur_r and "trend" in _ent_r) or
                        ("rang" in _cur_r and "rang" in _ent_r)
                    )
                    is_runner_eligible = (
                        curr_r_val >= 2.0 and
                        decayed_exp_r >= 0.20 and
                        reg_intact
                    )
                    if is_runner_eligible:
                        log_event("INFO", f"[{active_symbol} {iv}m] Label horizon reached ({countdown_str}) but runner extension granted (PnL: +{curr_r_val:.1f}R, Exp R: {decayed_exp_r:.2f}R). Extending hold.")
                    else:
                        exit_reason = "HORIZON_EXPIRY [LABEL_TIMEOUT]"
                        log_event("INFO", f"[{active_symbol} {iv}m] Trade reached full label horizon ({countdown_str}). Programmatic close.")
                
                if TRADE_MODE != "simulation":
                    if exit_reason is not None and not bybit_closed:
                        # Cancel scale-out limit order if it exists
                        scale_out_id = active_trade.get("bybit_scale_out_order_id")
                        if scale_out_id:
                            cancel_payload = {"category": "linear", "symbol": active_symbol, "orderId": scale_out_id}
                            bybit_post_request("/v5/order/cancel", cancel_payload)
                            print(f"[Bybit API] Canceled scale-out limit order {scale_out_id} due to programmatic exit.")
                        
                        # Place market close order
                        pos = get_bybit_position(active_symbol)
                        if pos:
                            qty_str = pos.get("size", "0")
                            if float(qty_str) > 0:
                                close_side = "Sell" if direction == "Bullish" else "Buy"
                                print(f"[Bybit API] Placing Market close order for {qty_str} {active_symbol} due to programmatic exit...")
                                close_res = place_bybit_order(symbol=active_symbol, side=close_side, qty=qty_str, reduce_only=True)
                                if close_res.get("retCode") == 0:
                                    # Strict verification: Confirm exchange position size is 0.0 before marking closed
                                    is_flat = False
                                    for v_att in range(5):
                                        time.sleep(0.4)
                                        v_pos = get_bybit_position(active_symbol)
                                        if v_pos and float(v_pos.get("size", 0.0)) == 0.0:
                                            is_flat = True
                                            break
                                        elif v_pos and float(v_pos.get("size", 0.0)) > 0.0:
                                            rem_sz = float(v_pos.get("size", 0.0))
                                            print(f"[{active_symbol} {iv}m] Programmatic exit residual: {rem_sz} remaining (attempt {v_att+1}). Resubmitting...")
                                            place_bybit_order(symbol=active_symbol, side=close_side, qty=format_bybit_qty(active_symbol, rem_sz), reduce_only=True)
                                    if is_flat:
                                        bybit_closed = True
                                    else:
                                        log_event("CRITICAL", f"[{active_symbol} {iv}m] Programmatic exit unconfirmed: Position size not 0.0 on Bybit!")
                                        active_trade["close_failed"] = True
                                else:
                                    log_event("ERROR", f"[{active_symbol} {iv}m] Market close order failed: {close_res.get('retMsg')}")
                                    active_trade["close_failed"] = True

                    if bybit_closed:
                        entry_time_ms = active_trade.get("entry_time")
                        if not entry_time_ms:
                            entry_time_ms = int((end_time - (int(iv) * 60)) * 1000)
                        
                        expected_qty = float(active_trade.get("qty", 0.0))
                        bybit_pnl_data = None
                        for pnl_attempt in range(5):
                            bybit_pnl_data = get_bybit_accumulated_closed_pnl(active_symbol, entry_time_ms, expected_total_qty=expected_qty)
                            if bybit_pnl_data is not None:
                                break
                            time.sleep(0.5)

                        if bybit_pnl_data:
                            bybit_realized_pnl = bybit_pnl_data["total_pnl"]
                            pnl_source = "EXCHANGE"
                            if bybit_pnl_data["avg_exit_price"] is not None:
                                bybit_exit_price = bybit_pnl_data["avg_exit_price"]
                            if bybit_pnl_data["total_entry_value"] is not None:
                                lev = float(active_trade.get("leverage", 1.0))
                                actual_margin = round(bybit_pnl_data["total_entry_value"] / lev, 2)
                                active_trade["position_size_usd"] = actual_margin
                                position_size_usd = actual_margin
                        else:
                            # Venue publication delay fallback: do NOT set bybit_realized_pnl to a gross estimate!
                            # Leaving bybit_realized_pnl as None ensures the downstream fee-aware local calculation
                            # (gross_pnl - fee_cost) is executed and tagged as ESTIMATED.
                            pnl_source = "ESTIMATED"
                            if bybit_exit_price is None:
                                bybit_exit_price = current_price

                        actual_exit = bybit_exit_price if bybit_exit_price is not None else current_price
                        tp_hit = (actual_exit >= take_profit) if direction == "Bullish" else (actual_exit <= take_profit)
                        sl_hit = (actual_exit <= stop_loss) if direction == "Bullish" else (actual_exit >= stop_loss)
                        atr_ref = active_trade.get("atr_dollars") or (entry_price * 0.01)
                        be_hit = abs(actual_exit - entry_price) < (0.25 * atr_ref)
                        
                        is_profit = (bybit_realized_pnl > 0.001) if (bybit_realized_pnl is not None) else ((actual_exit > entry_price + 0.0005 * entry_price) if direction == "Bullish" else (actual_exit < entry_price - 0.0005 * entry_price))
                        
                        if tp_hit and is_profit:
                            exit_reason = "TAKE PROFIT HIT [SUCCESS]"
                        elif (half_closed or active_trade.get("break_even_triggered")) and is_profit:
                            exit_reason = "TRAILING STOP / BREAK-EVEN HIT [SUCCESS]"
                        elif is_profit:
                            exit_reason = "PROFITABLE EXIT [SUCCESS]"
                        elif sl_hit:
                            exit_reason = "STOP LOSS HIT [FAIL]"
                        else:
                            exit_reason = "EXCHANGE / EARLY CLOSE [CONTROLLED]"

                if TRADE_MODE == "simulation":
                    is_exited = (exit_reason is not None)
                else:
                    is_exited = (exit_reason is not None and bybit_closed) or bybit_closed
                if is_exited:
                    # Maker vs Taker execution logic
                    is_stop_loss = "STOP LOSS" in str(exit_reason).upper() if exit_reason else True
                    
                    if is_stop_loss:
                        # Taker execution for Stop Loss market close
                        slippage_pct = 0.0
                        actual_price = bybit_exit_price if bybit_exit_price is not None else current_price
                        exit_reason = str(exit_reason)
                    else:
                        # Maker execution for Take Profit, Timer, etc.
                        slippage_pct = 0.0
                        actual_price = bybit_exit_price if bybit_exit_price is not None else current_price

                    price_diff = actual_price - predicted_price
                    price_diff_pct = (price_diff / predicted_price) * 100
                    price_accuracy = max(0.0, 100.0 - abs((actual_price - predicted_price) / actual_price * 100))
                    actual_change = actual_price - entry_price
                    actual_change_pct = (actual_change / entry_price) * 100
                    
                    # Calculate PnL (long vs short) with correct taker fees on leveraged size
                    raw_return_pct = actual_change_pct if direction == "Bullish" else -actual_change_pct
                    leverage = active_trade.get("leverage", 1.0)
                    gross_pnl = position_size_usd * (raw_return_pct * leverage / 100.0)
                    entry_fee_rate = 0.0002  # Assumed Maker entry
                    exit_fee_rate = 0.0002 if not is_stop_loss else 0.00055  # Limit exit vs Stop Loss market close
                    roundtrip_fee_rate = entry_fee_rate + exit_fee_rate
                    fee_cost = position_size_usd * leverage * roundtrip_fee_rate
                    
                    # Deduct accrued funding settlement cost (Finding #78)
                    funding_rate = get_funding_rate(active_symbol)
                    funding_intervals = max(0.0, float(candles_elapsed) * float(iv) / 480.0)
                    funding_dir_mult = 1.0 if direction == "Bullish" else -1.0
                    funding_cost = position_size_usd * leverage * funding_rate * funding_dir_mult * funding_intervals
                    realized_pnl = gross_pnl - fee_cost - funding_cost
                    net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
                    
                    if realized_pnl < -position_size_usd:
                        realized_pnl = -position_size_usd
                        net_return_pct = -100.0
                    
                    # Aggregate PnL and size for trade history logging if scaled out
                    original_size = float(active_trade.get("original_size", position_size_usd))
                    scaled_out_pnl = float(active_trade.get("scaled_out_pnl", 0.0))
                    
                    if TRADE_MODE != "simulation" and bybit_realized_pnl is not None:
                        # Finding #60: Bybit closed-pnl excludes settled funding payments; deduct funding_cost
                        total_pnl = round(bybit_realized_pnl - funding_cost, 2)
                        realized_pnl = round(total_pnl - scaled_out_pnl, 2)
                        net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
                        total_net_return_pct = round((total_pnl / original_size) * 100.0, 4)
                    else:
                        total_pnl = round(realized_pnl + scaled_out_pnl, 2)
                        total_net_return_pct = round((total_pnl / original_size) * 100.0, 4)
                    
                    # Log total trade outcome (including scale-outs) into KellyTracker (Finding #79)
                    global_kelly_tracker.log_trade(active_symbol, str(iv), total_pnl, total_net_return_pct)
                    
                    # Update simulated balance (only in simulation)
                    if TRADE_MODE == "simulation":
                        old_bal = bot_state.get("simulated_balance", 80.0)
                        new_bal = round(old_bal + position_size_usd + realized_pnl, 2)
                        bot_state["simulated_balance"] = new_bal
                    else:
                        new_bal = bot_state.get("simulated_balance", 0.0)
                    
                    actual_trend = "Bullish" if actual_change > 0 else "Bearish"
                    signal_correct = (actual_trend == direction)
                    
                    print("\n==================================================")
                    print(f"[{active_symbol} {iv}m TRADE EXITED]: {exit_reason} (Slippage: {slippage_pct:.3f}%)")
                    print(f"Start Price: {entry_price:.2f} | Exit Price: {actual_price:.2f}")
                    print(f"Actual Change: {actual_change:+.2f} ({actual_change_pct:+.4f}%)")
                    if active_trade.get("half_closed", False):
                        print(f"Total Size: ${original_size:.2f} (Scaled-Out) | Net Return: {total_net_return_pct:+.4f}% (weighted)")
                        print(f"Scaled-Out PnL: ${scaled_out_pnl:+.2f} | Remaining PnL: ${realized_pnl:+.2f} | Total PnL: ${total_pnl:+.2f}")
                    else:
                        print(f"Size: ${position_size_usd:.2f} | Net Return: {net_return_pct:+.4f}% (after {roundtrip_fee_rate*100.0:.3f}% fees)")
                        print(f"Realized PnL: ${realized_pnl:+.2f}")
                    # 81-Scenario Counterfactual Replay Matrix & Regret Analysis
                    try:
                        risk_usd_ref = active_trade.get("atr_dollars") or (entry_price * 0.01)
                        actual_r_val = round((actual_price - entry_price) / max(1e-6, risk_usd_ref) if direction == "Bullish" else (entry_price - actual_price) / max(1e-6, risk_usd_ref), 3)
                        
                        cf_res = counterfactual_replay_engine.run_81_scenario_replay(
                            trade_id=str(active_trade.get("trade_id", active_symbol)),
                            symbol=active_symbol,
                            interval=str(iv),
                            entry_price=entry_price,
                            exit_price=actual_price,
                            direction=direction,
                            realized_pnl=total_pnl,
                            planned_rr=float(active_trade.get("initial_planned_rr", 1.4)),
                            actual_r=actual_r_val
                        )
                        
                        best_cf = cf_res.get("best_scenario", {})
                        decision_outcome_db.update_outcome_and_regret(
                            decision_id=str(active_trade.get("trade_id", active_symbol)),
                            outcome_details={"realized_pnl": realized_pnl, "actual_r": actual_r_val, "exit_reason": exit_reason},
                            counterfactual_matrix=cf_res,
                            best_counterfactual_r=float(best_cf.get("simulated_r", actual_r_val)),
                            actual_r=actual_r_val
                        )
                        
                        regret_r = max(0.0, float(best_cf.get("simulated_r", actual_r_val)) - actual_r_val)
                        print(f"[{active_symbol} {iv}m Replay Matrix] 81 Scenarios Evaluated | Best Alternative: {best_cf.get('scenario_id')} (+{best_cf.get('simulated_r'):.2f}R) | Regret: +{regret_r:.2f}R")
                    except Exception as e:
                        print(f"[Counterfactual Replay Warning] {e}")

                    # Update Completed Trade History in global state
                    t_id = active_trade.get("trade_id") or f"tr_{active_symbol}_{int(active_trade.get('entry_time') or time.time())}"
                    active_trade["trade_id"] = t_id
                    completed_trade = {
                        "trade_id": t_id,
                        "symbol": active_symbol,
                        "entry_time": float(active_trade.get("entry_time", 0)),
                        "exit_time": float(time.time()),
                        "interval": str(iv),
                        "direction": str(direction),
                        "regime": str(active_trade.get("regime") or active_trade.get("entry_regime") or bot_state.get(f"regime_{active_symbol}_{iv}") or bot_state.get(f"regime_{iv}") or "Trending"),
                        "entry_regime": str(active_trade.get("entry_regime") or active_trade.get("regime") or bot_state.get(f"regime_{active_symbol}_{iv}") or bot_state.get(f"regime_{iv}") or "Trending"),
                        "entry_price": float(entry_price),
                        "exit_price": float(actual_price),
                        "change_pct": float(total_net_return_pct if active_trade.get("half_closed", False) else net_return_pct),
                        "success": bool(total_pnl > 0),
                        "signal_correct": bool(signal_correct),
                        "reason": str(exit_reason) + (" (Scale-Out)" if active_trade.get("half_closed", False) else ""),
                        "position_size_usd": float(position_size_usd),
                        "intended_size_usd": float(active_trade.get("intended_size_usd") or active_trade.get("original_size") or position_size_usd),
                        "original_size": float(original_size),
                        "pnl_usd": float(total_pnl),
                        "balance": float(new_bal),
                        "leverage": float(leverage),
                        "confidence": active_trade.get("confidence") if active_trade.get("confidence") == "MT" else float(active_trade.get("confidence") or 0.0),
                        "take_profit": float(active_trade.get("take_profit", 0.0)),
                        "stop_loss": float(active_trade.get("stop_loss", 0.0)),
                        "venue_closed_pnl": float(bybit_pnl_data.get("total_pnl")) if (isinstance(bybit_pnl_data, dict) and bybit_pnl_data.get("total_pnl") is not None) else None,
                        "venue_qty": float(bybit_pnl_data.get("total_qty")) if (isinstance(bybit_pnl_data, dict) and bybit_pnl_data.get("total_qty") is not None) else None,
                        "venue_entry_value": float(bybit_pnl_data.get("total_entry_value")) if (isinstance(bybit_pnl_data, dict) and bybit_pnl_data.get("total_entry_value") is not None) else None,
                        "stop_state": active_trade.get("stop_state", "INITIAL"),
                        "stop_state_meta": active_trade.get("stop_state_meta", {}),
                        "atr_dollars": float(active_trade.get("atr_dollars", 0.0)),
                        "fill_pct": float(active_trade.get("fill_pct", 100.0)),
                        "modeled_slippage_bps": float(active_trade.get("modeled_slippage_bps")) if active_trade.get("modeled_slippage_bps") is not None else None,
                        "realized_slippage_bps": float(active_trade.get("realized_slippage_bps")) if active_trade.get("realized_slippage_bps") is not None else None,
                        "bybit_order_id": active_trade.get("bybit_order_id"),
                        "bybit_scale_out_order_id": active_trade.get("bybit_scale_out_order_id"),
                        "pnl_source": pnl_source
                    }
                    # Finding #102 & R50: Atomically persist trade closure to SQLite with verified return
                    db_closed = False
                    try:
                        import database
                        db_closed = database.close_trade_atomically(completed_trade, tf=str(iv))
                    except Exception as ex_db_close:
                        log_event("WARNING", f"[DB Close Trade Atomically Warning] {ex_db_close}")
                        db_closed = False
                    if not db_closed:
                        # Retry once after short backoff to recover transient locks
                        import time as _t_close
                        _t_close.sleep(0.1)
                        try:
                            db_closed = database.close_trade_atomically(completed_trade, tf=str(iv))
                        except Exception as ex_retry:
                            log_event("WARNING", f"[DB Close Trade Retry Warning] {ex_retry}")
                            db_closed = False
                    if not db_closed:
                        log_event("CRITICAL", f"[Database Close Failure] Atomic persistence failed/rolled back for {completed_trade.get('trade_id')} ({active_symbol} {iv}m). Retaining trade in active list to prevent state desync.")
                        updated_trades.append(active_trade)
                        continue
                    active_trade["exit_processed"] = True
                    with active_trades_lock:
                        bot_state["trade_history"].append(completed_trade)
                        active_trades_updated = True
                    # Log to trade journal CSV
                    log_trade_journal(completed_trade)
                    
                    # Build Scale-Out details block if trade was half-closed
                    scale_out_block = ""
                    if active_trade.get("half_closed", False):
                        stage1_price = float(active_trade.get("scale_out_price", entry_price))
                        stage1_margin = float(active_trade.get("scaled_out_margin", original_size / 2.0))
                        stage1_pnl = float(active_trade.get("scaled_out_pnl", 0.0))
                        
                        stage2_price = actual_price
                        stage2_margin = float(position_size_usd)
                        stage2_pnl = float(realized_pnl)
                        
                        total_m = stage1_margin + stage2_margin
                        pct1 = round((stage1_margin / max(1e-6, total_m) * 100.0), 1) if total_m > 0 else 50.0
                        pct2 = round((stage2_margin / max(1e-6, total_m) * 100.0), 1) if total_m > 0 else 50.0
                        
                        stage1_title = "Partial Profit Secured" if stage1_pnl > 0 else "Partial Scale-Out Exit"
                        stage1_price_label = "Target Price" if stage1_pnl > 0 else "Exit Price"
                        
                        stage2_name = "Trailing Stop Hit" if "TRAILING" in str(exit_reason).upper() else "Take Profit Hit" if "TAKE PROFIT" in str(exit_reason).upper() else "Final Exit"
                        
                        scale_out_block = (
                            f"\n\n🥞 *Scale-Out Execution Details*\n"
                            f"• *Stage 1: {stage1_title} ({pct1}% Scale-Out)*\n"
                            f"  - {stage1_price_label}: `${stage1_price:.4f}`\n"
                            f"  - Returned Margin: `${stage1_margin:.2f}`\n"
                            f"  - PnL Realized: *${stage1_pnl:+.2f}*\n"
                            f"• *Stage 2: {stage2_name} ({pct2}% Remaining)*\n"
                            f"  - Exit Price: `${stage2_price:.4f}`\n"
                            f"  - Returned Margin: `${stage2_margin:.2f}`\n"
                            f"  - PnL Realized: *${stage2_pnl:+.2f}*"
                        )

                    # Deduplicate exit alerts to avoid sending duplicate close messages
                    exit_alert_key = (active_symbol, str(iv), str(active_trade.get("trade_id", "")), round(total_pnl, 2))
                    if exit_alert_key not in _dispatched_exit_alerts:
                        _dispatched_exit_alerts.add(exit_alert_key)
                        if len(_dispatched_exit_alerts) > 500:
                            _dispatched_exit_alerts.clear()

                        if total_pnl > 0:
                            exit_header = "🚀 *TAKE PROFIT HIT* 🚀" if "TAKE PROFIT" in str(exit_reason).upper() else "📈 *TRAILING STOP HIT (PROFITABLE)* 📈" if "TRAILING" in str(exit_reason).upper() else "🎉 *TRADE CLOSED WITH PROFIT* 🎉"
                            send_telegram_alert(
                                f"{exit_header}\n"
                                f"• *Asset*: {active_symbol}\n"
                                f"• *Interval*: {iv}m\n"
                                f"• *Direction*: {direction}\n"
                                f"• *Entry Price*: ${entry_price:.4f}\n"
                                f"• *Exit Price*: ${actual_price:.4f}\n"
                                f"• *Realized PnL*: *${total_pnl:+.2f}* (" + (f"{total_net_return_pct:+.2f}" if active_trade.get("half_closed", False) else f"{net_return_pct:+.2f}") + f"%)\n"
                                f"• *Exit Reason*: {exit_reason}" + (" (Scale-Out)" if active_trade.get("half_closed", False) else "") + "\n"
                                f"• *New Balance*: ${new_bal:.2f}"
                                f"{scale_out_block}"
                            )
                        else:
                            send_telegram_alert(
                                f"🔴 *POSITION CLOSED (AUTO)* 🔴\n"
                                f"• *Asset*: {active_symbol}\n"
                                f"• *Interval*: {iv}m\n"
                                f"• *Direction*: {direction}\n"
                                f"• *Exit Reason*: {exit_reason}" + (" (Scale-Out)" if active_trade.get("half_closed", False) else "") + "\n"
                                f"• *Entry Price*: ${entry_price:.4f}\n"
                                f"• *Exit Price*: ${actual_price:.4f}\n"
                                f"• *Realized PnL*: ${total_pnl:+.2f} (" + (f"{total_net_return_pct:+.2f}" if active_trade.get("half_closed", False) else f"{net_return_pct:+.2f}") + f"%)\n"
                                f"• *New Balance*: ${new_bal:.2f}"
                                f"{scale_out_block}"
                            )
                        
                        entry_ts_raw = float(active_trade.get("entry_time", 0))
                        entry_ts_sec = entry_ts_raw / 1000.0 if entry_ts_raw > 1e11 else entry_ts_raw
                        if entry_ts_sec > 0:
                            dur_sec = abs(time.time() - entry_ts_sec)
                            if dur_sec < 5.0:
                                send_telegram_alert(f"🚨 *CRITICAL EXECUTION ALERT*: `{active_symbol}` ({iv}m) stopped out within {dur_sec:.1f}s of entry — check SL placement/geometry!")
                                log_event("ERROR", f"[{active_symbol} {iv}m] Stopped out within {dur_sec:.1f}s of entry (Entry: ${entry_price:.4f}, Exit: ${actual_price:.4f})")
                    
                    
                    # Send email alert on any profitable trade exit
                    if total_pnl > 0:
                        subject = f"🚀 [UBOTE Profit Target] {active_symbol} {iv}m Closed with Profit!"
                        invested_margin_usd = original_size
                        leveraged_position_usd = original_size * leverage if leverage > 0 else original_size
                        
                        # Dynamic header based on exit reason
                        exit_title = "🎉 Take Profit Hit!" if "TAKE PROFIT" in str(exit_reason).upper() else "📈 Trailing Stop Hit (Profitable Close)!" if "TRAILING" in str(exit_reason).upper() else "✅ Trade Closed with Profit!"
                        
                        body = f"""
                        <html>
                        <body style="font-family: Arial, sans-serif; background-color: #0b0e14; color: #f0f3fa; padding: 20px; border-radius: 8px;">
                            <h2 style="color: #00b0ff; margin-bottom: 20px;">{exit_title}</h2>
                            <div style="background-color: #161a22; padding: 15px; border-radius: 6px; border-left: 4px solid #00c853;">
                                <table style="width: 100%; border-collapse: collapse;">
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3; width: 140px;"><b>Symbol:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace; font-size: 14px;">{active_symbol}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Timeframe:</b></td>
                                        <td style="padding: 6px 0;">{iv}m</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Direction:</b></td>
                                        <td style="padding: 6px 0; color: {'#00c853' if direction == 'Bullish' else '#ff3d00'}; font-weight: bold;">{direction}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Entry Price:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">${entry_price:.4f}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Exit Price:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">${actual_price:.4f}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Profit/Loss:</b></td>
                                        <td style="padding: 6px 0; color: #00c853; font-weight: bold; font-family: monospace;">+{total_pnl:+.2f} USD ({total_net_return_pct:+.4f}%)</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Exit Reason:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">{exit_reason}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Leveraged Position Size (USD):</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">${leveraged_position_usd:.2f} USD</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Leverage:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">{leverage}x</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Actual Investment (USD):</b></td>
                                        <td style="padding: 6px 0; font-family: monospace;">${invested_margin_usd:.2f} USD</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 6px 0; color: #8f9bb3;"><b>Account Balance:</b></td>
                                        <td style="padding: 6px 0; font-family: monospace; font-weight: bold;">${new_bal:.2f} USD</td>
                                    </tr>
                                </table>
                            </div>
                            <p style="font-size: 11px; color: #8f9bb3; margin-top: 20px;">Sent automatically by UBOTE Trading System.</p>
                        </body>
                        </html>
                        """
                        threading.Thread(target=send_email_notification, args=(subject, body), daemon=True).start()
                    
                    for p in bot_state["prediction_history"]:
                        if p.get("interval") == str(iv) and p.get("symbol") == active_symbol and p.get("status") == "Traded" and (not p.get("evaluation") or not p["evaluation"].get("evaluated")):
                            p["evaluation"] = {
                                "evaluated": True,
                                "exit_price": float(actual_price),
                                "change": float(actual_change if direction == "Bullish" else -actual_change),
                                "change_pct": float(raw_return_pct),
                                "success": bool(signal_correct)
                            }
                            bot_state.save_prediction(p)
                            break
                    active_trade["exit_processed"] = True
                    save_history()
                else:
                    updated_trades.append(active_trade)
            with active_trades_lock:
                # Finding #25: Merge with any concurrent manual trades added while the exit loop ran
                current_active_trades = bot_state.get(active_trade_key, [])
                if isinstance(current_active_trades, list):
                    evaluated_ids = {t.get("trade_id") or f"{t.get('symbol')}_{t.get('entry_time')}" for t in active_trades_list}
                    for cat in current_active_trades:
                        cat_id = cat.get("trade_id") or f"{cat.get('symbol')}_{cat.get('entry_time')}"
                        if cat_id not in evaluated_ids:
                            updated_trades.append(cat)
                bot_state[active_trade_key] = updated_trades
                try:
                    import database
                    database.save_active_trades(tf, updated_trades)
                except Exception as ex_db_save:
                    log_event("WARNING", f"[Active Trades Save Warning] {ex_db_save}")
        
        if active_trades_updated:
            save_history()

        # 3. Check for completed candle closes to search for a new signal

        # --- Daily Drawdown Circuit Breaker & Profit Goal ---
        today = datetime.now(timezone.utc).day
        current_equity_val = float(bot_state.get("live_balance", bot_state.get("wallet_balance", bot_state.get("simulated_balance", 80.0))))
        if current_equity_val <= 0:
            current_equity_val = float(bot_state.get("simulated_balance", 80.0))

        if bot_state["daily_drawdown_reset_day"] != today:
            bot_state["daily_drawdown_start_balance"] = current_equity_val
            bot_state["daily_drawdown_reset_day"] = today
            bot_state["circuit_breaker_active"] = False
            bot_state["daily_goal_reached"] = False
            print(f"[Circuit Breaker] Daily reset (UTC). Start balance: ${bot_state['daily_drawdown_start_balance']:.2f}")
        else:
            start_bal = bot_state["daily_drawdown_start_balance"]
            curr_bal = current_equity_val
            daily_dd_pct = (start_bal - curr_bal) / start_bal * 100 if start_bal > 0 else 0
            daily_profit = curr_bal - start_bal
            
            # Rule 3: GARCH-scaled Dynamic Daily Circuit Breaker
            btc_returns = None
            df_dict_val = locals().get("df_dict")
            if isinstance(df_dict_val, dict) and "BTCUSDT" in df_dict_val:
                btc_df = df_dict_val["BTCUSDT"]
                if isinstance(btc_df, pd.DataFrame) and "close" in btc_df.columns:
                    btc_returns = btc_df["close"].pct_change()

            dynamic_halt_pct = garch_vol_monitor.get_dynamic_circuit_breaker_pct(btc_returns)
            
            if daily_dd_pct >= dynamic_halt_pct:
                if not bot_state.get("circuit_breaker_active", False):
                    send_telegram_alert(f"⚠️ *DYNAMIC GARCH CIRCUIT BREAKER* ⚠️\n• Start Balance: ${start_bal:.2f}\n• Current Balance: ${curr_bal:.2f}\n• Daily Drawdown: *{daily_dd_pct:.2f}%* (>= {dynamic_halt_pct:.2f}% GARCH limit)\n• *Trading Halted* until reset.")
                bot_state["circuit_breaker_active"] = True
                print(f"[Circuit Breaker] TRIGGERED - Daily drawdown is {daily_dd_pct:.2f}% (>= {dynamic_halt_pct:.2f}% GARCH limit). Trading halted.")
            else:
                from config import CIRCUIT_BREAKER_RESUME_RATIO
                resume_pct = dynamic_halt_pct * CIRCUIT_BREAKER_RESUME_RATIO
                if daily_dd_pct < resume_pct:
                    if bot_state.get("circuit_breaker_active", False):
                        log_event("INFO", f"[Circuit Breaker] Released at {daily_dd_pct:.2f}% (< {resume_pct:.2f}%)")
                        send_telegram_alert(f"✅ *DYNAMIC CIRCUIT BREAKER RELEASED* ✅\n• Daily Drawdown: *{daily_dd_pct:.2f}%* (< {resume_pct:.2f}% recovery limit)\n• *Trading Resumed*.")
                        print(f"[Circuit Breaker] RELEASED - Daily drawdown {daily_dd_pct:.2f}% < {resume_pct:.2f}% limit. Trading resumed.")
                    bot_state["circuit_breaker_active"] = False

            # Rule 4: Rolling 5.0% Equity Target (replaces static $1000)
            rolling_equity_goal = max(100.0, curr_bal * 0.05)
            if daily_profit >= rolling_equity_goal and not bot_state.get("daily_goal_reached", False):
                bot_state["daily_goal_reached"] = True
                send_telegram_alert(f"🎉 *ROLLING EQUITY GOAL REACHED* 🎉\n• Daily Profit: *${daily_profit:.2f}* (5% Equity Target: ${rolling_equity_goal:.2f})\n• Current Account Value: ${curr_bal:.2f}\n• Continuing trading to maximize gains.")
                print(f"[Daily Goal] REACHED - daily profit of ${daily_profit:.2f} >= ${rolling_equity_goal:.2f} (5% target). Continuing trading.")
            elif daily_profit < rolling_equity_goal:
                bot_state["daily_goal_reached"] = False

        # --- Rule 5: Impact-Weighted News Window Guard ---
        def is_high_impact_news_window():
            """Rule 5: Impact-Weighted News Blackout (15m, 30m, 45m)."""
            news_active, news_reason = news_monitor.get_news_blackout_status()
            if news_active:
                return True, news_reason

            try:
                now_utc = datetime.now(timezone.utc)
                events = fetch_economic_calendar_cached()
                for ev_time in events:
                    diff = abs((now_utc - ev_time).total_seconds())
                    if diff <= 1800:  # 30 minute fallback window
                        return True, "High-Impact Economic Event"
            except Exception:
                pass
            return False, ""

        # --- Consecutive Losses & Post-Trade Single-Candle Cooldown ---
        def is_symbol_interval_cooling_off(symbol, interval):
            """
            Checks if a symbol and interval combination is in:
            1. Post-trade single candle cool-off (prevents immediate re-entry on same bar).
            2. A 6-hour cool-off period after suffering 2 consecutive loss trades.
            """
            trades = [t for t in bot_state.get("trade_history", []) if t.get("symbol") == symbol and str(t.get("interval")) == str(interval)]
            if not trades:
                return False, 0
                
            sorted_trades = sorted(trades, key=lambda x: float(x.get("exit_time", 0.0) or 0.0), reverse=True)
            latest_trade = sorted_trades[0]
            latest_exit_time = float(latest_trade.get("exit_time", 0.0) or 0.0)
            
            # 1. Post-Trade Single-Candle Pause
            try:
                iv_str = str(interval).strip()
                if iv_str.endswith("h"):
                    iv_minutes = int(iv_str[:-1]) * 60
                elif iv_str.endswith("m"):
                    iv_minutes = int(iv_str[:-1])
                else:
                    iv_minutes = int(iv_str)
            except Exception:
                iv_minutes = 60
            
            candle_duration_sec = iv_minutes * 60
            time_since_exit = time.time() - latest_exit_time
            if time_since_exit < candle_duration_sec:
                remaining_minutes = max(1, int((candle_duration_sec - time_since_exit) / 60))
                return True, remaining_minutes
            
            # 2. Consecutive Losses Cooldown
            if len(sorted_trades) >= 2:
                second_latest = sorted_trades[1]
                def _is_loss(t):
                    succ = str(t.get("success", "")).lower()
                    if succ in ["false", "0", "no"]:
                        return True
                    try:
                        pnl = float(t.get("pnl_usd", 0.0) or 0.0)
                        return pnl < 0.0
                    except Exception:
                        return False

                if _is_loss(latest_trade) and _is_loss(second_latest):
                    cooldown_duration = 6 * 3600  # 6 hours
                    if time_since_exit < cooldown_duration:
                        remaining_minutes = int((cooldown_duration - time_since_exit) / 60)
                        return True, remaining_minutes
                        
            return False, 0

        def get_learned_confidence_threshold(symbol: str, interval: str, regime: str) -> float:
            try:
                thresh_file = f"learned_thresholds_{interval}.json"
                if os.path.exists(thresh_file):
                    with open(thresh_file, "r") as f:
                        data = json.load(f)
                        key = f"{symbol}_{regime.lower()}"
                        if key in data:
                            return float(data[key])
                        elif regime.lower() in data:
                            return float(data[regime.lower()])
                        elif "default" in data:
                            return float(data["default"])
            except Exception:
                pass
            try:
                from config import TIMEFRAME_CONFIG, DYNAMIC_CONFIDENCE_THRESHOLDS
                if str(interval) in TIMEFRAME_CONFIG and "base_confidence_threshold" in TIMEFRAME_CONFIG[str(interval)]:
                    return float(TIMEFRAME_CONFIG[str(interval)]["base_confidence_threshold"])
                return float(DYNAMIC_CONFIDENCE_THRESHOLDS.get(str(interval), 0.55))
            except Exception:
                return 0.55

        reloaded_intervals = check_and_hot_reload_models()
        
        # --- Intelligent Boundary Window Candle Polling ---
        current_time_utc = datetime.now(timezone.utc)
        current_utc_hour = current_time_utc.hour
        current_utc_minute = current_time_utc.minute
        current_15m_block = (current_utc_hour * 60 + current_utc_minute) // 15
        
        # 1. Reset block state variables at a new 15-minute interval transition
        if current_15m_block != last_check_hour:
            last_check_hour = current_15m_block
            hour_check_complete = False
            completed_this_hour.clear()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] New 15m boundary block detected ({current_utc_hour:02d}:{current_utc_minute:02d} UTC). Resetting check status.")
            
        # 2. Determine if we are within the candle check window (first 4 minutes after boundary) or executing startup check
        is_in_check_window = (current_utc_minute % 15 < CANDLE_CHECK_WINDOW_MINS) or (not startup_check_done)
        
        check_queue = []
        is_startup = not startup_check_done
        
        if is_in_check_window and not hour_check_complete:
            # Throttle requests to CANDLE_CHECK_INTERVAL_SECS (e.g. 20s)
            if current_time - last_candle_check_time >= CANDLE_CHECK_INTERVAL_SECS:
                last_candle_check_time = current_time
                
                if is_startup:
                    # Startup initial check: query all supported symbols for all intervals in parallel
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Startup initial fast check: checking all {len(SUPPORTED_SYMBOLS)} symbols across all timeframes...")
                    check_queue = [(symbol, iv) for iv in ["15", "30", "60", "120", "240"] for symbol in SUPPORTED_SYMBOLS]
                    startup_check_done = True
                else:
                    # Regular transition checks: determine active intervals for this 15m boundary
                    active_intervals = ["15"]
                    if current_utc_minute // 15 % 2 == 0:
                        active_intervals.append("30")
                    if current_utc_minute < 15:
                        active_intervals.append("60")
                        if current_utc_hour % 2 == 0:
                            active_intervals.append("120")
                        if current_utc_hour % 4 == 0:
                            active_intervals.append("240")
                    
                    # We check how many of the currently expected ones are completed
                    expected_pairs = [(symbol, iv) for iv in active_intervals for symbol in SUPPORTED_SYMBOLS]
                    missing_pairs = [pair for pair in expected_pairs if pair not in completed_this_hour]
                    
                    if not missing_pairs:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] All active candle checks for {current_utc_hour:02d}:{current_utc_minute:02d} UTC boundary completed. Polling paused.")
                        hour_check_complete = True
                    else:
                        check_queue = missing_pairs

        forced_intervals = set()
        if is_startup:
            forced_intervals.update(["15", "30", "60", "120", "240"])
        # 3. Handle hot-reload queue inject
        if reloaded_intervals:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Hot-reloaded intervals detected: {reloaded_intervals}. Resetting processed timestamps and forcing check...")
            for iv_hr in reloaded_intervals:
                forced_intervals.add(iv_hr)
                # Clear interval specific tracker
                last_processed_timestamps[f"last_processed_{iv_hr}_ts"] = None
                # Clear per-symbol tracker
                for sym in SUPPORTED_SYMBOLS:
                    last_ts_key = f"last_processed_{sym}_{iv_hr}_ts"
                    last_processed_timestamps.pop(last_ts_key, None)
                    
            # Inject to queue
            check_queue.extend([(symbol, iv_hr) for iv_hr in reloaded_intervals for symbol in SUPPORTED_SYMBOLS])
        
        current_hour_dt = datetime(current_time_utc.year, current_time_utc.month, current_time_utc.day, current_time_utc.hour, tzinfo=timezone.utc)
        current_hour_ts = int(current_hour_dt.timestamp() * 1000)

        
        htf_cache = {}
        fetched_data = {}
        if check_queue:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Parallelise the BTC pre-fetch across all unique intervals to avoid blocking the main thread
            btc_hist_cache = {}
            unique_intervals = set(iv for sym, iv in check_queue)

            def _fetch_btc(iv_val):
                try:
                    return iv_val, get_history(symbol="BTCUSDT", interval=iv_val, limit=300)
                except Exception as _e:
                    log_event("WARNING", f"[BTC Pre-fetch] Error for {iv_val}: {type(_e).__name__} {_e}")
                    return iv_val, None

            with ThreadPoolExecutor(max_workers=min(len(unique_intervals), 8)) as btc_exec:
                btc_futures = {btc_exec.submit(_fetch_btc, iv_val): iv_val for iv_val in unique_intervals}
                try:
                    for fut in as_completed(btc_futures, timeout=25):
                        try:
                            iv_val, df_btc = fut.result()
                            if df_btc is not None:
                                btc_hist_cache[iv_val] = df_btc
                        except Exception as _e:
                            log_event("WARNING", f"[BTC Pre-fetch] Result error: {type(_e).__name__} {_e}")
                except Exception as _e:
                    log_event("WARNING", f"[BTC Pre-fetch] Batch timeout/incomplete: {type(_e).__name__} {_e}")

            # Pre-fetch derivatives (OI, funding, F&G) per-symbol once in parallel.
            # This avoids each of the 63 worker threads making its own blocking REST calls
            # to get_bybit_oi_history / get_bybit_funding_history (up to 50 paginated pages each).
            unique_symbols = list(set(sym for sym, iv in check_queue))
            deriv_cache = {}  # key: sym -> (df_oi, df_funding, df_fng)
            try:
                df_fng_global = get_fear_and_greed_history()
            except Exception:
                df_fng_global = pd.DataFrame(columns=["timestamp", "fear_greed"])

            def _fetch_deriv(sym):
                try:
                    df_oi = get_bybit_oi_history(symbol=sym, interval="15")
                except Exception:
                    df_oi = pd.DataFrame(columns=["timestamp", "open_interest"])
                try:
                    df_funding = get_bybit_funding_history(symbol=sym)
                except Exception:
                    df_funding = pd.DataFrame(columns=["timestamp", "funding_rate"])
                return sym, (df_oi, df_funding, df_fng_global)

            with ThreadPoolExecutor(max_workers=min(len(unique_symbols), 9)) as deriv_exec:
                deriv_futures = {deriv_exec.submit(_fetch_deriv, sym): sym for sym in unique_symbols}
                try:
                    for fut in as_completed(deriv_futures, timeout=45):
                        try:
                            sym, deriv_tuple = fut.result()
                            deriv_cache[sym] = deriv_tuple
                        except Exception as _e:
                            log_event("WARNING", f"[Deriv Pre-fetch] Result error: {type(_e).__name__} {_e}")
                except Exception as _e:
                    log_event("WARNING", f"[Deriv Pre-fetch] Batch timeout/incomplete: {type(_e).__name__} {_e}")
            
            def fetch_single_history(sym, interval_val):
                if sym == "BTCUSDT" and interval_val in btc_hist_cache:
                    df_raw_val = btc_hist_cache[interval_val]
                else:
                    df_raw_val = get_history(symbol=sym, interval=interval_val, limit=300, fail_if_stale=(interval_val not in forced_intervals))
                if df_raw_val is None or len(df_raw_val) < 2:
                    return sym, interval_val, None, None
                
                now_ms = time.time() * 1000.0
                interval_ms_val = int(interval_val) * 60 * 1000
                is_forced_val = interval_val in forced_intervals

                # Check fetch_ok attribute
                if not df_raw_val.attrs.get("fetch_ok", True) and not is_forced_val:
                    log_event("WARNING", f"[{sym} {interval_val}m] Candle fetch marked unsuccessful/stale (age={df_raw_val.attrs.get('last_bar_age_sec', 0):.1f}s). Skipping evaluation.")
                    return sym, interval_val, None, None

                df_completed_val = df_raw_val.iloc[:-1].copy()
                latest_completed_ts_val = int(df_completed_val.iloc[-1]["timestamp"])
                
                # Hard freshness assertion: Completed bar timestamp must be within 2.5 * interval_ms
                if (now_ms - latest_completed_ts_val) > (2.5 * interval_ms_val):
                    log_event("WARNING", f"[{sym} {interval_val}m] Stale candle rejected: completed bar age {(now_ms - latest_completed_ts_val)/1000:.1f}s exceeds threshold ({2.5*interval_ms_val/1000:.1f}s). Skipping evaluation.")
                    return sym, interval_val, df_raw_val, None

                last_ts_key_val = f"last_processed_{sym}_{interval_val}_ts"
                if not is_forced_val and last_processed_timestamps.get(last_ts_key_val) is not None:
                    if latest_completed_ts_val == last_processed_timestamps[last_ts_key_val]:
                        return sym, interval_val, df_raw_val, None
                
                # Fast check if candle is up to date before running heavy feature calculation
                expected_bar_open_val = (now_ms // interval_ms_val) * interval_ms_val
                expected_start_ms_val = expected_bar_open_val - interval_ms_val
                is_up_to_date_val = (latest_completed_ts_val >= expected_start_ms_val) or is_forced_val
                if not is_up_to_date_val:
                    return sym, interval_val, df_raw_val, None
                
                raw_attrs = dict(getattr(df_raw_val, "attrs", {}))
                df_target_val = df_completed_val.copy()
                if "is_synthetic" in df_raw_val.columns and "is_synthetic" not in df_target_val.columns:
                    df_target_val["is_synthetic"] = df_raw_val["is_synthetic"].iloc[:-1].values
                if sym != "BTCUSDT":
                    df_btc_val = btc_hist_cache.get(interval_val)
                    if df_btc_val is not None and len(df_btc_val) > 0:
                        df_btc_sub_val = df_btc_val[["timestamp", "close"]].rename(columns={"close": "close_btc"})
                        df_target_val = pd.merge(df_target_val, df_btc_sub_val, on="timestamp", how="left")
                        df_target_val["close_btc"] = df_target_val["close_btc"].ffill().bfill().fillna(df_target_val["close"])
                    else:
                        df_target_val["close_btc"] = df_target_val["close"]
                else:
                    df_target_val["close_btc"] = df_target_val["close"]
                
                # Use pre-fetched derivatives if available, otherwise fall back to live fetch
                if sym in deriv_cache:
                    df_oi_c, df_funding_c, df_fng_c = deriv_cache[sym]
                    df_btc_ref = btc_hist_cache.get(interval_val)
                    df_target_val = _merge_cached_derivatives(df_target_val, df_oi_c, df_funding_c, df_fng_c, df_btc=df_btc_ref, symbol=sym)
                else:
                    df_target_val = merge_derivatives_sentiment_features(df_target_val, symbol=sym, interval=interval_val)
                df_feat_val = add_features(df_target_val, symbol=sym, interval=interval_val)
                if df_feat_val is not None and not df_feat_val.empty:
                    for k, v in raw_attrs.items():
                        df_feat_val.attrs[k] = v
                    if "is_synthetic" in df_target_val.columns and "is_synthetic" not in df_feat_val.columns:
                        df_feat_val["is_synthetic"] = df_target_val["is_synthetic"]
                
                return sym, interval_val, df_raw_val, df_feat_val
 
            print(f"[Parallel Fetch] Querying {len(check_queue)} candle combinations in parallel...")
            t_start = time.time()
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_pair = {executor.submit(fetch_single_history, sym, iv): (sym, iv) for sym, iv in check_queue}
                try:
                    for fut in as_completed(future_to_pair, timeout=60):
                        sym, iv = future_to_pair[fut]
                        try:
                            _, _, df_raw_val, df_feat_val = fut.result()
                            if df_raw_val is not None:
                                fetched_data[(sym, iv)] = (df_raw_val, df_feat_val)
                        except Exception as e:
                            print(f"[Parallel Fetch] Error fetching {sym} {iv}: {type(e).__name__} {e}")
                except Exception as _e:
                    log_event("WARNING", f"[Parallel Fetch] Batch timeout/incomplete: {type(_e).__name__} {_e}")
            print(f"[Parallel Fetch] Completed in {time.time() - t_start:.2f} seconds.")
 
        # Update peak_balance dynamically for accurate real-time drawdown tracking (M-1)
        try:
            curr_equity = float(bot_state.get("simulated_balance", 80.0))
            if TRADE_MODE != "simulation":
                real_bal_val = get_real_bybit_balance_cached()
                if isinstance(real_bal_val, (int, float)) and real_bal_val > 0:
                    curr_equity = float(real_bal_val)
            peak_bal = float(bot_state.get("peak_balance", 0.0) or 0.0)
            if curr_equity > peak_bal:
                bot_state["peak_balance"] = curr_equity
                peak_bal = curr_equity

            # Finding #57: Hard 20% drawdown breach triggers emergency kill switch
            if peak_bal > 0:
                dd_pct = (peak_bal - curr_equity) / peak_bal
                max_dd_limit = getattr(config, "HARD_MAX_DRAWDOWN_LIMIT", 0.20)
                if dd_pct >= max_dd_limit:
                    log_event("CRITICAL", f"[TRADING_LOOP] Hard Drawdown Limit Breached: {dd_pct*100:.1f}% >= {max_dd_limit*100:.1f}% (Peak: ${peak_bal:.2f}, Current: ${curr_equity:.2f}). Triggering emergency kill switch.")
                    trigger_emergency_kill_switch(f"Hard 20% Drawdown Limit Breached ({dd_pct*100:.1f}%)")
                    time.sleep(10)
                    return
        except Exception as _ex_dd:
            log_event("WARNING", f"[TRADING_LOOP] Drawdown evaluation exception: {_ex_dd}")

        # Circuit Breaker Halt Guard (Micro-Trading Run Scoped)
        try:
            micro_ts_str = database.get_setting("micro_run_start_ts", "0")
            micro_ts = float(micro_ts_str) if micro_ts_str else 0.0
            if micro_ts == 0.0:
                micro_ts = time.time()
                database.set_setting("micro_run_start_ts", str(micro_ts))
            
            closed_all = database.get_completed_trades(limit=1000) if hasattr(database, "get_completed_trades") else []
            micro_trades = [t for t in closed_all if float(t.get("exit_time") or 0) >= micro_ts]
            persisted_trade_count = int(database.get_setting("closed_trade_count", "0") or 0)
            closed_trade_count = max(len(micro_trades), persisted_trade_count)
            cumulative_loss = -sum(float(t.get("venue_closed_pnl") or t.get("pnl_usd") or 0.0) for t in micro_trades if float(t.get("venue_closed_pnl") or t.get("pnl_usd") or 0.0) < 0)
            
            max_trades_cap = getattr(config, "MAX_LIVE_TRADES_CAP", 60)
            max_loss_cap = getattr(config, "MAX_LIVE_LOSS_CAP", 15.0)
            is_persisted_stopped = database.get_setting("bot_stopped") == "True"
            
            cb_ok, cb_reason = circuit_breaker.evaluate_micro_run_caps(closed_trade_count, cumulative_loss, max_trades_cap, max_loss_cap)
            if is_persisted_stopped or not cb_ok:
                bot_state["bot_running"] = False
                bot_state["bot_stopped"] = True
                database.set_setting("bot_running", "False")
                database.set_setting("bot_stopped", "True")
                log_event("WARNING", f"[TRADING_LOOP] Hard Circuit Breaker Triggered ({cb_reason}, Micro Trades: {closed_trade_count}/{max_trades_cap}, Micro Loss: ${cumulative_loss:.2f}/${max_loss_cap:.2f}, Persisted Stopped: {is_persisted_stopped}) — bot halted.")
                if not cb_ok:
                    try:
                        kill_reason = f"Hard Circuit Breaker ({cb_reason})"
                        trigger_emergency_kill_switch(kill_reason)
                    except Exception as ex_ks:
                        log_event("ERROR", f"[TRADING_LOOP] Error invoking trigger_emergency_kill_switch: {ex_ks}")
                time.sleep(10)
                return

            # Finding #121 & Finding #166 (#96): Production Circuit Breaker System Health Check
            raw_lat = state_manager.get("last_api_latency_ms", bot_state.get("last_api_latency_ms"))
            api_latency_ms = float(raw_lat) if raw_lat is not None else 100.0

            raw_bal_ts = bot_state.get("last_balance_sync_ts")
            if raw_bal_ts is None:
                raw_bal_ts = state_manager.get("last_balance_sync_ts")
            if raw_bal_ts is None and _last_balance_fetch > 0:
                raw_bal_ts = _last_balance_fetch
            bal_sync_ts = float(raw_bal_ts) if raw_bal_ts is not None else 0.0

            raw_inf = bot_state.get("last_inference_latency_ms")
            inference_lat_ms = float(raw_inf) if raw_inf is not None else 50.0

            db_healthy = True
            try:
                with database.db_lock:
                    c = database.get_db_connection()
                    c.execute("SELECT 1;").fetchone()
                    c.close()
            except Exception:
                db_healthy = False

            sh_ok, sh_reason = circuit_breaker.evaluate_system_health(
                exchange_latency_ms=api_latency_ms,
                last_balance_sync_ts=bal_sync_ts,
                db_healthy=db_healthy,
                inference_latency_ms=inference_lat_ms
            )
            if not sh_ok:
                log_event("WARNING", f"[TRADING_LOOP] System Health Circuit Breaker Triggered ({sh_reason}) — halting signal evaluation.")
                time.sleep(5)
                return
        except Exception as ex_cb:
            log_event("CRITICAL", f"[TRADING_LOOP] Circuit breaker check exception (failing closed): {ex_cb}")
            time.sleep(5)
            return

        just_opened_symbols = set()  # Symbols opened this cycle — block duplicates regardless of Bybit sync latency
        for symbol, iv in check_queue:
            # Global active execution guard check
            with active_execution_lock:
                in_execution = symbol in active_execution_symbols
            if in_execution:
                print(f"[{symbol} {iv}m] Skip signal check: A live order placement is currently executing in the background.")
                continue

            tf = tf_map[iv]
            active_trade_key = f"active_trade_{tf}"
            with active_trades_lock:
                active_trades_list = bot_state.get(active_trade_key, [])
                active_trades_list = filter_unprocessed_active_trades(active_trades_list)
                bot_state[active_trade_key] = active_trades_list
                
            if (symbol, iv) not in fetched_data:
                log_event("WARNING", f"[{symbol} {iv}m] Market data missing from fetched_data batch. Skipping evaluation.")
                continue
            df_raw, df = fetched_data[(symbol, iv)]
            if df is None or len(df) == 0:
                log_event("WARNING", f"[{symbol} {iv}m] DataFrame is empty or None. Skipping evaluation.")
                continue
                
            try:
                df_completed = df.copy()
                latest_completed_ts = int(df_completed.iloc[-1]["timestamp"])
 
                last_ts_key = f"last_processed_{symbol}_{iv}_ts"
                if last_processed_timestamps.get(last_ts_key) is None:
                    last_processed_timestamps[last_ts_key] = 0
                    print(f"Initialized completed candle timestamp tracking for {symbol} on {iv}m: {get_local_time_str(latest_completed_ts/1000)}")
 
                # Hard candle freshness check: Completed candle must have closed recently
                now_ms = time.time() * 1000.0
                is_fresh, candle_age_sec, max_allowed_age_sec = is_candle_fresh(latest_completed_ts, iv, now_ms=now_ms)
                if not is_fresh:
                    log_event("INFO", f"[{symbol} {iv}m] Stale candle skipped: completed bar closed {candle_age_sec:.1f}s ago (max allowed: {max_allowed_age_sec:.1f}s). Skipping.")
                    completed_this_hour.add((symbol, iv))
                    continue

                completed_this_hour.add((symbol, iv))
                
                if latest_completed_ts != last_processed_timestamps[last_ts_key]:
                    last_processed_timestamps[last_ts_key] = latest_completed_ts
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Completed {symbol} {iv}-minute candle evaluation triggered (TS: {latest_completed_ts})")
                    log_event("INFO", f"[{symbol} {iv}m] Evaluation start — candle {latest_completed_ts}")
                    rec = DecisionRecord(symbol=symbol, interval=str(iv))
                    rec.candle_timestamp = latest_completed_ts
                    decision_ts = time.time()
                    status_msg = "Abstain"
                    placed = False
                    async_spawned = False
                    exp_edge_bps = None
                    exp_r_val = None
                    cost_bps = None
                    position_size_usd = None
                    leverage_val = None
                    tcm_cost_bps = None
                    mhi_val = None
                    wallet_exceeded = False
                    bybit_success = False
                    try:
                    
                        latest_candle = df.iloc[-1]
                        
                        # S-3 Data Continuity Guard: Refuse evaluation if candle series has unserviceable gaps (> 3 bars)
                        synthetic_count = int(df["is_synthetic"].sum()) if "is_synthetic" in df.columns else int(getattr(df, "attrs", {}).get("synthetic_bar_count", 0))
                        gap_exceeded = False
                        max_g = 0
                        if getattr(df, "attrs", {}).get("gap_exceeded", False) or synthetic_count > 5:
                            gap_exceeded = True
                            max_g = max(int(getattr(df, "attrs", {}).get("max_consecutive_synthetic_bars", 0)), synthetic_count)
                        elif "timestamp" in df.columns and len(df) > 1:
                            expected_step_ms = int(iv) * 60 * 1000
                            ts_diffs = df["timestamp"].diff().dropna()
                            max_diff = ts_diffs.max() if len(ts_diffs) > 0 else expected_step_ms
                            if max_diff > expected_step_ms * 1.5:
                                max_g = int(round(max_diff / expected_step_ms)) - 1
                                if max_diff > expected_step_ms * 3.5:
                                    gap_exceeded = True
                        if not gap_exceeded and synthetic_count > 0:
                            max_g = max(max_g, synthetic_count)
                        
                        if gap_exceeded:
                            print(f"[{symbol} {iv}m Gap Guard] Window contains {max_g} consecutive missing bars (> 3) or {synthetic_count} synthetic bars (> 5). Abstaining from signal evaluation (Fail-Closed).")
                            log_event("WARNING", f"[{symbol} {iv}m] Unserviceable gap ({max_g} bars, synthetic: {synthetic_count}) in candle history — abstaining (Fail-Closed)")
                            continue
                    
                        from data_quality_engine import DataQualityEngine
                        from market_data_quality import MarketDataQualityMonitor

                        expected_step_ms = int(iv) * 60 * 1000
                        raw_max_diff = float(df["timestamp"].diff().dropna().max()) if (len(df) > 1 and "timestamp" in df.columns) else float(expected_step_ms)
                        excess_gap_sec = max(0.0, float((raw_max_diff - expected_step_ms) / 1000.0))
                        dq_res = DataQualityEngine().evaluate_data_quality(
                            missing_candles_count=int(max_g),
                            timestamp_gap_seconds=float(excess_gap_sec),
                            stale_feed_seconds=max(0.0, float(candle_age_sec)),
                            zero_price_detected=bool(float(latest_candle.get("close", 0.0)) <= 0.0)
                        )
                        if dq_res.get("severity") in ["CRITICAL", "HIGH"]:
                            log_event(dq_res.get("severity"), f"[{symbol} {iv}m DataQualityEngine] {dq_res.get('detail')} — Action: {dq_res.get('action')}. Abstaining.")
                            continue

                        if "_mdq_monitors" not in bot_state:
                            bot_state["_mdq_monitors"] = {}
                        mdq_key = f"{symbol}_{iv}"
                        if mdq_key not in bot_state["_mdq_monitors"]:
                            bot_state["_mdq_monitors"][mdq_key] = MarketDataQualityMonitor()
                        mdq = bot_state["_mdq_monitors"][mdq_key]
                        with _time_offset_lock:
                            srv_offset = _cached_time_offset if _cached_time_offset is not None else 0.0
                        mdq_res = mdq.evaluate_feed_health(
                            last_candle_timestamp=float((latest_completed_ts + expected_step_ms) / 1000.0),
                            server_time_ms=float(now_ms + srv_offset),
                            client_time_ms=float(now_ms),
                            ws_connected=bool(ws_connected),
                            interval_sec=float(int(iv) * 60)
                        )
                        if mdq_res.get("health_tier") == "RED" or mdq_res.get("tier") == "RED" or not mdq_res.get("trading_allowed", True):
                            reasons_str = mdq_res.get("reasons") or mdq_res.get("feed_status")
                            log_event("WARNING", f"[{symbol} {iv}m MDQ] Feed health RED: {reasons_str}. Abstaining.")
                            continue
                    
                        # Dynamic Regime Routing based on GMM Unsupervised Classifier
                        regime = classify_market_regime(df, interval=iv)
                        adx_regime = latest_candle["ADX"]

                        # Finding #84: Explicitly reset candidate loop-scoped governance state to prevent iteration leaks
                        _mdata = None
                        manifest_load_error = None
                        is_promoted_flag = None
                        abstain_reason = None
                    
                        # Ensure models for interval iv are loaded into memory on-demand
                        _tf = models_by_interval.get(iv, {})
                        if _tf.get("_fully_denied"):
                            pred_entry_dict = {
                                "symbol": symbol,
                                "interval": str(iv),
                                "predicted_change": 0.0,
                                "predicted_price": float(latest_candle.get("close", 0.0)),
                                "direction": "Offline (Denied)",
                                "raw_confidence": 0.0,
                                "calibrated_confidence": 0.0,
                                "manifest_mcc": 0.0,
                                "signal_source": "GOVERNANCE_DENIED",
                                "is_fallback": False,
                                "status": "Offline (Governance Denied)"
                            }
                            for k_suffix in [str(tf), str(iv)]:
                                bot_state[f"regime_{symbol}_{k_suffix}"] = regime
                                bot_state[f"adx_{symbol}_{k_suffix}"] = adx_regime
                                bot_state[f"latest_prediction_{symbol}_{k_suffix}"] = pred_entry_dict
                                if symbol == "BTCUSDT" or symbol == bot_state.get("active_symbol", "BTCUSDT"):
                                    bot_state[f"regime_{k_suffix}"] = regime
                                    bot_state[f"adx_{k_suffix}"] = adx_regime
                                    bot_state[f"latest_prediction_{k_suffix}"] = pred_entry_dict
                            continue
                        if not any(_tf.get(_r, {}).get("trend") for _r in ("trending", "ranging")):
                            load_model_weights(iv)
                            _tf = models_by_interval.get(iv, {})
                            if _tf.get("_fully_denied"):
                                pred_entry_dict = {
                                    "symbol": symbol,
                                    "interval": str(iv),
                                    "predicted_change": 0.0,
                                    "predicted_price": float(latest_candle.get("close", 0.0)),
                                    "direction": "Offline (Denied)",
                                    "raw_confidence": 0.0,
                                    "calibrated_confidence": 0.0,
                                    "manifest_mcc": 0.0,
                                    "signal_source": "GOVERNANCE_DENIED",
                                    "is_fallback": False,
                                    "status": "Offline (Governance Denied)"
                                }
                                for k_suffix in [str(tf), str(iv)]:
                                    bot_state[f"regime_{symbol}_{k_suffix}"] = regime
                                    bot_state[f"adx_{symbol}_{k_suffix}"] = adx_regime
                                    bot_state[f"latest_prediction_{symbol}_{k_suffix}"] = pred_entry_dict
                                    if symbol == "BTCUSDT" or symbol == bot_state.get("active_symbol", "BTCUSDT"):
                                        bot_state[f"regime_{k_suffix}"] = regime
                                        bot_state[f"adx_{k_suffix}"] = adx_regime
                                        bot_state[f"latest_prediction_{k_suffix}"] = pred_entry_dict
                                continue

                        if iv in models_by_interval:
                            models_tf = models_by_interval[iv]
                            from config import ENABLE_DYNAMIC_REGIME_ROUTING, DYNAMIC_REGIME_ROUTING_INTERVALS
                            is_dynamic_routing = ENABLE_DYNAMIC_REGIME_ROUTING or (str(iv) in DYNAMIC_REGIME_ROUTING_INTERVALS)
                            if is_dynamic_routing:
                                regime_key = "ranging" if "Ranging" in str(regime) else "trending"
                                served_regime = regime
                            else:
                                regime_key = "trending"
                                served_regime = f"Trending (Universal Baseline, Market: {regime})"

                            m_price = models_tf.get(regime_key, {}).get("price")
                            m_trend = models_tf.get(regime_key, {}).get("trend")
                            m_cal = models_tf.get(regime_key, {}).get("calibrator")
                            m_meta = models_tf.get(regime_key, {}).get("meta")
                            feat_list = models_tf.get(f"selected_features_{regime_key}") or models_tf.get("selected_features")

                            active_model_price = m_price
                            active_model_trend = m_trend
                            active_calibrator = m_cal
                            active_meta_model = m_meta
                            regime_name = f"{served_regime} (GMM)" if ENABLE_DYNAMIC_REGIME_ROUTING else served_regime

                            # C-1 Predictive Floor & Holdout Out-Of-Sample Governance Check
                            from config import (
                                MODEL_GOVERNANCE, TIMEFRAME_MIN_MCC, TIMEFRAME_MIN_BAL_ACC,
                                TIMEFRAME_MIN_HOLDOUT_MCC, TIMEFRAME_MIN_HOLDOUT_BAL_ACC
                            )
                            min_mcc_floor = TIMEFRAME_MIN_MCC.get(str(iv), TIMEFRAME_MIN_MCC.get("default", MODEL_GOVERNANCE.get("min_mcc", 0.05)))
                            min_bal_acc_floor = TIMEFRAME_MIN_BAL_ACC.get(str(iv), TIMEFRAME_MIN_BAL_ACC.get("default", MODEL_GOVERNANCE.get("min_balanced_accuracy", 0.36)))
                            min_holdout_mcc_floor = TIMEFRAME_MIN_HOLDOUT_MCC.get(str(iv), TIMEFRAME_MIN_HOLDOUT_MCC.get("default", MODEL_GOVERNANCE.get("min_holdout_mcc", 0.035)))
                            min_holdout_bal_acc_floor = TIMEFRAME_MIN_HOLDOUT_BAL_ACC.get(str(iv), TIMEFRAME_MIN_HOLDOUT_BAL_ACC.get("default", MODEL_GOVERNANCE.get("min_holdout_balanced_accuracy", 0.355)))

                            mcc_val = getattr(active_model_trend, "manifest_mcc", None) or models_tf.get(regime_key, {}).get("manifest_mcc") or models_tf.get("manifest_mcc")
                            mcc_min_val = getattr(active_model_trend, "manifest_mcc_min", None) or models_tf.get(regime_key, {}).get("manifest_mcc_min") or models_tf.get("manifest_mcc_min")
                            bal_acc_val = getattr(active_model_trend, "manifest_bal_acc", None) or models_tf.get(regime_key, {}).get("manifest_bal_acc") or models_tf.get("manifest_bal_acc")
                            holdout_mcc_val = getattr(active_model_trend, "holdout_mcc", None) or models_tf.get(regime_key, {}).get("holdout_mcc") or models_tf.get("holdout_mcc")
                            holdout_bal_acc_val = getattr(active_model_trend, "holdout_bal_acc", None) or models_tf.get(regime_key, {}).get("holdout_bal_acc") or models_tf.get("holdout_bal_acc")
                            holdout_ci95_low = getattr(active_model_trend, "holdout_ci95_low", None) or models_tf.get(regime_key, {}).get("holdout_ci95_low") or models_tf.get("holdout_ci95_low")
                            is_promoted_flag = getattr(active_model_trend, "promoted", None) if getattr(active_model_trend, "promoted", None) is not None else models_tf.get(regime_key, {}).get("promoted")

                            man_path = f"ensemble_{regime_key}_trend_{iv}_manifest.json"
                            if os.path.exists(man_path):
                                try:
                                    with open(man_path, "r") as mf:
                                        _mdata = json.load(mf)
                                        if mcc_val is None:
                                            mcc_val = extract_metric(_mdata, ["manifest_mcc"], ["cv_metrics", "mcc", "mean"], ["metrics", "mcc"])
                                        if mcc_min_val is None:
                                            mcc_min_val = extract_metric(_mdata, ["manifest_mcc_min"], ["cv_metrics", "mcc", "min"], ["metrics", "mcc_min"])
                                        if bal_acc_val is None:
                                            bal_acc_val = extract_metric(_mdata, ["manifest_bal_acc"], ["cv_metrics", "balanced_accuracy", "mean"], ["metrics", "balanced_accuracy"])
                                        if holdout_mcc_val is None:
                                            holdout_mcc_val = extract_metric(_mdata, ["holdout_mcc"], ["cv_metrics", "holdout_mcc"], ["metrics", "holdout_mcc"])
                                        if holdout_bal_acc_val is None:
                                            holdout_bal_acc_val = extract_metric(_mdata, ["holdout_balanced_accuracy"], ["cv_metrics", "holdout_balanced_accuracy"], ["metrics", "holdout_balanced_accuracy"])
                                        if holdout_ci95_low is None:
                                            _ci = _mdata.get("cv_metrics", {}).get("holdout_mcc_ci95") if isinstance(_mdata.get("cv_metrics"), dict) else None
                                            if isinstance(_ci, (list, tuple)) and len(_ci) >= 1:
                                                holdout_ci95_low = _ci[0]
                                        if is_promoted_flag is None:
                                            is_promoted_flag = _mdata.get("promoted", False)  # Finding #84: Fail-closed default
                                except Exception as mf_err:
                                    log_event("CRITICAL", f"Failed to load manifest {man_path}: {mf_err}")
                                    manifest_load_error = str(mf_err)
                            else:
                                manifest_load_error = f"Manifest {man_path} missing on disk"

                            abstain_reason = None
                            if "manifest_load_error" in locals() and manifest_load_error:
                                abstain_reason = f"Corrupted or unreadable manifest {man_path}: {manifest_load_error}"
                            elif m_price is None or m_trend is None or not feat_list:
                                abstain_reason = f"{served_regime} model offline"
                            elif active_calibrator is None or (isinstance(active_calibrator, dict) and active_calibrator.get("is_fallback", False)):
                                abstain_reason = f"{served_regime} calibrator missing or fallback (Fail-Closed)"
                            elif _mdata is not None and isinstance(_mdata, dict):
                                from config import is_manifest_degenerate
                                is_deg, deg_reason = is_manifest_degenerate(_mdata)
                                if is_deg:
                                    abstain_reason = f"Degenerate manifest: {deg_reason}"
                            
                            if not abstain_reason:
                                if mcc_val is not None and mcc_val < min_mcc_floor:
                                    abstain_reason = f"MCC {mcc_val:.4f} < floor {min_mcc_floor}"
                                elif mcc_min_val is not None and mcc_min_val < -0.05:
                                    abstain_reason = f"min CV MCC {mcc_min_val:.4f} < -0.05"
                                elif bal_acc_val is not None and bal_acc_val < min_bal_acc_floor:
                                    abstain_reason = f"BalAcc {bal_acc_val:.4f} < floor {min_bal_acc_floor}"
                                elif holdout_mcc_val is None or holdout_mcc_val < min_holdout_mcc_floor:
                                    abstain_reason = f"Holdout MCC ({holdout_mcc_val}) < floor {min_holdout_mcc_floor} or missing"
                                elif holdout_bal_acc_val is None or holdout_bal_acc_val < min_holdout_bal_acc_floor:
                                    abstain_reason = f"Holdout BalAcc ({holdout_bal_acc_val}) < floor {min_holdout_bal_acc_floor} or missing"
                                elif holdout_ci95_low is not None and holdout_ci95_low < -0.05:
                                    abstain_reason = f"Holdout CI95 lower bound {holdout_ci95_low:.4f} < -0.05"
                                elif is_promoted_flag is False:
                                    abstain_reason = f"{served_regime} model manifest promoted=False"

                            if abstain_reason:
                                log_event("WARNING", f"[{symbol} {iv}m ({regime_key})] {abstain_reason}. Abstaining.")
                                pred_entry_dict = {
                                    "symbol": symbol,
                                    "interval": str(iv),
                                    "predicted_change": 0.0,
                                    "predicted_price": float(latest_candle.get("close", 0.0)),
                                    "direction": "Abstain",
                                    "raw_confidence": 0.0,
                                    "calibrated_confidence": 0.0,
                                    "manifest_mcc": mcc_val,
                                    "signal_source": "GOVERNANCE_ABSTAIN",
                                    "is_fallback": False,
                                    "status": f"Abstain ({abstain_reason})"
                                }
                                for k_suffix in [str(tf), str(iv)]:
                                    bot_state[f"regime_{symbol}_{k_suffix}"] = regime_name
                                    bot_state[f"adx_{symbol}_{k_suffix}"] = adx_regime
                                    bot_state[f"latest_prediction_{symbol}_{k_suffix}"] = pred_entry_dict
                                    if symbol == "BTCUSDT" or symbol == bot_state.get("active_symbol", "BTCUSDT"):
                                        bot_state[f"regime_{k_suffix}"] = regime_name
                                        bot_state[f"adx_{k_suffix}"] = adx_regime
                                        bot_state[f"latest_prediction_{k_suffix}"] = pred_entry_dict
                                continue
                            
                            # C-1: Preserve strict train/serve feature distribution consistency
                            # Remove ad-hoc inference-time feature multiplier scaling
                            bybit_success = False
                            actual_qty = 0.0
                            raw_qty = 0.0
                            bybit_order_id = None
                            bybit_scale_out_order_id = None
                            latest_candle_weighted = latest_candle.copy()
                            if hasattr(latest_candle_weighted.index, "duplicated") and latest_candle_weighted.index.duplicated().any():
                                latest_candle_weighted = latest_candle_weighted[~latest_candle_weighted.index.duplicated(keep="first")]

                            from ensemble import get_model_feature_names
                            _exp_names = get_model_feature_names(active_model_trend)
                            if _exp_names and not all(str(n).startswith("Column_") for n in _exp_names):
                                _features_to_use = _exp_names
                            elif feat_list is not None:
                                _features_to_use = feat_list
                            else:
                                from core import features as master_features
                                _features_to_use = master_features

                            X_live_full, missing_model_features = check_live_feature_integrity(latest_candle_weighted, _features_to_use)
                            if missing_model_features:
                                log_event("WARNING", f"[{symbol} {iv}m] Live inference missing {len(missing_model_features)} expected model features: {missing_model_features[:5]}. Abstaining (fail-closed).")
                                rec.outcome = "REJECTED"
                                rec.reason_code = ReasonCode.PREDICTION_ERROR
                                rec.reject_reason = f"Missing {len(missing_model_features)} model features: {','.join(missing_model_features[:10])}"
                                write_decision(rec)
                                continue

                            try:
                                X_live = _slice_model_input(active_model_trend, X_live_full)
                            except Exception as ex_slice:
                                log_event("WARNING", f"[{symbol} {iv}m] Feature slice failed: {ex_slice}. Abstaining (fail-closed).")
                                rec.outcome = "REJECTED"
                                rec.reason_code = ReasonCode.PREDICTION_ERROR
                                rec.reject_reason = f"Feature slice failed: {ex_slice}"
                                write_decision(rec)
                                continue

                            # Item A: Interval-Specific Ensemble Weights (LightGBM & CatBoost-heavy for 15M/30M scalp accuracy)
                            if str(iv) == "15":
                                ensemble_weights = [0.10, 0.45, 0.45]
                            elif str(iv) == "30":
                                ensemble_weights = [0.15, 0.42, 0.43]
                            else:
                                ensemble_weights = [0.30, 0.20, 0.50] if "Trending" in regime_name else [0.30, 0.50, 0.20]
                        
                            t_inf_start = time.time()
                            try:
                                pred_pct = float(active_model_price.predict(X_live, weights=ensemble_weights)[0])
                                pred_change = pred_pct * float(latest_candle["close"])
                                predicted_price = float(latest_candle["close"]) + pred_change
                            
                                # 3-class probabilities with Conformal Uncertainty estimation
                                if hasattr(active_model_trend, "predict_with_uncertainty"):
                                    probs_arr, conformal_unc_score, conformal_is_uncertain = active_model_trend.predict_with_uncertainty(X_live, weights=ensemble_weights)
                                    probs = probs_arr[0]
                                else:
                                    probs = active_model_trend.predict_proba(X_live, weights=ensemble_weights)[0]
                                    conformal_unc_score = 0.0
                                    conformal_is_uncertain = False

                                # Finding #166 (Finding #96): Record live model inference latency
                                inf_lat_ms = (time.time() - t_inf_start) * 1000.0
                                bot_state["last_inference_latency_ms"] = inf_lat_ms
                                try:
                                    state_manager["last_inference_latency_ms"] = inf_lat_ms
                                except Exception as ex_inf:
                                    log_event("WARNING", f"state_manager latency notice: {ex_inf}")
                            except Exception as pred_err:
                                import traceback
                                err_msg = f"[{symbol} {iv}m CRITICAL PREDICTION ERROR] {type(pred_err).__name__}: {pred_err}"
                                log_event("WARNING", f"{err_msg}\n{traceback.format_exc()}")
                                print(f"{err_msg}. Aborting trade entry (Fail-Closed).")
                                status_msg = f"Skipped (Prediction Error: {type(pred_err).__name__} {pred_err})"
                                rec.reject_reason = status_msg
                                all_pass = False
                                continue

                            # Degenerate live prediction detector (requires statistically sufficient sample size N >= 30 per symbol)
                            _model_key = f"{symbol}_ensemble_{regime_key}_trend_{iv}"
                            if _model_key not in _recent_runtime_argmax:
                                from collections import deque
                                _recent_runtime_argmax[_model_key] = deque(maxlen=50)
                            _recent_runtime_argmax[_model_key].append(int(np.argmax(probs)))
                            if len(_recent_runtime_argmax[_model_key]) >= 30:
                                _shares = np.bincount(_recent_runtime_argmax[_model_key], minlength=3) / len(_recent_runtime_argmax[_model_key])
                                # One-sided directional collapse: exclusively Bearish (0) or Bullish (2) dominating
                                if len(_shares) >= 3 and (_shares[0] >= 0.95 or _shares[2] >= 0.95):
                                    log_event("WARNING", f"[{_model_key}] degenerate one-sided directional predictor: {_shares.round(3)} over {len(_recent_runtime_argmax[_model_key])} predictions — abstaining (Fail-Closed)")
                                    status_msg = f"Skipped (Degenerate Prediction: {_model_key})"
                                    continue
                        
                            if len(probs) >= 3:
                                prob_bearish = float(probs[0])
                                prob_neutral = float(probs[1])
                                prob_bullish = float(probs[2])
                            elif len(probs) == 2:
                                prob_bearish = float(probs[0])
                                prob_neutral = 0.0
                                prob_bullish = float(probs[1])
                            else:
                                val = float(probs[0])
                                prob_bearish = val if val < 0.5 else 0.0
                                prob_neutral = 0.0
                                prob_bullish = val if val >= 0.5 else 0.0

                            from ensemble import resolve_direction
                            ml_trend, ml_confidence = resolve_direction(probs, interval=str(iv))
                            raw_class_prob = prob_bullish if ml_trend == "Bullish" else (prob_bearish if ml_trend == "Bearish" else prob_neutral)

                            # 1. Calibrate the directional confidence (matching economic 2-class break-even scale)
                            calibrated_confidence = ml_confidence if ml_trend in ["Bullish", "Bearish"] else raw_class_prob
                            calibrator = active_calibrator
                            is_fallback_signal = False
                            signal_source_type = "ML_ENSEMBLE"
                            if calibrator is not None and ml_trend in ["Bullish", "Bearish"]:
                                from tools.beta_calibrator import calibrate_probability, is_calibrator_viable
                                if not is_calibrator_viable(calibrator):
                                    log_event("WARNING", f"[{symbol} {iv}m] Calibrator failed viability check (Fail-Closed). Abstaining.")
                                    is_fallback_signal = True
                                    signal_source_type = "UNVIABLE_CALIBRATOR"
                                    calibrated_confidence = 0.0
                                    ml_trend = "Neutral"
                                else:
                                    calibrated_confidence = calibrate_probability(ml_confidence, calibrator)
                                    method_name = calibrator.get("scaling_method", "calibration")
                                    print(f"[{symbol} {iv}m {method_name}] Dir Mass: {ml_confidence*100:.2f}% (Raw Class: {raw_class_prob*100:.2f}%) -> Calibrated: {calibrated_confidence*100:.2f}%")

                            # 2. Decision-layer neutral discount (default 0.0 to prevent double penalty)
                            neutral_coeff = getattr(config, "NEUTRAL_PENALTY_COEFFICIENT", 0.0)
                            if ml_trend in ("Bullish", "Bearish") and neutral_coeff > 0.0:
                                calibrated_confidence = min(0.95, calibrated_confidence * (1.0 - prob_neutral * neutral_coeff))

                            # Clip calibrated output away from 0.0 & 1.0 saturation boundaries (EPS = 1e-3)
                            calibrated_confidence = float(np.clip(calibrated_confidence, 1e-3, 1.0 - 1e-3))

                            # Governance MCC / Predictive Floor Check
                            from config import MODEL_GOVERNANCE, TIMEFRAME_MIN_MCC, TIMEFRAME_MIN_BAL_ACC, TIMEFRAME_MIN_HOLDOUT_MCC
                            min_mcc_floor = TIMEFRAME_MIN_MCC.get(str(iv), TIMEFRAME_MIN_MCC.get("default", MODEL_GOVERNANCE.get("min_mcc", 0.05)))
                            min_holdout_mcc_floor = TIMEFRAME_MIN_HOLDOUT_MCC.get(str(iv), TIMEFRAME_MIN_HOLDOUT_MCC.get("default", MODEL_GOVERNANCE.get("min_holdout_mcc", 0.035)))
                            _manifest_mcc_val = getattr(m_trend, "manifest_mcc", None)
                            _holdout_mcc_val = getattr(m_trend, "holdout_mcc", None)
                            if _manifest_mcc_val is None:
                                _manifest_mcc_val = locals().get("manifest_info", {}).get("manifest_mcc")
                            if _holdout_mcc_val is None:
                                _holdout_mcc_val = locals().get("manifest_info", {}).get("holdout_mcc")

                            if _manifest_mcc_val is not None and _manifest_mcc_val < min_mcc_floor:
                                log_event("WARNING", f"[{symbol} {iv}m] Model MCC ({_manifest_mcc_val:.4f}) below governance floor ({min_mcc_floor}). ABSTAIN.")
                                continue
                            if _holdout_mcc_val is not None and _holdout_mcc_val < min_holdout_mcc_floor:
                                log_event("WARNING", f"[{symbol} {iv}m] Model Holdout MCC ({_holdout_mcc_val:.4f}) below holdout floor ({min_holdout_mcc_floor}). ABSTAIN.")
                                continue

                            expected_pct_change = (abs(pred_change) / latest_candle["close"]) * 100

                            reg_dict = models_tf.get(regime_key, {})
                            model_ver = reg_dict.get("model_version") or getattr(m_trend, "model_version", None)
                            git_sha_val = reg_dict.get("git_sha")
                            manifest_schema_val = reg_dict.get("manifest_schema_version")
                            feature_contract_val = reg_dict.get("feature_contract_hash")
                            cal_ver = reg_dict.get("calibrator_version")
                            cal_ece = reg_dict.get("calibrator_ece")

                            pred_entry_dict = {
                                "timestamp": float(time.time()),
                                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "symbol": symbol,
                                "interval": str(iv),
                                "predicted_change": pred_change,
                                "predicted_price": predicted_price,
                                "direction": ml_trend,
                                "raw_confidence": ml_confidence,
                                "calibrated_confidence": calibrated_confidence,
                                "manifest_mcc": _manifest_mcc_val,
                                "signal_source": signal_source_type,
                                "is_fallback": is_fallback_signal,
                                "model_version": model_ver,
                                "git_sha": git_sha_val,
                                "manifest_schema_version": manifest_schema_val,
                                "feature_contract_hash": feature_contract_val,
                                "calibrator_version": cal_ver,
                                "calibrator_ece": cal_ece
                            }
                            for k_suffix in [str(tf), str(iv)]:
                                bot_state[f"regime_{symbol}_{k_suffix}"] = regime_name
                                bot_state[f"adx_{symbol}_{k_suffix}"] = adx_regime
                                bot_state[f"latest_prediction_{symbol}_{k_suffix}"] = pred_entry_dict
                                if symbol == "BTCUSDT" or symbol == bot_state.get("active_symbol", "BTCUSDT"):
                                    bot_state[f"regime_{k_suffix}"] = regime_name
                                    bot_state[f"adx_{k_suffix}"] = adx_regime
                                    bot_state[f"latest_prediction_{k_suffix}"] = pred_entry_dict

                            # F-1: Observe MCC Leverage Qualification status at prediction time
                            mcc_thresh = getattr(config, "MCC_LEVERAGE_QUALIFICATION_THRESHOLD", 0.15)
                            if _manifest_mcc_val is None or _manifest_mcc_val < mcc_thresh:
                                cons_caps = getattr(config, "CONSERVATIVE_LEVERAGE_CAPS", {})
                                mcc_cap = cons_caps.get(symbol, cons_caps.get("default", 3.0))
                                _mcc_str = f"{_manifest_mcc_val:.4f}" if _manifest_mcc_val is not None else "unavailable"
                                log_event("INFO", f"[{symbol} {iv}m F-1 Leverage Guard] Model MCC ({_mcc_str} < {mcc_thresh:.4f}) — Max leverage clamped to {mcc_cap:.1f}x.")

                            rec.regime = str(regime_name)
                            rec.direction = str(ml_trend)
                            rec.raw_confidence = float(ml_confidence)
                            rec.calibrated_conf = float(calibrated_confidence)
                            rec.signal_source = str(pred_entry_dict.get("signal_source") or "UNSET")
                            rec.is_fallback = int(pred_entry_dict.get("is_fallback", False))
                            rec.directional_mass_passed = 1 if str(iv) not in ["15", "30"] or ml_trend in ["Bullish", "Bearish"] else 0
                            rec.is_calibrated = 1 if (active_calibrator is not None and ml_trend in ["Bullish", "Bearish"]) else 0
                            rec.is_floor_scaled = 1 if (locals().get('is_floor_scaled', False)) else 0
                            rec._inputs["predicted_change"] = float(pred_change)
                            rec._inputs["expected_pct_change"] = float(pred_change)
                            if model_ver:
                                rec.model_version = str(model_ver)
                            if git_sha_val:
                                rec.git_sha = str(git_sha_val)
                            if manifest_schema_val is not None:
                                try:
                                    rec.manifest_schema = int(manifest_schema_val)
                                except (ValueError, TypeError) as _sch_err:
                                    log_event("DEBUG", f"manifest_schema conversion error: {_sch_err}")
                            if feature_contract_val:
                                rec.feature_hash = str(feature_contract_val)
                            if cal_ver:
                                rec.calibrator_version = str(cal_ver)
                            if cal_ece is not None:
                                try:
                                    rec.calibrator_ece = float(cal_ece)
                                except (ValueError, TypeError) as _ece_err:
                                    log_event("DEBUG", f"calibrator_ece conversion error: {_ece_err}")

                            log_event("INFO", f"[{symbol} {iv}m] probs=[{prob_bearish:.4f}, {prob_neutral:.4f}, {prob_bullish:.4f}] ml_conf={ml_confidence:.4f} cal={calibrated_confidence:.4f}")

                            print(f"[{iv}m] Regime Selected: {regime_name} | ML Output: {ml_trend} (Bull: {prob_bullish*100:.1f}%, Bear: {prob_bearish*100:.1f}%, Neut: {prob_neutral*100:.1f}%) | Raw Conf: {ml_confidence*100:.2f}% | Calibrated Conf: {calibrated_confidence*100:.2f}% | Expected Change: {pred_change:+.3f}")

                            # Determine dynamic confidence threshold based on trade economics (p* break-even payoff) + bounded modifiers
                            atr_norm_val = latest_candle["ATR_norm"]
                            entry_close = float(latest_candle["close"])
                            atr_dollars = float(latest_candle.get("ATR", entry_close * atr_norm_val))
                            is_ranging_regime = "Ranging" in str(regime_name)
                            bot_state[f"garch_sigma_{symbol}"] = float(atr_norm_val) if atr_norm_val > 0 else 0.015

                            # 1. Resolve exact order execution geometry (TP and SL) before economic gate
                            from config import TIMEFRAME_CONFIG
                            from trade_calculators import transaction_cost_model, UnifiedTargetGenerator, REALIZED_RR_HAIRCUT, get_realized_rr_haircut
                            cfg = TIMEFRAME_CONFIG.get(str(iv), {})
                            sl_multiplier = float(cfg.get("sl_mult", 0.85))

                            vol_factor = 1.0
                            if atr_norm_val > 0:
                                vol_factor = max(0.75, min(1.5, 1.5 - ((atr_norm_val - 0.003) / 0.005) * 0.75))

                            min_target = max(getattr(config, "MIN_TARGET_ATR_MULT", {}).get(str(iv), 1.5), 1.20 * sl_multiplier)
                            base_tp_target = max(cfg.get("tp_mult_ranging", 1.40) if is_ranging_regime else cfg.get("tp_mult_trending", 1.85), min_target)
                            tp_multiplier_adjusted = round(base_tp_target * vol_factor, 3)

                            # Volatility (ATR Percentile) Adjustment (±5%)
                            atr_series = pd.to_numeric(df_completed["ATR"], errors="coerce").tail(100) if (df_completed is not None and "ATR" in df_completed.columns) else None
                            vol_adj = 1.00
                            if atr_series is not None and len(atr_series.dropna()) > 10:
                                curr_atr = float(latest_candle.get("ATR", atr_dollars))
                                clean_atr = atr_series.dropna()
                                atr_percentile = float((clean_atr < curr_atr).mean() * 100.0) if len(clean_atr) > 0 else 50.0
                                if atr_percentile > 90.0:
                                    vol_adj = 0.95
                                elif atr_percentile < 20.0:
                                    vol_adj = 1.05
                            tp_multiplier_adjusted *= vol_adj

                            # Session Liquidity Adjustment
                            curr_utc_hour = datetime.now(timezone.utc).hour
                            if 6 <= curr_utc_hour < 8:
                                session_factor = 0.95
                            elif 12 <= curr_utc_hour < 16:
                                session_factor = 1.00
                            else:
                                session_factor = 0.98
                            tp_multiplier_adjusted *= session_factor

                            # Finding #159 (Finding #91): Single shared canonical trade geometry resolver
                            geom = trade_calculators.resolve_trade_geometry(
                                entry_price=entry_close,
                                direction=ml_trend,
                                interval=str(iv),
                                atr_dollars=atr_dollars,
                                base_sl_multiplier=float(cfg.get("sl_mult", sl_multiplier)),
                                base_tp_multiplier=tp_multiplier_adjusted,
                                df=df_completed,
                                symbol=symbol,
                                regime=regime_name,
                                volatility=atr_norm_val,
                                database_module=database
                            )
                            stop_loss_price = geom["stop_loss_price"]
                            take_profit_price = geom["take_profit_price"]
                            resolved_sl_dist = geom["sl_dist"]
                            tp_change = geom["tp_dist"]
                            sl_multiplier_adjusted = geom["sl_multiplier_adjusted"]
                            resolved_sl_m = float(resolved_sl_dist / max(1e-6, atr_dollars))
                            resolved_tp_m = geom["tp_multiplier_adjusted"]
                            tp_multiplier_adjusted = resolved_tp_m
                            struct_meta = geom["struct_meta"]
                            struct_sl_dist_pct = geom["struct_sl_dist_pct"]
                            scaled_lev = None

                            # Economic Break-Even Threshold (p*) based on exact resolved order geometry
                            _bars_per_day = max(1, round(1440 / max(1, int(iv))))
                            _adv_usd = float(df_completed["volume"].tail(_bars_per_day).sum() * entry_close) if ("volume" in df_completed.columns and len(df_completed) >= _bars_per_day) else 50_000_000.0
                            _current_bal = float(bot_state.get("live_balance", bot_state.get("wallet_balance", bot_state.get("simulated_balance", 80.0))))
                            _max_risk_frac = float(getattr(config, "MAX_RISK_PER_TRADE", 0.02))
                            _est_stop_dist = max(0.002, (resolved_sl_dist / max(1e-6, entry_close)))
                            _est_notional = (_current_bal * _max_risk_frac) / _est_stop_dist
                            _min_order = float(getattr(config, "MIN_ORDER_VALUE_USDT", 5.1))
                            _order_usd = max(_min_order, _est_notional)

                            # Finding #79 & Overturned R47: Calculate live market microstructure spread in bps via real top-of-book orderbook
                            ob_data = get_orderbook_imbalance_and_spread(symbol)
                            ob_spread = ob_data.get("spread", 0.0) if isinstance(ob_data, dict) else 0.0
                            if ob_spread > 0.0:
                                current_spread_bps = round(float(ob_spread * 10000.0), 2)
                            else:
                                _bid_px = float(latest_candle.get("bid", 0.0))
                                _ask_px = float(latest_candle.get("ask", 0.0))
                                if _ask_px > _bid_px > 0:
                                    current_spread_bps = round(((_ask_px - _bid_px) / ((_ask_px + _bid_px) / 2.0)) * 10000.0, 2)
                                else:
                                    current_spread_bps = 1.5
                            bot_state[f"current_spread_bps_{symbol}"] = current_spread_bps
                            bot_state[f"spread_bp_{symbol}"] = current_spread_bps
                            bot_state["current_spread_bps"] = current_spread_bps
                            bot_state[f"garch_sigma_{symbol}"] = float(atr_norm_val) if 'atr_norm_val' in locals() and atr_norm_val > 0 else 0.015

                            _val_s = bot_state.get(f"spread_bp_{symbol}")
                            if _val_s is None:
                                _val_s = bot_state.get(f"current_spread_bps_{symbol}")
                            if _val_s is None:
                                _val_s = bot_state.get("current_spread_bps")
                            _spread_bp = float(_val_s) if _val_s is not None else 1.5
                            _garch_sigma = float(bot_state.get(f"garch_sigma_{symbol}") or (atr_norm_val if atr_norm_val > 0 else 0.015))

                            # Overturned #150: Model realistic round-trip costs: taker exit is guaranteed on SL, entry may be maker chase or taker IOC
                            _is_extreme_vol = bool(atr_norm_val > 0.02 or "vol" in str(regime_name).lower())
                            _is_maker_entry = not _is_extreme_vol
                            _tcm = transaction_cost_model.estimate_transaction_cost(
                                symbol=symbol,
                                order_size_usd=_order_usd,
                                volume_24h_usd=_adv_usd,
                                bid_ask_spread_bp=_spread_bp,
                                garch_sigma=_garch_sigma,
                                is_maker=_is_maker_entry,
                                round_trip=True
                            )
                            cost_bps = float(_tcm.get("total_cost_bps", _tcm.get("total_cost_bp", 12.0)))  # canonical round-trip with maker entry + taker exit
                            rec.round_trip_cost_bp = float(cost_bps)
                            nominal_rr = (resolved_tp_m * atr_dollars) / max(1e-6, resolved_sl_dist)
                            realized_haircut = get_realized_rr_haircut(interval=str(iv), regime=str(regime_name), nominal_rr=nominal_rr)
                            effective_tp_m = resolved_tp_m * realized_haircut
                            p_star = resolved_sl_m / max(1e-6, (effective_tp_m + resolved_sl_m))
                            cost_adj = (cost_bps / 1e4) / max(1e-6, (effective_tp_m + resolved_sl_m) * max(1e-4, atr_norm_val))
                            economic_base_threshold = float(round(p_star + cost_adj, 4))
                            base_cfg_thresh = float(cfg.get("base_confidence_threshold", 0.0))
                            dynamic_conf_threshold = max(economic_base_threshold, base_cfg_thresh)
                            adjustments_applied = [("economic_base", dynamic_conf_threshold)]
                            exp_edge_bps = abs(float(expected_pct_change)) * 100.0 - cost_bps
                            rec.expected_value = float(exp_edge_bps)
                            exp_r_val = float(effective_tp_m / max(1e-6, resolved_sl_m))
                            rec.expected_rr = exp_r_val
                            rec.gate("cost", value=float(cost_bps), passed=bool(cost_adj <= 0.05))

                            # Calibrator Economic Viability Guard
                            from tools.beta_calibrator import is_calibrator_viable
                            if active_calibrator is None or not is_calibrator_viable(active_calibrator, min_required_p_star=economic_base_threshold):
                                log_event("WARNING", f"[{symbol} {iv}m Calibrator Guard] Active calibrator missing, fallback, or achievable ceiling cannot reach fee-inclusive break-even p* ({economic_base_threshold:.4f}). Abstaining (Fail-Closed).")
                                continue

                            # ADX Regime Floor Filter (Regime-aware: Ranging models trade 10-24 ADX; Trending requires high momentum)
                            is_ranging_regime = "Ranging" in regime_name
                            if is_ranging_regime:
                                min_adx_thresh = float(cfg.get("min_adx_ranging", 10.0 if str(iv) in ["15", "30"] else 12.0))
                            else:
                                min_adx_thresh = float(cfg.get("min_adx", 16.0 if str(iv) in ["15", "30"] else 24.0))
                            if adx_regime < min_adx_thresh:
                                log_event("INFO", f"[{symbol} {iv}m ADX Filter] ADX {adx_regime:.1f} < min required {min_adx_thresh:.1f}. Skipping lower-conviction chop.")
                                continue

                            # 2. Bounded Regime & Volatility Adjustments
                            if "Ranging" in regime_name:
                                regime_delta = 0.02 if str(iv) in ["15", "30"] else 0.04
                                dynamic_conf_threshold += regime_delta
                                adjustments_applied.append(("regime_ranging", regime_delta))
                            
                            # High Volatility Adjustment (ATR > 0.015)
                            if atr_norm_val > 0.015:
                                vol_delta = 0.05
                                dynamic_conf_threshold += vol_delta
                                adjustments_applied.append(("high_volatility", vol_delta))
                                
                            htf_decay_threshold_penalty = 0.0
                            if str(iv) == "15" and ml_trend in ["Bullish", "Bearish"]:
                                pred_30m_dict = bot_state.get(f"latest_prediction_{symbol}_30m") or bot_state.get(f"latest_prediction_{symbol}_30") or {}
                                pred_60m_dict = bot_state.get(f"latest_prediction_{symbol}_1h") or bot_state.get(f"latest_prediction_{symbol}_60m") or bot_state.get(f"latest_prediction_{symbol}_60") or {}

                                now_time_sec = time.time()
                                for pred_dict, label in [(pred_30m_dict, "30m"), (pred_60m_dict, "1h")]:
                                    p_dir = pred_dict.get("direction")
                                    p_ts = pred_dict.get("timestamp", now_time_sec)
                                    if p_dir in ["Bullish", "Bearish"] and p_dir != ml_trend:
                                        age_mins = max(0.0, (now_time_sec - p_ts) / 60.0)
                                        decay = 0.5 ** (age_mins / 30.0)
                                        htf_decay_threshold_penalty += 0.03 * decay

                                if htf_decay_threshold_penalty > 0:
                                    htf_decay_threshold_penalty = min(0.04, htf_decay_threshold_penalty)
                                    dynamic_conf_threshold += htf_decay_threshold_penalty
                                    adjustments_applied.append(("htf_decay_penalty", htf_decay_threshold_penalty))
                                    print(f"[{symbol} 15m Time-Decayed Penalty] HTF contradiction gate penalty (+{htf_decay_threshold_penalty*100:.1f}% required threshold).")
                            
                            # Recent 50-Trade Performance Decay Filter (Timeframe-Isolated)
                            interval_trades = [t for t in bot_state.get("trade_history", []) if isinstance(t, dict) and str(t.get("interval")) == str(iv)][-50:]
                            if len(interval_trades) >= 10:
                                win_count = sum(1 for t in interval_trades if float(t.get("pnl_usd", 0.0)) > 0)
                                recent_win_rate = (win_count / len(interval_trades)) * 100.0
                                if recent_win_rate < 45.0:
                                    perf_delta = 0.04
                                    dynamic_conf_threshold += perf_delta
                                    adjustments_applied.append(("performance_decay", perf_delta))
                                    print(f"[{symbol} {iv}m Performance Decay Filter] Interval win rate {recent_win_rate:.1f}% < 45% ({len(interval_trades)} trades). Raised threshold by +0.04 to {dynamic_conf_threshold:.2f}")
                            


                            # Asian Market Session Awareness (00:00 - 08:00 UTC)
                            utc_hour_now = datetime.now(timezone.utc).hour
                            if 0 <= utc_hour_now < 8:
                                dynamic_conf_threshold += 0.02
                                adjustments_applied.append(("asian_session", 0.02))
                                print(f"[{symbol} {iv}m Asian Session] UTC hour {utc_hour_now:02d}:00 in low-volatility Asian window (+0.02 threshold -> {dynamic_conf_threshold:.2f})")
                                
                            # Cap additive penalties so 3-class models (random chance 33.3%) are not pushed to unreachable binary thresholds,
                            # but never cap below effective_base (which would loosen the model's baseline gating)
                            effective_base = max(float(economic_base_threshold), float(base_cfg_thresh))
                            max_conf_cap = max(effective_base, 0.50 if str(iv) in ["15", "30", "60"] else 0.55)
                            dynamic_conf_threshold = max(effective_base, min(max_conf_cap, dynamic_conf_threshold))

                            # Finding #129: Compute Composite Uncertainty (U_ensemble + U_market) with distinct predictions and matching weights
                            from statistical_validation import statistical_validation
                            _mean_atr = float(df_completed["ATR_norm"].mean()) if (df_completed is not None and "ATR_norm" in df_completed.columns and len(df_completed) >= 20) else atr_norm_val
                            target_class_idx = 2 if ml_trend == "Bullish" else (0 if ml_trend == "Bearish" else 1)
                            ind_preds = {}
                            if hasattr(active_model_trend, "predict_individual_proba"):
                                try:
                                    ind_p_map = active_model_trend.predict_individual_proba(X_live)
                                    for m_k, p_arr in ind_p_map.items():
                                        if isinstance(p_arr, (list, np.ndarray)) and len(p_arr) > target_class_idx:
                                            ind_preds[m_k] = float(p_arr[target_class_idx])
                                except Exception as ex_ind:
                                    log_event("WARNING", f"Individual prediction notice: {ex_ind}")
                            if not ind_preds:
                                ref_prob = prob_bullish if ml_trend == "Bullish" else prob_bearish
                                ind_preds = {"xgb": float(ref_prob)}
                                log_event("WARNING", f"[{symbol} {iv}m] No individual learner predictions available; falling back to single model uncertainty.")
                            elif len(ind_preds) < 2:
                                log_event("WARNING", f"[{symbol} {iv}m] Only {len(ind_preds)} model prediction available ({list(ind_preds.keys())}); cross-learner disagreement unavailable.")

                            ens_w = getattr(active_model_trend, "weights", None)
                            if ens_w is not None and isinstance(ens_w, (list, tuple, np.ndarray)) and len(ens_w) == 3:
                                w_dict = {"xgb": float(ens_w[0]), "lgb": float(ens_w[1]), "cat": float(ens_w[2])}
                            elif "ensemble_weights" in locals() and isinstance(ensemble_weights, (list, tuple)) and len(ensemble_weights) == 3:
                                w_dict = {"xgb": float(ensemble_weights[0]), "lgb": float(ensemble_weights[1]), "cat": float(ensemble_weights[2])}
                            else:
                                w_dict = {"xgb": 0.3333, "lgb": 0.3333, "cat": 0.3334}

                            unc_metrics = statistical_validation.calculate_composite_uncertainty(
                                individual_predictions=ind_preds,
                                model_weights=w_dict,
                                atr_expansion_ratio=float(atr_norm_val / max(1e-4, _mean_atr)),
                                spread_bp=float(current_spread_bps)
                            )
                            u_tot = float(unc_metrics.get("u_total", 0.04))
                            bot_state["u_total"] = u_tot
                            for k_suffix in [str(tf), str(iv)]:
                                bot_state[f"u_total_{symbol}_{k_suffix}"] = u_tot
                                bot_state[f"u_total_{k_suffix}"] = u_tot

                            # Compute drift p-value from live PSI
                            last_psi = float(bot_state.get("last_psi", 0.04))
                            drift_p = round(max(0.001, min(0.999, 1.0 - (last_psi / 0.25))), 4)
                            bot_state["drift_p_val"] = drift_p

                            # Compute rolling 30-trade symbol Sharpe
                            sym_trades = [t for t in bot_state.get("trade_history", []) if t.get("symbol") == symbol][-30:]
                            if len(sym_trades) >= 5:
                                sym_returns = [float(t.get("change_pct", (t.get("pnl_usd", 0.0) / max(1.0, float(t.get("original_size", t.get("position_size_usd", 1.0)))) * 100.0))) for t in sym_trades]
                                from trade_calculators import calculate_rolling_sharpe
                                sym_sharpe = round(float(calculate_rolling_sharpe(sym_returns)), 2)
                            else:
                                sym_sharpe = 1.2
                            bot_state["symbol_sharpe"] = sym_sharpe

                            # Compute MHI score
                            from strategy_health_engine import strategy_health_engine
                            mhi_res = strategy_health_engine.compute_model_health_index(
                                recent_pnls=[float(t.get("pnl_usd", 0.0)) for t in bot_state.get("trade_history", [])[-50:]],
                                ece_score=float(bot_state.get("last_ece", 0.04)),
                                brier_score=float(bot_state.get("last_brier_score", 0.15)),
                                psi_score=float(bot_state.get("last_psi", 0.04)),
                                execution_health_score=85.0
                            )
                            mhi_val = mhi_res.get("mhi_score", 90.0)
                            bot_state["mhi_score"] = mhi_val
                            rec.mhi_score = float(mhi_val)
                            for k_suffix in [str(tf), str(iv)]:
                                bot_state[f"mhi_{k_suffix}"] = mhi_val
                                bot_state[f"mhi_{symbol}_{k_suffix}"] = mhi_val

                            # Adaptive Confidence Threshold Matrix for 15m
                            if str(iv) == "15":
                                adaptive_val = trade_calculators.calculate_adaptive_15m_threshold(
                                    regime=regime_name,
                                    drift_p_val=drift_p,
                                    u_total=u_tot,
                                    symbol_sharpe=sym_sharpe,
                                    base_threshold=economic_base_threshold
                                )
                                if adaptive_val > dynamic_conf_threshold:
                                    adapt_delta = round(adaptive_val - dynamic_conf_threshold, 4)
                                    dynamic_conf_threshold = adaptive_val
                                    adjustments_applied.append(("adaptive_15m_matrix", adapt_delta))

                            # Bayesian Cold-Start Adjustment (Trades 3-9)
                            bayesian_res = mlops_engine.get_bayesian_adjusted_threshold(iv, bot_state.get("trade_history", []))
                            if bayesian_res.get("confidence_boost", 0) > 0:
                                b_boost = bayesian_res["confidence_boost"]
                                dynamic_conf_threshold += b_boost
                                adjustments_applied.append(("bayesian_cold_start", b_boost))
                                print(f"[{symbol} {iv}m] {bayesian_res['note']} -> Threshold: {dynamic_conf_threshold*100:.2f}%")

                            with news_sentiment_lock:
                                current_sentiment = cached_news_sentiment
                            print(f"[{iv}m] Dynamic Confidence Threshold: {dynamic_conf_threshold * 100:.2f}% (Economic Base: {economic_base_threshold*100:.2f}%, Regime: {regime_name}, Volatility: {atr_norm_val * 100:.3f}%, Sentiment: {current_sentiment})")

                            # Meta-Classifier: Use as confidence MODIFIER instead of hard gate
                            meta_adjustment = 0.0
                            if ml_trend in ["Bullish", "Bearish"] and active_meta_model is not None:
                                try:
                                    _exp_meta_names = get_model_feature_names(active_meta_model)
                                    if _exp_meta_names and not all(str(n).startswith("Column_") for n in _exp_meta_names):
                                        _meta_feats_to_use = [f for f in _exp_meta_names if f in latest_candle_weighted.index]
                                    else:
                                        _meta_feats_to_use = features
                                    X_meta_live = latest_candle_weighted[_meta_feats_to_use].to_frame().T if isinstance(latest_candle_weighted[_meta_feats_to_use], pd.Series) else latest_candle_weighted[_meta_feats_to_use]
                                    X_meta_input = _slice_model_input(active_meta_model, X_meta_live)
                                    if hasattr(X_meta_input, "apply"):
                                        X_meta_input = X_meta_input.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                                    else:
                                        X_meta_input = np.nan_to_num(np.array(X_meta_input, dtype=float), nan=0.0)
                                    meta_pred = int(active_meta_model.predict(X_meta_input)[0])
                                    if meta_pred == 1:
                                        meta_adjustment = -0.05  # Lowers required gate threshold by 5%
                                        print(f"[{iv}m] Meta-Classifier: PASS (required gate threshold lowered by -5%)")
                                    else:
                                        meta_adjustment = +0.07  # Raises required gate threshold by 7%
                                        print(f"[{iv}m] Meta-Classifier: FAIL (required gate threshold raised by +7%)")
                                    dynamic_conf_threshold += meta_adjustment
                                    adjustments_applied.append(("meta_classifier", meta_adjustment))
                                except Exception as meta_err:
                                    print(f"[{iv}m Warning] Meta-Classifier prediction skipped: {meta_err}")

                            # Candlestick Pattern Alignment Overlay Boost (-4% Threshold Gate Lowering)
                            bull_patterns = ["cdl_hammer", "cdl_bullish_engulfing", "cdl_morning_star", "cdl_three_white_soldiers", "cdl_three_inside_up", "cdl_abandoned_baby_bull", "cdl_piercing_line", "cdl_tweezer_bottom", "cdl_marubozu_bull"]
                            bear_patterns = ["cdl_shooting_star", "cdl_bearish_engulfing", "cdl_evening_star", "cdl_three_black_crows", "cdl_three_inside_down", "cdl_dark_cloud_cover", "cdl_tweezer_top", "cdl_marubozu_bear"]
                            pattern_boost = False
                            if ml_trend == "Bullish" and any(latest_candle.get(p, 0) == 1 for p in bull_patterns):
                                pattern_boost = True
                            elif ml_trend == "Bearish" and any(latest_candle.get(p, 0) in [1, -1] for p in bear_patterns):
                                pattern_boost = True

                            if pattern_boost:
                                candlestick_delta = -0.04
                                dynamic_conf_threshold += candlestick_delta
                                adjustments_applied.append(("candlestick_pattern", candlestick_delta))
                                print(f"[{symbol} {iv}m Candlestick Overlay] Pattern Alignment Boost (required threshold lowered -4.0% to {dynamic_conf_threshold:.2f}) | Pure Calibrated Conf: {calibrated_confidence*100:.2f}%")

                            # Determine tracking status
                            # Softened contradiction: only block if regressor predicts > 0.05% in OPPOSITE direction
                            pred_pct = (abs(pred_change) / latest_candle["close"]) * 100
                            strong_conflict = (ml_trend == "Bullish" and pred_change < 0 and pred_pct > 0.05) or \
                                              (ml_trend == "Bearish" and pred_change > 0 and pred_pct > 0.05)
                        
                            is_cooling, remaining_mins = is_symbol_interval_cooling_off(symbol, iv)
                            news_event = ""
                        
                            # Hierarchical Confluence Check, Institutional HTF Waterfall & Decision Lineage
                            confluence_blocked = False
                            htf_trend = "Neutral"
                            macro_tf = ""
                            htf_mapping = {"15": "60", "30": "120", "60": "240", "120": "360"}
                            macro_iv = htf_mapping.get(str(iv), "")
                            htf_meta = {
                                "trend": "Neutral",
                                "trend_source": "NONE",
                                "ml_probability": 0.0,
                                "ml_prediction": "Neutral",
                                "model_age_days": 0,
                                "fallback_reason": "UNINITIALIZED",
                                "ema_fast": 0.0,
                                "ema_slow": 0.0,
                                "ema_slow_slope": 0.0,
                                "adx": 0.0,
                                "sma50": 0.0,
                                "consensus_score": "LOW",
                                "decision_timestamp": datetime.now(timezone.utc).isoformat(),
                                "model_version": f"{macro_iv}_v1" if macro_iv else "default"
                            }

                            if str(iv) in htf_mapping:
                                macro_tf = tf_map.get(str(macro_iv))
                                learned_threshold = get_learned_confidence_threshold(symbol, macro_iv, regime)

                                if macro_tf:
                                    macro_pred = bot_state.get(f"latest_prediction_{symbol}_{macro_tf}")
                                    ml_trend_dir = "Neutral"
                                    ml_prob = 0.0
                                    model_age_days = 0
                                    is_htf_stale = False

                                    if macro_pred and isinstance(macro_pred, dict):
                                        ml_trend_dir = macro_pred.get("direction", "Neutral")
                                        ml_prob = float(macro_pred.get("calibrated_confidence") or macro_pred.get("confidence") or 0.0)
                                        pred_dyn_thresh = macro_pred.get("dynamic_threshold")
                                        if pred_dyn_thresh is not None:
                                            learned_threshold = float(pred_dyn_thresh)
                                        
                                        # Check timestamp age of HTF prediction (> 2x HTF candle interval)
                                        pred_ts = macro_pred.get("timestamp") or macro_pred.get("eval_time") or 0.0
                                        macro_iv_sec = int(macro_iv) * 60 if str(macro_iv).isdigit() else 3600
                                        if pred_ts > 0 and (time.time() - pred_ts) > (2.0 * macro_iv_sec):
                                            is_htf_stale = True

                                    # Compute model age dynamically from manifest metadata timestamp if available
                                    manifest_paths = [
                                        f"ensemble_{regime.lower()}_trend_{macro_iv}_manifest.json",
                                        f"ensemble_trending_trend_{macro_iv}_manifest.json"
                                    ]
                                    for m_path in manifest_paths:
                                        if os.path.exists(m_path):
                                            try:
                                                with open(m_path, "r") as mf:
                                                    m_meta = json.load(mf)
                                                    m_created = m_meta.get("created_at") or m_meta.get("timestamp")
                                                    if m_created:
                                                        if isinstance(m_created, (int, float)):
                                                            model_age_days = max(0, int((time.time() - m_created) / 86400))
                                                        else:
                                                            dt = datetime.fromisoformat(str(m_created).replace("Z", "+00:00"))
                                                            model_age_days = max(0, (datetime.now(timezone.utc) - dt).days)
                                                    else:
                                                        mtime = os.path.getmtime(m_path)
                                                        model_age_days = max(0, int((time.time() - mtime) / 86400))
                                                    break
                                            except Exception:
                                                pass

                                    htf_meta["ml_prediction"] = ml_trend_dir
                                    htf_meta["ml_probability"] = ml_prob
                                    htf_meta["model_age_days"] = model_age_days

                                    # STEP 1: Check ML Model Freshness & Learned Confidence
                                    if ml_trend_dir in ["Bullish", "Bearish"] and ml_prob >= learned_threshold and model_age_days < 45 and not is_htf_stale:
                                        htf_trend = ml_trend_dir
                                        htf_meta["trend_source"] = "ML_MODEL"
                                        htf_meta["fallback_reason"] = "NONE"
                                    else:
                                        if is_htf_stale:
                                            fallback_reason = "SIGNAL_STALE"
                                        elif ml_trend_dir == "Neutral":
                                            fallback_reason = "MODEL_NEUTRAL"
                                        elif ml_prob < learned_threshold:
                                            fallback_reason = "LOW_CONFIDENCE"
                                        else:
                                            fallback_reason = "MODEL_STALE"
                                        htf_meta["fallback_reason"] = fallback_reason

                                        # STEP 2: EMA9 vs EMA21 + EMA21 Slope > 0 Technical Fallback
                                        try:
                                            from ta.trend import EMAIndicator, ADXIndicator, SMAIndicator
                                            htf_df = get_history(symbol=symbol, interval=str(macro_iv), limit=60, fail_if_stale=True)
                                            if htf_df is not None and not htf_df.empty and htf_df.attrs.get("fetch_ok", True) and len(htf_df) >= 50:
                                                s_e9 = EMAIndicator(htf_df["close"], window=9).ema_indicator()
                                                s_e21 = EMAIndicator(htf_df["close"], window=21).ema_indicator()
                                                s_adx = ADXIndicator(htf_df["high"], htf_df["low"], htf_df["close"], window=14).adx()
                                                s_sma50 = SMAIndicator(htf_df["close"], window=50).sma_indicator()

                                                e9_val = float(s_e9.iloc[-1]) if pd.notna(s_e9.iloc[-1]) else 0.0
                                                e21_val = float(s_e21.iloc[-1]) if pd.notna(s_e21.iloc[-1]) else 0.0
                                                e21_prev = float(s_e21.iloc[-3]) if len(s_e21) >= 3 and pd.notna(s_e21.iloc[-3]) else e21_val
                                                e21_slope = (e21_val - e21_prev) / (e21_prev or 1.0) * 100.0

                                                adx_val = float(s_adx.iloc[-1]) if pd.notna(s_adx.iloc[-1]) else 0.0
                                                sma50_val = float(s_sma50.iloc[-1]) if pd.notna(s_sma50.iloc[-1]) else 0.0
                                                latest_close = float(htf_df["close"].iloc[-1])

                                                htf_meta["ema_fast"] = e9_val
                                                htf_meta["ema_slow"] = e21_val
                                                htf_meta["ema_slow_slope"] = e21_slope
                                                htf_meta["adx"] = adx_val
                                                htf_meta["sma50"] = sma50_val

                                                ema_bullish = (e9_val > e21_val) and (e21_slope > 0.0)
                                                ema_bearish = (e9_val < e21_val) and (e21_slope < 0.0)

                                                if ema_bullish:
                                                    htf_trend = "Bullish"
                                                    htf_meta["trend_source"] = "EMA_FALLBACK"
                                                elif ema_bearish:
                                                    htf_trend = "Bearish"
                                                    htf_meta["trend_source"] = "EMA_FALLBACK"
                                                else:
                                                    # STEP 3: ADX + Close vs SMA50 Secondary Fallback
                                                    if adx_val >= 20.0:
                                                        if latest_close > sma50_val:
                                                            htf_trend = "Bullish"
                                                            htf_meta["trend_source"] = "ADX_FALLBACK"
                                                        elif latest_close < sma50_val:
                                                            htf_trend = "Bearish"
                                                            htf_meta["trend_source"] = "ADX_FALLBACK"
                                        except Exception as err:
                                            print(f"[{symbol} HTF Fallback Warning] Error computing technical waterfall: {err}")

                                    # STEP 4: HTF Consensus Scoring (HIGH, MEDIUM, LOW)
                                    htf_meta["trend"] = htf_trend
                                    ml_matches = (ml_trend_dir == htf_trend) and (htf_trend in ["Bullish", "Bearish"])
                                    if ml_matches and htf_meta["trend_source"] == "ML_MODEL":
                                        consensus = "HIGH"
                                    elif htf_trend in ["Bullish", "Bearish"] and htf_meta["trend_source"] in ["EMA_FALLBACK", "ADX_FALLBACK"]:
                                        consensus = "MEDIUM" if ml_trend_dir == "Neutral" else "LOW"
                                    else:
                                        consensus = "LOW"
                                    htf_meta["consensus_score"] = consensus

                                    # Save lineage metadata in bot_state
                                    bot_state[f"htf_trend_metadata_{symbol}_{iv}"] = htf_meta
                                    if str(macro_iv) == "240" or str(macro_tf) == "4h":
                                        bot_state[f"macro_trend_{symbol}_4h"] = htf_trend
                                        bot_state[f"htf_trend_{symbol}_4h"] = htf_trend
                                        if symbol == "BTCUSDT" or symbol == bot_state.get("active_symbol", "BTCUSDT"):
                                            bot_state["macro_trend_4h"] = htf_trend
                                            bot_state["htf_trend_4h"] = htf_trend

                                    if htf_trend in ["Bullish", "Bearish"] and ml_trend in ["Bullish", "Bearish"]:
                                        if ml_trend == htf_trend:
                                            dynamic_conf_threshold -= 0.08
                                            adjustments_applied.append(("macro_alignment", -0.08))
                                            print(f"[{symbol} {iv}m Macro Alignment Boost] Aligned with {macro_tf} ({htf_trend}, Source: {htf_meta['trend_source']}, Consensus: {consensus}). Threshold lowered (-8.0% to {dynamic_conf_threshold:.2f}) | Pure Calibrated Conf: {calibrated_confidence*100:.2f}%")
                                        else:
                                            dynamic_conf_threshold += 0.10
                                            confluence_blocked = True
                                            adjustments_applied.append(("macro_opposition", 0.10))
                                            print(f"[{symbol} {iv}m Macro Opposition Penalty] Signal opposes {macro_tf} ({htf_trend}, Source: {htf_meta['trend_source']}). Threshold raised (+10.0% to {dynamic_conf_threshold:.2f}) | Pure Calibrated Conf: {calibrated_confidence*100:.2f}%")

                            # Funding Rate Carry Overlay & Crowdedness Friction Guard
                            funding_rate = get_funding_rate(symbol)
                            funding_blocked = False
                            if funding_rate > 0.0003 and ml_trend == "Bullish":
                                penalty = min(0.06, (funding_rate - 0.0003) * 100.0 + 0.02)
                                dynamic_conf_threshold += penalty
                                adjustments_applied.append(("funding_carry_long", round(penalty, 3)))
                                print(f"[{symbol} {iv}m] Funding Carry Friction: High Long funding ({funding_rate*100:.3f}%) raised threshold (+{penalty*100:.1f}% to {dynamic_conf_threshold*100:.1f}%)")
                            elif funding_rate < -0.0003 and ml_trend == "Bearish":
                                penalty = min(0.06, (abs(funding_rate) - 0.0003) * 100.0 + 0.02)
                                dynamic_conf_threshold += penalty
                                adjustments_applied.append(("funding_carry_short", round(penalty, 3)))
                                print(f"[{symbol} {iv}m] Funding Carry Friction: High Short funding ({funding_rate*100:.3f}%) raised threshold (+{penalty*100:.1f}% to {dynamic_conf_threshold*100:.1f}%)")

                            if (ml_trend == "Bullish" and funding_rate > 0.0008) or (ml_trend == "Bearish" and funding_rate < -0.0008):
                                funding_blocked = True
                            
                            # Open Interest Momentum Guard
                            try:
                                oi_delta = df.iloc[-1].get("open_interest_pct_change", 0.0) * 100.0
                                if oi_delta < 0.5:
                                    dynamic_conf_threshold += 0.02
                                    adjustments_applied.append(("oi_momentum_guard", 0.02))
                                    print(f"[{symbol} {iv}m] OI Momentum Guard: Low Open Interest Delta ({oi_delta:+.2f}%) raised threshold to {dynamic_conf_threshold*100:.1f}%")
                            except Exception as e:
                                log_event("WARNING", f"[{symbol} {iv}m] Exception in OI Momentum Guard: {e}")

                            # Bound final threshold relative to economic base
                            from config import MAX_THRESHOLD_UPLIFT
                            effective_base = max(float(economic_base_threshold), float(base_cfg_thresh))
                            max_allowed_threshold = max(effective_base, min(0.65, effective_base + MAX_THRESHOLD_UPLIFT))
                            dynamic_conf_threshold = float(round(max(effective_base, min(max_allowed_threshold, dynamic_conf_threshold)), 4))
                        
                            # Log threshold lineage to prediction state
                            for k_suffix in [str(tf), str(iv)]:
                                if f"latest_prediction_{symbol}_{k_suffix}" in bot_state and isinstance(bot_state[f"latest_prediction_{symbol}_{k_suffix}"], dict):
                                    bot_state[f"latest_prediction_{symbol}_{k_suffix}"]["threshold_base"] = economic_base_threshold
                                    bot_state[f"latest_prediction_{symbol}_{k_suffix}"]["threshold_adjustments"] = adjustments_applied
                                    bot_state[f"latest_prediction_{symbol}_{k_suffix}"]["dynamic_threshold"] = dynamic_conf_threshold
                                if (symbol == "BTCUSDT" or symbol == bot_state.get("active_symbol", "BTCUSDT")) and f"latest_prediction_{k_suffix}" in bot_state and isinstance(bot_state[f"latest_prediction_{k_suffix}"], dict):
                                    bot_state[f"latest_prediction_{k_suffix}"]["threshold_base"] = economic_base_threshold
                                    bot_state[f"latest_prediction_{k_suffix}"]["threshold_adjustments"] = adjustments_applied
                                    bot_state[f"latest_prediction_{k_suffix}"]["dynamic_threshold"] = dynamic_conf_threshold
                            rec._inputs["dynamic_threshold"] = float(dynamic_conf_threshold)
                        
                            status_msg = "Pending"
                            active_trade_key = f"active_trade_{tf}"
                            active_trades_list = bot_state.get(active_trade_key, [])
                        
                            # Prevent duplicate parallel trades of the same symbol on ANY interval/timeframe
                            already_active = False
                            active_on_tf = None
                            if symbol in just_opened_symbols:
                                already_active = True
                                active_on_tf = "current_cycle"
                            else:
                                for check_tf in ["15m", "30m", "1h", "2h", "4h", "6h"]:
                                    # Normalize timeframe keys to check both active_trade_15m and active_trade_15
                                    keys_to_check = [f"active_trade_{check_tf}", f"active_trade_{check_tf.replace('m','').replace('h','0')}"]
                                    if check_tf == "1h": keys_to_check.append("active_trade_60")
                                    elif check_tf == "2h": keys_to_check.append("active_trade_120")
                                    elif check_tf == "4h": keys_to_check.append("active_trade_240")
                                    elif check_tf == "6h": keys_to_check.append("active_trade_360")
                                    
                                    for k in keys_to_check:
                                        if any(t.get("symbol") == symbol for t in bot_state.get(k, [])):
                                            already_active = True
                                            active_on_tf = check_tf
                                            break
                                    if already_active:
                                        break
                        
                            # Session Filter: temporarily allow 24-hour trading
                            utc_hour = datetime.now(timezone.utc).hour
                            in_session = True

                            flash_crash_active = check_flash_crash(symbol, max_drop_pct=3.0, window_minutes=5)
                            liq_score = get_liquidity_score(symbol)
                            low_liquidity = (liq_score < 0.3)
                            rec.liquidity_score = float(liq_score)

                            # Component 9: Realized Closed-Trade Expectancy Gate (Finding #105, #17)
                            exp_gate_blocked = False
                            exp_gate_msg = ""
                            exp_mode = getattr(config, "EXPECTANCY_GATE_MODE", "shadow")
                            if exp_mode != "disabled":
                                try:
                                    recent_closed = database.get_completed_trades(limit=50, symbol=symbol)
                                    interval_closed = [t for t in recent_closed if str(t.get("interval", "")).replace("m", "") == str(iv).replace("m", "")]
                                    if len(interval_closed) >= 15:
                                        wins = [float(t.get("change_pct", 0.0)) for t in interval_closed if float(t.get("pnl_usd", 0.0)) > 0]
                                        losses = [abs(float(t.get("change_pct", 0.0))) for t in interval_closed if float(t.get("pnl_usd", 0.0)) < 0]
                                        if len(wins) > 0 and len(losses) > 0:
                                            hist_wr = len(wins) / len(interval_closed)
                                            avg_win_p = sum(wins) / len(wins)
                                            avg_loss_p = sum(losses) / len(losses)
                                            from confluence_engine import evaluate_expectancy_gate
                                            exp_pass, hist_ev = evaluate_expectancy_gate(hist_wr, avg_win_p, avg_loss_p)
                                            if not exp_pass:
                                                if exp_mode == "active":
                                                    exp_gate_blocked = True
                                                    exp_gate_msg = f"Negative Historical EV ({hist_ev*100:+.2f}%) over last {len(interval_closed)} trades"
                                                else:
                                                    log_event("INFO", f"[{symbol} {iv}m] [Shadow Expectancy Gate] Negative Historical EV ({hist_ev*100:+.2f}%) over {len(interval_closed)} trades — would block in active mode.")
                                                    bot_state[f"shadow_expectancy_block_{symbol}_{iv}"] = True
                                except Exception as ex_exp:
                                    log_event("WARNING", f"Expectancy gate check error for {symbol} {iv}m: {ex_exp}")
                                    if exp_mode == "active":
                                        exp_gate_blocked = True
                                        exp_gate_msg = f"Expectancy check DB error (Fail-Closed): {ex_exp}"

                            # P3: Correlated Portfolio Cluster Exposure Guard
                            cluster_blocked = False
                            cluster_block_reason = ""
                            if ml_trend in ["Bullish", "Bearish"]:
                                all_open_positions = []
                                for _k_tf in ["15m", "30m", "1h", "2h", "4h"]:
                                    all_open_positions.extend(bot_state.get(f"active_trade_{_k_tf}", []))
                                from portfolio_risk import portfolio_risk_engine
                                cluster_approved, cluster_block_reason = portfolio_risk_engine.check_correlated_cluster_exposure(symbol, ml_trend, all_open_positions, max_same_cluster_count=2)
                                if not cluster_approved:
                                    cluster_blocked = True

                            if not bot_state.get("bot_running", True):
                                status_msg = "Skipped (Bot Stopped)"
                                rec.reason_code = ReasonCode.BOT_STOPPED
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Bot is currently stopped by the user.")
                            elif bot_state.get("circuit_breaker_active", False):
                                status_msg = "Skipped (Circuit Breaker)"
                                rec.reason_code = ReasonCode.CIRCUIT_BREAKER_ACTIVE
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Daily Drawdown Circuit Breaker is active.")
                            elif flash_crash_active:
                                status_msg = "Skipped (Flash Crash Block)"
                                rec.reason_code = ReasonCode.FLASH_CRASH_ACTIVE
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Flash crash detected (>3.0% drop in last 5 minutes).")
                            elif low_liquidity:
                                status_msg = "Skipped (Low Liquidity)"
                                rec.reason_code = ReasonCode.LOW_LIQUIDITY
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Insufficient L2 orderbook liquidity (Score: {liq_score:.2f} < 0.30).")
                            elif already_active:
                                status_msg = "Skipped (Already Active)"
                                rec.reason_code = ReasonCode.ALREADY_ACTIVE
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: A trade is already active for this symbol on the {active_on_tf} timeframe.")
                            elif cluster_blocked:
                                status_msg = "Skipped (Cluster Limit)"
                                rec.reason_code = ReasonCode.CLUSTER_LIMIT
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: {cluster_block_reason}")
                            elif exp_gate_blocked:
                                status_msg = f"Skipped ({exp_gate_msg})"
                                rec.reason_code = ReasonCode.EXPECTANCY_NEGATIVE
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: {exp_gate_msg}")
                            elif hasattr(bot_state, "get") and bot_state.get(f"kill_switch_halt_{iv}"):
                                status_msg = "Skipped (Kill Switch Halt)"
                                rec.reason_code = ReasonCode.DRAWDOWN_HALT
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Interval {iv}m is halted by statistical kill criteria.")
                            elif not in_session:
                                status_msg = "Skipped (Off-Session)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Outside London/NY session (UTC hour: {utc_hour}).")
                            elif is_cooling:
                                status_msg = "Skipped (Cool-Off)"
                                rec.reason_code = ReasonCode.COOL_OFF_ACTIVE
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Interval is in a 6-hour cool-off period after consecutive losses ({remaining_mins} mins remaining).")
                            elif funding_blocked:
                                status_msg = "Skipped (Funding Block)"
                                rec.reason_code = ReasonCode.FUNDING_BLOCK
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: High funding fee payment risk (Funding: {funding_rate*100:.3f}%).")
                            elif confluence_blocked:
                                status_msg = "Skipped (Macro Opposition)"
                                rec.reason_code = ReasonCode.MACRO_OPPOSITION
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: HTF macro trend opposes trade direction ({htf_trend}).")
                            elif ml_trend == "Neutral":
                                status_msg = "Skipped (Neutral)"
                                rec.reason_code = ReasonCode.NEUTRAL
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Model output is Neutral/Hold.")
                            elif strong_conflict:
                                status_msg = "Skipped (Contradiction)"
                                rec.reason_code = ReasonCode.CONTRADICTION
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Strong directional contradiction (Trend: {ml_trend}, Regressor: {pred_change:+.3f} [{pred_pct:.3f}%]).")
                            elif calibrated_confidence < dynamic_conf_threshold:
                                status_msg = "Skipped (Low Confidence)"
                                rec.reason_code = ReasonCode.CONFIDENCE_BELOW_DYNAMIC_THRESHOLD
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped (calibrated confidence {calibrated_confidence*100:.2f}% < {dynamic_conf_threshold*100:.2f}%).")
                            elif conformal_is_uncertain and ml_trend in ["Bullish", "Bearish"]:
                                status_msg = "Skipped (High Conformal Uncertainty)"
                                rec.reason_code = ReasonCode.HIGH_CONFORMAL_UNCERTAINTY
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: High ensemble disagreement / conformal uncertainty score ({conformal_unc_score:.3f}).")

                            if status_msg == "Pending":
                                current_spread_bps = float(bot_state.get(f"current_spread_bps_{symbol}", bot_state.get("current_spread_bps", 3.5))) if "bot_state" in globals() and hasattr(bot_state, "get") else 3.5
                                u_tot_live = float(bot_state.get("u_total", 0.04)) if "bot_state" in globals() and hasattr(bot_state, "get") else 0.04
                                rec.spread_bp = current_spread_bps
                                
                                # Expected R:R of the target setup relative to minimum floor across all intervals
                                _iv_cfg = bot_state.get("optimized_timeframe_config", {}).get(str(iv), {}) if "bot_state" in globals() and hasattr(bot_state, "get") else {}
                                _iv_base = TIMEFRAME_CONFIG.get(str(iv), {"sl_mult": 0.9, "tp_mult_ranging": 1.4, "tp_mult_trending": 1.8})
                                _iv_sl = float(_iv_cfg.get("sl_mult", _iv_base.get("sl_mult", 0.9)))
                                adx_regime_threshold = float(getattr(config, "REGIME_ADX_ENTER_BY_INTERVAL", {}).get(str(iv), 28.0))
                                _iv_tp = float(_iv_cfg.get("tp_mult_trending" if latest_candle.get("ADX", 0.0) >= adx_regime_threshold else "tp_mult_ranging", 1.4))
                                exp_r_val = float(_iv_tp / max(1e-6, _iv_sl))
                                rec.expected_rr = exp_r_val

                                # Estimate per-symbol 24h ADV from candle volume * price
                                _bars_24h = max(10, round(1440 / max(1, int(iv))))
                                _adv_usd = float(df["volume"].tail(_bars_24h).sum() * df["close"].iloc[-1]) if (df is not None and "volume" in df.columns and "close" in df.columns and len(df) >= 10) else 50_000_000.0
                                _current_bal_eval = float(bot_state.get("live_balance", bot_state.get("wallet_balance", bot_state.get("simulated_balance", 80.0)))) if "bot_state" in globals() and hasattr(bot_state, "get") else 80.0
                                _min_order_eval = float(getattr(config, "MIN_ORDER_VALUE_USDT", 5.1))
                                _est_stop_dist_eval = max(0.002, (_iv_sl * atr_norm_val))
                                _est_notional_eval = (_current_bal_eval * float(getattr(config, "MAX_RISK_PER_TRADE", 0.02))) / max(1e-4, _est_stop_dist_eval)
                                _order_usd = max(_min_order_eval, _est_notional_eval)
                                _spread_val_eval = bot_state.get(f"spread_bp_{symbol}")
                                if _spread_val_eval is None:
                                    _spread_val_eval = bot_state.get(f"current_spread_bps_{symbol}")
                                if _spread_val_eval is None:
                                    _spread_val_eval = current_spread_bps
                                _spread_bp_eval = float(_spread_val_eval)
                                _garch_sigma_eval = float(bot_state.get(f"garch_sigma_{symbol}") or (atr_norm_val if 'atr_norm_val' in locals() and atr_norm_val > 0 else 0.015))
                                _is_maker_eval = not bool(atr_norm_val > 0.02)
                                _tcm_res = transaction_cost_model.estimate_transaction_cost(
                                    symbol=symbol,
                                    order_size_usd=_order_usd,
                                    volume_24h_usd=_adv_usd,
                                    bid_ask_spread_bp=_spread_bp_eval,
                                    garch_sigma=_garch_sigma_eval,
                                    is_maker=_is_maker_eval,
                                    round_trip=True
                                )
                                tcm_cost_bps = float(_tcm_res.get("total_cost_bps", _tcm_res.get("total_cost_bp", 12.0)))
                                rec.round_trip_cost_bp = float(tcm_cost_bps)
                                exp_edge_bps = abs(float(expected_pct_change)) * 100.0 - tcm_cost_bps
                                rec.expected_value = float(exp_edge_bps)
                                rec.gate("cost", value=float(tcm_cost_bps), passed=bool(exp_edge_bps > 0))

                                if str(iv) == "15":
                                    vol_series = df["volume"] if (df is not None and "volume" in df.columns) else None
                                    vol_20th = float(vol_series.quantile(0.20)) if (vol_series is not None and len(vol_series.dropna()) >= 20) else 0.0
                                    curr_vol = float(latest_candle.get("volume", 0.0))
                                    mean_atr_24h = float(df["ATR_norm"].mean()) if (df is not None and "ATR_norm" in df.columns and len(df) >= 20) else atr_norm_val

                                    # Adaptive spread limit: 5.0 bps for BTC/ETH, 8.0 for major alts, 15.0 for others
                                    if symbol in ["BTCUSDT", "ETHUSDT"]:
                                        max_spread_bps = 5.0
                                    elif symbol in ["SOLUSDT", "BNBUSDT", "XRPUSDT"]:
                                        max_spread_bps = 8.0
                                    else:
                                        max_spread_bps = 15.0

                                    if curr_vol < vol_20th and vol_20th > 0:
                                        status_msg = "Skipped (Volume Compression <20th Pct)"
                                        rec.reason_code = ReasonCode.VOLUME_COMPRESSION
                                        print(f"[{symbol} 15m Filter] Trade skipped: Volume ({curr_vol:.1f}) < 20th percentile ({vol_20th:.1f}).")
                                    elif atr_norm_val > (1.5 * mean_atr_24h):
                                        status_msg = "Skipped (ATR Spike >1.5x Mean)"
                                        rec.reason_code = ReasonCode.ATR_SPIKE
                                        print(f"[{symbol} 15m Filter] Trade skipped: ATR spike ({atr_norm_val*100:.2f}%) > 1.5x 24h mean ({mean_atr_24h*100:.2f}%).")
                                    elif current_spread_bps > max_spread_bps:
                                        status_msg = f"Skipped (Spread Widening >{max_spread_bps:.1f} bps)"
                                        rec.reason_code = ReasonCode.SPREAD_WIDENING
                                        print(f"[{symbol} 15m Filter] Trade skipped: Spread ({current_spread_bps:.1f} bps) exceeds {max_spread_bps:.1f} bps limit.")
                                    elif exp_r_val < 1.0:
                                        status_msg = "Skipped (Expected R < 1.0R)"
                                        rec.reason_code = ReasonCode.RR_BELOW_FLOOR
                                        print(f"[{symbol} 15m Filter] Trade skipped: Expected R ({exp_r_val:.2f}R) < 1.00R floor.")
                                    elif exp_edge_bps <= 0:
                                        status_msg = "Skipped (TCM Net Edge <= 0)"
                                        rec.reason_code = ReasonCode.TCM_NET_EDGE_NEGATIVE
                                        print(f"[{symbol} 15m Filter] Trade skipped: TCM Expected Net Edge ({exp_edge_bps:.1f} bps) is non-positive.")
                                    elif u_tot_live >= 0.20:
                                        status_msg = "Skipped (Uncertainty U >= 0.20)"
                                        print(f"[{symbol} 15m Filter] Trade skipped: Total Ensemble Uncertainty ({u_tot_live:.3f}) >= 0.20 threshold.")

                            if status_msg == "Pending":
                                # Check news window proximity status for logging/blocking purposes
                                in_news_window, news_event = is_high_impact_news_window()
                                if in_news_window:
                                    print(f"[{iv}m News Block] Trade skipped: high-impact news event window active ({news_event}).")
                                    status_msg = "Skipped (News Block)"
                                    continue
                                
                                with news_sentiment_lock:
                                    news_sentiment = cached_news_sentiment
                                    latest_titles = cached_news_titles
                                    df_1h_confluence = fetched_data.get((symbol, "60"), (None, None))[1]
                                    if df_1h_confluence is None:
                                        df_1h_confluence = df
                                    all_pass, confluence_results, confluence_score_pct = check_pre_trade_confluence(
                                        latest_candle["close"], df_1h_confluence, ml_trend, news_sentiment, expected_pct_change, iv, symbol=symbol, htf_cache=htf_cache,
                                        calibrated_confidence=calibrated_confidence, dynamic_conf_threshold=dynamic_conf_threshold, get_history_fn=get_history,
                                        get_orderbook_fn=get_orderbook_imbalance, choppiness_fn=choppiness_index, bot_state_dict=bot_state, current_regime=regime
                                    )

                                    # Update global confluence status
                                    bot_state[f"confluence_results_{tf}"] = {
                                        "approved": all_pass,
                                        "checks": confluence_results
                                    }

                                    print(f"\n==================================================")
                                    print(f"[{iv}m] PRE-TRADE CONFLUENCE ANALYSIS REPORT")
                                    print("--------------------------------------------------")
                                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbol: {symbol}")
                                    print(f"Signal: {ml_trend} | Calibrated Confidence: {calibrated_confidence * 100:.2f}%")
                                    print(f"Current Price: {latest_candle['close']:.2f} | Predicted Price: {predicted_price:.2f} (Expected: {pred_change:+.3f} [{expected_pct_change:.3f}%])")
                                    print("--------------------------------------------------")
                                    print("Checks Status:")
                                    for idx, (check_name, res_val) in enumerate(confluence_results.items(), 1):
                                        status_str = "[PASS]" if res_val["pass"] else "[FAIL]"
                                        print(f"  {status_str} {idx}. {check_name.replace('_', ' '):<22}: {res_val['detail']}")
                                
                                    if all_pass:
                                        status_msg = "Traded"
                                        print("--------------------------------------------------")
                                        print(f"CONFLUENCE RESULT: APPROVED ({confluence_results.get('_Score_Summary', {}).get('detail', 'Score check passed')})")
                                        print("==================================================\n")

                                        # Use exact resolved order execution geometry (aligned with economic gate p*)
                                        raw_entry_price = entry_close
                                        entry_price = raw_entry_price
                                        raw_sl_dist = resolved_sl_dist

                                        # Adaptive Structural Swing Stop & Recency Guard for intraday timeframes (15m, 30m, 60m)
                                        if str(iv) in ["15", "30", "60"]:
                                            base_sl_pct = max(0.6, (atr_dollars * 1.0 / entry_price) * 100.0)
                                            scaled_lev, is_valid_lev = trade_calculators.scale_leverage_for_fixed_risk(
                                                base_leverage=5.0,
                                                base_sl_pct=base_sl_pct,
                                                structural_sl_pct=struct_sl_dist_pct if struct_sl_dist_pct is not None else ((raw_sl_dist / entry_price) * 100.0)
                                            )
                                            if not is_valid_lev:
                                                print(f"[{symbol} {iv}m Filter] Trade skipped: Scaled leverage ({scaled_lev}x) below 1.5x floor limit.")
                                                status_msg = "Skipped (Leverage Floor < 1.5x)"
                                                all_pass = False
                                            else:
                                                print(f"[{iv}m Structural Stop] Entry: {entry_price:.4f} | Structural SL: {stop_loss_price:.4f} (Dist: {struct_sl_dist_pct:.2f}%, Window: {struct_meta['window']}b, Quality: {struct_meta['quality_score']}/100) -> Scaled Leverage: {scaled_lev:.2f}x")
                                        else:
                                            log_event("INFO", f"[{iv}m ML Targets] Entry: {entry_price:.2f} | Dynamic SL: {stop_loss_price:.2f} (SL Dist: {raw_sl_dist:.2f}) | Regressor TP: {take_profit_price:.2f}")


                                        # Calibrated Position Sizing based on Isotonic Probability (Kelly scaling)
                                        c_prob = float(calibrated_confidence)
                                        current_hour_pkt = get_pkt_time().hour
                                        is_golden_hour = 18 <= current_hour_pkt < 21
                                    
                                        # Pre-calculate active trade stats needed for dynamic sizing
                                        with active_execution_lock:
                                            in_flight_margin_val = sum(active_execution_margins.values())
                                        total_active_size = sum(t.get("position_size_usd", 0.0) for tf_key in ["15m", "30m", "1h", "2h", "4h", "6h"] for t in bot_state.get(f"active_trade_{tf_key}", [])) + in_flight_margin_val
                                        current_bal = bot_state.get("simulated_balance", 80.0)
                                        if TRADE_MODE != "simulation":
                                            real_bal = get_real_bybit_balance_cached(force=True)
                                            if isinstance(real_bal, (int, float)) and real_bal > 0:
                                                current_bal = real_bal
                                        cov_multiplier, net_risk = calculate_covariance_multiplier(symbol, ml_trend, bot_state=bot_state)
                                    
                                        # Base size dynamic calculation using Risk Engine Conservative Kelly (Wilson CI / Bootstrap)
                                        from config import (
                                            MIN_POSITION_BALANCE_FRAC, MAX_POSITION_BALANCE_FRAC,
                                            CVAR_TAIL_PERCENTILE, CVAR_FALLBACK, DAILY_LOSS_BUDGET_FRAC
                                        )
                                        _latest_pred = bot_state.get(f"latest_prediction_{symbol}_{iv}") or bot_state.get(f"latest_prediction_{symbol}_{tf}") or {}
                                        _mcc_val = _latest_pred.get("manifest_mcc") if isinstance(_latest_pred, dict) else None

                                        executed_sl_dist = abs(entry_price - stop_loss_price)
                                        executed_tp_dist = abs(take_profit_price - entry_price)
                                        realized_sl_m = executed_sl_dist / max(1e-6, atr_dollars)
                                        realized_tp_m = executed_tp_dist / max(1e-6, atr_dollars)

                                        # Record labelled vs executed geometry for auditability (Finding #118, #84)
                                        rec.snapshot(
                                            labelled_sl_mult=float(cfg.get("sl_mult", 1.0)),
                                            labelled_tp_mult=float(cfg.get("tp_mult_trending" if "TRENDING" in str(regime_name).upper() else "tp_mult_ranging", 1.5)),
                                            executed_sl_mult=float(realized_sl_m),
                                            executed_tp_mult=float(realized_tp_m),
                                            executed_sl_dist=float(executed_sl_dist),
                                            executed_tp_dist=float(executed_tp_dist)
                                        )

                                        scaled_kelly = risk_engine.compute_conservative_kelly(
                                            calibrated_confidence=calibrated_confidence,
                                            tp_multiplier=realized_tp_m,
                                            sl_multiplier=realized_sl_m,
                                            interval=str(iv),
                                            trade_history=bot_state.get("trade_history", []),
                                            mcc_val=_mcc_val,
                                            haircut=realized_haircut,
                                            atr_norm=atr_norm_val,
                                            cost_bps=cost_bps
                                        )
                                        rec.kelly_effective = float(scaled_kelly)
                                    
                                        if scaled_kelly <= 0.0:
                                            print(f"[{symbol} {iv}m Kelly Sizing] Scaled Kelly is non-positive ({scaled_kelly:.4f}) — abstaining from trade entry (Fail-Closed).")
                                            log_event("INFO", f"[{symbol} {iv}m] Scaled Kelly non-positive ({scaled_kelly:.4f}) — abstaining from entry.")
                                            status_msg = "Skipped (Kelly Edge <= 0)"
                                            rec.reason_code = ReasonCode.KELLY_EDGE_NON_POSITIVE
                                            continue

                                        # Joint Risk Budget Allocation (MHI-governed fractional Kelly + Portfolio Heat + Liquidity Capping)
                                        portfolio_heat = min(1.0, total_active_size / max(1.0, current_bal))
                                        mhi_val = float(bot_state.get(f"mhi_{iv}", bot_state.get("mhi_score", 70.0)))
                                        rec.mhi_score = float(mhi_val)
                                        th_recs = bot_state.get("trade_history", [])
                                        df_th = pd.DataFrame(th_recs) if th_recs else df_completed
                                        budget_res = risk_engine.joint_risk_budget_allocator.allocate_risk_budget(
                                            symbol=symbol,
                                            entry_price=entry_price,
                                            atr_dollars=atr_dollars,
                                            atr_norm=atr_norm_val,
                                            calibrated_confidence=calibrated_confidence,
                                            direction=ml_trend,
                                            total_equity=current_bal,
                                            portfolio_heat=portfolio_heat,
                                            mhi_score=mhi_val,
                                            df_completed=df_th,
                                            trade_history=th_recs,
                                            interval=str(iv),
                                            mcc_val=_mcc_val,
                                            stop_distance=abs(entry_price - stop_loss_price),
                                            target_distance=abs(entry_price - take_profit_price)
                                        )
                                        rec.mhi_cap = float(budget_res.get("mhi_cap", 1.0)) if isinstance(budget_res, dict) else 1.0
                                        rec.kelly_raw = float(budget_res.get("raw_kelly", scaled_kelly)) if isinstance(budget_res, dict) else scaled_kelly

                                        if not budget_res.get("execution_permitted", True):
                                            rej_reason = budget_res.get("reason", "Halted by Risk Budget Allocator")
                                            print(f"[{symbol} {iv}m Joint Risk Budget Guard] Trade rejected: {rej_reason}")
                                            log_event("INFO", f"[{symbol} {iv}m] Trade rejected by JointRiskBudgetAllocator: {rej_reason}")
                                            status_msg = f"Skipped ({rej_reason})"
                                            rec.reason_code = ReasonCode.RISK_CHECKLIST_BLOCKED
                                            continue

                                        # Finding #141: Soft floor so conservative Kelly fractions are not artificially inflated above daily loss budget
                                        f_clamped = min(MAX_POSITION_BALANCE_FRAC, scaled_kelly)
                                    
                                        # Dimensional Kelly: f_clamped is the fraction of total capital at risk at the stop loss.
                                        # Capital at Risk = current_bal * f_clamped.
                                        # Target Notional = (current_bal * f_clamped) / stop_loss_frac.
                                        stop_loss_frac = max(0.002, abs(entry_price - stop_loss_price) / max(1e-9, entry_price))
                                        raw_notional_usd = (current_bal * f_clamped) / stop_loss_frac
                                        alloc_notional_usd = budget_res.get("position_size")
                                        if alloc_notional_usd is not None and alloc_notional_usd > 0:
                                            target_notional_usd = min(raw_notional_usd, alloc_notional_usd)
                                        else:
                                            target_notional_usd = raw_notional_usd
                                    
                                        # Covariance multiplier to account for existing correlations
                                        target_notional_usd = target_notional_usd * cov_multiplier
                                    
                                        # Volatility Regime Sizing Multiplier (Sweet spot 1.2x boost, extreme vol 0.5x, flat chop 0.3x)
                                        vol_regime_mult = risk_engine.get_volatility_regime_multiplier(atr_norm_val, iv)
                                        target_notional_usd = target_notional_usd * vol_regime_mult
                                        
                                        # Timeframe-Weighted Capital Allocation Multiplier (Finding #144 & #153)
                                        tf_sizing_mult = risk_engine.get_timeframe_sizing_multiplier(iv)
                                        target_notional_usd = target_notional_usd * tf_sizing_mult
                                        log_event("INFO", f"[{symbol} {iv}m Timeframe & Vol Sizing] VolMult: {vol_regime_mult:.2f}x | TFMult: {tf_sizing_mult:.2f}x -> Target Notional: ${target_notional_usd:.2f}")

                                        # Phase 1 Continuous Learning Engine Risk Multiplier (Enforces >= 50 closed trades floor)
                                        from learning_engine import continuous_learning_engine
                                        learning_risk_mult = continuous_learning_engine.get_risk_multiplier({
                                            "symbol": symbol,
                                            "interval": str(iv),
                                            "confidence": calibrated_confidence,
                                            "regime": regime_name
                                        })
                                        target_notional_usd = target_notional_usd * learning_risk_mult
                                        print(f"[{symbol} {iv}m Learning Engine Sizing] Multiplier: {learning_risk_mult:.2f}x -> Target Notional: ${target_notional_usd:.2f}")
                                    
                                        # CVaR (Expected Shortfall) Risk Constraint with Realized Daily Loss & Open Risk Tracking (Finding #141)
                                        try:
                                            hist_close = pd.to_numeric(df["close"], errors="coerce").values
                                            hist_close = hist_close[np.isfinite(hist_close) & (hist_close > 0)]
                                            if len(hist_close) > 30:
                                                returns_pct = (hist_close[1:] - hist_close[:-1]) / hist_close[:-1]
                                                returns_pct = returns_pct[np.isfinite(returns_pct)]
                                                if len(returns_pct) > 30 and float(np.std(returns_pct)) > 1e-6:
                                                    returns_sorted = np.sort(returns_pct)
                                                    alpha_idx = max(1, int(len(returns_sorted) * CVAR_TAIL_PERCENTILE))
                                                    tail_losses = returns_sorted[:alpha_idx]
                                                    tail_losses = tail_losses[np.isfinite(tail_losses)]
                                                    cvar_raw = abs(float(np.mean(tail_losses))) if len(tail_losses) > 0 else CVAR_FALLBACK
                                                    cvar_95 = max(0.01, min(1.0, cvar_raw)) if (np.isfinite(cvar_raw) and cvar_raw > 1e-4) else CVAR_FALLBACK
                                                else:
                                                    cvar_95 = CVAR_FALLBACK
                                            else:
                                                cvar_95 = CVAR_FALLBACK
                                            
                                            # Finding #141 & Finding #160 (#92): Daily loss budget tracking accumulator
                                            now_utc = datetime.now(timezone.utc)
                                            start_of_day_ts = now_utc.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                                            realized_loss_today = 0.0
                                            try:
                                                recent_completed = database.get_completed_trades(limit=200)
                                                for ct in recent_completed:
                                                    exit_ts = float(ct.get("exit_time") or 0.0)
                                                    if exit_ts >= start_of_day_ts:
                                                        pnl_val = float(ct.get("pnl_usd") or 0.0)
                                                        if pnl_val < 0.0:
                                                            realized_loss_today += abs(pnl_val)
                                            except Exception as ex_dt:
                                                log_event("CRITICAL", f"Error computing realized loss today (failing closed): {ex_dt}")
                                                realized_loss_today = current_bal * DAILY_LOSS_BUDGET_FRAC

                                            open_risk_usd = 0.0
                                            try:
                                                for _tf_k in ["15m", "30m", "1h", "2h", "4h", "6h", "15", "30", "60", "120", "240", "360"]:
                                                    for _op in bot_state.get(f"active_trade_{_tf_k}", []):
                                                        _p_entry = float(_op.get("entry_price", 0.0))
                                                        _p_sl = float(_op.get("stop_loss", 0.0))
                                                        _p_qty = float(_op.get("qty", 0.0))
                                                        if _p_entry > 0 and _p_sl > 0 and _p_qty > 0:
                                                            open_risk_usd += abs(_p_entry - _p_sl) * _p_qty
                                                        else:
                                                            _p_sz = float(_op.get("position_size_usd", 0.0))
                                                            _p_lev = float(_op.get("leverage", 1.0))
                                                            _p_sl_pct = float(_op.get("stop_loss_pct", 0.01))
                                                            open_risk_usd += (_p_sz * _p_lev * _p_sl_pct)
                                            except Exception as ex_or:
                                                log_event("WARNING", f"Error computing open risk: {ex_or}")
                                                open_risk_usd += (current_bal * 0.02)

                                            in_flight_risk_usd = float(bot_state.get("in_flight_risk_usd", 0.0))
                                            daily_loss_budget = current_bal * DAILY_LOSS_BUDGET_FRAC
                                            remaining_daily_budget = max(0.0, daily_loss_budget - realized_loss_today - open_risk_usd - in_flight_risk_usd)
                                            max_cvar_notional = remaining_daily_budget / (cvar_95 + 1e-8)
                                            log_event("INFO", f"[{iv}m CVaR Guard] 95% CVaR: {cvar_95*100:.2f}% | Daily Budget: ${daily_loss_budget:.2f} (Realized Loss: ${realized_loss_today:.2f}, Open Risk: ${open_risk_usd:.2f}, In-Flight: ${in_flight_risk_usd:.2f}, Remaining: ${remaining_daily_budget:.2f}) -> Max Notional Allowed: ${max_cvar_notional:.2f}")
                                        except Exception as cvar_err:
                                            log_event("CRITICAL", f"[CVaR Error] {cvar_err} — failing closed")
                                            max_cvar_notional = 0.0
                                            remaining_daily_budget = 0.0

                                        # Finding #160 (Finding #92 / Finding #44): Explicit rejection on exhausted daily loss budget including in-flight risk
                                        if remaining_daily_budget <= 0.0 or max_cvar_notional <= 0.0:
                                            abstain_reason = f"DAILY_LOSS_BUDGET_EXHAUSTED (Realized: ${realized_loss_today:.2f} + Open: ${open_risk_usd:.2f} + InFlight: ${in_flight_risk_usd:.2f} >= Budget: ${daily_loss_budget:.2f})"
                                            log_event("WARNING", f"[{symbol} {iv}m] Skipped: {abstain_reason}")
                                            rec.reject_reason = abstain_reason
                                            all_pass = False
                                            continue

                                        golden_mult = float(getattr(config, "GOLDEN_HOUR_MULTIPLIER", 1.0))
                                        if is_golden_hour and golden_mult > 1.0:
                                            # Golden Hour: boost slot allocation size up to hard risk limit
                                            from risk_limits import HARD_MAX_RISK_PER_TRADE_PCT
                                            max_allowed_notional_golden = (current_bal * HARD_MAX_RISK_PER_TRADE_PCT) / max(1e-4, stop_loss_frac)
                                            target_notional_usd = min(target_notional_usd * golden_mult, max_allowed_notional_golden)
                                            log_event("INFO", f"[{iv}m Golden Hour Kelly Sizing] Kelly Fraction: {scaled_kelly:.4f} -> Risk Fraction: {f_clamped*100:.1f}% -> Golden Target Notional: ${target_notional_usd:.2f} (Covariance: {cov_multiplier:.2f}x, Multiplier: {golden_mult}x)")
                                        else:
                                            print(f"[{iv}m Kelly Sizing] Kelly Fraction: {scaled_kelly:.4f} -> Risk Fraction: {f_clamped*100:.1f}% -> Target Notional: ${target_notional_usd:.2f} (Covariance: {cov_multiplier:.2f}x)")

                                        # Finding #128: CVaR daily loss budget strictly bounds target notional
                                        target_notional_usd = min(target_notional_usd, max_cvar_notional)
                                        
                                        # Base margin estimate before final leverage clamp
                                        from risk_limits import HARD_TIMEFRAME_MAX_LEVERAGE_CAPS
                                        tf_hard_cap = HARD_TIMEFRAME_MAX_LEVERAGE_CAPS.get(str(iv), 5.0)
                                        position_size_usd = target_notional_usd / max(1.0, tf_hard_cap)
                                        original_kelly_size = float(position_size_usd) # Keep intended size pre-clamp
                                        print(f"[{iv}m Trade Size Boundary Check] Target notional (CVaR constrained): ${target_notional_usd:.2f}")

                                        # Calculate Kelly parameters for logs and metadata (preserving variables for downstream use)
                                        kelly_fraction = scaled_kelly

                                        # 3. Concurrent Position Limit Check
                                        from config import MAX_CONCURRENT_POSITIONS
                                        total_active_count = sum(len(bot_state.get(f"active_trade_{tf_k}", [])) for tf_k in ["15m", "30m", "1h", "2h", "4h", "6h"]) + len(active_execution_symbols)

                                        # Ensure total size of active trades does not exceed the wallet balance
                                        min_bal_limit = 2.0
                                        min_size_limit = 2.0
                                    
                                        wallet_exceeded = False
                                        if total_active_count >= MAX_CONCURRENT_POSITIONS:
                                            log_event("WARNING", f"[{symbol} {iv}m] Skipped: MAX_CONCURRENT_POSITIONS ({MAX_CONCURRENT_POSITIONS}) reached")
                                            status_msg = f"Skipped (Max Concurrent Positions {MAX_CONCURRENT_POSITIONS} Reached)"
                                            wallet_exceeded = True
                                        elif current_bal <= min_bal_limit:
                                            print(f"[{symbol} {iv}m] Trade skipped: Wallet balance (${current_bal:.2f}) must be greater than ${min_bal_limit:.2f} to open new trades.")
                                            status_msg = "Skipped (Insufficient Balance)"
                                            wallet_exceeded = True
                                            send_telegram_alert(
                                                f"⚠️ *SIGNAL PASSED (SKIPPED - LOW BALANCE)* ⚠️\n"
                                                f"• *Asset*: {symbol}\n"
                                                f"• *Interval*: {iv}m\n"
                                                f"• *Direction*: {ml_trend}\n"
                                                f"• *Calibrated Confidence*: {calibrated_confidence * 100:.2f}%\n"
                                                f"• *Detail*: Wallet balance (${current_bal:.2f}) must be greater than ${min_bal_limit:.2f}."
                                            )
                                        elif total_active_size + position_size_usd > current_bal:
                                            remaining_bal = current_bal - total_active_size
                                            if remaining_bal >= min_size_limit:
                                                print(f"[{symbol} {iv}m] Sizing scaled down from ${position_size_usd:.2f} to ${remaining_bal:.2f} to fit remaining wallet balance (Total Active: ${total_active_size:.2f}, Wallet: ${current_bal:.2f}).")
                                                position_size_usd = remaining_bal
                                            else:
                                                print(f"[{symbol} {iv}m] Trade skipped: Insufficient wallet balance to maintain minimum ${min_size_limit:.2f} trade size (Total Active: ${total_active_size:.2f}, Wallet: ${current_bal:.2f}, Proposed: ${position_size_usd:.2f}).")
                                                status_msg = "Skipped (Exceeds Wallet)"
                                                wallet_exceeded = True
                                                send_telegram_alert(
                                                    f"⚠️ *SIGNAL PASSED (SKIPPED - LOW BALANCE)* ⚠️\n"
                                                    f"• *Asset*: {symbol}\n"
                                                    f"• *Interval*: {iv}m\n"
                                                    f"• *Direction*: {ml_trend}\n"
                                                    f"• *Calibrated Confidence*: {calibrated_confidence * 100:.2f}%\n"
                                                    f"• *Detail*: Insufficient wallet balance to maintain minimum ${min_size_limit:.2f} size (Total Active: ${total_active_size:.2f}, Wallet: ${current_bal:.2f}, Proposed: ${position_size_usd:.2f})."
                                                )

                                        if not wallet_exceeded:
                                            # B-1: Continuous Leverage Scaling: scale smoothly from 1.5x (at dynamic threshold) to 50x (at 100% confidence)
                                            c = float(calibrated_confidence)
                                            min_conf = dynamic_conf_threshold
                                            min_lev_start = getattr(config, "MIN_LEVERAGE_RAMP_START", 1.5)
                                            if c >= min_conf:
                                                leverage_val = min_lev_start + ((c - min_conf) / max(1e-9, 1.0 - min_conf)) * (50.0 - min_lev_start)
                                            else:
                                                leverage_val = min_lev_start
                                        
                                            # Base continuous leverage from confidence ramp
                                            if scaled_lev is not None and scaled_lev > 0 and str(iv) == "15":
                                                leverage_val = min(leverage_val, float(scaled_lev))

                                            # Risk check: cap leverage so stop loss doesn't exceed 90% of capital, based on ACTUAL stop distance
                                            actual_sl_dist = abs(entry_price - stop_loss_price)
                                            stop_loss_pct = (actual_sl_dist / max(1e-9, entry_price)) * 100.0
                                            max_safe_lev = 90.0 / stop_loss_pct if stop_loss_pct > 0 else 100.0
                                        
                                            if symbol == "BTCUSDT":
                                                lev_cap = 30.0
                                            elif symbol in ["ETHUSDT", "BNBUSDT", "SOLUSDT"]:
                                                lev_cap = 20.0
                                            else:
                                                lev_cap = 5.0

                                            # Governance Hard Timeframe Leverage Cap
                                            lev_cap = min(lev_cap, tf_hard_cap)

                                            # F-1: MCC Leverage Qualification Threshold Clamp
                                            mcc_val = pred_entry_dict.get("manifest_mcc") if 'pred_entry_dict' in locals() else (locals().get("pred_info", {}).get("manifest_mcc"))
                                            mcc_thresh = getattr(config, "MCC_LEVERAGE_QUALIFICATION_THRESHOLD", 0.15)
                                            if mcc_val is None or mcc_val < mcc_thresh:
                                                cons_caps = getattr(config, "CONSERVATIVE_LEVERAGE_CAPS", {})
                                                mcc_cap = cons_caps.get(symbol, cons_caps.get("default", 3.0))
                                                lev_cap = min(lev_cap, mcc_cap)
                                                _shown = f"{mcc_val:.4f}" if mcc_val is not None else "unavailable"
                                                log_event("INFO", f"[{symbol} {iv}m F-1 Leverage Guard] Model MCC ({_shown} < {mcc_thresh:.4f}) — Clamped max leverage to {lev_cap:.1f}x.")

                                            # Volatility-based leverage scaling cap
                                            atr_pct_of_price = (atr_dollars / entry_price) * 100.0
                                            if atr_pct_of_price > 3.0:
                                                vol_lev_cap = 2.0 if symbol in ["BTCUSDT", "ETHUSDT"] else 1.0
                                                lev_cap = min(lev_cap, vol_lev_cap)
                                                print(f"[{symbol} {iv}m Volatility-Scaled Leverage] Extreme Volatility Detected (ATR = {atr_pct_of_price:.2f}% of price). Capped leverage to {lev_cap}x.")
                                            elif atr_pct_of_price > 1.5:
                                                lev_cap = lev_cap * 0.5
                                                print(f"[{symbol} {iv}m Volatility-Scaled Leverage] High Volatility Detected (ATR = {atr_pct_of_price:.2f}% of price). Halved leverage cap to {lev_cap}x.")
                                        
                                            # F-4: Apply Golden Hour multiplier before final lev_cap and max_safe_lev single clamp
                                            current_hour_pkt = get_pkt_time().hour
                                            if 18 <= current_hour_pkt < 21:
                                                leverage_val *= 2.0
                                            
                                            # Sharpe-Adaptive Leverage Multiplier (Dynamic drawdown safety)
                                            sharpe_mult = calculate_recent_performance_leverage_multiplier(days=7)
                                            leverage_val = leverage_val * sharpe_mult
                                            
                                            leverage_val = round(max(1.0, min(lev_cap, min(leverage_val, max_safe_lev))), 1)

                                            # === FIX 1 & 3: COMBINED PRE-FLIGHT ENTRY VALIDATION ===
                                            is_valid, adjusted_struct, struct_log = validate_trade_structure(
                                                entry_price=entry_price,
                                                stop_price=stop_loss_price,
                                                tp_price=take_profit_price,
                                                atr_dollars=atr_dollars,
                                                leverage=leverage_val,
                                                interval=iv,
                                                symbol=symbol,
                                                direction=ml_trend
                                            )
                                            if not is_valid:
                                                print(f"[{symbol} {iv}m PRE-FLIGHT REJECT] Trade submission aborted: {struct_log}")
                                                status_msg = "Skipped (Min R:R Floor Reject)"
                                                continue

                                            if struct_log != "OK":
                                                print(f"[{symbol} {iv}m Pre-Flight Audit] {struct_log}")

                                            orig_sl_price = stop_loss_price
                                            orig_tp_price = take_profit_price
                                            new_stop_price = adjusted_struct["stop_price"]
                                            new_tp_price = adjusted_struct["tp_price"]
                                            leverage_val = adjusted_struct["leverage"]

                                            # Reconcile geometry: If validate_trade_structure widened stop or capped TP, verify economic gate and recompute Kelly (Finding #52, #60, #16, #26)
                                            if abs(new_stop_price - orig_sl_price) > 1e-6 or abs(new_tp_price - orig_tp_price) > 1e-6:
                                                from trade_calculators import passes_economic_gate, calculate_required_p
                                                fr_struct = get_funding_rate(symbol)
                                                cfg_lookahead = TIMEFRAME_CONFIG.get(str(iv), {}).get("lookahead", 10)
                                                horizon_hours_struct = (float(iv) * float(cfg_lookahead)) / 60.0
                                                exp_funding_struct = max(0.0, (fr_struct if ml_trend == "Bullish" else -fr_struct) * (horizon_hours_struct / 8.0))
                                                if not passes_economic_gate(entry=entry_price, tp=new_tp_price, sl=new_stop_price, conf=calibrated_confidence, expected_funding_frac=exp_funding_struct):
                                                    _req_p = calculate_required_p(entry=entry_price, tp=new_tp_price, sl=new_stop_price, expected_funding_frac=exp_funding_struct)
                                                    print(f"[{symbol} {iv}m PRE-FLIGHT REJECT] Adjusted trade structure fails economic gate (requires {_req_p:.3f} > conf {calibrated_confidence:.3f}). Aborting entry.")
                                                    status_msg = "Skipped (Adjusted Structure Fails Economic Gate)"
                                                    continue

                                                # Finding #16 & #26: Re-evaluate dynamic confidence threshold on adjusted geometry
                                                post_sl_m = abs(entry_price - new_stop_price) / max(1e-6, atr_dollars)
                                                post_tp_m = abs(new_tp_price - entry_price) / max(1e-6, atr_dollars)
                                                post_eff_tp = post_tp_m * realized_haircut
                                                post_p_star = post_sl_m / max(1e-6, (post_eff_tp + post_sl_m))
                                                post_cost_adj = (cost_bps / 1e4) / max(1e-6, (post_eff_tp + post_sl_m) * max(1e-4, atr_norm_val))
                                                post_dynamic_threshold = float(round(post_p_star + post_cost_adj, 4))
                                                if 'adjustments_applied' in locals():
                                                    for adj_name, adj_val in adjustments_applied:
                                                        if adj_name != "economic_base":
                                                            post_dynamic_threshold += float(adj_val)
                                                if calibrated_confidence < post_dynamic_threshold:
                                                    log_event("WARNING", f"[{symbol} {iv}m PRE-FLIGHT REJECT] Adjusted trade structure below dynamic confidence threshold (requires {post_dynamic_threshold:.3f} > conf {calibrated_confidence:.3f}). Aborting entry.")
                                                    status_msg = "Skipped (Adjusted Structure Below Dynamic Confidence Threshold)"
                                                    continue
                                                p_star = post_p_star

                                                # Finding #52 & #16: Recompute Kelly with post-sanitizer geometry and measured cost_bps
                                                re_scaled_kelly = risk_engine.compute_conservative_kelly(
                                                    calibrated_confidence=calibrated_confidence,
                                                    tp_multiplier=post_tp_m,
                                                    sl_multiplier=post_sl_m,
                                                    interval=str(iv),
                                                    trade_history=bot_state.get("trade_history", []),
                                                    mcc_val=_mcc_val,
                                                    haircut=realized_haircut,
                                                    atr_norm=atr_norm_val,
                                                    cost_bps=cost_bps
                                                )
                                                if re_scaled_kelly <= 0.0:
                                                    log_event("INFO", f"[{symbol} {iv}m] Recomputed Kelly non-positive ({re_scaled_kelly:.4f}) post-sanitizer — abstaining from entry.")
                                                    status_msg = "Skipped (Kelly Edge <= 0 Post-Sanitizer)"
                                                    rec.reason_code = ReasonCode.KELLY_EDGE_NON_POSITIVE
                                                    continue
                                                new_f_clamped = min(MAX_POSITION_BALANCE_FRAC, re_scaled_kelly)
                                                if f_clamped > 0:
                                                    target_notional_usd = target_notional_usd * (new_f_clamped / f_clamped)
                                                scaled_kelly = re_scaled_kelly
                                                f_clamped = new_f_clamped
                                                rec.kelly_effective = float(scaled_kelly)

                                            take_profit_price = new_tp_price

                                            # Proportional stop-adjustment to preserve all upstream risk budgets (Joint Allocator, Covariance, Vol Regime, Learning Engine, CVaR)
                                            new_stop_loss_frac = max(0.002, abs(entry_price - new_stop_price) / max(1e-9, entry_price))
                                            if abs(new_stop_loss_frac - stop_loss_frac) > 1e-9:
                                                target_notional_usd = target_notional_usd * (stop_loss_frac / new_stop_loss_frac)
                                            stop_loss_price = new_stop_price
                                            target_notional_usd = min(target_notional_usd, current_bal * lev_cap)

                                            cfg = TIMEFRAME_CONFIG.get(str(iv), {"lookahead": 10})
                                            lookahead = cfg.get("lookahead", 10)
                                            duration_seconds = int(iv) * 60.0 * lookahead
                                            import uuid
                                            trade_uuid = str(uuid.uuid4())
                                            
                                            # Check free available margin and minimum exchange requirements
                                            available_margin = max(0.0, current_bal - total_active_size)
                                            min_exchange_notional = getattr(config, "MIN_ORDER_VALUE_USDT", 5.1)
                                            min_req_margin = min_exchange_notional / max(1.0, leverage_val)
                                            raw_target_margin = target_notional_usd / max(1.0, leverage_val)

                                            if raw_target_margin < min_req_margin:
                                                if available_margin >= min_req_margin:
                                                    # Bump to minimum exchange order size ($5.10 USDT) for small balance accounts
                                                    raw_target_margin = min_req_margin
                                                    target_notional_usd = min_exchange_notional
                                                    print(f"[{symbol} {iv}m Position Size Bump] Sized up to exchange minimum notional (${min_exchange_notional:.2f}).")
                                                else:
                                                    log_event("INFO", f"[{symbol} {iv}m] Risk budget allocation (${raw_target_margin:.2f}) below exchange minimum (${min_req_margin:.2f}). Cleanly skipping entry.")
                                                    status_msg = "Skipped (Below Risk Allocation Floor)"
                                                    continue

                                            if available_margin < min_req_margin:
                                                log_event("WARNING", f"[{symbol} {iv}m] Insufficient free margin (${available_margin:.2f}) for required position margin (${min_req_margin:.2f}). Skipping entry.")
                                                status_msg = "Skipped (Insufficient Free Margin)"
                                                continue

                                            # Calculate initial proposed position size cleanly clamped to available margin
                                            position_size_usd = min(available_margin, raw_target_margin)

                                            # Pre-Trade Signal Guard Check
                                            pred_info = bot_state.get(f"latest_prediction_{symbol}_{iv}") or bot_state.get(f"latest_prediction_{symbol}_{iv}m") or {}
                                            if pred_info.get("is_fallback", False) or pred_info.get("signal_source") in ["RULE_BASED_FALLBACK", "UNSET"]:
                                                position_size_usd *= 0.50
                                                print(f"[{symbol} {iv}m Signal Guard] Rule-based fallback signal detected: Applied 50% position sizing penalty.")

                                            # Apply P4 Anti-Martingale scaling to candidate sizing before risk gates
                                            from risk_engine import calculate_anti_martingale_risk_multiplier
                                            peak_eq = float(bot_state.get("peak_balance", current_bal) or current_bal)
                                            am_res = calculate_anti_martingale_risk_multiplier(
                                                current_bal,
                                                peak_eq,
                                                bot_state.get("trade_history", [])
                                            )
                                            am_mult = float(am_res.get("multiplier", 1.0))
                                            # Invariant: Never inflate sizing above 1.0x if currently in a drawdown (Finding #81)
                                            if current_bal < peak_eq * 0.985:
                                                am_mult = min(1.0, am_mult)
                                            position_size_usd = max(0.0, float(position_size_usd * am_mult))
                                            # Re-clamp candidate position size to available free margin to avoid exceeding allocation limit
                                            position_size_usd = min(available_margin * 0.95, position_size_usd)
                                            if am_mult != 1.0:
                                                print(f"[{symbol} {iv}m Anti-Martingale] Applied streak multiplier {am_mult:.2f}x to candidate size -> ${position_size_usd:.2f}")

                                            # Adaptive Volume Gate Check
                                            vol_pass, vol_msg, vol_pctile = adaptive_volume_gate.check(symbol, kline_df=df_completed)
                                            print(f"[{symbol} {iv}m Volume Gate] {vol_msg}")
                                            if not vol_pass:
                                                print(f"[{symbol} {iv}m Volume Gate Block] Trade entry aborted.")
                                                status_msg = "Skipped (Volume Gate Block)"
                                                wallet_exceeded = True
                                                bybit_success = False

                                            active_trades_list = [t for tf_key in ["15m", "30m", "1h", "2h", "4h", "6h"] for t in bot_state.get(f"active_trade_{tf_key}", [])]
                                            df_dict = {symbol: df_completed}
                                            for t in active_trades_list:
                                                pos_sym = t.get("symbol")
                                                if pos_sym and pos_sym != symbol and pos_sym not in df_dict:
                                                    try:
                                                        df_pos = get_history(symbol=pos_sym, interval=str(iv), limit=100)
                                                        if df_pos is not None and not df_pos.empty:
                                                            df_dict[pos_sym] = df_pos
                                                    except Exception:
                                                        pass
                                            log_event("INFO", f"[{symbol} {iv}m] Journalling decision: {status_msg}")
                                            # rec already initialized at start of evaluation
                                            rec.snapshot(
                                                prediction=pred_info,
                                                equity=float(bot_state.get("live_balance", bot_state.get("wallet_balance", bot_state.get("simulated_balance", 80.0)))),
                                                open_positions_count=len(active_trades_list),
                                                wallet_exceeded=wallet_exceeded
                                            )
                                            rec.signal_source      = str(pred_info.get("signal_source") or rec.signal_source or "UNSET")
                                            rec.is_fallback        = int(pred_info.get("is_fallback") if pred_info.get("is_fallback") is not None else (rec.is_fallback or 0))
                                            rec.direction          = ml_trend or rec.direction
                                            rec.raw_confidence     = pred_info.get("raw_confidence") if pred_info.get("raw_confidence") is not None else rec.raw_confidence
                                            rec.calibrated_conf    = pred_info.get("calibrated_confidence") if pred_info.get("calibrated_confidence") is not None else rec.calibrated_conf
                                            rec.calibrator_version = pred_info.get("calibrator_version") or rec.calibrator_version
                                            rec.calibrator_ece     = pred_info.get("calibrator_ece") if pred_info.get("calibrator_ece") is not None else rec.calibrator_ece
                                            rec.model_version      = pred_info.get("model_version") or rec.model_version
                                            rec.feature_hash       = pred_info.get("feature_contract_hash") or pred_info.get("feature_hash") or rec.feature_hash
                                            rec.manifest_schema    = pred_info.get("manifest_schema_version") if pred_info.get("manifest_schema_version") is not None else rec.manifest_schema
                                            rec.git_sha            = pred_info.get("git_sha") or rec.git_sha
                                            rec.regime             = pred_info.get("regime_mode") or pred_info.get("regime") or rec.regime or str(regime_name)
                                            rec.adx                = pred_info.get("adx") if pred_info.get("adx") is not None else (rec.adx or adx_regime)
                                            rec.atr_norm           = pred_info.get("atr_norm") if pred_info.get("atr_norm") is not None else (rec.atr_norm or atr_norm_val)
                                            rec.liquidity_score    = float(liq_score) if 'liq_score' in locals() else bot_state.get("liquidity_score", 1.0)
                                            rec.spread_bp          = float(current_spread_bps) if 'current_spread_bps' in locals() else rec.spread_bp
                                            rec.expected_value     = float(exp_edge_bps) if 'exp_edge_bps' in locals() else rec.expected_value
                                            rec.expected_rr        = float(exp_r_val) if 'exp_r_val' in locals() else rec.expected_rr
                                            rec.round_trip_cost_bp = float(cost_bps) if 'cost_bps' in locals() else rec.round_trip_cost_bp
                                            rec.position_size_usd  = position_size_usd
                                            rec.leverage           = leverage_val

                                            try:
                                                passed_checklist, checklist_msg, dd_mult, capped_size = risk_engine.evaluate_pre_trade_checklist(
                                                    symbol, position_size_usd, leverage_val, active_trades_list, bot_state, df_dict, interval=str(iv), direction=ml_trend, journal=rec
                                                )
                                                rec.outcome = "APPROVED" if (passed_checklist and not wallet_exceeded) else "REJECTED"
                                                rec.reject_reason = None if (passed_checklist and not wallet_exceeded) else (checklist_msg if not passed_checklist else "Wallet allocation exceeded")
                                                if not (passed_checklist and not wallet_exceeded):
                                                    rec.reason_code = ReasonCode.MARGIN_GUARD_EXCEEDED if wallet_exceeded else ReasonCode.RISK_CHECKLIST_BLOCKED
                                                if passed_checklist and not wallet_exceeded:
                                                    rec.position_size_usd = min(capped_size, max(0.0, float(capped_size * dd_mult)))
                                            except Exception as risk_err:
                                                rec.outcome = "ERROR"
                                                rec.reject_reason = f"Risk checklist exception: {risk_err}"
                                                print(f"[{symbol} {iv}m CRITICAL RISK CHECKLIST EXCEPTION] {risk_err}. Aborting trade entry (Fail-Closed).")
                                                passed_checklist = False
                                                checklist_msg = f"REJECTED: Risk Checklist Exception ({risk_err})"
                                                dd_mult = 0.0
                                                capped_size = 0.0

                                            print(f"[{symbol} {iv}m Pre-Trade Checklist] {checklist_msg}")
                                            if not passed_checklist or wallet_exceeded:
                                                print(f"[{symbol} {iv}m Risk Checklist Block] Trade entry aborted.")
                                                if not passed_checklist:
                                                    status_msg = "Skipped (Risk Checklist Block)"
                                                wallet_exceeded = True
                                                bybit_success = False
                                            else:
                                                # Finding #154: Defensive validation for invalid entry_price / current_price to prevent ZeroDivisionError
                                                import math
                                                if entry_price is None or (isinstance(entry_price, float) and (math.isnan(entry_price) or entry_price <= 0)) or float(entry_price) <= 0:
                                                    log_event("WARNING", f"[{symbol} {iv}m Position Sizing] Invalid entry_price ({entry_price}). Aborting trade execution.")
                                                    wallet_exceeded = True
                                                    bybit_success = False
                                                    status_msg = "Skipped (Invalid Entry Price)"
                                                    continue

                                                # Finding N2 & R48: Enforce pre-order minimum stop floor BEFORE sizing, Kelly, and risk guards
                                                min_atr_mult = 1.25 if float(leverage_val) > 10.0 else 1.0
                                                min_sl_cfg = getattr(config, "MIN_SL_PCT_CONFIG", {})
                                                min_sl_pct = float(min_sl_cfg.get(str(iv), min_sl_cfg.get("default", 0.008)))
                                                min_allowed_sl_dist = max(atr_dollars * min_atr_mult, entry_price * min_sl_pct)
                                                current_sl_dist = abs(entry_price - stop_loss_price)
                                                if current_sl_dist < min_allowed_sl_dist:
                                                    target_rr = abs(take_profit_price - entry_price) / max(1e-6, current_sl_dist)
                                                    new_sl_dist = min_allowed_sl_dist
                                                    if str(ml_trend).upper() in ["BULLISH", "LONG", "BUY"]:
                                                        stop_loss_price = entry_price - new_sl_dist
                                                        take_profit_price = entry_price + max(abs(take_profit_price - entry_price), new_sl_dist * target_rr)
                                                    else:
                                                        stop_loss_price = entry_price + new_sl_dist
                                                        take_profit_price = entry_price - max(abs(take_profit_price - entry_price), new_sl_dist * target_rr)
                                                    sl_multiplier_adjusted = new_sl_dist / max(1e-6, atr_dollars)
                                                    tp_multiplier_adjusted = abs(take_profit_price - entry_price) / max(1e-6, atr_dollars)
                                                    resolved_sl_dist = new_sl_dist

                                                # Final position size strictly clamped to validated checklist cap with drawdown scaling
                                                position_size_usd = min(capped_size, max(0.0, float(capped_size * dd_mult)))
                                                leveraged_size = position_size_usd * leverage_val
                                                bot_state["position_size_usd"] = float(leveraged_size)
                                                raw_qty = leveraged_size / entry_price if entry_price > 0 else 0.0
                                                qty_str = format_bybit_qty(symbol, raw_qty)
                                                qty_val = float(qty_str) if qty_str else 0.0

                                                original_notional = leveraged_size
                                                original_stop_dist = abs(entry_price - stop_loss_price)
                                                original_risk_usd = (original_notional / max(1e-8, entry_price)) * original_stop_dist
                                                is_oversized_trade = False

                                                # Enforce minimum order value from config (default 5.1 USDT) on final post-checklist size
                                                min_order_value = getattr(config, "MIN_ORDER_VALUE_USDT", 5.1)
                                                if qty_val * entry_price < min_order_value:
                                                    step = get_bybit_min_qty_step(symbol)
                                                    required_qty = min_order_value / entry_price
                                                    import math
                                                    if step > 0:
                                                        qty_val = math.ceil(required_qty / step) * step
                                                        qty_str = format_bybit_qty(symbol, qty_val)
                                                        qty_val = float(qty_str)
                                                        raw_qty = qty_val
                                                    
                                                    scaled_notional = qty_val * entry_price
                                                    clamped_val = float(position_size_usd)
                                                    final_val = float(scaled_notional / max(1e-9, leverage_val))
                                                    if final_val > clamped_val:
                                                        log_event("INFO", f"[{symbol} {iv}m] Scaled UP to min order value: ${clamped_val:.2f} -> ${final_val:.2f}")
                                                
                                                    # Priority 1: Maintain structural stop width & Brownian noise clearance envelope
                                                    min_atr_mult = 1.25 if float(leverage_val) > 10.0 else 1.0
                                                    min_sl_cfg = getattr(config, "MIN_SL_PCT_CONFIG", {})
                                                    min_sl_pct = float(min_sl_cfg.get(str(iv), min_sl_cfg.get("default", 0.008)))
                                                    min_allowed_sl_dist = max(atr_dollars * min_atr_mult, entry_price * min_sl_pct)
                                                    
                                                    # Keep structural stop distance; only compress if stop is excessively wide
                                                    new_stop_dist = max(original_stop_dist, min_allowed_sl_dist)
                                                
                                                    if str(ml_trend).upper() in ["BULLISH", "LONG", "BUY"]:
                                                        new_sl_price = entry_price - new_stop_dist
                                                    else:
                                                        new_sl_price = entry_price + new_stop_dist

                                                    scaled_risk_usd = (scaled_notional / max(1e-8, entry_price)) * new_stop_dist
                                                
                                                    # Priority 2: Risk Check — Strictly bounded by MAX_SCALED_RISK_CAP_RATIO (1.10x) (Finding #91)
                                                    max_allowed_risk_ratio = getattr(config, "MAX_SCALED_RISK_CAP_RATIO", 1.10)
                                                    max_allowed_risk_usd = original_risk_usd * max_allowed_risk_ratio
                                                    if scaled_risk_usd > max_allowed_risk_usd:
                                                        print(f"[{symbol} {iv}m Risk Guard] REJECTED: Scaling to ${scaled_notional:.2f} would exceed risk budget (Scaled: ${scaled_risk_usd:.2f} vs Max: ${max_allowed_risk_usd:.2f})")
                                                        status_msg = "Skipped (Exceeds Risk Cap)"
                                                        wallet_exceeded = True
                                                    elif final_val > available_margin:
                                                        print(f"[{symbol} {iv}m Margin Guard] REJECTED: Min order value required margin (${final_val:.2f}) exceeds free available margin (${available_margin:.2f}). Trade entry aborted.")
                                                        status_msg = "Skipped (Insufficient Free Margin for Min Order)"
                                                        wallet_exceeded = True
                                                    else:
                                                        if new_stop_dist > original_stop_dist + 1e-4:
                                                            from trade_calculators import passes_economic_gate, calculate_required_p
                                                            re_sl_m = new_stop_dist / max(1e-6, atr_dollars)
                                                            re_eff_tp = resolved_tp_m * realized_haircut
                                                            re_p_star = re_sl_m / max(1e-6, (re_eff_tp + re_sl_m))
                                                            re_cost_adj = (cost_bps / 1e4) / max(1e-6, (re_eff_tp + re_sl_m) * max(1e-4, atr_norm_val))
                                                            re_gate_threshold = float(round(re_p_star + re_cost_adj, 4))
                                                            econ_pass = passes_economic_gate(entry=entry_price, tp=take_profit_price, sl=new_sl_price, conf=calibrated_confidence)
                                                            req_p_val = calculate_required_p(entry=entry_price, tp=take_profit_price, sl=new_sl_price)
                                                            if calibrated_confidence < max(re_gate_threshold, req_p_val) or not econ_pass:
                                                                log_event("WARNING", f"[{symbol} {iv}m Min Order Risk] REJECTED: Widened min-order stop fails economic break-even gate (requires {max(re_gate_threshold, req_p_val):.3f} > conf {calibrated_confidence:.3f}). Aborting entry.")
                                                                status_msg = "Skipped (Widened Stop Below Economic Break-Even)"
                                                                wallet_exceeded = True
                                                            else:
                                                                p_star = req_p_val

                                                        if not wallet_exceeded:
                                                            stop_loss_price = new_sl_price
                                                            # Finding #47: Re-evaluate portfolio checklist against scaled-up final_val
                                                            if final_val > position_size_usd:
                                                                try:
                                                                    re_passed, re_msg, re_dd_mult, re_capped_size = risk_engine.evaluate_pre_trade_checklist(
                                                                        symbol, final_val, leverage_val, active_trades_list, bot_state, df_dict, interval=str(iv), direction=ml_trend, journal=None
                                                                    )
                                                                    if not re_passed:
                                                                        log_event("WARNING", f"[{symbol} {iv}m Min Order Risk] Scaled up size (${final_val:.2f}) failed re-evaluated portfolio checklist: {re_msg}. Aborting entry.")
                                                                        status_msg = f"Skipped (Min Order Bump Checklist Fail: {re_msg})"
                                                                        wallet_exceeded = True
                                                                    elif re_capped_size is not None and float(re_capped_size) < (final_val - 1e-4):
                                                                        log_event("WARNING", f"[{symbol} {iv}m Min Order Risk] Scaled up size (${final_val:.2f}) exceeds risk limit cap (${float(re_capped_size):.2f}). Aborting entry.")
                                                                        status_msg = f"Skipped (Min Order Bump Exceeds Cap: {re_msg})"
                                                                        wallet_exceeded = True
                                                                    else:
                                                                        final_val = min(final_val, float(re_capped_size if re_capped_size is not None else final_val))
                                                                except Exception as _chk_err:
                                                                    log_event("WARNING", f"[{symbol} {iv}m Min Order Risk] Checklist re-eval exception: {_chk_err}")
                                                                    wallet_exceeded = True
                                                            if not wallet_exceeded:
                                                                position_size_usd = min(available_margin, final_val)
                                                                is_oversized_trade = True
                                                                bot_state["position_size_usd"] = float(position_size_usd * leverage_val)
                                                                print(f"[{symbol} {iv}m API] Enforced minimum order value (${scaled_notional:.2f}). Applied noise-clearance SL (${new_stop_dist:.4f}) with total risk ${scaled_risk_usd:.2f}.")

                                                # Priority 3: Balance Guard - If margin exceeds 90% of free available margin, reject trade.
                                                required_margin = (qty_val * entry_price) / max(1e-8, leverage_val)
                                                if not wallet_exceeded and required_margin > available_margin * 0.90:
                                                    print(f"[{symbol} {iv}m Margin Guard] REJECTED: Required margin (${required_margin:.2f}) exceeds 90% of free available margin (${available_margin:.2f}). Trade entry aborted.")
                                                    status_msg = "Skipped (Exceeds Free Margin)"
                                                    wallet_exceeded = True

                                                # Post-Floor Geometry & Economic Viability Recheck
                                                post_floor_pass = True
                                                try:
                                                    trade_calculators.assert_valid_geometry(ml_trend, entry_price, stop_loss_price, take_profit_price, symbol=f"{symbol} {iv}m")
                                                except ValueError as geom_err:
                                                    log_event("ERROR", str(geom_err))
                                                    status_msg = f"Skipped (Invalid {ml_trend} Geometry)"
                                                    post_floor_pass = False
                                            
                                                if post_floor_pass and all_pass and not wallet_exceeded:
                                                    from trade_calculators import passes_economic_gate, calculate_required_p
                                                    # Finding #52: Recompute Kelly with post-floor SL
                                                    final_sl_m = abs(entry_price - stop_loss_price) / max(1e-6, atr_dollars)
                                                    final_tp_m = abs(take_profit_price - entry_price) / max(1e-6, atr_dollars)
                                                    re_scaled_kelly_final = risk_engine.compute_conservative_kelly(
                                                        calibrated_confidence=calibrated_confidence,
                                                        tp_multiplier=final_tp_m,
                                                        sl_multiplier=final_sl_m,
                                                        interval=str(iv),
                                                        trade_history=bot_state.get("trade_history", []),
                                                        mcc_val=_mcc_val,
                                                        haircut=realized_haircut,
                                                        atr_norm=atr_norm_val,
                                                        cost_bps=cost_bps
                                                    )
                                                    if re_scaled_kelly_final <= 0.0:
                                                        log_event("WARNING", f"[{symbol} {iv}m] Post-floor SL widening reduced Kelly edge to non-positive ({re_scaled_kelly_final:.4f}). Aborting entry.")
                                                        status_msg = "Skipped (Kelly Edge <= 0 Post-Floor)"
                                                        post_floor_pass = False
                                                    else:
                                                        rec.kelly_effective = float(re_scaled_kelly_final)
                                                        fr_pf = get_funding_rate(symbol)
                                                        exp_funding_pf = max(0.0, (fr_pf if ml_trend == "Bullish" else -fr_pf) * (duration_seconds / 3600.0 / 8.0))
                                                        if not passes_economic_gate(entry=entry_price, tp=take_profit_price, sl=stop_loss_price, conf=calibrated_confidence, expected_funding_frac=exp_funding_pf):
                                                            final_sl_dist = abs(entry_price - stop_loss_price)
                                                            final_tp_dist = abs(take_profit_price - entry_price)
                                                            final_rr = (final_tp_dist / max(1e-9, final_sl_dist))
                                                            _req_p = calculate_required_p(entry=entry_price, tp=take_profit_price, sl=stop_loss_price, expected_funding_frac=exp_funding_pf)
                                                            log_event("WARNING", f"[{symbol} {iv}m] Post-floor SL widening reduced R:R to {final_rr:.2f}; calibrated confidence {calibrated_confidence:.3f} < required threshold {_req_p:.3f}. Aborting entry.")
                                                            status_msg = f"Skipped (Post-Floor R:R {final_rr:.2f} Econ Fail)"
                                                            post_floor_pass = False
                                                        else:
                                                            p_star = calculate_required_p(entry=entry_price, tp=take_profit_price, sl=stop_loss_price, expected_funding_frac=exp_funding_pf)
                                            
                                                if not post_floor_pass or not all_pass:
                                                    wallet_exceeded = True

                                                if not wallet_exceeded:
                                                    # Terminal Risk-at-Stop Hard Boundary Assertion
                                                    from risk_limits import HARD_MAX_RISK_PER_TRADE_PCT
                                                    min_sl_cfg = getattr(config, "MIN_SL_PCT_CONFIG", {})
                                                    min_sl_pct = float(min_sl_cfg.get(str(iv), min_sl_cfg.get("default", 0.008)))
                                                    min_allowed_sl = max(atr_dollars * 1.0, entry_price * min_sl_pct)
                                                    final_stop_dist = max(abs(entry_price - stop_loss_price), min_allowed_sl)
                                                    if final_stop_dist > abs(entry_price - stop_loss_price) + 1e-6:
                                                        stop_loss_price = (entry_price - final_stop_dist) if ml_trend == "Bullish" else (entry_price + final_stop_dist)
                                                    terminal_risk_usd = (qty_val * final_stop_dist)
                                                    max_terminal_risk_usd = current_bal * HARD_MAX_RISK_PER_TRADE_PCT
                                                    if terminal_risk_usd > max_terminal_risk_usd + 1e-6:
                                                        max_allowed_q = max_terminal_risk_usd / max(1e-8, final_stop_dist)
                                                        c_qty_str = format_bybit_qty(symbol, max_allowed_q)
                                                        c_qty_val = float(c_qty_str) if c_qty_str else 0.0
                                                        if c_qty_val * entry_price < min_order_value:
                                                            log_event("WARNING", f"[{symbol} {iv}m Terminal Risk Guard] Clamping risk (${terminal_risk_usd:.2f} -> ${max_terminal_risk_usd:.2f}) reduces order below min notional (${min_order_value:.2f}). Aborting entry.")
                                                            status_msg = "Skipped (Risk Cap Exceeds Min Notional)"
                                                            wallet_exceeded = True
                                                        else:
                                                            log_event("INFO", f"[{symbol} {iv}m Terminal Risk Guard] Clamped risk from ${terminal_risk_usd:.2f} to ${max_terminal_risk_usd:.2f} (3% hard cap). Qty: {qty_str} -> {c_qty_str}")
                                                            qty_val = c_qty_val
                                                            raw_qty = c_qty_val
                                                            qty_str = c_qty_str
                                                            position_size_usd = (raw_qty * entry_price) / max(1.0, leverage_val)

                                                if not wallet_exceeded:
                                                    from execution_validator import ExecutionValidator

                                                    # Fetch live market bid/ask/last to guard against executing when price has already breached the stop loss or drifted adversely
                                                    live_bid, live_ask, live_last = get_bybit_bid_ask(symbol)
                                                    if live_bid is not None and live_ask is not None and live_bid > 0 and live_ask > 0:
                                                        live_current_price = (live_bid + live_ask) / 2.0
                                                    elif live_last is not None and live_last > 0:
                                                        live_current_price = live_last
                                                    else:
                                                        log_event("WARNING", f"[{symbol} {iv}m Adverse Drift Guard] Cannot retrieve live market price. Skipping entry (Fail-Closed).")
                                                        status_msg = "Skipped (Market Price Unavailable)"
                                                        wallet_exceeded = True
                                                        bybit_success = False
                                                        live_current_price = None

                                                    if live_current_price is not None:
                                                        # Adverse Price-Drift Check (comparing live mid to stale candle close entry)
                                                        max_adverse_drift = max(0.25 * atr_dollars, entry_price * 0.0025)
                                                        if str(ml_trend).upper() in ("BUY", "LONG", "BULLISH") and (entry_price - live_current_price) > max_adverse_drift:
                                                            adverse_pts = entry_price - live_current_price
                                                            log_event("WARNING", f"[{symbol} {iv}m Adverse Drift Guard] Live price ({live_current_price:.2f}) drifted {adverse_pts:.2f} below entry ({entry_price:.2f}) > max allowed {max_adverse_drift:.2f} (0.25 ATR). Skipping.")
                                                            status_msg = "Skipped (Adverse Price Drift)"
                                                            wallet_exceeded = True
                                                            bybit_success = False
                                                        elif str(ml_trend).upper() in ("SELL", "SHORT", "BEARISH") and (live_current_price - entry_price) > max_adverse_drift:
                                                            adverse_pts = live_current_price - entry_price
                                                            log_event("WARNING", f"[{symbol} {iv}m Adverse Drift Guard] Live price ({live_current_price:.2f}) drifted {adverse_pts:.2f} above entry ({entry_price:.2f}) > max allowed {max_adverse_drift:.2f} (0.25 ATR). Skipping.")
                                                            status_msg = "Skipped (Adverse Price Drift)"
                                                            wallet_exceeded = True
                                                            bybit_success = False

                                                    # Fetch real top-of-book depth from orderbook imbalance
                                                    top_book_depth = 50000.0
                                                    try:
                                                        obi_res = bybit_get_orderbook_imbalance(symbol, depth=10)
                                                        if obi_res and obi_res.get("status") == "OK":
                                                            mid_ref = float(obi_res.get("mid_price", live_current_price) or live_current_price)
                                                            top_depth_calc = float((obi_res.get("bid_vol", 0.0) + obi_res.get("ask_vol", 0.0)) * mid_ref)
                                                            if top_depth_calc > 0:
                                                                top_book_depth = top_depth_calc
                                                    except Exception as ex_ob:
                                                        log_event("WARNING", f"Failed to fetch orderbook depth for {symbol}: {ex_ob}")

                                                    if not wallet_exceeded:
                                                        ev_valid, ev_msg = ExecutionValidator(max_portfolio_heat=getattr(config, "MAX_PORTFOLIO_HEAT", 0.35)).validate_order(
                                                            symbol=symbol,
                                                            direction=ml_trend,
                                                            entry_price=entry_price,
                                                            stop_loss_price=stop_loss_price,
                                                            take_profit_price=take_profit_price,
                                                            position_size_usd=position_size_usd,
                                                            live_price=live_current_price,
                                                            top_book_depth_usd=top_book_depth,
                                                            portfolio_heat=portfolio_heat,
                                                            atr_norm=float(pred_info.get("atr_norm", 0.01)) if isinstance(pred_info, dict) and pred_info.get("atr_norm") is not None else 0.01
                                                        )
                                                        if not ev_valid:
                                                            log_event("WARNING", f"[{symbol} {iv}m ExecutionValidator] REJECTED: {ev_msg}")
                                                            status_msg = f"Skipped ({ev_msg})"
                                                            wallet_exceeded = True
                                                            bybit_success = False
                                                        else:
                                                            raw_qty = float(qty_str) if qty_str else qty_val
                                                            actual_qty = raw_qty
                                                            bybit_success = True
                                                            bybit_order_id = None
                                                            bybit_scale_out_order_id = None
                                                
                                                if TRADE_MODE != "simulation":
                                                    # Live trading execution offloaded to background thread to minimize latency
                                                    just_opened_symbols.add(symbol)
                                                    if bybit_success:
                                                        actual_qty = raw_qty if raw_qty > 0 else (float((position_size_usd * leverage_val) / entry_price) if entry_price > 0 else 0.0)
                                                        actual_notional_val = float(actual_qty * entry_price)
                                                        actual_margin_usd = float(actual_notional_val / leverage_val) if leverage_val > 0 else float(position_size_usd)
                                                        with active_execution_lock:
                                                            active_execution_symbols.add(symbol)
                                                            active_execution_margins[symbol] = float(actual_margin_usd)
                                                            active_execution_notional[symbol] = float(actual_notional_val)
                                                            intended_risk_usd = abs(float(entry_price) - float(stop_loss_price)) * float(actual_qty)
                                                            active_execution_risks[symbol] = float(intended_risk_usd)
                                                            bot_state["in_flight_risk_usd"] = sum(active_execution_risks.values())
                                                        threading.Thread(
                                                            target=execute_bybit_trade_async,
                                                            args=(symbol, iv, tf, ml_trend, leverage_val, qty_str, raw_qty, entry_price, stop_loss_price, take_profit_price, position_size_usd, kelly_fraction, calibrated_confidence, ml_confidence, dynamic_conf_threshold, latest_completed_ts, latest_candle, pred_change, predicted_price, atr_dollars, tp_multiplier_adjusted, sl_multiplier_adjusted, df_completed, trade_uuid, duration_seconds, active_trade_key, is_oversized_trade, original_kelly_size, decision_ts),
                                                            kwargs={"journal_rec": rec},
                                                            daemon=True
                                                        ).start()
                                                        placed = True
                                                        async_spawned = True
                                                        status_msg = "Traded"
                                                        bybit_success = False # Skip the simulation path for this trade

                                            if bybit_success:
                                                actual_notional_val = float(actual_qty * entry_price)
                                                actual_margin_usd = float(actual_notional_val / leverage_val) if leverage_val > 0 else float(position_size_usd)
                                                actual_size_usd = actual_margin_usd
                                                active_trade = {
                                                    "trade_id": f"{symbol}_{trade_uuid}",
                                                    "bybit_order_id": bybit_order_id,
                                                    "bybit_scale_out_order_id": bybit_scale_out_order_id,
                                                    "symbol": symbol,
                                                    "entry_price": float(entry_price),
                                                    "predicted_price": float(predicted_price),
                                                    "stop_loss": float(stop_loss_price),
                                                    "take_profit": float(take_profit_price),
                                                    "initial_stop_loss": float(stop_loss_price),
                                                    "initial_take_profit": float(take_profit_price),
                                                    "initial_rr": float(abs(take_profit_price - entry_price) / max(1e-9, abs(entry_price - stop_loss_price))),
                                                    "kelly_size_usd": float(original_kelly_size) if 'original_kelly_size' in locals() else float(position_size_usd),
                                                    "clamped_size_usd": float(position_size_usd),
                                                    "final_size_usd": float(actual_size_usd),
                                                    "direction": str(ml_trend),
                                                    "end_time": float(time.time() + duration_seconds),
                                                    "entry_time": int(time.time() * 1000),
                                                    "atr_dollars": float(atr_dollars),
                                                    "entry_atr": float(atr_dollars),
                                                    "highest_price": float(entry_price),
                                                    "lowest_price": float(entry_price),
                                                    "break_even_triggered": False,
                                                    "half_closed": False,
                                                    "original_size": float(position_size_usd),
                                                    "position_size_usd": actual_size_usd,
                                                    "scaled_out_pnl": 0.0,
                                                    "kelly_fraction": float(kelly_fraction),
                                                    "leverage": float(leverage_val),
                                                    "confidence": float(calibrated_confidence),
                                                    "qty": float(actual_qty),
                                                    "original_qty": float(actual_qty),
                                                    "fill_pct": round((actual_qty / raw_qty) * 100.0, 2) if raw_qty > 0 else 100.0
                                                }
                                            
                                                # Send Telegram alert for successful prediction/trade entry
                                                entry_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                                send_telegram_alert(
                                                    f"🟢 *POSITION OPENED (SUCCESSFUL SIGNAL)* 🟢\n"
                                                    f"• *Asset*: {symbol}\n"
                                                    f"• *Interval*: {iv}m\n"
                                                    f"• *Direction*: {ml_trend}\n"
                                                    f"• *Entry Price*: ${float(entry_price):.4f}\n"
                                                    f"• *Entry Time*: {entry_time_str}\n"
                                                    f"• *Take Profit*: ${float(take_profit_price):.4f}\n"
                                                    f"• *Stop Loss*: ${float(stop_loss_price):.4f}\n"
                                                    f"• *Calibrated Confidence*: {calibrated_confidence * 100:.2f}%\n"
                                                    f"• *Leverage*: {leverage_val:.1f}x\n"
                                                    f"• *Position Size (Margin)*: ${actual_margin_usd:.2f} (Value: ${actual_notional_val:.2f})\n"
                                                    f"• *Execution Mode*: {TRADE_MODE.upper()}"
                                                )
                                            
                                                with active_trades_lock:
                                                    current_trades = bot_state.get(active_trade_key, [])
                                                    if not isinstance(current_trades, list):
                                                        current_trades = []
                                                    current_trades = list(current_trades)
                                                    current_trades.append(active_trade)
                                                    bot_state[active_trade_key] = current_trades

                                                # Finding #49: In simulation mode, confirm trade ID and execution outcome upon storing active trade
                                                rec.trade_id = f"{symbol}_{trade_uuid}"
                                                rec.outcome = "EXECUTED"
                                            
                                                # Mark symbol as opened this cycle — prevents duplicate opens due to Bybit sync latency
                                                just_opened_symbols.add(symbol)
                                                # Sync positions immediately to load live Bybit state parameters
                                                if TRADE_MODE != "simulation":
                                                    sync_active_positions_from_bybit()
                                            
                                                # Deduct size from wallet balance immediately (only in simulation)
                                                if TRADE_MODE == "simulation":
                                                    bot_state["simulated_balance"] = round(bot_state["simulated_balance"] - position_size_usd, 2)
                                                placed = True
                                                status_msg = "Traded"
                                            
                                                print(f"[{symbol} {iv}m] Trade Opened: {ml_trend} at price {entry_price:.2f} (SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}, Slippage: {slippage_pct:.3f}%)")
                                    else:
                                        status_msg = "Skipped (Confluence Failed)"
                                        failed_list = [name.replace('_', ' ') for name, res_val in confluence_results.items() if not res_val["pass"] and name != '_Score_Summary']
                                        print("--------------------------------------------------")
                                        print(f"CONFLUENCE RESULT: REJECTED ({confluence_results.get('_Score_Summary', {}).get('detail', 'Score too low')})")
                                        print(f"Failed checks: {', '.join(failed_list)}")
                                        print("==================================================\n")
                        


                            # Update existing prediction from SignalEvaluator or append new entry with final live gate status
                            matched_pred = None
                            c_ts_target = int(latest_completed_ts * 1000) if latest_completed_ts < 1e11 else int(latest_completed_ts)
                            window_ms = max(int(iv) * 60 * 1000 * 3, 1800000)
                            for p in reversed(bot_state.get("prediction_history", [])):
                                if str(p.get("symbol")) == str(symbol) and str(p.get("interval")) == str(iv):
                                    p_c_ts = p.get("candle_timestamp") or p.get("timestamp") or 0
                                    p_c_ts_conv = int(p_c_ts * 1000) if p_c_ts < 1e11 else int(p_c_ts)
                                    if abs(p_c_ts_conv - c_ts_target) <= window_ms:
                                        if p.get("status") == "Pending Risk Evaluation" or matched_pred is None:
                                            matched_pred = p
                                            if p.get("status") == "Pending Risk Evaluation":
                                                break
                                        break

                            if matched_pred is None:
                                for p in reversed(bot_state.get("prediction_history", [])):
                                    if str(p.get("symbol")) == str(symbol) and str(p.get("interval")) == str(iv):
                                        if abs(float(time.time()) - float(p.get("timestamp", 0))) <= 1800:
                                            matched_pred = p
                                            break

                            if matched_pred is not None:
                                matched_pred["timestamp"] = float(time.time())
                                matched_pred["status"] = str(status_msg)
                                matched_pred["direction"] = str(ml_trend)
                                matched_pred["ref_price"] = float(latest_candle["close"])
                                matched_pred["predicted_change"] = float(pred_change)
                                matched_pred["predicted_price"] = float(predicted_price)
                                matched_pred["calibrated_confidence"] = float(calibrated_confidence)
                                matched_pred["raw_confidence"] = float(ml_confidence)
                                matched_pred["dynamic_threshold"] = float(dynamic_conf_threshold)
                                matched_pred["threshold_base"] = float(economic_base_threshold) if 'economic_base_threshold' in locals() else None
                                matched_pred["threshold_adjustments"] = adjustments_applied if 'adjustments_applied' in locals() else []
                                try:
                                    database.save_prediction(matched_pred)
                                except Exception as ex_db1:
                                    log_event("WARNING", f"[Prediction DB] Failed saving matched prediction: {ex_db1}")
                            else:
                                bot_state["prediction_history"].append({
                                    "prediction_id": f"{symbol}_{iv}_{int(c_ts_target)}",
                                    "symbol": symbol,
                                    "timestamp": float(time.time()),
                                    "candle_timestamp": c_ts_target,
                                    "interval": str(iv),
                                    "direction": str(ml_trend),
                                    "ref_price": float(latest_candle["close"]),
                                    "predicted_change": float(pred_change),
                                    "predicted_price": float(predicted_price),
                                    "status": str(status_msg),
                                    "calibrated_confidence": float(calibrated_confidence),
                                    "raw_confidence": float(ml_confidence),
                                    "dynamic_threshold": float(dynamic_conf_threshold),
                                    "threshold_base": float(economic_base_threshold) if 'economic_base_threshold' in locals() else None,
                                    "threshold_adjustments": adjustments_applied if 'adjustments_applied' in locals() else [],
                                    "evaluation": {
                                        "evaluated": False,
                                        "exit_price": None,
                                        "change": None,
                                        "change_pct": None,
                                        "success": None
                                    }
                                })
                                try:
                                    database.save_prediction(bot_state["prediction_history"][-1])
                                except Exception as ex_db2:
                                    log_event("WARNING", f"[Prediction DB] Failed saving new prediction: {ex_db2}")
                            
                                max_pred_cap = getattr(config, "MAX_PREDICTION_HISTORY_MEMORY", 500)
                                if len(bot_state["prediction_history"]) > max_pred_cap:
                                    bot_state["prediction_history"] = bot_state["prediction_history"][-max_pred_cap:]
                        
                            evaluate_predictions(df_completed, iv, symbol)
                            save_history()
                        
                    finally:
                        rec.status_msg = status_msg
                        if not getattr(rec, "reject_reason", None) and status_msg not in ("Pending", "Traded", ""):
                            rec.reject_reason = status_msg
                        if not getattr(rec, "reason_code", None) and status_msg not in ("Pending", "Traded", ""):
                            rec.reason_code = map_status_to_reason_code(status_msg)
                        
                        # Populate and snapshot remaining economic and sizing metrics if available
                        if 'exp_edge_bps' in locals() and exp_edge_bps is not None and rec.expected_value is None:
                            rec.expected_value = float(exp_edge_bps)
                        if 'exp_r_val' in locals() and exp_r_val is not None and rec.expected_rr is None:
                            rec.expected_rr = float(exp_r_val)
                        if 'cost_bps' in locals() and cost_bps is not None and rec.round_trip_cost_bp is None:
                            rec.round_trip_cost_bp = float(cost_bps)
                        if 'mhi_val' in locals() and mhi_val is not None and rec.mhi_score is None:
                            rec.mhi_score = float(mhi_val)
                        # Finding #29: Intended size and leverage recorded even on pre-sizing rejections
                        if rec.position_size_usd is None:
                            if 'position_size_usd' in locals() and position_size_usd is not None:
                                rec.position_size_usd = float(position_size_usd)
                            else:
                                rec.position_size_usd = float(round(current_bal * getattr(config, "RISK_PER_TRADE_PCT", 0.01), 2))
                        if rec.leverage is None:
                            if 'leverage_val' in locals() and leverage_val is not None:
                                rec.leverage = float(leverage_val)
                            else:
                                rec.leverage = 1.0
                        rec.snapshot(
                            status_msg=status_msg,
                            expected_value=rec.expected_value,
                            expected_rr=rec.expected_rr,
                            round_trip_cost_bp=rec.round_trip_cost_bp,
                            position_size_usd=rec.position_size_usd,
                            leverage=rec.leverage,
                            mhi_score=rec.mhi_score
                        )

                        if not async_spawned:
                            if placed:
                                rec.outcome = "EXECUTED"
                            elif status_msg.startswith("REJECTED"):
                                rec.outcome = "REJECTED"
                            elif status_msg.startswith("Skipped") or status_msg in ("Abstain", "Pending"):
                                rec.outcome = "SKIPPED"
                            elif rec.outcome == "APPROVED":
                                rec.outcome = "EXECUTED" if placed else "SKIPPED"
                            elif rec.outcome == "ERROR":
                                pass
                            else:
                                rec.outcome = "SKIPPED"
                            write_decision(rec)
                            log_event("INFO", f"[{symbol} {iv}m] Journalling decision: {status_msg}")
                        else:
                            log_event("INFO", f"[{symbol} {iv}m] Async execution spawned; decision journaling delegated to async executor.")
                        last_processed_timestamps[last_ts_key] = latest_completed_ts
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                traceback.print_exc()
                print(f"Error checking {iv}m candle close signals: {e}")
                send_telegram_alert(
                    f"⚠️ *ERROR IN CANDLE CHECK SIGNAL CHECK* ⚠️\n"
                    f"• *Asset*: {symbol}\n"
                    f"• *Interval*: {iv}m\n"
                    f"• *Error*: {str(e)}\n"
                    f"• *Detail*: Failed during signal processing cycle."
                )

        # Clean temporary DataFrame caches and trim memory after candle checks
        if check_queue:
            fetched_data.clear()
            htf_cache.clear()
            max_pred_cap = getattr(config, "MAX_PREDICTION_HISTORY_MEMORY", 500)
            if len(bot_state.get("prediction_history", [])) > max_pred_cap:
                bot_state["prediction_history"] = bot_state["prediction_history"][-max_pred_cap:]
            try:
                import gc, ctypes
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
            except Exception:
                pass

        # Model Drift Check (every 4 hours)
        current_hour_utc = datetime.now(timezone.utc).hour
        if current_hour_utc % 4 == 0 and datetime.now(timezone.utc).minute == 0:
            drift_res = mlops_engine.check_model_drift("15", bot_state.get("trade_history", []), window=100)
            if drift_res.get("status") == "ALERT":
                alert_text = "\n".join(drift_res.get("alerts", []))
                send_telegram_alert(
                    f"🚨 *MODEL DRIFT ALERT DETECTED* 🚨\n"
                    f"• *Interval*: 15m\n"
                    f"• *Accuracy*: {drift_res.get('accuracy', 0)*100:.1f}%\n"
                    f"• *High-Conf Win Rate*: {drift_res.get('high_conf_wr', 0)*100:.1f}%\n"
                    f"• *Alerts*:\n{alert_text}"
                )

        try:
            import gc, ctypes
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        except Exception:
            pass

        time.sleep(30)  # Candles close every 15m minimum — no need to check faster than 30s

def run_order_flow_persister():
    print("[Order Flow] Persister background thread started.")
    import sqlite3
    from data import DB_PATH, db_write_lock
    while True:
        time.sleep(60) # Dump statistics every 60 seconds
        try:
            now_ts = time.time()
            # Align to minute mark
            minute_ts = float(int(now_ts / 60) * 60)
            
            with order_flow_lock:
                # Extract and clear/reset current CVD & OFI buffers
                to_write = []
                for sym, state in list(order_flow_data.items()):
                    minute_delta = state.get("cvd", 0.0)
                    state["cumulative_cvd"] = state.get("cumulative_cvd", 0.0) + minute_delta
                    cvd_val = state["cumulative_cvd"]
                    ofi_val = state.get("ofi", 0.0) or minute_delta
                    ob_imb = state.get("ob_imbalance_L2", 0.0)
                    ob_spr = state.get("ob_spread_L2", 0.0)
                    liq_l = state.get("liq_long_1h", 0.0)
                    liq_s = state.get("liq_short_1h", 0.0)
                    
                    to_write.append((sym, minute_ts, cvd_val, ofi_val, ob_imb, ob_spr, liq_l, liq_s))
                    # Reset accumulators for the next minute
                    state["cvd"] = 0.0
                    state["ofi"] = 0.0
                    state["liq_long_1h"] = 0.0
                    state["liq_short_1h"] = 0.0
            
            if to_write:
                with db_write_lock:
                    conn = sqlite3.connect(DB_PATH, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS historical_order_flow (
                            symbol TEXT,
                            timestamp REAL,
                            cvd REAL,
                            ofi REAL,
                            ob_imbalance_L2 REAL,
                            ob_spread_L2 REAL,
                            liq_long_1h REAL,
                            liq_short_1h REAL,
                            PRIMARY KEY (symbol, timestamp)
                        )
                    """)
                    conn.executemany(
                        "INSERT OR REPLACE INTO historical_order_flow (symbol, timestamp, cvd, ofi, ob_imbalance_L2, ob_spread_L2, liq_long_1h, liq_short_1h) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        to_write
                    )
                    conn.commit()
                    conn.close()
        except Exception as e:
            print(f"[Order Flow Persister Error] {e}")

def safe_main():
    while True:
        try:
            main()
        except Exception as e:
            import traceback
            try:
                traceback_str = traceback.format_exc()
            except Exception:
                traceback_str = str(e)
            try:
                traceback.print_exc()
            except Exception:
                pass
            try:
                print(f"CRITICAL ERROR in main loop: {e}")
            except Exception:
                pass
            try:
                err_clean = str(e).replace("`", "'")
                send_telegram_alert(
                    f"🔴 *CRITICAL RUNTIME ERROR* 🔴\n"
                    f"• *Error*: `{err_clean}`\n"
                    f"• *Action*: Restarting main bot loop...\n\n"
                    f"```\n{traceback_str[:300]}...\n```"
                )
            except Exception:
                pass
            time.sleep(15)  # wait before restarting main

if __name__ == "__main__":
    import threading
    print("[SYSTEM] Entered main execution block.", flush=True)
    print("[SYSTEM] Launching Flask dashboard thread on port 5001...", flush=True)
    t_flask = threading.Thread(target=run_flask, name="flask-dashboard", daemon=True)
    t_flask.start()
    print("[SYSTEM] Flask dashboard thread launched.", flush=True)
    
    # Start main bot loop in background thread
    print("[SYSTEM] Launching safe_main bot loop thread...", flush=True)
    threading.Thread(target=safe_main, name="main-bot-loop", daemon=True).start()
    # Start background Telegram command listener thread
    threading.Thread(target=start_telegram_command_listener, args=(bot_state, bot_state_lock, active_trades_lock), name="telegram-listener", daemon=True).start()
    
    try:
        send_telegram_alert(f"🤖 *BTC Trading Bot Started successfully on {TRADE_MODE.upper()} mode.*")
    except Exception as ex_tg:
        print(f"[Telegram Notice] Could not send startup alert: {ex_tg}")

    # Start background news sentiment updater thread
    threading.Thread(target=run_news_sentiment_updater, name="news-sentiment", daemon=True).start()
    # Start background Bybit balance updater thread
    threading.Thread(target=run_bybit_balance_updater, name="bybit-balance", daemon=True).start()
    # Start Bybit WebSocket feed in a background thread
    threading.Thread(target=start_ws, name="ws-public-feed", daemon=True).start()
    # Start Bybit Private WebSocket feed in a background thread
    threading.Thread(target=start_private_ws, name="ws-private-feed", daemon=True).start()
    # Start WebSocket keep-alive watchdog thread in a background thread
    threading.Thread(target=run_websocket_watchdog, name="ws-watchdog", daemon=True).start()
    # Start Bybit REST API fallback price updater thread
    threading.Thread(target=run_fallback_price_updater, name="price-fallback", daemon=True).start()
    # Start background order flow persister thread
    threading.Thread(target=run_order_flow_persister, name="order-flow-persister", daemon=True).start()
    # Start daily database and trade journal backup thread
    threading.Thread(target=run_daily_backup_scheduler, name="daily-backup-scheduler", daemon=True).start()
    # Start daily 00:00 UTC performance summary report thread
    threading.Thread(target=run_daily_summary_scheduler, name="daily-summary-scheduler", daemon=True).start()
    # Start statistical governance scheduler background thread
    try:
        from background_schedulers import run_statistical_governance_scheduler
        threading.Thread(target=run_statistical_governance_scheduler, name="stat-governance-scheduler", daemon=True).start()
    except Exception as ex_sgs:
        log_event("WARNING", f"[StatGovernance Launch Warning] {ex_sgs}")
    # Start pain feedback verifier background thread
    threading.Thread(target=run_pain_feedback_verifier, name="pain-feedback-verifier", daemon=True).start()
    # Start signal evaluator background thread
    try:
        from signal_evaluator import run_signal_evaluator_loop
        threading.Thread(target=run_signal_evaluator_loop, args=(bot_state,), name="signal-evaluator", daemon=True).start()
    except Exception as ex_se:
        log_event("WARNING", f"[SignalEvaluator Launch Warning] {ex_se}")

    # Check champion model age and warn if stale (> 14 days) (Finding #40)
    try:
        check_champion_models_staleness(max_age_days=14.0)
    except Exception as ex_stale:
        log_event("WARNING", f"[Model Governance] Model staleness check warning: {ex_stale}")

    # Start automated weekly rolling retraining scheduler thread (Sundays 00:00 UTC) (Finding #40)
    threading.Thread(target=run_rolling_retrain_scheduler, name="rolling-retrain-scheduler", daemon=True).start()

    # Keep main thread alive
    while True:
        time.sleep(1)