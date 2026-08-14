from typing import Optional
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import time
import json
import re
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
from datetime import datetime, timezone
from kelly_tracker import global_kelly_tracker
from volatility_clusterer import volatility_clusterer
from gmm_trail import gmm_trailing_engine
from garch_monitor import garch_vol_monitor
from news_monitor import news_monitor
from decay_calibrator import decay_calibrator
import database
import trade_calculators
import exit_manager
from trade_calculators import transaction_cost_model, calculate_break_even_stop, UnifiedTargetGenerator, calculate_probabilistic_utility_bootstrap
from statistical_validation import statistical_validation
from decision_outcome_db import decision_outcome_db
from meta_learning_engine import meta_learning_engine
from causal_attribution_engine import causal_attribution_engine
from counterfactual_replay_engine import counterfactual_replay_engine
from probabilistic_policy_selector import probabilistic_policy_selector
from hierarchical_bayesian_engine import hierarchical_bayesian_engine
from drift_attribution_engine import drift_attribution_engine
from automatic_research_reporter import automatic_research_reporter
from exit_policy_engine import exit_policy_engine, PortfolioUtilityOptimizer, generate_continuous_policy_vector, log_checksummed_exit_decision
from order_state_machine import StopState, StopStateMachine
from secret_manager import get_secure_env

from bybit_client import (
    bybit_get_request,
    bybit_post_request,
    get_real_bybit_balance_cached,
    get_all_bybit_positions,
    place_bybit_order,
    format_bybit_price,
    format_bybit_qty,
    get_bybit_time_offset,
    execute_bybit_order_ws_or_rest
)
from confluence_engine import check_pre_trade_confluence
from telegram_bot import send_telegram_alert, execute_telegram_api_call
from telegram_listener import run_manual_confluence_report, start_telegram_command_listener
from websocket_client import init_bybit_websocket_listeners, get_ws_status
from dashboard_routes import dashboard_bp
from risk_limits import assert_risk_governance_invariants
from config_verifier import assert_shared_constants_aligned
from decision_journal import DecisionRecord, write_decision

# F-09 Governance Startup Lock: Assert hard safety bounds before trading initialization
assert_risk_governance_invariants()
assert_shared_constants_aligned()

ACTIVE_TRADE_TF_KEYS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h"]


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
    "5m": 1.8,
    "15": 2.0, "15m": 2.0,
    "30": 2.5, "30m": 2.5,
    "60": 3.0, "1h": 3.0,
    "120": 4.0, "2h": 4.0,
    "240": 4.0, "4h": 4.0,
    "360": 4.0, "6h": 4.0
}

def compute_be_trigger_distance(atr_dollars, leverage, interval, mfe_trigger_atr_multiple, entry_price=0.0, min_pct_floor=0.0):
    """
    FIX 2: Compute minimum favorable move before break-even activates.
    Enforces a minimum 1.0x ATR distance for leverage > 10x to prevent premature BE chop-outs.
    """
    base_be_dist = max(mfe_trigger_atr_multiple * atr_dollars, entry_price * min_pct_floor)
    if leverage > 10.0:
        min_be_dist = 1.0 * atr_dollars
        final_be_dist = max(base_be_dist, min_be_dist)
        return final_be_dist
    return base_be_dist

def validate_trade_structure(entry_price, stop_price, tp_price, atr_dollars, leverage, interval, symbol, direction):
    """
    UNIVERSAL TRADE STRUCTURE SANITIZER: Pre-flight gate before order placement.
    Validates & adjusts R:R ratio, minimum stop width, and leverage compatibility.
    Returns: (is_valid, adjusted_dict, log_reason_str)
    """
    stop_dist = abs(entry_price - stop_price)
    tp_dist = abs(tp_price - entry_price)
    
    adjusted = {
        "stop_price": stop_price,
        "tp_price": tp_price,
        "leverage": leverage,
        "stop_dist": stop_dist,
        "tp_dist": tp_dist
    }
    logs = []
    
    # 1. Enforce minimum stop width for ALL trades
    min_stop = atr_dollars * 1.0 if leverage > 10.0 else atr_dollars * 0.75
    if stop_dist < min_stop:
        if leverage > 10.0:
            adjusted["leverage"] = 10.0
            logs.append(f"[LEVERAGE_CAPPED] {symbol} {interval} leverage reduced from {leverage:.1f}x to 10.0x & SL widened from ${stop_dist:.4f} to 1.0x ATR (${min_stop:.4f})")
        else:
            logs.append(f"[STOP_WIDENED] {symbol} {interval} SL widened from ${stop_dist:.4f} to 0.75x ATR (${min_stop:.4f})")
            
        required_stop = min_stop
        if direction == "Bearish":
            adjusted["stop_price"] = entry_price + required_stop
        else:
            adjusted["stop_price"] = entry_price - required_stop
        adjusted["stop_dist"] = required_stop

    # 2. Universal R:R Ratio Capping by timeframe (Max Cap)
    iv_str = str(interval).replace("m", "")
    max_rr = MAX_RR_RATIO.get(str(interval), MAX_RR_RATIO.get(iv_str, 4.0))
    current_rr = adjusted["tp_dist"] / adjusted["stop_dist"] if adjusted["stop_dist"] > 0 else 0.0
    
    if current_rr > max_rr:
        max_allowed_tp_dist = adjusted["stop_dist"] * max_rr
        if direction == "Bearish":
            adjusted["tp_price"] = entry_price - max_allowed_tp_dist
        else:
            adjusted["tp_price"] = entry_price + max_allowed_tp_dist
        adjusted["tp_dist"] = max_allowed_tp_dist
        current_rr = max_rr
        logs.append(f"[TP_CAPPED_UNIVERSAL] {symbol} {interval} R:R capped from {tp_dist/adjusted['stop_dist']:.1f}:1 to {max_rr:.1f}:1 (TP dist reduced from ${tp_dist:.4f} to ${max_allowed_tp_dist:.4f})")
        
    # 3. Minimum R:R Ratio Floor Gate (Dynamic TP optimization if below min_rr)
    min_rr = MIN_RR_RATIO.get(str(interval), MIN_RR_RATIO.get(iv_str, 2.0))
    if current_rr < min_rr:
        try:
            from trade_frequency_optimizer import trade_frequency_optimizer
            opt_tp, new_rr, adjusted_flag = trade_frequency_optimizer.optimize_tp_target_for_rr(
                entry_price=entry_price, stop_price=adjusted["stop_price"], atr_dollars=atr_dollars, direction=direction, min_rr_required=min_rr
            )
            if adjusted_flag:
                adjusted["tp_price"] = opt_tp
                adjusted["tp_dist"] = abs(opt_tp - entry_price)
                current_rr = min_rr
                logs.append(f"[TP_OPTIMIZED_RR] {symbol} {interval} TP target adjusted to ${opt_tp:.4f} to satisfy {min_rr:.1f}:1 R:R floor")
        except Exception as e:
            pass

    if current_rr < min_rr:
        logs.append(f"[REJECT_MIN_RR] {symbol} {interval} R:R {current_rr:.1f}:1 is below minimum floor {min_rr:.1f}:1")
        return False, adjusted, "; ".join(logs)
        
    return True, adjusted, "; ".join(logs) if logs else "OK"

class AdaptiveVolumeGate:
    def __init__(self, lookback_days=30, optimization_window=500):
        self.lookback_days = lookback_days
        self.optimization_window = optimization_window
        self.threshold_cache = {}
        self.last_optimized = {}

    def get_volume_percentile(self, symbol, kline_df=None):
        try:
            if kline_df is not None and "volume" in kline_df.columns and len(kline_df) >= 10:
                volumes = kline_df["volume"].values
                current_vol = volumes[-1]
                percentile = float(np.mean(volumes <= current_vol))
                return percentile
        except Exception:
            pass
        return 1.0

    def optimize_threshold(self, symbol):
        try:
            trades = database.get_trade_history(limit=self.optimization_window)
            sym_trades = [t for t in trades if isinstance(t, dict) and t.get("symbol") == symbol]
            if len(sym_trades) < 20:
                return 0.25
            
            def _safe_vol(t):
                try:
                    if t.get("raw_data"):
                        return float(json.loads(t["raw_data"]).get("vol_pctile", 1.0))
                except Exception:
                    pass
                return 1.0

            best_threshold = 0.25
            best_profit = -float('inf')
            for threshold in np.arange(0.10, 0.51, 0.05):
                allowed = [t for t in sym_trades if _safe_vol(t) >= threshold]
                if len(allowed) < 5:
                    continue
                pnl_sum = sum(float(t.get("pnl_usd", 0.0)) for t in allowed)
                if pnl_sum > best_profit:
                    best_profit = pnl_sum
                    best_threshold = threshold
            self.threshold_cache[symbol] = float(best_threshold)
            self.last_optimized[symbol] = time.time()
            return float(best_threshold)
        except Exception:
            return 0.25

    def check(self, symbol, kline_df=None):
        current_pct = self.get_volume_percentile(symbol, kline_df=kline_df)
        last_opt = self.last_optimized.get(symbol, 0)
        if time.time() - last_opt > 86400 * 7 or symbol not in self.threshold_cache:
            threshold = self.optimize_threshold(symbol)
        else:
            threshold = self.threshold_cache[symbol]
            
        if current_pct < threshold:
            return False, f"VOLUME_GATE_BLOCKED: {symbol} 4H volume at {current_pct:.1%} (Threshold: {threshold:.1%})", current_pct
        return True, f"VOLUME_GATE_PASSED: {symbol} 4H volume at {current_pct:.1%}", current_pct

class MFEBreakEvenTrigger:
    def __init__(self, lookback_trades=150, min_sample_size=15):
        self.lookback_trades = lookback_trades
        self.min_sample_size = min_sample_size
        self.trigger_cache = {}

    def get_trigger_multiple(self, symbol, timeframe="60"):
        key = (symbol, str(timeframe))
        if key in self.trigger_cache:
            return self.trigger_cache[key]
            
        try:
            trades = database.get_trade_history(limit=self.lookback_trades)
            sym_winning_trades = [
                t for t in trades
                if isinstance(t, dict) and t.get("symbol") == symbol and float(t.get("pnl_usd", 0.0)) > 0
            ]
            mfe_ratios = []
            for t in sym_winning_trades:
                atr = float(t.get("atr_dollars", 0.0))
                if atr > 0:
                    raw = {}
                    if t.get("raw_data"):
                        try:
                            raw = json.loads(t["raw_data"])
                        except Exception:
                            raw = {}
                    mfe_val = float(raw.get("mfe", 0.0))
                    if mfe_val > 0:
                        mfe_ratios.append(mfe_val / atr)
            if len(mfe_ratios) >= self.min_sample_size:
                trig = float(np.percentile(mfe_ratios, 25))
                trig = float(np.clip(trig, 0.8, 2.0))
                self.trigger_cache[key] = trig
                return trig
        except Exception:
            pass
        return 0.85 if str(timeframe) not in ["15", "30"] else 0.65

adaptive_volume_gate = AdaptiveVolumeGate()
mfe_be_trigger = MFEBreakEvenTrigger()

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
                # Strip ANSI escape codes to keep logs clean in Telegram
                clean_msg = re.sub(r'\x1b\[[0-9;]*[mK]', '', str_msg.strip())
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.logs_list.append(f"[{timestamp}] {clean_msg}")
                if len(self.logs_list) > self.capacity:
                    self.logs_list.pop(0)
        except Exception:
            pass
                
    def flush(self):
        try:
            self.original_stdout.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.original_stdout, name)

log_buffer = CircularLogBuffer(capacity=100)
# sys.stdout = log_buffer

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
CANDLE_CHECK_WINDOW_MINS = int(os.environ.get("CANDLE_CHECK_WINDOW_MINS", "15"))
CANDLE_CHECK_INTERVAL_SECS = int(os.environ.get("CANDLE_CHECK_INTERVAL_SECS", "20"))
BALANCE_UPDATE_INTERVAL_SECS = int(os.environ.get("BALANCE_UPDATE_INTERVAL_SECS", "120"))
POSITION_SYNC_INTERVAL_SECS = float(os.environ.get("POSITION_SYNC_INTERVAL_SECS", "30.0"))
POSITION_SYNC_IDLE_INTERVAL_SECS = float(os.environ.get("POSITION_SYNC_IDLE_INTERVAL_SECS", "120.0"))

import config
from config import TIMEFRAME_CONFIG
import trading_engine
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
            connector = aiohttp.TCPConnector(ssl=False, limit=100, keepalive_timeout=30)
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


def choppiness_index(df, window=None):
    """0-100 scale. >61.8 = choppy, <38.2 = trending"""
    if window is None:
        import config
        window = getattr(config, "DEFAULT_INDICATOR_WINDOW", 14)
    if df is None or len(df) < window:
        return 50.0
    high_max = df['high'].rolling(window).max()
    low_min = df['low'].rolling(window).min()
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    atr_sum = tr.rolling(window).sum()
    price_range = high_max - low_min
    ci = 100 * np.log10(atr_sum / (price_range + 1e-8)) / np.log10(window)
    return float(ci.iloc[-1]) if not np.isnan(ci.iloc[-1]) else 50.0

def is_news_blackout(now_utc, interval):
    """15M/30M avoid trading around major scheduled economic news (e.g., FOMC, CPI, NFP)"""
    if str(interval) not in ["15", "30"]:
        return False
    minute = now_utc.minute
    hour = now_utc.hour
    if hour in [13, 14, 18, 19]:
        if 45 <= minute or minute <= 15:
            return True
    return False

def check_flash_crash(symbol: str, max_drop_pct: float = 3.0, window_minutes: int = 5) -> bool:
    """Block 15M/30M entries if price dropped >3% in last 5 minutes"""
    try:
        df_1m = get_history(symbol=symbol, interval="1", limit=window_minutes + 2)
        if df_1m is None or len(df_1m) < window_minutes:
            return False
        recent_high = df_1m["high"].iloc[-window_minutes:].max()
        current_low = df_1m["low"].iloc[-1]
        drop_pct = ((recent_high - current_low) / (recent_high + 1e-8)) * 100.0
        return drop_pct > max_drop_pct
    except Exception:
        return False

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
        
        target_chat_id = chat_id if chat_id else os.environ.get("TELEGRAM_CHAT_ID", "8957269359")
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
            df = get_history(symbol=symbol, interval=str(interval), limit=100)
            if df is not None and len(df) >= 21:
                df_completed = df.iloc[:-1].copy()
                ema9 = float(EMAIndicator(df_completed["close"], window=9).ema_indicator().iloc[-1])
                ema21 = float(EMAIndicator(df_completed["close"], window=21).ema_indicator().iloc[-1])
                with self._lock:
                    self._cache[key] = {"ema9": ema9, "ema21": ema21, "timestamp": now}
                return ema9, ema21
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
        
def estimate_liquidation_pool(df_history, direction, entry_price):
    """
    Estimates the location of the nearest high-leverage liquidation pool
    based on historical swing highs/lows (support and resistance levels).
    """
    import numpy as np
    
    # Look at the last 60 candles to find recent swing high/low
    lookback = min(len(df_history), 60)
    df_recent = df_history.iloc[-lookback:]
    
    if direction == "Bullish":
        # We are Long. Take Profit is above entry.
        # Short sellers entered near the recent swing high. Their liquidations (buy stops)
        # are clustered 1% to 2% above the swing high (representing 100x and 50x leverage liquidations).
        swing_high = float(df_recent["high"].max())
        # Target just inside the 50x/100x liquidation pool (1.2% above swing high)
        liq_pool_target = swing_high * 1.012
        # Ensure it's higher than entry price
        return max(liq_pool_target, entry_price * 1.005)
    elif direction == "Bearish":
        # We are Short. Take Profit is below entry.
        # Long buyers entered near the recent swing low. Their liquidations (sell stops)
        # are clustered 1% to 2% below the swing low.
        swing_low = float(df_recent["low"].min())
        # Target just inside the 50x/100x liquidation pool (1.2% below swing low)
        liq_pool_target = swing_low * 0.988
        # Ensure it's lower than entry price
        return min(liq_pool_target, entry_price * 0.995)
    else:
        return entry_price

# Telegram listener extracted to telegram_listener.py


print("[System Debug] Importing ta...")
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
print("[System Debug] ta imported.")
print("[System Debug] Importing data.py...")
from data import get_history, merge_derivatives_sentiment_features, classify_market_regime
print("[System Debug] Importing Flask...")
from flask import Flask, jsonify, render_template, request, make_response

# ==========================================
# WEB DASHBOARD CONFIGURATION & STATE
# ==========================================
app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.register_blueprint(dashboard_bp)

bot_logs = []
logs_lock = threading.Lock()
retraining_lock = threading.Lock()
active_trades_lock = threading.Lock()

from state_manager import state_manager as bot_state

def trigger_emergency_kill_switch(reason: str = "Manual Trigger"):
    print(f"[EMERGENCY KILL SWITCH] Triggered! Reason: {reason}")
    bot_state["bot_running"] = False
    send_telegram_alert(f"🚨 *EMERGENCY KILL SWITCH ACTIVATED* 🚨\n• Reason: `{reason}`\n• Action: Halting bot & closing open positions at market.")
    try:
        if TRADE_MODE != "simulation":
            bybit_post_request("/v5/order/cancel-all", {"category": "linear", "settleCoin": "USDT"})
            positions = get_all_bybit_positions()
            for p in (positions or []):
                sym = p.get("symbol")
                sz = float(p.get("size", "0"))
                side = p.get("side")
                if sz > 0 and sym:
                    close_side = "Sell" if side == "Buy" else "Buy"
                    bybit_post_request("/v5/order/create", {
                        "category": "linear",
                        "symbol": sym,
                        "side": close_side,
                        "orderType": "Market",
                        "qty": str(sz),
                        "timeInForce": "IOC",
                        "reduceOnly": True
                    })
    except Exception as err:
        print(f"[Kill Switch Error] Failed executing emergency close: {err}")

from functools import wraps
import hmac

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        expected_key = get_secure_env("DASHBOARD_API_KEY", "").strip()
        client_key = request.headers.get("X-API-KEY")
        if not expected_key or not client_key or not hmac.compare_digest(client_key.strip().encode("utf-8"), expected_key.encode("utf-8")):
            return jsonify({"error": "Unauthorized", "message": "Missing or invalid API key."}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_ip_whitelist(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        allowed_ips = get_secure_env("ALLOWED_DASHBOARD_IPS", "").strip()
        if allowed_ips:
            ip_list = [ip.strip() for ip in allowed_ips.split(",") if ip.strip()]
            trusted_proxies = [ip.strip() for ip in get_secure_env("TRUSTED_PROXIES", "").split(",") if ip.strip()]
            if request.remote_addr in trusted_proxies and request.headers.get("X-Forwarded-For"):
                client_ip = request.headers.get("X-Forwarded-For").split(",")[0].strip()
            else:
                client_ip = request.remote_addr
            if client_ip not in ip_list and client_ip not in ["127.0.0.1", "::1"]:
                return jsonify({"error": "Forbidden", "message": f"IP {client_ip} not allowed."}), 403
        return f(*args, **kwargs)
    return decorated_function



@app.route("/killswitch", methods=["POST"])
@require_api_key
def killswitch_endpoint():
    trigger_emergency_kill_switch("HTTP /killswitch Request")
    return jsonify({"status": "KILL_SWITCH_ACTIVATED", "message": "All orders cancelled and bot halted."})


cached_news_sentiment = "Neutral"
cached_news_titles = []
news_sentiment_lock = threading.Lock()

# Thread-safe real-time Order Flow (CVD & OFI)
from config import SUPPORTED_SYMBOLS
order_flow_lock = threading.Lock()
order_flow_data = {s: {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0} for s in SUPPORTED_SYMBOLS}

# Thread-safe active background order execution guard
active_execution_lock = threading.Lock()
active_execution_symbols = set()

economic_calendar_cache = None
last_calendar_fetch = 0.0
economic_calendar_lock = threading.Lock()

# Re-entrant lock for thread-safe access to bot_state and file IO
bot_state_lock = threading.RLock()

HISTORY_FILE = "/data/dashboard_history.json" if os.path.exists("/data") and os.access("/data", os.W_OK) else "dashboard_history.json"

def save_history():
    with bot_state_lock:
        # Deduplicate completed trades
        trades = bot_state.get("trade_history", [])
        if trades:
            try:
                import database
                db_trades = database.get_trade_history(limit=500)
                if db_trades:
                    trades.extend(db_trades)
            except Exception:
                pass
            
            seen_keys = set()
            deduped = []
            for t in sorted(trades, key=lambda x: float(x.get("exit_time") or 0.0)):
                sym = t.get("symbol")
                dir_str = str(t.get("direction", "")).lower()
                entry_p = round(float(t.get("entry_price") or 0.0), 2)
                exit_p = round(float(t.get("exit_price") or 0.0), 2)
                exit_t = int(float(t.get("exit_time") or 0.0) // 60)
                key = (sym, dir_str, entry_p, exit_p, exit_t)
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append(t)
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
        
    sorted_trades = sorted(trades, key=lambda x: float(x.get("exit_time", 0.0)))
    deduped = []
    for t in sorted_trades:
        duplicate = False
        t_exit = float(t.get("exit_time", 0.0))
        t_entry_p = round(float(t.get("entry_price", 0.0)), 4)
        t_exit_p = round(float(t.get("exit_price", 0.0)), 4)
        t_sym = t.get("symbol")
        t_iv = str(t.get("interval"))
        t_dir = t.get("direction")
        
        for existing in deduped:
            if (t_sym == existing.get("symbol") and str(t_iv) == str(existing.get("interval")) and 
                t_dir == existing.get("direction") and 
                abs(t_entry_p - round(float(existing.get("entry_price", 0.0)), 4)) < 1e-4 and 
                abs(t_exit_p - round(float(existing.get("exit_price", 0.0)), 4)) < 1e-4 and 
                abs(t_exit - float(existing.get("exit_time", 0.0))) < 43200):
                duplicate = True
                break
                
        if not duplicate:
            deduped.append(t)
            
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

    # 2. Sync from AWS Server API if running locally
    elif not space_id:
        try:
            server_ip_default = os.environ.get("SERVER_IP", "127.0.0.1")
            aws_host = os.environ.get("TARGET_AWS_SERVER") or os.environ.get("SYNC_SERVER_URL") or server_ip_default

            if not aws_host.startswith("http://") and not aws_host.startswith("https://"):
                aws_host = f"http://{aws_host}"
            if ":" not in aws_host.replace("http://", "").replace("https://", ""):
                sync_port = os.environ.get("PORT", "5001")
                aws_host = f"{aws_host}:{sync_port}"
            sync_url = f"{aws_host.rstrip('/')}/api/status"
            print(f"Syncing: Attempting to pull latest history from AWS Server API ({sync_url})...")
            resp = requests.get(sync_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                remote_trades = data.get("trade_history", [])
                remote_predictions = data.get("prediction_history", [])
                remote_balance = data.get("simulated_balance", 80.0)
                
                # Filter out old 5m intervals
                remote_trades = [t for t in remote_trades if str(t.get("interval", "60")) != "5"]
                remote_predictions = [p for p in remote_predictions if str(p.get("interval", "60")) != "5"]
                
                if len(remote_trades) > 0 or len(remote_predictions) > 0:
                    bot_state["simulated_balance"] = remote_balance
                    bot_state["trade_history"] = remote_trades
                    bot_state["prediction_history"] = remote_predictions
                    # NOTE: Do NOT overwrite active_trade_* keys from the API response.
                    # StateManager already loads active trades from SQLite on startup.
                    # The API response may lag behind (Redis caching) and overwriting
                    # would wipe valid trades recovered from the DB.
                    
                    # Migrate legacy active trades from DB-loaded state
                    for tf_key in ACTIVE_TRADE_TF_KEYS:
                        migrate_active_trades(bot_state[f"active_trade_{tf_key}"])
                        
                    bot_state["bot_running"] = data.get("bot_running", True)
                    bot_state["fresh_reset_v3"] = data.get("fresh_reset_v3", False)
                    print(f"Sync Success: Loaded {len(remote_trades)} trades and {len(remote_predictions)} predictions from AWS Server ({sync_url}).")
                    
                    # Startup Balance Audit
                    active_margin = sum(t.get("position_size_usd", 0.0) for tf_key in ACTIVE_TRADE_TF_KEYS for t in bot_state.get(f"active_trade_{tf_key}", []))
                    print(f"[Startup Sync Balance Audit] Cash Balance: ${bot_state['simulated_balance']:.2f} | Active Position Margin: ${active_margin:.2f} | Total Account Value: ${bot_state['simulated_balance'] + active_margin:.2f}")
                    
                    save_history()
                    return
        except Exception as e:
            print(f"[AWS Sync] Could not fetch state from AWS Server ({sync_url}): {e}")


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
                
                # Load active trades
                bot_state["active_trade_15m"] = data.get("active_trade_15m", [])
                bot_state["active_trade_30m"] = data.get("active_trade_30m", [])
                bot_state["active_trade_1h"] = data.get("active_trade_1h", [])
                bot_state["active_trade_2h"] = data.get("active_trade_2h", [])
                bot_state["active_trade_4h"] = data.get("active_trade_4h", [])
                bot_state["active_trade_6h"] = data.get("active_trade_6h", [])
                
                # Migrate legacy active trades
                for tf_key in ACTIVE_TRADE_TF_KEYS:
                    migrate_active_trades(bot_state[f"active_trade_{tf_key}"])
                    
                bot_state["bot_running"] = data.get("bot_running", True)
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
    bot_state["active_trade_15m"] = []
    bot_state["active_trade_30m"] = []
    bot_state["active_trade_1h"] = []
    bot_state["active_trade_2h"] = []
    bot_state["active_trade_4h"] = []
    bot_state["active_trade_6h"] = []

    # Force auto-reset if it's the first time running this updated version
    if not bot_state.get("fresh_reset_v3", False):
        print("[System Reset] Migrating history to fresh reset v3. Setting balance to 80.0 and clearing all old trades.")
        bot_state["simulated_balance"] = 80.0
        bot_state["daily_drawdown_start_balance"] = 80.0
        bot_state["trade_history"] = []
        bot_state["prediction_history"] = []
        bot_state["active_trade_15m"] = []
        bot_state["active_trade_30m"] = []
        bot_state["active_trade_1h"] = []
        bot_state["active_trade_2h"] = []
        bot_state["active_trade_4h"] = []
        bot_state["active_trade_6h"] = []
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

_cached_time_offset = None
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
                    _cached_time_offset = offset
                    _last_time_sync = time.time()
                return offset
        except Exception as e:
            if attempt == 2:
                print(f"[Bybit API Error] Failed to sync time after 3 attempts: {e}")
            time.sleep(1)
    return 0

def bybit_post_request(endpoint, payload):
    import time
    import hmac
    import hashlib
    
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    
    if not api_key or not api_secret:
        return {"retCode": -1, "retMsg": "API keys missing"}
        
    offset = get_bybit_time_offset()
    timestamp = str(int(time.time() * 1000) + offset)
    recv_window = "5000"
    
    payload_str = json.dumps(payload)
    val_str = timestamp + api_key + recv_window + payload_str
    sign = hmac.new(
        api_secret.encode("utf-8"),
        val_str.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": sign,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    
    url = f"{BYBIT_BASE_URL}{endpoint}"
    
    async def do_post(url, headers, json_data):
        proxy_dict = get_bybit_proxies()
        proxy_url = proxy_dict.get("https") or proxy_dict.get("http") if proxy_dict else None
        timeout = aiohttp.ClientTimeout(total=8)
        async with _aiohttp_session.post(url, headers=headers, json=json_data, proxy=proxy_url, timeout=timeout) as resp:
            status = resp.status
            try:
                data = await resp.json()
            except Exception:
                text = await resp.text()
                data = {"retCode": status, "retMsg": f"HTTP Error: {text}"}
            
            # Adaptive rate limiting dynamic sleep
            try:
                remaining = int(resp.headers.get("X-Bapi-Limit-Remaining", 100))
                if remaining <= 5:
                    print(f"[Rate Limiter Warning] Post remaining limit is low: {remaining}. Sleeping 1s...")
                    await asyncio.sleep(1.0)
            except Exception:
                pass
                
            return status, data

    max_retries = 3
    for attempt in range(max_retries):
        try:
            _ensure_async_loop()
            future = asyncio.run_coroutine_threadsafe(do_post(url, headers, payload), _async_loop)
            status, res = future.result(timeout=10)
            if status == 200:
                return res
            else:
                if status in [402, 403, 429, 502, 503, 504] and attempt < max_retries - 1:
                    time.sleep(1 + attempt * 1.5)
                    continue
                return res if isinstance(res, dict) else {"retCode": status, "retMsg": str(res)}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 1.5)
                continue
            return {"retCode": -1, "retMsg": f"Connection Error: {e}"}

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

_ws_responses = {}
_ws_responses_lock = threading.Lock()

_order_exec_lock = threading.Lock()

def execute_bybit_order_ws_or_rest(endpoint, payload):
    global private_ws_connected, active_private_ws
    import uuid
    # C7: Add unique clientOrderId (orderLinkId) for request deduplication
    if endpoint == "/v5/order/create" and "orderLinkId" not in payload:
        payload["orderLinkId"] = f"cl_{payload.get('symbol', 'generic')}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        
    op_map = {
        "/v5/order/create": "order.create",
        "/v5/order/cancel": "order.cancel"
    }
    op = op_map.get(endpoint)
    
    with _order_exec_lock:
        if op and private_ws_connected and active_private_ws:
            req_id = f"req_{payload.get('symbol', 'generic')}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:4]}"
            ws_payload = {
                "op": op,
                "reqId": req_id,
                "args": [payload]
            }
            try:
                print(f"[WebSocket Private Execution] Sending {op} reqId={req_id}")
                active_private_ws.send(json.dumps(ws_payload))
                
                # Wait for response (timeout 2.0s)
                timeout = 2.0
                start_t = time.time()
                while time.time() - start_t < timeout:
                    with _ws_responses_lock:
                        if len(_ws_responses) > 500:
                            _ws_responses.clear()
                        if req_id in _ws_responses:
                            resp = _ws_responses.pop(req_id)
                            print(f"[WebSocket Private Execution] Received response for reqId={req_id} retCode={resp.get('retCode')}")
                            return resp
                    time.sleep(0.05)  # Reduced from 0.01 to cut spin rate by 5x
                print(f"[WebSocket Private Execution Warning] Timeout waiting for reqId={req_id}. Falling back to REST...")
            except Exception as e:
                print(f"[WebSocket Private Execution Error] {e}. Falling back to REST...")
                
        # C8: Atomic Fallback to standard REST API request
        return bybit_post_request(endpoint, payload)


def place_bybit_order(symbol, side, qty, price=None, sl=None, tp=None, reduce_only=False, order_type="Market", order_link_id=None):
    # C1: Configurable order type & price bound slippage control
    order_type_str = "Limit" if (order_type == "Limit" and price is not None) else "Market"
    link_id = order_link_id or f"bot_{symbol}_{int(time.time()*1000)}"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": order_type_str,
        "qty": str(qty),
        "timeInForce": "GTC" if order_type_str == "Limit" else "IOC",
        "positionIdx": 0,
        "orderLinkId": str(link_id)[:36]
    }
    if price is not None:
        payload["price"] = format_bybit_price(symbol, price)
    if reduce_only:
        payload["reduceOnly"] = True
    if sl:
        payload["stopLoss"] = format_bybit_price(symbol, sl)
    if tp:
        payload["takeProfit"] = format_bybit_price(symbol, tp)
        
    res = execute_bybit_order_ws_or_rest("/v5/order/create", payload)
    return res


def get_bybit_order_details(symbol, order_id):
    params = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id
    }
    # 1. Try active/realtime orders first
    res = bybit_get_request("/v5/order/realtime", params)
    if res.get("retCode") == 0:
        orders = res.get("result", {}).get("list", [])
        if orders:
            return orders[0]
            
    # 2. Fall back to order history
    res = bybit_get_request("/v5/order/history", params)
    if res.get("retCode") == 0:
        orders = res.get("result", {}).get("list", [])
        if orders:
            return orders[0]
    return None

def cancel_bybit_order(symbol, order_id):
    cancel_payload = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id
    }
    return execute_bybit_order_ws_or_rest("/v5/order/cancel", cancel_payload)

def bybit_get_request(endpoint, query_params):
    import time
    import hmac
    import hashlib
    import urllib.parse
    
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    
    if not api_key or not api_secret:
        return {"retCode": -1, "retMsg": "API keys missing"}
        
    offset = get_bybit_time_offset()
    timestamp = str(int(time.time() * 1000) + offset)
    recv_window = "5000"
    
    query_string = urllib.parse.urlencode(query_params)
    
    val_str = timestamp + api_key + recv_window + query_string
    sign = hmac.new(
        api_secret.encode("utf-8"),
        val_str.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": sign,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    
    url = f"{BYBIT_BASE_URL}{endpoint}?{query_string}"
    
    async def do_get(url, headers):
        proxy_dict = get_bybit_proxies()
        proxy_url = proxy_dict.get("https") or proxy_dict.get("http") if proxy_dict else None
        timeout = aiohttp.ClientTimeout(total=8)
        async with _aiohttp_session.get(url, headers=headers, proxy=proxy_url, timeout=timeout) as resp:
            status = resp.status
            try:
                data = await resp.json()
            except Exception:
                text = await resp.text()
                data = {"retCode": status, "retMsg": f"HTTP Error: {text}"}
            
            # Adaptive rate limiting dynamic sleep
            try:
                remaining = int(resp.headers.get("X-Bapi-Limit-Remaining", 100))
                if remaining <= 5:
                    print(f"[Rate Limiter Warning] Get remaining limit is low: {remaining}. Sleeping 1s...")
                    await asyncio.sleep(1.0)
            except Exception:
                pass
                
            return status, data

    max_retries = 3
    for attempt in range(max_retries):
        try:
            _ensure_async_loop()
            future = asyncio.run_coroutine_threadsafe(do_get(url, headers), _async_loop)
            status, res = future.result(timeout=10)
            if status == 200:
                return res
            else:
                if status in [402, 403, 429, 502, 503, 504] and attempt < max_retries - 1:
                    time.sleep(1 + attempt * 1.5)
                    continue
                return res if isinstance(res, dict) else {"retCode": status, "retMsg": str(res)}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 1.5)
                continue
            return {"retCode": -1, "retMsg": f"Connection Error: {e}"}

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

def get_bybit_accumulated_closed_pnl(symbol, entry_time_ms):
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
        
        for item in pnl_list:
            updated_time = int(item.get("updatedTime", 0))
            if updated_time >= entry_time_ms:
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

def update_bybit_stop_loss(symbol, sl_price, active_trade=None):
    if active_trade:
        qty_val = float(active_trade.get("qty", 0.0))
        side = "Buy" if active_trade.get("direction") == "Bullish" else "Sell"
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
        
    live_price = bot_state.get(f"live_price_{symbol}")
    if live_price is None:
        live_price = get_fallback_price(symbol)
        
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
        
    live_price = bot_state.get(f"live_price_{symbol}")
    if live_price is None:
        live_price = get_fallback_price(symbol)
        
    if live_price is not None:
        if side == "Buy" or side == "Long":  # Long position: Take Profit must be > current price
            if tp_price <= live_price:
                log_event("WARNING", f"[{symbol}] TP {tp_price} <= price {live_price} on a long — skipped")
                print(f"[Bybit API] Take Profit update skipped for Long {symbol}: Proposed TP {tp_price:.4f} is <= current price {live_price:.4f}.")
                return False
        else:  # Short position: Take Profit must be < current price
            if tp_price >= live_price:
                log_event("WARNING", f"[{symbol}] TP {tp_price} >= price {live_price} on a short — skipped")
                print(f"[Bybit API] Take Profit update skipped for Short {symbol}: Proposed TP {tp_price:.4f} is >= current price {live_price:.4f}.")
                return False

    payload = {
        "category": "linear",
        "symbol": symbol,
        "takeProfit": format_bybit_price(symbol, tp_price),
        "positionIdx": 0
    }
    res = bybit_post_request("/v5/position/trading-stop", payload)
    if res.get("retCode") == 0:
        print(f"[Bybit API] Successfully updated Take Profit on Bybit to {tp_price:.4f} for {symbol}.")
        return True
    elif "not modified" in str(res.get("retMsg", "")).lower() or res.get("retCode") == 130089:
        print(f"[Bybit API] Take Profit for {symbol} is already set to {tp_price:.4f} (not modified).")
        return True
    else:
        print(f"[Bybit API Error] Failed to update Take Profit for {symbol}: {res.get('retMsg')}")
        return False

def place_bybit_limit_order(symbol, side, qty, price, sl=None, tp=None, reduce_only=False):
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Limit",
        "qty": str(qty),
        "price": format_bybit_price(symbol, price),
        "timeInForce": "GTC",
        "positionIdx": 0
    }
    if reduce_only:
        payload["reduceOnly"] = True
    if sl:
        payload["stopLoss"] = format_bybit_price(symbol, sl)
    if tp:
        payload["takeProfit"] = format_bybit_price(symbol, tp)
    res = execute_bybit_order_ws_or_rest("/v5/order/create", payload)
    return res

def place_bybit_taker_ioc_order(symbol, side, qty, sl=None, tp=None, reduce_only=False):
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "IOC",
        "positionIdx": 0
    }
    if reduce_only:
        payload["reduceOnly"] = True
    if sl:
        payload["stopLoss"] = format_bybit_price(symbol, sl)
    if tp:
        payload["takeProfit"] = format_bybit_price(symbol, tp)
    res = execute_bybit_order_ws_or_rest("/v5/order/create", payload)
    return res

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

def get_bybit_last_execution(symbol):
    res = bybit_get_request("/v5/execution/list", {"category": "linear", "symbol": symbol, "limit": 1})
    if res.get("retCode") == 0:
        exec_list = res.get("result", {}).get("list", [])
        if exec_list:
            return exec_list[0]
    return None

def get_real_bybit_balance():
    api_key = os.getenv("BYBIT_API_KEY", "").strip()
    api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    
    if not api_key or not api_secret:
        return "API_KEYS_MISSING"
        
    max_balance = 0.0
    geo_blocked_encountered = False
    
    for account_type in ["UNIFIED", "CONTRACT", "SPOT", "FUND"]:
        if account_type == "FUND":
            res = bybit_get_request("/v5/asset/transfer/query-account-coins-balance", {"accountType": "FUND"})
        else:
            res = bybit_get_request("/v5/account/wallet-balance", {"accountType": account_type})
            
        ret_code = res.get("retCode")
        if ret_code == 0:
            if account_type == "FUND":
                balances = res.get("result", {}).get("balance", [])
                fund_sum = 0.0
                for b_item in balances:
                    coin_name = b_item.get("coin", "")
                    coin_bal = float(b_item.get("walletBalance", "0"))
                    if coin_name in ["USDT", "USDC"]:
                        fund_sum += coin_bal
                    elif coin_name == "BTC":
                        fund_sum += coin_bal * float(get_fallback_price("BTCUSDT") or 60000.0)
                    elif coin_name == "ETH":
                        fund_sum += coin_bal * float(get_fallback_price("ETHUSDT") or 33000.0)
                    elif coin_name == "SOL":
                        fund_sum += coin_bal * float(get_fallback_price("SOLUSDT") or 140.0)
                max_balance = max(max_balance, fund_sum)
            else:
                list_data = res.get("result", {}).get("list", [])
                if list_data:
                    total_equity = list_data[0].get("totalEquity") or list_data[0].get("totalWalletBalance") or "0"
                    max_balance = max(max_balance, float(total_equity))
        else:
            ret_msg = res.get("retMsg", "")
            # If the response is HTTP error (retCode is HTTP status code)
            if isinstance(ret_code, int) and (400 <= ret_code <= 599):
                print(f"[Bybit Balance] HTTP {ret_code} for {account_type}: {ret_msg}")
                if ret_code == 403 and ("cloudfront" in ret_msg.lower() or "block" in ret_msg.lower()):
                    geo_blocked_encountered = True
            else:
                # Suppress legacy warnings (10001, 10003) for Unified accounts
                if not (ret_code in [10001, 10003] and account_type in ["SPOT", "CONTRACT", "FUND"]):
                    print(f"[Bybit Balance] Query error for {account_type}: Code {ret_code} - {ret_msg}")
                    
    if max_balance > 0.0:
        return max_balance
    if geo_blocked_encountered:
        return "GEO_BLOCKED"
    return 0.0

def get_real_bybit_balance_cached(force=False):
    global _cached_balance, _last_balance_fetch
    now = time.time()
    if force or (now - _last_balance_fetch > BALANCE_UPDATE_INTERVAL_SECS):
        try:
            val = get_real_bybit_balance()
            with _balance_lock:
                _cached_balance = val
                _last_balance_fetch = now
            if TRADE_MODE != "simulation" and isinstance(val, (int, float)) and val > 0:
                bot_state["simulated_balance"] = val
        except Exception as e:
            print(f"[Bybit Balance] Error in balance update (forced={force}): {e}")
    with _balance_lock:
        return _cached_balance

def run_bybit_balance_updater():
    global _cached_balance, _last_balance_fetch
    print("[Bybit Balance] Background updater thread started.")
    # Fetch immediately at startup
    try:
        val = get_real_bybit_balance()
        with _balance_lock:
            _cached_balance = val
            _last_balance_fetch = time.time()
        if TRADE_MODE != "simulation" and isinstance(val, (int, float)) and val > 0:
            bot_state["simulated_balance"] = val
        print(f"[Bybit Balance] Startup background update success: {val}")
    except Exception as e:
        print(f"[Bybit Balance] Startup background update error: {e}")
        
    while True:
        time.sleep(BALANCE_UPDATE_INTERVAL_SECS)  # Query Bybit balance periodically based on configuration
        try:
            val = get_real_bybit_balance()
            with _balance_lock:
                _cached_balance = val
                _last_balance_fetch = time.time()
            if TRADE_MODE != "simulation" and isinstance(val, (int, float)) and val > 0:
                bot_state["simulated_balance"] = val
        except Exception as e:
            print(f"[Bybit Balance] Error in background balance update: {e}")



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
JOURNAL_HEADER = ["timestamp", "symbol", "interval", "direction", "entry_price", "exit_price", "pnl_usd", "change_pct", "success", "reason", "balance", "leverage", "confidence"]

def log_trade_journal(trade: dict):
    """Append a closed trade to trade_journal.csv."""
    import csv
    write_header = not os.path.exists(JOURNAL_PATH)
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

FUNDING_ARB_THRESHOLD = 0.001  # 0.1% — above this, shorts earn funding income
FUNDING_ARB_SIZE_USD = 20.0    # Fixed notional size per arbitrage position

def run_funding_rate_arbitrage_monitor():
    """Monitor funding rates. When rate > 0.1%, open a small short to collect funding income."""
    print("[Funding Arb] Monitor started.")
    time.sleep(60)  # Wait for bot to initialize
    while True:
        try:
            for sym in SUPPORTED_SYMBOLS:
                rate = get_funding_rate(sym)
                arb_key = f"funding_arb_{sym}"
                existing = bot_state.get(arb_key)

                if rate > FUNDING_ARB_THRESHOLD and not existing:
                    # High positive funding — shorts earn. Open small short.
                    print(f"[Funding Arb] {sym} funding rate {rate*100:.4f}% > 0.1%. Opening arb short.")
                    if TRADE_MODE == "live":
                        curr_sym_price = bot_state.get(f"live_price_{sym}") or get_fallback_price(sym) or 1.0
                        qty_str = format_bybit_qty(sym, FUNDING_ARB_SIZE_USD / max(1e-6, curr_sym_price))
                        res = place_bybit_order(symbol=sym, side="Sell", qty=qty_str, reduce_only=False)
                        if res and res.get("retCode") == 0:
                            bot_state[arb_key] = {"qty": qty_str, "open_rate": rate}
                            send_telegram_alert(
                                f"💰 *FUNDING ARB OPENED*\n"
                                f"• Asset: {sym}\n"
                                f"• Funding Rate: {rate*100:.4f}%\n"
                                f"• Side: Short (collecting funding)\n"
                                f"• Size: ${FUNDING_ARB_SIZE_USD}"
                            )
                elif existing and rate < FUNDING_ARB_THRESHOLD * 0.3:
                    # Funding rate has normalized — close the arb short
                    print(f"[Funding Arb] {sym} funding rate normalized ({rate*100:.4f}%). Closing arb short.")
                    if TRADE_MODE == "live":
                        qty_str = existing["qty"]
                        res = place_bybit_order(symbol=sym, side="Buy", qty=qty_str, reduce_only=True)
                        if res and res.get("retCode") == 0:
                            bot_state.pop(arb_key, None)
                            send_telegram_alert(
                                f"✅ *FUNDING ARB CLOSED*\n"
                                f"• Asset: {sym}\n"
                                f"• Current Rate: {rate*100:.4f}%"
                            )
        except Exception as e:
            print(f"[Funding Arb] Error: {e}")
        time.sleep(300)  # Check every 5 minutes

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
            
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(DB_FILE):
                    zipf.write(DB_FILE, os.path.basename(DB_FILE))
                if os.path.exists(JOURNAL_PATH):
                    zipf.write(JOURNAL_PATH, os.path.basename(JOURNAL_PATH))
                    
            print(f"[Backup Scheduler] Created local compressed backup: {zip_filename}")
            
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
            pain_feedback.verify_pending_pain_trades(database_module=database, fetch_kline_func=get_history)
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
            today_date_str = time.strftime("%Y-%m-%d", now_gm)
            # Sleep until next midnight UTC
            seconds_to_midnight = 86400 - (now_gm.tm_hour * 3600 + now_gm.tm_min * 60 + now_gm.tm_sec)
            time.sleep(max(1, seconds_to_midnight))
            
            with bot_state_lock:
                last_date = bot_state.get("last_daily_summary_date", "")
                if last_date != today_date_str:
                    bot_state["last_daily_summary_date"] = today_date_str
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
    print("[Scheduler] Automated weekly Sunday retraining scheduler started.")
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            # Sunday is weekday 6. Check if it is Sunday between 00:00 and 00:20 UTC.
            if now_utc.weekday() == 6 and now_utc.hour == 0 and now_utc.minute < 20:
                print(f"[Scheduler] Sunday 00:00 UTC detected. Triggering weekly model retraining...")
                retrain_models_thread(is_manual=False)
                # Sleep 45 minutes to prevent double trigger within the same hour
                time.sleep(2700)
        except Exception as e:
            print(f"[Scheduler] Error in weekly retraining scheduler: {e}")
        
        # Sleep for 15 minutes before checking time again
        time.sleep(900)

import logging
import wsgiref.simple_server
from werkzeug.serving import run_simple

def run_flask():
    import sys
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    port = int(os.environ.get("PORT", 5001))
    flask_host = os.environ.get("FLASK_HOST", "0.0.0.0")
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
        # Load selected features for both regimes (ensuring feature file matches model mtime)
        selected_features_filename = f"selected_features_{iv}.json"
        trending_features_filename = f"selected_features_{iv}_trending.json"
        ranging_features_filename = f"selected_features_{iv}_ranging.json"
        model_trending_filename = f"ensemble_trending_trend_{iv}_xgb.json"
        feat_trending = None
        for f_name in [trending_features_filename, selected_features_filename, "selected_features_30.json"]:
            if os.path.exists(f_name):
                try:
                    with open(f_name, "r") as f:
                        feat_trending = json.load(f)
                        if feat_trending:
                            break
                except Exception:
                    pass
        if feat_trending is None:
            from core import features as master_features
            feat_trending = master_features

        feat_ranging = None
        for f_name in [ranging_features_filename, selected_features_filename, "selected_features_30.json"]:
            if os.path.exists(f_name):
                try:
                    with open(f_name, "r") as f:
                        feat_ranging = json.load(f)
                        if feat_ranging:
                            break
                except Exception:
                    pass
        if feat_ranging is None:
            from core import features as master_features
            feat_ranging = master_features
                
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
        
        from config import SUPPORTED_MANIFEST_SCHEMA_VERSION

        def check_startup_manifest_health(prefix: str) -> bool:
            manifest_path = f"{prefix}_manifest.json"
            if not os.path.exists(manifest_path):
                print(f"[CRITICAL ALERT] Model manifest missing for {prefix}. Engaging RULE_BASED_FALLBACK for interval {iv}.")
                return False
            try:
                with open(manifest_path, "r") as f:
                    m = json.load(f)
                schema_v = m.get("manifest_schema_version", 1)
                if schema_v > SUPPORTED_MANIFEST_SCHEMA_VERSION or schema_v < 1:
                    print(f"[CRITICAL ALERT] Model manifest schema version mismatch ({schema_v} > {SUPPORTED_MANIFEST_SCHEMA_VERSION}) for {prefix}. Engaging RULE_BASED_FALLBACK for interval {iv}.")
                    return False
                from model_governance import model_governance_engine
                model_governance_engine.log_barrier_manifest_audit("BTCUSDT", str(iv), m)
                return True
            except Exception as e:
                print(f"[CRITICAL ALERT] Corrupted model manifest for {prefix}: {e}. Engaging RULE_BASED_FALLBACK for interval {iv}.")
                return False

        from mlops_engine import load_production_model_from_registry

        reg_model_trending, ver_trending = load_production_model_from_registry(interval=str(iv), regime="trending", live_features=feat_trending)
        if reg_model_trending is not None:
            models_by_interval[iv]["trending"]["trend"] = reg_model_trending
            models_by_interval[iv]["trending"]["model_version"] = ver_trending
        elif os.path.exists(f"{prefixes['trending_trend']}_xgb.json") and check_startup_manifest_health(prefixes['trending_trend']):
            models_by_interval[iv]["trending"]["trend"] = load_ensemble_classifier(prefixes["trending_trend"], n_features_trending, feature_names=feat_trending)
            models_by_interval[iv]["trending"]["model_version"] = f"btc_{iv}m_trending_clf:v1.0"

        if os.path.exists(f"{prefixes['trending_price']}_xgb.json") and check_startup_manifest_health(prefixes['trending_price']):
            models_by_interval[iv]["trending"]["price"] = load_ensemble_regressor(prefixes["trending_price"], n_features_trending, feature_names=feat_trending)
        if os.path.exists(prefixes["trending_meta"]):
            meta_clf = XGBClassifier()
            meta_clf.load_model(prefixes["trending_meta"])
            models_by_interval[iv]["trending"]["meta"] = meta_clf

        reg_model_ranging, ver_ranging = load_production_model_from_registry(interval=str(iv), regime="ranging", live_features=feat_ranging)
        if reg_model_ranging is not None:
            models_by_interval[iv]["ranging"]["trend"] = reg_model_ranging
            models_by_interval[iv]["ranging"]["model_version"] = ver_ranging
        elif os.path.exists(f"{prefixes['ranging_trend']}_xgb.json") and check_startup_manifest_health(prefixes['ranging_trend']):
            models_by_interval[iv]["ranging"]["trend"] = load_ensemble_classifier(prefixes["ranging_trend"], n_features_ranging, feature_names=feat_ranging)
            models_by_interval[iv]["ranging"]["model_version"] = f"btc_{iv}m_ranging_clf:v1.0"
        if os.path.exists(f"{prefixes['ranging_price']}_xgb.json") and check_startup_manifest_health(prefixes['ranging_price']):
            models_by_interval[iv]["ranging"]["price"] = load_ensemble_regressor(prefixes["ranging_price"], n_features_ranging, feature_names=feat_ranging)
        if os.path.exists(prefixes["ranging_meta"]):
            meta_clf = XGBClassifier()
            meta_clf.load_model(prefixes["ranging_meta"])
            models_by_interval[iv]["ranging"]["meta"] = meta_clf

        # Fallback to trending model if ranging model files do not exist on disk
        if models_by_interval[iv]["ranging"]["trend"] is None:
            models_by_interval[iv]["ranging"]["trend"] = models_by_interval[iv]["trending"]["trend"]
            models_by_interval[iv]["ranging"]["model_version"] = models_by_interval[iv]["trending"].get("model_version", f"btc_{iv}m_trending_clf:v1.0")
            print(f"[Model Load Fallback] Ranging trend model missing for {iv}m; using trending model fallback.")
        if models_by_interval[iv]["ranging"]["price"] is None:
            models_by_interval[iv]["ranging"]["price"] = models_by_interval[iv]["trending"]["price"]
            print(f"[Model Load Fallback] Ranging price model missing for {iv}m; using trending model fallback.")
        if models_by_interval[iv]["ranging"]["meta"] is None:
            models_by_interval[iv]["ranging"]["meta"] = models_by_interval[iv]["trending"].get("meta")
            
        # Load calibrators if they exist, or default to identity mapping
        trending_cal_file = f"calibrator_trending_{iv}.json"
        if os.path.exists(trending_cal_file):
            with open(trending_cal_file, "r") as f:
                models_by_interval[iv]["trending"]["calibrator"] = json.load(f)
            print(f"Loaded Isotonic Regression calibrator: {trending_cal_file}")
        else:
            models_by_interval[iv]["trending"]["calibrator"] = {"X": [0.0, 1.0], "y": [0.0, 1.0]}
            print(f"Initialized identity calibrator for trending_{iv}")

        ranging_cal_file = f"calibrator_ranging_{iv}.json"
        if os.path.exists(ranging_cal_file):
            with open(ranging_cal_file, "r") as f:
                models_by_interval[iv]["ranging"]["calibrator"] = json.load(f)
            print(f"Loaded Isotonic Regression calibrator: {ranging_cal_file}")
        else:
            models_by_interval[iv]["ranging"]["calibrator"] = {"X": [0.0, 1.0], "y": [0.0, 1.0]}
            print(f"Initialized identity calibrator for ranging_{iv}")
            
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
        print(f"Warning: Could not load ensemble models for interval {iv}: {e}")



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
                print(f"[Hot-Reload] Recalculated calibration thresholds for {iv} (p95: {p95:.2f}, max_conf: {max_conf:.2f})")
            except Exception as e:
                print(f"[Hot-Reload] Warning: Could not recalculate thresholds for {iv}m: {e}")
            reloaded.append(iv)
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
                                found_symbols.add(sym)
                                if sym == "BTCUSDT":
                                    live_price = val
                                    bot_state["live_price"] = val
                                    bot_state["last_update"] = time.time()
                                    if not ws_active:
                                        last_ws_update_time = time.time()

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
                                        bot_state[f"live_price_{sym}"] = float(val_str)
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
                                    if sym == "BTCUSDT":
                                        live_price = binance_prices[sym]
                                        bot_state["live_price"] = binance_prices[sym]
                                        bot_state["last_update"] = time.time()
                                        if not ws_active:
                                            last_ws_update_time = time.time()
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
                        order_flow_data[sym] = {"cvd": 0.0, "ofi": 0.0, "prev_bid_price": 0.0, "prev_ask_price": 0.0, "prev_bid_size": 0.0, "prev_ask_size": 0.0, "latest_bids": [], "latest_asks": [], "ob_imbalance_L2": 0.0, "ob_spread_L2": 0.0, "liq_long_1h": 0.0, "liq_short_1h": 0.0}
                    
                    state = order_flow_data[sym]
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
                ping_interval=20, ping_timeout=10,
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

def on_private_message(ws, message):
    import json
    import time
    global last_private_ws_update_time, _last_ws_pos_sync_time
    last_private_ws_update_time = time.time()
    try:
        data = json.loads(message)
        op = data.get("op")
        topic = data.get("topic")
        
        # Capture operations responses (e.g. order.create or order.cancel callbacks)
        req_id = data.get("reqId")
        if req_id:
            with _ws_responses_lock:
                _ws_responses[req_id] = data
                if len(_ws_responses) > 200:
                    _ws_responses.clear()
                
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
            
            # Debounce position sync to max once every 3.0s to avoid thread storms
            now_t = time.time()
            if now_t - _last_ws_pos_sync_time >= 3.0:
                _last_ws_pos_sync_time = now_t
                import threading
                threading.Thread(target=sync_active_positions_from_bybit, daemon=True).start()
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
                ping_interval=20, ping_timeout=10,
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


def add_features(df):
    return features_module.add_features(df, fetch_calendar_callback=fetch_economic_calendar_cached)

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
                df = add_features(df)
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
                if avg_score > 0.15:
                    sentiment = "Bullish"
                elif avg_score < -0.15:
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
                    "bullish", "surge", "surges", "surging", "rally", "rallies", "rallying", "breakout", "skyrocket", "skyrockets",
                    "gain", "gains", "gaining", "all-time high", "ath", "soar", "soars", "soaring", "pump", "pumping", "pumps",
                    "rebound", "rebounds", "rebounding", "recovery", "recovers", "recovering", "uptrend", "outperform", "outperforms",
                    "bounce", "bouncing", "climb", "climbs", "climbing", "record high", "milestone", "moon", "soaring",
                    # Flows & Institutional
                    "inflow", "inflows", "accumulation", "accumulating", "accumulate", "adoption", "approval", "approved", "approves",
                    "institutional", "profit", "profits", "profitable", "upward", "bull", "bulls", "optimistic", "growth", "expanding",
                    "buy", "buying", "bought", "reserve", "treasury", "sec approval", "etf approval", "partnership", "mainnet", "upgrade",
                    "stimulus", "rate cut", "rate cuts", "dovish", "easing", "support level", "holder", "holders"
                }
                bearish_keywords = {
                    # Market Price Action & Losses
                    "bearish", "crash", "crashes", "crashing", "dump", "dumps", "dumping", "plunge", "plunges", "plunging",
                    "drop", "drops", "dropping", "fall", "falls", "falling", "slide", "slides", "sliding", "tumble", "tumbles", "tumbling",
                    "collapse", "collapses", "collapsing", "selloff", "sell-off", "selloffs", "downtrend", "slump", "slumps", "retreat",
                    "retreats", "bleeding", "capitulation", "correction", "wilt", "wilts", "wilted",
                    # Risk, Hacks & Failures
                    "hack", "hacked", "hacks", "exploit", "exploited", "exploits", "stolen", "drain", "drained", "scam", "rugpull",
                    "fraud", "bankruptcy", "bankrupt", "bankruptcies", "insolvent", "insolvency", "liquidation", "liquidations", "liquidated",
                    "outflow", "outflows", "loss", "losses", "bear", "bears", "pessimistic", "panic", "sell", "selling", "sold",
                    "cut headcount", "layoff", "layoffs", "cuts 1", "cuts 2", "job cuts",
                    # Regulatory & Macro Hardship
                    "ban", "banned", "banning", "bans", "lawsuit", "lawsuits", "sued", "suing", "sec", "crackdown", "probe", "investigation",
                    "subpoena", "fine", "fined", "penalty", "rate hike", "rate hikes", "hawkish", "inflation", "recession", "war", "restriction"
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
                if avg_score > 0.10:
                    sentiment = "Bullish"
                elif avg_score < -0.10:
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
        if avg_score > 0.15:
            sentiment = "Bullish"
        elif avg_score < -0.15:
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
        
    # WebSocket Cache Check (Instantly bypass HTTP request if cache is warm)
    with order_flow_lock:
        cached = order_flow_data.get(symbol)
        if cached and cached.get("latest_bids") and cached.get("latest_asks"):
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
                    "ask_depth": float(ask_depth_usd)
                }
            except Exception:
                pass

    from data import get_orderbook_imbalance as data_get_ob
    return data_get_ob(symbol=symbol)


# ==========================================
# CONFIDENCE CALIBRATION & HISTORICAL STATS
# ==========================================
def calculate_historical_thresholds(model_trend, interval):
    if model_trend is None:
        return 0.55, 0.75
    print(f"Fetching historical data to calibrate confidence percentiles (last 5,000 candles for {SYMBOL} + BTCUSDT on {interval}m interval)...")
    try:
        df_target = get_history(symbol=SYMBOL, interval=interval, limit=1000, pages=5)
        df_btc = get_history(symbol="BTCUSDT", interval=interval, limit=1000, pages=5)
        
        if df_target is not None and len(df_target) > 0 and df_btc is not None and len(df_btc) > 0:
            df_btc_sub = df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"})
            df = pd.merge(df_target, df_btc_sub, on="timestamp", how="inner")
            if len(df) > 0:
                df = merge_derivatives_sentiment_features(df, symbol=SYMBOL, interval=interval)
                df = add_features(df)
                
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
        print(f"Error fetching funding rate: {e}")
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

def calculate_covariance_multiplier(new_symbol, new_direction):
    """
    Calculates a position sizing multiplier based on empirical rolling portfolio covariance.
    Penalizes highly correlated assets in the same direction.
    Allows offsetting/hedging for assets in opposite directions.
    """
    is_stressed = False
    try:
        df_vol = get_history(symbol=new_symbol, interval="60", limit=30)
        if df_vol is not None and not df_vol.empty and "ATR_norm" in df_vol.columns:
            rolling_atr = df_vol["ATR_norm"].tail(30)
            atr_mean = rolling_atr.mean()
            atr_std = rolling_atr.std()
            vol_z_score = (df_vol["ATR_norm"].iloc[-1] - atr_mean) / (atr_std + 1e-8) if atr_std > 0 else 0.0
            is_stressed = vol_z_score > 2.0
            if is_stressed:
                print(f"[Stress Covariance] Volatility Z-score: {vol_z_score:.2f} > 2.0. Stressed correlation mode active.")
    except Exception as e:
        log_event("DEBUG", f"Stress evaluation fallback for {new_symbol}: {e}")

    def get_correlation(s1, s2):
        if s1 == s2:
            return 1.0
        if is_stressed:
            return 0.95
        return get_rolling_correlation(s1, s2)

    # Collect active trades from all timeframes
    open_trades = []
    for tf_key in ACTIVE_TRADE_TF_KEYS:
        open_trades.extend(bot_state.get(f"active_trade_{tf_key}", []))

    if not open_trades:
        return 1.0, 0.0

    total_risk = 0.0
    breakdown = []
    
    for t in open_trades:
        open_sym = t.get("symbol")
        open_dir = t.get("direction")
        if not open_sym or not open_dir:
            continue
        r = get_correlation(new_symbol, open_sym)
        
        if new_direction == open_dir:
            impact = r
            risk_type = "CONCENTRATION"
        else:
            impact = -r
            risk_type = "HEDGE"
            
        total_risk += impact
        breakdown.append(f"  - Active: {open_sym} {open_dir} | Correlation: {r:.2f} | Risk impact: {impact:+.2f} ({risk_type})")

    if total_risk <= 0:
        multiplier = 1.0
    else:
        multiplier = 1.0 / (1.0 + total_risk)
        multiplier = max(0.20, min(1.0, multiplier))

    print(f"\n[Portfolio Covariance Analysis] New Entry: {new_symbol} {new_direction}")
    for item in breakdown:
        print(item)
    print(f"  - Total Net Correlation Risk: {total_risk:+.2f} -> Covariance Multiplier: {multiplier:.2f}x\n")

    return float(multiplier), float(total_risk)

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
                        bot_state[f"live_price_{sym}"] = val
                        found_symbols.add(sym)
                        if sym == "BTCUSDT":
                            live_price = val
                            bot_state["live_price"] = val
                            last_ws_update_time = time.time()
                            bot_state["last_update"] = last_ws_update_time
        
        # Fall back to external sources (Binance/Coinbase) for any missing symbols (e.g. LINKUSDT on testnet)
        for sym in SUPPORTED_SYMBOLS:
            if sym not in found_symbols:
                val = get_fallback_price(sym)
                if val is not None:
                    bot_state[f"live_price_{sym}"] = val
                    if sym == "BTCUSDT" and live_price is None:
                        live_price = val
                        bot_state["live_price"] = val
                        last_ws_update_time = time.time()
                        bot_state["last_update"] = last_ws_update_time
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

        # Re-sync bot_state active trades with unified lock acquisition hierarchy
        with active_execution_lock:
            with active_trades_lock:
                matched_symbols = set()
            
            for tf_key in ACTIVE_TRADE_TF_KEYS:
                current_trades = bot_state.get(f"active_trade_{tf_key}", [])
                if not isinstance(current_trades, list):
                    current_trades = []
                
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
                            print(f"[Side Mismatch Guard] WARNING: {symbol} in {tf_key} has direction {trade_direction} but Bybit position is {pos_side}! Force-closing to prevent inverted SL/TP.")
                            if TRADE_MODE != "simulation":
                                close_side = "Sell" if pos_side == "Buy" else "Buy"
                                place_bybit_order(symbol, close_side, str(pos.get("size", t["qty"])), reduce_only=True)
                                if t.get("bybit_scale_out_order_id"):
                                    cancel_bybit_order(symbol, t["bybit_scale_out_order_id"])
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
                            
                        # Fix Take Profit if it is unset (0.0)
                        if t.get("take_profit", 0.0) == 0.0:
                            if direction == "Bullish":
                                t["take_profit"] = max(mark_price + 1.25 * current_atr, avg_price + 1.25 * current_atr)
                            else:
                                t["take_profit"] = min(mark_price - 1.25 * current_atr, avg_price - 1.25 * current_atr)
                            if TRADE_MODE != "simulation":
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
                                else:
                                    t["stop_loss"] = bybit_sl
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
                        if not t.get("exit_processed", False):
                            t["bybit_closed"] = True
                            updated_trades.append(t)
                        else:
                            print(f"[Sync Cleanup] Removed already processed closed trade for {symbol} ({tf_key}).")
                
                bot_state[f"active_trade_{tf_key}"] = updated_trades
    
            # Reconstruct any open positions on Bybit that are NOT in bot_state (orphaned/manual positions)
            recovered = 0
            for symbol, pos in open_positions.items():
                with active_execution_lock:
                    in_active_execution = symbol in active_execution_symbols
                if in_active_execution:
                    print(f"[Crash Recovery] Skipped recovery scan for {symbol} - trade is currently being executed async.")
                    continue
                # FIX: Re-check ALL active timeframes live in bot_state at this moment.
                # matched_symbols is built at the START of the sync loop and may be stale
                # if a manual trade was added to bot_state between the sync loop start and now.
                currently_tracked = any(
                    any(t.get("symbol") == symbol for t in bot_state.get(f"active_trade_{k}", []))
                    for k in ACTIVE_TRADE_TF_KEYS
                )
                if currently_tracked:
                    print(f"[Crash Recovery] Skipped recovery for {symbol} - already tracked in current bot_state (live re-check).")
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
                    
                    # Calculate proper ATR on recovery
                    calc_atr = abs(avg_price - sl_price) / 0.75 if sl_price > 0 else 0.015 * avg_price
                    if calc_atr > 0.05 * avg_price or calc_atr == 0:
                        calc_atr = 0.015 * avg_price
                        log_event("WARNING", f"[{symbol}] ATR unavailable in recovery — using 1.5% fallback ({calc_atr:.4f})")
                    
                    # Sanitize TP and SL on recovery
                    if tp_price == 0.0:
                        if direction == "Bullish":
                            tp_price = max(mark_price + 1.25 * calc_atr, avg_price + 1.25 * calc_atr)
                        else:
                            tp_price = min(mark_price - 1.25 * calc_atr, avg_price - 1.25 * calc_atr)
                            
                    if sl_price == 0.0 or abs(avg_price - sl_price) > 3.0 * calc_atr:
                        if direction == "Bullish":
                            sl_price = avg_price - 0.75 * calc_atr
                            if liq_price > 0.0 and sl_price <= liq_price:
                                sl_price = min(avg_price * 0.999, liq_price + 0.2 * calc_atr)
                            if sl_price >= avg_price:
                                log_event("ERROR", f"[{symbol}] Liquidation too close for a valid long stop — leaving exchange SL")
                                send_telegram_alert(f"🚨 {symbol}: cannot place valid SL, liq {liq_price} too close to entry {avg_price}")
                                sl_price = None
                        else:
                            sl_price = avg_price + 0.75 * calc_atr
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
     
                    # Dynamic timeframe and confidence resolution from prediction history matching symbol and direction
                    matched_tf = "1h"  # fallback default
                    matched_confidence = 0.0
                    for p in reversed(bot_state.get("prediction_history", [])):
                        if p.get("symbol") == symbol and p.get("direction") == direction:
                            # Verify if it was within the last 48 hours to avoid matching ancient predictions
                            if abs(p.get("timestamp", 0) - time.time()) < 86400 * 2:
                                token_val_not_used = None
                                matched_tf_interval = p.get("interval", "60")
                                tf_map_inv = {"5": "5m", "15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h", "360": "6h"}
                                matched_tf = tf_map_inv.get(matched_tf_interval, "1h")
                                matched_confidence = float(p.get("calibrated_confidence", p.get("confidence", 0.63)))
                                break

                    recovered_trade = {
                        "trade_id": f"{symbol}_{trade_uuid}_recovered",
                        "bybit_order_id": entry_order_id,
                        "bybit_scale_out_order_id": scale_out_order_id,
                        "symbol": symbol,
                        "entry_price": avg_price,
                        "predicted_price": avg_price,
                        "stop_loss": sl_price,
                        "take_profit": tp_price,
                        "direction": direction,
                        "end_time": float(time.time() + 3600 * 48),
                        "entry_time": int(time.time() * 1000),
                        "atr_dollars": calc_atr,
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
                        "recovered": True
                    }
                    
                    tf_key = matched_tf
                    # Cross-timeframe duplicate guard: skip if symbol already active in ANY timeframe
                    already_in_any_tf = any(
                        any(t.get("symbol") == symbol for t in bot_state.get(f"active_trade_{k}", []))
                        for k in ACTIVE_TRADE_TF_KEYS
                    )
                    if already_in_any_tf:
                        print(f"[Crash Recovery] Skipped duplicate recovery for {symbol} - already tracked in an active timeframe.")
                        continue
                    active_trades_list = bot_state.get(f"active_trade_{tf_key}", [])
                    if not isinstance(active_trades_list, list):
                        active_trades_list = []
                    active_trades_list.append(recovered_trade)
                    bot_state[f"active_trade_{tf_key}"] = active_trades_list
                    try:
                        import database
                        database.save_active_trades(tf_key, [recovered_trade])
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
    try:
        _execute_bybit_trade_async_inner(*args, **kwargs)
    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"[CRITICAL EXECUTION ERROR] Async trade execution failed for {symbol}: {err_msg}")
        traceback.print_exc()
        send_telegram_alert(f"🚨 *Async Trade Execution Error* 🚨\n• Symbol: `{symbol}`\n• Error: `{err_msg}`")
    finally:
        with active_execution_lock:
            active_execution_symbols.discard(symbol)

def _execute_bybit_trade_async_inner(symbol, iv, tf, ml_trend, leverage_val, qty_str, raw_qty, entry_price, stop_loss_price, take_profit_price, position_size_usd, kelly_fraction, calibrated_confidence, ml_confidence, dynamic_conf_threshold, latest_completed_ts, latest_candle, pred_change, predicted_price, atr_dollars, tp_multiplier_adjusted, sl_multiplier_adjusted, df_completed, trade_uuid, duration_seconds, active_trade_key, is_oversized=False, intended_size_usd=None):

    if latest_candle is None:
        latest_candle = {}
    if df_completed is None:
        df_completed = pd.DataFrame()
        
    bybit_success = True
    bybit_order_id = None
    bybit_scale_out_order_id = None
    actual_qty = raw_qty
    order_res = {}
    
    # 1. Live Exchange Position Guard
    try:
        pos_list = get_all_bybit_positions()
        if pos_list:
            existing_pos = next((p for p in pos_list if p.get("symbol") == symbol and float(p.get("size", "0")) > 0), None)
            if existing_pos:
                print(f"[{symbol} {iv}m API Block] Live order placement skipped: a live position already exists on Bybit.")
                sync_active_positions_from_bybit()
                return
    except Exception as pos_check_err:
        print(f"[{symbol} {iv}m API Warning] Live Position Guard check failed: {pos_check_err}")

    # SL/TP Geometry Direction Guard (Hard Abort — Do NOT auto-correct)
    try:
        trade_calculators.assert_valid_geometry(ml_trend, entry_price, stop_loss_price, take_profit_price, symbol=f"{symbol} {iv}m")
    except ValueError as geom_err:
        log_event("ERROR", str(geom_err))
        send_telegram_alert(f"🚨 *CRITICAL ORDER ABORT*: {symbol} {iv}m invalid geometry: SL={stop_loss_price}, Entry={entry_price}, TP={take_profit_price}")
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
            order_res = place_bybit_taker_ioc_order(symbol, side, qty_str, sl=stop_loss_price, tp=take_profit_price)
            if order_res.get("retCode") == 0:
                bybit_order_id = order_res.get("result", {}).get("orderId")
                bybit_success = True
                time.sleep(0.5)
                order_details = get_bybit_order_details(symbol, bybit_order_id)
                if order_details:
                    entry_price = float(order_details.get("avgPrice", entry_price))
                    actual_qty = float(order_details.get("cumExecQty", raw_qty))
                else:
                    fill_exec = get_bybit_last_execution(symbol)
                    if fill_exec:
                        entry_price = float(fill_exec.get("execPrice", entry_price))
                    actual_qty = raw_qty
            else:
                print(f"[{symbol} {iv}m API ERROR] Taker IOC order failed: {order_res.get('retMsg')}")
        else:
            # Normal Volatility: Place Limit Maker entry order with dynamic price chasing
            for chase in range(5):
                bid, ask, last = get_bybit_bid_ask(symbol)
                if bid is None or ask is None:
                    bid, ask = entry_price, entry_price
                limit_price = bid if side == "Buy" else ask
                print(f"[{symbol} {iv}m API] Placing Limit Maker order at ${limit_price:.4f} (Chase {chase+1}/5)...")
                order_res = place_bybit_limit_order(symbol, side, qty_str, limit_price)
                if order_res.get("retCode") == 0:
                    bybit_order_id = order_res.get("result", {}).get("orderId")
                    filled = False
                    for idx in range(24):
                        time.sleep(0.5)
                        with _ws_filled_orders_lock:
                            ws_details = _ws_filled_orders.get(bybit_order_id)
                        if ws_details:
                            entry_price = float(ws_details.get("avgPrice", limit_price))
                            actual_qty = float(ws_details.get("cumExecQty", raw_qty))
                            filled = True
                            bybit_success = True
                            print(f"[{symbol} {iv}m API] Order fill detected via WebSocket in {idx*0.5:.1f}s.")
                            break
                        if idx > 0 and idx % 6 == 0:
                            order_details = get_bybit_order_details(symbol, bybit_order_id)
                            if order_details:
                                status = order_details.get("orderStatus")
                                if status == "Filled":
                                    entry_price = float(order_details.get("avgPrice", limit_price))
                                    actual_qty = float(order_details.get("cumExecQty", raw_qty))
                                    filled = True
                                    bybit_success = True
                                    break
                                elif status in ["Cancelled", "Rejected"]:
                                    break
                    if filled:
                        print(f"[{symbol} {iv}m API] Success! Maker Limit Order filled at ${entry_price:.4f}.")
                        break
                    else:
                        print(f"[{symbol} {iv}m API] Order unfilled after 12s. Cancelling and re-quoting...")
                        cancel_bybit_order(symbol, bybit_order_id)
                        time.sleep(0.5)
                        final_details = get_bybit_order_details(symbol, bybit_order_id)
                        if final_details:
                            status = final_details.get("orderStatus")
                            cum_qty = float(final_details.get("cumExecQty", 0.0))
                            if status == "Filled" or cum_qty > 0:
                                entry_price = float(final_details.get("avgPrice", limit_price))
                                actual_qty = cum_qty if cum_qty > 0 else raw_qty
                                filled = True
                                bybit_success = True
                                print(f"[{symbol} {iv}m API] Success! Maker Limit Order filled/partially filled during cancel request at ${entry_price:.4f} (Qty: {actual_qty}).")
                                break
                else:
                    print(f"[{symbol} {iv}m API WARNING] Limit order placement failed: {order_res.get('retMsg')} (waiting 2s before retry)")
                    time.sleep(2)
                    
            # Fallback to Market order if limit chases failed (re-check WS fill cache first to prevent race condition)
            if not bybit_success:
                with _ws_filled_orders_lock:
                    if bybit_order_id and bybit_order_id in _ws_filled_orders:
                        bybit_success = True
                        print(f"[{symbol} {iv}m API] Order fill confirmed in WS cache. Skipping market order fallback.")
            
            if not bybit_success:
                atr_norm = float(latest_candle.get("ATR_norm", 0.0))
                if atr_norm >= 0.015:
                    print(f"[{symbol} {iv}m API BLOCK] Limit chases failed. Market fallback blocked due to extreme volatility (ATR_norm: {atr_norm:.4f} >= 0.015).")
                else:
                    print(f"[{symbol} {iv}m API] All Limit Maker chases failed. Falling back to Market order to guarantee entry...")
                    order_res = place_bybit_order(symbol, side, qty_str)
                    if order_res.get("retCode") == 0:
                        bybit_order_id = order_res.get("result", {}).get("orderId")
                        bybit_success = True
                        time.sleep(0.5)
                        order_details = get_bybit_order_details(symbol, bybit_order_id)
                        if order_details:
                            entry_price = float(order_details.get("avgPrice", entry_price))
                            actual_qty = float(order_details.get("cumExecQty", raw_qty))
                        else:
                            fill_exec = get_bybit_last_execution(symbol)
                            if fill_exec:
                                entry_price = float(fill_exec.get("execPrice", entry_price))
                            actual_qty = raw_qty
                            
        min_fill_pct = getattr(config, "MIN_ACCEPTABLE_FILL_PCT", 0.60)
        fill_ratio = (actual_qty / raw_qty) if (raw_qty > 0 and actual_qty > 0) else (1.0 if bybit_success else 0.0)
        if bybit_success and fill_ratio < min_fill_pct:
            log_event("WARNING", f"[{symbol} {iv}m API] Fill ratio {fill_ratio*100:.1f}% below {min_fill_pct*100:.0f}% threshold. Reversing partial fill...")
            if getattr(config, "RESIDUAL_ACTION", "CLOSE") == "CLOSE":
                opp_side = "Sell" if side == "Buy" else "Buy"
                place_bybit_taker_ioc_order(symbol, opp_side, format_bybit_qty(symbol, actual_qty))
                bybit_success = False

        if bybit_success:
            # F-7: Timeframe-Adaptive Minimum Stop Floor via config.MIN_SL_PCT_CONFIG
            min_sl_cfg = getattr(config, "MIN_SL_PCT_CONFIG", {})
            min_sl_pct = min_sl_cfg.get(str(iv), min_sl_cfg.get("default", 0.008)) * 100.0

            min_sl_dist = entry_price * (min_sl_pct / 100.0)
            atr_sl_dist = sl_multiplier_adjusted * atr_dollars
            
            # Preserve approved pre-flight stop loss if provided
            preflight_sl = stop_loss_price
            preflight_sl_dist = abs(entry_price - float(preflight_sl)) if preflight_sl is not None else 0.0

            if preflight_sl_dist > 0:
                raw_sl_dist = max(preflight_sl_dist, min_sl_dist)
                sl_source = "PREFLIGHT_APPROVED"
                sl_override_reason = "Pre-flight Approved SL Preserved"
            elif atr_sl_dist >= min_sl_dist:
                raw_sl_dist = atr_sl_dist
                sl_source = "ATR"
                sl_override_reason = "ATR Multiplier Applied"
            else:
                raw_sl_dist = min_sl_dist
                sl_source = "MIN_FLOOR"
                sl_override_reason = f"Minimum Risk Floor ({min_sl_pct:.2f}%) Triggered"
            
            # Institutional Meta Exit Policy & Unified Target Generator
            current_regime = "STRONG_TREND" if tp_multiplier_adjusted > 1.8 else "RANGING"
            policy_vec = generate_continuous_policy_vector(current_regime, confidence=calibrated_confidence)
            
            cand_sl = (entry_price - raw_sl_dist) if ml_trend == "Bullish" else (entry_price + raw_sl_dist)
            cand_tp_temp = (entry_price + 1.5 * tp_multiplier_adjusted * atr_dollars) if ml_trend == "Bullish" else (entry_price - 1.5 * tp_multiplier_adjusted * atr_dollars)
            
            boot_ci = calculate_probabilistic_utility_bootstrap(
                symbol=symbol,
                entry_price=entry_price,
                candidate_tp=cand_tp_temp,
                candidate_sl=cand_sl,
                direction=ml_trend,
                win_prob=max(0.50, min(0.90, calibrated_confidence)),
                loss_prob=1.0 - max(0.50, min(0.90, calibrated_confidence)),
                leverage=leverage_val,
                position_size_usd=position_size_usd
            )
            
            unified_res = UnifiedTargetGenerator.compute_targets(
                policy_vector=policy_vec,
                bootstrap_ci=boot_ci,
                entry_price=entry_price,
                direction=ml_trend,
                atr_dollars=atr_dollars,
                symbol=symbol,
                df_history=df_completed
            )
            
            stop_loss_price = cand_sl
            take_profit_price = unified_res["take_profit_price"]

            # F-2: Scale TP target with SL floor expansion & check horizon reachability
            if sl_source == "MIN_FLOOR":
                from config import TIMEFRAME_CONFIG
                import math
                cfg = TIMEFRAME_CONFIG.get(str(iv), {})
                target_rr = cfg.get("tp_mult_trending", 1.85) / max(1e-9, cfg.get("sl_mult", 0.8))
                final_sl_dist = abs(entry_price - stop_loss_price)
                new_tp_dist = final_sl_dist * target_rr
                
                lookahead = cfg.get("lookahead", 10)
                reach_factor = getattr(config, "HORIZON_REACHABILITY_FACTOR", 0.90)
                max_reachable = math.sqrt(lookahead) * atr_dollars * reach_factor
                
                if new_tp_dist > max_reachable:
                    log_event("WARNING", f"[{symbol} {iv}m F-2 Reachability Guard] Floor-scaled TP distance (${new_tp_dist:.4f}) exceeds horizon reach (${max_reachable:.4f}). Aborting trade entry.")
                    send_telegram_alert(f"⚠️ [{symbol} {iv}m] Trade aborted — floor-scaled TP target (${new_tp_dist:.4f}) exceeds horizon reach (${max_reachable:.4f}).")
                    opp_side = "Sell" if side == "Buy" else "Buy"
                    try:
                        place_bybit_taker_ioc_order(symbol, opp_side, format_bybit_qty(symbol, actual_qty))
                    except Exception as ex_flat:
                        log_event("ERROR", f"[{symbol} {iv}m Emergency Flatten Error] {ex_flat}")
                    return
                
                take_profit_price = (entry_price + new_tp_dist) if ml_trend == "Bullish" else (entry_price - new_tp_dist)
                is_floor_scaled = True
                log_event("INFO", f"[{symbol} {iv}m F-2 Target Scaling] Scaled TP to ${take_profit_price:.4f} (R:R {target_rr:.2f}) to match MIN_FLOOR SL (${stop_loss_price:.4f}).")
                
            from trade_calculators import passes_economic_gate, assert_valid_geometry
            try:
                assert_valid_geometry(ml_trend, entry_price, stop_loss_price, take_profit_price, symbol)
            except ValueError as geo_err:
                log_event("WARNING", f"[{symbol} {iv}m] Geometry assertion failed: {geo_err}. Aborting.")
                opp_side = "Sell" if side == "Buy" else "Buy"
                try:
                    place_bybit_taker_ioc_order(symbol, opp_side, format_bybit_qty(symbol, actual_qty))
                except Exception as ex_flat:
                    log_event("ERROR", f"[{symbol} {iv}m Emergency Flatten Error] {ex_flat}")
                return

            if not passes_economic_gate(entry=entry_price, tp=take_profit_price, sl=stop_loss_price, conf=calibrated_confidence):
                _sl_dist = abs(entry_price - stop_loss_price)
                _tp_dist = abs(take_profit_price - entry_price)
                _sl_frac = _sl_dist / max(1e-9, entry_price)
                _tp_frac = _tp_dist / max(1e-9, entry_price)
                _required_p = (_sl_frac + 0.0016) / max(1e-9, (_sl_frac + _tp_frac))
                log_event("WARNING", f"[{symbol} {iv}m] MIN_FLOOR widened SL to {min_sl_pct:.2f}%: R:R {_tp_dist/_sl_dist:.2f} requires {_required_p:.3f}, have {calibrated_confidence:.3f}. Aborting.")
                send_telegram_alert(f"⚠️ [{symbol} {iv}m] Trade aborted — post-floor R:R {_tp_dist/_sl_dist:.2f} (requires {_required_p:.3f}, have {calibrated_confidence:.3f})")
                opp_side = "Sell" if side == "Buy" else "Buy"
                try:
                    place_bybit_taker_ioc_order(symbol, opp_side, format_bybit_qty(symbol, actual_qty))
                except Exception as ex_flat:
                    log_event("ERROR", f"[{symbol} {iv}m Emergency Flatten Error] {ex_flat}")
                return

            # 4. Set SL/TP on active position on Bybit
            temp_trade = {"qty": str(actual_qty), "direction": ml_trend}
            update_bybit_stop_loss(symbol, stop_loss_price, active_trade=temp_trade)
            update_bybit_take_profit(symbol, take_profit_price, active_trade=temp_trade)
            
            # Place scale-out limit order on Bybit
            limit_side = "Sell" if ml_trend == "Bullish" else "Buy"
            limit_price = entry_price + 1.0 * atr_dollars if ml_trend == "Bullish" else entry_price - 1.0 * atr_dollars
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
            "sl_source": str(sl_source),
            "min_sl_pct": float(min_sl_pct),
            "atr_sl_dist": float(atr_sl_dist),
            "min_sl_dist": float(min_sl_dist),
            "initial_planned_rr": float(init_planned_rr),
            "direction": str(ml_trend),
            "end_time": float(time.time() + duration_seconds),
            "entry_time": int(time.time() * 1000),
            "atr_dollars": float(atr_dollars),
            "highest_price": float(entry_price),
            "lowest_price": float(entry_price),
            "swing_low_3b": float(df_completed["low"].tail(3).min()) if (df_completed is not None and not df_completed.empty and "low" in df_completed.columns) else float(entry_price),
            "swing_high_3b": float(df_completed["high"].tail(3).max()) if (df_completed is not None and not df_completed.empty and "high" in df_completed.columns) else float(entry_price),
            "break_even_triggered": False,
            "half_closed": False,
            "original_size": float(position_size_usd),
            "intended_size_usd": float(intended_size_usd if intended_size_usd is not None else position_size_usd),
            "position_size_usd": actual_size_usd,
            "scaled_out_pnl": 0.0,
            "kelly_fraction": float(kelly_fraction),
            "leverage": float(leverage_val),
            "confidence": float(calibrated_confidence),
            "qty": float(actual_qty),
            "original_qty": float(actual_qty),
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

def main():
    global live_price, last_ws_update_time
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
                last_ws_update_time = time.time()
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
        if live_price is None or (current_time - last_ws_update_time > 30.0):
            fallback_price = get_fallback_price()
            if fallback_price is not None:
                print(f"[{get_pkt_time().strftime('%H:%M:%S')}] WebSocket/Fallback price is stale or disconnected. Fetching price: {fallback_price:.2f}")
                live_price = fallback_price
                last_ws_update_time = current_time
            
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
            if all_open_trades:
                portfolio_rebal = PortfolioUtilityOptimizer.optimize_portfolio_capital(all_open_trades)

            for iv in ["15", "30", "60", "120", "240"]:
                tf = tf_map[iv]
                active_trade_key = f"active_trade_{tf}"
                active_trades_list = bot_state.get(active_trade_key, [])
                if not isinstance(active_trades_list, list):
                    active_trades_list = [] if active_trades_list is None else [active_trades_list]
                    bot_state[active_trade_key] = active_trades_list
                
                updated_trades = []
                for active_trade in active_trades_list:
                    active_symbol = active_trade.get("symbol", "BTCUSDT")
                    symbol_price = bot_state.get(f"live_price_{active_symbol}")
                    if symbol_price is None:
                        symbol_price = get_fallback_price(active_symbol)
                    if symbol_price is None:
                        updated_trades.append(active_trade)
                        continue
                    current_price = symbol_price
                    
                    stop_loss = active_trade["stop_loss"]
                    take_profit = active_trade["take_profit"]
                    direction = active_trade["direction"]
                    end_time = active_trade["end_time"]
                    entry_price = active_trade["entry_price"]
                    predicted_price = active_trade["predicted_price"]
    
                    # Bybit Live position query and state tracking
                    bybit_closed = False
                    bybit_scaled_out = False
                    bybit_exit_price = None
                    bybit_realized_pnl = None
                    bybit_pnl_data = None
                    
                    if TRADE_MODE != "simulation":
                        if active_trade.get("bybit_closed", False):
                            bybit_closed = True
                        else:
                            # Detect scale-out fill from cached and stored qty values
                            original_qty = active_trade.get("original_qty", active_trade.get("qty", 0.0))
                            current_qty = active_trade.get("qty", 0.0)
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
                                        if time.time() - stuck_since > 600:  # 10-minute timeout
                                            print(f"[{active_symbol}] Scale-out stuck >10 min ({status_msg}). Cancelling stale order and marking half_closed.")
                                            cancel_bybit_order(active_symbol, scale_out_order_id)
                                            active_trade["half_closed"] = True
                                            active_trade.pop("scale_out_stuck_since", None)
                                        else:
                                            print(f"[{active_symbol}] Size check indicates scale-out, but limit order status is not Filled ({status_msg}). Waiting.")
                                else:
                                    # Fallback if no scale-out order ID is attached: require price to have actually reached scale-out target
                                    atr_d = active_trade.get("atr_dollars", 0.015 * entry_price)
                                    reached_scale_target = False
                                    if direction == "Bullish" and active_trade.get("highest_price", entry_price) >= entry_price + 1.0 * atr_d:
                                        reached_scale_target = True
                                    elif direction == "Bearish" and active_trade.get("lowest_price", entry_price) <= entry_price - 1.0 * atr_d:
                                        reached_scale_target = True
                                        
                                    if reached_scale_target:
                                        bybit_scaled_out = True
                                    else:
                                        # Reset original_qty to current_qty to prevent continuous false scale-out triggers
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

                    min_pct_floor = risk_engine.auto_stop_floor.get_floor(active_symbol, database_module=database) if hasattr(risk_engine, 'auto_stop_floor') else volatility_clusterer.get_symbol_break_even_floor(active_symbol)
                    be_mult = mfe_be_trigger.get_trigger_multiple(active_symbol, timeframe=str(iv))
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
                    current_move = abs(current_price - entry_price)
                    progress_pct = (current_move / total_tp_range) if total_tp_range > 0 else 0.0
                    
                    fib_locks = getattr(config, "FIBONACCI_STEP_LOCKS", {0.618: 0.55, 0.50: 0.40, 0.382: 0.25})
                    locked_pct = 0.0
                    for threshold in sorted(fib_locks.keys(), reverse=True):
                        if progress_pct >= threshold:
                            locked_pct = fib_locks[threshold]
                            break
                        
                    if locked_pct > 0.0:
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

                    # Scale-Out (50% partial profit taking at 1.0 * ATR)
                    from config import TAKER_FEE_PCT, SCALE_OUT_CONFIG
                    scale_out_mult = SCALE_OUT_CONFIG.get("atr_trigger_mult", 1.0)
                    scale_out_portion = SCALE_OUT_CONFIG.get("position_portion", 0.50)
                    half_closed = active_trade.get("half_closed", False)
                    trigger_scale_out = False
                    if not half_closed:
                        if TRADE_MODE != "simulation":
                            trigger_scale_out = bybit_scaled_out
                        else:
                            if direction == "Bullish" and current_price >= entry_price + scale_out_mult * atr_dollars:
                                trigger_scale_out = True
                            elif direction == "Bearish" and current_price <= entry_price - scale_out_mult * atr_dollars:
                                trigger_scale_out = True
    
                    if trigger_scale_out and not half_closed:
                        if direction == "Bullish":
                            # Scale-Out Triggered for Long
                            half_closed = True
                            active_trade["half_closed"] = True
                            
                            # Close configured portion of the position
                            closed_size = round(position_size_usd * scale_out_portion, 2)
                            remaining_size = round(position_size_usd - closed_size, 2)
                            
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
                            
                            # Move stop loss to fee-adjusted break-even
                            target_sl = calculate_break_even_stop(direction, entry_price, current_price, atr_dollars)
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, target_sl, active_trade)
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
                            
                        elif direction == "Bearish":
                            # Scale-Out Triggered for Short
                            half_closed = True
                            active_trade["half_closed"] = True
                            
                            # Close configured portion of position
                            closed_size = round(position_size_usd * scale_out_portion, 2)
                            remaining_size = round(position_size_usd - closed_size, 2)
                            
                            # Calculate profit on closed portion (correct taker fee on leveraged size)
                            raw_return_pct = ((entry_price - current_price) / entry_price) * 100.0
                            lev = active_trade.get("leverage", 1.0)
                            gross_pnl = closed_size * (raw_return_pct * lev / 100.0)
                            taker_fee_cost = closed_size * lev * TAKER_FEE_PCT  # exit side only
                            pnl_usd = round(gross_pnl - taker_fee_cost, 2)
                            if pnl_usd < -closed_size:
                                pnl_usd = -closed_size
                                net_return_pct = -100.0
                            else:
                                net_return_pct = round((pnl_usd / closed_size) * 100.0, 2) if closed_size > 0 else 0.0
                            
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
                            
                            # Move stop loss to fee-adjusted break-even floor
                            target_sl = calculate_break_even_stop(direction, entry_price, current_price, atr_dollars)
                            
                            if TRADE_MODE != "simulation":
                                success = update_bybit_stop_loss(active_symbol, target_sl, active_trade)
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
                    entry_time_ms = active_trade.get("entry_time") or (current_time * 1000)
                    tf_mins = max(1, int(iv))
                    candles_elapsed = int((time.time() - (entry_time_ms / 1000.0)) / (tf_mins * 60))
                    
                    atr_dollars = active_trade.get("atr_dollars") or max(1e-6, entry_price * 0.01)
                    highest_p = active_trade.get("highest_price", current_price)
                    lowest_p = active_trade.get("lowest_price", current_price)
                    pnl_dist_mfe = (highest_p - entry_price) if direction == "Bullish" else (entry_price - lowest_p)
                    risk_dist_mfe = abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 1e-6 else atr_dollars
                    mfe_r = round(pnl_dist_mfe / risk_dist_mfe, 2)
                    
                    curr_regime = bot_state.get(f"regime_{iv}", "Trending") if "bot_state" in globals() and isinstance(bot_state, dict) else "Trending"
                    
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
                        entry_regime=str(active_trade.get("entry_regime", curr_regime)),
                        current_regime=str(curr_regime),
                        mhi_status=float(bot_state.get(f"mhi_{iv}", bot_state.get("mhi_score", 100.0))) if "bot_state" in globals() and isinstance(bot_state, dict) else 100.0
                    )
                    
                    if hierarchy_eval.get("should_exit"):
                        exit_reason = f"EXIT HIERARCHY LEVEL {hierarchy_eval.get('exit_level')}: {hierarchy_eval.get('exit_reason')}"
                        print(f"[{active_symbol} {iv}m Exit Hierarchy Triggered] Level {hierarchy_eval.get('exit_level')} -> {hierarchy_eval.get('exit_reason')} | Exit Score: {hierarchy_eval.get('exit_score')}")

                    
                    # 3. Simulation mode SL/TP price checks
                    if TRADE_MODE == "simulation" and not exit_reason:
                        if direction == "Bullish":
                            if current_price <= stop_loss:
                                exit_reason = "TRAILING STOP HIT [SUCCESS]" if half_closed else "STOP LOSS HIT [FAIL]"
                            elif current_price >= take_profit:
                                exit_reason = "TAKE PROFIT HIT [SUCCESS]"
                        else:
                            if current_price >= stop_loss:
                                exit_reason = "TRAILING STOP HIT [SUCCESS]" if half_closed else "STOP LOSS HIT [FAIL]"
                            elif current_price <= take_profit:
                                exit_reason = "TAKE PROFIT HIT [SUCCESS]"
                    
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
                                        bybit_closed = True
                                        time.sleep(0.5)

                        if bybit_closed:
                            entry_time_ms = active_trade.get("entry_time")
                            if not entry_time_ms:
                                entry_time_ms = int((end_time - (int(iv) * 60)) * 1000)
                            
                            bybit_pnl_data = get_bybit_accumulated_closed_pnl(active_symbol, entry_time_ms)
                            if bybit_pnl_data:
                                bybit_realized_pnl = bybit_pnl_data["total_pnl"]
                                if bybit_pnl_data["avg_exit_price"] is not None:
                                    bybit_exit_price = bybit_pnl_data["avg_exit_price"]
                                if bybit_pnl_data["total_entry_value"] is not None:
                                    lev = float(active_trade.get("leverage", 1.0))
                                    actual_margin = round(bybit_pnl_data["total_entry_value"] / lev, 2)
                                    active_trade["position_size_usd"] = actual_margin
                                    position_size_usd = actual_margin

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
                            else:
                                exit_reason = "STOP LOSS HIT [FAIL]"
    
                    is_exited = (exit_reason is not None) or bybit_closed
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
                        realized_pnl = gross_pnl - fee_cost
                        net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
                        
                        if realized_pnl < -position_size_usd:
                            realized_pnl = -position_size_usd
                            net_return_pct = -100.0
                        
                        # Log trade into KellyTracker for dynamic Quarter-Kelly sizing
                        global_kelly_tracker.log_trade(active_symbol, str(iv), realized_pnl, net_return_pct)
                        
                        # Aggregate PnL and size for trade history logging if scaled out
                        original_size = float(active_trade.get("original_size", position_size_usd))
                        scaled_out_pnl = float(active_trade.get("scaled_out_pnl", 0.0))
                        
                        if TRADE_MODE != "simulation" and bybit_realized_pnl is not None:
                            total_pnl = round(bybit_realized_pnl, 2)
                            realized_pnl = round(total_pnl - scaled_out_pnl, 2)
                            net_return_pct = (realized_pnl / position_size_usd) * 100.0 if position_size_usd > 0 else 0.0
                            total_net_return_pct = round((total_pnl / original_size) * 100.0, 4)
                        else:
                            total_pnl = round(realized_pnl + scaled_out_pnl, 2)
                            total_net_return_pct = round((total_pnl / original_size) * 100.0, 4)
                        
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
                        bot_state["trade_history"].append({
                            "symbol": active_symbol,
                            "exit_time": float(time.time()),
                            "interval": str(iv),
                            "direction": str(direction),
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
                            "bybit_scale_out_order_id": active_trade.get("bybit_scale_out_order_id")
                        })
                        # Log to trade journal CSV
                        log_trade_journal(bot_state["trade_history"][-1])
                        
                        # Build Scale-Out details block if trade was half-closed
                        scale_out_block = ""
                        if active_trade.get("half_closed", False):
                            stage1_price = float(active_trade.get("scale_out_price", entry_price))
                            stage1_margin = float(active_trade.get("scaled_out_margin", original_size / 2.0))
                            stage1_pnl = float(active_trade.get("scaled_out_pnl", 0.0))
                            
                            stage2_price = actual_price
                            stage2_margin = float(position_size_usd)
                            stage2_pnl = float(realized_pnl)
                            
                            stage2_name = "Trailing Stop Hit" if "TRAILING" in str(exit_reason).upper() else "Take Profit Hit" if "TAKE PROFIT" in str(exit_reason).upper() else "Final Exit"
                            
                            scale_out_block = (
                                f"\n\n🥞 *Scale-Out Execution Details*\n"
                                f"• *Stage 1: Partial Profit Locked (50% Scale-Out)*\n"
                                f"  - Target Price: `${stage1_price:.4f}`\n"
                                f"  - Returned Margin: `${stage1_margin:.2f}`\n"
                                f"  - PnL Realized: *${stage1_pnl:+.2f}*\n"
                                f"• *Stage 2: {stage2_name} (Remaining 50%)*\n"
                                f"  - Exit Price: `${stage2_price:.4f}`\n"
                                f"  - Returned Margin: `${stage2_margin:.2f}`\n"
                                f"  - PnL Realized: *${stage2_pnl:+.2f}*"
                            )

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
                bot_state[active_trade_key] = updated_trades
            
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

        # --- Consecutive Losses Cooldown Circuit Breaker ---
        def is_symbol_interval_cooling_off(symbol, interval):
            """
            Checks if a symbol and interval combination is in a 6-hour cool-off period
            after suffering 2 consecutive loss trades.
            """
            trades = [t for t in bot_state.get("trade_history", []) if t.get("symbol") == symbol and str(t.get("interval")) == str(interval)]
            if len(trades) < 2:
                return False, 0
                
            sorted_trades = sorted(trades, key=lambda x: float(x.get("exit_time", 0.0) or 0.0), reverse=True)
            
            latest_trade = sorted_trades[0]
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

            is_latest_loss = _is_loss(latest_trade)
            is_second_loss = _is_loss(second_latest)
            
            if is_latest_loss and is_second_loss:
                exit_time = float(latest_trade.get("exit_time", 0.0) or 0.0)
                cooldown_duration = 6 * 3600  # 6 hours
                time_elapsed = time.time() - exit_time
                if time_elapsed < cooldown_duration:
                    remaining_minutes = int((cooldown_duration - time_elapsed) / 60)
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
            except Exception:
                pass
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
            from concurrent.futures import ThreadPoolExecutor
            
            # Fetch BTCUSDT history once per interval to cache and share among workers (avoids rate limits and duplicate REST calls)
            btc_hist_cache = {}
            unique_intervals = set(iv for sym, iv in check_queue)
            for iv_val in unique_intervals:
                df_btc = get_history(symbol="BTCUSDT", interval=iv_val, limit=300)
                btc_hist_cache[iv_val] = df_btc
            
            def fetch_single_history(sym, interval_val):
                if sym == "BTCUSDT" and interval_val in btc_hist_cache:
                    df_raw_val = btc_hist_cache[interval_val]
                else:
                    df_raw_val = get_history(symbol=sym, interval=interval_val, limit=300)
                if df_raw_val is None or len(df_raw_val) < 2:
                    return sym, interval_val, None, None
                
                df_completed_val = df_raw_val.iloc[:-1].copy()
                latest_completed_ts_val = int(df_completed_val.iloc[-1]["timestamp"])
                
                last_ts_key_val = f"last_processed_{sym}_{interval_val}_ts"
                if last_processed_timestamps.get(last_ts_key_val) is not None:
                    if latest_completed_ts_val == last_processed_timestamps[last_ts_key_val]:
                        return sym, interval_val, df_raw_val, None
                
                # Fast check if candle is up to date before running heavy feature calculation
                interval_ms_val = int(interval_val) * 60 * 1000
                expected_start_ms_val = current_hour_ts - interval_ms_val
                is_forced_val = interval_val in forced_intervals
                is_up_to_date_val = (latest_completed_ts_val >= expected_start_ms_val) or is_startup or is_forced_val
                if not is_up_to_date_val:
                    return sym, interval_val, df_raw_val, None
                
                df_target_val = df_completed_val.copy()
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
                
                df_target_val = merge_derivatives_sentiment_features(df_target_val, symbol=sym, interval=interval_val)
                df_feat_val = add_features(df_target_val)
                
                return sym, interval_val, df_raw_val, df_feat_val
 
            print(f"[Parallel Fetch] Querying {len(check_queue)} candle combinations in parallel...")
            t_start = time.time()
            with ThreadPoolExecutor(max_workers=6) as executor:
                future_to_pair = {executor.submit(fetch_single_history, sym, iv): (sym, iv) for sym, iv in check_queue}
                for fut in future_to_pair:
                    sym, iv = future_to_pair[fut]
                    try:
                        _, _, df_raw_val, df_feat_val = fut.result(timeout=60)  # HTTP timeout is 10s per request
                        if df_raw_val is not None:
                            fetched_data[(sym, iv)] = (df_raw_val, df_feat_val)
                    except Exception as e:
                        print(f"[Parallel Fetch] Error fetching {sym} {iv}: {type(e).__name__} {e}")
            print(f"[Parallel Fetch] Completed in {time.time() - t_start:.2f} seconds.")
 
        # Update peak_balance dynamically for accurate real-time drawdown tracking (M-1)
        try:
            curr_equity = float(bot_state.get("simulated_balance", 80.0))
            if TRADE_MODE != "simulation":
                real_bal_val = get_real_bybit_balance_cached()
                if isinstance(real_bal_val, (int, float)) and real_bal_val > 0:
                    curr_equity = float(real_bal_val)
            if curr_equity > float(bot_state.get("peak_balance", 0.0)):
                bot_state["peak_balance"] = curr_equity
        except Exception:
            pass

        # Circuit Breaker Halt Guard (Micro-Trading Run Scoped)
        try:
            micro_ts_str = database.get_setting("micro_run_start_ts", "0")
            micro_ts = float(micro_ts_str) if micro_ts_str else 0.0
            if micro_ts == 0.0:
                micro_ts = time.time()
                database.set_setting("micro_run_start_ts", str(micro_ts))
            
            closed_all = database.get_completed_trades(limit=1000) if hasattr(database, "get_completed_trades") else []
            micro_trades = [t for t in closed_all if float(t.get("exit_time", 0)) >= micro_ts]
            closed_trade_count = len(micro_trades)
            cumulative_loss = -sum(float(t.get("venue_closed_pnl", t.get("pnl_usd", 0.0))) for t in micro_trades if float(t.get("venue_closed_pnl", t.get("pnl_usd", 0.0))) < 0)
            
            max_trades_cap = getattr(config, "MAX_LIVE_TRADES_CAP", 60)
            max_loss_cap = getattr(config, "MAX_LIVE_LOSS_CAP", 15.0)
            
            if closed_trade_count >= max_trades_cap or cumulative_loss >= max_loss_cap:
                bot_state["bot_running"] = False
                bot_state["bot_stopped"] = True
                database.set_setting("bot_running", "False")
                database.set_setting("bot_stopped", "True")
                log_event("WARNING", f"[TRADING_LOOP] Hard Circuit Breaker Triggered (Micro Trades: {closed_trade_count}/{max_trades_cap}, Micro Loss: ${cumulative_loss:.2f}/${max_loss_cap:.2f}) — bot halted.")
                print(f"[TRADING_LOOP] Hard Circuit Breaker Triggered (Micro Trades: {closed_trade_count}/{max_trades_cap}, Micro Loss: ${cumulative_loss:.2f}/${max_loss_cap:.2f}) — bot halted.")
                return
        except Exception as ex_cb:
            log_event("WARNING", f"[TRADING_LOOP] Circuit breaker check exception: {ex_cb}")

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
                if not isinstance(active_trades_list, list):
                    active_trades_list = [] if active_trades_list is None else [active_trades_list]
                active_trades_list = [t for t in active_trades_list if isinstance(t, dict) and not t.get("bybit_closed") and not t.get("closed")]
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
 
                # Validate if candle is up to date based on expected window boundary
                interval_ms = int(iv) * 60 * 1000
                expected_start_ms = current_hour_ts - interval_ms
                is_forced = iv in forced_intervals
                is_up_to_date = (latest_completed_ts >= expected_start_ms) or is_startup or is_forced
                
                if not is_up_to_date:
                    # Candle is stale, wait for exchange to finalize the new candle
                    continue
                    
                completed_this_hour.add((symbol, iv))
                
                if latest_completed_ts != last_processed_timestamps[last_ts_key]:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New completed {symbol} {iv}-minute candle detected (TS: {latest_completed_ts})")
                    log_event("INFO", f"[{symbol} {iv}m] Evaluation start — candle {latest_completed_ts}")
                    rec = DecisionRecord(symbol=symbol, interval=str(iv))
                    rec.candle_timestamp = latest_completed_ts
                    status_msg = "Abstain"
                    placed = False
                    try:
                    
                        latest_candle = df.iloc[-1]
                    
                        # Dynamic Regime Routing based on GMM Unsupervised Classifier
                        regime = classify_market_regime(df, interval=iv)
                        adx_regime = latest_candle["ADX"]
                    
                        # Ensure models for interval iv are loaded into memory on-demand
                        if models_by_interval.get(iv, {}).get("trending", {}).get("trend") is None:
                            load_model_weights(iv)

                        if iv in models_by_interval:
                            models_tf = models_by_interval[iv]
                            regime_key = regime.lower() if regime in ["Trending", "Ranging"] else "trending"
                            m_price = models_tf.get(regime_key, {}).get("price")
                            m_trend = models_tf.get(regime_key, {}).get("trend")
                            m_cal = models_tf.get(regime_key, {}).get("calibrator")
                            m_meta = models_tf.get(regime_key, {}).get("meta")
                            feat_list = models_tf.get(f"selected_features_{regime_key}") or models_tf.get("selected_features")
                            served_regime = regime

                            if m_price is None or m_trend is None or not feat_list:
                                alt_key = "ranging" if regime_key == "trending" else "trending"
                                alt_price = models_tf.get(alt_key, {}).get("price")
                                alt_trend = models_tf.get(alt_key, {}).get("trend")
                                alt_cal = models_tf.get(alt_key, {}).get("calibrator")
                                alt_meta = models_tf.get(alt_key, {}).get("meta")
                                alt_feats = models_tf.get(f"selected_features_{alt_key}") or models_tf.get("selected_features")
                                if alt_price is not None and alt_trend is not None and alt_feats:
                                    log_event("WARNING", f"[{symbol} {iv}m] {regime} model incomplete — falling back to {alt_key.title()} ensemble & feature contract")
                                    m_price, m_trend, m_cal, m_meta, feat_list = alt_price, alt_trend, alt_cal, alt_meta, alt_feats
                                    served_regime = f"{alt_key.title()}_Fallback"
                                else:
                                    log_event("CRITICAL", f"[{symbol} {iv}m] {regime} model & fallback unavailable — skipping interval (Fail-Closed)")
                                    continue

                            active_model_price = m_price
                            active_model_trend = m_trend
                            active_calibrator = m_cal
                            active_meta_model = m_meta
                            regime_name = f"{served_regime} (GMM)"
                            
                            # C-1: Preserve strict train/serve feature distribution consistency
                            # Remove ad-hoc inference-time feature multiplier scaling
                            latest_candle_weighted = latest_candle.copy()
                            if hasattr(latest_candle_weighted.index, "duplicated") and latest_candle_weighted.index.duplicated().any():
                                latest_candle_weighted = latest_candle_weighted[~latest_candle_weighted.index.duplicated(keep="first")]

                            from ensemble import get_model_feature_names
                            _exp_names = get_model_feature_names(active_model_trend)
                            if _exp_names and not all(str(n).startswith("Column_") for n in _exp_names):
                                _features_to_use = _exp_names
                                if isinstance(latest_candle_weighted, pd.Series):
                                    X_live_full = latest_candle_weighted.to_frame().T.reindex(columns=_exp_names, fill_value=0.0)
                                else:
                                    X_live_full = latest_candle_weighted.reindex(columns=_exp_names, fill_value=0.0)
                            elif feat_list is not None:
                                _features_to_use = feat_list
                                _avail = [f for f in _features_to_use if f in latest_candle_weighted.index]
                                X_live_full = latest_candle_weighted[_avail].to_frame().T if isinstance(latest_candle_weighted[_avail], pd.Series) else latest_candle_weighted[_avail]
                            else:
                                from core import features as master_features
                                _features_to_use = master_features
                                _avail = [f for f in _features_to_use if f in latest_candle_weighted.index]
                                X_live_full = latest_candle_weighted[_avail].to_frame().T if isinstance(latest_candle_weighted[_avail], pd.Series) else latest_candle_weighted[_avail]

                            X_live = _slice_model_input(active_model_trend, X_live_full)

                            # Item A: Interval-Specific Ensemble Weights (LightGBM & CatBoost-heavy for 15M/30M scalp accuracy)
                            if str(iv) == "15":
                                ensemble_weights = [0.10, 0.45, 0.45]
                            elif str(iv) == "30":
                                ensemble_weights = [0.15, 0.42, 0.43]
                            else:
                                ensemble_weights = [0.30, 0.20, 0.50] if "Trending" in regime_name else [0.30, 0.50, 0.20]
                        
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
                            except Exception as pred_err:
                                print(f"[{symbol} {iv}m CRITICAL PREDICTION ERROR] Model prediction exception: {pred_err}. Aborting trade entry (Fail-Closed).")
                                status_msg = "Skipped (Prediction Error)"
                                all_pass = False
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
                                prob_bearish = float(probs[0]) if float(probs[0]) < 0.5 else 0.0
                                prob_neutral = 0.0
                                prob_bullish = float(probs[0]) if float(probs[0]) >= 0.5 else 0.0

                            _p = {"Bearish": prob_bearish, "Neutral": prob_neutral, "Bullish": prob_bullish}
                            ml_trend = max(_p, key=_p.get)
                            ml_confidence = _p[ml_trend]
                            neutral_coeff = getattr(config, "NEUTRAL_PENALTY_COEFFICIENT", 0.0)
                            if ml_trend in ("Bullish", "Bearish") and neutral_coeff > 0.0:
                                ml_confidence = min(0.95, ml_confidence * (1.0 - prob_neutral * neutral_coeff))

                            # Apply Isotonic Regression probability calibration if available
                            calibrated_confidence = ml_confidence
                            calibrator = active_calibrator
                            if calibrator is not None and "X" in calibrator and "y" in calibrator and ml_trend in ["Bullish", "Bearish"]:
                                calibrated_confidence = float(np.interp(ml_confidence, calibrator["X"], calibrator["y"]))
                                print(f"[{symbol} {iv}m Isotonic Calibration] Raw: {ml_confidence*100:.2f}% -> Pure Calibrated: {calibrated_confidence*100:.2f}%")
                            else:
                                calibrated_confidence = ml_confidence

                            # Clip calibrated output away from 0.0 & 1.0 saturation boundaries (EPS = 1e-3)
                            calibrated_confidence = float(np.clip(calibrated_confidence, 1e-3, 1.0 - 1e-3))


                            
                            # Item D: Exponential Time-Decayed Cross-Interval Penalty applied to THRESHOLD GATE (Fix Recommendation #8)
                            htf_decay_threshold_penalty = 0.0
                            if str(iv) == "15" and ml_trend in ["Bullish", "Bearish"]:
                                pred_30m_dict = bot_state.get(f"latest_prediction_{symbol}_30m") or bot_state.get("latest_prediction_30m") or {}
                                pred_60m_dict = bot_state.get(f"latest_prediction_{symbol}_1h") or bot_state.get("latest_prediction_1h") or {}
                            
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
                                    print(f"[{symbol} 15m Time-Decayed Penalty] HTF contradiction gate penalty (+{htf_decay_threshold_penalty*100:.1f}% required threshold). Calibrated conf preserved -> {calibrated_confidence*100:.2f}%")

                            expected_pct_change = (abs(pred_change) / latest_candle["close"]) * 100

                            # Update global state prediction metrics for this timeframe
                            pred_entry_dict = {
                                "predicted_change": pred_change,
                                "predicted_price": predicted_price,
                                "direction": ml_trend,
                                "raw_confidence": ml_confidence,
                                "calibrated_confidence": calibrated_confidence,
                                "signal_source": "ML_ENSEMBLE",
                                "is_fallback": False
                            }
                            bot_state[f"regime_{symbol}_{tf}"] = regime_name
                            bot_state[f"adx_{symbol}_{tf}"] = adx_regime
                            bot_state[f"regime_{tf}"] = regime_name
                            bot_state[f"adx_{tf}"] = adx_regime
                            bot_state[f"latest_prediction_{symbol}_{tf}"] = pred_entry_dict
                            bot_state[f"latest_prediction_{tf}"] = pred_entry_dict

                            rec.regime = str(regime_name)
                            rec.direction = str(ml_trend)
                            rec.raw_conf = float(ml_confidence)
                            rec.calibrated_conf = float(calibrated_confidence)
                            rec.signal_source = str(pred_entry_dict.get("signal_source") or "UNSET")
                            rec.is_fallback = int(pred_entry_dict.get("is_fallback", False))
                            rec.directional_mass_passed = 1 if str(iv) not in ["15", "30"] or ml_trend in ["Bullish", "Bearish"] else 0
                            rec.is_calibrated = 1 if (active_calibrator is not None and ml_trend in ["Bullish", "Bearish"]) else 0
                            rec.is_floor_scaled = 1 if (locals().get('is_floor_scaled', False)) else 0
                            rec._inputs["predicted_change"] = float(pred_change)
                            rec._inputs["expected_pct_change"] = float(pred_change)

                            log_event("INFO", f"[{symbol} {iv}m] probs=[{prob_bearish:.4f}, {prob_neutral:.4f}, {prob_bullish:.4f}] ml_conf={ml_confidence:.4f} cal={calibrated_confidence:.4f}")

                            print(f"[{iv}m] Regime Selected: {regime_name} | ML Output: {ml_trend} (Bull: {prob_bullish*100:.1f}%, Bear: {prob_bearish*100:.1f}%, Neut: {prob_neutral*100:.1f}%) | Raw Conf: {ml_confidence*100:.2f}% | Calibrated Conf: {calibrated_confidence*100:.2f}% | Expected Change: {pred_change:+.3f}")

                            # Determine dynamic confidence threshold based on trade economics (p* break-even payoff) + bounded modifiers
                            atr_norm_val = latest_candle["ATR_norm"]
                        
                            # 1. Economic Base Threshold (p* break-even payoff threshold + transaction costs)
                            from config import TIMEFRAME_CONFIG
                            from trade_calculators import transaction_cost_model, UnifiedTargetGenerator
                            cfg = TIMEFRAME_CONFIG.get(str(iv), {})
                            base_tp_m = cfg.get("tp_mult_trending", 1.85)
                            base_sl_m = cfg.get("sl_mult", 0.8)
                            atr_dollars = float(latest_candle.get("ATR", latest_candle["close"] * 0.01))
                            entry_close = float(latest_candle["close"])
                            actual_tp_m = UnifiedTargetGenerator.resolve_tp_multiplier(
                                interval=str(iv), entry_price=entry_close,
                                atr_dollars=atr_dollars, base_tp_m=base_tp_m
                            )
                            _bars_per_day = max(1, round(1440 / max(1, int(iv))))
                            _adv_usd = float(df_completed["volume"].tail(_bars_per_day).sum() * entry_close) if ("volume" in df_completed.columns and len(df_completed) >= _bars_per_day) else 50_000_000.0
                            _order_usd = float(bot_state.get("position_size_usd", 1000.0))
                            _tcm = transaction_cost_model.estimate_transaction_cost(
                                order_size_usd=_order_usd,
                                volume_24h_usd=_adv_usd,
                                is_maker=True,
                            )
                            cost_bps = _tcm["total_cost_bps"] * 2.0  # round-trip
                            p_star = base_sl_m / (actual_tp_m + base_sl_m)
                            economic_base_threshold = float(round(p_star + (cost_bps / 1e4) / (actual_tp_m + base_sl_m), 4))
                        
                            dynamic_conf_threshold = economic_base_threshold
                            adjustments_applied = [("economic_base", economic_base_threshold)]

                            # 2. Bounded Regime & Volatility Adjustments
                            if "Ranging" in regime_name:
                                regime_delta = 0.03 if str(iv) in ["15", "30"] else 0.05
                                dynamic_conf_threshold += regime_delta
                                adjustments_applied.append(("regime_ranging", regime_delta))
                            
                            # High Volatility Adjustment (ATR > 0.015)
                            if atr_norm_val > 0.015:
                                vol_delta = 0.05
                                dynamic_conf_threshold += vol_delta
                                adjustments_applied.append(("high_volatility", vol_delta))
                                
                            if htf_decay_threshold_penalty > 0:
                                dynamic_conf_threshold += htf_decay_threshold_penalty
                                adjustments_applied.append(("htf_decay_penalty", htf_decay_threshold_penalty))
                            
                            # Recent 50-Trade Performance Decay Filter
                            recent_trades = bot_state.get("trade_history", [])[-50:]
                            if len(recent_trades) >= 10:
                                win_count = sum(1 for t in recent_trades if float(t.get("pnl_usd", 0.0)) > 0)
                                recent_win_rate = (win_count / len(recent_trades)) * 100.0
                                if recent_win_rate < 45.0:
                                    perf_delta = 0.10
                                    dynamic_conf_threshold += perf_delta
                                    adjustments_applied.append(("performance_decay", perf_delta))
                                    print(f"[{symbol} {iv}m Performance Decay Filter] Win rate {recent_win_rate:.1f}% < 45%. Raised threshold by +0.10 to {dynamic_conf_threshold:.2f}")
                            


                            # Asian Market Session Awareness (00:00 - 08:00 UTC)
                            utc_hour_now = datetime.now(timezone.utc).hour
                            if 0 <= utc_hour_now < 8:
                                dynamic_conf_threshold += 0.05
                                adjustments_applied.append(("asian_session", 0.05))
                                print(f"[{symbol} {iv}m Asian Session] UTC hour {utc_hour_now:02d}:00 in low-volatility Asian window (+0.05 threshold -> {dynamic_conf_threshold:.2f})")
                                
                            # Adaptive Confidence Threshold Matrix for 15m
                            if str(iv) == "15":
                                drift_p = bot_state.get("drift_p_val", 0.50) if "bot_state" in globals() and isinstance(bot_state, dict) else 0.50
                                u_tot = float(bot_state.get("u_total", 0.04)) if "bot_state" in globals() and isinstance(bot_state, dict) else 0.04
                                sym_sharpe = float(bot_state.get("symbol_sharpe", 1.2)) if "bot_state" in globals() and isinstance(bot_state, dict) else 1.2
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
                                    macro_pred = bot_state.get(f"latest_prediction_{symbol}_{macro_tf}") or bot_state.get(f"latest_prediction_{macro_tf}")
                                    ml_trend_dir = "Neutral"
                                    ml_prob = 0.0
                                    model_age_days = 0

                                    if macro_pred and isinstance(macro_pred, dict):
                                        ml_trend_dir = macro_pred.get("direction", "Neutral")
                                        ml_prob = float(macro_pred.get("calibrated_confidence") or macro_pred.get("confidence") or 0.0)
                                        model_age_days = int(macro_pred.get("model_age_days") or 0)

                                    htf_meta["ml_prediction"] = ml_trend_dir
                                    htf_meta["ml_probability"] = ml_prob
                                    htf_meta["model_age_days"] = model_age_days

                                    # STEP 1: Check ML Model Freshness & Learned Confidence
                                    if ml_trend_dir in ["Bullish", "Bearish"] and ml_prob >= learned_threshold and model_age_days < 45:
                                        htf_trend = ml_trend_dir
                                        htf_meta["trend_source"] = "ML_MODEL"
                                        htf_meta["fallback_reason"] = "NONE"
                                    else:
                                        fallback_reason = "MODEL_NEUTRAL" if ml_trend_dir == "Neutral" else ("LOW_CONFIDENCE" if ml_prob < learned_threshold else "MODEL_STALE")
                                        htf_meta["fallback_reason"] = fallback_reason

                                        # STEP 2: EMA9 vs EMA21 + EMA21 Slope > 0 Technical Fallback
                                        try:
                                            from ta.trend import EMAIndicator, ADXIndicator, SMAIndicator
                                            htf_df = get_history(symbol=symbol, interval=str(macro_iv), limit=60)
                                            if htf_df is not None and len(htf_df) >= 50:
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

                            # Funding Rate Carry Overlay
                            funding_rate = get_funding_rate(symbol)
                            funding_blocked = False
                            if funding_rate > 0.0005 and ml_trend == "Bullish":
                                dynamic_conf_threshold += 0.05
                                adjustments_applied.append(("funding_carry_long", 0.05))
                                print(f"[{symbol} {iv}m] Funding Carry Adjustment: Positive funding rate ({funding_rate*100:.3f}%) raised Long threshold to {dynamic_conf_threshold*100:.1f}%")
                            elif funding_rate < -0.0005 and ml_trend == "Bearish":
                                dynamic_conf_threshold += 0.05
                                adjustments_applied.append(("funding_carry_short", 0.05))
                                print(f"[{symbol} {iv}m] Funding Carry Adjustment: Negative funding rate ({funding_rate*100:.3f}%) raised Short threshold to {dynamic_conf_threshold*100:.1f}%")

                            if ml_trend == "Bullish" and funding_rate > 0.001:
                                funding_blocked = True
                            elif ml_trend == "Bearish" and funding_rate < -0.001:
                                funding_blocked = True
                            
                            # Open Interest Momentum Guard
                            try:
                                oi_delta = df.iloc[-1].get("open_interest_pct_change", 0.0) * 100.0
                                if oi_delta < 0.5:
                                    dynamic_conf_threshold += 0.05
                                    adjustments_applied.append(("oi_momentum_guard", 0.05))
                                    print(f"[{symbol} {iv}m] OI Momentum Guard: Low Open Interest Delta ({oi_delta:+.2f}%) raised threshold to {dynamic_conf_threshold*100:.1f}%")
                            except Exception as e:
                                print(f"[{symbol} {iv}m] Exception in OI Momentum Guard: {e}")

                            # Bound final threshold relative to economic base
                            from config import MAX_THRESHOLD_UPLIFT
                            max_allowed_threshold = min(0.85, economic_base_threshold + MAX_THRESHOLD_UPLIFT)
                            dynamic_conf_threshold = float(round(min(max_allowed_threshold, max(0.40, dynamic_conf_threshold)), 4))
                        
                            # Log threshold lineage to prediction state
                            if f"latest_prediction_{tf}" in bot_state and isinstance(bot_state[f"latest_prediction_{tf}"], dict):
                                bot_state[f"latest_prediction_{tf}"]["threshold_base"] = economic_base_threshold
                                bot_state[f"latest_prediction_{tf}"]["threshold_adjustments"] = adjustments_applied
                                bot_state[f"latest_prediction_{tf}"]["dynamic_threshold"] = dynamic_conf_threshold
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

                            flash_crash_active = check_flash_crash(symbol, max_drop_pct=3.0, window_minutes=5) if str(iv) in ["15", "30"] else False
                            liq_score = get_liquidity_score(symbol)
                            low_liquidity = (str(iv) in ["15", "30"] and liq_score < 0.3)

                            if not bot_state.get("bot_running", True):
                                status_msg = "Skipped (Bot Stopped)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Bot is currently stopped by the user.")
                            elif bot_state.get("circuit_breaker_active", False):
                                status_msg = "Skipped (Circuit Breaker)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Daily Drawdown Circuit Breaker is active.")
                            elif flash_crash_active:
                                status_msg = "Skipped (Flash Crash Block)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Flash crash detected (>3.0% drop in last 5 minutes).")
                            elif low_liquidity:
                                status_msg = "Skipped (Low Liquidity)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Insufficient L2 orderbook liquidity (Score: {liq_score:.2f} < 0.30).")
                            elif already_active:
                                status_msg = "Skipped (Already Active)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: A trade is already active for this symbol on the {active_on_tf} timeframe.")
                            elif not in_session:
                                status_msg = "Skipped (Off-Session)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Outside London/NY session (UTC hour: {utc_hour}).")
                            elif is_cooling:
                                status_msg = "Skipped (Cool-Off)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Interval is in a 6-hour cool-off period after consecutive losses ({remaining_mins} mins remaining).")
                            elif confluence_blocked:
                                status_msg = "Skipped (HTF Trend Block)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Counter-trend relative to macro timeframe ({macro_tf} trend: {htf_trend}).")
                            elif funding_blocked:
                                status_msg = "Skipped (Funding Block)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: High funding fee payment risk (Funding: {funding_rate*100:.3f}%).")
                            elif ml_trend == "Neutral":
                                status_msg = "Skipped (Neutral)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Model output is Neutral/Hold.")
                            elif strong_conflict:
                                status_msg = "Skipped (Contradiction)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: Strong directional contradiction (Trend: {ml_trend}, Regressor: {pred_change:+.3f} [{pred_pct:.3f}%]).")
                            elif calibrated_confidence < dynamic_conf_threshold:
                                status_msg = "Skipped (Low Confidence)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped (calibrated confidence {calibrated_confidence*100:.2f}% < {dynamic_conf_threshold*100:.2f}%).")
                            elif conformal_is_uncertain and ml_trend in ["Bullish", "Bearish"]:
                                status_msg = "Skipped (High Conformal Uncertainty)"
                                log_event("WARNING", f"[{symbol} {iv}m] Prediction skipped: High ensemble disagreement / conformal uncertainty score ({conformal_unc_score:.3f}).")

                            if status_msg == "Pending":
                                # Refinements 2, 8, 9, 10: 15m Institutional Hardening Filters
                                if str(iv) == "15":
                                    # Refinement 2: Liquidity & Volatility Compression Filter
                                    vol_series = pd.to_numeric(df["volume"], errors="coerce") if (df is not None and "volume" in df.columns) else None
                                    vol_20th = float(vol_series.quantile(0.20)) if (vol_series is not None and len(vol_series.dropna()) >= 20) else 0.0
                                    curr_vol = float(latest_candle.get("volume", 0.0))
                                    mean_atr_24h = float(df["ATR_norm"].mean()) if (df is not None and "ATR_norm" in df.columns and len(df) >= 20) else atr_norm_val
                                    current_spread_bps = float(bot_state.get("current_spread_bps", 3.5)) if "bot_state" in globals() and isinstance(bot_state, dict) else 3.5
                                    u_tot_live = float(bot_state.get("u_total", 0.04)) if "bot_state" in globals() and isinstance(bot_state, dict) else 0.04
                                    exp_r_val = abs(float(expected_pct_change)) / max(1e-4, atr_norm_val)

                                    # C-2: estimate per-symbol 24h ADV from candle volume * price
                                    _adv_usd = float(df["volume"].tail(96).sum() * df["close"].iloc[-1]) if (df is not None and "volume" in df.columns and "close" in df.columns and len(df) >= 10) else 50_000_000.0
                                    _order_usd = float(bot_state.get("position_size_usd", 1000.0)) if "bot_state" in globals() and isinstance(bot_state, dict) else 1000.0
                                    tcm_cost_bps = transaction_cost_model.estimate_transaction_cost(order_size_usd=_order_usd, volume_24h_usd=_adv_usd, is_maker=True).get("total_cost_bps", 5.0)
                                    exp_edge_bps = abs(float(expected_pct_change)) * 100.0 - tcm_cost_bps

                                    if curr_vol < vol_20th and vol_20th > 0:
                                        status_msg = "Skipped (Volume Compression <20th Pct)"
                                        print(f"[{symbol} 15m Filter] Trade skipped: Volume ({curr_vol:.1f}) < 20th percentile ({vol_20th:.1f}).")
                                    elif atr_norm_val > (1.5 * mean_atr_24h):
                                        status_msg = "Skipped (ATR Spike >1.5x Mean)"
                                        print(f"[{symbol} 15m Filter] Trade skipped: ATR spike ({atr_norm_val*100:.2f}%) > 1.5x 24h mean ({mean_atr_24h*100:.2f}%).")
                                    elif current_spread_bps > 4.5:
                                        status_msg = "Skipped (Spread Widening >4.5 bps)"
                                        print(f"[{symbol} 15m Filter] Trade skipped: Spread ({current_spread_bps:.1f} bps) exceeds 4.5 bps limit.")
                                    elif exp_r_val < 1.0:
                                        status_msg = "Skipped (Expected R < 1.0R)"
                                        print(f"[{symbol} 15m Filter] Trade skipped: Expected R ({exp_r_val:.2f}R) < 1.00R floor.")
                                    elif exp_edge_bps <= 0:
                                        status_msg = "Skipped (TCM Net Edge <= 0)"
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
                                
                                with news_sentiment_lock:
                                    news_sentiment = cached_news_sentiment
                                    latest_titles = cached_news_titles
                                    all_pass, confluence_results, confluence_score_pct = check_pre_trade_confluence(
                                        latest_candle["close"], df, ml_trend, news_sentiment, expected_pct_change, iv, symbol=symbol, htf_cache=htf_cache,
                                        calibrated_confidence=calibrated_confidence, dynamic_conf_threshold=dynamic_conf_threshold, get_history_fn=get_history,
                                        get_orderbook_fn=get_orderbook_imbalance, choppiness_fn=choppiness_index, bot_state_dict=bot_state
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
                                    
                                        atr_norm_val = latest_candle["ATR_norm"]
                                        atr_dollars = atr_norm_val * latest_candle["close"]
                                    
                                        # Align stop loss and take profit multipliers dynamically from baseline TIMEFRAME_CONFIG or release-gate approved walk-forward state
                                        optimized_cfg = bot_state.get("optimized_timeframe_config", {}).get(str(iv), {}) if "bot_state" in globals() and hasattr(bot_state, "get") else {}
                                        baseline_cfg = TIMEFRAME_CONFIG.get(str(iv), {
                                            "lookahead": 10,
                                            "sl_mult": 0.8,
                                            "tp_mult_ranging": 1.45,
                                            "tp_mult_trending": 1.75
                                        })
                                        # Merge approved optimizations over baseline defaults
                                        cfg = {**baseline_cfg, **optimized_cfg}
                                        adx_val = latest_candle.get("ADX", 0.0)
                                        sl_multiplier = cfg.get("sl_mult", 1.0)
                                        vol_factor = 1.0
                                        if atr_norm_val > 0:
                                            vol_factor = 1.5 - ((atr_norm_val - 0.003) / 0.005) * 0.75
                                            vol_factor = max(0.75, min(1.5, vol_factor))

                                        base_tp_target = cfg.get("tp_mult_trending", 2.0) if adx_val >= 20.0 else cfg.get("tp_mult_ranging", 1.5)
                                        tp_multiplier_adjusted = round(base_tp_target * vol_factor, 3)
                                        print(f"[{iv}m Target Config] ADX: {adx_val:.1f} | Base TP Mult: {base_tp_target:.2f}x (Vol Factor: {vol_factor:.2f}x) -> Effective TP Mult: {tp_multiplier_adjusted:.2f}x | SL Mult: {sl_multiplier:.2f}x")

                                        # 1. Volatility (ATR Percentile) Adjustment (±5%)
                                        atr_series = pd.to_numeric(df_completed["ATR"], errors="coerce").tail(100) if (df_completed is not None and "ATR" in df_completed.columns) else None
                                        vol_adj = 1.00
                                        if atr_series is not None and len(atr_series.dropna()) > 10:
                                            curr_atr = float(latest_candle.get("ATR", atr_dollars))
                                            clean_atr = atr_series.dropna()
                                            atr_percentile = float((clean_atr < curr_atr).mean() * 100.0) if len(clean_atr) > 0 else 50.0
                                            if atr_percentile > 90.0:
                                                vol_adj = 0.95  # Extreme volatility: tighten target before exhaustion reversal
                                            elif atr_percentile < 20.0:
                                                vol_adj = 1.05  # Quiet market: expand target for breakout extension
                                        tp_multiplier_adjusted *= vol_adj

                                        # 2. Session Liquidity Adjustment
                                        curr_utc_hour = datetime.now(timezone.utc).hour
                                        if 6 <= curr_utc_hour < 8:
                                            session_factor = 0.95  # Late Asian session: lower liquidity
                                        elif 12 <= curr_utc_hour < 16:
                                            session_factor = 1.00  # London / NY overlap: prime liquidity
                                        else:
                                            session_factor = 0.98
                                        tp_multiplier_adjusted *= session_factor

                                        # 3. Walk-Forward Optimal Rounding (0.05 precision)
                                        from trade_calculators import UnifiedTargetGenerator
                                        tp_m = UnifiedTargetGenerator.resolve_tp_multiplier(
                                            interval=str(iv), entry_price=float(latest_candle["close"]),
                                            atr_dollars=atr_dollars, base_tp_m=tp_multiplier_adjusted
                                        )
                                        tp_change = tp_m * atr_dollars

                                        print(f"[{iv}m Target Alignment] ADX: {adx_val:.1f} | Dynamic multipliers: SL = {sl_multiplier}x, TP = {tp_multiplier_adjusted:.2f}x (Vol: {vol_adj:.2f}x, Session: {session_factor:.2f}x)")
                                    
                                        # Maker execution: zero entry slippage for limit orders
                                        slippage_pct = 0.0
                                        raw_entry_price = float(latest_candle["close"])
                                        entry_price = raw_entry_price

                                        # Take-Profit distance tp_change resolved via UnifiedTargetGenerator above
                                    
                                        # Dynamically adjust Stop Loss multiplier based on prediction confidence
                                        sl_multiplier_adjusted = sl_multiplier
                                        if calibrated_confidence > dynamic_conf_threshold:
                                            confidence_ratio = (calibrated_confidence - dynamic_conf_threshold) / (1.0 - dynamic_conf_threshold)
                                            # Scale SL down by up to 30% for maximum confidence trades
                                            sl_multiplier_adjusted = sl_multiplier * (1.0 - 0.3 * confidence_ratio)
                                        
                                        # Refinements 3, 4, 7: Adaptive Structural Swing Stop & Recency Guard for 15m
                                        if str(iv) == "15":
                                            struct_sl, struct_sl_dist_pct, struct_meta = trade_calculators.calculate_adaptive_structural_stop(
                                                df_recent=df_completed,
                                                entry_price=entry_price,
                                                direction=ml_trend,
                                                atr_val=atr_dollars,
                                                regime=regime_name,
                                                volatility=atr_norm_val
                                            )
                                            stop_loss_price = struct_sl
                                            raw_sl_dist = abs(entry_price - stop_loss_price)
                                        
                                            # Refinements 5 & 6: Dynamic Leverage Scaling & Floor
                                            base_sl_pct = max(0.4, (atr_dollars * 0.75 / entry_price) * 100.0)
                                            scaled_lev, is_valid_lev = trade_calculators.scale_leverage_for_fixed_risk(
                                                base_leverage=7.5,
                                                base_sl_pct=base_sl_pct,
                                                structural_sl_pct=struct_sl_dist_pct
                                            )
                                            if not is_valid_lev:
                                                print(f"[{symbol} 15m Filter] Trade skipped: Scaled leverage ({scaled_lev}x) below 1.5x floor limit.")
                                                status_msg = "Skipped (Leverage Floor < 1.5x)"
                                                all_pass = False

                                            take_profit_price = (entry_price + tp_change) if ml_trend == "Bullish" else (entry_price - tp_change)
                                            print(f"[15m Structural Stop] Entry: {entry_price:.4f} | Structural SL: {stop_loss_price:.4f} (Dist: {struct_sl_dist_pct:.2f}%, Window: {struct_meta['window']}b, Quality: {struct_meta['quality_score']}/100) -> Scaled Leverage: {scaled_lev:.2f}x")
                                        else:
                                            raw_sl_dist = risk_engine.calculate_final_stop_distance(
                                                entry_price, atr_dollars, symbol, df=df_completed, gmm_multiplier=sl_multiplier_adjusted, database_module=database
                                            )
                                            if ml_trend == "Bullish":
                                                stop_loss_price = entry_price - raw_sl_dist
                                                take_profit_price = entry_price + tp_change
                                            else:
                                                stop_loss_price = entry_price + raw_sl_dist
                                                take_profit_price = entry_price - tp_change
                                            print(f"[{iv}m ML Targets] Entry: {entry_price:.2f} | Dynamic SL: {stop_loss_price:.2f} (Mult: {sl_multiplier_adjusted:.2f}x) | Regressor TP: {take_profit_price:.2f} (Expected: {pred_change:+.3f})")


                                        # Calibrated Position Sizing based on Isotonic Probability (Kelly scaling)
                                        c_prob = float(calibrated_confidence)
                                        current_hour_pkt = get_pkt_time().hour
                                        is_golden_hour = 18 <= current_hour_pkt < 21
                                    
                                        # Pre-calculate active trade stats needed for dynamic sizing
                                        total_active_size = sum(t.get("position_size_usd", 0.0) for tf_key in ["15m", "30m", "1h", "2h", "4h", "6h"] for t in bot_state.get(f"active_trade_{tf_key}", []))
                                        current_bal = bot_state.get("simulated_balance", 80.0)
                                        if TRADE_MODE != "simulation":
                                            real_bal = get_real_bybit_balance_cached(force=True)
                                            if isinstance(real_bal, (int, float)) and real_bal > 0:
                                                current_bal = real_bal
                                        cov_multiplier, net_risk = calculate_covariance_multiplier(symbol, ml_trend)
                                    
                                        # Base size dynamic calculation using Risk Engine Conservative Kelly (Wilson CI / Bootstrap)
                                        from config import (
                                            MIN_POSITION_BALANCE_FRAC, MAX_POSITION_BALANCE_FRAC,
                                            CVAR_TAIL_PERCENTILE, CVAR_FALLBACK, DAILY_LOSS_BUDGET_FRAC
                                        )
                                        _latest_pred = bot_state.get(f"latest_prediction_{iv}", {})
                                        _mcc_val = _latest_pred.get("manifest_mcc") if isinstance(_latest_pred, dict) else None
                                        scaled_kelly = risk_engine.compute_conservative_kelly(
                                            calibrated_confidence=calibrated_confidence,
                                            tp_multiplier=tp_multiplier_adjusted,
                                            sl_multiplier=sl_multiplier,
                                            interval=str(iv),
                                            trade_history=bot_state.get("trade_history", []),
                                            mcc_val=_mcc_val
                                        )
                                    
                                        # Enforce bounds on balance fraction (Min 2%, Max 15% per trade from config)
                                        f_clamped = max(MIN_POSITION_BALANCE_FRAC, min(MAX_POSITION_BALANCE_FRAC, scaled_kelly))
                                    
                                        # Sizing before leverage
                                        position_size_usd = current_bal * f_clamped
                                    
                                        # Covariance multiplier to account for existing correlations
                                        position_size_usd = position_size_usd * cov_multiplier
                                    
                                        # Volatility Regime Sizing Multiplier (Sweet spot 1.2x boost, extreme vol 0.5x, flat chop 0.3x)
                                        vol_regime_mult = risk_engine.get_volatility_regime_multiplier(atr_norm_val, iv)
                                        position_size_usd = position_size_usd * vol_regime_mult
                                        print(f"[{symbol} {iv}m Volatility Regime Sizing] Multiplier: {vol_regime_mult:.2f}x -> Position Size: ${position_size_usd:.2f}")

                                        # Phase 1 Continuous Learning Engine Risk Multiplier (Enforces >= 50 closed trades floor)
                                        from learning_engine import continuous_learning_engine
                                        learning_risk_mult = continuous_learning_engine.get_risk_multiplier({
                                            "symbol": symbol,
                                            "interval": str(iv),
                                            "confidence": calibrated_confidence,
                                            "regime": regime_name
                                        })
                                        position_size_usd = position_size_usd * learning_risk_mult
                                        print(f"[{symbol} {iv}m Learning Engine Sizing] Multiplier: {learning_risk_mult:.2f}x -> Position Size: ${position_size_usd:.2f}")
                                    
                                        # CVaR (Expected Shortfall) Risk Constraint
                                        try:
                                            hist_close = df["close"].values
                                            if len(hist_close) > 30:
                                                returns_pct = (hist_close[1:] - hist_close[:-1]) / hist_close[:-1]
                                                returns_sorted = np.sort(returns_pct)
                                                alpha_idx = max(1, int(len(returns_sorted) * CVAR_TAIL_PERCENTILE))
                                                tail_losses = returns_sorted[:alpha_idx]
                                                cvar_95 = abs(float(np.mean(tail_losses))) if len(tail_losses) > 0 else CVAR_FALLBACK
                                            else:
                                                cvar_95 = CVAR_FALLBACK
                                            daily_loss_budget = current_bal * DAILY_LOSS_BUDGET_FRAC
                                            max_cvar_size = daily_loss_budget / (cvar_95 + 1e-8)
                                            print(f"[{iv}m CVaR Guard] 95% CVaR: {cvar_95*100:.2f}% | Max Risk Size Allowed: ${max_cvar_size:.2f}")
                                            position_size_usd = min(position_size_usd, max_cvar_size)
                                        except Exception as cvar_err:
                                            print(f"[CVaR Error] {cvar_err}")
                                        if is_golden_hour:
                                            # Golden Hour: Double the target slot allocation size
                                            position_size_usd = position_size_usd * 2.0
                                            print(f"[{iv}m Golden Hour Kelly Sizing] Kelly Fraction: {scaled_kelly:.4f} -> Clamped: {f_clamped*100:.1f}% -> Golden Target: ${position_size_usd:.2f} (Covariance: {cov_multiplier:.2f}x)")
                                        else:
                                            print(f"[{iv}m Kelly Sizing] Kelly Fraction: {scaled_kelly:.4f} -> Clamped: {f_clamped*100:.1f}% -> Final Size: ${position_size_usd:.2f} (Covariance: {cov_multiplier:.2f}x)")
                                        
                                        # Clip to minimum Bybit order requirement (e.g. $2.0)
                                        position_size_usd = max(2.0, position_size_usd)
                                        original_kelly_size = float(position_size_usd) # Keep intended size pre-clamp
                                        print(f"[{iv}m Trade Size Boundary Check] Final size before leverage (CVaR constrained): ${position_size_usd:.2f}")

                                        # Calculate Kelly parameters for logs and metadata (preserving variables for downstream use)
                                        kelly_fraction = scaled_kelly

                                        # 3. Concurrent Position Limit Check
                                        from config import MAX_CONCURRENT_POSITIONS
                                        total_active_count = sum(len(bot_state.get(f"active_trade_{tf_k}", [])) for tf_k in ["15m", "30m", "1h", "2h", "4h", "6h"])

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
                                            if 'scaled_lev' in locals() and str(iv) == "15" and scaled_lev is not None and scaled_lev > 0:
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

                                            # F-1: MCC Leverage Qualification Threshold Clamp
                                            mcc_val = pred_entry_dict.get("manifest_mcc") if 'pred_entry_dict' in locals() else (locals().get("pred_info", {}).get("manifest_mcc"))
                                            mcc_thresh = getattr(config, "MCC_LEVERAGE_QUALIFICATION_THRESHOLD", 0.15)
                                            if mcc_val is not None and mcc_val < mcc_thresh:
                                                cons_caps = getattr(config, "CONSERVATIVE_LEVERAGE_CAPS", {})
                                                mcc_cap = cons_caps.get(symbol, cons_caps.get("default", 3.0))
                                                lev_cap = min(lev_cap, mcc_cap)
                                                log_event("INFO", f"[{symbol} {iv}m F-1 Leverage Guard] Model MCC ({mcc_val:.4f} < {mcc_thresh:.4f}) — Clamped max leverage to {lev_cap:.1f}x.")

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

                                            stop_loss_price = adjusted_struct["stop_price"]
                                            take_profit_price = adjusted_struct["tp_price"]
                                            leverage_val = adjusted_struct["leverage"]

                                            cfg = TIMEFRAME_CONFIG.get(str(iv), {"lookahead": 10})
                                            lookahead = cfg.get("lookahead", 10)
                                            duration_seconds = int(iv) * 60.0 * lookahead
                                            import uuid
                                            trade_uuid = str(uuid.uuid4())
                                            # Calculate quantity (qty) in coins rounded according to symbol requirements
                                            leveraged_size = position_size_usd * leverage_val
                                            raw_qty = leveraged_size / entry_price
                                            qty_str = format_bybit_qty(symbol, raw_qty)
                                            qty_val = float(qty_str)
                                         
                                            original_notional = qty_val * entry_price
                                            original_stop_dist = abs(entry_price - stop_loss_price)
                                            original_risk_usd = (original_notional / entry_price) * original_stop_dist
                                            is_oversized_trade = False

                                            # Enforce minimum order value from config (default 5.1 USDT)
                                            min_order_value = getattr(config, "MIN_ORDER_VALUE_USDT", 5.1)
                                            if qty_val * entry_price < min_order_value:
                                                step = get_bybit_min_qty_step(symbol)
                                                required_qty = min_order_value / entry_price
                                                import math
                                                if step > 0:
                                                    qty_val = round(required_qty / step) * step
                                                    qty_str = format_bybit_qty(symbol, qty_val)
                                                    qty_val = float(qty_str)
                                                    raw_qty = qty_val
                                                
                                                scaled_notional = qty_val * entry_price
                                                clamped_val = float(position_size_usd)
                                                final_val = float(scaled_notional / max(1e-9, leverage_val))
                                                if final_val > clamped_val:
                                                    log_event("INFO", f"[{symbol} {iv}m] Scaled UP to min order value: ${clamped_val:.2f} -> ${final_val:.2f}")
                                            
                                                # Priority 1: Tighten stop distance proportionally to keep dollar risk constant
                                                scale_ratio = original_notional / scaled_notional if scaled_notional > 0 else 1.0
                                                new_stop_dist = original_stop_dist * scale_ratio
                                            
                                                # Enforce absolute floor: Never compress SL tighter than 0.60x ATR to prevent spread noise stop-outs
                                                min_allowed_sl_dist = atr_dollars * 0.60
                                                if new_stop_dist < min_allowed_sl_dist:
                                                    new_stop_dist = min_allowed_sl_dist
                                                    print(f"[{symbol} {iv}m Risk Guard] Capped SL compression to 0.60x ATR (${min_allowed_sl_dist:.4f}) to protect against spread noise.")
                                            
                                                if str(ml_trend).upper() in ["BULLISH", "LONG", "BUY"]:
                                                    new_sl_price = entry_price - new_stop_dist
                                                else:
                                                    new_sl_price = entry_price + new_stop_dist

                                                
                                                scaled_risk_usd = (scaled_notional / max(1e-8, entry_price)) * new_stop_dist
                                            
                                                # Priority 2: Hard Cap - Never exceed 110% of approved original risk
                                                if scaled_risk_usd > original_risk_usd * 1.10:
                                                    print(f"[{symbol} {iv}m Risk Guard] REJECTED: Scaling to ${scaled_notional:.2f} would exceed 110% of approved risk (Scaled: ${scaled_risk_usd:.2f} vs Approved: ${original_risk_usd:.2f})")
                                                    status_msg = "Skipped (Exceeds 110% Risk Cap)"
                                                    wallet_exceeded = True
                                                else:
                                                    stop_loss_price = new_sl_price
                                                    is_oversized_trade = True
                                                    print(f"[{symbol} {iv}m API] Enforced minimum order value (${scaled_notional:.2f}). Tightened SL from ${original_stop_dist:.2f} to ${new_stop_dist:.2f} to keep risk constant at ${scaled_risk_usd:.2f}.")

                                            # Priority 3: Balance Guard - Remove auto-leverage escalation. If margin doesn't fit within 90% of balance, reject trade.
                                            required_margin = (qty_val * entry_price) / max(1e-8, leverage_val)
                                        
                                            # Post-Floor Geometry & Economic Viability Recheck
                                            all_pass = True
                                            try:
                                                trade_calculators.assert_valid_geometry(ml_trend, entry_price, stop_loss_price, take_profit_price, symbol=f"{symbol} {iv}m")
                                            except ValueError as geom_err:
                                                log_event("ERROR", str(geom_err))
                                                status_msg = f"Skipped (Invalid {ml_trend} Geometry)"
                                                all_pass = False
                                        
                                            if all_pass:
                                                # Post-Floor R:R & Economic Re-check (Closing the loop on widened SL)
                                                final_sl_dist = abs(entry_price - stop_loss_price)
                                                final_tp_dist = abs(take_profit_price - entry_price)
                                                final_rr = (final_tp_dist / max(1e-9, final_sl_dist))
                                                cost_frac = 0.0016
                                                required_p = (1.0 / (1.0 + max(1e-9, final_rr))) + cost_frac

                                                if calibrated_confidence < required_p:
                                                    log_event("WARNING", f"[{symbol} {iv}m] Post-floor SL widening reduced R:R to {final_rr:.2f}; calibrated confidence {calibrated_confidence:.3f} < required threshold {required_p:.3f}. Aborting entry.")
                                                    status_msg = f"Skipped (Post-Floor R:R {final_rr:.2f} Econ Fail)"
                                                    all_pass = False
                                        
                                            if not all_pass:
                                                wallet_exceeded = True
                                        
                                            if not wallet_exceeded and required_margin > current_bal * 0.90:
                                                print(f"[{symbol} {iv}m Margin Guard] REJECTED: Required margin (${required_margin:.2f}) exceeds 90% of available wallet balance (${current_bal:.2f}). Trade entry aborted.")
                                                status_msg = "Skipped (Exceeds Wallet Margin)"
                                                wallet_exceeded = True

                                            # Adaptive Volume Gate Check
                                            vol_pass, vol_msg, vol_pctile = adaptive_volume_gate.check(symbol, kline_df=df_completed)
                                            print(f"[{symbol} {iv}m Volume Gate] {vol_msg}")
                                            if not vol_pass:
                                                print(f"[{symbol} {iv}m Volume Gate Block] Trade entry aborted.")
                                                status_msg = "Skipped (Volume Gate Block)"
                                                wallet_exceeded = True
                                                bybit_success = False

                                            # Pre-Trade Risk Checklist Check
                                            pred_info = bot_state.get(f"latest_prediction_{iv}") or bot_state.get(f"latest_prediction_{iv}m") or {}
                                            if pred_info.get("is_fallback", False) or pred_info.get("signal_source") in ["RULE_BASED_FALLBACK", "UNSET"]:
                                                position_size_usd *= 0.50
                                                print(f"[{symbol} {iv}m Signal Guard] Rule-based fallback signal detected: Applied 50% position sizing penalty.")

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
                                            rec.signal_source      = str(pred_info.get("signal_source") or "UNSET")
                                            rec.is_fallback        = int(pred_info.get("is_fallback", False))
                                            rec.direction          = ml_trend
                                            rec.raw_confidence     = pred_info.get("raw_confidence")
                                            rec.calibrated_conf    = pred_info.get("calibrated_confidence")
                                            rec.calibrator_version = pred_info.get("calibrator_version")
                                            rec.calibrator_ece     = pred_info.get("calibrator_ece")
                                            rec.model_version      = pred_info.get("model_version")
                                            rec.feature_hash       = pred_info.get("feature_contract_hash") or pred_info.get("feature_hash")
                                            rec.manifest_schema    = pred_info.get("manifest_schema_version")
                                            rec.git_sha            = pred_info.get("git_sha")
                                            rec.regime             = pred_info.get("regime_mode") or pred_info.get("regime")
                                            rec.adx                = pred_info.get("adx")
                                            rec.atr_norm           = pred_info.get("atr_norm")
                                            rec.liquidity_score    = bot_state.get("liquidity_score", 1.0)
                                            rec.position_size_usd  = position_size_usd
                                            rec.leverage           = leverage_val

                                            try:
                                                passed_checklist, checklist_msg, dd_mult, capped_size = risk_engine.evaluate_pre_trade_checklist(
                                                    symbol, position_size_usd, leverage_val, active_trades_list, bot_state, df_dict, interval=str(iv), direction=ml_trend, journal=rec
                                                )
                                                rec.outcome = "EXECUTED" if (passed_checklist and not wallet_exceeded) else "REJECTED"
                                                rec.reject_reason = None if (passed_checklist and not wallet_exceeded) else checklist_msg
                                                if passed_checklist and not wallet_exceeded:
                                                    rec.position_size_usd = capped_size * dd_mult
                                                    rec.trade_id = f"{symbol}_{trade_uuid}"
                                            except Exception as risk_err:
                                                rec.outcome = "ERROR"
                                                rec.reject_reason = f"Risk checklist exception: {risk_err}"
                                                print(f"[{symbol} {iv}m CRITICAL RISK CHECKLIST EXCEPTION] {risk_err}. Aborting trade entry (Fail-Closed).")
                                                passed_checklist = False
                                                checklist_msg = f"REJECTED: Risk Checklist Exception ({risk_err})"
                                                dd_mult = 0.0
                                                capped_size = 0.0
                                            finally:
                                                pass

                                            print(f"[{symbol} {iv}m Pre-Trade Checklist] {checklist_msg}")
                                            if not passed_checklist or wallet_exceeded:
                                                print(f"[{symbol} {iv}m Risk Checklist Block] Trade entry aborted.")
                                                if not passed_checklist:
                                                    status_msg = "Skipped (Risk Checklist Block)"
                                                wallet_exceeded = True
                                                bybit_success = False
                                            else:
                                                position_size_usd = capped_size * dd_mult

                                                # Set Bybit Leverage and Place Order if in live/testnet mode
                                                bybit_success = True
                                                bybit_order_id = None
                                                bybit_scale_out_order_id = None
                                            
                                                if TRADE_MODE != "simulation":
                                                    # Live trading execution offloaded to background thread to minimize latency
                                                    just_opened_symbols.add(symbol)
                                                    if bybit_success:
                                                        with active_execution_lock:
                                                            active_execution_symbols.add(symbol)
                                                        actual_qty = raw_qty if raw_qty > 0 else (float((position_size_usd * leverage_val) / entry_price) if entry_price > 0 else 0.0)
                                                        actual_notional_val = float(actual_qty * entry_price)
                                                        actual_margin_usd = float(actual_notional_val / leverage_val) if leverage_val > 0 else float(position_size_usd)
                                                        threading.Thread(
                                                            target=execute_bybit_trade_async,
                                                            args=(symbol, iv, tf, ml_trend, leverage_val, qty_str, raw_qty, entry_price, stop_loss_price, take_profit_price, position_size_usd, kelly_fraction, calibrated_confidence, ml_confidence, dynamic_conf_threshold, latest_completed_ts, latest_candle, pred_change, predicted_price, atr_dollars, tp_multiplier_adjusted, sl_multiplier_adjusted, df_completed, trade_uuid, duration_seconds, active_trade_key, is_oversized_trade, original_kelly_size),
                                                            daemon=True
                                                        ).start()
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
                                            
                                                # Mark symbol as opened this cycle — prevents duplicate opens due to Bybit sync latency
                                                just_opened_symbols.add(symbol)
                                                # Sync positions immediately to load live Bybit state parameters
                                                if TRADE_MODE != "simulation":
                                                    sync_active_positions_from_bybit()
                                            
                                                # Deduct size from wallet balance immediately (only in simulation)
                                                if TRADE_MODE == "simulation":
                                                    bot_state["simulated_balance"] = round(bot_state["simulated_balance"] - position_size_usd, 2)
                                            
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
                            else:
                                bot_state["prediction_history"].append({
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
                            
                                max_pred_cap = getattr(config, "MAX_PREDICTION_HISTORY_MEMORY", 500)
                                if len(bot_state["prediction_history"]) > max_pred_cap:
                                    bot_state["prediction_history"] = bot_state["prediction_history"][-max_pred_cap:]
                        
                            evaluate_predictions(df_completed, iv, symbol)
                            save_history()
                        
                    finally:
                        rec.status_msg = status_msg
                        if not getattr(rec, "reject_reason", None) and status_msg not in ("Pending", ""):
                            rec.reject_reason = status_msg
                        if not hasattr(rec, "outcome") or not rec.outcome or rec.outcome == "ERROR":
                            rec.outcome = "EXECUTED" if placed else ("REJECTED" if status_msg.startswith("REJECTED") else "SKIPPED")
                        write_decision(rec)
                        log_event("INFO", f"[{symbol} {iv}m] Journalling decision: {status_msg}")
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
                    cvd_val = state.get("cvd", 0.0)
                    ofi_val = state.get("ofi", 0.0)
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
    threading.Thread(target=start_telegram_command_listener, args=(bot_state, bot_state_lock), name="telegram-listener", daemon=True).start()
    
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
    # Start funding rate arbitrage monitor thread
    threading.Thread(target=run_funding_rate_arbitrage_monitor, name="funding-arb-monitor", daemon=True).start()
    # Start daily database and trade journal backup thread
    threading.Thread(target=run_daily_backup_scheduler, name="daily-backup-scheduler", daemon=True).start()
    # Start daily 00:00 UTC performance summary report thread
    threading.Thread(target=run_daily_summary_scheduler, name="daily-summary-scheduler", daemon=True).start()
    from signal_evaluator import run_signal_evaluator_loop
    threading.Thread(target=run_signal_evaluator_loop, args=(bot_state,), name="signal-evaluator", daemon=True).start()

    # Keep main thread alive
    while True:
        time.sleep(1)