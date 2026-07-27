import os
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone


PAIN_FEEDBACK_FILE = "pain_feedback_state.json"
pain_lock = threading.Lock()

class PainFeedbackLoop:
    def __init__(self, state_file=PAIN_FEEDBACK_FILE):
        self.state_file = state_file
        self.adjustments = self.load_state()  # symbol -> {base_floor, adjusted_floor, applied_at, decay_days, reason}

    def load_state(self):
        with pain_lock:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[PainFeedbackLoop] Error loading state file: {e}")
            return {}

    def save_state(self):
        with pain_lock:
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(self.adjustments, f, indent=2)
            except Exception as e:
                print(f"[PainFeedbackLoop] Error saving state file: {e}")

    def register_pain_trade(self, symbol, entry_price, exit_price, take_profit, current_floor):
        """Raise min floor for this symbol due to a pain trade."""
        if not entry_price or entry_price == 0:
            return
            
        adverse_move = abs(exit_price - entry_price) / entry_price
        needed_floor = adverse_move * 1.3  # 30% safety buffer
        
        new_floor = max(current_floor * 1.15, needed_floor)
        new_floor = max(0.005, min(new_floor, 0.020))  # Hard cap at 2.0%, min 0.5%
        
        self.adjustments[symbol] = {
            'base_floor': current_floor,
            'adjusted_floor': new_floor,
            'applied_at': datetime.now(timezone.utc).isoformat(),
            'decay_days': 30,
            'reason': f"Pain trade: stopped at {adverse_move:.2%}, reversed to TP"
        }
        self.save_state()
        print(f"[PainFeedbackLoop ALERT] Raised {symbol} min floor from {current_floor:.2%} to {new_floor:.2%} (Decay: 30 days)")

    def get_effective_floor(self, symbol):
        """Get floor with pain adjustment applied, decayed over time."""
        with pain_lock:
            if symbol not in self.adjustments:
                return None
            
            adj = self.adjustments[symbol]

            try:
                applied_at = datetime.fromisoformat(adj['applied_at'])
                if applied_at.tzinfo is None:
                    applied_at = applied_at.replace(tzinfo=timezone.utc)
            except Exception:
                return None
                
            days_since = (datetime.now(timezone.utc) - applied_at).total_seconds() / 86400.0
            decay_days = adj.get('decay_days', 30)
            
            if days_since >= decay_days:
                # Expired, remove adjustment safely
                del self.adjustments[symbol]
                try:
                    with open(self.state_file, 'w') as f:
                        json.dump(self.adjustments, f, indent=2)
                except Exception as e:
                    print(f"[PainFeedbackLoop] Error saving state file: {e}")
                return None
            
            # Linear decay back to base
            decay_progress = days_since / float(decay_days)
            base = adj.get('base_floor', 0.008)
            adjusted = adj.get('adjusted_floor', 0.015)
            
            effective = adjusted - (adjusted - base) * decay_progress
            return effective


    def verify_pending_pain_trades(self, database_module=None, fetch_kline_func=None):
        """Asynchronously verify 24h post-exit whether closed trades reached TP after being stopped out."""
        if not database_module or not hasattr(database_module, 'get_pending_pain_checks'):
            return

        try:
            pending = database_module.get_pending_pain_checks()
            now_ts = time.time()

            
            for p in pending:
                exit_time = float(p.get('exit_time') or 0)
                if exit_time == 0:
                    continue
                    
                # Evaluate after 24h elapsed since exit
                if now_ts - exit_time >= 86400:
                    symbol = p.get('symbol')
                    entry_price = float(p.get('entry_price') or 0)
                    exit_price = float(p.get('exit_price') or 0)
                    take_profit = float(p.get('take_profit') or 0)
                    direction = str(p.get('direction', 'LONG')).upper()
                    trade_id = p.get('trade_id')
                    
                    hit_tp_after_exit = False
                    if fetch_kline_func and take_profit > 0:
                        try:
                            try:
                                df_klines = fetch_kline_func(symbol=symbol, interval="15", limit=300)
                            except Exception:
                                df_klines = fetch_kline_func(symbol, start_ts=int(exit_time*1000), end_ts=int((exit_time + 86400)*1000))

                            if df_klines is not None and not df_klines.empty and 'high' in df_klines.columns and 'low' in df_klines.columns:
                                if 'timestamp' in df_klines.columns:
                                    sample_ts = float(df_klines['timestamp'].iloc[0])
                                    exit_time_ts = exit_time * 1000.0 if sample_ts > 1e11 else exit_time
                                    exit_time_end_ts = (exit_time + 86400) * 1000.0 if sample_ts > 1e11 else (exit_time + 86400)
                                    df_post = df_klines[(df_klines['timestamp'] >= exit_time_ts) & (df_klines['timestamp'] <= exit_time_end_ts)]
                                else:
                                    df_post = df_klines

                                if not df_post.empty:
                                    if direction == 'LONG':
                                        max_high = float(df_post['high'].max())
                                        hit_tp_after_exit = max_high >= take_profit
                                    else:
                                        min_low = float(df_post['low'].min())
                                        hit_tp_after_exit = min_low <= take_profit
                        except Exception as kerr:
                            print(f"[PainFeedbackLoop] Error fetching post-exit klines for {symbol}: {kerr}")


                    if hit_tp_after_exit:
                        current_floor = 0.008
                        self.register_pain_trade(symbol, entry_price, exit_price, take_profit, current_floor)
                        
                    database_module.delete_pending_pain_check(trade_id)
        except Exception as e:
            print(f"[PainFeedbackLoop] Error in verify_pending_pain_trades: {e}")

# Global instance for app-wide access
pain_feedback = PainFeedbackLoop()

