import os
import json
import hashlib
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional

ENGINE_VERSION = "3.0"
REGISTRY_FILE = "policy_registry.json"
DEFAULT_POLICY_DIR = "policies"

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 hex digest of a JSON configuration file."""
    if not os.path.exists(filepath):
        return "UNKNOWN_HASH"
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        print(f"[ExitPolicyEngine Warning] Error computing SHA256 for {filepath}: {e}")
        return "ERROR_HASH"

class ExitPolicyEngine:
    def __init__(self, registry_file: str = REGISTRY_FILE):
        self.registry_file = registry_file
        self.active_champion_id = "policy_v3"
        self.rollback_target_id = "policy_v2"
        self.champion_config: Dict[str, Any] = {}
        self.champion_hash: str = ""
        self.shadow_configs: Dict[str, Dict[str, Any]] = {}
        self.shadow_hashes: Dict[str, str] = {}
        self.load_registry_and_policies()

    def load_registry_and_policies(self):
        """Loads policy registry and instantiates active Champion, RollbackTarget, and Shadow configs."""
        if not os.path.exists(self.registry_file):
            print(f"[ExitPolicyEngine Warning] Registry file {self.registry_file} not found. Using defaults.")
            self._use_fallback_champion()
            return

        try:
            with open(self.registry_file, "r") as f:
                registry = json.load(f)
            
            self.active_champion_id = registry.get("active_champion", "policy_v3")
            self.rollback_target_id = registry.get("rollback_target", "policy_v2")
            
            policies_meta = registry.get("policies", {})
            
            # Load Champion
            champion_meta = policies_meta.get(self.active_champion_id, {})
            champ_path = champion_meta.get("policy_path", f"{DEFAULT_POLICY_DIR}/{self.active_champion_id}.json")
            
            if os.path.exists(champ_path):
                self.champion_config = self._load_and_validate_policy(champ_path)
                self.champion_hash = compute_file_sha256(champ_path)
            else:
                print(f"[ExitPolicyEngine Warning] Champion path {champ_path} not found. Trying RollbackTarget.")
                rollback_meta = policies_meta.get(self.rollback_target_id, {})
                rb_path = rollback_meta.get("policy_path", f"{DEFAULT_POLICY_DIR}/{self.rollback_target_id}.json")
                if os.path.exists(rb_path):
                    self.champion_config = self._load_and_validate_policy(rb_path)
                    self.champion_hash = compute_file_sha256(rb_path)
                else:
                    self._use_fallback_champion()

            # Load Shadow policies (if any registered with status == 'Shadow')
            for p_id, p_info in policies_meta.items():
                if p_info.get("status") == "Shadow" and p_id != self.active_champion_id:
                    path = p_info.get("policy_path", f"{DEFAULT_POLICY_DIR}/{p_id}.json")
                    if os.path.exists(path):
                        try:
                            self.shadow_configs[p_id] = self._load_and_validate_policy(path)
                            self.shadow_hashes[p_id] = compute_file_sha256(path)
                        except Exception as shadow_err:
                            print(f"[ExitPolicyEngine Warning] Failed to load shadow policy {p_id}: {shadow_err}")

        except Exception as e:
            print(f"[ExitPolicyEngine Error] Failed to load registry: {e}. Falling back to default champion.")
            self._use_fallback_champion()

    def _load_and_validate_policy(self, filepath: str) -> Dict[str, Any]:
        """Loads JSON policy file and asserts engine compatibility version."""
        with open(filepath, "r") as f:
            cfg = json.load(f)
        
        min_ver = cfg.get("min_engine_version", "1.0")
        if float(min_ver) > float(ENGINE_VERSION):
            raise ValueError(f"Incompatible policy {filepath}: requires min_engine_version {min_ver} > current {ENGINE_VERSION}")
        
        return cfg

    def _use_fallback_champion(self):
        """Fallback champion configuration if policy files are missing."""
        self.champion_hash = "FALLBACK_DEFAULT_HASH"
        self.champion_config = {
            "policy_id": "policy_fallback",
            "version": "3.0.0",
            "min_engine_version": "3.0",
            "parameters": {
                "RANGING": { "scale_out_pct": 0.25, "scale_out_atr_mult": 1.2, "be_trigger_atr_mult": 1.5, "be_safety_margin_atr": 0.10 },
                "MODERATE_TREND": { "scale_out_pct": 0.20, "scale_out_atr_mult": 1.5, "be_trigger_atr_mult": 1.5, "be_safety_margin_atr": 0.10 },
                "STRONG_TREND": { "scale_out_pct": 0.00, "scale_out_atr_mult": 0.0, "be_trigger_atr_mult": 2.0, "be_safety_margin_atr": 0.10 }
            }
        }

    def compute_be_buffer(self, symbol: str, leverage: float, entry_price: float, atr_dollars: float, safety_margin_atr: float = 0.10) -> float:
        """
        Calculates exact symbol overhead Break-Even buffer:
        BE_buffer = fees (entry + exit taker) + expected_slippage + safety_margin
        """
        round_trip_fee_pct = 0.0011 # 0.055% * 2 taker fee
        est_slippage_pct = 0.0005   # 0.05% expected slippage
        
        overhead_pct = round_trip_fee_pct + est_slippage_pct
        price_overhead = entry_price * overhead_pct
        safety_buffer = atr_dollars * safety_margin_atr
        
        return float(price_overhead + safety_buffer)

    def evaluate_stagnation_gate(
        self,
        pnl_usd: float,
        current_atr: float,
        entry_atr: float,
        current_volume: float,
        avg_volume: float,
        trade_age_hours: float,
        stagnation_age_hours: float,
        price_dev: float,
        adx_val: float,
        regime: str
    ) -> Tuple[bool, str]:
        """
        5-Factor Stagnation Gate:
        Only triggers if:
        1. Negative PnL (pnl_usd < 0)
        2. ATR Contraction (current_atr < 0.8 * entry_atr)
        3. Volume Contraction (current_volume < 0.7 * avg_volume)
        4. No Structure Improvement (price_dev < 0.5 * entry_atr)
        5. Low ADX / Choppy Regime (adx_val < 18.0 OR regime in ["CHOPPY", "RANGING"])
        """
        if trade_age_hours < stagnation_age_hours:
            return False, "Age below threshold"
            
        c1 = pnl_usd < 0.0
        c2 = current_atr < (0.8 * entry_atr) if entry_atr > 0 else False
        c3 = current_volume < (0.7 * avg_volume) if avg_volume > 0 else False
        c4 = price_dev < (0.5 * entry_atr) if entry_atr > 0 else False
        c5 = (adx_val < 18.0) or (str(regime).upper() in ["CHOPPY", "RANGING"])
        
        if c1 and c2 and c3 and c4 and c5:
            reason = f"5-FACTOR STAGNATION (Age: {trade_age_hours:.1f}h, Net: ${pnl_usd:+.2f}, ADX: {adx_val:.1f})"
            return True, reason
        else:
            return False, "Stagnation criteria not fully satisfied"

    def evaluate_hybrid_trailing_stop(
        self,
        direction: str,
        current_price: float,
        entry_price: float,
        stop_loss: float,
        swing_price: Optional[float],
        atr_dollars: float
    ) -> float:
        """
        Evaluates Hybrid Market Structure + ATR Trailing Stop:
        Long:  max(Swing Low - 0.25*ATR, ATR Trailing SL, current_SL) (requires current_price > entry_price)
        Short: min(Swing High + 0.25*ATR, ATR Trailing SL, current_SL) (requires current_price < entry_price)
        """
        atr_buffer = 0.25 * atr_dollars
        min_sl_dist = 0.60 * atr_dollars
        if direction == "Bullish":
            if current_price <= entry_price:
                return stop_loss
            atr_trail = current_price - (1.5 * atr_dollars)
            struct_trail = (swing_price - atr_buffer) if swing_price and swing_price > 0 else atr_trail
            candidate_sl = max(atr_trail, struct_trail)
            candidate_sl = min(candidate_sl, current_price - min_sl_dist)
            return max(stop_loss, candidate_sl)
        else:
            if current_price >= entry_price:
                return stop_loss
            atr_trail = current_price + (1.5 * atr_dollars)
            struct_trail = (swing_price + atr_buffer) if swing_price and swing_price > 0 else atr_trail
            candidate_sl = min(atr_trail, struct_trail)
            candidate_sl = max(candidate_sl, current_price + min_sl_dist)
            return min(stop_loss, candidate_sl)

    @staticmethod
    def _resolve_regime_key(regime: str, adx_val: float = 15.0) -> str:
        r = str(regime).upper()
        if "RANG" in r or "CHOP" in r:
            return "RANGING"
        if "STRONG" in r or ("HIGH VOL" in r and "TREND" in r) or (("TREND" in r) and float(adx_val) >= 30.0):
            return "STRONG_TREND"
        if "TREND" in r or "MODERATE" in r or float(adx_val) >= 25.0:
            return "MODERATE_TREND"
        from logger import log_event
        log_event("WARNING", f"[Exit Policy Engine] Unrecognized regime '{regime}' (ADX={adx_val:.1f}). Falling back to RANGING.")
        return "RANGING"

    def evaluate_exit(
        self,
        active_trade: Dict[str, Any],
        current_price: float,
        current_time: float,
        regime: str = "RANGING",
        adx_val: float = 15.0,
        current_volume: float = 100.0,
        avg_volume: float = 120.0,
        swing_price: Optional[float] = None,
        current_atr: Optional[float] = None
    ) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
        """
        Evaluates the active Champion exit policy for live execution,
        runs parallel Shadow policy simulations, and generates an Exit Decision Trace.
        Returns: (exit_reason, active_trade_updates, exit_decision_trace)
        """
        regime_key = self._resolve_regime_key(regime, adx_val)
        params = self.champion_config.get("parameters", {}).get(regime_key, {})
        if not params:
            params = self.champion_config.get("parameters", {}).get(regime.upper(), {})
        if not params:
            params = self.champion_config.get("parameters", {}).get("RANGING", {})

        direction = active_trade.get("direction", "Bullish")
        entry_price = float(active_trade.get("entry_price", current_price))
        take_profit = float(active_trade.get("take_profit", current_price * 1.05))
        stop_loss = float(active_trade.get("stop_loss", current_price * 0.95))
        half_closed = active_trade.get("half_closed", False)
        position_size_usd = float(active_trade.get("position_size_usd", 15.0))
        leverage = float(active_trade.get("leverage", 10.0))
        entry_atr = float(active_trade.get("entry_atr") or active_trade.get("atr_dollars") or (entry_price * 0.01))
        atr_dollars = float(current_atr) if current_atr is not None else float(active_trade.get("current_atr") or entry_atr)
        
        updates = {}
        exit_reason = None
        
        # 1. Scale-out Check
        scale_out_pct = float(params.get("scale_out_pct", 0.25))
        scale_out_atr_mult = float(params.get("scale_out_atr_mult", 1.2))
        be_trigger_atr_mult = float(params.get("be_trigger_atr_mult", 1.5))
        be_safety_margin_atr = float(params.get("be_safety_margin_atr", 0.10))
        
        trigger_scale_out = False
        if not half_closed and scale_out_pct > 0.0:
            if direction == "Bullish" and current_price >= (entry_price + scale_out_atr_mult * atr_dollars):
                trigger_scale_out = True
            elif direction == "Bearish" and current_price <= (entry_price - scale_out_atr_mult * atr_dollars):
                trigger_scale_out = True
                
        if trigger_scale_out and not half_closed:
            updates["trigger_scale_out"] = True
            updates["scale_out_pct"] = scale_out_pct

        # 2. Check Break-Even trigger if price moved past be_trigger_atr_mult without scale-out
        if not active_trade.get("break_even_triggered"):
            be_dist = be_trigger_atr_mult * atr_dollars
            be_reached = (current_price >= entry_price + be_dist) if direction == "Bullish" else (current_price <= entry_price - be_dist)
            if be_reached:
                be_buffer = self.compute_be_buffer(active_trade.get("symbol", "BTCUSDT"), leverage, entry_price, atr_dollars, be_safety_margin_atr)
                target_sl = (entry_price + be_buffer) if direction == "Bullish" else (entry_price - be_buffer)
                is_tighter = (target_sl > stop_loss + 1e-4) if direction == "Bullish" else (target_sl < stop_loss - 1e-4)
                if is_tighter:
                    updates["new_stop_loss"] = target_sl
                    stop_loss = target_sl
                updates["break_even_triggered"] = True

        # 3. Check Take Profit Hit (Cleaned fix: allowed regardless of half_closed)
        if direction == "Bullish":
            if current_price >= take_profit:
                exit_reason = "TAKE PROFIT HIT [SUCCESS]"
            elif current_price <= stop_loss:
                exit_reason = "TRAILING STOP HIT [SUCCESS]" if (half_closed or active_trade.get("break_even_triggered")) else "STOP LOSS HIT [FAIL]"
        else:
            if current_price <= take_profit:
                exit_reason = "TAKE PROFIT HIT [SUCCESS]"
            elif current_price >= stop_loss:
                exit_reason = "TRAILING STOP HIT [SUCCESS]" if (half_closed or active_trade.get("break_even_triggered")) else "STOP LOSS HIT [FAIL]"

        # 4. Check Stagnation Gate (Finding #139)
        if not exit_reason and params.get("enable_stagnation_gate", True):
            entry_time_ms = active_trade.get("entry_time")
            if entry_time_ms:
                entry_sec = (float(entry_time_ms) / 1000.0) if float(entry_time_ms) > 1e11 else float(entry_time_ms)
                trade_age_hours = (time.time() - entry_sec) / 3600.0
                trade_iv = str(active_trade.get("interval", "15")).replace("m", "").replace("h", "")
                iv_min = int(trade_iv) if trade_iv.isdigit() else 15
                stagnation_age_hours = max(1.5, (iv_min * 12) / 60.0) # 12 candles scaled by interval
                price_dev = abs(current_price - entry_price)
                
                # Approximate current PnL
                raw_ret = ((current_price - entry_price)/entry_price)*100.0 if direction == "Bullish" else ((entry_price - current_price)/entry_price)*100.0
                est_pnl = position_size_usd * (raw_ret * leverage / 100.0)
                
                is_stagnant, stag_reason = self.evaluate_stagnation_gate(
                    pnl_usd=est_pnl,
                    current_atr=atr_dollars,
                    entry_atr=entry_atr,
                    current_volume=current_volume,
                    avg_volume=avg_volume,
                    trade_age_hours=trade_age_hours,
                    stagnation_age_hours=stagnation_age_hours,
                    price_dev=price_dev,
                    adx_val=adx_val,
                    regime=regime
                )
                if is_stagnant:
                    exit_reason = stag_reason

        # 5. Hybrid Trailing Stop Update
        if not exit_reason and params.get("enable_structure_trailing", True):
            updated_sl = self.evaluate_hybrid_trailing_stop(direction, current_price, entry_price, stop_loss, swing_price, atr_dollars)
            is_tighter = (updated_sl > stop_loss + 1e-4) if direction == "Bullish" else (updated_sl < stop_loss - 1e-4)
            if is_tighter:
                updates["new_stop_loss"] = updated_sl
                stop_loss = updated_sl

        # 6. Generate Exit Decision Trace
        risk_dist = abs(entry_price - float(active_trade.get("original_sl", stop_loss)))
        risk_dist = max(1e-6, risk_dist)
        
        mfe_dist = (active_trade.get("max_high", current_price) - entry_price) if direction == "Bullish" else (entry_price - active_trade.get("min_low", current_price))
        mfe_dist = max(0.0, mfe_dist)
        
        captured_dist = (current_price - entry_price) if direction == "Bullish" else (entry_price - current_price)
        
        easy_r = round(mfe_dist / risk_dist, 2)
        captured_r = round(captured_dist / risk_dist, 2)
        exit_eff = round(captured_r / easy_r, 2) if easy_r > 0 else 0.0

        trace = {
            "policy_id": self.champion_config.get("policy_id", self.active_champion_id),
            "policy_hash": self.champion_hash,
            "regime": regime,
            "scaleout_triggered": trigger_scale_out or half_closed,
            "scaleout_pct": scale_out_pct if (trigger_scale_out or half_closed) else 0.0,
            "be_triggered": active_trade.get("break_even_triggered", False) or updates.get("break_even_triggered", False),
            "stagnation": "STAGNATION" in str(exit_reason).upper(),
            "timer_exit": "TIMER" in str(exit_reason).upper(),
            "final_reason": exit_reason or "ACTIVE",
            "captured_r": captured_r,
            "easy_r": easy_r,
            "exit_efficiency": max(0.0, exit_eff)
        }

        # 7. Evaluate Parallel Shadow Policy Simulations
        self.evaluate_shadow_simulations(active_trade, current_price, current_time, regime, adx_val, current_volume, avg_volume, swing_price, current_atr=atr_dollars)

        return exit_reason, updates, trace

    def evaluate_shadow_simulations(
        self,
        active_trade: Dict[str, Any],
        current_price: float,
        current_time: float,
        regime: str,
        adx_val: float,
        current_volume: float,
        avg_volume: float,
        swing_price: Optional[float] = None,
        current_atr: Optional[float] = None
    ):
        """Runs shadow candidate policies in parallel without sending real exchange orders."""
        regime_key = self._resolve_regime_key(regime, adx_val)
        for p_id, shadow_cfg in self.shadow_configs.items():
            try:
                shadow_params = shadow_cfg.get("parameters", {}).get(regime_key, {})
                if not shadow_params:
                    shadow_params = shadow_cfg.get("parameters", {}).get(regime.upper(), {})
                if not shadow_params:
                    shadow_params = shadow_cfg.get("parameters", {}).get("RANGING", {})

                direction = active_trade.get("direction", "Bullish")
                entry_price = float(active_trade.get("entry_price", current_price))
                stop_loss = float(active_trade.get("stop_loss", current_price * 0.95))
                take_profit = float(active_trade.get("take_profit", current_price * 1.05))
                entry_atr = float(active_trade.get("entry_atr") or active_trade.get("atr_dollars") or (entry_price * 0.01))
                atr_dollars = float(current_atr) if current_atr is not None else float(active_trade.get("current_atr") or entry_atr)

                # 1. Shadow Trailing Stop Evaluation
                shadow_sl = self.evaluate_hybrid_trailing_stop(
                    direction=direction,
                    current_price=current_price,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    swing_price=swing_price,
                    atr_dollars=atr_dollars
                )

                # 2. Shadow Stagnation Check
                stag_hours = float(shadow_params.get("stagnation_exit_hours", 2.0))
                entry_time = float(active_trade.get("entry_time", current_time))
                if entry_time > 1e11:
                    entry_time = entry_time / 1000.0
                trade_age_hours = (current_time - entry_time) / 3600.0
                is_long = direction == "Bullish"
                pnl_usd = (current_price - entry_price) * float(active_trade.get("qty", 1.0)) if is_long else (entry_price - current_price) * float(active_trade.get("qty", 1.0))
                price_dev = abs(current_price - entry_price)

                stag_exit, stag_reason = self.evaluate_stagnation_gate(
                    pnl_usd=pnl_usd,
                    current_atr=atr_dollars,
                    entry_atr=entry_atr,
                    current_volume=current_volume,
                    avg_volume=avg_volume,
                    trade_age_hours=trade_age_hours,
                    stagnation_age_hours=stag_hours,
                    price_dev=price_dev,
                    adx_val=adx_val,
                    regime=regime
                )

                shadow_exit_reason = None
                if stag_exit:
                    shadow_exit_reason = stag_reason
                elif is_long and current_price <= shadow_sl:
                    shadow_exit_reason = "SHADOW_TRAILING_STOP_HIT"
                elif is_long and current_price >= take_profit:
                    shadow_exit_reason = "SHADOW_TAKE_PROFIT_HIT"
                elif not is_long and current_price >= shadow_sl:
                    shadow_exit_reason = "SHADOW_TRAILING_STOP_HIT"
                elif not is_long and current_price <= take_profit:
                    shadow_exit_reason = "SHADOW_TAKE_PROFIT_HIT"

                if not hasattr(self, "_shadow_evaluations"):
                    self._shadow_evaluations = {}
                self._shadow_evaluations[p_id] = {
                    "last_eval_time": current_time,
                    "last_price": current_price,
                    "regime": regime,
                    "active_trade_id": active_trade.get("trade_id", "none"),
                    "shadow_sl": shadow_sl,
                    "shadow_exit_reason": shadow_exit_reason,
                    "stag_exit": stag_exit,
                    "pnl_usd": pnl_usd
                }
            except Exception as shadow_err:
                pass

    def get_shadow_evaluations(self) -> Dict[str, Any]:
        """Returns the most recent shadow policy simulation states."""
        return getattr(self, "_shadow_evaluations", {})

    def select_and_lock_meta_policy(
        self,
        regime: str = "Trending",
        garch_vol: float = 0.015,
        adx_val: float = 25.0
    ) -> Dict[str, str]:
        """
        Evaluates market context at trade entry and LOCKS the selected Meta-Policy.
        Mid-trade policy switching is explicitly prohibited to prevent attribution contamination.
        """
        regime_upper = str(regime).upper()
        if "TRENDING" in regime_upper and garch_vol <= 0.015:
            policy_id = "Trail_Wide"
            reason = "Trending regime with low volatility — using wide trailing stop (2.5x ATR)"
        elif "TRENDING" in regime_upper and garch_vol > 0.015:
            policy_id = "Trail_Tight"
            reason = "Trending regime with high volatility — using tight trailing stop (1.2x ATR) + scale-out"
        elif "RANGING" in regime_upper:
            policy_id = "Structure_Bound"
            reason = "Ranging regime — using structural swing high/low barrier exit"
        else:
            policy_id = "Time_Decay"
            reason = "Choppy regime — using linear time-decay exit"

        return {
            "locked_policy": policy_id,
            "policy_reason": reason,
            "entry_regime": regime,
            "entry_volatility": f"{garch_vol*100:.2f}%",
            "entry_adx": f"{adx_val:.1f}",
            "locked_timestamp": time.time()
        }

    def calculate_uncertainty_execution_policy(self, u_total: float = 0.06) -> Dict[str, Any]:
        """
        Ties total uncertainty (U_total) directly to execution management.
        """
        u = float(u_total)
        if u < 0.05:
            return {"execution_policy": "Full size, normal TP/SL", "sizing_mult": 1.00, "sl_mult": 1.00, "scaleout_pct": 0.50, "shadow_only": False}
        elif u < 0.10:
            return {"execution_policy": "Slight size reduction, standard exits", "sizing_mult": 0.90, "sl_mult": 1.00, "scaleout_pct": 0.50, "shadow_only": False}
        elif u < 0.20:
            return {"execution_policy": "Reduced size, wider stop, smaller initial scale-out", "sizing_mult": 0.65, "sl_mult": 1.25, "scaleout_pct": 0.25, "shadow_only": False}
        else:
            return {"execution_policy": "Paper trade or shadow-only evaluation", "sizing_mult": 0.00, "sl_mult": 1.50, "scaleout_pct": 0.00, "shadow_only": True}

    def evaluate_10_level_exit_hierarchy(
        self,
        symbol: str,
        interval: str,
        current_price: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        direction: str,
        candles_elapsed: int,
        expected_r: float = 1.20,
        mfe_r: float = 0.0,
        entry_regime: str = "Trending",
        current_regime: str = "Trending",
        garch_vol: float = 0.015,
        rolling_vol_20th_pct: float = 0.010,
        atr_ratio: float = 1.0,
        mhi_status: float = 100.0,
        incoming_signal_expected_r: Optional[float] = None,
        portfolio_heat_full: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates active trade against 10-level institutional exit hierarchy:
          1. Take Profit
          2. Stop Loss
          3. Catastrophic Risk
          4. Expected R Collapse
          5. Opportunity Cost Rotation
          6. Regime Shift
          7. Volatility Collapse
          8. MFE Stagnation
          9. Adaptive Time Decay
          10. Conditional Hard Timeout Safety Net
        """
        tf_clean = str(interval).replace("m", "")
        # Finding #140: Derive soft and hard limits directly from TIMEFRAME_CONFIG lookahead horizon
        import config
        tf_cfg = getattr(config, "TIMEFRAME_CONFIG", {}).get(tf_clean, {})
        lookahead = int(tf_cfg.get("lookahead", 12))
        base_soft = max(16, int(lookahead))
        hard_limit = max(base_soft + 2, int(lookahead * 1.5))

        import config
        # H-2: Continuous ATR Expansion/Compression Soft Timeout Adjustment
        sens = getattr(config, "ATR_TIMEOUT_SENSITIVITY", 5.0)
        max_adj = getattr(config, "ATR_TIMEOUT_MAX_ADJ", 2.0)
        atr_adj = float(np.clip((float(atr_ratio) - 1.0) * sens, -max_adj, max_adj))
        
        # C-1: Model Health Index (MHI) Continuous Patience Scaling
        if isinstance(mhi_status, dict):
            mhi_num = float(mhi_status.get("mhi", 100.0))
        else:
            try:
                mhi_num = float(mhi_status)
            except (ValueError, TypeError):
                mhi_num = 100.0
        mhi_floor = getattr(config, "MHI_MULT_FLOOR", 0.50)
        mhi_mult = float(np.clip(mhi_num / 100.0, mhi_floor, 1.0))
        # Finding #90: Floor soft limit at full label lookahead horizon
        soft_limit = max(int(lookahead), int(round((base_soft + atr_adj) * mhi_mult)))

        is_long = direction.upper() in ["BUY", "LONG", "BULLISH"]
        pnl_dist = (current_price - entry_price) if is_long else (entry_price - current_price)
        risk_dist = abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 1e-6 else 1.0
        current_r = round(pnl_dist / risk_dist, 2)

        # M-1: Timeframe & Regime-Adaptive Time Decay Rate
        decay_policy = getattr(config, "TIME_DECAY_POLICY", {})
        regime_key = "trending" if "TRENDING" in current_regime.upper() else ("ranging" if "RANGING" in current_regime.upper() else "unknown")
        tf_map = decay_policy.get(regime_key, {})
        decay_rate = float(tf_map.get(tf_clean, tf_map.get("default", 0.10)))
        age_ratio = float(candles_elapsed) / max(1, soft_limit)
        
        # M-2: Time Decay Floor Governance Policy (decay floor falls to 0.0 once hard limit exceeded)
        decay_floor = float(getattr(config, "TIME_DECAY_FLOOR_EXCEEDED", 0.00)) if candles_elapsed > hard_limit else float(getattr(config, "TIME_DECAY_FLOOR", 0.20))
        decay_factor = max(decay_floor, 1.0 - decay_rate * max(0.0, age_ratio - 1.0))
        decayed_expected_r = round(expected_r * decay_factor, 3)

        # H-1: Unified Exit Score Policy & Continuous Ramping
        exit_policy = getattr(config, "EXIT_SCORE_POLICY", {})
        w_norm_r = exit_policy.get("norm_r_weight", 0.30)
        w_regime = exit_policy.get("regime_weight", 0.20)
        w_vol = exit_policy.get("volatility_weight", 0.20)
        w_decay = exit_policy.get("decay_weight", 0.15)
        w_opp = exit_policy.get("opportunity_weight", 0.15)
        opp_hurdle = exit_policy.get("opp_hurdle", 1.5)

        norm_r = min(1.0, max(0.0, decayed_expected_r / 2.0))
        regime_intact = 1.0 if entry_regime.upper() == current_regime.upper() else 0.0
        vol_pct_score = float(np.clip(float(garch_vol) / max(1e-9, float(rolling_vol_20th_pct)), 0.0, 1.0))
        
        incoming_r = float(incoming_signal_expected_r) if incoming_signal_expected_r is not None else 0.0
        if portfolio_heat_full and incoming_r > 0:
            opp_score = float(np.clip(1.0 - (incoming_r / max(1e-9, expected_r * opp_hurdle)), 0.0, 1.0))
        else:
            opp_score = 1.0
        
        exit_score = round(
            w_norm_r * norm_r +
            w_regime * regime_intact +
            w_vol * vol_pct_score +
            w_decay * decay_factor +
            w_opp * opp_score, 3
        )

        # LEVEL 1: Take Profit
        if (is_long and current_price >= take_profit) or (not is_long and current_price <= take_profit):
            return self._build_exit_result(1, "Take Profit Hit", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 2: Stop Loss
        if (is_long and current_price <= stop_loss) or (not is_long and current_price >= stop_loss):
            return self._build_exit_result(2, "Stop Loss Hit", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 3: Catastrophic Risk Controls
        if current_r <= -3.0:
            return self._build_exit_result(3, "Catastrophic Risk Limit Breach", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 4: Expected R Collapse (after soft timeout AND trade must be underwater)
        # A trade at breakeven or profit should NOT be killed by time decay alone
        if candles_elapsed >= soft_limit and decayed_expected_r < 0.25 and current_r < 0.0:
            return self._build_exit_result(4, "Expected R Collapse (<0.25R)", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 5: Opportunity Cost Rotation (Recommendation 3)
        if portfolio_heat_full and incoming_signal_expected_r and incoming_signal_expected_r > (expected_r * 1.5):
            return self._build_exit_result(5, f"Opportunity Cost Rotation (New signal {incoming_signal_expected_r:.2f}R > 1.5x active {expected_r:.2f}R)", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 6: Regime Shift
        if entry_regime.upper() == "TRENDING" and current_regime.upper() == "RANGING" and candles_elapsed >= soft_limit:
            return self._build_exit_result(6, "Regime Shift (Trending -> Ranging)", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 7: Volatility Collapse (Recommendation 6: relative to 20th percentile)
        if candles_elapsed >= soft_limit and garch_vol < rolling_vol_20th_pct and abs(current_r) < 0.2:
            return self._build_exit_result(7, "Volatility Collapse (<20th percentile vol)", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 8: MFE Stagnation (Recommendation 4)
        if mfe_r >= 2.0 and current_r < 0.3 and candles_elapsed >= 12:
            return self._build_exit_result(8, f"MFE Stagnation (Reached +{mfe_r:.1f}R MFE but current PnL is +{current_r:.1f}R)", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 9: Adaptive Time Decay & Exit Score
        if candles_elapsed >= soft_limit and exit_score < 0.35:
            return self._build_exit_result(9, f"Adaptive Time Decay Exit (Unified Exit Score {exit_score:.2f} < 0.35)", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        # LEVEL 10: Conditional Hard Timeout Safety Net (Recommendation 1)
        if candles_elapsed >= hard_limit:
            # Runner Exception: Allow extension if PnL > +2.0R, Expected R >= 0.20R and Regime intact
            if current_r >= 2.0 and decayed_expected_r >= 0.20 and regime_intact == 1.0:
                return self._build_exit_result(10, f"Hard Timeout Runner Extension Granted (PnL: +{current_r:.1f}R, Exp R: {decayed_expected_r:.2f}R)", False, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)
            else:
                return self._build_exit_result(10, "Conditional Hard Timeout Reached", True, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

        return self._build_exit_result(0, "Hold Active Trade", False, current_r, decayed_expected_r, exit_score, candles_elapsed, soft_limit, hard_limit)

    def _build_exit_result(
        self,
        level: int,
        reason: str,
        should_exit: bool,
        current_r: float,
        decayed_expected_r: float,
        exit_score: float,
        candles_elapsed: int,
        soft_limit: int,
        hard_limit: int
    ) -> Dict[str, Any]:
        """Builds structured attribution report for exit decisions."""
        return {
            "should_exit": should_exit,
            "exit_level": level,
            "exit_reason": reason,
            "current_r": current_r,
            "decayed_expected_r": decayed_expected_r,
            "exit_score": exit_score,
            "candles_elapsed": candles_elapsed,
            "soft_limit_candles": soft_limit,
            "hard_limit_candles": hard_limit,
            "attribution": {
                "level": level,
                "reason": reason,
                "timestamp": time.time(),
                "exit_score": exit_score,
                "decayed_expected_r": decayed_expected_r
            }
        }


class PortfolioUtilityOptimizer:
    """
    Institutional Portfolio Utility Optimizer:
    Evaluates all active open trades in the portfolio simultaneously.
    Ranks trades by Net Utility, identifies low-utility trades (stagnant/low expected return)
    and commands early close or scale-out to harvest margin for high-utility runner trades.
    """
    @staticmethod
    def optimize_portfolio_capital(
        active_trades: Dict[str, Dict[str, Any]],
        portfolio_heat_max: float = 0.80
    ) -> Dict[str, Any]:
        """
        Ranks active trades by current_r / net_utility.
        Returns rebalancing recommendations dict:
        - "close_trades": list of trade_ids to harvest margin
        - "scale_out_trades": list of trade_ids to reduce size by 50%
        - "prioritize_trades": list of high-utility trade_ids to hold/extend
        """
        if not active_trades:
            return {"close_trades": [], "scale_out_trades": [], "prioritize_trades": []}

        ranked_trades = []
        for trade_id, trade in active_trades.items():
            entry_price = trade.get("entry_price", 1.0)
            current_price = trade.get("current_price", entry_price)
            direction = trade.get("direction", "Bullish")
            is_long = str(direction).upper() in ["BUY", "LONG", "BULLISH"]
            
            pnl_pct = ((current_price - entry_price) / max(1e-4, entry_price)) * 100.0 if is_long else ((entry_price - current_price) / max(1e-4, entry_price)) * 100.0
            expected_r = trade.get("expected_r", 1.20)
            net_utility = pnl_pct * expected_r
            
            ranked_trades.append((trade_id, net_utility, trade))

        # Sort descending by net_utility
        ranked_trades.sort(key=lambda x: x[1], reverse=True)

        close_list = []
        scale_out_list = []
        prioritize_list = []

        import config
        util_policy = getattr(config, "PORTFOLIO_UTILITY_POLICY", {})
        min_util = util_policy.get("min_utility_threshold", 0.20)
        max_stag = util_policy.get("max_stagnant_candles", 15)
        scaleout_util = util_policy.get("scaleout_utility_threshold", 0.50)

        total_trades = len(ranked_trades)
        for idx, (t_id, util, t_data) in enumerate(ranked_trades):
            # Do NOT market-close active trades on normal dips — Stop Loss governs risk.
            # Only harvest margin if the trade is confirmed stagnant past max_stagnant_candles.
            candles = t_data.get("candles_elapsed", 0)
            if util < min_util and candles > max_stag:
                close_list.append(t_id)
            elif idx >= total_trades // 2 and util < scaleout_util and not t_data.get("half_closed", False):
                scale_out_list.append(t_id)
            else:
                prioritize_list.append(t_id)

        return {
            "close_trades": close_list,
            "scale_out_trades": scale_out_list,
            "prioritize_trades": prioritize_list,
            "ranked_utility": [(t_id, round(util, 2)) for t_id, util, _ in ranked_trades]
        }


def generate_continuous_policy_vector(regime: str, adx_val: float = 25.0, confidence: float = 0.80) -> Dict[str, Any]:
    """
    Generates continuous policy control vectors without bucket quantization loss.
    """
    r = regime.upper()
    if "STRONG" in r:
        target_mult = 2.2 + min(0.5, (adx_val - 25.0) * 0.02)
        partial_split = [0.20, 0.20, 0.60]
        ext_step = 0.15
        trail = 0.70
    elif "RANGING" in r or "CHOP" in r:
        target_mult = 1.3 + min(0.3, confidence * 0.2)
        partial_split = [0.40, 0.40, 0.20]
        ext_step = 0.25
        trail = 0.50
    else: # MODERATE
        target_mult = 1.6 + min(0.4, (adx_val - 15.0) * 0.02)
        partial_split = [0.30, 0.30, 0.40]
        ext_step = 0.20
        trail = 0.60

    return {
        "target_multiplier": round(target_mult, 3),
        "partial_split": partial_split,
        "extension_step_r": round(ext_step, 2),
        "trail_factor": round(trail, 2)
    }


def log_checksummed_exit_decision(record_dict: Dict[str, Any], filepath: str = "ExitDecisionDataset.jsonl"):
    """
    Logs checksummed exit decision trace record to ExitDecisionDataset.jsonl for counterfactual replay learning.
    """
    meta_record = {
        "schema_version": "3.0.0",
        "timestamp": time.time(),
        "git_commit": "c65080f",
        "policy_checksum": compute_file_sha256(REGISTRY_FILE) if os.path.exists(REGISTRY_FILE) else "DEFAULT_HASH",
        "trade_data": record_dict
    }
    try:
        with open(filepath, "a") as f:
            f.write(json.dumps(meta_record) + "\n")
    except Exception as e:
        print(f"[ExitDecisionDataset Error] Failed to append decision record: {e}")


exit_policy_engine = ExitPolicyEngine()




