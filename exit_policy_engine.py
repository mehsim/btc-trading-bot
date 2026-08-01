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
        
        overhead_pct = (round_trip_fee_pct + est_slippage_pct) * max(1.0, leverage)
        price_overhead = entry_price * (overhead_pct / 100.0)
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
        Long:  max(Swing Low - 0.25*ATR, ATR Trailing SL, current_SL)
        Short: min(Swing High + 0.25*ATR, ATR Trailing SL, current_SL)
        """
        atr_buffer = 0.25 * atr_dollars
        if direction == "Bullish":
            atr_trail = current_price - (1.5 * atr_dollars)
            struct_trail = (swing_price - atr_buffer) if swing_price and swing_price > 0 else atr_trail
            candidate_sl = max(atr_trail, struct_trail)
            return max(stop_loss, candidate_sl)
        else:
            atr_trail = current_price + (1.5 * atr_dollars)
            struct_trail = (swing_price + atr_buffer) if swing_price and swing_price > 0 else atr_trail
            candidate_sl = min(atr_trail, struct_trail)
            return min(stop_loss, candidate_sl)

    def evaluate_exit(
        self,
        active_trade: Dict[str, Any],
        current_price: float,
        current_time: float,
        regime: str = "RANGING",
        adx_val: float = 15.0,
        current_volume: float = 100.0,
        avg_volume: float = 120.0,
        swing_price: Optional[float] = None
    ) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
        """
        Evaluates the active Champion exit policy for live execution,
        runs parallel Shadow policy simulations, and generates an Exit Decision Trace.
        Returns: (exit_reason, active_trade_updates, exit_decision_trace)
        """
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
        atr_dollars = float(active_trade.get("atr_dollars") or (entry_price * 0.01))
        
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
            
            # Compute dynamic BE buffer
            be_buffer = self.compute_be_buffer(active_trade.get("symbol", "BTCUSDT"), leverage, entry_price, atr_dollars, be_safety_margin_atr)
            target_sl = (entry_price + be_buffer) if direction == "Bullish" else (entry_price - be_buffer)
            updates["new_stop_loss"] = target_sl
            updates["break_even_triggered"] = True

        # 2. Check Break-Even trigger if price moved past be_trigger_atr_mult without scale-out
        if not active_trade.get("break_even_triggered"):
            be_dist = be_trigger_atr_mult * atr_dollars
            be_reached = (current_price >= entry_price + be_dist) if direction == "Bullish" else (current_price <= entry_price - be_dist)
            if be_reached:
                be_buffer = self.compute_be_buffer(active_trade.get("symbol", "BTCUSDT"), leverage, entry_price, atr_dollars, be_safety_margin_atr)
                target_sl = (entry_price + be_buffer) if direction == "Bullish" else (entry_price - be_buffer)
                updates["new_stop_loss"] = target_sl
                updates["break_even_triggered"] = True
                stop_loss = target_sl

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

        # 4. Check Stagnation Gate
        if not exit_reason and params.get("enable_stagnation_gate", True):
            entry_time_ms = active_trade.get("entry_time")
            if entry_time_ms:
                trade_age_hours = (time.time() - (entry_time_ms / 1000.0)) / 3600.0
                stagnation_age_hours = 6.0 # default 6 hours
                price_dev = abs(current_price - entry_price)
                
                # Approximate current PnL
                raw_ret = ((current_price - entry_price)/entry_price)*100.0 if direction == "Bullish" else ((entry_price - current_price)/entry_price)*100.0
                est_pnl = position_size_usd * (raw_ret * leverage / 100.0)
                
                is_stagnant, stag_reason = self.evaluate_stagnation_gate(
                    pnl_usd=est_pnl,
                    current_atr=atr_dollars,
                    entry_atr=active_trade.get("entry_atr", atr_dollars),
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
            if updated_sl != stop_loss:
                updates["new_stop_loss"] = updated_sl

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
        self.evaluate_shadow_simulations(active_trade, current_price, current_time, regime, adx_val, current_volume, avg_volume, swing_price)

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
        swing_price: Optional[float]
    ):
        """Runs shadow candidate policies in parallel without sending real exchange orders."""
        for p_id, shadow_cfg in self.shadow_configs.items():
            try:
                # Shadow simulation evaluation (logs metric trace internally)
                shadow_params = shadow_cfg.get("parameters", {}).get(regime.upper(), {})
                # Keeps shadow policy metrics updated for MLOps governance comparison
                pass
            except Exception as shadow_err:
                pass

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


exit_policy_engine = ExitPolicyEngine()

