"""
learning_engine.py
-------------------
Phase 1 Central Continuous Learning Engine & Event Queue.
Coordinates non-blocking post-trade ingestion, diagnosis, pattern mining, calibration tracking,
counterfactual evaluation, and risk multiplier resolution.
"""

import threading
import queue
import time
from typing import Dict, Any

from experience_db import save_trade_experience
from decision_snapshot import build_decision_snapshot
from feature_health_monitor import feature_health_monitor
from calibration_tracker import record_trade_outcome
from failure_attribution_engine import failure_attribution_engine
from decision_outcome_replay import decision_outcome_replay
from pattern_miner import pattern_miner
from knowledge_base import save_rule, get_active_rules
from risk_multiplier import risk_multiplier_engine
from regime_memory import record_regime_trade
from counterfactual_engine import counterfactual_engine
from learning_scorer import calculate_learning_score
from drift_monitor import drift_monitor
from shap_store import save_shap_record
from learning_report import generate_trade_learning_report
from audit_logger import log_learning_action
from schema_validator import schema_validator

from trade_calculators import safe_float
from feature_availability import record_feature_sample
from regime_transition_analyzer import record_transition_trade

LEARNING_ENGINE_VERSION = "v1.0.0"

# Background Queue for Non-Blocking Processing
learning_event_queue = queue.Queue(maxsize=1000)

class ContinuousLearningEngine:
    def __init__(self):
        self._worker_thread = None
        self._lock = threading.Lock()

    def _ensure_worker_started(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            with self._lock:
                if self._worker_thread is None or not self._worker_thread.is_alive():
                    self._worker_thread = threading.Thread(target=self._event_loop_worker, daemon=True)
                    self._worker_thread.start()

    def _event_loop_worker(self):
        while True:
            try:
                event = learning_event_queue.get()
                if event and isinstance(event, dict):
                    event_type = event.get("type")
                    if event_type == "TRADE_CLOSED":
                        trade = event.get("trade", {})
                        self._process_trade_closed(trade)
                learning_event_queue.task_done()
            except Exception as e:
                print(f"[LearningEngine Worker Error] {e}")

    def _process_trade_closed(self, trade: Dict[str, Any]):
        """
        Executes the full Phase 1 learning pipeline for a closed trade.
        Safe — wrapped in try/except so failures never affect main trade processing.
        """
        try:
            trade_id = trade.get("trade_id") or f"{trade.get('symbol')}_{int(safe_float(trade.get('exit_time'), time.time()))}"
            symbol = trade.get("symbol", "BTCUSDT")
            pnl = safe_float(trade.get("pnl_usd", 0.0))
            conf = safe_float(trade.get("confidence", 0.60), 0.60)
            is_win = (pnl >= 0)
            
            log_learning_action("TRADE_CLOSED_RECEIVED", "learning_engine", trade_id=trade_id, details={"symbol": symbol, "pnl": pnl})
            
            # 1. Feature Availability Tracking
            features_to_track = ["adx", "atr_pct", "rsi", "funding_rate", "oi_z_score", "confidence"]
            for f in features_to_track:
                v = trade.get(f)
                record_feature_sample(f, is_available=(v is not None and not (isinstance(v, float) and (v != v))))
            
            # 2. Feature Health Inspection & Schema Validation
            is_healthy, health_issues = feature_health_monitor.inspect_record(trade)
            is_valid, schema_errors = schema_validator.validate_record(trade)
            if not is_healthy or not is_valid:
                log_learning_action("SCHEMA_WARNING", "schema_validator", trade_id=trade_id, details={"health_issues": health_issues, "schema_errors": schema_errors})
                
            # 3. Calibration Tracking
            record_trade_outcome(confidence=conf, is_win=is_win)
            log_learning_action("CALIBRATION_UPDATED", "calibration_tracker", trade_id=trade_id)
            
            # 4. Decision Snapshot
            raw_snap = trade.get("decision_snapshot")
            if not isinstance(raw_snap, dict):
                raw_snap = build_decision_snapshot(
                    symbol=symbol,
                    direction=trade.get("direction", "LONG"),
                    confidence=conf,
                    market_regime=trade.get("interval", "60")
                )
                
            # 5. Failure Attribution Diagnosis
            attribution = failure_attribution_engine.diagnose_loss(trade) if not is_win else {}
            if attribution:
                log_learning_action("ATTRIBUTION_DIAGNOSED", "failure_attribution_engine", trade_id=trade_id, details=attribution)
            
            # 6. Decision Outcome Replay
            replay = decision_outcome_replay.replay_trade(trade)
            
            # 7. Counterfactual Scenarios
            cf_scenarios = counterfactual_engine.evaluate_scenarios(trade)
            
            # 8. Calculate Learning Score & Enqueue if >= 80
            individual_brier = round((conf - (1.0 if is_win else 0.0)) ** 2, 4)
            trade["individual_brier_loss"] = individual_brier
            learning_score = calculate_learning_score(trade, cf_scenarios)
            
            # 9. Regime Memory Tagging & Transition Analyzer
            regime_type = trade.get("market_regime", "TRENDING")
            from_regime = trade.get("prev_market_regime", regime_type)
            realized_r = safe_float(trade.get("realized_r", 0.0))
            regime_id = record_regime_trade(regime_type=regime_type, pnl_usd=pnl, realized_r=realized_r)
            if from_regime != regime_type:
                record_transition_trade(from_regime, regime_type, is_win, realized_r)
                log_learning_action("REGIME_TRANSITION_RECORDED", "regime_transition_analyzer", trade_id=trade_id, details={"from": from_regime, "to": regime_type})
            
            # 10. Save Complete Trade Experience Record
            exp_record = dict(trade)
            exp_record["trade_id"] = trade_id
            exp_record["learning_engine_version"] = LEARNING_ENGINE_VERSION
            exp_record["decision_snapshot"] = raw_snap
            exp_record["failure_attribution"] = attribution
            exp_record["decision_replay"] = replay
            exp_record["learning_score"] = learning_score
            exp_record["regime_id"] = regime_id
            save_trade_experience(exp_record)
            
            # 11. SHAP Recording
            raw_shap = trade.get("shap_values")
            if isinstance(raw_shap, dict):
                save_shap_record(trade_id=trade_id, symbol=symbol, shap_dict=raw_shap)
                
            # 12. Pattern Mining & Rule Generation (With Provenance & Recency Decay)
            patterns = pattern_miner.mine_patterns(limit=200)
            for p in patterns:
                if p["is_significant"] and p["win_rate"] < 0.45:
                    rule_id = f"RULE_{p['cluster_key'].replace('|', '_')}"
                    save_rule({
                        "rule_id": rule_id,
                        "cluster_key": p["cluster_key"],
                        "sample_size": p["sample_size"],
                        "win_rate": p["win_rate"],
                        "avg_r": p["avg_r"],
                        "ci_lower": p["ci_95_lower"],
                        "ci_upper": p["ci_95_upper"],
                        "evidence_score": round(min(100.0, p["sample_size"] * 2.0), 1),
                        "recommendation": f"Shrink position size on {p['cluster_key']} (Win Rate: {p['win_rate']*100:.1f}%)",
                        "trade_ids": p.get("trade_ids", [])
                    })
                    log_learning_action("KNOWLEDGE_RULE_EVALUATED", "knowledge_base", trade_id=trade_id, details={"rule_id": rule_id})
                    
            # 13. Learning Report Generation
            report = generate_trade_learning_report(exp_record)
            log_learning_action("LEARNING_REPORT_GENERATED", "learning_report", trade_id=trade_id, details={"score": learning_score})
            print(f"[LearningEngine] Processed trade {trade_id} (Score: {learning_score:.0f}/100 | Ver: {LEARNING_ENGINE_VERSION})")
            
        except Exception as e:
            import traceback
            print(f"[LearningEngine Error] Non-critical ingest error for trade {trade.get('trade_id')}: {e}\n{traceback.format_exc()}")


    def on_trade_closed(self, trade_record: dict):
        """
        Public entry point called by save_completed_trade() hook.
        Pushes event into learning_event_queue with timeout & saturation tracking.
        """
        try:
            self._ensure_worker_started()
            learning_event_queue.put_nowait({"type": "TRADE_CLOSED", "trade": trade_record})
        except queue.Full:
            t_id = trade_record.get('trade_id') if isinstance(trade_record, dict) else 'unknown'
            print(f"[LearningEngine Queue Full] Learning event queue saturated; non-blockingly dropped event for trade {t_id}")
        except Exception as e:
            t_id = trade_record.get('trade_id') if isinstance(trade_record, dict) else 'unknown'
            print(f"[LearningEngine Queue Error] Failed to enqueue learning event for trade {t_id}: {e}")

    def get_risk_multiplier(self, signal_context: dict) -> float:
        """
        Public risk multiplier query for position sizer.
        Returns float in range [0.50, 1.00]. Fails safe to 0.75 on exception.
        """
        try:
            return risk_multiplier_engine.get_risk_multiplier(signal_context)
        except Exception as e:
            print(f"[LearningEngine RiskMult Fail-Safe] Exception evaluating risk multiplier: {e}. Falling back to 0.75.")
            return 0.75

continuous_learning_engine = ContinuousLearningEngine()
