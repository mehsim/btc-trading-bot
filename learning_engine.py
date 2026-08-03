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

# Background Queue for Non-Blocking Processing
learning_event_queue = queue.Queue(maxsize=1000)

class ContinuousLearningEngine:
    def __init__(self):
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
            trade_id = trade.get("trade_id") or f"{trade.get('symbol')}_{int(trade.get('exit_time', time.time()))}"
            symbol = trade.get("symbol", "BTCUSDT")
            pnl = float(trade.get("pnl_usd", 0.0))
            conf = float(trade.get("confidence", 0.60))
            is_win = (pnl >= 0)
            
            # 1. Feature Health Inspection
            is_healthy, health_issues = feature_health_monitor.inspect_record(trade)
            if not is_healthy:
                print(f"[LearningEngine Warning] Trade {trade_id} has health issues: {health_issues}")
                
            # 2. Calibration Tracking
            record_trade_outcome(confidence=conf, is_win=is_win)
            
            # 3. Decision Snapshot
            raw_snap = trade.get("decision_snapshot")
            if not isinstance(raw_snap, dict):
                raw_snap = build_decision_snapshot(
                    symbol=symbol,
                    direction=trade.get("direction", "LONG"),
                    confidence=conf,
                    market_regime=trade.get("interval", "60")
                )
                
            # 4. Failure Attribution Diagnosis
            attribution = failure_attribution_engine.diagnose_loss(trade) if not is_win else {}
            
            # 5. Decision Outcome Replay
            replay = decision_outcome_replay.replay_trade(trade)
            
            # 6. Counterfactual Scenarios
            cf_scenarios = counterfactual_engine.evaluate_scenarios(trade)
            
            # 7. Calculate Learning Score & Enqueue if >= 80
            individual_brier = round((conf - (1.0 if is_win else 0.0)) ** 2, 4)
            trade["individual_brier_loss"] = individual_brier
            learning_score = calculate_learning_score(trade, cf_scenarios)
            
            # 8. Regime Memory Tagging
            regime_type = trade.get("market_regime", "TRENDING")
            realized_r = float(trade.get("realized_r", 0.0))
            regime_id = record_regime_trade(regime_type=regime_type, pnl_usd=pnl, realized_r=realized_r)
            
            # 9. Save Complete Trade Experience Record
            exp_record = dict(trade)
            exp_record["trade_id"] = trade_id
            exp_record["decision_snapshot"] = raw_snap
            exp_record["failure_attribution"] = attribution
            exp_record["decision_replay"] = replay
            exp_record["learning_score"] = learning_score
            exp_record["regime_id"] = regime_id
            save_trade_experience(exp_record)
            
            # 10. SHAP Recording
            raw_shap = trade.get("shap_values")
            if isinstance(raw_shap, dict):
                save_shap_record(trade_id=trade_id, symbol=symbol, shap_dict=raw_shap)
                
            # 11. Pattern Mining & Rule Generation (Stage 1 Rule-Based)
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
                        "recommendation": f"Shrink position size on {p['cluster_key']} (Win Rate: {p['win_rate']*100:.1f}%)"
                    })
                    
            # 12. Learning Report Generation
            report = generate_trade_learning_report(exp_record)
            print(f"[LearningEngine] Processed trade {trade_id} (Score: {learning_score:.0f}/100)")
            
        except Exception as e:
            print(f"[LearningEngine Error] Non-critical ingest error for trade {trade.get('trade_id')}: {e}")

    def on_trade_closed(self, trade_record: dict):
        """
        Public entry point called by save_completed_trade() hook.
        Non-blocking: pushes event into learning_event_queue.
        """
        try:
            learning_event_queue.put_nowait({"type": "TRADE_CLOSED", "trade": trade_record})
        except Exception as e:
            print(f"[LearningEngine Queue Error] {e}")

    def get_risk_multiplier(self, signal_context: dict) -> float:
        """
        Public risk multiplier query for position sizer.
        Returns float in range [0.50, 1.00].
        """
        try:
            return risk_multiplier_engine.get_risk_multiplier(signal_context)
        except Exception as e:
            print(f"[LearningEngine RiskMult Error] {e}")
            return 1.0

continuous_learning_engine = ContinuousLearningEngine()
