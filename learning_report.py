"""
learning_report.py
-------------------
Phase 1A: Automated Research Report Generator.
Generates institutional Markdown/Text research summaries for completed trades.
"""

from typing import Dict, Any
from trade_calculators import safe_float

def generate_trade_learning_report(record: Dict[str, Any]) -> str:
    trade_id = record.get("trade_id", "UNKNOWN")
    symbol = record.get("symbol", "UNKNOWN")
    outcome = record.get("trade_outcome", "UNKNOWN")
    pnl = safe_float(record.get("pnl_usd", 0.0))
    realized_r = safe_float(record.get("realized_r", 0.0))
    conf = safe_float(record.get("confidence", 0.0)) * 100.0
    regime = record.get("market_regime", "TRENDING")
    
    snap = record.get("decision_snapshot", {})
    reason_codes = snap.get("reason_codes", ["N/A"]) if isinstance(snap, dict) else ["N/A"]
    
    attrib = record.get("failure_attribution", {})
    
    report = f"""
===================================================================
  INSTITUTIONAL TRADE RESEARCH REPORT — {symbol} ({trade_id})
===================================================================

  Trade Outcome:         {outcome} (${pnl:+.2f} | {realized_r:+.2f}R)
  Model Confidence:      {conf:.1f}%
  Market Regime:         {regime}
  Trigger Reason Codes:  {', '.join(reason_codes)}

  --- DIAGNOSTICS & ATTRIBUTION ---
"""
    if outcome == "WIN":
        report += "  Status: Successful trade. Expected setup parameters validated.\n"
    else:
        report += f"  Failure Attribution Breakdown:\n"
        if isinstance(attrib, dict) and attrib:
            for factor, data in attrib.items():
                if isinstance(data, dict):
                    pct = data.get("pct", 0)
                    evidence = data.get("evidence", "")
                    report += f"    • {factor}: {pct}% ({evidence})\n"
                else:
                    report += f"    • {factor}: {data}\n"
        else:
            report += "    • Diagnostic analysis pending sample aggregation.\n"
            
    brier = safe_float(record.get("individual_brier_loss", 0.0))
    report += f"\n  Individual Brier Loss: {brier:.4f}\n"
    report += "===================================================================\n"
    return report

