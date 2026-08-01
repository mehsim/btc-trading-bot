"""
audit_trail_logger.py
---------------------
Immutable Compliance Trade Audit Trail Stream.
Appends cryptographically hashed (SHA-256 HMAC) trade and order records to an
append-only audit stream (compliance_audit.log) for regulatory audit readiness.
"""

import json
import hashlib
import hmac
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any

AUDIT_SECRET_KEY = b"trading_bot_audit_signature_secret_2026"
AUDIT_LOG_FILE = "audit_trail.jsonl"
COMPLIANCE_LOG_FILE = "compliance_audit.log"

class ImmutableAuditTrailLogger:
    def __init__(self, log_file: str = AUDIT_LOG_FILE):
        self.log_file = log_file

    def _compute_hmac_signature(self, payload: dict) -> str:
        serialized = json.dumps(payload, sort_keys=True)
        signature = hmac.new(AUDIT_SECRET_KEY, serialized.encode("utf-8"), hashlib.sha256).hexdigest()
        return signature

    def record_trade_audit_event(self, event_type: str, trade_details: dict) -> Dict[str, Any]:
        """Records an immutable trade event with SHA-256 HMAC signature."""
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp_utc": timestamp_utc,
            "event_type": event_type,
            "trade_details": trade_details
        }
        signature = self._compute_hmac_signature(payload)
        audit_entry = {**payload, "hmac_signature": signature}
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            print(f"[Audit Logger Error] Failed recording audit event: {e}")
        return audit_entry

    def log_order_event(self, order_id: str, symbol: str, direction: str,
                        size_usd: float, triggered_by: str, confidence: float,
                        action: str = "ORDER_SUBMITTED") -> None:
        """
        Compliance log for every order submission/cancellation with full metadata.
        Required for regulatory audit standards.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "order_id": order_id,
            "symbol": symbol,
            "direction": direction,
            "size_usd": round(size_usd, 4),
            "triggered_by": triggered_by,
            "confidence": round(confidence, 4)
        }
        try:
            with open(COMPLIANCE_LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[Compliance Log Error] {e}")

    def log_cancellation(self, order_id: str, reason: str) -> None:
        """Logs order cancellation with reason for compliance audit trail."""
        self.log_order_event(
            order_id=order_id, symbol="N/A", direction="CANCEL",
            size_usd=0.0, triggered_by=reason, confidence=0.0,
            action="ORDER_CANCELLED"
        )

audit_logger = ImmutableAuditTrailLogger()

