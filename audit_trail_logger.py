"""
audit_trail_logger.py
---------------------
Immutable Compliance Trade Audit Trail Stream (MEDIUM-5 Remediation).
Appends cryptographically hashed (SHA-256 HMAC) trade records to an append-only
audit stream file (audit_trail.jsonl) for regulatory and compliance auditing.
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

class ImmutableAuditTrailLogger:
    def __init__(self, log_file: str = AUDIT_LOG_FILE):
        self.log_file = log_file

    def _compute_hmac_signature(self, payload: dict) -> str:
        serialized = json.dumps(payload, sort_keys=True)
        signature = hmac.new(AUDIT_SECRET_KEY, serialized.encode("utf-8"), hashlib.sha256).hexdigest()
        return signature

    def record_trade_audit_event(self, event_type: str, trade_details: dict) -> Dict[str, Any]:
        """
        Records an immutable trade event with SHA-256 HMAC signature.
        """
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

audit_logger = ImmutableAuditTrailLogger()
