from typing import Optional
"""
logger.py
---------
Structured JSON logger with correlation_id tracking, log-level filtering,
and ISO-8601 UTC timestamps for high-reliability observability.
"""

import json
import logging
import uuid
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", str(uuid.uuid4())),
            "module": record.module,
            "line_no": record.lineno
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logger(name="trading_bot", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger

bot_logger = setup_logger("trading_bot")

def log_event(level: str, msg: str, correlation_id: Optional[str] = None, extra: Optional[dict] = None):
    lvl = getattr(logging, level.upper(), logging.INFO)
    extra_dict = {"correlation_id": correlation_id or str(uuid.uuid4())}
    if extra:
        extra_dict.update(extra)
    bot_logger.log(lvl, msg, extra=extra_dict)
