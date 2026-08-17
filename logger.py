from typing import Optional, List, Dict
"""
logger.py
---------
Structured JSON logger with correlation_id tracking, log-level filtering,
ISO-8601 UTC timestamps, in-memory live rolling buffer for dashboard streaming,
and rotating file logging for persistent observability.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
import uuid
import sys
import os
import re
import collections
import threading
from datetime import datetime, timezone

_LIVE_LOG_BUFFER = collections.deque(maxlen=200)
_LIVE_LOG_LOCK = threading.Lock()

def add_to_live_log(msg: str):
    """Add a sanitized, timestamped log line to the in-memory dashboard buffer."""
    if not msg or not msg.strip():
        return
    clean_msg = re.sub(r'\x1b\[[0-9;]*[mK]', '', msg.strip())
    # If JSON formatted, extract the clean readable text
    if clean_msg.startswith("{") and clean_msg.endswith("}"):
        try:
            item = json.loads(clean_msg)
            ts = item.get("timestamp_utc", "")[:19].split("T")[-1]
            lvl = item.get("level", "INFO")
            body = item.get("message", "")
            clean_msg = f"[{ts} UTC] [{lvl}] {body}"
        except (ValueError, TypeError, KeyError):
            pass
    if not clean_msg.startswith("["):
        ts_now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        clean_msg = f"[{ts_now} UTC] {clean_msg}"
    with _LIVE_LOG_LOCK:
        _LIVE_LOG_BUFFER.append(clean_msg)

def get_recent_logs(max_lines: int = 40) -> List[str]:
    """Retrieve the latest live log lines for dashboard rendering."""
    with _LIVE_LOG_LOCK:
        return list(_LIVE_LOG_BUFFER)[-max_lines:]

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

class MemoryLogHandler(logging.Handler):
    """Logging handler that pipes events into the live rolling buffer for dashboard streaming."""
    def emit(self, record):
        try:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            msg = record.getMessage()
            lvl = record.levelname
            add_to_live_log(f"[{ts} UTC] [{lvl}] {msg}")
        except (AttributeError, ValueError, TypeError):
            pass

def setup_logger(name="trading_bot", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        # 1. Stdout JSON handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(JSONFormatter())
        logger.addHandler(stream_handler)
        
        # 2. Live in-memory stream handler for web dashboard
        mem_handler = MemoryLogHandler()
        logger.addHandler(mem_handler)

        # 3. Rotating file handler (bot.log, 5MB max, 2 backups)
        try:
            log_file = "bot.log"
            file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)
        except (IOError, OSError, ValueError):
            pass
    return logger

bot_logger = setup_logger("trading_bot")

def log_event(level: str, msg: str, correlation_id: Optional[str] = None, extra: Optional[dict] = None):
    lvl = getattr(logging, level.upper(), logging.INFO)
    extra_dict = {"correlation_id": correlation_id or str(uuid.uuid4())}
    if extra:
        extra_dict.update(extra)
    bot_logger.log(lvl, msg, extra=extra_dict)
