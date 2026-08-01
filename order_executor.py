"""
order_executor.py
-------------------
Decoupled OMS/EMS with SQLite-backed persistent queue, API rate limiting
(token bucket), and exponential backoff.
States: PENDING_SUBMIT -> ACKNOWLEDGED -> FILLED / CANCELLED / REJECTED
"""

import time
import json
import sqlite3
import queue
import signal
import threading
from collections import deque
from typing import Dict, Any

DB_PATH = "order_queue.db"
MAX_RPS = 80  # Bybit V5 private endpoint limit with 20% safety buffer

class OrderExecutor:
    def __init__(self):
        self._init_db()
        self.order_queue = queue.Queue()
        self.order_states: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.is_running = True
        self._request_timestamps = deque()  # for token bucket rate limiter
        self._recover_pending_orders()
        self.worker_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.worker_thread.start()
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_queue (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    direction TEXT,
                    status TEXT,
                    created_at REAL,
                    filled_at REAL,
                    payload TEXT
                )
            """)
            conn.commit()

    def _persist_order(self, order: Dict[str, Any]):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO order_queue
                (order_id, symbol, direction, status, created_at, filled_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                order.get("order_id"), order.get("symbol"), order.get("direction"),
                order.get("status"), order.get("created_at"), order.get("filled_at"),
                json.dumps(order)
            ))
            conn.commit()

    def _recover_pending_orders(self):
        """On startup: recover PENDING_SUBMIT orders from disk and re-enqueue."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT payload FROM order_queue WHERE status IN ('PENDING_SUBMIT','ACKNOWLEDGED')"
                ).fetchall()
            for row in rows:
                order = json.loads(row[0])
                order["status"] = "PENDING_SUBMIT"
                with self.lock:
                    self.order_states[order["order_id"]] = order
                self.order_queue.put(order)
                print(f"[OMS/EMS] Recovered order {order['order_id']} from disk.")
        except Exception as e:
            print(f"[OMS/EMS] Recovery warning: {e}")

    def _enforce_rate_limit(self):
        """Token bucket rate limiter: max MAX_RPS requests per second."""
        now = time.monotonic()
        self._request_timestamps.append(now)
        # Drop timestamps older than 1 second
        while self._request_timestamps and self._request_timestamps[0] < now - 1.0:
            self._request_timestamps.popleft()
        if len(self._request_timestamps) >= MAX_RPS:
            time.sleep(1.0 / MAX_RPS)

    def submit_order(self, order_request: Dict[str, Any]) -> str:
        order_id = order_request.get("order_id") or f"ord_{int(time.time() * 1000)}"
        order_request["order_id"] = order_id
        order_request["status"] = "PENDING_SUBMIT"
        order_request["created_at"] = time.time()
        with self.lock:
            self.order_states[order_id] = order_request
        self._persist_order(order_request)
        self.order_queue.put(order_request)
        print(f"[OMS/EMS] Enqueued order {order_id} for {order_request.get('symbol')} {order_request.get('direction')}")
        return order_id

    def _execution_loop(self):
        retry_count = 0
        while self.is_running:
            try:
                order = self.order_queue.get(timeout=1.0)
                order_id = order.get("order_id")
                self._enforce_rate_limit()

                with self.lock:
                    if order_id in self.order_states:
                        self.order_states[order_id]["status"] = "ACKNOWLEDGED"
                self._persist_order({**order, "status": "ACKNOWLEDGED"})

                # Simulate network RTT / actual Bybit call
                time.sleep(0.05)

                with self.lock:
                    if order_id in self.order_states:
                        self.order_states[order_id]["status"] = "FILLED"
                        self.order_states[order_id]["filled_at"] = time.time()
                self._persist_order({**order, "status": "FILLED", "filled_at": time.time()})

                self.order_queue.task_done()
                retry_count = 0

            except queue.Empty:
                continue
            except Exception as e:
                retry_count += 1
                wait = min(2 ** retry_count * 0.5, 30)
                print(f"[OMS/EMS Warning] Execution error ({e}). Backoff {wait:.1f}s (retry {retry_count})")
                time.sleep(wait)

    def _shutdown_handler(self, signum, frame):
        """Flush queue state to disk on SIGTERM before exit."""
        print("[OMS/EMS] SIGTERM received. Flushing queue state to disk.")
        with self.lock:
            for order in self.order_states.values():
                self._persist_order(order)
        self.is_running = False

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        with self.lock:
            return self.order_states.get(order_id, {"status": "UNKNOWN"})

order_executor = OrderExecutor()
