import time
import random
import uuid
from enum import Enum

class OrderState(Enum):
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    SETTLED = "SETTLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

import threading

from api_telemetry import global_api_telemetry

class IdempotencyCache:
    def __init__(self, ttl_seconds: float = 60.0):
        self.ttl = ttl_seconds
        self._cache = {}
        self._lock = threading.Lock()

    def is_duplicate(self, client_order_id: str) -> bool:
        with self._lock:
            self._clean()
            return client_order_id in self._cache

    def add(self, client_order_id: str):
        with self._lock:
            self._clean()
            self._cache[client_order_id] = time.time()

    def _clean(self):
        now = time.time()
        _, _, dynamic_ttl = global_api_telemetry.get_telemetry_params()
        self.ttl = dynamic_ttl
        expired = [k for k, t in self._cache.items() if now - t > self.ttl]
        for k in expired:
            del self._cache[k]

idempotency_cache = IdempotencyCache(ttl_seconds=60.0)

def generate_client_order_id(symbol: str, side: str) -> str:
    ts = int(time.time() * 1000)
    nonce = str(uuid.uuid4())[:4]
    clean_sym = symbol.replace("/", "").replace("-", "").replace("USDT", "")[:8]
    cl_id = f"B_{clean_sym}_{side.upper()[0]}_{ts}_{nonce}"[:36]
    idempotency_cache.add(cl_id)
    return cl_id

def calculate_exponential_backoff_with_jitter(attempt: int, base_delay: float = None, max_delay: float = None, jitter_pct: float = 0.20) -> float:
    dyn_base, dyn_max, _ = global_api_telemetry.get_telemetry_params()
    eff_base = base_delay if base_delay is not None else dyn_base
    eff_max = max_delay if max_delay is not None else dyn_max

    delay = min(eff_max, eff_base * (2 ** attempt))
    jitter = delay * jitter_pct * (random.random() * 2 - 1) # ±20% jitter
    return max(0.1, delay + jitter)

class ManagedOrder:
    def __init__(self, client_order_id: str, symbol: str, side: str, order_type: str, qty: float, price: float = None):
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.qty = qty
        self.price = price
        self.filled_qty = 0.0
        self.state = OrderState.SUBMITTED
        self.history = [(time.time(), OrderState.SUBMITTED.value)]

    def transition_to(self, new_state: OrderState, detail: str = ""):
        self.state = new_state
        self.history.append((time.time(), f"{new_state.value}: {detail}".strip()))

    def update_fill(self, fill_qty: float, fill_price: float):
        self.filled_qty += fill_qty
        if self.filled_qty >= self.qty:
            self.transition_to(OrderState.FILLED, f"Filled {self.filled_qty}/{self.qty} @ {fill_price}")
        else:
            self.transition_to(OrderState.PARTIALLY_FILLED, f"Partial fill {self.filled_qty}/{self.qty} @ {fill_price}")
