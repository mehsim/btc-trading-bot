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
        expired = [k for k, t in self._cache.items() if now - t > self.ttl]
        for k in expired:
            del self._cache[k]

idempotency_cache = IdempotencyCache(ttl_seconds=60.0)

def generate_client_order_id(symbol: str, side: str) -> str:
    ts = int(time.time() * 1000)
    nonce = str(uuid.uuid4())[:6]
    clean_sym = symbol.replace("/", "").replace("-", "")
    cl_id = f"BOT_{clean_sym}{side.upper()}{ts}_{nonce}"
    idempotency_cache.add(cl_id)
    return cl_id

def calculate_exponential_backoff_with_jitter(attempt: int, base_delay: float = 0.5, max_delay: float = 30.0, jitter_pct: float = 0.20) -> float:
    delay = min(max_delay, base_delay * (2 ** attempt))
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
