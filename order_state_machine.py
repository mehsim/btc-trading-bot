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

    def update_fill(self, fill_qty: float, fill_price: float, is_cumulative: bool = False):
        if is_cumulative:
            self.filled_qty = max(self.filled_qty, float(fill_qty))
        else:
            self.filled_qty += float(fill_qty)
        if self.filled_qty >= self.qty:
            self.transition_to(OrderState.FILLED, f"Filled {self.filled_qty}/{self.qty} @ {fill_price}")
        else:
            self.transition_to(OrderState.PARTIALLY_FILLED, f"Partial fill {self.filled_qty}/{self.qty} @ {fill_price}")


class StopState(Enum):
    INITIAL = "INITIAL"
    TRAILING = "TRAILING"
    BREAK_EVEN = "BREAK_EVEN"
    PROFIT_LOCK = "PROFIT_LOCK"
    STRUCTURAL_TRAIL = "STRUCTURAL_TRAIL"
    FINAL_RUNNER = "FINAL_RUNNER"

STATE_HIERARCHY_RANK = {
    StopState.INITIAL: 1,
    StopState.TRAILING: 2,
    StopState.BREAK_EVEN: 3,
    StopState.PROFIT_LOCK: 4,
    StopState.STRUCTURAL_TRAIL: 5,
    StopState.FINAL_RUNNER: 6,
}

class StopStateMachine:
    """
    Institutional Stop State Machine & Monotonic Invariant Validator.
    Enforces strict forward state transitions and monotonic locked risk rules:
    LockedRisk(t+1) >= LockedRisk(t)  (for Longs: SL(t+1) >= SL(t))
    """
    @staticmethod
    def can_transition(current_state_str: str, target_state_str: str) -> bool:
        try:
            curr_state = StopState[current_state_str] if isinstance(current_state_str, str) else current_state_str
            targ_state = StopState[target_state_str] if isinstance(target_state_str, str) else target_state_str
        except (KeyError, TypeError):
            return True # fallback if unmapped string
        
        curr_rank = STATE_HIERARCHY_RANK.get(curr_state, 1)
        targ_rank = STATE_HIERARCHY_RANK.get(targ_state, 1)

        # Monotonic state rank requirement: target_rank >= curr_rank
        return targ_rank >= curr_rank

    @staticmethod
    def validate_monotonic_stop_update(
        direction: str,
        current_sl: float,
        proposed_sl: float,
        current_state_str: str,
        target_state_str: str
    ) -> tuple[bool, str]:
        """
        Validates monotonic stop price movement:
        Long: proposed_sl >= current_sl
        Short: proposed_sl <= current_sl
        State: target_state rank >= current_state rank
        """
        if not StopStateMachine.can_transition(current_state_str, target_state_str):
            return False, f"Illegal backward state transition from {current_state_str} to {target_state_str}"

        is_long = str(direction).upper() in ["BUY", "LONG", "BULLISH"]
        if is_long:
            if proposed_sl < current_sl - 1e-4:
                return False, f"Monotonic violation for Long: proposed SL {proposed_sl:.4f} < current SL {current_sl:.4f}"
        else:
            if proposed_sl > current_sl + 1e-4:
                return False, f"Monotonic violation for Short: proposed SL {proposed_sl:.4f} > current SL {current_sl:.4f}"

        return True, "Monotonic Stop Invariant Passed"

