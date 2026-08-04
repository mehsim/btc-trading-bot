import re
import math
import numpy as np
from typing import Optional


def sanitize_symbol(symbol: Optional[str]) -> str:
    """Sanitizes and validates crypto trading symbols (e.g. BTCUSDT)."""
    if not symbol:
        return "BTCUSDT"
    
    clean_sym = re.sub(r'[^A-Za-z0-9]', '', str(symbol)).upper()
    if not clean_sym:
        return "BTCUSDT"
        
    if not clean_sym.endswith("USDT") and not clean_sym.endswith("PERP"):
        clean_sym += "USDT"
        
    return clean_sym

def validate_price(price: float, default_price: float = 0.0, max_price: float = 500000.0) -> float:
    """Validates and ensures non-negative, finite price values bounded within valid market range."""
    try:
        val = float(price)
        if val <= 0.0 or val > max_price or np.isnan(val) or np.isinf(val):
            return float(default_price)

        return val
    except Exception:
        return float(default_price)

def validate_quantity(qty: float, min_qty: float = 0.0001) -> float:
    """Validates non-negative order quantity."""
    try:
        val = float(qty)
        if val <= 0.0 or np.isnan(val) or np.isinf(val):
            return float(min_qty)
        return val
    except Exception:
        return float(min_qty)
