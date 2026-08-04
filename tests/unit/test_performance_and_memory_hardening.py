import time
import collections
import pandas as pd
import pytest
from confluence_engine import get_valid_htf_cache, set_valid_htf_cache

def test_deque_memory_bounding():
    logs = collections.deque(maxlen=500)
    for i in range(1000):
        logs.append(f"Log message {i}")
    assert len(logs) == 500
    assert logs[0] == "Log message 500"
    assert logs[-1] == "Log message 999"

def test_htf_cache_ttl_eviction():
    htf_cache = {}
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    key = ("BTCUSDT", "240")

    # Set cache with current timestamp
    set_valid_htf_cache(htf_cache, key, df)
    cached_df = get_valid_htf_cache(htf_cache, key, ttl_seconds=900.0)
    assert cached_df is not None
    assert len(cached_df) == 3

    # Manually expire cache entry by setting timestamp 1000 seconds ago
    htf_cache[key] = (df, time.time() - 1000.0)
    expired_df = get_valid_htf_cache(htf_cache, key, ttl_seconds=900.0)
    assert expired_df is None
