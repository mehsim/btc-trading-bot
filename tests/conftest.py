"""
tests/conftest.py
-----------------
Pytest global configuration & concurrency race detection harness.
Sets micro-switch interval (1 microsecond) to widen race condition windows ~1000x
and enables faulthandler to dump stack trace and terminate hangs on deadlocks.
"""

import sys
import faulthandler
import pytest

# Widen race condition windows ~1000x during test execution
sys.setswitchinterval(0.000001)

# Enable automatic deadlock / hang detection: dumps tracebacks if test suite hangs > 120s
try:
    faulthandler.dump_traceback_later(120, exit=True)
except Exception:
    pass


import os

if os.path.exists(".manifest_hmac_secret"):
    with open(".manifest_hmac_secret", "r") as f:
        _sec = f.read().strip()
    if _sec:
        os.environ.setdefault("MANIFEST_HMAC_SECRET", _sec)
else:
    os.environ.setdefault("MANIFEST_HMAC_SECRET", "test-only-deterministic-key-v1")


@pytest.fixture(autouse=True, scope="session")
def _manifest_secret():
    if os.path.exists(".manifest_hmac_secret"):
        with open(".manifest_hmac_secret", "r") as f:
            _sec = f.read().strip()
        if _sec:
            os.environ.setdefault("MANIFEST_HMAC_SECRET", _sec)
    else:
        os.environ.setdefault("MANIFEST_HMAC_SECRET", "test-only-deterministic-key-v1")


@pytest.fixture(autouse=True)
def reset_deadlock_timer():
    """Resets the deadlock timer before each test run."""
    try:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(120, exit=True)
    except Exception:
        pass
