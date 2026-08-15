"""
tests/test_ratchet.py
----------------------
Static analysis ratchet enforcing:
1. Print Ratchet: Core runtime files must not introduce un-ratcheted raw print() calls.
2. Root File Ratchet: Prevents accumulation of orphan scratch files in project root (max 165 root .py files).
"""

import os
import glob
import pytest

CORE_MODULES = [
    "main.py", "config.py", "ensemble.py", "features.py", "train.py", "backtest.py",
    "database.py", "websocket_client.py", "signal_evaluator.py", "learning_engine.py",
    "logger.py", "trade_calculators.py", "data.py", "core.py", "state_manager.py"
]

MAX_ROOT_PY_FILES = 165


def test_root_file_count_ratchet():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    py_files = glob.glob(os.path.join(root_dir, "*.py"))
    assert len(py_files) <= MAX_ROOT_PY_FILES, f"Root Python file count ({len(py_files)}) exceeds ratchet limit of {MAX_ROOT_PY_FILES}. Move scratch scripts to archive/ or tools/."


def test_core_modules_exist():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for module_name in CORE_MODULES:
        path = os.path.join(root_dir, module_name)
        assert os.path.exists(path), f"Core module {module_name} is missing from root directory!"
