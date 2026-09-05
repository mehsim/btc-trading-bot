"""
tools/check_print_ratchet.py
-----------------------------
Fix 3 Ratchet: Asserts that unstructured print() calls in core runtime modules do not increase.
Excludes offline CLI tools, training scripts, tests, and standalone scripts.
"""

import ast
import json
import os
import sys

# Dynamic baseline file — shared with check_silent_handlers.py
_BASELINE_FILE = os.path.join(os.path.dirname(__file__), "ratchet_baseline.json")

def _load_baseline() -> int:
    try:
        with open(_BASELINE_FILE) as f:
            return json.load(f).get("print_calls", 999)
    except Exception:
        return 999

def _save_baseline(count: int) -> None:
    data = {}
    try:
        with open(_BASELINE_FILE) as f:
            data = json.load(f)
    except Exception:
        pass
    current_bl = int(data.get("print_calls", count))
    if count < current_bl:
        data["print_calls"] = count
        data["print_count"] = count
        with open(_BASELINE_FILE, "w") as f:
            json.dump(data, f, indent=2)

# Directories and files to EXCLUDE from the print ratchet (offline scripts where print is expected)
EXCLUDE_DIRS = {
    ".venv", "venv", "build", "dist", ".git", ".pytest_cache",
    "node_modules", "mlartifacts", "tools", "scripts", "tests", "archive"
}

EXCLUDE_FILES = {
    "dashboard.py",
    "setup.py",
    "background_ml_tester.py",
    "soak_replay.py",
    "train.py",
    "auto_retrain_optuna.py",
    "retrain_worker.py"
}


class PrintCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.print_count = 0

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.print_count += 1
        self.generic_visit(node)


def count_runtime_prints(root_dir: str = ".") -> int:
    total_prints = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py") and f not in EXCLUDE_FILES:
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        tree = ast.parse(file.read(), filename=path)
                        visitor = PrintCallVisitor()
                        visitor.visit(tree)
                        total_prints += visitor.print_count
                except Exception:
                    pass
    return total_prints


if __name__ == "__main__":
    count = count_runtime_prints(".")
    baseline = _load_baseline()
    print(f"[Print Ratchet] Detected {count} print() calls in core runtime modules (Baseline: {baseline})")

    if count > baseline:
        print(f"\u274c REGRESSION DETECTED: Core runtime print() count rose {baseline} \u2192 {count}.")
        print("   Use log_event() from logger.py instead of raw print() in runtime paths.")
        sys.exit(1)
    else:
        if count < baseline:
            _save_baseline(count)  # ratchet down: baseline becomes new lower count
            print(f"✅ OK: {count} print() calls < baseline {baseline}. Ratchet lowered to {count}.")
        else:
            print(f"✅ OK: {count} print() calls == baseline {baseline}.")
        sys.exit(0)
