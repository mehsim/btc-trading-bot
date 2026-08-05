"""
tools/check_print_ratchet.py
-----------------------------
Fix 3 Ratchet: Asserts that unstructured print() calls in core runtime modules do not increase.
Excludes offline CLI tools, training scripts, tests, and standalone scripts.
"""

import ast
import os
import sys

# Baseline print() count for core runtime files (ratchets down only)
PRINT_BASELINE = 1026

# Directories and files to EXCLUDE from the print ratchet (offline scripts where print is expected)
EXCLUDE_DIRS = {
    ".venv", "venv", "build", "dist", ".git", ".pytest_cache",
    "node_modules", "mlartifacts", "tools", "scripts", "tests"
}

EXCLUDE_FILES = {
    "dashboard.py",
    "setup.py",
    "background_ml_tester.py",
    "soak_replay.py"
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
    print(f"[Print Ratchet] Detected {count} print() calls in core runtime modules (Baseline: {PRINT_BASELINE})")
    
    if count > PRINT_BASELINE:
        print(f"❌ REGRESSION DETECTED: Core runtime print() count rose {PRINT_BASELINE} → {count}.")
        print("   Use log_event() from logger.py instead of raw print() in runtime paths.")
        sys.exit(1)
    else:
        print(f"✅ OK: {count} print() calls <= baseline {PRINT_BASELINE}.")
        sys.exit(0)
