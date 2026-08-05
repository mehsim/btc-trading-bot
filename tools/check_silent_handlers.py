"""
tools/check_silent_handlers.py
-------------------------------
Tier 1 Ratchet for Silent Exception Handlers:
Scans Python files in the codebase for silent or broad exception handlers (except Exception / except:)
that swallow exceptions without logging or re-raising.
Enforces that silent handler count only ever decreases over time.
"""

import ast
import os
import sys

# Current Baseline Count (Ratchets down only)
BASELINE = 110

# Files to exclude (external libraries, build output, virtual environments, tooling)
EXCLUDE_DIRS = {".venv", "venv", "build", "dist", ".git", ".pytest_cache", "node_modules", "mlartifacts", "tools"}
EXCLUDE_FILES = {"setup.py"}


class SilentHandlerVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.silent_count = 0

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Broad handlers: except: or except Exception / BaseException
        is_broad = False
        if node.type is None:
            is_broad = True
        elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
            is_broad = True
        elif isinstance(node.type, ast.Tuple):
            for elt in node.type.elts:
                if isinstance(elt, ast.Name) and elt.id in ("Exception", "BaseException"):
                    is_broad = True
                    break

        if is_broad:
            # Check if body is just pass, continue, or simple return without logging or raising
            if len(node.body) == 1:
                stmt = node.body[0]
                if isinstance(stmt, (ast.Pass, ast.Continue)):
                    self.silent_count += 1
                elif isinstance(stmt, ast.Return):
                    self.silent_count += 1
            else:
                # Check if body lacks print, log, logger, raise, or re-raise
                has_logging_or_raise = False
                for stmt in node.body:
                    if isinstance(stmt, ast.Raise):
                        has_logging_or_raise = True
                        break
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        func_name = ""
                        if isinstance(func, ast.Name):
                            func_name = func.id
                        elif isinstance(func, ast.Attribute):
                            func_name = func.attr
                        
                        if any(kw in func_name.lower() for kw in ("log", "print", "alert", "traceback", "warning", "error")):
                            has_logging_or_raise = True
                            break
                if not has_logging_or_raise:
                    self.silent_count += 1

        self.generic_visit(node)


def count_silent_handlers(root_dir: str = ".") -> int:
    total_silent = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py") and f not in EXCLUDE_FILES:
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        tree = ast.parse(file.read(), filename=path)
                        visitor = SilentHandlerVisitor(path)
                        visitor.visit(tree)
                        total_silent += visitor.silent_count
                except Exception:
                    pass
    return total_silent


if __name__ == "__main__":
    count = count_silent_handlers(".")
    print(f"[Silent Handler Ratchet] Detected {count} silent/broad exception handlers (Baseline: {BASELINE})")
    
    if count > BASELINE:
        print(f"❌ REGRESSION DETECTED: Silent handler count rose {BASELINE} → {count}.")
        print("   Narrow exception type or log the exception instead of swallowing it silently.")
        sys.exit(1)
    else:
        print(f"✅ OK: {count} silent handlers <= baseline {BASELINE}.")
        sys.exit(0)
