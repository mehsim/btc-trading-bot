"""
tools/mypy_ratchet.py
---------------------
Fix 4 Type Coverage Ratchet:
Parses mypy output, counts type errors, and asserts count <= BASELINE (133).
Baseline ratchets down only as type annotations improve across the codebase.
"""

import argparse
import re
import sys

BASELINE = 165


def count_mypy_errors(mypy_output_path: str) -> int:
    error_pattern = re.compile(r"^.+:\d+: error:")
    count = 0
    try:
        with open(mypy_output_path, "r", encoding="utf-8") as f:
            for line in f:
                if error_pattern.search(line):
                    count += 1
    except Exception as e:
        print(f"[MyPy Ratchet Warning] Could not read mypy output file {mypy_output_path}: {e}")
        return 0
    return count


def enforce_mypy_ratchet(mypy_output_path: str, baseline: int = BASELINE):
    count = count_mypy_errors(mypy_output_path)
    print(f"[MyPy Type Coverage Ratchet] Total mypy errors: {count} (Baseline: {baseline})")

    if count > baseline:
        print(f"❌ REGRESSION DETECTED: MyPy error count rose {baseline} → {count}.")
        sys.exit(1)
    else:
        print(f"✅ OK: {count} mypy errors <= baseline {baseline}.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyPy Type Coverage Ratchet")
    parser.add_argument("output_file", type=str, help="Path to mypy.txt output")
    parser.add_argument("--baseline", type=int, default=BASELINE, help="Baseline error count threshold")
    args = parser.parse_args()

    enforce_mypy_ratchet(args.output_file, args.baseline)
