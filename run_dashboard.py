"""
run_dashboard.py — One-click dashboard launcher for IDE (VS Code / PyCharm).

Hardcoded paths point to files that are committed in the repository so the
dashboard works out-of-the-box without any CLI arguments.

To use a different predictions file, change INPUT_FILE below.
"""

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths (relative to this file — works regardless of working directory)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "data", "predicted_sample.csv")
RAW_COUNTS = os.path.join(BASE_DIR, "models", "raw_counts_per_date.csv")
DASHBOARD  = os.path.join(BASE_DIR, "dashboard.py")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for path, label in [(INPUT_FILE, "input_file"), (RAW_COUNTS, "raw_counts")]:
        if not os.path.exists(path):
            print(f"[run_dashboard] ERROR: {label} not found: {path}")
            sys.exit(1)

    print(f"[run_dashboard] input_file : {INPUT_FILE}")
    print(f"[run_dashboard] raw_counts : {RAW_COUNTS}")
    print()

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", DASHBOARD,
        "--",
        "--input_file", INPUT_FILE,
        "--raw_counts", RAW_COUNTS,
    ])
