#!/usr/bin/env python3
"""Read-only local preparation report. Never provisions or prints credentials."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finmini.runpod_guard import launch_readiness

if __name__ == "__main__":
    print(json.dumps(launch_readiness(balance_usd=None, cli_authenticated=False,
                                     local_tests_passed=False), indent=2))
    sys.exit(2)
