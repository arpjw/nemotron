#!/usr/bin/env python3
"""Fail-closed launch entrypoint: emit the required gate, never provision."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finmini.runpod_guard import launch_readiness

if __name__ == "__main__":
    report = launch_readiness(balance_usd=None, cli_authenticated=False,
                              local_tests_passed=False)
    report["procedure"] = "RUNPOD.md"
    report["remote_runner"] = "scripts/gpu_pilot.py"
    print(json.dumps(report, indent=2))
    raise SystemExit(2)
