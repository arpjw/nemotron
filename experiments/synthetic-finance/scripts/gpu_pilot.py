#!/usr/bin/env python3
"""Finite remote GPU smoke/train/dev workflow. Does not provision or stop a pod."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finmini.runpod_guard import GateError, run_bounded


def plan(root: Path = ROOT) -> list[tuple[str, list[str], int]]:
    py = sys.executable
    return [
        ("smoke", [py, "-m", "finmini.train", "--data", str(root / "data/frozen/train.jsonl"),
                   "--output", str(root / "runs/smoke-adapter"), "--max-steps", "2",
                   "--gradient-accumulation", "1", "--max-seconds", "300"], 360),
        ("smoke_reload", [py, "-m", "finmini.inference", "--data", str(root / "data/frozen/dev.jsonl"),
                          "--output", str(root / "runs/smoke-reload"), "--adapter",
                          str(root / "runs/smoke-adapter"), "--limit", "2", "--max-seconds", "300"], 360),
        ("base_dev", [py, "-m", "finmini.inference", "--data", str(root / "data/frozen/dev.jsonl"),
                      "--output", str(root / "runs/base-dev"), "--max-seconds", "900"], 960),
        ("pilot", [py, "-m", "finmini.train", "--data", str(root / "data/frozen/train.jsonl"),
                   "--output", str(root / "runs/pilot-adapter"), "--max-steps", "100",
                   "--gradient-accumulation", "8", "--max-seconds", "2400"], 2460),
        ("adapter_dev", [py, "-m", "finmini.inference", "--data", str(root / "data/frozen/dev.jsonl"),
                         "--output", str(root / "runs/adapter-dev"), "--adapter",
                         str(root / "runs/pilot-adapter"), "--max-seconds", "900"], 960),
        ("score_base", [py, "-m", "finmini.scoring", "--data", str(root / "data/frozen/dev.jsonl"),
                        "--predictions", str(root / "runs/base-dev/predictions.jsonl"),
                        "--output", str(root / "runs/base-dev-score.json")], 30),
        ("score_comparison", [py, "-m", "finmini.scoring", "--data", str(root / "data/frozen/dev.jsonl"),
                              "--predictions", str(root / "runs/adapter-dev/predictions.jsonl"),
                              "--baseline", str(root / "runs/base-dev/predictions.jsonl"),
                              "--output", str(root / "runs/dev-comparison.json")], 60),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-supervised", action="store_true")
    args = parser.parse_args()
    commands = plan()
    if not args.execute_supervised:
        print(json.dumps({"execute": False, "commands": commands,
                          "warning": "Process timeout does not stop GPU billing."}, indent=2))
        return 0
    for filename in ("train.jsonl", "dev.jsonl", "manifest.json"):
        if not (ROOT / "data" / "frozen" / filename).is_file():
            raise GateError(f"Missing data/frozen/{filename}")
    from finmini.data import verify_dataset
    verify_dataset(ROOT / "data/frozen")
    runs = ROOT / "runs"
    # Reject partial earlier runs. Recovery is a new reviewed run, never overwrite.
    if runs.exists():
        raise GateError("runs/ already exists; preserve and review the previous run")
    runs.mkdir()
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    start = time.monotonic()
    status = {"complete": False, "steps": [], "process_timeout_seconds": 5400}
    try:
        for name, command, limit in commands:
            remaining = 5400 - (time.monotonic() - start)
            if remaining <= 0:
                raise GateError("Pilot process deadline reached")
            begun = time.monotonic()
            step = {"name": name, "returncode": None, "seconds": None}
            status["steps"].append(step)
            try:
                result = run_bounded(command, timeout_seconds=min(limit, remaining), cwd=ROOT)
                step["returncode"] = result
            except BaseException as exc:
                step["error_type"] = type(exc).__name__
                raise
            finally:
                step["seconds"] = time.monotonic() - begun
            if result:
                return result
        status["complete"] = True
        return 0
    finally:
        status["elapsed_seconds"] = time.monotonic() - start
        (runs / "pilot_status.json").write_text(json.dumps(status, indent=2))
        print("Pilot process finished. Export artifacts and verify pod termination; GPU billing may continue.")


if __name__ == "__main__":
    raise SystemExit(main())
