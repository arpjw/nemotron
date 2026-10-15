import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RunpodScriptTests(unittest.TestCase):
    def test_launch_and_preflight_are_fail_closed(self):
        for script in ("runpod_preflight.py", "runpod_launch.py"):
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                                    capture_output=True, text=True, timeout=10,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse(report["paid_launch_ready"])
            self.assertTrue(any("teardown" in item for item in report["blockers"]))
    def test_runner_default_only_prints_finite_plan(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/gpu_pilot.py")],
                                capture_output=True, text=True, timeout=10,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["execute"])
        self.assertLessEqual(sum(row[2] for row in report["commands"]), 5400)
        names = [row[0] for row in report["commands"]]
        self.assertLess(names.index("smoke_reload"), names.index("base_dev"))
        self.assertGreater(names.index("score_comparison"), names.index("adapter_dev"))
        for name, command, limit in report["commands"]:
            self.assertGreater(limit, 0)
            self.assertNotIn("test.jsonl", " ".join(command))
            self.assertNotIn("robustness.jsonl", " ".join(command))
            if not name.startswith("score_"):
                self.assertIn("--max-seconds", command)
            self.assertIn("data/frozen/", " ".join(command))
    def test_no_unsupported_timer_in_executable_scripts(self):
        for path in (ROOT / "scripts").glob("*.py"):
            self.assertNotIn("--terminate-after", path.read_text())
            self.assertNotIn("--stop-after", path.read_text())

if __name__ == "__main__": unittest.main()
