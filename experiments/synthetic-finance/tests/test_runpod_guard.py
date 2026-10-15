import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from finmini.runpod_guard import GateError, Ledger, estimate, launch_readiness, money, run_bounded


class FakePod:
    def __init__(self, name):
        self.pod = {"id": "pod1", "name": name, "cost": "0.89"}
        self.deletes = 0
        self.fail_delete = False
    def get(self, pod_id):
        assert pod_id == "pod1"
        return self.pod
    def delete(self, pod_id):
        assert pod_id == "pod1"
        self.deletes += 1
        if self.fail_delete:
            raise ConnectionError("simulated transport loss")
        self.pod = None


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger.json"
        self.ledger = Ledger(self.path)
    def tearDown(self):
        self.temp.cleanup()
    def reserve(self, run_id="run1", **kwargs):
        defaults = dict(stage="pilot", allocation_usd="10", hourly_rate="0.89",
                        duration_seconds=7200, storage_usd="1", now=100)
        defaults.update(kwargs)
        return self.ledger.reserve(run_id, **defaults)
    def active(self):
        record = self.reserve()
        client = FakePod(record["expected_name"])
        self.ledger.record_creation("run1", client.pod, now=101)
        return client
    def test_budget_math(self):
        self.assertEqual(estimate("0.89", 7200, "1"), money("2.78"))
    def test_invalid_money(self):
        for value in ["NaN", "Infinity", "-1", None, "bad"]:
            with self.assertRaises(GateError):
                money(value)
    def test_pilot_caps_rate_and_time(self):
        for override in [dict(allocation_usd="10.01"), dict(hourly_rate="1.01"),
                         dict(duration_seconds=7201), dict(duration_seconds=float("nan")),
                         dict(allocation_usd="1")]:
            with self.assertRaises(GateError):
                self.reserve(**override)
    def test_duplicate_reservation_and_cumulative_pilot(self):
        self.reserve()
        for name in ["run1", "run2"]:
            with self.assertRaises(GateError): self.reserve(name)
    def test_cumulative_budget(self):
        self.reserve("one", stage="training", allocation_usd="95")
        with self.assertRaises(GateError): self.reserve("two")
    def test_controller_resumes_and_is_idempotent(self):
        client = self.active()
        resumed = Ledger(self.path)
        self.assertEqual(resumed.tick("run1", client, now=200), "active")
        self.assertEqual(resumed.tick("run1", client, now=7300), "closed")
        self.assertEqual(resumed.tick("run1", client, now=7400), "closed")
        self.assertEqual(client.deletes, 1)
    def test_failed_delete_retries_on_restart(self):
        client = self.active()
        client.fail_delete = True
        with self.assertRaises(ConnectionError): self.ledger.tick("run1", client, now=7400)
        client.fail_delete = False
        self.assertEqual(Ledger(self.path).tick("run1", client, now=7401), "closed")
    def test_completion_intent_survives_failed_delete_before_deadline(self):
        client = self.active()
        client.fail_delete = True
        with self.assertRaises(ConnectionError):
            self.ledger.tick("run1", client, now=200, job_finished=True)
        client.fail_delete = False
        self.assertEqual(Ledger(self.path).tick("run1", client, now=201), "closed")
    def test_unrelated_pod_refused(self):
        client = self.active()
        client.pod["name"] = "another-project"
        with self.assertRaises(GateError): self.ledger.tick("run1", client, now=7400)
        self.assertEqual(client.deletes, 0)
    def test_rate_change_triggers_teardown(self):
        client = self.active()
        client.pod["cost"] = "1.50"
        self.assertEqual(self.ledger.tick("run1", client, now=200), "closed")
    def test_rate_overrun_intent_survives_failed_delete(self):
        client = self.active()
        client.pod["cost"] = "2"
        client.fail_delete = True
        with self.assertRaises(ConnectionError):
            self.ledger.tick("run1", client, now=200)
        client.fail_delete = False
        client.pod["cost"] = "0.89"
        self.assertEqual(Ledger(self.path).tick("run1", client, now=201), "closed")
    def test_expensive_receipt_remains_attributed(self):
        record = self.reserve()
        client = FakePod(record["expected_name"])
        client.pod["cost"] = "2"
        self.ledger.record_creation("run1", client.pod, now=101)
        self.assertEqual(self.ledger.tick("run1", client, now=102), "closed")
    def test_foreign_creation_receipt_refused(self):
        self.reserve()
        with self.assertRaises(GateError):
            self.ledger.record_creation("run1", {"id":"foreign", "name":"foreign", "cost":".2"}, now=101)
    def test_billing_does_not_release_reservation_before_close(self):
        self.active()
        with self.assertRaises(GateError): self.ledger.settle("run1", billed_usd="1")
    def test_settlement_is_idempotent_not_overwritable(self):
        client = self.active()
        self.ledger.tick("run1", client, now=200, job_finished=True)
        self.ledger.settle("run1", billed_usd="1")
        self.ledger.settle("run1", billed_usd="1")
        with self.assertRaises(GateError): self.ledger.settle("run1", billed_usd="0")
    def test_corrupt_ledger_fail_closed(self):
        self.path.write_text('{"project":"foreign", "schema":1}')
        with self.assertRaises(GateError): self.reserve()
    def test_launch_always_requires_durable_integration(self):
        report = launch_readiness(balance_usd="100", cli_authenticated=True,
                                  local_tests_passed=True, launch_authorized=True)
        self.assertFalse(report["paid_launch_ready"])
        self.assertEqual(len(report["blockers"]), 1)
    def test_process_timeout(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], timeout_seconds=.05)
    def test_process_exit_code(self):
        self.assertEqual(run_bounded([sys.executable, "-c", "raise SystemExit(3)"], timeout_seconds=2), 3)

if __name__ == "__main__": unittest.main()
