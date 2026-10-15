"""Fail-closed budget planning and a restartable *local* cleanup controller.

No provisioning API is implemented. A local controller is not a durable spending cap.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator, Protocol

PROJECT = "nemotron-mini-finance"
TOTAL_CAP = Decimal("100")
PILOT_CAP = Decimal("10")
RATE_CEILING = Decimal("1.00")
MAX_PILOT_SECONDS = 7200


class GateError(ValueError):
    """A required spending or ownership condition was not met."""


class PodClient(Protocol):
    def get(self, pod_id: str) -> dict[str, object] | None: ...
    def delete(self, pod_id: str) -> None: ...


def money(value: object) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise GateError("Invalid money value") from exc
    if not result.is_finite() or result < 0:
        raise GateError("Money must be finite and nonnegative")
    return result


def seconds(value: object) -> float:
    if isinstance(value, bool):
        raise GateError("Invalid time")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise GateError("Invalid time") from exc
    if not math.isfinite(result) or result < 0:
        raise GateError("Time must be finite and nonnegative")
    return result


def estimate(hourly_rate: object, duration_seconds: object, storage_usd: object = "0") -> Decimal:
    return money(hourly_rate) * Decimal(str(seconds(duration_seconds))) / 3600 + money(storage_usd)


def launch_readiness(*, balance_usd: object | None, cli_authenticated: bool,
                     local_tests_passed: bool, launch_authorized: bool = False) -> dict[str, object]:
    """No user-supplied boolean can bypass the missing durable teardown integration."""
    blockers = ["Durable provider-side teardown is not integrated or verified."]
    if balance_usd is None:
        blockers.append("Available account balance is unverified.")
    elif money(balance_usd) < PILOT_CAP:
        blockers.append("Available balance is below the $10 pilot allocation.")
    if not cli_authenticated:
        blockers.append("CLI authentication and SSH transfer path need verification.")
    if not local_tests_passed:
        blockers.append("Local validation must be confirmed for this launch session.")
    if not launch_authorized:
        blockers.append("Paid launch authorization has not been recorded.")
    return {"project": PROJECT, "paid_launch_ready": False, "blockers": blockers,
            "total_cap_usd": str(TOTAL_CAP), "pilot_cap_usd": str(PILOT_CAP),
            "rate_ceiling_usd_per_hour": str(RATE_CEILING),
            "max_pilot_seconds": MAX_PILOT_SECONDS}


class Ledger:
    """POSIX locked, atomic JSON ledger. Keep on local disk, not NFS.

    Reservations remain charged against the budget until final billing is entered.
    Local estimates are never represented as provider billing.
    """
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def transaction(self) -> Iterator[dict[str, object]]:
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if data.get("schema") != 1 or data.get("project") != PROJECT:
                    raise GateError("Unrecognized ledger; refusing mutation")
            else:
                data = {"schema": 1, "project": PROJECT, "runs": {}}
            yield data
            fd, name = tempfile.mkstemp(prefix=".ledger-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(data, stream, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, self.path)
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(name):
                    os.unlink(name)

    def reserve(self, run_id: str, *, stage: str, allocation_usd: object,
                hourly_rate: object, duration_seconds: object,
                storage_usd: object, now: object) -> dict[str, object]:
        if not run_id or not run_id.replace("-", "").isalnum():
            raise GateError("Run ID must be nonempty alphanumeric/hyphen")
        if stage not in {"pilot", "training", "evaluation"}:
            raise GateError("Unknown stage")
        allocation, rate = money(allocation_usd), money(hourly_rate)
        duration, start = seconds(duration_seconds), seconds(now)
        if allocation <= 0 or rate <= 0 or rate > RATE_CEILING or duration <= 0:
            raise GateError("Invalid allocation, duration, or hourly rate")
        if stage == "pilot" and (allocation > PILOT_CAP or duration > MAX_PILOT_SECONDS):
            raise GateError("Pilot exceeds $10 or two hours")
        if estimate(rate, duration, storage_usd) > allocation:
            raise GateError("Estimated compute plus storage exceeds allocation")
        record: dict[str, object] = {
            "stage": stage, "allocation_usd": str(allocation), "hourly_rate": str(rate),
            "duration_seconds": duration, "storage_usd": str(money(storage_usd)),
            "created_at": start, "deadline": start + duration, "status": "reserved",
            "expected_name": f"{PROJECT}-{run_id}", "pod_id": None,
            "final_billed_usd": None,
        }
        with self.transaction() as data:
            runs = data["runs"]
            if not isinstance(runs, dict):
                raise GateError("Malformed runs ledger")
            if run_id in runs:
                raise GateError("Run already reserved; resume it instead")
            commitments = sum((money(r["final_billed_usd"] if r["final_billed_usd"] is not None
                                     else r["allocation_usd"]) for r in runs.values()), Decimal(0))
            if commitments + allocation > TOTAL_CAP:
                raise GateError("Cumulative project budget exceeded")
            if stage == "pilot":
                pilot = sum((money(r["allocation_usd"]) for r in runs.values()
                             if r["stage"] == "pilot"), Decimal(0))
                if pilot + allocation > PILOT_CAP:
                    raise GateError("Cumulative pilot allocation exceeds $10")
            runs[run_id] = record
        return record

    def record_creation(self, run_id: str, receipt: dict[str, object], *, now: object) -> None:
        """Only pass the direct successful create response, never an inventory match.

        This function stores provenance, not cryptographic proof of ownership. A future
        launcher must call it directly with its create response. No launcher exists yet.
        """
        with self.transaction() as data:
            runs = data["runs"]
            record = runs[run_id]  # type: ignore[index]
            pod_id = receipt.get("id")
            if not isinstance(pod_id, str) or not pod_id or receipt.get("name") != record["expected_name"]:
                raise GateError("Creation receipt does not match this run")
            if record["status"] == "active" and record["pod_id"] == pod_id:
                return
            if record["status"] != "reserved" or seconds(now) >= record["deadline"]:
                raise GateError("Run cannot accept a new creation receipt")
            if any(r["pod_id"] == pod_id for r in runs.values()):
                raise GateError("Pod is already attributed to another run")
            actual_rate = money(receipt.get("cost"))
            # Preserve receipt even when actual cost exceeds the approved quote so
            # the controller can immediately tear down the expensive resource.
            record.update(pod_id=pod_id, actual_hourly_rate=str(actual_rate), status="active")
            if actual_rate > money(record["hourly_rate"]) or actual_rate > RATE_CEILING:
                record["stop_reason"] = "actual_rate_exceeds_quote"

    def tick(self, run_id: str, client: PodClient, *, now: object,
             job_finished: bool = False) -> str:
        """One controller step. DELETE retries are safe after a crash or transport loss.

        API exceptions leave the ledger active, so a restart retries cleanup. The
        client must map a genuine 404 to None and propagate authentication/network errors.
        """
        # Persist requested completion before any network operation. If GET or
        # DELETE fails, a restarted controller still knows teardown was requested.
        if job_finished:
            with self.transaction() as data:
                record = data["runs"][run_id]  # type: ignore[index]
                if record["status"] == "active":
                    record["stop_reason"] = "job_finished"
        with self.transaction() as data:
            record = data["runs"][run_id]  # type: ignore[index]
            if record["status"] == "closed":
                return "closed"
            if record["status"] != "active":
                return "reserved"
            pod_id = record["pod_id"]
            pod = client.get(pod_id)
            if pod is None:
                record["status"] = "closed"
                record["closed_at"] = seconds(now)
                return "closed"
            if pod.get("id") != pod_id or pod.get("name") != record["expected_name"]:
                raise GateError("Live pod identity mismatch; refusing teardown")
            observed_rate = money(pod.get("cost"))
            exceeded_rate = observed_rate > money(record["hourly_rate"])
            expired = seconds(now) >= record["deadline"]
            if not (expired or job_finished or exceeded_rate or record.get("stop_reason")):
                return "active"
            # Commit every teardown intent before DELETE, including rate overruns.
            record["stop_reason"] = record.get("stop_reason") or ("deadline" if expired else "rate_exceeded")
        # Caller must preserve checkpoints continuously on a network volume.
        # Deadline cleanup proceeds even when final artifact copying failed.
        client.delete(pod_id)
        if client.get(pod_id) is not None:
            return "termination_pending"
        with self.transaction() as data:
            record = data["runs"][run_id]  # type: ignore[index]
            if record["pod_id"] != pod_id:
                raise GateError("Recorded pod changed during cleanup")
            record["status"] = "closed"
            record["closed_at"] = seconds(now)
        return "closed"

    def settle(self, run_id: str, *, billed_usd: object) -> None:
        with self.transaction() as data:
            record = data["runs"][run_id]  # type: ignore[index]
            if record["status"] != "closed":
                raise GateError("Cannot settle an open run")
            amount = str(money(billed_usd))
            if record["final_billed_usd"] not in (None, amount):
                raise GateError("Conflicting final billing; requires manual reconciliation")
            record["final_billed_usd"] = amount


def run_bounded(command: list[str], *, timeout_seconds: float, cwd: Path | None = None) -> int:
    """Bound a local process group. This does NOT stop or delete a cloud pod."""
    limit = seconds(timeout_seconds)
    if limit <= 0 or not command:
        raise GateError("Command and positive finite timeout required")
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return process.wait(timeout=limit)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # A child can ignore TERM even after its parent has exited. Always finish
        # the process group, not only the immediate subprocess.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
