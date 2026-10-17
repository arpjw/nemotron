"""Freeze a development-selected graph adapter before one test comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import MODEL_ID, MODEL_REVISION, file_hash, write_json


def make_selection(report: dict, *, report_hash: str) -> dict:
    if report.get("split") != "dev" or report.get("candidate_inference") is None or report.get("baseline_inference") is None:
        raise ValueError("complete base and adapter development report required")
    candidate = report["candidate_inference"]
    base = report["baseline_inference"]
    paired = report.get("paired", {})
    if paired.get("candidate_minus_baseline", 0) <= 0:
        raise ValueError("adapter did not improve development success rate")
    if candidate.get("model_id") != MODEL_ID or candidate.get("model_revision") != MODEL_REVISION:
        raise ValueError("wrong model")
    if base.get("adapter_sha256") is not None or not candidate.get("adapter_sha256"):
        raise ValueError("base/adapter comparison missing")
    return {"version": 1, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "dev_report_sha256": report_hash, "dev_dataset_sha256": report["dataset_sha256"],
            "adapter_sha256": candidate["adapter_sha256"],
            "dev_success_base": report["baseline"]["success_rate"],
            "dev_success_adapter": report["candidate"]["success_rate"],
            "selection_rule": "positive development success-rate difference; one prospective pilot"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("selection output already exists")
    report = json.loads(args.dev_report.read_text())
    selection = make_selection(report, report_hash=file_hash(args.dev_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, selection)
    print(json.dumps({"selected_adapter_sha256": selection["adapter_sha256"],
                      "dev_report_sha256": selection["dev_report_sha256"]}))


if __name__ == "__main__":
    main()
