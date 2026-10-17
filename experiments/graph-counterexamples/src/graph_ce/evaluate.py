"""Exact complete-denominator graph-witness evaluation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import load_split
from .graph import Graph
from .model import file_hash, write_json
from .task import Task


def parse_graph(raw: object, n: int) -> Graph | None:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        return Graph.from_json(value, expected_n=n)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def score(rows: list[dict], predictions: list[dict]) -> dict:
    ids = [row["task"]["id"] for row in rows]
    mapping = {row["id"]: row for row in predictions if isinstance(row, dict) and isinstance(row.get("id"), str)}
    if not rows or len(ids) != len(set(ids)) or len(mapping) != len(predictions) or set(mapping) != set(ids):
        raise ValueError("predictions must cover unique task IDs exactly")
    outcomes = []
    for row in rows:
        task = Task.from_json(row["task"])
        prediction = mapping[task.id]
        graph = parse_graph(prediction.get("raw", prediction.get("graph")), task.n)
        margin = task.margin(graph) if graph is not None else None
        outcomes.append({"id": task.id, "valid": graph is not None,
                         "success": margin is not None and margin > 0,
                         "margin": margin, "edges": len(graph.edges) if graph is not None else None})
    successes = [outcome for outcome in outcomes if outcome["success"]]
    return {"n": len(outcomes), "successes": len(successes),
            "success_rate": len(successes) / len(outcomes),
            "valid_json_graphs": sum(outcome["valid"] for outcome in outcomes),
            "mean_margin_on_success": sum(item["margin"] for item in successes) / len(successes) if successes else None,
            "mean_edges_on_success": sum(item["edges"] for item in successes) / len(successes) if successes else None,
            "outcomes": outcomes}


def paired_interval(first: dict, second: dict, *, draws: int = 2000, seed: int = 871) -> dict:
    a = {row["id"]: row for row in first["outcomes"]}
    b = {row["id"]: row for row in second["outcomes"]}
    if not a or set(a) != set(b):
        raise ValueError("paired results need identical IDs")
    differences = [int(b[key]["success"]) - int(a[key]["success"]) for key in sorted(a)]
    rng = random.Random(seed)
    samples = sorted(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
                     for _ in range(draws))
    return {"metric": "success_rate", "candidate_minus_baseline": sum(differences) / len(differences),
            "lower_95": samples[int(.025 * (draws - 1))],
            "upper_95": samples[int(.975 * (draws - 1))], "draws": draws,
            "method": "paired task percentile bootstrap"}


def load_predictions(path: Path, rows: list[dict]) -> list[dict]:
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    manifest_path = path.parent / "inference_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        split = rows[0]["task"]["split"]
        if (manifest.get("status") != "completed" or manifest.get("split") != split or
                manifest.get("rows") != len(rows) or not manifest.get("full_split") or
                manifest.get("predictions_sha256") != file_hash(path)):
            raise ValueError("incomplete or incompatible model inference")
    return predictions


def model_manifest(path: Path, rows: list[dict]) -> dict | None:
    manifest_path = path.parent / "inference_manifest.json"
    if not manifest_path.exists():
        return None
    value = json.loads(manifest_path.read_text())
    if (value.get("status") != "completed" or value.get("split") != rows[0]["task"]["split"] or
            value.get("rows") != len(rows) or not value.get("full_split") or
            value.get("predictions_sha256") != file_hash(path)):
        raise ValueError("invalid model inference manifest")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--require-model", action="store_true", help="require compatible full-split base and adapter manifests")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("test split requires explicit --allow-test after development selection")
    rows = load_split(args.data, args.split)
    candidate_manifest = model_manifest(args.predictions, rows)
    if (args.require_model or args.split == "test") and candidate_manifest is None:
        parser.error("model inference manifest required")
    candidate = score(rows, load_predictions(args.predictions, rows))
    report = {"split": args.split, "dataset_sha256": file_hash(args.data / f"{args.split}.jsonl"),
              "predictions_sha256": file_hash(args.predictions), "candidate": candidate,
              "candidate_inference": candidate_manifest,
              "claim_limit": "synthetic bounded-size graph tasks; one task-generation seed; no theorem proof"}
    if args.baseline:
        baseline_manifest = model_manifest(args.baseline, rows)
        if args.require_model or args.split == "test":
            if baseline_manifest is None or candidate_manifest is None:
                parser.error("both model inference manifests required")
            for key in ("model_id", "model_revision", "dataset_sha256", "generation"):
                if candidate_manifest.get(key) != baseline_manifest.get(key):
                    raise ValueError(f"base and adapter {key} differ")
            for name in ("model.py", "infer.py"):
                if (not candidate_manifest.get("source_sha256", {}).get(name) or
                        candidate_manifest["source_sha256"][name] != baseline_manifest.get("source_sha256", {}).get(name)):
                    raise ValueError("base and adapter decoder code differs")
            if baseline_manifest.get("adapter_sha256") is not None or not candidate_manifest.get("adapter_sha256"):
                raise ValueError("comparison requires untouched base and completed adapter")
            if args.split == "test" and (not candidate_manifest.get("selection_sha256") or
                                            candidate_manifest["selection_sha256"] != baseline_manifest.get("selection_sha256")):
                raise ValueError("test inference selection receipts differ")
        baseline = score(rows, load_predictions(args.baseline, rows))
        report.update(baseline=baseline, baseline_inference=baseline_manifest,
                      baseline_predictions_sha256=file_hash(args.baseline),
                      paired=paired_interval(baseline, candidate))
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"split": args.split, "n": candidate["n"], "successes": candidate["successes"],
                      "success_rate": candidate["success_rate"], "paired": report.get("paired")}))


if __name__ == "__main__":
    main()
