"""Strict, dependency-free evaluation for the frozen financial snippet task."""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from typing import Any, Mapping, Sequence

KEYS = frozenset({"status", "value", "unit", "evidence"})
DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]{2}\Z")
EVIDENCE = re.compile(r"L[1-9][0-9]*\Z")
METRICS = ("exact", "schema", "status", "value", "unit", "evidence")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def valid_response(value: object) -> bool:
    """Validate types before values, including bool-vs-number ambiguities."""
    if not isinstance(value, dict) or set(value) != KEYS:
        return False
    status, number, unit, evidence = (
        value["status"], value["value"], value["unit"], value["evidence"]
    )
    if not isinstance(status, str) or status not in ("ok", "unanswerable"):
        return False
    if not isinstance(evidence, list) or any(
        not isinstance(item, str) or EVIDENCE.fullmatch(item) is None
        for item in evidence
    ):
        return False
    if len(set(evidence)) != len(evidence):
        return False
    if status == "unanswerable":
        return number is None and unit is None and evidence == []
    return (
        isinstance(number, str)
        and DECIMAL.fullmatch(number) is not None
        and isinstance(unit, str)
        and unit in ("USD", "percent")
        and bool(evidence)
    )


def parse_response(text: str) -> dict[str, Any] | None:
    """Reject extra prose, duplicate keys, numeric JSON values, and NaN."""
    if not isinstance(text, str) or len(text) > 100_000:
        return None
    try:
        response = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, TypeError, RecursionError):
        return None
    return response if valid_response(response) else None


def score_response(text: str, expected: Mapping[str, Any]) -> dict[str, bool]:
    """Score against a trusted reference; invalid references fail loudly."""
    if not valid_response(dict(expected)):
        raise ValueError("Invalid reference response")
    parsed = parse_response(text)
    if parsed is None:
        return dict.fromkeys(METRICS, False)
    flags = {
        "schema": True,
        "status": parsed["status"] == expected["status"],
        "value": parsed["value"] == expected["value"],
        "unit": parsed["unit"] == expected["unit"],
        "evidence": set(parsed["evidence"]) == set(expected["evidence"]),
    }
    return {"exact": all(flags.values()), **flags}


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]], predictions: Mapping[str, str]
) -> dict[str, Any]:
    """Missing outputs fail; extra IDs and duplicate reference IDs are errors."""
    ids = [row["id"] for row in records]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Evaluation requires nonempty, uniquely identified records")
    if set(predictions) - set(ids):
        raise ValueError("Predictions contain IDs outside this evaluation set")
    totals: Counter[str] = Counter()
    slices: dict[str, Counter[str]] = {}
    outcomes = []
    true_abstain = predicted_abstain = correct_abstain = 0
    answerable_value = answerable_n = 0
    for row in records:
        text = predictions.get(row["id"], "")
        flags = score_response(text, row["response"])
        totals.update({key: int(value) for key, value in flags.items()})
        for dimension in ("task", "layout"):
            bucket = slices.setdefault(f"{dimension}:{row[dimension]}", Counter())
            bucket["n"] += 1
            bucket["correct"] += flags["exact"]
        actual_abstain = row["response"]["status"] == "unanswerable"
        parsed = parse_response(text)
        guessed_abstain = parsed is not None and parsed["status"] == "unanswerable"
        true_abstain += actual_abstain
        predicted_abstain += guessed_abstain
        correct_abstain += actual_abstain and guessed_abstain
        if not actual_abstain:
            answerable_n += 1
            answerable_value += flags["value"] and flags["status"] and flags["unit"]
        outcomes.append({"id": row["id"], "group_id": row["group_id"], **flags})
    n = len(records)
    return {
        "n": n,
        "counts": {metric: totals[metric] for metric in METRICS},
        "rates": {metric: totals[metric] / n for metric in METRICS},
        "slices": {
            name: {**dict(counts), "accuracy": counts["correct"] / counts["n"]}
            for name, counts in sorted(slices.items())
        },
        "abstention": {
            "true": true_abstain,
            "predicted": predicted_abstain,
            "correct": correct_abstain,
            "precision": correct_abstain / predicted_abstain if predicted_abstain else None,
            "recall": correct_abstain / true_abstain if true_abstain else None,
        },
        "answerable_value": {
            "n": answerable_n,
            "correct": answerable_value,
            "accuracy": answerable_value / answerable_n if answerable_n else None,
        },
        "outcomes": outcomes,
    }


def paired_accuracy_interval(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    samples: int = 2000,
    seed: int = 531,
) -> dict[str, float | int]:
    """Percentile 95% paired bootstrap, resampling complete latent groups."""
    if isinstance(samples, bool) or samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    a = {row["id"]: row for row in baseline}
    b = {row["id"]: row for row in candidate}
    if not a or set(a) != set(b) or len(a) != len(baseline) or len(b) != len(candidate):
        raise ValueError("Paired results must have identical unique IDs")
    groups: dict[str, list[int]] = {}
    for key in sorted(a):
        if a[key]["group_id"] != b[key]["group_id"]:
            raise ValueError("Paired group mismatch")
        if type(a[key]["exact"]) is not bool or type(b[key]["exact"]) is not bool:
            raise ValueError("Exact outcomes must be booleans")
        groups.setdefault(a[key]["group_id"], []).append(
            int(b[key]["exact"]) - int(a[key]["exact"])
        )
    values = list(groups.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(sum(v) for v in selected) / sum(len(v) for v in selected))
    draws.sort()
    return {
        "delta": sum(sum(v) for v in values) / len(a),
        "lower_95": draws[math.floor(0.025 * (samples - 1))],
        "upper_95": draws[math.ceil(0.975 * (samples - 1))],
        "groups": len(groups),
        "samples": samples,
        "seed": seed,
    }


def validate_comparison_provenance(
    prediction_path: "Path", data_path: "Path", ids: list[str]
) -> dict[str, Any]:
    """Bind official comparisons to complete, equivalent inference runs."""
    from pathlib import Path
    from finmini.model import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, sha256_file

    path = Path(prediction_path)
    metadata = json.loads((path.parent / "inference_manifest.json").read_text(encoding="utf-8"))
    required = {
        "status": "completed",
        "dataset_sha256": sha256_file(data_path),
        "predictions_sha256": sha256_file(path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "system_prompt": SYSTEM_PROMPT,
        "quantization": "NF4-double-BF16",
        "rows": len(ids),
        "dataset_rows": len(ids),
        "subset": False,
        "selected_ids": ids,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Inference provenance mismatch: {key}")
    generation = metadata.get("generation")
    if not isinstance(generation, dict) or set(generation) != {"do_sample", "num_beams", "max_new_tokens"}:
        raise ValueError("Unexpected generation configuration")
    if generation["do_sample"] is not False or type(generation["num_beams"]) is not int or generation["num_beams"] != 1:
        raise ValueError("Comparison requires greedy single-beam generation")
    if type(generation["max_new_tokens"]) is not int or not 1 <= generation["max_new_tokens"] < 1024:
        raise ValueError("Invalid generation token budget")
    return metadata


def main() -> None:
    """Score predictions, with an explicit acknowledgement for sealed tests."""
    import argparse
    from pathlib import Path
    from finmini.data import digest, verify_dataset

    parser = argparse.ArgumentParser(description="Strict scoring of financial snippet predictions")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--allow-sealed", action="store_true", help="Acknowledge final test use; no subsequent tuning against these scores")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Score output already exists; use a fresh report path")
    if args.data.stem in ("test", "robustness") and not args.allow_sealed:
        parser.error("Sealed final evaluation requires --allow-sealed")
    manifest = verify_dataset(args.data.parent)
    if args.data.name not in manifest["files"]:
        parser.error("Data must be a manifest-listed split file")
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]

    def load_predictions(path: Path) -> dict[str, str]:
        predictions: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(item, dict) or set(item) != {"id", "prediction"}:
                raise ValueError("Prediction rows require exactly id and prediction")
            if not isinstance(item["id"], str) or not isinstance(item["prediction"], str):
                raise ValueError("Prediction ID and prediction must be strings")
            if item["id"] in predictions:
                raise ValueError("Duplicate prediction ID")
            predictions[item["id"]] = item["prediction"]
        return predictions

    report = evaluate_predictions(rows, load_predictions(args.predictions))
    report["provenance"] = {
        "data_sha256": digest(args.data.read_bytes()),
        "predictions_sha256": digest(args.predictions.read_bytes()),
        "scorer_sha256": digest(Path(__file__).read_bytes()),
        "sealed_evaluation": args.data.stem in ("test", "robustness"),
        "scope": "original synthetic financial snippets only",
    }
    if args.baseline:
        ids = [row["id"] for row in rows]
        candidate_metadata = validate_comparison_provenance(args.predictions, args.data, ids)
        baseline_metadata = validate_comparison_provenance(args.baseline, args.data, ids)
        if baseline_metadata.get("adapter_sha256") is not None:
            raise ValueError("Baseline must be the untouched model")
        adapter_hash = candidate_metadata.get("adapter_sha256")
        if not isinstance(adapter_hash, str) or re.fullmatch(r"[0-9a-f]{64}", adapter_hash) is None:
            raise ValueError("Candidate requires a recorded adapter checksum")
        if baseline_metadata["generation"] != candidate_metadata["generation"]:
            raise ValueError("Baseline and adapter decoding settings differ")
        candidate_predictions = load_predictions(args.predictions)
        baseline_predictions = load_predictions(args.baseline)
        if set(candidate_predictions) != set(ids) or set(baseline_predictions) != set(ids):
            raise ValueError("Official paired comparison requires every prediction")
        baseline = evaluate_predictions(rows, baseline_predictions)
        report["comparison"] = paired_accuracy_interval(baseline["outcomes"], report["outcomes"])
        report["provenance"]["baseline_predictions_sha256"] = digest(args.baseline.read_bytes())
        report["provenance"]["candidate_inference_manifest_sha256"] = digest((args.predictions.parent / "inference_manifest.json").read_bytes())
        report["provenance"]["baseline_inference_manifest_sha256"] = digest((args.baseline.parent / "inference_manifest.json").read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"n": report["n"], "rates": report["rates"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
