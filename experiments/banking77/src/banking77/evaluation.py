"""Closed-taxonomy metrics; primary constrained classification and raw format diagnostics."""
from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path


def canonical_label(text: str, taxonomy: list[str]) -> str | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    return stripped if stripped in taxonomy else None


def evaluate(rows: list[dict], predictions: dict[str, dict], taxonomy: list[str]) -> dict:
    ids = [row["id"] for row in rows]
    if not ids or len(ids) != len(set(ids)) or set(predictions) != set(ids):
        raise ValueError("predictions must cover unique evaluation IDs exactly")
    if any(row["label"] not in taxonomy for row in rows):
        raise ValueError("reference outside taxonomy")
    result = {"n": len(rows), "primary": "constrained_macro_f1", "modes": {}, "outcomes": []}
    modes = {}
    for mode in ("constrained", "raw"):
        guessed = {row["id"]: canonical_label(predictions[row["id"]].get(mode), taxonomy) for row in rows}
        correct = sum(guessed[row["id"]] == row["label"] for row in rows)
        valid = sum(label is not None for label in guessed.values())
        per_class = {}
        for label in taxonomy:
            tp = sum(row["label"] == label and guessed[row["id"]] == label for row in rows)
            fp = sum(row["label"] != label and guessed[row["id"]] == label for row in rows)
            fn = sum(row["label"] == label and guessed[row["id"]] != label for row in rows)
            per_class[label] = {"support": tp + fn, "correct": tp, "precision": tp / (tp + fp) if tp + fp else 0.0,
                                "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
        result["modes"][mode] = {"correct": correct, "accuracy": correct / len(rows), "valid": valid,
                                  "malformed": len(rows) - valid, "valid_label_accuracy": correct / valid if valid else None,
                                  "macro_f1": sum(item["f1"] for item in per_class.values()) / len(taxonomy), "per_class": per_class}
        modes[mode] = guessed
    result["outcomes"] = [{"id": row["id"], "group_id": row["group_id"],
                           "reference": row["label"], "constrained_label": modes["constrained"][row["id"]],
                           "constrained_correct": modes["constrained"][row["id"]] == row["label"],
                           "raw_correct": modes["raw"][row["id"]] == row["label"],
                           "raw_malformed": modes["raw"][row["id"]] is None} for row in rows]
    return result


def paired_interval(base: list[dict], candidate: list[dict], samples: int = 2000) -> dict:
    if type(samples) is not int or samples < 100:
        raise ValueError("at least100 bootstrap samples required")
    a = {row["id"]: row for row in base}
    b = {row["id"]: row for row in candidate}
    if set(a) != set(b) or len(a) != len(base) or len(b) != len(candidate) or not a:
        raise ValueError("paired results need identical unique IDs")
    groups = {}
    for key, row in a.items():
        if row["group_id"] != b[key]["group_id"]:
            raise ValueError("paired group mismatch")
        groups.setdefault(row["group_id"], []).append(int(b[key]["constrained_correct"]) - int(row["constrained_correct"]))
    values = list(groups.values())
    rng = random.Random(871)
    draws = []
    for _ in range(samples):
        chosen = [values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(sum(group) for group in chosen) / sum(len(group) for group in chosen))
    draws.sort()
    return {"delta": sum(sum(group) for group in values) / len(a), "lower_95": draws[int(.025 * (samples - 1))],
            "upper_95": draws[int(.975 * (samples - 1))], "metric": "constrained_accuracy", "method": "paired group percentile bootstrap", "samples": samples}


def paired_f1_interval(base: list[dict], candidate: list[dict], taxonomy: list[str], samples: int = 2000) -> dict:
    """Bootstrap macro-F1 over paired complete duplicate groups and all77 labels."""
    if type(samples) is not int or samples < 100:
        raise ValueError("at least100 bootstrap samples required")
    a = {row["id"]: row for row in base}
    b = {row["id"]: row for row in candidate}
    if not a or set(a) != set(b) or len(a) != len(base) or len(b) != len(candidate):
        raise ValueError("paired F1 results need identical unique IDs")
    label_index = {label: index for index, label in enumerate(taxonomy)}
    groups = {}
    for key, row in a.items():
        other = b[key]
        if row["group_id"] != other["group_id"] or row["reference"] != other["reference"]:
            raise ValueError("paired F1 group or reference mismatch")
        groups.setdefault(row["group_id"], []).append((label_index[row["reference"]], label_index.get(row["constrained_label"]), label_index.get(other["constrained_label"])))
    values = list(groups.values())
    def delta(selected):
        support = [0] * len(taxonomy)
        predicted = [[0] * len(taxonomy) for _ in range(2)]
        correct = [[0] * len(taxonomy) for _ in range(2)]
        for group in selected:
            for reference, first, second in group:
                support[reference] += 1
                for mode, guessed in enumerate((first, second)):
                    if guessed is not None:
                        predicted[mode][guessed] += 1
                        correct[mode][guessed] += guessed == reference
        scores = [sum(2 * correct[mode][i] / (support[i] + predicted[mode][i]) if support[i] + predicted[mode][i] else 0 for i in range(len(taxonomy))) / len(taxonomy) for mode in range(2)]
        return scores[1] - scores[0]
    rng = random.Random(871)
    draws = sorted(delta([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples))
    return {"metric": "constrained_macro_f1", "delta": delta(values), "lower_95": draws[int(.025 * (samples - 1))],
            "upper_95": draws[int(.975 * (samples - 1))], "method": "paired group percentile bootstrap; fixed full taxonomy", "samples": samples}


def load_predictions(path: Path, data: Path, ids: list[str]) -> tuple[dict, dict]:
    from .model import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, MAX_NEW_TOKENS, sha256_file
    metadata = json.loads((path.parent / "inference_manifest.json").read_text())
    expected = {"status": "completed", "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "system_prompt": SYSTEM_PROMPT,
                "dataset_sha256": sha256_file(data), "taxonomy_sha256": sha256_file(data.parent / "taxonomy.json"),
                "predictions_sha256": sha256_file(path), "rows": len(ids), "dataset_rows": len(ids), "subset": False,
                "selected_ids": ids, "quantization": "NF4-double-BF16", "sealed": data.stem == "test",
                "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "modes": ["taxonomy_trie", "raw"]}}
    if data.stem == "test":
        expected["selection_sha256"] = sha256_file(data.parent / "selection.json")
        expected["test_guard_sha256"] = sha256_file(Path(__file__).with_name("test_guard.py"))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("inference provenance mismatch")
    if data.stem == "test":
        from .test_guard import validate_test_receipt
        validate_test_receipt(data.parent, metadata)
    predictions = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if not isinstance(item, dict) or set(item) != {"id", "constrained", "raw"} or any(not isinstance(value, str) for value in item.values()) or item["id"] in predictions:
            raise ValueError("invalid or duplicate prediction row")
        predictions[item["id"]] = item
    if set(predictions) != set(ids):
        raise ValueError("prediction IDs mismatch")
    return predictions, metadata


def passes_development_gate(macro_f1: float, interval: dict, gate: dict) -> bool:
    return (macro_f1 >= gate["macro_f1_at_least"] and interval["delta"] >= gate["gain_over_base_at_least"]
            and interval["lower_95"] > gate["paired_macro_f1_lower_95_strictly_above"])


def main() -> None:
    from .data import load_frozen_records
    from .model import load_taxonomy, sha256_file
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-sealed", action="store_true")
    args = parser.parse_args()
    if args.data.name not in ("dev.jsonl", "test.jsonl") or (args.data.name == "test.jsonl" and not args.allow_sealed):
        parser.error("only frozen dev or explicitly acknowledged test is evaluable")
    selection = None
    if args.data.stem == "test":
        if not args.baseline:
            parser.error("official-test report requires the untouched baseline comparison")
        from .sealed_test import load_test_records
        rows = load_test_records(args.data)
        selection = json.loads((args.data.parent / "selection.json").read_text())
    else:
        rows = load_frozen_records(args.data, allowed_split="dev")
    taxonomy = load_taxonomy(args.data.parent)
    ids = [row["id"] for row in rows]
    predictions, metadata = load_predictions(args.predictions, args.data, ids)
    report = {"candidate": evaluate(rows, predictions, taxonomy), "candidate_inference": metadata,
              "data_sha256": sha256_file(args.data), "evaluator_sha256": sha256_file(Path(__file__)),
              "claim_limit": "single-seed BANKING77 few-shot intent classification; no broad finance claim", "sealed": args.data.stem == "test"}
    if args.baseline:
        base, base_metadata = load_predictions(args.baseline, args.data, ids)
        if base_metadata.get("adapter_sha256") is not None:
            raise ValueError("baseline must be untouched model")
        if not re.fullmatch(r"[0-9a-f]{64}", metadata.get("adapter_sha256") or "") or metadata.get("adapter_epoch_complete") is not True:
            raise ValueError("candidate must be a completed full-epoch adapter")
        if not metadata.get("versions") or metadata["versions"] != base_metadata.get("versions"):
            raise ValueError("base and adapter runtime versions differ or are unrecorded")
        for filename in ("model.py", "inference.py"):
            digest = metadata.get("source_sha256", {}).get(filename)
            if not digest or digest != base_metadata.get("source_sha256", {}).get(filename):
                raise ValueError("base and adapter inference source code differs or is unrecorded")
        if selection is not None:
            from .test_guard import validate_test_binding
            validate_test_binding(selection, metadata, adapter=True)
            validate_test_binding(selection, base_metadata, adapter=False)
        report["baseline"] = evaluate(rows, base, taxonomy)
        report["baseline_inference"] = base_metadata
        report["paired_accuracy"] = paired_interval(report["baseline"]["outcomes"], report["candidate"]["outcomes"])
        report["paired_macro_f1"] = paired_f1_interval(report["baseline"]["outcomes"], report["candidate"]["outcomes"], taxonomy)
        if args.data.stem == "test":
            from .test_guard import clean_test_rows
            audit = json.loads((args.data.parent / "audit.json").read_text())
            clean = clean_test_rows(rows, audit)
            clean_ids = {row["id"] for row in clean}
            report["official_all_rows"] = True
            report["selection_sha256"] = sha256_file(args.data.parent / "selection.json")
            if clean:
                clean_candidate = evaluate(clean, {key: value for key, value in predictions.items() if key in clean_ids}, taxonomy)
                clean_base = evaluate(clean, {key: value for key, value in base.items() if key in clean_ids}, taxonomy)
                report["clean_sensitivity"] = {"n": len(clean), "candidate": clean_candidate, "baseline": clean_base,
                                               "paired_macro_f1": paired_f1_interval(clean_base["outcomes"], clean_candidate["outcomes"], taxonomy),
                                               "paired_accuracy": paired_interval(clean_base["outcomes"], clean_candidate["outcomes"])}
            else:
                report["clean_sensitivity"] = {"n": 0, "available": False, "reason": "all official rows quarantined by overlap/conflict audit"}
            report["quarantine_audit"] = audit
        if args.data.stem == "dev":
            interval = report["paired_macro_f1"]
            gate_path = Path(__file__).resolve().parents[2] / "docs/GATE_V2.json"
            gate = json.loads(gate_path.read_text())
            if gate["data_manifest_sha256"] != sha256_file(args.data.parent / "manifest.json") or gate["primary_metric"] != "constrained_macro_f1" or gate["bootstrap"]["samples"] != 2000 or gate["bootstrap"]["seed"] != 871:
                raise ValueError("prospective gate does not match frozen data or bootstrap")
            report["development_gate"] = {"definition": gate, "sha256": sha256_file(gate_path),
                                          "passed": passes_development_gate(report["candidate"]["modes"]["constrained"]["macro_f1"], interval, gate)}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["candidate"]["modes"]["constrained"]["accuracy"]))


if __name__ == "__main__":
    main()
