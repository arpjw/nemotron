"""Exact one-flip scoring, paired summaries, and development selection."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from book import verify
from edit_model import MODEL_ID, MODEL_REVISION, file_hash, write_json
from edit_study import BLUE_BOOK, N, PAIRS, RED_BOOK, edge_list, flip, load_split, score, violating_spines


def parse_action(raw: str) -> tuple[int, int] | None:
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"u", "v"} or \
            any(type(value[key]) is not int for key in ("u", "v")):
        return None
    u, v = value["u"], value["v"]
    return (u, v) if 0 <= u < v < N else None


def score_predictions(rows: list[dict], predictions: list[dict]) -> dict:
    if len(rows) != len(predictions) or any(set(prediction) != {"id", "raw"} or
                                            prediction["id"] != row["id"]
                                            for row, prediction in zip(rows, predictions)):
        raise ValueError("prediction IDs, count, or schema mismatch")
    records = []
    for row, prediction in zip(rows, predictions):
        start = [int(word, 16) for word in row["rows_hex"]]
        action = parse_action(prediction["raw"])
        next_loss = row["loss"]
        if action is not None:
            flip(start, *action)
            next_loss = score(start)[0]
            exact = verify(N, edge_list(start), RED_BOOK, BLUE_BOOK)
            if (next_loss == 0) != exact["valid"]:
                raise AssertionError("score and independent checker disagree")
        records.append({"id": row["id"], "action": list(action) if action else None,
                        "initial_loss": row["loss"], "next_loss": next_loss,
                        "improved": next_loss < row["loss"], "solved": next_loss == 0,
                        "teacher_next_loss": row["teacher_next_loss"]})
    count = len(records)
    return {"rows": count, "valid_actions": sum(item["action"] is not None for item in records),
            "improvements": sum(item["improved"] for item in records),
            "solved": sum(item["solved"] for item in records),
            "mean_loss_reduction": sum(item["initial_loss"] - item["next_loss"]
                                       for item in records) / count,
            "mean_regret_to_teacher": sum(item["next_loss"] - item["teacher_next_loss"]
                                          for item in records) / count,
            "records": records}


def random_predictions(rows: list[dict]) -> list[dict]:
    return [{"id": row["id"], "raw": json.dumps(dict(zip(("u", "v"),
                                                     random.Random(871 + index).choice(PAIRS))))}
            for index, row in enumerate(rows)]


def visible_spine_predictions(rows: list[dict]) -> list[dict]:
    predictions = []
    for index, row in enumerate(rows):
        current = [int(word, 16) for word in row["rows_hex"]]
        _, u, v, _ = random.Random(971 + index).choice(violating_spines(current)[:12])
        predictions.append({"id": row["id"], "raw": json.dumps({"u": u, "v": v})})
    return predictions


def paired_interval(candidate: dict, baseline: dict, *, draws: int = 2000, seed: int = 871) -> list[float]:
    a, b = candidate["records"], baseline["records"]
    if len(a) != len(b) or any(x["id"] != y["id"] for x, y in zip(a, b)):
        raise ValueError("unpaired records")
    differences = [(x["initial_loss"] - x["next_loss"]) -
                   (y["initial_loss"] - y["next_loss"]) for x, y in zip(a, b)]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(differences) for _ in differences) / len(differences)
                     for _ in range(draws))
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws)]]


def load_predictions(directory: Path, data: Path, split: str) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "inference_manifest.json").read_text())
    if manifest.get("status") != "completed" or not manifest.get("full_split") or \
            manifest.get("split") != split or manifest.get("model_id") != MODEL_ID or \
            manifest.get("model_revision") != MODEL_REVISION or \
            manifest.get("dataset_sha256") != file_hash(data / f"{split}.jsonl") or \
            manifest.get("predictions_sha256") != file_hash(directory / "predictions.jsonl"):
        raise ValueError("inference provenance mismatch")
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    return manifest, predictions


def evaluate(data: Path, split: str, base: Path, adapter: Path) -> dict:
    rows = load_split(data, split)
    base_manifest, base_predictions = load_predictions(base, data, split)
    adapter_manifest, adapter_predictions = load_predictions(adapter, data, split)
    if base_manifest.get("adapter_sha256") is not None or not adapter_manifest.get("adapter_sha256"):
        raise ValueError("expected untouched base and trained adapter")
    base_score = score_predictions(rows, base_predictions)
    adapter_score = score_predictions(rows, adapter_predictions)
    random_score = score_predictions(rows, random_predictions(rows))
    visible_score = score_predictions(rows, visible_spine_predictions(rows))
    return {"split": split, "dataset_sha256": file_hash(data / f"{split}.jsonl"),
            "base_inference": base_manifest, "adapter_inference": adapter_manifest,
            "base": base_score, "adapter": adapter_score, "random_one": random_score,
            "visible_spine": visible_score,
            "paired_adapter_minus_base_95pct": paired_interval(adapter_score, base_score),
            "paired_adapter_minus_random_95pct": paired_interval(adapter_score, random_score)}


def selection(report: dict, report_sha256: str) -> dict:
    if report.get("split") != "dev" or not report.get("adapter_inference", {}).get("adapter_sha256"):
        raise ValueError("development adapter report required")
    if report["adapter"]["mean_loss_reduction"] <= max(
            0.0,
            report["base"]["mean_loss_reduction"], report["random_one"]["mean_loss_reduction"],
            report["visible_spine"]["mean_loss_reduction"]):
        raise ValueError("adapter did not beat base and one-flip controls on development")
    return {"version": 1, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "dev_dataset_sha256": report["dataset_sha256"], "dev_report_sha256": report_sha256,
            "adapter_sha256": report["adapter_inference"]["adapter_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.selection_output and args.selection_output.exists()):
        parser.error("output exists")
    report = evaluate(args.data, args.split, args.base, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    if args.selection_output:
        if args.split != "dev":
            parser.error("selection is development only")
        write_json(args.selection_output, selection(report, file_hash(args.output)))
    print(json.dumps({key: {name: value for name, value in report[key].items() if name != "records"}
                      for key in ("base", "adapter", "random_one", "visible_spine")}, sort_keys=True))


if __name__ == "__main__":
    main()
