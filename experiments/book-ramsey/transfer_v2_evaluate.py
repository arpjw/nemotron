"""Score frozen B6/B6 transfer predictions against exact local controls."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from book import verify
from edit_evaluate import paired_interval
from edit_model import MODEL_ID, MODEL_REVISION, write_json
from transfer_v2 import (N, BLUE_BOOK, PAIRS, RED_BOOK, edge_list, flip, load_split,
                         score, sha256, spines)


def parse_action(raw: str) -> tuple[int, int] | None:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"u", "v"} or \
            any(type(value[key]) is not int for key in ("u", "v")):
        return None
    u, v = value["u"], value["v"]
    return (u, v) if 0 <= u < v < N else None


def score_actions(rows: list[dict], actions: list[tuple[int, int] | None]) -> dict:
    if len(rows) != len(actions):
        raise ValueError("action count mismatch")
    records = []
    for row, action in zip(rows, actions):
        adjacency = [int(word, 16) for word in row["rows_hex"]]
        next_loss = row["loss"]
        if action is not None:
            flip(adjacency, *action)
            next_loss = score(adjacency)[0]
            exact = verify(N, edge_list(adjacency), RED_BOOK, BLUE_BOOK)
            if (next_loss == 0) != exact["valid"]:
                raise AssertionError("exact score and verifier disagree")
        records.append({"id": row["id"], "action": list(action) if action else None,
                        "initial_loss": row["loss"], "next_loss": next_loss,
                        "improved": next_loss < row["loss"], "solved": next_loss == 0})
    return {"rows": len(records), "valid_actions": sum(x["action"] is not None for x in records),
            "improvements": sum(x["improved"] for x in records),
            "solved": sum(x["solved"] for x in records),
            "mean_loss_reduction": sum(x["initial_loss"] - x["next_loss"]
                                       for x in records) / len(records), "records": records}


def load_predictions(directory: Path, data: Path, split: str, *, adapter: bool) -> list[tuple[int, int] | None]:
    manifest = json.loads((directory / "inference_manifest.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("version") != 2 or \
            manifest.get("split") != split or manifest.get("model_id") != MODEL_ID or \
            manifest.get("model_revision") != MODEL_REVISION or \
            manifest.get("dataset_sha256") != sha256(data / f"{split}.jsonl") or \
            manifest.get("predictions_sha256") != sha256(directory / "predictions.jsonl") or \
            bool(manifest.get("adapter_sha256")) != adapter:
        raise ValueError("transfer inference provenance mismatch")
    if adapter and manifest["adapter_sha256"] != \
            "606c4efd0a33a808399daaab7a580ad3253181d1c65cece5667083153104e2cf":
        raise ValueError("wrong adapter")
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    rows = load_split(data, split)
    if len(predictions) != len(rows) or any(set(p) != {"id", "raw"} or p["id"] != row["id"]
                                            for p, row in zip(predictions, rows)):
        raise ValueError("prediction IDs, count, or schema mismatch")
    return [parse_action(p["raw"]) for p in predictions]


def controls(rows: list[dict]) -> dict[str, dict]:
    random_actions = []
    visible_actions = []
    shown_best_actions = []
    teacher_actions = []
    for index, row in enumerate(rows):
        adjacency = [int(word, 16) for word in row["rows_hex"]]
        random_actions.append(random.Random(871 + index).choice(PAIRS))
        visible = spines(adjacency)[:12]
        _, u, v, _ = random.Random(971 + index).choice(visible)
        visible_actions.append((u, v))
        candidates = []
        for _, u, v, _ in visible:
            flip(adjacency, u, v)
            candidates.append((score(adjacency)[0], u, v))
            flip(adjacency, u, v)
        _, u, v = min(candidates)
        shown_best_actions.append((u, v))
        teacher_actions.append(tuple(row["teacher_action"]))
    return {"random_edge": score_actions(rows, random_actions),
            "random_visible_spine": score_actions(rows, visible_actions),
            "best_visible_twelve": score_actions(rows, shown_best_actions),
            "exact_teacher": score_actions(rows, teacher_actions)}


def greedy_repair(rows: list[dict]) -> dict:
    started = time.monotonic()
    results = []
    for row in rows:
        adjacency = [int(word, 16) for word in row["rows_hex"]]
        current = row["loss"]
        calls = 0
        for _ in range(5):
            if current == 0:
                break
            candidates = []
            for u, v in PAIRS:
                flip(adjacency, u, v)
                candidates.append((score(adjacency)[0], u, v))
                flip(adjacency, u, v)
                calls += 1
            best, u, v = min(candidates)
            if best >= current:
                break
            flip(adjacency, u, v)
            current = best
        exact = verify(N, edge_list(adjacency), RED_BOOK, BLUE_BOOK)
        if (current == 0) != exact["valid"]:
            raise AssertionError("greedy repair verification mismatch")
        results.append({"id": row["id"], "best_loss": current,
                        "success": exact["valid"], "calls": calls})
    return {"successes": sum(item["success"] for item in results),
            "mean_calls": sum(item["calls"] for item in results) / len(results),
            "elapsed_seconds": time.monotonic() - started, "records": results}


def paired_anchor_interval(rows: list[dict], candidate: dict, baseline: dict,
                           *, draws: int = 2000, seed: int = 872) -> list[float]:
    """Bootstrap source graph6 records, rather than treating relabelings as independent."""
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["anchor_id"], []).append(index)
    anchors = sorted(groups)
    deltas = [(a["initial_loss"] - a["next_loss"]) -
              (b["initial_loss"] - b["next_loss"])
              for a, b in zip(candidate["records"], baseline["records"])]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        indices = [index for _ in anchors for index in groups[rng.choice(anchors)]]
        samples.append(sum(deltas[index] for index in indices) / len(indices))
    samples.sort()
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws)]]


def evaluate(data: Path, split: str, base: Path, adapter: Path) -> dict:
    rows = load_split(data, split)
    base_score = score_actions(rows, load_predictions(base, data, split, adapter=False))
    adapter_score = score_actions(rows, load_predictions(adapter, data, split, adapter=True))
    controls_score = controls(rows)
    return {"version": 2, "split": split, "dataset_sha256": sha256(data / f"{split}.jsonl"),
            "base": base_score, "adapter": adapter_score, **controls_score,
            "greedy_five_pass": greedy_repair(rows),
            "paired_adapter_minus_base_95pct": paired_interval(adapter_score, base_score),
            "paired_adapter_minus_base_anchor_95pct": paired_anchor_interval(
                rows, adapter_score, base_score)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    report = evaluate(args.data, args.split, args.base, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({name: {key: value for key, value in report[name].items() if key != "records"}
                      for name in ("base", "adapter", "random_edge", "random_visible_spine",
                                   "best_visible_twelve", "exact_teacher", "greedy_five_pass")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
