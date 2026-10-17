"""Independently replay saved transfer actions with set-based book counting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from book import verify
from transfer_v2 import N, RED_BOOK, BLUE_BOOK, load_split, sha256
from transfer_v2_evaluate import parse_action


def independent_loss(edges: set[tuple[int, int]]) -> int:
    vertices = set(range(N))
    red_neighbors = [set() for _ in range(N)]
    for u, v in edges:
        red_neighbors[u].add(v)
        red_neighbors[v].add(u)
    loss = 0
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) in edges:
                pages = len(red_neighbors[u] & red_neighbors[v])
                loss += max(0, pages - RED_BOOK + 1)
            else:
                common_blue = (vertices - red_neighbors[u] - {u, v}) & \
                              (vertices - red_neighbors[v] - {u, v})
                loss += max(0, len(common_blue) - BLUE_BOOK + 1)
    return loss


def audit(data: Path, split: str, predictions: Path, report: Path, method: str) -> dict:
    rows = load_split(data, split)
    actions = [json.loads(line) for line in predictions.read_text().splitlines()]
    result = json.loads(report.read_text())
    records = result[method]["records"]
    if len(actions) != len(rows) or len(records) != len(rows):
        raise ValueError("record count mismatch")
    for row, prediction, recorded in zip(rows, actions, records):
        if row["id"] != prediction["id"] or row["id"] != recorded["id"]:
            raise ValueError("record alignment mismatch")
        adjacency = [int(word, 16) for word in row["rows_hex"]]
        edges = {(u, v) for u in range(N) for v in range(u + 1, N) if adjacency[u] >> v & 1}
        before = independent_loss(edges)
        if before != row["loss"] or before != recorded["initial_loss"]:
            raise AssertionError("start loss mismatch")
        action = parse_action(prediction["raw"])
        if action is not None:
            if action in edges:
                edges.remove(action)
            else:
                edges.add(action)
        after = independent_loss(edges)
        if after != recorded["next_loss"] or \
                (list(action) if action else None) != recorded["action"]:
            raise AssertionError("action replay mismatch")
        checked = verify(N, [list(edge) for edge in sorted(edges)], RED_BOOK, BLUE_BOOK)
        if (after == 0) != checked["valid"]:
            raise AssertionError("zero-loss certificate mismatch")
    return {"version": 2, "split": split, "method": method,
            "rows": len(rows), "dataset_sha256": sha256(data / f"{split}.jsonl"),
            "predictions_sha256": sha256(predictions), "report_sha256": sha256(report),
            "independent_replay": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--method", choices=("base", "adapter"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = audit(args.data, args.split, args.predictions, args.report, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
