"""Fixed-budget random-search baseline; no model or GPU required."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import load_split, random_graph, sha256
from .graph import Graph
from .task import Task


def search(task: Task, *, budget: int = 32, seed: int = 0) -> tuple[Graph, int]:
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = random.Random(seed)
    best = None
    for _ in range(budget):
        graph = random_graph(task.n, rng)
        margin = task.margin(graph)
        key = (margin > 0, margin, -len(graph.edges))
        if best is None or key > best[0]:
            best = (key, graph)
    assert best is not None
    return best[1], task.margin(best[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=32)
    args = parser.parse_args()
    rows = load_split(args.data, args.split)
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for index, row in enumerate(rows):
            task = Task.from_json(row["task"])
            graph, margin = search(task, budget=args.budget, seed=871 + index)
            stream.write(json.dumps({"id": task.id, "graph": graph.to_json(), "margin": margin}) + "\n")
    print(json.dumps({"rows": len(rows), "successes": sum(json.loads(line)["margin"] > 0 for line in args.output.read_text().splitlines()),
                      "budget": args.budget, "predictions_sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
