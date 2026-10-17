"""Try a frozen development puzzle and receive exact verifier feedback."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_split
from .evaluate import parse_graph
from .graph import invariants
from .task import Task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    rows = load_split(Path(args.data), "dev")
    if not 0 <= args.index < len(rows):
        parser.error("index outside development split")
    task = Task.from_json(rows[args.index]["task"])
    print(task.prompt())
    for attempt in range(1, 6):
        try:
            raw = input(f"graph {attempt}/5> ")
        except EOFError:
            break
        graph = parse_graph(raw, task.n)
        if graph is None:
            print("Invalid JSON graph. Use sorted unique edge pairs and the required n.")
            continue
        margin = task.margin(graph)
        print(f"margin={margin}; invariants={invariants(graph)}")
        if margin > 0:
            print("Counterexample verified.")
            break


if __name__ == "__main__":
    main()
