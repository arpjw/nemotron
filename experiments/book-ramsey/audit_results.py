"""Recheck saved candidates and report their exact constraint violations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from book import verify
from search import N, energy, necessary_condition_excess, parity_condition_excess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("*.json")):
        if path.name == "candidate-audit.json":
            continue
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or "red_edges" not in value or "loss" not in value:
            continue
        n, red_edges = value["n"], value["red_edges"]
        exact = verify(n, red_edges, 4, 9)
        adjacency = [0] * n
        for u, v in red_edges:
            adjacency[u] |= 1 << v
            adjacency[v] |= 1 << u
        loss, m, _ = energy(adjacency)
        if n != N or loss != value["loss"] or m != exact["red_edges"]:
            raise AssertionError(f"inconsistent saved candidate: {path}")
        rows.append({
            "file": path.name,
            "method": value["method"],
            "seed": value["seed"],
            "loss": loss,
            "red_edges": m,
            "degree_condition_excess": necessary_condition_excess(adjacency, m),
            "parity_condition_excess": parity_condition_excess(adjacency, m),
            "red_violations": exact["red_violations"],
            "blue_violations": exact["blue_violations"],
            "valid": exact["valid"],
        })
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
