"""Deterministic synthetic task generation and frozen-split verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .graph import Graph, from_mask, invariants
from .task import FEATURES, Task

SEED = 20261015
COUNTS = {"train": 120, "dev": 30, "test": 30}
PAIR_SPLITS = {"train": 32, "dev": 8, "test": 8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def random_graph(n: int, rng: random.Random) -> Graph:
    probability = rng.choice((0.18, 0.35, 0.5, 0.65, 0.82))
    edges = tuple((u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < probability)
    return Graph(n, edges)


def sample_graphs(n: int, rng: random.Random, count: int = 64) -> list[Graph]:
    graphs = [from_mask(n, 0), from_mask(n, (1 << (n * (n - 1) // 2)) - 1)]
    graphs.append(Graph(n, tuple(sorted((u, (u + 1) % n) if u < (u + 1) % n
                                        else ((u + 1) % n, u) for u in range(n)))))
    graphs.extend(random_graph(n, rng) for _ in range(count - len(graphs)))
    return graphs


def generate(seed: int = SEED, counts: dict[str, int] | None = None) -> dict[str, list[dict]]:
    """Freeze invariant-pair families across splits before generating tasks.

    Targets come from candidate graphs sampled before the later evaluation run.
    No model outcomes or evaluation predictions influence this generator.
    """
    rng = random.Random(seed)
    pairs = [(left, right) for left in FEATURES for right in FEATURES if left != right]
    rng.shuffle(pairs)
    partitions = {}
    start = 0
    for split, size in PAIR_SPLITS.items():
        partitions[split] = pairs[start:start + size]
        start += size
    output: dict[str, list[dict]] = {}
    for split, size in (counts or COUNTS).items():
        if split not in partitions or size < 1:
            raise ValueError("invalid split or count")
        rows = []
        for index in range(size):
            lhs, rhs = partitions[split][index % len(partitions[split])]
            n = 5 + (index // len(partitions[split])) % 4
            a, b = rng.randint(1, 3), rng.randint(1, 3)
            graphs = sample_graphs(n, rng)
            interior = [graph for graph in graphs if 2 <= len(graph.edges) <= n * (n - 1) // 2 - 2]
            differences = sorted(a * invariants(graph)[lhs] - b * invariants(graph)[rhs] for graph in interior)
            # Put the threshold near the high tail. The exact verifier, not this
            # sample, decides whether any submitted graph is a counterexample.
            c = min(differences[int(0.90 * (len(differences) - 1))], differences[-1] - 1)
            witnesses = [graph for graph in interior
                         if a * invariants(graph)[lhs] - b * invariants(graph)[rhs] > c]
            assert witnesses
            witness = min(witnesses, key=lambda graph: (len(graph.edges), -(a * invariants(graph)[lhs] - b * invariants(graph)[rhs] - c), graph.edges))
            task = Task(f"{split}-{index:04d}", split, n, lhs, rhs, a, b, c)
            rows.append({"task": task.to_json(), "target": witness.to_json(), "target_margin": task.margin(witness)})
        output[split] = rows
    return output


def write_dataset(directory: Path, *, seed: int = SEED, counts: dict[str, int] | None = None) -> dict:
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    data = generate(seed, counts)
    manifest = {"version": 1, "seed": seed, "counts": {split: len(rows) for split, rows in data.items()},
                "partition": "disjoint ordered invariant pairs; task signatures unique within split", "files": {}}
    for split, rows in data.items():
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
        manifest["files"][path.name] = sha256(path)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_split(directory: Path, split: str) -> list[dict]:
    if split not in COUNTS:
        raise ValueError("unknown split")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("version") != 1 or set(manifest.get("files", {})) != {"train.jsonl", "dev.jsonl", "test.jsonl"}:
        raise ValueError("invalid data manifest")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"changed frozen split: {name}")
    seen: set[tuple] = set()
    families: dict[str, set[tuple[str, str]]] = {}
    selected = []
    for name in manifest["files"]:
        part = name.removesuffix(".jsonl")
        rows = [json.loads(line) for line in (directory / name).read_text().splitlines()]
        if len(rows) != manifest["counts"][part]:
            raise ValueError("count mismatch")
        families[part] = set()
        for row in rows:
            if set(row) != {"task", "target", "target_margin"}:
                raise ValueError("wrong row schema")
            task = Task.from_json(row["task"])
            graph = Graph.from_json(row["target"], expected_n=task.n)
            if task.split != part or task.margin(graph) != row["target_margin"] or row["target_margin"] <= 0:
                raise ValueError("invalid witness or split")
            signature = (task.n, task.lhs, task.rhs, task.a, task.b, task.c)
            if signature in seen:
                raise ValueError("duplicate task")
            seen.add(signature)
            families[part].add((task.lhs, task.rhs))
        if part == split:
            selected = rows
    if any(families[a] & families[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("invariant-pair leakage across splits")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output), indent=2))


if __name__ == "__main__":
    main()
