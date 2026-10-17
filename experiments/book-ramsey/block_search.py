"""Search 3-by-9 block-circulant, 12-regular B4/B9 candidates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

from book import verify
from search import N, certificate, energy, flip


def groups() -> list[list[list[tuple[int, int]]]]:
    result: list[list[list[tuple[int, int]]]] = []
    for block in range(3):
        within = []
        for distance in range(1, 5):
            within.append([(block * 9 + i, block * 9 + (i + distance) % 9) for i in range(9)])
        result.append(within)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        between = []
        for shift in range(9):
            between.append([(left * 9 + i, right * 9 + (i + shift) % 9) for i in range(9)])
        result.append(between)
    return result


GROUPS = groups()
DEFAULT_COUNTS = [2, 2, 2, 4, 4, 4]


def graph(selected: list[set[int]]) -> list[int]:
    adjacency = [0] * N
    for group, choices in zip(GROUPS, selected):
        for choice in choices:
            for u, v in group[choice]:
                flip(adjacency, u, v)
    return adjacency


def run(seed: int, seconds: float, out: Path, counts: list[int] | None = None) -> dict[str, object]:
    rng = random.Random(seed)
    counts = DEFAULT_COUNTS if counts is None else counts
    if len(counts) != len(GROUPS) or any(not 0 <= count <= len(group)
                                         for group, count in zip(GROUPS, counts)):
        raise ValueError("invalid orbit counts")
    mutable_groups = [i for i, (group, count) in enumerate(zip(GROUPS, counts))
                      if 0 < count < len(group)]
    if not mutable_groups:
        raise ValueError("no mutable orbit group")
    selected = [set(rng.sample(range(len(group)), count)) for group, count in zip(GROUPS, counts)]
    adjacency = graph(selected)
    current, red_edges, _ = energy(adjacency)
    assert red_edges == 162 and all(row.bit_count() == 12 for row in adjacency)
    best = current
    best_step = 0
    proposals = accepted = 0
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        proposals += 1
        group_index = rng.choice(mutable_groups)
        old = rng.choice(tuple(selected[group_index]))
        new = rng.choice(tuple(set(range(len(GROUPS[group_index]))) - selected[group_index]))
        selected[group_index].remove(old)
        selected[group_index].add(new)
        candidate_adj = graph(selected)
        candidate, _, _ = energy(candidate_adj)
        fraction = (proposals % 5000) / 5000
        temperature = 4.0 * (0.1 / 4.0) ** fraction
        if candidate <= current or rng.random() < math.exp((current - candidate) / temperature):
            current = candidate
            adjacency = candidate_adj
            accepted += 1
        else:
            selected[group_index].remove(new)
            selected[group_index].add(old)
        if current < best:
            best = current
            best_step = proposals
            result = certificate(adjacency)
            result.update({"loss": best, "step": proposals, "seed": seed,
                           "method": "block_circulant", "selected_orbits": [sorted(s) for s in selected]})
            out.write_text(json.dumps(result, sort_keys=True) + "\n")
            if best == 0:
                break
    summary: dict[str, object] = {
        "method": "block_circulant", "seed": seed, "counts": counts, "best_loss": best,
        "best_step": best_step, "proposals": proposals, "accepted": accepted,
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }
    if best == 0:
        candidate = json.loads(out.read_text())
        exact = verify(candidate["n"], candidate["red_edges"], 4, 9)
        summary["exact_check"] = exact
        if not exact["valid"]:
            raise AssertionError("search loss and verifier disagree")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--out", type=Path, default=Path("search-results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1417, 1518, 1619])
    parser.add_argument("--counts", type=int, nargs=6, default=DEFAULT_COUNTS)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        result = run(seed, args.seconds, args.out / f"block_circulant-{seed}.json", args.counts)
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
        if result["best_loss"] == 0:
            break
    (args.out / "block-summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
