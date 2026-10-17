"""Bounded 12-regular graph search using degree-preserving two-edge switches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

from book import verify
from search import N, PAIRS, certificate, energy, flip


def start_graph() -> tuple[list[int], list[tuple[int, int]]]:
    adjacency = [0] * N
    for u, v in PAIRS:
        distance = min((v - u) % N, (u - v) % N)
        if 1 <= distance <= 6:
            flip(adjacency, u, v)
    edges = [(u, v) for u, v in PAIRS if adjacency[u] >> v & 1]
    assert len(edges) == 162 and all(row.bit_count() == 12 for row in adjacency)
    return adjacency, edges


def propose(rng: random.Random, adjacency: list[int], edges: list[tuple[int, int]]) -> tuple[int, int, tuple[int, int], tuple[int, int]] | None:
    i = rng.randrange(len(edges))
    j = rng.randrange(len(edges) - 1)
    if j >= i:
        j += 1
    u, v = edges[i]
    x, y = edges[j]
    if rng.randrange(2):
        u, v = v, u
    if rng.randrange(2):
        x, y = y, x
    if len({u, v, x, y}) != 4 or adjacency[u] >> x & 1 or adjacency[v] >> y & 1:
        return None
    return i, j, (min(u, x), max(u, x)), (min(v, y), max(v, y))


def apply_switch(adjacency: list[int], old1: tuple[int, int], old2: tuple[int, int],
                 new1: tuple[int, int], new2: tuple[int, int]) -> None:
    for u, v in (old1, old2, new1, new2):
        flip(adjacency, u, v)


def randomize(rng: random.Random, adjacency: list[int], edges: list[tuple[int, int]], switches: int) -> None:
    accepted = 0
    while accepted < switches:
        move = propose(rng, adjacency, edges)
        if move is None:
            continue
        i, j, new1, new2 = move
        apply_switch(adjacency, edges[i], edges[j], new1, new2)
        edges[i], edges[j] = new1, new2
        accepted += 1


def run(seed: int, seconds: float, out: Path) -> dict[str, object]:
    rng = random.Random(seed)
    adjacency, edges = start_graph()
    randomize(rng, adjacency, edges, 4000)
    current, red_edges, _ = energy(adjacency)
    assert red_edges == 162 and all(row.bit_count() == 12 for row in adjacency)
    best = current
    best_step = 0
    evaluated = 1
    accepted = 0
    start = time.monotonic()
    step = 0
    while time.monotonic() - start < seconds:
        step += 1
        move = propose(rng, adjacency, edges)
        if move is None:
            continue
        i, j, new1, new2 = move
        old1, old2 = edges[i], edges[j]
        apply_switch(adjacency, old1, old2, new1, new2)
        candidate, _, _ = energy(adjacency)
        evaluated += 1
        fraction = (step % 10000) / 10000
        temperature = 2.5 * (0.1 / 2.5) ** fraction
        if candidate <= current or rng.random() < math.exp((current - candidate) / temperature):
            current = candidate
            edges[i], edges[j] = new1, new2
            accepted += 1
        else:
            apply_switch(adjacency, old1, old2, new1, new2)
        if current < best:
            best = current
            best_step = step
            result = certificate(adjacency)
            result.update({"loss": best, "step": step, "seed": seed, "method": "regular_switch"})
            out.write_text(json.dumps(result, sort_keys=True) + "\n")
            if best == 0:
                break
    summary: dict[str, object] = {
        "method": "regular_switch", "seed": seed, "best_loss": best,
        "best_step": best_step, "proposals": step, "evaluated": evaluated,
        "accepted": accepted, "elapsed_seconds": round(time.monotonic() - start, 3),
    }
    if best == 0:
        candidate = json.loads(out.read_text())
        exact = verify(candidate["n"], candidate["red_edges"], 4, 9)
        summary["exact_check"] = exact
        if not exact["valid"]:
            raise AssertionError("two independent losses disagree")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=Path("search-results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[811, 912, 1013])
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        result = run(seed, args.seconds, args.out / f"regular_switch-{seed}.json")
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
        if result["best_loss"] == 0:
            break
    (args.out / "regular-summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
