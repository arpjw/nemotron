"""Bounded exploratory search for a 27-vertex B4/B9 Ramsey witness.

This is a heuristic. Only an exact zero-loss certificate proves a lower bound.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from book import decode_graph6, verify

N = 27
PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N)]
FULL = (1 << N) - 1


def energy(adjacency: list[int]) -> tuple[int, int, int]:
    loss = 0
    red_edges = 0
    blue_edges = 0
    for u, v in PAIRS:
        if adjacency[u] >> v & 1:
            red_edges += 1
            pages = (adjacency[u] & adjacency[v]).bit_count()
            loss += max(0, pages - 3)
        else:
            blue_edges += 1
            pages = (FULL & ~(adjacency[u] | adjacency[v]) & ~(1 << u) & ~(1 << v)).bit_count()
            loss += max(0, pages - 8)
    return loss, red_edges, blue_edges


def necessary_condition_excess(adjacency: list[int], red_edges: int) -> int:
    """Positive part of the degree condition for a B4/B9 avoider."""
    squared_degrees = sum(row.bit_count() ** 2 for row in adjacency)
    return max(0, 3 * squared_degrees - 146 * red_edges + 11934)


def parity_condition_excess(adjacency: list[int], red_edges: int) -> int:
    """Positive part of the stronger condition from red triangle parity."""
    degrees = [row.bit_count() for row in adjacency]
    return max(0, 3 * sum(degree * degree for degree in degrees)
               + sum(degree & 1 for degree in degrees)
               - 146 * red_edges + 11934)


def flip(adjacency: list[int], u: int, v: int) -> None:
    adjacency[u] ^= 1 << v
    adjacency[v] ^= 1 << u


def random_graph(rng: random.Random, probability: float = 0.47) -> list[int]:
    adjacency = [0] * N
    for u, v in PAIRS:
        if rng.random() < probability:
            flip(adjacency, u, v)
    return adjacency


def extend_srg(rng: random.Random, graphs: list[list[int]]) -> list[int]:
    edges = rng.choice(graphs)
    adjacency = [0] * N
    permutation = list(range(26))
    rng.shuffle(permutation)
    for u, v in edges:
        flip(adjacency, permutation[u], permutation[v])
    for v in rng.sample(range(26), 12):
        flip(adjacency, v, 26)
    return adjacency


def certificate(adjacency: list[int]) -> dict[str, object]:
    return {"n": N, "red_edges": [[u, v] for u, v in PAIRS if adjacency[u] >> v & 1]}


def search(seed: int, method: str, steps: int, seconds: float, graphs: list[list[int]], out: Path) -> dict[str, object]:
    rng = random.Random(seed)
    adjacency = random_graph(rng) if method in {"random", "random_anneal"} else extend_srg(rng, graphs)
    current, red_edges, _ = energy(adjacency)
    current_score = current + (0.5 * necessary_condition_excess(adjacency, red_edges)
                               if method == "constrained_anneal" else 0)
    best = current
    best_at = 0
    start = time.monotonic()
    evaluated = 1
    improvements: list[dict[str, int]] = []
    for step in range(1, steps + 1):
        if time.monotonic() - start >= seconds:
            break
        u, v = rng.choice(PAIRS)
        flip(adjacency, u, v)
        candidate, red_edges, _ = energy(adjacency)
        candidate_score = candidate + (0.5 * necessary_condition_excess(adjacency, red_edges)
                                       if method == "constrained_anneal" else 0)
        evaluated += 1
        if method == "random":
            accept = True
        else:
            # Cyclic annealing allows escape from early local minima.
            fraction = (step % 4000) / 4000
            temperature = 1.8 * (0.08 / 1.8) ** fraction
            accept = candidate_score <= current_score or rng.random() < math.exp((current_score - candidate_score) / temperature)
        if accept:
            current = candidate
            current_score = candidate_score
        else:
            flip(adjacency, u, v)
        if current < best:
            best = current
            best_at = step
            result = certificate(adjacency)
            result["loss"] = best
            result["step"] = step
            result["seed"] = seed
            result["method"] = method
            out.write_text(json.dumps(result, sort_keys=True) + "\n")
            improvements.append({"step": step, "loss": best})
            if best == 0:
                break
    elapsed = time.monotonic() - start
    summary: dict[str, object] = {
        "seed": seed, "method": method, "evaluated": evaluated,
        "best_loss": best, "best_at_step": best_at,
        "elapsed_seconds": round(elapsed, 3), "improvements": improvements,
    }
    if best == 0:
        candidate = json.loads(out.read_text())
        exact = verify(candidate["n"], candidate["red_edges"], 4, 9)
        summary["exact_check"] = exact
        if not exact["valid"]:
            raise AssertionError("search energy and independent checker disagree")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--out", type=Path, default=Path("work/book-ramsey/search-results"))
    args = parser.parse_args()
    if args.seconds <= 0 or args.steps <= 0:
        parser.error("seconds and steps must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("sr261034.g6")
    graphs = [decode_graph6(line)[1] for line in source.read_text().splitlines() if line.strip()]
    if len(graphs) != 10 or not all(verify(26, edges, 4, 9)["valid"] for edges in graphs):
        raise RuntimeError("SRG calibration failed")
    summaries = []
    for method, seed in [("random", 101), ("random_anneal", 202), ("random_anneal", 303),
                         ("srg_anneal", 404), ("srg_anneal", 505),
                         ("constrained_anneal", 606), ("constrained_anneal", 707)]:
        summary = search(seed, method, args.steps, args.seconds, graphs,
                         args.out / f"{method}-{seed}.json")
        print(json.dumps(summary, sort_keys=True), flush=True)
        summaries.append(summary)
        if summary["best_loss"] == 0:
            break
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
