"""Bounded annealing inside the necessary degree-feasible region."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

from book import verify
from regular_search import randomize, start_graph
from search import N, PAIRS, certificate, energy, flip, parity_condition_excess


def run(seed: int, seconds: float, out: Path) -> dict[str, object]:
    rng = random.Random(seed)
    adjacency, edges = start_graph()
    randomize(rng, adjacency, edges, 4000)
    current, red_edges, _ = energy(adjacency)
    assert red_edges == 162 and parity_condition_excess(adjacency, red_edges) == 0
    best = current
    best_step = 0
    step = evaluated = accepted = infeasible = 0
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        step += 1
        u, v = rng.choice(PAIRS)
        flip(adjacency, u, v)
        new_m = red_edges + (1 if adjacency[u] >> v & 1 else -1)
        if not 157 <= new_m <= 171 or parity_condition_excess(adjacency, new_m) > 0:
            flip(adjacency, u, v)
            infeasible += 1
            continue
        candidate, _, _ = energy(adjacency)
        evaluated += 1
        fraction = (step % 8000) / 8000
        temperature = 2.0 * (0.07 / 2.0) ** fraction
        if candidate <= current or rng.random() < math.exp((current - candidate) / temperature):
            current = candidate
            red_edges = new_m
            accepted += 1
        else:
            flip(adjacency, u, v)
        if current < best:
            best = current
            best_step = step
            result = certificate(adjacency)
            result.update({"loss": best, "step": step, "seed": seed, "method": "feasible_flip"})
            out.write_text(json.dumps(result, sort_keys=True) + "\n")
            if best == 0:
                break
    summary: dict[str, object] = {
        "method": "feasible_flip", "seed": seed, "best_loss": best,
        "best_step": best_step, "proposals": step, "evaluated": evaluated,
        "accepted": accepted, "infeasible": infeasible,
        "elapsed_seconds": round(time.monotonic() - start, 3),
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
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--out", type=Path, default=Path("search-results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1114, 1215, 1316])
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        result = run(seed, args.seconds, args.out / f"feasible_flip-{seed}.json")
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
        if result["best_loss"] == 0:
            break
    (args.out / "feasible-summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
