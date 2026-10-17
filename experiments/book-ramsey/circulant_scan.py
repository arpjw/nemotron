"""Exhaust all undirected degree-12 circulant graphs on 27 vertices."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from book import verify
from search import N, PAIRS, certificate, energy, flip


def main() -> None:
    best_loss = 10**9
    best_distances: tuple[int, ...] | None = None
    best_adjacency: list[int] | None = None
    scanned = 0
    best_count = 0
    for distances in itertools.combinations(range(1, 14), 6):
        adjacency = [0] * N
        for u, v in PAIRS:
            if min(v - u, N - v + u) in distances:
                flip(adjacency, u, v)
        loss, red_edges, _ = energy(adjacency)
        assert red_edges == 162 and all(row.bit_count() == 12 for row in adjacency)
        scanned += 1
        if loss < best_loss:
            best_loss, best_distances, best_adjacency = loss, distances, adjacency
            best_count = 1
        elif loss == best_loss:
            best_count += 1
    assert best_distances is not None and best_adjacency is not None
    candidate = certificate(best_adjacency)
    exact = verify(candidate["n"], candidate["red_edges"], 4, 9)
    assert (best_loss == 0) == exact["valid"]
    result = {
        "family": "27-vertex undirected circulants with 6 of 13 step distances",
        "scanned": scanned,
        "best_loss": best_loss,
        "best_count": best_count,
        "best_distances": list(best_distances),
        "exact_check_of_one_best": exact,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    Path(__file__).with_name("search-results").mkdir(exist_ok=True)
    Path(__file__).with_name("search-results").joinpath("circulant-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
