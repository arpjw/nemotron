"""Small simple graphs and exact integer-valued invariants.

The deliberately small vertex cap makes every reported witness independently
checkable without a numerical eigensolver or an approximate graph algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MAX_VERTICES = 8


@dataclass(frozen=True)
class Graph:
    n: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.n) is not int or not 1 <= self.n <= MAX_VERTICES:
            raise ValueError(f"n must be an integer in 1..{MAX_VERTICES}")
        if type(self.edges) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 or
            any(type(vertex) is not int for vertex in pair) or
            not 0 <= pair[0] < pair[1] < self.n for pair in self.edges
        ) or tuple(sorted(set(self.edges))) != self.edges:
            raise ValueError("edges must be sorted, unique pairs 0 <= u < v < n")

    @classmethod
    def from_json(cls, value: object, *, expected_n: int | None = None) -> Graph:
        if not isinstance(value, dict) or set(value) != {"n", "edges"}:
            raise ValueError("graph must contain exactly n and edges")
        raw = value["edges"]
        if not isinstance(raw, list) or any(not isinstance(pair, list) or len(pair) != 2 for pair in raw):
            raise ValueError("edges must be a list of pairs")
        graph = cls(value["n"], tuple(tuple(pair) for pair in raw))
        if expected_n is not None and graph.n != expected_n:
            raise ValueError("wrong vertex count")
        return graph

    def to_json(self) -> dict:
        return {"n": self.n, "edges": [list(pair) for pair in self.edges]}


def from_mask(n: int, mask: int) -> Graph:
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    if type(mask) is not int or mask < 0 or mask >= 1 << len(pairs):
        raise ValueError("edge mask out of range")
    return Graph(n, tuple(pair for index, pair in enumerate(pairs) if mask & (1 << index)))


@lru_cache(maxsize=32768)
def invariants(graph: Graph) -> dict[str, int]:
    n = graph.n
    adj = [0] * n
    for u, v in graph.edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    degree = [bits.bit_count() for bits in adj]

    def is_clique(mask: int) -> bool:
        return all((adj[v] & (mask ^ (1 << v))) == (mask ^ (1 << v))
                   for v in range(n) if mask & (1 << v))

    def is_independent(mask: int) -> bool:
        return all(not (adj[v] & mask) for v in range(n) if mask & (1 << v))

    omega = max(mask.bit_count() for mask in range(1 << n) if is_clique(mask))
    alpha = max(mask.bit_count() for mask in range(1 << n) if is_independent(mask))

    @lru_cache(maxsize=None)
    def chromatic(remaining: int, classes: tuple[int, ...]) -> int:
        if not remaining:
            return len(classes)
        vertex = max((v for v in range(n) if remaining & (1 << v)), key=lambda v: degree[v])
        next_remaining = remaining ^ (1 << vertex)
        best = n
        for index, members in enumerate(classes):
            if not adj[vertex] & members:
                updated = list(classes)
                updated[index] |= 1 << vertex
                best = min(best, chromatic(next_remaining, tuple(updated)))
        return min(best, chromatic(next_remaining, classes + (1 << vertex,)))

    @lru_cache(maxsize=None)
    def matching(remaining: int) -> int:
        if not remaining:
            return 0
        bit = remaining & -remaining
        u = bit.bit_length() - 1
        rest = remaining ^ bit
        best = matching(rest)
        neighbors = adj[u] & rest
        while neighbors:
            vbit = neighbors & -neighbors
            best = max(best, 1 + matching(rest ^ vbit))
            neighbors ^= vbit
        return best

    return {"n": n, "m": len(graph.edges), "delta": max(degree), "alpha": alpha,
            "omega": omega, "chi": chromatic((1 << n) - 1, ()),
            "nu": matching((1 << n) - 1), "alpha_omega": alpha * omega}
