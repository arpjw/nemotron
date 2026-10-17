"""Restricted, auditable language of graph inequalities."""
from __future__ import annotations

from dataclasses import dataclass

from .graph import Graph, invariants

FEATURES = ("n", "m", "delta", "alpha", "omega", "chi", "nu", "alpha_omega")
SYMBOLS = {"n": "|V|", "m": "|E|", "delta": "Δ", "alpha": "α", "omega": "ω",
           "chi": "χ", "nu": "ν", "alpha_omega": "αω"}


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    n: int
    lhs: str
    rhs: str
    a: int
    b: int
    c: int

    def __post_init__(self) -> None:
        if not self.id or self.split not in {"train", "dev", "test"}:
            raise ValueError("invalid task identity or split")
        if self.lhs not in FEATURES or self.rhs not in FEATURES or self.lhs == self.rhs:
            raise ValueError("invalid invariant pair")
        if any(type(value) is not int for value in (self.n, self.a, self.b, self.c)):
            raise ValueError("integer task parameters required")
        if not 4 <= self.n <= 8 or not 1 <= self.a <= 3 or not 1 <= self.b <= 3 or not -100 <= self.c <= 100:
            raise ValueError("task parameters out of range")

    def margin(self, graph: Graph) -> int:
        if graph.n != self.n:
            raise ValueError("wrong graph order")
        values = invariants(graph)
        return self.a * values[self.lhs] - self.b * values[self.rhs] - self.c

    def prompt(self) -> str:
        return (f"Find a counterexample among simple undirected graphs on vertices 0..{self.n - 1} "
                f"to the claim {self.a}{SYMBOLS[self.lhs]}(G) ≤ {self.b}{SYMBOLS[self.rhs]}(G) "
                f"+ ({self.c}). Here χ is chromatic number, ω clique number, α independence "
                "number, ν maximum matching size, Δ maximum degree, |V| vertex count, "
                "|E| edge count, and αω "
                "their product. Return only JSON: {\"n\": integer, \"edges\": [[u,v],...]}. "
                "Use sorted unique edges with u < v. A valid answer must make left minus right positive.")

    def to_json(self) -> dict:
        return {key: getattr(self, key) for key in ("id", "split", "n", "lhs", "rhs", "a", "b", "c")}

    @classmethod
    def from_json(cls, value: dict) -> Task:
        if set(value) != {"id", "split", "n", "lhs", "rhs", "a", "b", "c"}:
            raise ValueError("wrong task schema")
        return cls(**value)
