"""Exact certificate checker for two-color book Ramsey constructions.

B_k is k triangles sharing a spine edge. Red edges are listed explicitly;
all other pairs are blue. An edge's same-color common-neighbor count equals
the number of pages in its largest monochromatic book.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def decode_graph6(line: str) -> tuple[int, list[list[int]]]:
    """Decode the ordinary graph6 form for 2..62 vertices."""
    data = line.strip().encode("ascii")
    if not data or not 65 <= data[0] <= 125:
        raise ValueError("expected ordinary graph6 with 2..62 vertices")
    n = data[0] - 63
    expected = 1 + (n * (n - 1) // 2 + 5) // 6
    if len(data) != expected or any(not 63 <= byte <= 126 for byte in data[1:]):
        raise ValueError("invalid graph6 length or byte")
    bits = [(byte - 63) >> shift & 1 for byte in data[1:] for shift in range(5, -1, -1)]
    edges = []
    position = 0
    for v in range(1, n):
        for u in range(v):
            if bits[position]:
                edges.append([u, v])
            position += 1
    edges.sort()
    return n, edges


def verify(n: int, red_edges: list[list[int]], red_book: int, blue_book: int) -> dict[str, int | bool]:
    if type(n) is not int or not 2 <= n <= 64:
        raise ValueError("n must be an integer in 2..64")
    if type(red_book) is not int or type(blue_book) is not int or min(red_book, blue_book) < 1:
        raise ValueError("book sizes must be positive integers")
    if not isinstance(red_edges, list) or any(
        not isinstance(pair, list) or len(pair) != 2 or
        any(type(vertex) is not int for vertex in pair) or
        not 0 <= pair[0] < pair[1] < n
        for pair in red_edges
    ):
        raise ValueError("red_edges must be ascending vertex pairs")
    pairs = [tuple(pair) for pair in red_edges]
    if pairs != sorted(set(pairs)):
        raise ValueError("red_edges must be sorted and unique")

    adjacency = [0] * n
    for u, v in pairs:
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u
    full = (1 << n) - 1
    max_red_pages = max_blue_pages = 0
    red_violations = blue_violations = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u] & (1 << v):
                pages = (adjacency[u] & adjacency[v]).bit_count()
                max_red_pages = max(max_red_pages, pages)
                red_violations += pages >= red_book
            else:
                blue_u = full & ~adjacency[u] & ~(1 << u)
                blue_v = full & ~adjacency[v] & ~(1 << v)
                pages = (blue_u & blue_v).bit_count()
                max_blue_pages = max(max_blue_pages, pages)
                blue_violations += pages >= blue_book
    return {
        "n": n,
        "red_edges": len(pairs),
        "blue_edges": n * (n - 1) // 2 - len(pairs),
        "max_red_pages": max_red_pages,
        "max_blue_pages": max_blue_pages,
        "red_violations": red_violations,
        "blue_violations": blue_violations,
        "valid": red_violations == 0 and blue_violations == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--red-book", type=int, default=4)
    parser.add_argument("--blue-book", type=int, default=9)
    args = parser.parse_args()
    value = json.loads(args.certificate.read_text())
    if not isinstance(value, dict) or set(value) != {"n", "red_edges"}:
        parser.error("certificate must contain exactly n and red_edges")
    print(json.dumps(verify(value["n"], value["red_edges"], args.red_book, args.blue_book), sort_keys=True))


if __name__ == "__main__":
    main()
