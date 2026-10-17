"""Independent small-graph and known-construction checks."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import unittest

from book import decode_graph6, verify


def brute_valid(n: int, red: set[tuple[int, int]], red_book: int, blue_book: int) -> bool:
    def same_color(color: str, u: int, v: int) -> bool:
        edge = (min(u, v), max(u, v))
        return (edge in red) == (color == "red")

    for u, v in itertools.combinations(range(n), 2):
        for color, limit in (("red", red_book), ("blue", blue_book)):
            if not same_color(color, u, v):
                continue
            pages = sum(same_color(color, u, w) and same_color(color, v, w)
                        for w in range(n) if w not in (u, v))
            if pages >= limit:
                return False
    return True


class BookVerifierTests(unittest.TestCase):
    def test_all_five_vertex_graphs(self) -> None:
        pairs = list(itertools.combinations(range(5), 2))
        for mask in range(1 << len(pairs)):
            red = {pair for i, pair in enumerate(pairs) if mask >> i & 1}
            edges = [list(pair) for pair in sorted(red)]
            for red_book, blue_book in ((1, 2), (2, 1), (2, 2)):
                with self.subTest(mask=mask, red_book=red_book, blue_book=blue_book):
                    self.assertEqual(
                        verify(5, edges, red_book, blue_book)["valid"],
                        brute_valid(5, red, red_book, blue_book),
                    )

    def test_known_26_vertex_graphs(self) -> None:
        source = Path(os.environ.get("BOOK_GRAPH6_SOURCE", Path(__file__).with_name("sr261034.g6")))
        if not source.exists():
            self.skipTest("download McKay's sr261034.g6 to run calibration")
        graphs = [decode_graph6(line) for line in source.read_text().splitlines() if line.strip()]
        self.assertEqual(len(graphs), 10)
        for n, edges in graphs:
            outcome = verify(n, edges, 4, 9)
            self.assertEqual(n, 26)
            self.assertEqual(outcome["red_edges"], 130)
            self.assertEqual(outcome["max_red_pages"], 3)
            self.assertEqual(outcome["max_blue_pages"], 8)
            self.assertTrue(outcome["valid"])


if __name__ == "__main__":
    unittest.main()
