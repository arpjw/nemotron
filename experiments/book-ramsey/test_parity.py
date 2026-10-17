"""Check the counting identity and its integer edge-count consequence."""
from __future__ import annotations

import random
import unittest

from search import N, PAIRS, flip, parity_condition_excess


class ParityBoundTests(unittest.TestCase):
    def test_triangle_deficit_identity(self) -> None:
        rng = random.Random(3619)
        full = (1 << N) - 1
        for _ in range(100):
            adjacency = [0] * N
            for u, v in PAIRS:
                if rng.random() < 0.47:
                    flip(adjacency, u, v)
            degrees = [row.bit_count() for row in adjacency]
            red_edges = sum(degrees) // 2
            red_deficit = blue_deficit = 0
            incident_red_deficit = [0] * N
            for u, v in PAIRS:
                if adjacency[u] >> v & 1:
                    deficit = 3 - (adjacency[u] & adjacency[v]).bit_count()
                    red_deficit += deficit
                    incident_red_deficit[u] += deficit
                    incident_red_deficit[v] += deficit
                else:
                    blue_u = full & ~adjacency[u] & ~(1 << u)
                    blue_v = full & ~adjacency[v] & ~(1 << v)
                    blue_deficit += 8 - (blue_u & blue_v).bit_count()
            self.assertEqual(2 * (red_deficit + blue_deficit),
                             146 * red_edges - 11934 - 3 * sum(d * d for d in degrees))
            self.assertTrue(all(deficit % 2 == degree % 2
                                for deficit, degree in zip(incident_red_deficit, degrees)))
            self.assertEqual(parity_condition_excess(adjacency, red_edges),
                             max(0, sum(degree & 1 for degree in degrees)
                                 - 2 * (red_deficit + blue_deficit)))

    def test_balanced_degree_integer_range(self) -> None:
        possible = []
        for red_edges in range(352):
            low, remainder = divmod(2 * red_edges, N)
            degrees = [low] * (N - remainder) + [low + 1] * remainder
            if 3 * sum(d * d for d in degrees) + sum(d & 1 for d in degrees) \
                    <= 146 * red_edges - 11934:
                possible.append(red_edges)
        self.assertEqual(possible, list(range(157, 172)))


if __name__ == "__main__":
    unittest.main()
