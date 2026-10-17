"""Cross-check the optional SAT encoding on every five-vertex graph."""
from __future__ import annotations

import unittest

from book import verify

try:
    from pysat.solvers import Solver
    from sat_search import build
except ImportError:
    Solver = None
    build = None


@unittest.skipIf(Solver is None, "python-sat is not installed")
class SATEncodingTests(unittest.TestCase):
    def test_all_five_vertex_assignments(self) -> None:
        assert Solver is not None and build is not None
        for red_book, blue_book in ((1, 2), (2, 1), (2, 2)):
            cnf, _, edge = build(5, red_book, blue_book, symmetry_break=False)
            pairs = list(edge)
            with Solver(name="glucose4", bootstrap_with=cnf.clauses) as solver:
                for mask in range(1 << len(pairs)):
                    red = [list(pair) for i, pair in enumerate(pairs) if mask >> i & 1]
                    assumptions = [variable if mask >> i & 1 else -variable
                                   for i, variable in enumerate(edge.values())]
                    self.assertEqual(
                        solver.solve(assumptions=assumptions),
                        verify(5, red, red_book, blue_book)["valid"],
                    )


if __name__ == "__main__":
    unittest.main()
