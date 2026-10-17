import unittest

from graph_ce.graph import Graph, from_mask, invariants


class GraphTests(unittest.TestCase):
    def test_known_graphs(self):
        empty = Graph(5, ())
        self.assertEqual(invariants(empty), {"n": 5, "m": 0, "delta": 0, "alpha": 5,
                                             "omega": 1, "chi": 1, "nu": 0, "alpha_omega": 5})
        complete = from_mask(4, (1 << 6) - 1)
        self.assertEqual(invariants(complete)["chi"], 4)
        self.assertEqual(invariants(complete)["nu"], 2)
        cycle = Graph(5, ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4)))
        self.assertEqual({key: invariants(cycle)[key] for key in ("alpha", "omega", "chi", "nu")},
                         {"alpha": 2, "omega": 2, "chi": 3, "nu": 2})

    def test_chromatic_and_matching_independent_oracles_n4(self):
        # Exhaustive graphs, with a separate direct color-assignment oracle.
        for mask in range(1 << 6):
            graph = from_mask(4, mask)
            edge_set = set(graph.edges)
            valid_color_counts = []
            for code in range(4 ** 4):
                colors = tuple((code // (4 ** vertex)) % 4 for vertex in range(4))
                if all(colors[u] != colors[v] for u, v in edge_set):
                    valid_color_counts.append(len(set(colors)))
            self.assertEqual(invariants(graph)["chi"], min(valid_color_counts))
            possible = [0]
            for subset in range(1 << len(graph.edges)):
                chosen = [edge for index, edge in enumerate(graph.edges) if subset & (1 << index)]
                vertices = [vertex for edge in chosen for vertex in edge]
                if len(vertices) == len(set(vertices)):
                    possible.append(len(chosen))
            self.assertEqual(invariants(graph)["nu"], max(possible))

    def test_reject_bad_graph(self):
        for value in ({"n": 5, "edges": [[0, 1], [0, 1]]},
                      {"n": 5, "edges": [[1, 0]]},
                      {"n": True, "edges": []},
                      {"n": 5, "edges": [[0, 5]]}):
            with self.assertRaises(ValueError):
                Graph.from_json(value)


if __name__ == "__main__":
    unittest.main()
