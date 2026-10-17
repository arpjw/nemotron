import json
import os
import tempfile
import unittest
from pathlib import Path

from book import verify
from transfer_v2 import (ANCHORS, BLUE_BOOK, N, RED_BOOK, edge_list, flip,
                         generate, load_anchors, load_split, score)
from transfer_v2_audit import independent_loss
from transfer_v2_evaluate import (controls, greedy_repair, paired_anchor_interval,
                                  parse_action, score_actions)


class TransferV2Tests(unittest.TestCase):
    def setUp(self):
        self.source = Path(os.environ.get("BOOK_TRANSFER_SOURCE", Path(__file__).with_name("sr251256.g6")))
        if not self.source.exists():
            self.skipTest("external graph6 source is absent")

    def test_anchors_and_exact_score(self):
        anchors = load_anchors(self.source)
        self.assertEqual(len(anchors), 15)
        for rows in anchors:
            self.assertEqual(score(rows), (0, 0, 0))
            self.assertTrue(verify(N, edge_list(rows), RED_BOOK, BLUE_BOOK)["valid"])
            self.assertEqual(independent_loss({tuple(edge) for edge in edge_list(rows)}), 0)
        rows = anchors[0].copy()
        flip(rows, 0, 1)
        checked = verify(N, edge_list(rows), RED_BOOK, BLUE_BOOK)
        self.assertEqual(score(rows)[1:], (checked["red_violations"], checked["blue_violations"]))
        self.assertEqual(independent_loss({tuple(edge) for edge in edge_list(rows)}), score(rows)[0])

    def test_frozen_split_and_controls(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data"
            generate(self.source, data, counts={"dev": 2, "test": 2})
            dev = load_split(data, "dev")
            test = load_split(data, "test")
            self.assertTrue(all(row["anchor_id"] in ANCHORS["dev"] for row in dev))
            self.assertTrue(all(row["anchor_id"] in ANCHORS["test"] for row in test))
            self.assertEqual(controls(dev)["exact_teacher"]["valid_actions"], 2)
            self.assertLessEqual(greedy_repair(dev)["mean_calls"], 1500)
            teacher = score_actions(dev, [tuple(row["teacher_action"]) for row in dev])
            no_action = score_actions(dev, [None, None])
            interval = paired_anchor_interval(dev, teacher, no_action)
            self.assertEqual(len(interval), 2)
            self.assertLessEqual(interval[0], interval[1])
            self.assertIsNone(parse_action('{"u":true,"v":3}'))
            self.assertIsNone(parse_action('{"u":0,"v":25}'))
            self.assertEqual(score_actions(dev, [None, None])["mean_loss_reduction"], 0)
            path = data / "test.jsonl"
            path.write_text(path.read_text() + json.dumps({}) + "\n")
            with self.assertRaises(ValueError):
                load_split(data, "test")


if __name__ == "__main__":
    unittest.main()
