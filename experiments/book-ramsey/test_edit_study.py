import json
import os
import tempfile
import unittest
from pathlib import Path

from book import verify
from edit_baseline import rollout
from edit_evaluate import parse_action, score_predictions
from edit_model import encode_record
from edit_study import (ANCHORS, BLUE_BOOK, N, PAIRS, RED_BOOK, SOURCE_SHA256,
                        edge_list, flip, generate, load_anchors, load_split, score)


class EditStudyTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(os.environ.get("BOOK_GRAPH6_SOURCE", Path(__file__).with_name("sr261034.g6")))
        if not self.source.exists():
            self.skipTest("external McKay graph6 file is absent")

    def test_checker_and_flip(self):
        rows = load_anchors(self.source)[0]
        self.assertEqual(score(rows), (0, 0, 0))
        self.assertTrue(verify(N, edge_list(rows), RED_BOOK, BLUE_BOOK)["valid"])
        u, v = PAIRS[0]
        flip(rows, u, v)
        exact = verify(N, edge_list(rows), RED_BOOK, BLUE_BOOK)
        self.assertEqual(score(rows)[1:], (exact["red_violations"], exact["blue_violations"]))
        flip(rows, u, v)
        self.assertEqual(score(rows)[0], 0)

    def test_manifest_and_split(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data"
            manifest = generate(self.source, data, counts={"train": 3, "dev": 2, "test": 2})
            self.assertEqual(manifest["source_sha256"], SOURCE_SHA256)
            for split in ANCHORS:
                rows = load_split(data, split)
                self.assertTrue(all(row["anchor_id"] in ANCHORS[split] for row in rows))
            path = data / "test.jsonl"
            path.write_text(path.read_text() + json.dumps({}) + "\n")
            with self.assertRaises(ValueError):
                load_split(data, "test")

    def test_rollout_budget_and_exact_success(self):
        rows = load_anchors(self.source)[0]
        flip(rows, *PAIRS[0])
        result = rollout(rows, method="greedy", proposals=325, seed=1)
        self.assertLessEqual(result["evaluated"], 325)
        self.assertEqual(result["success"], result["best_loss"] == 0)

    def test_response_mask_and_scoring(self):
        class FakeTokenizer:
            eos_token_id = 99

            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]

            def encode(self, text, **kwargs):
                return [ord(char) for char in text]

        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data"
            generate(self.source, data, counts={"train": 1, "dev": 1, "test": 1})
            row = load_split(data, "train")[0]
            encoded = encode_record(FakeTokenizer(), row)
            self.assertEqual(encoded["labels"][:3], [-100] * 3)
            self.assertEqual(encoded["labels"][-1], 99)
            self.assertIsNone(parse_action('{"u":0,"v":0}'))
            self.assertIsNone(parse_action('{"u":true,"v":2}'))
            u, v = row["teacher_action"]
            result = score_predictions([row], [{"id": row["id"], "raw": json.dumps({"u": u, "v": v})}])
            self.assertEqual(result["records"][0]["next_loss"], row["teacher_next_loss"])


if __name__ == "__main__":
    unittest.main()
