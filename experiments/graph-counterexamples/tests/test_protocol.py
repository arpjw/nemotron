import json
import tempfile
import unittest
from pathlib import Path

from graph_ce.baseline import search
from graph_ce.data import load_split, write_dataset
from graph_ce.evaluate import paired_interval, parse_graph, score
from graph_ce.graph import Graph
from graph_ce.model import encode_record
from graph_ce.select import make_selection
from graph_ce.task import Task


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


class ProtocolTests(unittest.TestCase):
    def test_frozen_splits_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "data"
            write_dataset(directory, counts={"train": 3, "dev": 2, "test": 2})
            train = load_split(directory, "train")
            dev = load_split(directory, "dev")
            self.assertEqual(len(train), 3)
            self.assertFalse({(r["task"]["lhs"], r["task"]["rhs"]) for r in train} &
                             {(r["task"]["lhs"], r["task"]["rhs"]) for r in dev})
            (directory / "dev.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                load_split(directory, "train")

    def test_exact_scoring_and_malformed_denominator(self):
        task = Task("dev-0000", "dev", 5, "chi", "omega", 1, 1, 0)
        cycle = Graph(5, ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4)))
        row = {"task": task.to_json(), "target": cycle.to_json(), "target_margin": 1}
        good = score([row], [{"id": task.id, "raw": json.dumps(cycle.to_json())}])
        bad = score([row], [{"id": task.id, "raw": "a graph probably exists"}])
        self.assertEqual((good["success_rate"], bad["success_rate"]), (1.0, 0.0))
        self.assertEqual(bad["valid_json_graphs"], 0)
        self.assertEqual(paired_interval(bad, good, draws=100)["candidate_minus_baseline"], 1.0)
        self.assertIsNone(parse_graph(json.dumps({"n": 5, "edges": [[1, 0]]}), 5))
        guess, margin = search(task, budget=100, seed=1)
        self.assertEqual(margin, task.margin(guess))

    def test_response_only_mask_and_overflow(self):
        task = Task("train-0000", "train", 5, "chi", "omega", 1, 1, 0)
        graph = Graph(5, ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4)))
        encoded = encode_record(FakeTokenizer(), {"task": task.to_json(), "target": graph.to_json()})
        self.assertEqual(encoded["labels"][:3], [-100] * 3)
        self.assertEqual(encoded["labels"][-1], 99)
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))

    def test_selection_requires_improvement_and_model_pair(self):
        report = {"split": "dev", "dataset_sha256": "a" * 64,
                  "candidate_inference": {"model_id": "nvidia/Nemotron-Mini-4B-Instruct",
                                           "model_revision": "791833e92ebddb0bc2c1007f6d2b6764f886a2ae",
                                           "adapter_sha256": "b" * 64},
                  "baseline_inference": {"adapter_sha256": None},
                  "candidate": {"success_rate": 0.5}, "baseline": {"success_rate": 0.3},
                  "paired": {"candidate_minus_baseline": 0.2}}
        self.assertEqual(make_selection(report, report_hash="c" * 64)["adapter_sha256"], "b" * 64)
        report["paired"]["candidate_minus_baseline"] = 0
        with self.assertRaises(ValueError):
            make_selection(report, report_hash="c" * 64)


if __name__ == "__main__":
    unittest.main()
