from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from finmini.scoring import evaluate_predictions, paired_accuracy_interval, parse_response, score_response


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"status": "ok", "value": "10.00", "unit": "USD", "evidence": ["L1", "L2"]}

    def test_whitespace_key_and_evidence_order(self):
        text = '\n {"evidence":["L2", "L1"],"unit":"USD","status":"ok","value":"10.00"} '
        self.assertTrue(score_response(text, self.expected)["exact"])

    def test_invalid_outputs(self):
        valid = json.dumps(self.expected)
        invalid = [
            "answer: " + valid, valid + " garbage", "```json\n" + valid + "\n```",
            '{"status":"ok","status":"ok","value":"10.00","unit":"USD","evidence":["L1"]}',
            '{"status":"ok","value":NaN,"unit":"USD","evidence":["L1"]}',
            '[]', 'null', valid[:-1], "[" * 1500,
        ]
        for replacement in (True, 10, float("inf"), "10", "10.0", "1e1", "+10.00", "010.00", None, {}):
            invalid.append(json.dumps({**self.expected, "value": replacement}))
        for replacement in (None, {}, [], [True], ["L0"], ["L1", "L1"], ["L-1"]):
            invalid.append(json.dumps({**self.expected, "evidence": replacement}))
        invalid.append(json.dumps({**self.expected, "extra": "no"}))
        invalid.append(json.dumps({**self.expected, "unit": {"USD": 1}}))
        invalid.append(json.dumps({**self.expected, "status": ["ok"]}))
        for text in invalid:
            with self.subTest(text=text[:120]):
                self.assertFalse(score_response(text, self.expected)["exact"])
                self.assertIsNone(parse_response(text))

    def test_wrong_but_well_formed(self):
        for key, value in (("value", "11.00"), ("unit", "percent"), ("evidence", ["L99"])):
            flags = score_response(json.dumps({**self.expected, key: value}), self.expected)
            self.assertTrue(flags["schema"])
            self.assertFalse(flags["exact"])
            self.assertFalse(flags[key])

    def test_abstention(self):
        response = {"status": "unanswerable", "value": None, "unit": None, "evidence": []}
        self.assertTrue(score_response(json.dumps(response), response)["exact"])
        self.assertFalse(score_response(json.dumps(response), self.expected)["exact"])
        self.assertIsNone(parse_response(json.dumps({**response, "evidence": ["L1"]})))
        with self.assertRaises(ValueError):
            score_response("{}", {})

    def test_missing_outputs_are_not_dropped(self):
        rows = [
            {"id": "a", "group_id": "a", "response": self.expected, "task": "extract", "layout": "one"},
            {"id": "b", "group_id": "b", "response": self.expected, "task": "extract", "layout": "one"},
        ]
        report = evaluate_predictions(rows, {"a": json.dumps(self.expected)})
        self.assertEqual(report["n"], 2)
        self.assertEqual(report["rates"]["exact"], 0.5)
        self.assertEqual(report["abstention"]["precision"], None)
        with self.assertRaises(ValueError):
            evaluate_predictions(rows, {"alien": "{}"})
        with self.assertRaises(ValueError):
            evaluate_predictions([rows[0], rows[0]], {})

    def test_cli_scoring_and_sealed_guard(self):
        from finmini.data import DEFAULT_COUNTS, write_dataset
        from finmini.scoring import main
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frozen = base / "frozen"
            write_dataset(frozen, counts=dict.fromkeys(DEFAULT_COUNTS, 3))
            rows = [json.loads(x) for x in (frozen / "dev.jsonl").read_text().splitlines()]
            predictions = base / "predictions.jsonl"
            predictions.write_text("".join(json.dumps({"id": row["id"], "prediction": row["target"]}) + "\n" for row in rows))
            output = base / "score.json"
            argv = ["score", "--data", str(frozen / "dev.jsonl"), "--predictions", str(predictions), "--output", str(output)]
            with patch("sys.argv", argv), patch("builtins.print"):
                main()
            self.assertEqual(json.loads(output.read_text())["rates"]["exact"], 1)
            with patch("sys.argv", argv), self.assertRaises(SystemExit):
                main()
            argv[2] = str(frozen / "test.jsonl")
            argv[-1] = str(base / "test-score.json")
            with patch("sys.argv", argv), self.assertRaises(SystemExit):
                main()
            copied = frozen / "dev-copy.jsonl"
            copied.write_bytes((frozen / "test.jsonl").read_bytes())
            argv[2] = str(copied)
            with patch("sys.argv", argv), self.assertRaises(SystemExit):
                main()

    def test_official_comparison_provenance(self):
        from finmini.data import DEFAULT_COUNTS, write_dataset
        from finmini.model import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, sha256_file
        from finmini.scoring import main, validate_comparison_provenance
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frozen = base / "frozen"
            write_dataset(frozen, counts=dict.fromkeys(DEFAULT_COUNTS, 3))
            data = frozen / "dev.jsonl"
            rows = [json.loads(x) for x in data.read_text().splitlines()]
            ids = [row["id"] for row in rows]
            for name, adapter in (("baseline", None), ("candidate", "a" * 64)):
                directory = base / name
                directory.mkdir()
                predictions = directory / "predictions.jsonl"
                predictions.write_text("".join(json.dumps({"id": row["id"], "prediction": row["target"]}) + "\n" for row in rows))
                metadata = {
                    "status": "completed", "dataset_sha256": sha256_file(data),
                    "predictions_sha256": sha256_file(predictions),
                    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                    "system_prompt": SYSTEM_PROMPT, "quantization": "NF4-double-BF16",
                    "rows": len(ids), "dataset_rows": len(ids), "subset": False,
                    "selected_ids": ids, "adapter_sha256": adapter,
                    "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 192},
                }
                (directory / "inference_manifest.json").write_text(json.dumps(metadata))
            candidate = base / "candidate" / "predictions.jsonl"
            baseline = base / "baseline" / "predictions.jsonl"
            output = base / "comparison.json"
            argv = ["score", "--data", str(data), "--predictions", str(candidate), "--baseline", str(baseline), "--output", str(output)]
            with patch("sys.argv", argv), patch("builtins.print"):
                main()
            self.assertEqual(json.loads(output.read_text())["comparison"]["delta"], 0)
            manifest_file = candidate.parent / "inference_manifest.json"
            original = json.loads(manifest_file.read_text())
            for key, value in (("status", "failed"), ("dataset_sha256", "wrong"), ("model_revision", "wrong"), ("system_prompt", "wrong"), ("predictions_sha256", "wrong"), ("subset", True)):
                manifest_file.write_text(json.dumps({**original, key: value}))
                with self.assertRaises(ValueError):
                    validate_comparison_provenance(candidate, data, ids)
            manifest_file.write_text(json.dumps({**original, "generation": {**original["generation"], "max_new_tokens": 100}}))
            argv[-1] = str(base / "mismatched.json")
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "decoding"):
                main()

    def test_bootstrap_group_and_identity_checks(self):
        a = [{"id": str(i), "group_id": str(i // 2), "exact": False} for i in range(6)]
        b = [{**row, "exact": True} for row in a]
        stats = paired_accuracy_interval(a, b, samples=100)
        self.assertEqual(stats["delta"], 1)
        self.assertEqual(stats["lower_95"], 1)
        self.assertEqual(stats["upper_95"], 1)
        self.assertEqual(stats["groups"], 3)
        self.assertEqual(stats, paired_accuracy_interval(a, b, samples=100))
        with self.assertRaises(ValueError):
            paired_accuracy_interval(a, b[:-1], samples=100)


if __name__ == "__main__":
    unittest.main()
