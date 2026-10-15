from __future__ import annotations

import copy
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from finmini.data import (
    DEFAULT_COUNTS, LAYOUTS, build_prompt, generate_dataset, money,
    oracle, validate_records, verify_dataset, write_dataset,
)


class DataTests(unittest.TestCase):
    def test_decimal_and_negative_rounding(self):
        self.assertEqual(money(Decimal("1.005")), "1.01")
        self.assertEqual(money(Decimal("-1.005")), "-1.01")
        self.assertEqual(money(Decimal("-0.001")), "0.00")

    def test_oracle_independent_examples(self):
        facts = [
            {"metric": "revenue", "period": "2022", "usd": "3.00", "line_id": "L2"},
            {"metric": "revenue", "period": "2023", "usd": "4.00", "line_id": "L1"},
            {"metric": "operating_income", "period": "2023", "usd": "999.00", "line_id": "L3"},
        ]
        self.assertEqual(oracle(facts, "growth", "2023", "2022"), {
            "status": "ok", "value": "33.33", "unit": "percent", "evidence": ["L1", "L2"]})
        self.assertEqual(oracle(facts, "difference", "2023", "2022")["value"], "1.00")
        self.assertEqual(oracle(facts, "extract", "2023", "2022")["evidence"], ["L1"])
        self.assertEqual(oracle(facts[1:], "growth", "2023", "2022")["status"], "unanswerable")
        facts[0]["usd"] = "0.00"
        self.assertEqual(oracle(facts, "growth", "2023", "2022")["status"], "unanswerable")
        facts[0]["usd"] = "-1.00"
        self.assertEqual(oracle(facts, "growth", "2023", "2022")["status"], "unanswerable")
        self.assertEqual(oracle(facts, "difference", "2023", "2022")["value"], "5.00")
        facts.append({"metric": "revenue", "period": "2023", "usd": "7.00", "line_id": "L4"})
        self.assertEqual(oracle(facts, "extract", "2023", "2022")["status"], "unanswerable")

    def test_generation_reproducible_and_disjoint(self):
        counts = dict.fromkeys(DEFAULT_COUNTS, 90)
        first = generate_dataset(counts=counts)
        self.assertEqual(first, generate_dataset(counts=counts))
        self.assertNotEqual(first, generate_dataset(seed=1730, counts=counts))
        for split, rows in first.items():
            self.assertEqual(sum(r["response"]["status"] == "unanswerable" for r in rows), 18)
            self.assertEqual({r["layout"] for r in rows}, set(LAYOUTS[split]))
            self.assertEqual({r["metadata"]["mode"] for r in rows}, {"answerable", "missing", "conflict", "zero"})
            for layout in LAYOUTS[split]:
                self.assertEqual({r["task"] for r in rows if r["layout"] == layout}, {"extract", "difference", "growth"})
        validate_records(first)

    def test_public_entities_do_not_encode_answerability_index(self):
        import re
        dataset = generate_dataset(counts=dict.fromkeys(DEFAULT_COUNTS, 90))
        names = set()
        for split, rows in dataset.items():
            for index, row in enumerate(rows):
                entity = row["input"]["entity"]
                self.assertRegex(entity, r"^Fictional_[0-9a-f]{20}$")
                self.assertNotIn(split, entity)
                self.assertNotIn(entity, names)
                names.add(entity)
                # The former public-index rule had 100% status accuracy. The
                # opaque hash suffix no longer exposes that construction index.
                self.assertNotEqual(entity, f"Fictional_{split}_{index:05d}")
            correct = sum((int(row["input"]["entity"][-4:], 16) % 5 == 0) ==
                          (row["response"]["status"] == "unanswerable") for row in rows)
            self.assertLess(correct / len(rows), 0.9)
        revenues = []
        for row in dataset["train"]:
            facts = row["metadata"]["facts"]
            for revenue in [f for f in facts if f["metric"] == "revenue"]:
                income = next((f for f in facts if f["metric"] == "operating_income" and f["period"] == revenue["period"]), None)
                if income:
                    revenues.append(Decimal(income["usd"]) * 5 == Decimal(revenue["usd"]))
        self.assertFalse(any(revenues))

    def test_source_length_is_not_an_answerability_label(self):
        dataset = generate_dataset(counts=dict.fromkeys(DEFAULT_COUNTS, 90))
        for rows in dataset.values():
            self.assertEqual({len(row["metadata"]["facts"]) for row in rows}, {7})
            for row in rows:
                expected_lines = 8 if row["layout"] == "instruction_distractor" else 7
                self.assertEqual(len(row["input"]["lines"]), expected_lines)
        answerable_extractions = [r for r in dataset["train"] if r["task"] == "extract" and r["response"]["status"] == "ok"]
        counts = {sum(f["metric"] == "revenue" for f in r["metadata"]["facts"]) for r in answerable_extractions}
        self.assertEqual(counts, {1, 3})

    def test_full_default_counts(self):
        dataset = generate_dataset()
        self.assertEqual({k: len(v) for k, v in dataset.items()}, DEFAULT_COUNTS)
        self.assertEqual({k: sum(r["response"]["status"] == "unanswerable" for r in rows) for k, rows in dataset.items()},
                         {k: v // 5 for k, v in DEFAULT_COUNTS.items()})

    def test_prompt_allowlist(self):
        dataset = generate_dataset(counts=dict.fromkeys(DEFAULT_COUNTS, 3))
        row = dataset["train"][0]
        original = row["prompt"]
        row["metadata"]["secret"] = "GOLD_SENTINEL"
        row["response"] = {"secret": "GOLD_SENTINEL"}
        self.assertEqual(build_prompt(row["input"]), original)
        self.assertNotIn("GOLD_SENTINEL", build_prompt(row["input"]))
        with self.assertRaises(ValueError):
            build_prompt({**row["input"], "target": "GOLD_SENTINEL"})

    def test_validator_detects_leaks_and_bad_labels(self):
        rows = generate_dataset(counts=dict.fromkeys(DEFAULT_COUNTS, 3))
        changed = copy.deepcopy(rows)
        changed["dev"][0]["group_id"] = changed["train"][0]["group_id"]
        with self.assertRaises(ValueError):
            validate_records(changed)
        changed = copy.deepcopy(rows)
        changed["train"][1]["response"]["value"] = "0.00"
        with self.assertRaises(ValueError):
            validate_records(changed)
        changed = copy.deepcopy(rows)
        changed["test"][0]["layout"] = "sentences"
        with self.assertRaises(ValueError):
            validate_records(changed)
        changed = copy.deepcopy(rows)
        changed["train"][1]["prompt"] += " gold answer"
        with self.assertRaises(ValueError):
            validate_records(changed)

    def test_manifest_freeze_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "frozen"
            manifest = write_dataset(folder, counts=dict.fromkeys(DEFAULT_COUNTS, 6))
            self.assertEqual(verify_dataset(folder), manifest)
            with self.assertRaises(FileExistsError):
                write_dataset(folder)
            with (folder / "train.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "digest"):
                verify_dataset(folder)

    def test_invalid_sizes(self):
        for counts in ({"train": 1}, dict.fromkeys(DEFAULT_COUNTS, 0), dict.fromkeys(DEFAULT_COUNTS, True)):
            with self.assertRaises(ValueError):
                generate_dataset(counts=counts)


if __name__ == "__main__":
    unittest.main()
