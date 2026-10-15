"""Synthetic-only official-test guards. Never read publisher test bytes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from banking77.model import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, MAX_NEW_TOKENS
from banking77.test_guard import validate_test_binding, reserve_test_inference, clean_test_rows


def protocol():
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "system_prompt": SYSTEM_PROMPT,
            "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "modes": ["taxonomy_trie", "raw"]}}


class TestGuards(unittest.TestCase):
    def test_selected_adapter_prompt_and_decoding_enforced(self):
        selected = {**protocol(), "adapter_sha256": "a" * 64}
        metadata = {**protocol(), "adapter_sha256": "a" * 64, "adapter_epoch_complete": True}
        validate_test_binding(selected, metadata, adapter=True)
        validate_test_binding(selected, {**protocol(), "adapter_sha256": None}, adapter=False)
        for change in ({"adapter_sha256": "b" * 64}, {"system_prompt": "new prompt"}, {"generation": {}}, {"adapter_epoch_complete": False}):
            with self.assertRaises(ValueError):
                validate_test_binding(selected, {**metadata, **change}, adapter=True)

    def test_reservation_prevents_second_run_but_allows_two_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "selection.json").write_text("{}")
            (directory / "test.jsonl").write_text("synthetic-only")
            reserve_test_inference(directory, directory / "base", adapter=False, metadata={"adapter_sha256": None})
            with self.assertRaises(FileExistsError):
                reserve_test_inference(directory, directory / "other", adapter=False, metadata={"adapter_sha256": None})
            reserve_test_inference(directory, directory / "adapter", adapter=True, metadata={"adapter_sha256": "a" * 64})

    def test_clean_audit_preserves_order_and_rejects_invalid_partition(self):
        rows = [{"id": "first"}, {"id": "second"}, {"id": "third"}]
        self.assertEqual(clean_test_rows(rows, {"clean_ids": ["third", "first"], "quarantined_ids": ["second"]}), [rows[0], rows[2]])
        for audit in ({"clean_ids": ["first"], "quarantined_ids": []}, {"clean_ids": ["first", "first"], "quarantined_ids": ["second", "third"]}):
            with self.assertRaises(ValueError):
                clean_test_rows(rows, audit)

    def test_official_inference_requires_acknowledgement_and_forbids_subsets(self):
        from banking77.inference import main
        with patch("banking77.sealed_test.load_test_records") as reader:
            with self.assertRaises(ValueError):
                main(["--data", "fake/test.jsonl", "--output", "unused"])
            with self.assertRaises(ValueError):
                main(["--data", "fake/test.jsonl", "--output", "unused", "--allow-sealed", "--frozen-data", "fake-frozen", "--limit", "1"])
            reader.assert_not_called()

    def test_official_adapter_uses_original_frozen_training_directory(self):
        from banking77.inference import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test = root / "official"; test.mkdir()
            frozen = root / "frozen"
            selected = {**protocol(), "adapter_sha256": "a" * 64}
            (test / "selection.json").write_text(json.dumps(selected))
            (test / "test.jsonl").write_text("synthetic-only")
            (test / "taxonomy.json").write_text(json.dumps([f"label_{i}" for i in range(77)]))
            adapter = {"adapter_sha256": "a" * 64, "epoch_complete": True}
            with patch("banking77.sealed_test.load_test_records", return_value=[{"id": "synthetic", "prompt": "public"}]), patch("banking77.test_guard.validate_frozen_training"), patch("banking77.inference.validate_adapter", return_value=adapter) as check, patch("banking77.inference.load_tokenizer"), patch("banking77.inference.LabelTrie"), patch("banking77.inference.prompt_tokens", return_value=[1]), patch("banking77.inference.load_base", side_effect=RuntimeError("stop before any GPU")):
                with self.assertRaisesRegex(RuntimeError, "stop before any GPU"):
                    main(["--data", str(test / "test.jsonl"), "--output", str(root / "result"), "--adapter", str(root / "adapter"), "--allow-sealed", "--frozen-data", str(frozen)])
            check.assert_called_once_with(root / "adapter", frozen)
            manifest = json.loads((root / "result/inference_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue((test / "inference-selected-adapter.receipt.json").exists())

    def test_official_report_keeps_all_rows_and_separate_clean_sensitivity(self):
        from banking77.evaluation import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"id": "one", "group_id": "g1", "label": "a"}, {"id": "two", "group_id": "g2", "label": "b"}]
            (root / "test.jsonl").write_text("synthetic-only")
            (root / "selection.json").write_text(json.dumps({**protocol(), "adapter_sha256": "a" * 64}))
            (root / "audit.json").write_text(json.dumps({"clean_ids": ["one"], "quarantined_ids": ["two"]}))
            candidate = {"one": {"constrained": "a", "raw": "a"}, "two": {"constrained": "a", "raw": "a"}}
            baseline = {"one": {"constrained": "b", "raw": "b"}, "two": {"constrained": "b", "raw": "b"}}
            common = {**protocol(), "versions": {"synthetic": "1"}, "source_sha256": {"model.py": "m", "inference.py": "i"}}
            meta = {**common, "adapter_sha256": "a" * 64, "adapter_epoch_complete": True}
            base_meta = {**common, "adapter_sha256": None}
            output = root / "report.json"
            args = ["evaluation", "--data", str(root / "test.jsonl"), "--predictions", "candidate.jsonl", "--baseline", "base.jsonl", "--output", str(output), "--allow-sealed"]
            with patch("sys.argv", args), patch("banking77.sealed_test.load_test_records", return_value=rows), patch("banking77.model.load_taxonomy", return_value=["a", "b"]), patch("banking77.evaluation.load_predictions", side_effect=[(candidate, meta), (baseline, base_meta)]):
                main()
            report = json.loads(output.read_text())
            self.assertEqual(report["candidate"]["n"], 2)
            self.assertEqual(report["clean_sensitivity"]["n"], 1)
            self.assertEqual(report["paired_macro_f1"]["delta"], 0)
            self.assertGreater(report["clean_sensitivity"]["paired_macro_f1"]["delta"], 0)
            self.assertTrue(report["official_all_rows"])
            self.assertNotIn("development_gate", report)


    def test_empty_clean_partition_is_valid(self):
        self.assertEqual(clean_test_rows([{"id": "one"}], {"clean_ids": [], "quarantined_ids": ["one"]}), [])

    def test_receipt_integrity_and_role(self):
        from banking77.model import sha256_file
        from banking77.test_guard import validate_test_receipt
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "selection.json").write_text("{}")
            (root / "test.jsonl").write_text("synthetic-only")
            receipt = reserve_test_inference(root, root / "base", adapter=False, metadata={"adapter_sha256": None})
            metadata = {"adapter_sha256": None, "test_inference_receipt": receipt.name,
                        "test_inference_receipt_sha256": sha256_file(receipt),
                        "selection_sha256": sha256_file(root / "selection.json"),
                        "dataset_sha256": sha256_file(root / "test.jsonl")}
            validate_test_receipt(root, metadata)
            with self.assertRaises(ValueError):
                validate_test_receipt(root, {**metadata, "adapter_sha256": "a" * 64})
            receipt.write_text("{}")
            with self.assertRaises(ValueError):
                validate_test_receipt(root, metadata)
