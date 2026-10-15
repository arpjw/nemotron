"""CPU tests require only Python; GPU behavior remains explicitly unverified."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from finmini.model import (MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, encode_record, load_records,
                           prompt_tokens, sha256_file, validate_adapter, load_frozen_records)
from finmini.inference import generate_response
from finmini.train import build_parser


TARGET = '{"status":"ok","value":"12.00","unit":"USD","evidence":["L1"]}'


class Tokenizer:
    eos_token_id = 3
    pad_token_id = 3

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        return [10, 11, 12]

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [ord(c) + 100 for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(x - 100) for x in tokens)


class EncodingTests(unittest.TestCase):
    def test_response_mask_and_terminal_token(self):
        tokenizer = Tokenizer()
        row = {"id": "r", "prompt": "extract", "target": TARGET, "metadata": {"secret": "not in prompt"}}
        encoded = encode_record(tokenizer, row)
        self.assertEqual(encoded["labels"][:3], [-100] * 3)
        self.assertEqual(encoded["labels"][3:], encoded["input_ids"][3:])
        self.assertEqual(encoded["labels"][-1], 3)
        self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))
        self.assertEqual(tokenizer.messages, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "extract"}])

    def test_exact_boundary_and_overflow_rejection(self):
        row = {"prompt": "x", "target": TARGET}
        size = len(TARGET) + 4
        self.assertEqual(len(encode_record(Tokenizer(), row, size)["input_ids"]), size)
        with self.assertRaisesRegex(ValueError, "requires"):
            encode_record(Tokenizer(), row, size - 1)

    def test_bad_targets_and_lengths(self):
        for target in ("", "not json", "[]", "null", "1"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                encode_record(Tokenizer(), {"prompt": "x", "target": target})
        for limit in (0, 1025):
            with self.assertRaises(ValueError):
                encode_record(Tokenizer(), {"prompt": "x", "target": "{}"}, limit)
        with self.assertRaises(ValueError):
            prompt_tokens(Tokenizer(), " ")

    def test_split_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps({"id": "a", "split": "test"}) + "\n")
            with self.assertRaisesRegex(ValueError, "only train"):
                load_records(path, "train")
            path.write_text('{"id":"a"}\n{"id":"a"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_records(path)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_records(path)

    def test_frozen_manifest_alias_drift_and_sealed_gates(self):
        from finmini.data import write_dataset
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "frozen"
            write_dataset(directory, counts={"train": 6, "dev": 6, "test": 6, "robustness": 6})
            self.assertEqual(len(load_frozen_records(directory / "train.jsonl", training=True)), 6)
            with self.assertRaisesRegex(ValueError, "requires --allow-sealed"):
                load_frozen_records(directory / "test.jsonl")
            self.assertEqual(len(load_frozen_records(directory / "test.jsonl", allow_sealed=True)), 6)
            with self.assertRaisesRegex(ValueError, "requires the frozen train"):
                load_frozen_records(directory / "dev.jsonl", training=True)
            (directory / "alias.jsonl").write_bytes((directory / "test.jsonl").read_bytes())
            with self.assertRaisesRegex(ValueError, "manifest-listed"):
                load_frozen_records(directory / "alias.jsonl", allow_sealed=True)
            with (directory / "train.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "digest or size"):
                load_frozen_records(directory / "train.jsonl", training=True)

    def test_base_loader_uses_offline_pinned_restricted_weights(self):
        from finmini.model import load_base
        captured = {}
        result = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True))
        def from_pretrained(model_id, **kwargs):
            captured.update(model_id=model_id, **kwargs)
            return result
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True), bfloat16="BF16")
        transformers = types.SimpleNamespace(AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=from_pretrained), BitsAndBytesConfig=lambda **kw: kw)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            self.assertIs(load_base(training=True), result)
        self.assertEqual(captured["model_id"], MODEL_ID)
        self.assertEqual(captured["revision"], MODEL_REVISION)
        self.assertTrue(captured["local_files_only"])
        self.assertTrue(captured["weights_only"])
        self.assertFalse(captured["use_safetensors"])
        self.assertFalse(captured["trust_remote_code"])
        self.assertFalse(result.config.use_cache)

    def test_reject_nonpositive_steps(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--data", "a", "--output", "b", "--max-steps", "0"])

    def test_adapter_provenance_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "adapter_model.safetensors").write_bytes(b"test-weights")
            manifest = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "system_prompt": SYSTEM_PROMPT,
                        "status": "completed", "adapter_sha256": sha256_file(path / "adapter_model.safetensors")}
            (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": MODEL_ID, "peft_type": "LORA"}))
            manifest["adapter_config_sha256"] = sha256_file(path / "adapter_config.json")
            manifest["training_sha256"] = "current-data"
            (path / "run_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(validate_adapter(path, expected_training_sha256="current-data"), manifest)
            with self.assertRaisesRegex(ValueError, "training dataset"):
                validate_adapter(path, expected_training_sha256="old-data")
            original_config = (path / "adapter_config.json").read_bytes()
            (path / "adapter_config.json").write_bytes(original_config + b" ")
            with self.assertRaisesRegex(ValueError, "config checksum"):
                validate_adapter(path)
            (path / "adapter_config.json").write_bytes(original_config)

            (path / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_adapter(path)
            manifest["status"] = "failed"
            (path / "run_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "did not complete"):
                validate_adapter(path)


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        row, column = key
        return self.data[row][column]


class NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class GenerationTests(unittest.TestCase):
    def test_cli_subset_manifest_and_no_label_leak(self):
        from finmini.inference import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "dev.jsonl"
            rows = [{"id": str(i), "prompt": f"public-{i}", "target": "PRIVATE", "metadata": "PRIVATE"} for i in range(3)]
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            model = types.SimpleNamespace(eval=lambda: None)
            with patch("finmini.inference.load_frozen_records", return_value=rows), patch("finmini.inference.load_tokenizer", return_value=Tokenizer()), patch("finmini.inference.load_base", return_value=model), patch("finmini.inference.generate_response", return_value="{}") as generate:
                main(["--data", str(data), "--output", str(root / "run"), "--limit", "2"])
            self.assertEqual([call.args[2] for call in generate.call_args_list], ["public-0", "public-1"])
            manifest = json.loads((root / "run/inference_manifest.json").read_text())
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["dataset_rows"], 3)
            self.assertTrue(manifest["subset"])
            self.assertEqual(manifest["selected_ids"], ["0", "1"])
            self.assertEqual(manifest["status"], "completed")

    def test_termination_marks_inference_failed(self):
        from finmini.inference import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "dev.jsonl"
            rows = [{"id": "1", "prompt": "public"}]
            data.write_text(json.dumps(rows[0]) + "\n")
            model = types.SimpleNamespace(eval=lambda: None)
            with patch("finmini.inference.load_frozen_records", return_value=rows), patch("finmini.inference.load_tokenizer", return_value=Tokenizer()), patch("finmini.inference.load_base", return_value=model), patch("finmini.inference.generate_response", side_effect=SystemExit(143)):
                with self.assertRaises(SystemExit):
                    main(["--data", str(data), "--output", str(root / "run")])
            manifest = json.loads((root / "run/inference_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error_type"], "SystemExit")
            self.assertNotIn("predictions_sha256", manifest)

    def test_only_prompt_input_deterministic_and_continuation_only(self):
        class Model:
            device = "test-device"
            def generate(self, **kwargs):
                self.arguments = kwargs
                return FakeTensor([[10, 11, 12, ord('{') + 100, ord('}') + 100]])
        torch = types.SimpleNamespace(tensor=lambda x, **kw: FakeTensor(x), long="long",
                                      ones_like=lambda x: FakeTensor([[1] * len(x.data[0])]), inference_mode=NoGrad)
        model, tokenizer = Model(), Tokenizer()
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(generate_response(model, tokenizer, "prompt"), "{}")
        self.assertEqual(model.arguments["input_ids"].data, [[10, 11, 12]])
        self.assertIs(model.arguments["do_sample"], False)
        self.assertEqual(model.arguments["num_beams"], 1)
        self.assertEqual(model.arguments["max_new_tokens"], 192)
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                generate_response(model, tokenizer, "prompt", 1022)


if __name__ == "__main__":
    unittest.main()
