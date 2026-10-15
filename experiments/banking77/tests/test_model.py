from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from banking77.model import LabelTrie, encode_record, load_taxonomy, MAX_LENGTH, SYSTEM_PROMPT
from banking77.train import accumulation_windows


class Tokenizer:
    eos_token_id = 3
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return [10, 11, 12]
    def encode(self, text, **kwargs):
        return [ord(char) + 100 for char in text]


class ModelTests(unittest.TestCase):
    def test_only_prompt_and_target_supervised_with_eos(self):
        tokenizer = Tokenizer()
        row = {"prompt": "public", "target": "Refund_not_showing_up", "metadata": "SECRET", "label": "WRONG"}
        encoded = encode_record(tokenizer, row, [row["target"]])
        self.assertEqual(encoded["labels"][:3], [-100] * 3)
        self.assertEqual(encoded["labels"][3:], encoded["input_ids"][3:])
        self.assertEqual(encoded["labels"][-1], 3)
        self.assertEqual(tokenizer.messages, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "public"}])

    def test_unknown_label_and_overflow_rejected(self):
        with self.assertRaises(ValueError):
            encode_record(Tokenizer(), {"prompt": "a", "target": "unknown"}, ["known"])
        long = "x" * MAX_LENGTH
        with self.assertRaises(ValueError):
            encode_record(Tokenizer(), {"prompt": "a", "target": long}, [long])

    def test_trie_shared_prefix_complete_paths_and_invalid_suffix(self):
        tokenizer = Tokenizer()
        trie = LabelTrie(tokenizer, ["card", "card_arrival", "Refund_not_showing_up"])
        for label in ["card", "card_arrival", "Refund_not_showing_up"]:
            prefix = []
            for token in tokenizer.encode(label) + [3]:
                self.assertIn(token, trie.allowed(prefix))
                prefix.append(token)
        self.assertEqual(trie.allowed(tokenizer.encode("card")), [3, ord('_') + 100])
        with self.assertRaises(ValueError):
            trie.allowed([9999])

    def test_exact_publisher_taxonomy_not_normalized(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            labels = [f"class_{i}" for i in range(75)] + ["reverted_card_payment?", "Refund_not_showing_up"]
            (path / "taxonomy.json").write_text(json.dumps(labels))
            self.assertEqual(load_taxonomy(path), labels)
            labels[-1] = labels[0]
            (path / "taxonomy.json").write_text(json.dumps(labels))
            with self.assertRaises(ValueError):
                load_taxonomy(path)

    def test_one_epoch_has_no_omissions_repeats_or_partial_window_underweight(self):
        windows = list(accumulation_windows(list(range(10)), 4))
        self.assertEqual(list(map(len, windows)), [4, 4, 2])
        self.assertEqual([item for window in windows for item in window], list(range(10)))
        self.assertEqual(sum(1 / len(windows[-1]) for _ in windows[-1]), 1)
