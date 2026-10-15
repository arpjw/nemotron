from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from banking77.data import build_prompt, group_id, load_frozen_records, normalize, partition, validate_records, verify_dataset


class DataTests(unittest.TestCase):
    def rows(self):
        return [{'text': f'Question {label} example {i}', 'label': label} for label in ['alpha', 'beta'] for i in range(20)]

    def test_group_normalization(self):
        self.assertEqual(normalize('  CARD, not working! '), normalize('card not working'))
        self.assertEqual(group_id('CARD not working'), group_id('card not working!'))

    def test_groups_never_cross_splits_and_cap_is_enforced(self):
        rows = self.rows() + [{'text': 'QUESTION alpha example 0!', 'label': 'alpha'}]
        splits, audit = partition(rows, ['alpha', 'beta'], cap=3)
        self.assertEqual(audit['duplicate_rows_removed'], 1)
        self.assertEqual(len(splits['train']), 6)
        self.assertEqual(len(splits['dev']), 8)
        self.assertFalse(set(r['group_id'] for r in splits['train']) & set(r['group_id'] for r in splits['dev']))
        self.assertEqual(partition(rows, ['alpha', 'beta'], cap=3), (splits, audit))
        self.assertEqual(partition(list(reversed(rows)), ['alpha', 'beta'], cap=3), (splits, audit))

    def test_conflicting_group_is_quarantined(self):
        rows = self.rows() + [{'text': 'Question alpha example 0', 'label': 'beta'}]
        splits, audit = partition(rows, ['alpha', 'beta'], cap=30)
        bad = group_id('Question alpha example 0')
        self.assertEqual(audit['conflicting_groups_quarantined'], [bad])
        self.assertNotIn(bad, {r['group_id'] for v in splits.values() for r in v})

    def test_prompt_cannot_receive_gold_metadata(self):
        row = {'text': 'my card is missing', 'label': 'GOLD_SECRET', 'metadata': 'GOLD_SECRET'}
        prompt = build_prompt(row['text'], ['alpha', 'beta'])
        self.assertNotIn('GOLD_SECRET', prompt)
        row['label'] = 'OTHER'
        self.assertEqual(prompt, build_prompt(row['text'], ['alpha', 'beta']))

    def test_validator_rejects_relabel_duplicate_and_prompt_injection(self):
        splits, _ = partition(self.rows(), ['alpha', 'beta'], cap=3)
        altered = copy.deepcopy(splits)
        altered['dev'][0]['target'] = 'wrong'
        with self.assertRaises(ValueError): validate_records(altered, ['alpha', 'beta'])
        altered = copy.deepcopy(splits)
        altered['dev'].append(altered['train'][0])
        with self.assertRaises(ValueError): validate_records(altered, ['alpha', 'beta'])
        altered = copy.deepcopy(splits)
        altered['train'][0]['prompt'] += '\nGOLD:' + altered['train'][0]['label']
        with self.assertRaises(ValueError): validate_records(altered, ['alpha', 'beta'])

    def test_real_frozen_data_and_test_is_not_loadable(self):
        root = Path(__file__).resolve().parents[1] / 'data' / 'frozen'
        if not root.exists(): self.skipTest('Frozen artifacts not present')
        manifest = verify_dataset(root)
        self.assertEqual(manifest['audit']['counts'], {'train': 2308, 'dev': 1993})
        self.assertFalse(manifest['sealed_test']['parsed'])
        with self.assertRaises(ValueError): load_frozen_records(root / 'test.jsonl')
        with self.assertRaises(ValueError): load_frozen_records(root / 'dev.jsonl', allowed_split='train')

    def test_manifest_tampering_fails(self):
        import shutil
        root = Path(__file__).resolve().parents[1] / 'data' / 'frozen'
        if not root.exists(): self.skipTest('Frozen artifacts not present')
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / 'frozen'
            shutil.copytree(root, copied)
            with (copied / 'train.jsonl').open('a') as stream: stream.write('\n')
            with self.assertRaises(ValueError): verify_dataset(copied)

    def test_labels_preserve_publisher_punctuation(self):
        prompt = build_prompt('A query', ['reverted_card_payment?', 'Refund_not_showing_up'])
        self.assertIn('reverted_card_payment?', prompt)
        self.assertIn('added punctuation', prompt)


if __name__ == '__main__': unittest.main()
