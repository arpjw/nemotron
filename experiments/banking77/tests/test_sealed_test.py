import hashlib
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from banking77.data import canonical, sha256
from banking77.sealed_test import export_test, load_test_records
from banking77.sealed_test import transform_test, validate_selection


class SealedTestTests(unittest.TestCase):
    def transform(self, raw, references=()):
        sha = hashlib.sha256(raw).hexdigest()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        return transform_test(raw, ['A', 'B?'], list(references), sha, blob, 4)

    def test_overlap_conflict_quarantined_but_official_rows_preserved(self):
        rows, audit = self.transform(b'text,category\nHELLO!,A\nunique,B?\ncollision,A\nCollision!,B?\n', [{'text': 'hello', 'label': 'A'}])
        self.assertEqual(len(rows), 4)
        self.assertEqual(audit['clean_ids'], ['official-test-0001'])
        self.assertEqual(len(audit['quarantined_ids']), 3)
        self.assertEqual(len(audit['conflicting_groups']), 1)
        self.assertEqual(rows[1]['target'], 'B?')
        self.assertNotIn(rows[1]['id'], rows[1]['prompt'])

    def test_integrity_fail_before_decode(self):
        with self.assertRaisesRegex(ValueError, 'integrity'):
            transform_test(b'\xff', ['A'], [], 'wrong', 'wrong', 1)

    def test_unknown_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid test row'):
            self.transform(b'text,category\na,A\nb,B?\nc,A\nd,C\n')

    def fixture(self):
        candidate = dict(status='completed', adapter_epoch_complete=True, adapter_sha256='a'*64, system_prompt='sys', generation={'decoder': 'trie'}, model_id='m', model_revision='r')
        baseline = dict(candidate, adapter_sha256=None)
        gate = dict(data_manifest_sha256='manifest', macro_f1_at_least=.75, gain_over_base_at_least=.05, paired_macro_f1_lower_95_strictly_above=0)
        report = dict(sealed=False, candidate_inference=candidate, baseline_inference=baseline, development_gate={'passed': True, 'definition': gate}, candidate={'modes': {'constrained': {'macro_f1': .8}}}, paired_macro_f1={'delta': .1, 'lower_95': .01})
        selection = {'development_report_sha256': 'report', 'data_manifest_sha256': 'manifest', **{k: candidate[k] for k in ('adapter_sha256','system_prompt','generation','model_id','model_revision')}}
        return report, selection, gate

    def test_passed_gate_and_exact_selection_required(self):
        report, selection, gate = self.fixture()
        validate_selection(report, selection, 'report', 'manifest', gate)
        selection['adapter_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'Selection'):
            validate_selection(report, selection, 'report', 'manifest', gate)

    def test_pass_boolean_cannot_override_failed_metric(self):
        report, selection, gate = self.fixture()
        report['paired_macro_f1']['lower_95'] = 0
        with self.assertRaisesRegex(ValueError, 'passed development'):
            validate_selection(report, selection, 'report', 'manifest', gate)

    def test_atomic_export_reservation_and_reload_with_synthetic_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / 'frozen'
            frozen.mkdir()
            taxonomy = ['A', 'B?'] + [f'label{i}' for i in range(75)]
            (frozen / 'taxonomy.json').write_text(canonical(taxonomy) + '\n')
            (frozen / 'manifest.json').write_text('{}')
            for split in ('train', 'dev'):
                (frozen / f'{split}.jsonl').write_text(canonical({'id': split, 'text': split, 'label': 'A'}) + '\n')
            raw = root / 'test.csv'
            raw.write_text('text,category\n' + 'new query,A\n' * 3080)
            payload = raw.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
            report, selection, gate = self.fixture()
            manifest_sha = sha256(frozen / 'manifest.json')
            gate['data_manifest_sha256'] = manifest_sha
            gate_path = root / 'gate.json'
            gate_path.write_text(json.dumps(gate))
            report['development_gate']['sha256'] = sha256(gate_path)
            report['data_sha256'] = sha256(frozen / 'dev.jsonl')
            for metadata in (report['candidate_inference'], report['baseline_inference']):
                metadata.update(subset=False, selected_ids=['dev'], dataset_sha256=report['data_sha256'], taxonomy_sha256=sha256(frozen / 'taxonomy.json'))
            report_path = root / 'report.json'
            report_path.write_text(json.dumps(report))
            selection.update(development_report_sha256=sha256(report_path), data_manifest_sha256=manifest_sha)
            selection_path = root / 'selection.json'
            selection_path.write_text(json.dumps(selection))
            with patch('banking77.sealed_test.verify_dataset'), patch.dict('banking77.sealed_test.EXPECTED', {'test.csv': sha256(raw)}), patch('banking77.sealed_test.TEST_BLOB', blob):
                output = root / 'export'
                original = {path: path.read_bytes() for path in (report_path, selection_path, gate_path)}
                def change_sources_after_validation(*args):
                    for source in original:
                        source.write_text('{}')
                    return transform_test(*args)
                with patch('banking77.sealed_test.transform_test', side_effect=change_sources_after_validation):
                    export_test(frozen, raw, report_path, selection_path, output, gate_path)
                for source, name in ((report_path, 'development_report.json'), (selection_path, 'selection.json'), (gate_path, 'gate.json')):
                    self.assertEqual((output / name).read_bytes(), original[source])
                    source.write_bytes(original[source])
                self.assertEqual(len(load_test_records(output / 'test.jsonl')), 3080)
                with self.assertRaises(FileExistsError):
                    export_test(frozen, raw, report_path, selection_path, root / 'second', gate_path)
                manifest_path = output / 'manifest.json'
                manifest_bytes = manifest_path.read_bytes()
                manifest = json.loads(manifest_bytes)
                manifest['export_code_sha256'] = '0' * 64
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'code changed'):
                    load_test_records(output / 'test.jsonl')
                manifest_path.write_bytes(manifest_bytes)
                (output / 'test.jsonl').write_text('modified')
                with self.assertRaisesRegex(ValueError, 'modified'):
                    load_test_records(output / 'test.jsonl')


if __name__ == '__main__':
    unittest.main()
