"""One-time official-test release after a passed development gate and selection."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from .data import EXPECTED, build_prompt, canonical, group_id, normalize, sha256, verify_dataset

TEST_BLOB = '799687a8367359432985b8b13d85a2baf73f92dd'


def validate_selection(report: dict, selection: dict, report_sha: str, manifest_sha: str, gate: dict) -> None:
    from .evaluation import passes_development_gate
    candidate = report['candidate_inference']
    baseline = report['baseline_inference']
    if (report.get('sealed') is not False or report['development_gate'].get('passed') is not True
            or report['development_gate']['definition'] != gate
            or gate['data_manifest_sha256'] != manifest_sha
            or not passes_development_gate(report['candidate']['modes']['constrained']['macro_f1'], report['paired_macro_f1'], gate)):
        raise ValueError('A passed development comparison is required')
    if candidate.get('status') != 'completed' or candidate.get('adapter_epoch_complete') is not True or baseline.get('status') != 'completed' or baseline.get('adapter_sha256') is not None:
        raise ValueError('Completed adapter and untouched baseline required')
    digest = candidate.get('adapter_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid selected adapter hash')
    expected = {'development_report_sha256': report_sha, 'data_manifest_sha256': manifest_sha,
                **{k: candidate[k] for k in ('adapter_sha256', 'system_prompt', 'generation', 'model_id', 'model_revision')}}
    if selection != expected:
        raise ValueError('Selection receipt does not bind the selected adapter, prompt and decoder')
    for key in ('system_prompt', 'generation', 'model_id', 'model_revision'):
        if baseline[key] != candidate[key]:
            raise ValueError('Development comparators differ')


def transform_test(raw: bytes, taxonomy: list[str], references: list[dict], expected_sha: str, expected_blob: str, expected_rows: int) -> tuple[list[dict], dict]:
    """Pure transform for synthetic tests; production caller supplies pinned constants."""
    if hashlib.sha256(raw).hexdigest() != expected_sha or hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != expected_blob:
        raise ValueError('Official raw test integrity failed')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))
    if reader.fieldnames != ['text', 'category']:
        raise ValueError('Unexpected CSV schema')
    source = list(reader)
    if len(source) != expected_rows:
        raise ValueError('Official test count mismatch')
    known: dict[str, set[str]] = {}
    for row in references:
        known.setdefault(group_id(row['text']), set()).add(row['label'])
    test_labels: dict[str, set[str]] = {}
    for row in source:
        if set(row) != {'text', 'category'} or row['category'] not in taxonomy or not normalize(row['text']):
            raise ValueError('Invalid test row')
        test_labels.setdefault(group_id(row['text']), set()).add(row['category'])
    overlaps = set(known) & set(test_labels)
    conflicts = {g for g, labels in test_labels.items() if len(labels | known.get(g, set())) > 1}
    records = [{'id': f'official-test-{i:04d}', 'group_id': group_id(r['text']), 'split': 'test', 'text': r['text'],
                'label': r['category'], 'target': r['category'], 'prompt': build_prompt(r['text'], taxonomy)} for i, r in enumerate(source)]
    excluded = overlaps | conflicts
    audit = {'overlap_scope': 'used capped frozen train plus frozen dev only; unused publisher training rows not audited', 'official_rows': len(records), 'overlap_groups': sorted(overlaps), 'conflicting_groups': sorted(conflicts),
             'duplicate_groups': sorted(g for g in test_labels if sum(r['group_id'] == g for r in records) > 1),
             'quarantined_ids': [r['id'] for r in records if r['group_id'] in excluded],
             'clean_ids': [r['id'] for r in records if r['group_id'] not in excluded],
             'policy': 'Official rows unchanged; overlaps and conflicts quarantined from clean-subset sensitivity only.'}
    return records, audit


def export_test(frozen: Path, raw: Path, report_path: Path, selection_path: Path, output: Path, gate_path: Path) -> dict:
    verify_dataset(frozen)
    snapshots = {name: path.read_bytes() for name, path in (('development_report.json', report_path), ('selection.json', selection_path), ('gate.json', gate_path))}
    report = json.loads(snapshots['development_report.json'])
    selection = json.loads(snapshots['selection.json'])
    gate = json.loads(snapshots['gate.json'])
    validate_selection(report, selection, hashlib.sha256(snapshots['development_report.json']).hexdigest(), sha256(frozen / 'manifest.json'), gate)
    if report['development_gate']['sha256'] != hashlib.sha256(snapshots['gate.json']).hexdigest():
        raise ValueError('Gate definition hash mismatch')
    if report['data_sha256'] != sha256(frozen / 'dev.jsonl'):
        raise ValueError('Development report references another split')
    taxonomy = json.loads((frozen / 'taxonomy.json').read_text())
    if len(taxonomy) != 77 or len(set(taxonomy)) != 77:
        raise ValueError('Canonical 77-label taxonomy required')
    dev_ids = [json.loads(line)['id'] for line in (frozen / 'dev.jsonl').read_text().splitlines()]
    for metadata in (report['candidate_inference'], report['baseline_inference']):
        if metadata.get('subset') is not False or metadata.get('selected_ids') != dev_ids or metadata.get('dataset_sha256') != report['data_sha256'] or metadata.get('taxonomy_sha256') != sha256(frozen / 'taxonomy.json'):
            raise ValueError('Full frozen development provenance required')
    if output.exists():
        raise FileExistsError(output)
    # Reserve before reading sealed bytes. Retain reservation even on failure: no quiet retries.
    reservation = raw.parent / 'official-test-export.receipt.json'
    with reservation.open('x') as stream:
        stream.write(canonical({'status': 'reserved', 'selection': selection, 'output': str(output.resolve())}) + '\n')
    references = [json.loads(line) for split in ('train', 'dev') for line in (frozen / f'{split}.jsonl').read_text().splitlines()]
    records, audit = transform_test(raw.read_bytes(), taxonomy, references, EXPECTED['test.csv'], TEST_BLOB, 3080)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.test-export-', dir=output.parent))
    try:
        (temporary / 'test.jsonl').write_text(''.join(canonical(r) + '\n' for r in records))
        for name, value in [('taxonomy.json', taxonomy), ('audit.json', audit)]:
            (temporary / name).write_text(canonical(value) + '\n')
        for name, content in snapshots.items():
            (temporary / name).write_bytes(content)
        manifest = {'version': 1, 'split': 'test', 'rows': 3080, 'raw_sha256': EXPECTED['test.csv'], 'raw_git_blob_sha1': TEST_BLOB,
                    'selection': selection, 'export_code_sha256': sha256(Path(__file__)),
                    'files': {p.name: sha256(p) for p in temporary.iterdir()}}
        (temporary / 'manifest.json').write_text(canonical(manifest) + '\n')
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def load_test_records(path: Path) -> list[dict]:
    if path.name != 'test.jsonl':
        raise ValueError('Expected official test.jsonl')
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    required = {'test.jsonl', 'taxonomy.json', 'audit.json', 'selection.json', 'development_report.json', 'gate.json'}
    if set(manifest['files']) != required or manifest['raw_sha256'] != EXPECTED['test.csv'] or manifest['raw_git_blob_sha1'] != TEST_BLOB:
        raise ValueError('Invalid official-test export manifest')
    if manifest.get('export_code_sha256') != sha256(Path(__file__)):
        raise ValueError('Test export code changed after release')
    snapshots = {name: (path.parent / name).read_bytes() for name in required}
    for name, digest in manifest['files'].items():
        if hashlib.sha256(snapshots[name]).hexdigest() != digest:
            raise ValueError('Official-test export modified')
    selection = json.loads(snapshots['selection.json'])
    if manifest['selection'] != selection:
        raise ValueError('Selection mismatch')
    report = json.loads(snapshots['development_report.json'])
    gate = json.loads(snapshots['gate.json'])
    validate_selection(report, selection, hashlib.sha256(snapshots['development_report.json']).hexdigest(), selection['data_manifest_sha256'], gate)
    if report['development_gate']['sha256'] != hashlib.sha256(snapshots['gate.json']).hexdigest():
        raise ValueError('Released gate differs from development gate')
    rows = [json.loads(line) for line in snapshots['test.jsonl'].decode('utf-8').splitlines()]
    if hashlib.sha256(snapshots['taxonomy.json']).hexdigest() != report['candidate_inference']['taxonomy_sha256']:
        raise ValueError('Publisher taxonomy changed after selection')
    taxonomy = json.loads(snapshots['taxonomy.json'])
    if len(rows) != 3080 or len(taxonomy) != 77 or len(set(taxonomy)) != 77 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Test size or taxonomy mismatch')
    for row in rows:
        if row['split'] != 'test' or row['label'] not in taxonomy or row['target'] != row['label'] or row['group_id'] != group_id(row['text']) or row['prompt'] != build_prompt(row['text'], taxonomy):
            raise ValueError('Invalid test row')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'raw', 'report', 'selection', 'output', 'gate'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-official-test', action='store_true', required=True)
    args = parser.parse_args()
    export_test(args.frozen, args.raw, args.report, args.selection, args.output, args.gate)


if __name__ == '__main__':
    main()
