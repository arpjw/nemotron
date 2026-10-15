"""Locked CPU baselines fitted only on the capped BANKING77 training split."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from .data import SEED, load_frozen_records, sha256, verify_dataset

CONFIG = {
    'word': {'analyzer': 'word', 'ngram_range': [1, 2], 'min_df': 2, 'max_features': 50000, 'sublinear_tf': True},
    'character': {'analyzer': 'char_wb', 'ngram_range': [3, 5], 'min_df': 2, 'max_features': 50000, 'sublinear_tf': True},
    'classifier': {'C': 4.0, 'max_iter': 1000, 'solver': 'lbfgs', 'random_state': SEED},
}


def make_classifier() -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.linear_model import LogisticRegression
    features = []
    for name in ('word', 'character'):
        config = dict(CONFIG[name])
        config['ngram_range'] = tuple(config['ngram_range'])
        features.append((name, TfidfVectorizer(**config)))
    return Pipeline([('features', FeatureUnion(features)), ('classifier', LogisticRegression(**CONFIG['classifier']))])


def metrics(labels: list[str], predictions: list[str], taxonomy: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    if not labels or len(labels) != len(predictions) or any(x not in taxonomy for x in labels + predictions):
        raise ValueError('Invalid metric inputs')
    return {
        'n': len(labels), 'accuracy': float(accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(labels, predictions, labels=taxonomy, average='macro', zero_division=0)),
        'per_label': classification_report(labels, predictions, labels=taxonomy, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(labels, predictions, labels=taxonomy).tolist(),
        'label_order': taxonomy,
    }


def run(directory: Path, output: Path) -> dict[str, Any]:
    import importlib.metadata
    import warnings
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    if output.exists():
        raise FileExistsError('Baseline output already exists')
    manifest = verify_dataset(directory)
    train = load_frozen_records(directory / 'train.jsonl', allowed_split='train')
    dev = load_frozen_records(directory / 'dev.jsonl', allowed_split='dev')
    taxonomy = json.loads((directory / 'taxonomy.json').read_text())
    counts = Counter(row['label'] for row in train)
    majority = max(taxonomy, key=lambda label: counts[label])  # publisher order breaks ties
    classifier = make_classifier()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        classifier.fit([row['text'] for row in train], [row['label'] for row in train])
    predictions = {
        'majority': [majority] * len(dev),
        'tfidf_logistic': classifier.predict([row['text'] for row in dev]).tolist(),
    }
    labels = [row['label'] for row in dev]
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, predicted in predictions.items():
        path = output / f'{name}_predictions.jsonl'
        with path.open('x', encoding='utf-8') as stream:
            for row, label in zip(dev, predicted):
                stream.write(json.dumps({'id': row['id'], 'prediction': label}) + '\n')
        reports[name] = metrics(labels, predicted, taxonomy)
    joblib.dump(classifier, output / 'tfidf_logistic.joblib')
    report = {
        'status': 'completed', 'split': 'dev', 'official_test_evaluated': False,
        'config': CONFIG, 'majority_label': majority,
        'versions': {name: importlib.metadata.version(name) for name in ('scikit-learn', 'numpy', 'scipy', 'joblib')},
        'provenance': {
            'train_sha256': manifest['files']['train.jsonl']['sha256'],
            'dev_sha256': manifest['files']['dev.jsonl']['sha256'],
            'taxonomy_sha256': manifest['files']['taxonomy.json']['sha256'],
            'data_manifest_sha256': sha256(directory / 'manifest.json'),
            'baseline_code_sha256': sha256(Path(__file__)),
            'classifier_sha256': sha256(output / 'tfidf_logistic.joblib'),
            'prediction_sha256': {name: sha256(output / f'{name}_predictions.jsonl') for name in predictions},
        },
        'scores': reports,
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/frozen'))
    parser.add_argument('--output', type=Path, default=Path('results/cpu-baselines'))
    args = parser.parse_args()
    report = run(args.data, args.output)
    print(json.dumps({name: {key: value[key] for key in ('n', 'accuracy', 'macro_f1')} for name, value in report['scores'].items()}, indent=2))


if __name__ == '__main__':
    main()
