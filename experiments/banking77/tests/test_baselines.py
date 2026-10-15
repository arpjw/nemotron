import unittest
try:
    import sklearn
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
from banking77.baselines import make_classifier, metrics

@unittest.skipUnless(AVAILABLE, 'scikit-learn not installed')
class BaselineTests(unittest.TestCase):
    def test_metric_includes_all_labels(self):
        result = metrics(['a', 'b'], ['a', 'a'], ['a', 'b'])
        self.assertEqual(result['accuracy'], 0.5)
        self.assertAlmostEqual(result['macro_f1'], 1/3)

    def test_vocabulary_does_not_fit_development(self):
        pipeline = make_classifier()
        pipeline.fit(['card broken red', 'card broken blue', 'transfer pending red', 'transfer pending blue'], ['a', 'a', 'b', 'b'])
        word = pipeline.named_steps['features'].transformer_list[0][1]
        before = dict(word.vocabulary_)
        pipeline.predict(['secretdevonlytoken'])
        self.assertEqual(word.vocabulary_, before)
        self.assertNotIn('secretdevonlytoken', word.vocabulary_)

    def test_invalid_predictions_are_not_silently_dropped(self):
        with self.assertRaises(ValueError): metrics(['a', 'b'], ['a'], ['a', 'b'])
        with self.assertRaises(ValueError): metrics(['a'], ['unknown'], ['a', 'b'])
