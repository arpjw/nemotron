import unittest
from banking77.evaluation import canonical_label, evaluate, paired_interval, paired_f1_interval, passes_development_gate


class EvaluationTests(unittest.TestCase):
    def test_malformed_not_coerced_and_case_punctuation_preserved(self):
        labels = ["reverted_card_payment?", "Refund_not_showing_up"]
        self.assertIsNone(canonical_label('"Refund_not_showing_up"', labels))
        self.assertIsNone(canonical_label("refund_not_showing_up", labels))
        self.assertEqual(canonical_label(" reverted_card_payment?\n", labels), labels[0])

    def test_primary_separate_from_raw_format_accuracy(self):
        labels = ["a", "b"]
        rows = [{"id": "1", "group_id": "1", "label": "a"}, {"id": "2", "group_id": "2", "label": "b"}]
        predictions = {"1": {"constrained": "a", "raw": "Answer: a"}, "2": {"constrained": "b", "raw": "b"}}
        result = evaluate(rows, predictions, labels)
        self.assertEqual(result["modes"]["constrained"]["accuracy"], 1)
        self.assertEqual(result["modes"]["raw"]["accuracy"], .5)
        self.assertEqual(result["modes"]["raw"]["malformed"], 1)
        self.assertEqual(result["modes"]["raw"]["valid_label_accuracy"], 1)
        with self.assertRaises(ValueError):
            evaluate(rows, {"1": predictions["1"]}, labels)

    def test_paired_constant_difference(self):
        base = [{"id": "1", "group_id": "1", "constrained_correct": False}]
        candidate = [{"id": "1", "group_id": "1", "constrained_correct": True}]
        result = paired_interval(base, candidate, samples=100)
        self.assertEqual((result["delta"], result["lower_95"], result["upper_95"]), (1, 1, 1))

    def test_macro_f1_interval_is_named_and_distinct_from_accuracy(self):
        base = [{"id": "1", "group_id": "1", "reference": "a", "constrained_label": None}]
        candidate = [{"id": "1", "group_id": "1", "reference": "a", "constrained_label": "a"}]
        result = paired_f1_interval(base, candidate, ["a", "b"], samples=100)
        self.assertEqual(result["metric"], "constrained_macro_f1")
        self.assertEqual((result["delta"], result["lower_95"], result["upper_95"]), (.5, .5, .5))

    def test_gate_requires_absolute_quality_gain_and_positive_uncertainty_bound(self):
        gate = {"macro_f1_at_least": .75, "gain_over_base_at_least": .05, "paired_macro_f1_lower_95_strictly_above": 0}
        self.assertTrue(passes_development_gate(.8, {"delta": .1, "lower_95": .01}, gate))
        self.assertFalse(passes_development_gate(.74, {"delta": .1, "lower_95": .01}, gate))
        self.assertFalse(passes_development_gate(.8, {"delta": .04, "lower_95": .01}, gate))
        self.assertFalse(passes_development_gate(.8, {"delta": .1, "lower_95": 0}, gate))
