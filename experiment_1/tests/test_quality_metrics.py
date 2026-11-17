import unittest
from meeting_evaluation.metrics import reference_metrics, rouge_l_f1
from experiment_1.components.token_accounting import token_totals


class QualityMetricsTest(unittest.TestCase):
    def test_repeated_words_are_clipped_and_order_matters(self):
        q = reference_metrics("a a a b", "a b")
        self.assertEqual(q["rouge_1_precision"], .5)
        self.assertEqual(q["rouge_1_recall"], 1)
        self.assertEqual(q["rouge_2_precision"], 1/3)
        self.assertEqual(q["rouge_l_f1"], rouge_l_f1("a a a b", "a b"))
        self.assertEqual(reference_metrics("a b", "b a")["rouge_2_f1"], 0)
        self.assertEqual(reference_metrics("", "a")["rouge_l_precision"], 0)
        self.assertIsNone(reference_metrics("a", "")["rouge_1_f1"])

    def test_missing_accounting_is_not_silently_zero(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            token_totals({"token_accounting": {"source_prefill_tokens": 2}})
