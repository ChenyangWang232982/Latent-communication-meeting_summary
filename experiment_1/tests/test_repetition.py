import unittest
from experiment_1.components.repetition import repetition_metrics

class RepetitionTest(unittest.TestCase):
    def test_duplicates_across_sections_and_no_semantic_guess(self):
        result = repetition_metrics("**Actions**\n- Alice: Review.\n**Issues**\n- Alice: Review.\n- Bob: Review.")
        self.assertEqual(result["duplicate_bullet_excess"], 1)
        self.assertEqual(result["max_identical_bullet_count"], 2)
        self.assertEqual(repetition_metrics("")["duplicate_bullet_fraction"], 0)
