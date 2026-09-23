"""Regression tests for oracle-evidence QASPER preparation."""

from __future__ import annotations

import unittest

from interlat_same_model.prepare_oracle_qasper import answer_text, unique_nonempty


class OraclePreparationTest(unittest.TestCase):
    def test_answer_priority(self):
        self.assertEqual(answer_text({"extractive_spans": ["Inception V3", "biLSTM"]}), "Inception V3 biLSTM")
        self.assertEqual(answer_text({"yes_no": True}), "yes")
        self.assertEqual(answer_text({"unanswerable": True}), "unanswerable")

    def test_evidence_is_deduplicated(self):
        self.assertEqual(unique_nonempty([" fact ", "fact", "", "other"]), ["fact", "other"])


if __name__ == "__main__":
    unittest.main()
