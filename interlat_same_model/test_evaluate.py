"""Tests for the text-control prompt construction used in evaluation."""

from __future__ import annotations

import unittest

from interlat_same_model.evaluate import baseline_user_prompt


class EvaluationControlTest(unittest.TestCase):
    def test_question_only_control_omits_sender_plan(self):
        prompt = baseline_user_prompt("Which datasets were used?")
        self.assertIn("Which datasets were used?", prompt)
        self.assertNotIn("Internal plan:", prompt)

    def test_text_plan_control_contains_plan_and_question(self):
        prompt = baseline_user_prompt("Which datasets were used?", "Europarl and MultiUN")
        self.assertIn("Internal plan:\nEuroparl and MultiUN", prompt)
        self.assertIn("Question: Which datasets were used?", prompt)


if __name__ == "__main__":
    unittest.main()
