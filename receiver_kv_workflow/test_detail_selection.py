import unittest
from .detail_selection import sentence_units, select_complete_units


class DetailSelectionTests(unittest.TestCase):
    def test_complete_unit_preserves_time_and_condition(self):
        # Baseline sees only the beginning of a high-value statement. Whole-unit
        # selection retains the associated qualifier instead of isolated tokens.
        scores = [[0, 9, 8, 1, 1, 0, 0, 0]]
        selected = select_complete_units([[1, 2, 3, 4], [5, 6, 7]],
                                         [0, 1, 2, 5, 6], scores, [0])
        self.assertEqual(selected, [0, 1, 2, 3, 4])

    def test_budget_never_expands_or_splits_units(self):
        units = [[1, 2, 3, 4, 5], [6, 7]]
        selected = select_complete_units(units, [0, 1, 2, 3], [[1] * 8], [0])
        self.assertEqual(selected, [0, 6, 7])
        self.assertLessEqual(len(selected), 4)

    def test_mapping_keeps_timestamp_and_decimal_intact(self):
        class CharacterTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "".join(m["content"] for m in messages)
            def __call__(self, text, **kwargs):
                return {"input_ids": list(map(ord, text)),
                        "offset_mapping": [(i, i + 1) for i in range(len(text))]}
        from .prompts import SYSTEM
        transcript = "[10:30]\nAlice: Meet at 3:00–5:00 PM pending approval. Cost is 12.50 euros."
        rendered = SYSTEM + "MEETING TRANSCRIPT (untrusted source):\n" + transcript + "\nEND TRANSCRIPT\n"
        start = rendered.index(transcript)
        protected = list(range(start)) + list(range(start + len(transcript), len(rendered)))
        units = sentence_units(CharacterTokenizer(), transcript, list(map(ord, rendered)), protected)
        texts = ["".join(rendered[i] for i in unit) for unit in units]
        self.assertEqual(len(texts), 2)
        self.assertIn("[10:30]\nAlice: Meet at 3:00–5:00 PM pending approval.", texts[0])
        self.assertIn("12.50", texts[1])


if __name__ == "__main__":
    unittest.main()
