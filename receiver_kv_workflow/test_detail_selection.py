import unittest
from .detail_selection import expand_unit_neighbors, sentence_boundaries, sentence_units, select_complete_units


class DetailSelectionTests(unittest.TestCase):
    def test_neighbor_expansion_includes_two_complete_units_each_side(self):
        units = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        selected = expand_unit_neighbors([7, 8], units, [0], radius=2)
        self.assertEqual(selected, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    def test_neighbor_expansion_clamps_at_document_ends(self):
        units = [[1], [2], [3], [4]]
        self.assertEqual(expand_unit_neighbors([1], units, [0], radius=2), [0, 1, 2, 3])
    def test_repair_is_kept_with_retracted_phrase(self):
        for repair in ("Sorry, how it executes.", "Sorry. How it executes.",
                       "I mean, how it executes."):
            text = "Show why it failed. " + repair + " Next topic."
            boundaries = sentence_boundaries(text)
            self.assertEqual(text[:boundaries[0]], "Show why it failed. " + repair + " ")

    def test_ordinary_sentence_and_new_speaker_remain_separate(self):
        text = "We will send it. Next topic.\n\nBob: Sorry, I missed that."
        boundaries = sentence_boundaries(text)
        self.assertEqual(text[:boundaries[0]], "We will send it. ")
        self.assertIn(text.index("Bob:"), boundaries)

    def test_repair_group_respects_budget(self):
        text = "It failed. Sorry, it executed. Next."
        ends = sentence_boundaries(text)
        units = [list(range(start, end)) for start, end in zip([0] + ends, ends)]
        scores = [[9.0 if i < 10 else 1.0 for i in range(len(text))]]
        selected = select_complete_units(units, list(range(12)), scores, [])
        self.assertFalse(set(selected) & set(units[0]))
        self.assertLessEqual(len(selected), 12)

    def test_audit_distinguishes_missing_condition_from_retained_evidence(self):
        from unittest.mock import patch
        from .routing_audit import build_audit
        text = "Meet 3-5 pending approval."
        offsets = [(i, i + 1) for i in range(len(text))]
        with patch("receiver_kv_workflow.routing_audit.source_offsets", return_value=(offsets, 0)), patch(
            "receiver_kv_workflow.routing_audit.sentence_units", return_value=[list(range(len(text))) ]
        ):
            audit = build_audit(None, text, [], [], list(range(8)), list(range(8)), {}, "fixed")
        self.assertEqual(audit["selected_excerpts"][0]["text"], "Meet 3-5")
        self.assertEqual(audit["sentence_coverage"][0]["status"], "partial")
        self.assertEqual(audit["sentence_coverage"][0]["source"][0]["text"], text)

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
