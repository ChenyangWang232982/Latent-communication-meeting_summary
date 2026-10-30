import unittest

from .prepare_qmsum import normalize, speaker_transcript


class QMSumPreparationTests(unittest.TestCase):
    def test_writes_one_general_query_per_speaker_labelled_meeting(self):
        rows = [{
            "meeting_id": "m1",
            "meeting_transcripts": [
                {"speaker": "Alice", "content": "We need a decision."},
                {"speaker": "Bob", "content": "I agree."},
            ],
            "general_query_list": [
                {"query": "Summarize the whole meeting.", "answer": "Alice raised a decision and Bob agreed."},
                {"query": "Another general query.", "answer": "Ignored to avoid duplicate meetings."},
            ],
        }]
        records = normalize(rows, max_records=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "m1")
        self.assertEqual(records[0]["transcript"], "Alice: We need a decision.\nBob: I agree.")

    def test_rejects_missing_speaker_labels(self):
        with self.assertRaises(ValueError):
            speaker_transcript([{"content": "No speaker."}])


if __name__ == "__main__":
    unittest.main()
