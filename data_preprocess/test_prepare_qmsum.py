import json
from pathlib import Path
import tempfile
import unittest

from .prepare_qmsum import _read, normalize, speaker_transcript


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

    def test_rejects_nonempty_turn_without_speaker_and_skips_empty_turn(self):
        self.assertEqual(speaker_transcript([{"content": "No speaker."}]), "")
        self.assertEqual(
            speaker_transcript([{"speaker": "Alice", "content": "Hello."}, {"speaker": "", "content": ""}]),
            "Alice: Hello.",
        )

    def test_reads_all_directory_of_individual_meetings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "all"
            root.mkdir()
            (root / "meeting.json").write_text(json.dumps({"meeting_id": "m1"}), encoding="utf-8")
            self.assertEqual(_read(root), [{"meeting_id": "m1"}])


if __name__ == "__main__":
    unittest.main()
