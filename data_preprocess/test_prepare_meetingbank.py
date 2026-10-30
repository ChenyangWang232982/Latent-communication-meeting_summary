import json
from pathlib import Path
import tempfile
import unittest

from .prepare_meetingbank import export_rows


class MeetingBankPreparationTests(unittest.TestCase):
    def test_official_schema_exports_normalized_jsonl(self):
        rows = [
            {"id": 4, "uid": "Town_4", "transcript": "Meeting transcript", "summary": "Human summary"},
            {"id": 5, "uid": "Town_5", "transcript": "", "summary": "Missing source"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "meetingbank_test.jsonl"
            written, skipped, columns = export_rows(rows, output, max_records=0, overwrite=False)
            self.assertEqual((written, skipped), (1, 1))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"id": "Town_4", "transcript": "Meeting transcript", "summary": "Human summary"})
            self.assertIn("uid", columns)

    def test_refuses_unrequested_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "meetingbank_test.jsonl"
            output.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_rows([], output, max_records=0, overwrite=False)

    def test_diarized_turns_are_rendered_for_speaker_selection(self):
        rows = [{"uid": "Town_6", "turns": [
            {"speaker": "Chair", "content": "Open the meeting."},
            {"speaker": "Member", "content": "I support this."},
        ], "summary": "The chair opened and a member supported."}]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "meetingbank_speaker_test.jsonl"
            written, _, _ = export_rows(rows, output, max_records=25, overwrite=False, require_speakers=True)
            self.assertEqual(written, 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["transcript"], "Chair: Open the meeting.\nMember: I support this.")


if __name__ == "__main__":
    unittest.main()
