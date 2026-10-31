import json
from pathlib import Path
import tempfile
import unittest

from .prepare_meetingbank import _official_meetingbank_rows, export_rows


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

    def test_official_root_joins_split_to_diarized_item_turns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "Metadata"
            (metadata / "Splits").mkdir(parents=True)
            (metadata / "MeetingBank.json").write_text(json.dumps({
                "Town_1": {"itemInfo": {"Agenda A": {"transcripts": [
                    {"speaker": 2, "text": "We approve the proposal."},
                    {"speaker": 3, "text": "I agree."},
                ]}}
            }}), encoding="utf-8")
            (metadata / "Splits" / "test.json").write_text(json.dumps({
                "id": "Town_1_Agenda A", "summary": "The proposal was approved."
            }) + "\n", encoding="utf-8")
            rows = _official_meetingbank_rows(root, "test")
            self.assertEqual(rows, [{
                "uid": "Town_1_Agenda A",
                "summary": "The proposal was approved.",
                "turns": [
                    {"speaker": "Speaker_2", "content": "We approve the proposal."},
                    {"speaker": "Speaker_3", "content": "I agree."},
                ],
            }])


if __name__ == "__main__":
    unittest.main()
