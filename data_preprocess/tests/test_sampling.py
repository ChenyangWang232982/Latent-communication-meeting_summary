"""Sampling caps apply after filtering and never duplicate scarce meetings."""
import json
from pathlib import Path
import tempfile
import unittest

from data_preprocess.prepare_qmsum import normalize
from data_preprocess.prepare_meetingbank import export_rows


class SamplingTest(unittest.TestCase):
    def test_qmsum_cap_all_and_deterministic_sampling_after_filtering(self):
        rows = [{'meeting_id': str(i), 'meeting_transcripts': [{'speaker': 'Alice', 'content': str(i)}],
                 'general_query_list': [{'query': 'Summarize', 'answer': 'Reference'}]} for i in range(130)]
        class Filter:
            def retain(self, row):
                return int(row['id']) < 110
        selected = normalize(rows + rows[:3], max_records=100, token_filter=Filter())
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({r['id'] for r in selected}), 100)
        self.assertTrue(all(int(r['id']) < 110 for r in selected))
        self.assertEqual(selected, normalize(rows, max_records=100, token_filter=Filter()))
        self.assertEqual(len(normalize(rows[:36], max_records=100)), 36)
        self.assertEqual(len(normalize(rows, max_records=0)), 130)

    def test_meetingbank_cap_all_and_filter_before_sample(self):
        rows = [{'uid': str(i), 'turns': [{'speaker': 'Alice', 'content': str(i)}],
                 'summary': 'Reference'} for i in range(130)]
        class Filter:
            def retain(self, row):
                return int(row['id']) < 110
        with tempfile.TemporaryDirectory() as directory:
            def export(name, source, cap, token_filter=None):
                output = Path(directory) / name
                count, _, _ = export_rows(source, output, max_records=cap, overwrite=False,
                                           require_speakers=True, token_filter=token_filter)
                records = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(count, len(records))
                return records
            first = export('a.jsonl', rows, 100, Filter())
            self.assertEqual(first, export('b.jsonl', rows, 100, Filter()))
            self.assertEqual(len(first), 100)
            self.assertEqual(len({r['id'] for r in first}), 100)
            self.assertTrue(all(int(r['id']) < 110 for r in first))
            self.assertEqual(len(export('small.jsonl', rows[:36], 100)), 36)
            self.assertEqual(len(export('all.jsonl', rows, 0)), 130)
