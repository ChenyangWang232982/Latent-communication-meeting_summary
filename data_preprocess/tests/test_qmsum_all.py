import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from data_preprocess.prepare_all import DOMAINS, run


class QMSumBatchTest(unittest.TestCase):
    def test_single_tokenizer_and_independent_domain_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for domain in DOMAINS.values():
                folder = root / 'data' / domain / 'all'
                folder.mkdir(parents=True)
                (folder / 'one.json').write_text(json.dumps({
                    'meeting_id': domain, 'meeting_transcripts': [{'speaker': 'Alice', 'content': 'Hello'}],
                    'general_query_list': [{'query': 'Summarize', 'answer': 'A greeting'}]}), encoding='utf-8')
            args = argparse.Namespace(input_root=root, output_dir=root/'output', domains=list(DOMAINS),
                max_records=100, seed=7, overwrite=False, tokenizer_model='fake',
                min_transcript_tokens=0, max_transcript_tokens=None)
            tokenizer = SimpleNamespace(encode=lambda text, **kw: text.split())
            target = 'data_preprocess.prepare_all.load_token_filter'
            with patch(target, return_value=SimpleNamespace(tokenizer=tokenizer)) as loader:
                results = run(args)
                self.assertEqual(loader.call_count, 1)
            for name, result in results.items():
                self.assertEqual(result['status'], 'completed')
                report = json.loads((args.output_dir / f'{name}.jsonl.report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['token_filter']['valid_records_scanned'], 1)
                self.assertEqual(report['records_written'], 1)
            original = (args.output_dir/'ami.jsonl').read_bytes()
            # One existing output fails independently; remaining domains still run.
            (args.output_dir/'icsi.jsonl').unlink()
            (args.output_dir/'qmsum_committee.jsonl').unlink()
            with patch(target, return_value=SimpleNamespace(tokenizer=tokenizer)):
                retry = run(args)
            self.assertEqual(retry['ami']['status'], 'failed')
            self.assertEqual(retry['icsi']['status'], 'completed')
            self.assertEqual(retry['qmsum_committee']['status'], 'completed')
            self.assertEqual((args.output_dir/'ami.jsonl').read_bytes(), original)
            # MeetingBank participates in the same batch using the same tokenizer.
            meetingbank = root/'meetingbank.jsonl'
            meetingbank.write_text(json.dumps({'uid': 'mb1', 'turns': [{'speaker': 'Bob', 'content': 'Hello'}],
                                               'summary': 'A greeting'})+'\n', encoding='utf-8')
            args.domains = [*DOMAINS, 'meetingbank']
            args.meetingbank_root, args.split, args.overwrite = meetingbank, 'test', True
            with patch(target, return_value=SimpleNamespace(tokenizer=tokenizer)) as loader:
                all_results = run(args)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(len(all_results), 4)
            self.assertTrue(all(row['status'] == 'completed' for row in all_results.values()))
            mb_report = json.loads((args.output_dir/'meetingbank.jsonl.report.json').read_text(encoding='utf-8'))
            self.assertEqual(mb_report['token_filter']['valid_records_scanned'], 1)
