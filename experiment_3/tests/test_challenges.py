import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiment_3.challenge_cases import CASES
from experiment_3.challenge_report import export
from experiment_3.prepare_challenges import prepare


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ExpandedChallengeTest(unittest.TestCase):
    def sources(self, root, count=26):
        path = root / 'ami.jsonl'
        originals = [dict(id=str(i), transcript=f'Project Manager: Original meeting {i}.\nMarketing: End.',
                          summary='Original reference only.') for i in range(count)]
        path.write_text(''.join(json.dumps(r)+'\n' for r in originals), encoding='utf-8')
        return path, originals

    def test_twenty_new_sources_exact_text_and_no_answers_in_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, originals = self.sources(root)
            folder = prepare(source, root / 'challenge')
            keys = json.loads((folder / 'answer_key.json').read_text())
            rows = [json.loads(line) for line in (folder / 'meetings.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 20)
            self.assertEqual(len({r['id'] for r in rows}), 20)
            self.assertEqual(Counter(k['category'] for k in keys.values()),
                             dict(revision=4, owner_change=4, condition=4, scope=4, no_change=4))
            for row, original, addition in zip(rows, originals[6:], CASES):
                self.assertEqual(set(row), {'id', 'transcript'})
                key = keys[row['id']]
                self.assertEqual(key['source_id'], original['id'])
                self.assertEqual(row['transcript'], addition['opening']+'\n'+original['transcript']+'\n'+addition['closing'])
                self.assertNotIn(key['expected'], row['transcript'])
                for label in ('opening', 'closing'):
                    a, b = key[label]['character_range']
                    self.assertEqual(row['transcript'][a:b], key[label]['text'])
            with self.assertRaises(FileExistsError):
                prepare(source, folder)

    def test_not_enough_sources_does_not_silently_shrink_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self.sources(root, 25)
            with self.assertRaisesRegex(ValueError, '20 source meetings'):
                prepare(source, root / 'challenge')
            self.assertFalse((root / 'challenge').exists())

    def test_report_keeps_missing_cases_and_does_not_invent_target_verdicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self.sources(root)
            dataset = prepare(source, root / 'challenge')
            definition = json.loads((dataset / 'manifest.json').read_text())
            keys = json.loads((dataset / 'answer_key.json').read_text())
            sid = next(iter(keys))
            workflow, direct = root / 'workflow', root / 'direct'
            save(workflow / 'manifest.json', dict(input_sha256=definition['meetings_sha256'],
                                                  conditions=['baseline', 'semantic']))
            save(direct / 'manifest.json', dict(dataset=definition))
            case = workflow / '00000'
            save(case / 'source.json', dict(id=sid))
            for name, hit in (('baseline', False), ('semantic', True)):
                (case / f'{name}.summary.txt').write_text('Example summary', encoding='utf-8')
            save(case / 'target_coverage.json', {name: dict(
                opening=dict(fully_retained=True), closing=dict(fully_retained=True),
                both_in_same_local_window=hit) for name, hit in (('baseline', False), ('semantic', True))})
            save(direct / '00000' / 'result.json', dict(id=sid))
            (direct / '00000' / 'summary.txt').write_text('Direct answer', encoding='utf-8')
            out = export(dataset, workflow, direct, root / 'report')
            with (out / 'target_review.csv').open(encoding='utf-8-sig') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 60)
            self.assertEqual(sum(r['status'] == 'completed' for r in rows), 3)
            self.assertTrue(all(r['verdict'] == '' for r in rows))
            self.assertTrue(all(r['co_location'] == '' for r in rows if r['condition'] == 'direct_text'))
            report = (out / 'coverage.txt').read_text()
            self.assertIn('baseline | 1 | 1 | 0 | 0.0%', report)
            self.assertIn('semantic | 1 | 1 | 1 | 100.0%', report)
            with self.assertRaises(FileExistsError):
                export(dataset, workflow, direct, out)


if __name__ == '__main__':
    unittest.main()
