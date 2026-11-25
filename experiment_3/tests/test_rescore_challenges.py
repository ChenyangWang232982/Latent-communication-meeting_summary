import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from receiver_kv_workflow.engine import ReceiverKVConfig
from experiment_3.rescore_challenges import prepare, rescore


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ChallengeRescoreTest(unittest.TestCase):
    def fixture(self, root):
        run = root/'batch'
        workflow, direct = run/'workflow'/'run', run/'direct'/'run'/'direct_text'
        ids = ['ami_1_endpoint', 'ami_2_endpoint']
        config = asdict(ReceiverKVConfig(device='cpu'))
        save(workflow/'manifest.json', dict(sample_ids=ids, input_sha256='same', config=config))
        save(direct/'manifest.json', dict(sample_ids=ids, dataset=dict(meetings_sha256='same'), config=config))
        originals = []
        for i, sid in enumerate(ids):
            source = f'Alice: Meeting {i}.\nBob: Keep the plan.'
            reference = 'Keep the plan.'
            originals.append(dict(id=str(i+1), transcript=source, summary=reference))
            case = workflow/f'{i:05d}'
            save(case/'source.json', dict(id=sid, transcript='Marketing: Synthetic opening.\n'+source+'\nProject Manager: Synthetic closing.'))
            for condition in ('baseline', 'semantic'):
                (case/f'{condition}.summary.txt').write_text(reference, encoding='utf-8')
                save(case/f'{condition}.specialists.json', dict(stages=dict(final=dict(output_cap_hit=False))))
            dc = direct/f'{i:05d}_{sid}'
            save(dc/'result.json', dict(id=sid, variants=dict(direct_text=dict(output_cap_hit=False))))
            (dc/'summary.txt').write_text(reference, encoding='utf-8')
        refs = root/'ami.jsonl'
        refs.write_text(''.join(json.dumps(row)+'\n' for row in originals), encoding='utf-8')
        return run, refs, workflow, direct

    def test_original_reference_only_and_one_shared_bertscore_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, refs, _, _ = self.fixture(root)
            before = refs.read_bytes()
            def score(samples, model, layers, device, *, variants):
                self.assertEqual(len(samples), 6)
                self.assertEqual(variants, ('summary',))
                self.assertTrue(all(s['reference']=='Keep the plan.' for s in samples))
                for sample in samples:
                    sample['quality']['summary'].update(bertscore_precision=.8, bertscore_recall=.9, bertscore_f1=.85)
                return dict(model=model, layers=layers, device=device)
            with patch('experiment_3.rescore_challenges.add_bertscore', side_effect=score) as mocked:
                out = rescore(run, refs, output_dir=root/'evaluation', device='cpu')
            self.assertEqual(mocked.call_count, 1)
            result = json.loads((out/'summary.json').read_text())
            self.assertIsNone(result['factual_accuracy'])
            self.assertEqual(result['paired'], 2)
            self.assertEqual(result['means']['semantic']['bertscore_precision'], .8)
            self.assertEqual(refs.read_bytes(), before)
            with (out/'per_sample.csv').open(encoding='utf-8-sig') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)

    def test_same_id_with_wrong_original_body_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run, refs, workflow, _ = self.fixture(Path(temp))
            path = workflow/'00000/source.json'
            value = json.loads(path.read_text())
            value['transcript'] = value['transcript'].replace('Keep the plan.', 'A different source.')
            save(path, value)
            with self.assertRaisesRegex(ValueError, 'original meeting body differs'):
                prepare(run, refs)

    def test_missing_output_excludes_whole_pair_not_zero_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            run, refs, workflow, _ = self.fixture(Path(temp))
            (workflow/'00001/semantic.summary.txt').unlink()
            _, samples, missing, _, _ = prepare(run, refs)
            self.assertEqual(len(samples), 3)
            self.assertEqual(missing, ['ami_2_endpoint'])


if __name__ == '__main__':
    unittest.main()
