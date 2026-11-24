import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from receiver_kv_workflow.engine import ReceiverKVConfig
from experiment_2.components.config import DirectConfig
from experiment_3.evaluate import evaluate
from experiment_3.prepare_comparison import prepare


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


class EvaluationTest(unittest.TestCase):
    def fixture(self, root):
        workflow, direct = root / 'workflow', root / 'direct'
        save(workflow / 'manifest.json', dict(sample_ids=['a', 'b', 'bad', 'missing'],
                                             config=asdict(ReceiverKVConfig(device='cpu'))))
        save(direct / 'manifest.json', dict(sample_ids=['a', 'b', 'bad'], config=asdict(DirectConfig(device='cpu'))))
        for i, sid in enumerate(('a', 'b')):
            source, reference = 'Alice: The plan was approved.', 'The plan was approved.'
            save(workflow / f'{i:05d}' / 'source.json', dict(id=sid, transcript=source, reference=reference))
            for condition in ('baseline', 'semantic'):
                (workflow / f'{i:05d}' / f'{condition}.summary.txt').write_text(reference, encoding='utf-8')
                save(workflow / f'{i:05d}' / f'{condition}.specialists.json', dict(transcript_tokens=100,
                    stages=dict(local=dict(output_cap_hit=False), final=dict(output_cap_hit=False))))
                save(workflow / f'{i:05d}' / f'{condition}.metrics.json', dict(logical_total_tokens=25,
                    standalone_pipeline_seconds=3, logical_input_tokens=17, logical_output_tokens=8,
                    total_embedding_input_tokens=100, local_kv_position_events=80, final_kv_positions=70))
            save(direct / 'samples' / f'{i:05d}.json', dict(id=sid, status='completed', source=source,
                reference=reference, result=dict(answers=dict(direct_text=reference), variants=dict(direct_text=dict(
                    output_cap_hit=False, pipeline_seconds_excluding_model_load=1,
                    token_accounting=dict(source_prefill_tokens=0, selection_probe_tokens=0,
                        specialist_prompt_tokens=0, specialist_handoff_tokens=0,
                        receiver_prompt_tokens=5, receiver_output_tokens=3))))))
        save(workflow / '00002' / 'failure.json', dict(error='context limit'))
        save(direct / 'samples' / '00002.json', dict(id='bad', status='failed', error='context limit'))
        return workflow, direct

    def test_saved_pairs_reuse_metrics_and_one_shared_bertscore_call(self):
        with tempfile.TemporaryDirectory() as temp:
            workflow, direct = self.fixture(Path(temp))
            def fake(samples, model, layers, device, *, variants):
                self.assertEqual(len(samples), 6)
                self.assertEqual(variants, ('summary',))
                for s in samples:
                    s['quality']['summary'].update(bertscore_precision=.8, bertscore_recall=.9, bertscore_f1=.85)
                return dict(model=model, layers=layers)
            with patch('experiment_3.evaluate.add_bertscore', side_effect=fake) as scorer:
                folder = evaluate(workflow, direct, bertscore_model='mock', device='cpu')
                self.assertEqual(scorer.call_count, 1)
            report = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['paired_samples'], 2)
            self.assertEqual(report['workflow_failed_ids'], ['bad'])
            self.assertEqual(report['workflow_missing_ids'], ['missing'])
            self.assertEqual(report['means']['semantic']['total_tokens'], 25)
            self.assertEqual(report['means']['direct_text']['total_tokens'], 8)
            self.assertIsNone(report['means']['semantic']['cuda_peak_mib'])
            comparison = report['comparisons']['semantic']['overall']['metrics']
            self.assertEqual(comparison['rouge_l_f1']['difference_workflow_minus_direct'], 0)
            self.assertAlmostEqual(comparison['bertscore_f1']['workflow'], .85)
            self.assertTrue((folder / 'per_sample.csv').exists())

    def test_reference_and_source_mismatch_are_not_silently_paired(self):
        with tempfile.TemporaryDirectory() as temp:
            workflow, direct = self.fixture(Path(temp))
            path = workflow / '00000' / 'source.json'
            original = json.loads(path.read_text())
            save(path, dict(original, transcript='Different source'))
            with self.assertRaisesRegex(ValueError, 'Source/reference mismatch'):
                evaluate(workflow, direct)
            save(path, dict(original, reference=''))
            with self.assertRaisesRegex(ValueError, 'no reference summary'):
                evaluate(workflow, direct)

    def test_prepare_preserves_original_references_and_skips_development_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'all.jsonl'
            source.write_text(''.join(json.dumps(dict(id=str(i), transcript=f'Alice: Record {i}.',
                summary=f'Reference {i}.'))+'\n' for i in range(8)), encoding='utf-8')
            output = prepare(source, root / 'input' / 'frozen.jsonl')
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r['id'] for r in rows], ['3', '4', '5'])
            self.assertEqual(rows[0]['summary'], 'Reference 3.')
            with self.assertRaises(FileExistsError):
                prepare(source, output)

    def test_semantic_only_run_does_not_require_baseline_files(self):
        with tempfile.TemporaryDirectory() as temp:
            workflow, direct = self.fixture(Path(temp))
            path = workflow / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['conditions'] = ['semantic']
            save(path, manifest)
            for file in workflow.glob('*/baseline.*'):
                file.unlink()
            output = evaluate(workflow, direct)
            result = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(set(result['means']), {'semantic', 'direct_text'})
            self.assertEqual(result['paired_samples'], 2)

    def test_generation_settings_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            workflow, direct = self.fixture(Path(temp))
            path = direct / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['config']['summary_seed'] = 99
            save(path, manifest)
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                evaluate(workflow, direct)


if __name__ == '__main__':
    unittest.main()
