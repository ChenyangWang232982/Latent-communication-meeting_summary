import unittest
import json
import tempfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment_3.components.random_control import random_windows


class RandomEvidenceTest(unittest.TestCase):
    def fixture(self):
        blocks = [list(range(i*8, i*8+8)) for i in range(12)]
        texts = {i:f'Speaker: Distinct agenda item {i} includes detail {i}.' for i in range(12)}
        base = [dict(block_ids=[0], primary_block_ids=[0], supplement_block_ids=[],
                     positions=blocks[0], matches=[], added_positions=0)]
        semantic = [dict(base[0], supplement_block_ids=[6, 10],
                         positions=sorted(blocks[0]+blocks[6]+blocks[10]))]
        return blocks, texts, base, semantic

    def test_fixed_seed_count_budget_and_source_subset(self):
        blocks, texts, base, semantic = self.fixture()
        selected = [p for block in blocks for p in block]
        kwargs = dict(record_id='meeting', seed=7, neighbor_radius=1, trials=64)
        a = random_windows(blocks, texts, selected, base, semantic, **kwargs)
        b = random_windows(blocks, texts, selected, base, semantic, **kwargs)
        self.assertEqual(a, b)
        w = a[0]
        self.assertEqual(len(w['supplement_block_ids']), 2)
        self.assertEqual(w['added_positions'], 16)
        self.assertEqual(w['matching']['absolute_position_difference'], 0)
        self.assertTrue(set(w['positions']) <= set(selected))
        self.assertFalse(set(w['supplement_block_ids']) & {0, 1})
        self.assertTrue(set(base[0]['positions']) <= set(w['positions']))

    def test_no_padding_when_semantic_added_nothing(self):
        blocks, texts, base, _ = self.fixture()
        selected = [p for b in blocks for p in b]
        w = random_windows(blocks, texts, selected, base, base, record_id='a')[0]
        self.assertEqual(w['supplement_block_ids'], [])
        self.assertEqual(w['positions'], base[0]['positions'])

    def test_whole_blocks_and_unavoidable_budget_mismatch_are_reported(self):
        blocks, texts, base, semantic = self.fixture()
        # Only ten-position distant blocks are selected; reference target requests 16.
        blocks = [list(range(i*10, i*10+10)) for i in range(12)]
        base[0]['positions'] = blocks[0]
        semantic[0]['positions'] = blocks[0]+list(range(1000,1016))
        selected = [p for b in blocks for p in b]
        w = random_windows(blocks, texts, selected, base, semantic, record_id='a')[0]
        self.assertEqual(w['matching']['signed_position_difference'], 4)
        for block_id in w['supplement_block_ids']:
            self.assertTrue(set(blocks[block_id]) <= set(w['positions']))

    def test_dropped_positions_and_duplicate_content_are_not_restored(self):
        blocks, texts, base, semantic = self.fixture()
        texts[3] = texts[2]
        selected = [p for i,b in enumerate(blocks) if i != 11 for p in b]
        w = random_windows(blocks, texts, selected, base, semantic, record_id='a')[0]
        self.assertNotIn(11, w['supplement_block_ids'])
        self.assertEqual(len({texts[i] for i in w['supplement_block_ids']}), 2)

    def test_replay_generates_only_new_control_without_selectors_or_embedding(self):
        from receiver_kv_workflow.engine import ReceiverKVConfig
        from experiment_1.components.config import sha256
        from experiment_3.random_control import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'old'; old.mkdir(); case = old/'00000'; case.mkdir()
            def save(path, value): path.write_text(json.dumps(value), encoding='utf-8')
            project = Path(__file__).resolve().parents[2]
            names = ('chunk_specialist_workflow/components/generation.py','receiver_kv_workflow/cache.py',
                     'receiver_kv_workflow/prompts.py','receiver_kv_workflow/speaker_blocks.py','receiver_kv_workflow/source_mapping.py')
            save(old/'manifest.json', dict(sample_ids=['a'], config=asdict(ReceiverKVConfig(device='cpu',vector_routing=True)),
                exclude_neighbor_blocks=0, code_sha256={n:sha256((project/n).read_bytes()) for n in names}))
            pinned_revision='a'*40
            save(old/'runtime.json', dict(model_revision=pinned_revision))
            save(case/'source.json', dict(id='a',transcript='Alpha: First.\nBeta: Last.'))
            base = dict(block_ids=[0],primary_block_ids=[0],supplement_block_ids=[],positions=[0,1,3],matches=[],added_positions=0)
            sem = dict(base,block_ids=[0,1],supplement_block_ids=[1],positions=[0,1,2,3])
            selection = dict(source_positions=4, selected_positions=[0,1,2,3],selected_sha256='saved')
            stage = dict(prompt_tokens=2,output_tokens=1,source_kv_positions=4)
            save(case/'selection.json', selection)
            save(case/'baseline.specialists.json',dict(stages=dict(local=stage,final=stage)))
            save(case/'baseline.metrics.json',dict(logical_input_tokens=132))
            save(case/'semantic.specialists.json',dict(windows=[sem],synthesis_selected_positions=[0,1,2,3],transcript_tokens=10))
            class FakeRunner:
                def __init__(self,config,progress):
                    assert not config.vector_routing
                    assert Path(config.model).name==pinned_revision
                    self.config=config;self.progress=progress;self.device=SimpleNamespace(type='cpu')
                    self.model=SimpleNamespace(config=SimpleNamespace(_commit_hash=None))
                    self.tokenizer=None;self.torch=SimpleNamespace(inference_mode=nullcontext)
                def _sync(self): pass
                def _tensor(self,ids): return ids
            args=SimpleNamespace(source_run=old,num_samples=0,output_dir=root/'new',random_seed=7,
                                 match_trials=10,plan_only=False,device='cpu',answer_key=None)
            detail=dict(stages=dict(local=stage,final=stage),pipeline_seconds=.1)
            with patch('experiment_3.random_control.saved_evidence',return_value=(
                    [[1],[2]],{0:'Alpha: Original first.',1:'Beta: Different last.'},[base],selection)), \
                 patch('huggingface_hub.snapshot_download', return_value=str(root/'snapshots'/pinned_revision)), \
                 patch('receiver_kv_workflow.engine.ReceiverKVRunner',FakeRunner), \
                 patch('receiver_kv_workflow.prompts.prompt_parts',return_value=([10,11,12,13],[],[0,3])), \
                 patch('receiver_kv_workflow.speaker_blocks.speaker_units',return_value=[[1],[2]]), \
                 patch('receiver_kv_workflow.cache.forward_tokens',return_value=SimpleNamespace(past_key_values=object())), \
                 patch('chunk_specialist_workflow.components.generation.run_specialists',return_value=('Random only',detail)) as generated:
                out=run(args)
            self.assertEqual(generated.call_count,1)
            self.assertEqual(generated.call_args.kwargs['windows'][0]['positions'],[0,1,2,3])
            self.assertTrue(generated.call_args.kwargs['semantic_context'])
            self.assertEqual((out/'00000/random.summary.txt').read_text().strip(),'Random only')
            self.assertEqual(json.loads((out/'status.json').read_text())['completed'],1)
            runtime=json.loads((out/'runtime.json').read_text())
            self.assertEqual(runtime['model_revision'],pinned_revision)
            self.assertIsNone(runtime['config_reported_revision'])
            self.assertFalse((out/'00000/baseline.summary.txt').exists())

    def test_recorded_model_revision_is_resolved_before_loading(self):
        from experiment_3.random_control import replay_model_snapshot
        revision = 'a'*40
        snapshot = f'/cache/models--example--model/snapshots/{revision}'
        with patch('huggingface_hub.snapshot_download', return_value=snapshot) as download:
            path, metadata = replay_model_snapshot('example/model', revision)
        self.assertEqual(path, snapshot)
        self.assertEqual(download.call_args.kwargs['revision'], revision)
        self.assertEqual(metadata['pinned_revision'], revision)

    def test_resolved_snapshot_must_match_recorded_commit(self):
        from experiment_3.random_control import replay_model_snapshot
        with patch('huggingface_hub.snapshot_download', return_value='/cache/snapshots/'+'b'*40):
            with self.assertRaisesRegex(ValueError, 'different snapshot'):
                replay_model_snapshot('example/model', 'a'*40)


if __name__=='__main__': unittest.main()
