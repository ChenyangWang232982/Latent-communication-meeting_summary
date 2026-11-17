import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from grounded_specialist_workflow.components.structure import parse_topics


class TopicWorkflowTest(unittest.TestCase):
    def test_discovery_windows_cover_source_and_reuse_topic_coordinates(self):
        from grounded_specialist_workflow.components.discovery import discovery_windows, merge_topics
        blocks = [dict(block_id=i, positions=list(range(i*10, (i+1)*10)), text=str(i)) for i in range(64)]
        windows = list(discovery_windows(blocks))
        self.assertEqual(len(windows[0]), 8)
        self.assertEqual([b['block_id'] for b in windows[1]][:2], [6, 7])
        self.assertEqual({b['block_id'] for w in windows for b in w}, set(range(64)))
        catalogue = []
        merge_topics(catalogue, json.dumps({'topics': [dict(id=None, title='Plan', scope='Plan choice',
            block_ids=[6, 7])]}), windows[0])
        merge_topics(catalogue, json.dumps({'topics': [dict(id='topic_000', title='Plan revised',
            block_ids=[6, 7, 8, 9])]}), windows[1])
        self.assertEqual(len(catalogue), 1)
        self.assertEqual(catalogue[0]['source_block_ids'], [6, 7, 8, 9])
        # A long indivisible turn remains intact, and the next window still advances.
        oversized = [dict(block_id=i, positions=list(range(size))) for i, size in enumerate([1000, 1, 1])]
        self.assertEqual([b['block_id'] for w in discovery_windows(oversized) for b in w], [0, 1, 2])

    def test_thinking_decode_keeps_json_and_counts_all_generated_tokens(self):
        import torch
        from receiver_kv_workflow.engine import ReceiverKVConfig
        from grounded_specialist_workflow.components.pipeline import GroundedSpecialistRunner, SpecialistConfig
        runner = GroundedSpecialistRunner.__new__(GroundedSpecialistRunner)
        runner.config = ReceiverKVConfig(device='cpu')
        runner.specialists = SpecialistConfig(specialist_thinking=True)
        runner.torch, runner.device, runner.context_limit = torch, torch.device('cpu'), 100
        runner.progress = lambda _: None
        runner._log_lock, runner._metrics_lock = threading.Lock(), threading.Lock()
        runner.stages, runner.model = {}, object()
        pieces = {1: 'reason', 2: '\n', 3: '</think>', 4: '{"reports": []}', 5: '<eos>'}
        runner.tokenizer = SimpleNamespace(encode=lambda text, **kw: [3],
            decode=lambda ids, **kw: ''.join(pieces[i] for i in ids if not kw.get('skip_special_tokens') or i not in (3, 5)))
        def generate(*args, **kwargs):
            for index, token in enumerate([1, 2, 3, 4, 5]):
                kwargs['on_token'](None, token, index)
            return 'unparsed reasoning and JSON', 5, False, {'answer_tokens': 4, 'terminal_tokens': 1}
        with tempfile.TemporaryDirectory() as temp, patch('receiver_kv_workflow.cache.generate', side_effect=generate):
            runner.artifacts = Path(temp)
            answer = runner._decode('specialist_group_000', [1])
            self.assertEqual(json.loads(answer), {'reports': []})
            metrics = runner.stages['specialist_group_000']
            self.assertEqual(metrics['output_tokens'], 5)
            self.assertEqual(metrics['generation_counts'], dict(thinking_tokens=2, delimiter_tokens=1,
                answer_tokens=1, terminal_tokens=1, thinking_complete=True))
            self.assertEqual((Path(temp)/'specialist_group_000.thinking.txt').read_text(), 'reason\n')
            self.assertIn('</think>', (Path(temp)/'specialist_group_000.raw.txt').read_text())

    def test_decode_generator_is_independent_of_global_rng(self):
        import torch
        from receiver_kv_workflow.cache import generate_from_prefill
        logits = torch.zeros(1, 1, 8)
        output = SimpleNamespace(logits=logits, past_key_values=None)
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=99))
        tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda ids, **kw: str(ids))
        def run(perturb):
            return generate_from_prefill(model, tokenizer, output, next_position=1, context_limit=25,
                temperature=.7, generator=torch.Generator().manual_seed(12),
                on_token=(lambda *a: torch.manual_seed(999)) if perturb else None)
        with patch('receiver_kv_workflow.cache.forward_tokens', return_value=output):
            self.assertEqual(run(False), run(True))

    def test_topic_cache_mutation_does_not_modify_source_or_other_topic(self):
        import torch
        from transformers.cache_utils import DynamicCache
        from receiver_kv_workflow.cache import select_cache, pairs
        source = DynamicCache()
        tensor = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
        source.update(tensor.clone(), tensor.clone(), 0)
        first, second = select_cache(source, [0, 2, 3]), select_cache(source, [0, 3, 4])
        pairs(first)[0][0].fill_(-1)
        self.assertTrue(torch.equal(pairs(source)[0][0], tensor))
        self.assertTrue(torch.equal(pairs(second)[0][0], tensor[:, :, [0, 3, 4]]))

    def test_topics_are_dynamic_and_block_markers_preserve_speaker_grouping(self):
        from grounded_specialist_workflow.components.evidence import annotate_transcript
        from receiver_kv_workflow.speaker_blocks import speaker_ranges
        topics = parse_topics(json.dumps({'topics': [{'title': f'Issue {i}'} for i in range(7)]}))
        self.assertEqual(len(topics), 7)
        self.assertEqual(topics[-1]['id'], 'topic_006')
        source = 'Alice: Hello.\nAlice: More.\nBob: Later.\n'
        annotated = annotate_transcript(source)
        self.assertEqual(len(speaker_ranges(annotated)), 2)
        self.assertEqual(annotated.replace('[B0]\n', '').replace('[B1]\n', ''), source)

    def test_dynamic_topic_queue_chronological_kv_and_ordered_synthesis(self):
        import torch
        import time
        from receiver_kv_workflow.engine import ReceiverKVConfig
        from grounded_specialist_workflow.components.pipeline import SpecialistConfig, GroundedSpecialistRunner
        runner = GroundedSpecialistRunner.__new__(GroundedSpecialistRunner)
        runner.config = ReceiverKVConfig(device='cpu', vector_routing=True, role_budget_ratio=.5,
                                         vector_weight=0, speaker_neighbor_blocks=0)
        runner.specialists = SpecialistConfig(specialist_workers=2)
        runner.torch, runner.device, runner.context_limit = torch, torch.device('cpu'), 1000
        runner.progress = lambda _: None
        runner._log_lock = threading.Lock()
        runner._metrics_lock = threading.Lock()
        runner.tokenizer = SimpleNamespace(encode=lambda text, **k: [int(text)])
        blocks = [dict(block_id=i, positions=list(range(1+4*i, 5+4*i)), relative_position=i/3,
                       text=f'Speaker {i}: text', speaker=str(i)) for i in range(4)]
        calls, encoding_calls = [], []
        def encode(texts):
            encoding_calls.append(texts)
            return torch.eye(4).repeat((len(texts)+3)//4, 1)[:len(texts)], {'embedding_input_tokens': len(texts)}
        runner.semantic_router = SimpleNamespace(encode=encode)
        def scores(model, source, query, **kwargs):
            values = [0.0] * 17
            for block_id in (query.item() % 4, (query.item()+2) % 4):
                for i in blocks[block_id]['positions']:
                    values[i] = 1.0
            return values
        def decode(stage, ids, **kwargs):
            calls.append((stage, kwargs.get('positions')))
            runner.stages[stage] = {'prompt_tokens': 2, 'output_tokens': 2}
            if stage.startswith('topic_discovery_'):
                return json.dumps({'topics': [{'id': None if stage.endswith('_000') else f'topic_{i:03d}',
                    'title': str(i), 'scope': f'Issue {i}',
                    'block_ids': [i % 4]} for i in range(topic_count)]})
            if stage.startswith('specialist_'):
                if stage == 'specialist_group_000':
                    time.sleep(.03)  # Finish later than another queued topic.
                groups = json.loads((runner.artifacts/'groups.json').read_text())
                group = next(g for g in groups if stage == 'specialist_'+g['id'])
                return json.dumps({'reports': [dict(topic_id=t['id'], topic=t['title'],
                    key_facts=['Explicit detail'], discussion='Initial proposal then deferral',
                    decisions=[], actions=[], open_issues=['Pending'], final_status='deferred')
                    for t in group['topics']]})
            return 'Final summary'
        runner._decode = decode
        runner.model = object()
        target = 'grounded_specialist_workflow.components.pipeline.'
        with tempfile.TemporaryDirectory() as temp, \
             patch(target+'annotate_transcript', side_effect=lambda text: text), \
             patch(target+'prompt_parts', return_value=(list(range(17)), [], [0])), \
             patch(target+'source_suffix', return_value=[1]), \
             patch(target+'source_blocks', return_value=blocks), \
             patch(target+'discovery_windows', return_value=[blocks]) as window_mock, \
             patch(target+'topic_query', side_effect=lambda topic: topic['title']), \
             patch(target+'chat', return_value=[1]), \
             patch('receiver_kv_workflow.cache.forward_tokens', return_value=SimpleNamespace(past_key_values=object())), \
             patch('receiver_kv_workflow.cache.size_bytes', return_value=100), \
             patch('receiver_kv_workflow.scoring.query_scores', side_effect=scores):
            for topic_count, width in ((0, 0), (3, 0), (5, 2), (3, 1)):
                with self.subTest(topics=topic_count, width=width):
                    runner.specialists = SpecialistConfig(specialist_workers=2, topics_per_specialist=width)
                    window_count = 2 if topic_count == 3 else 1
                    window_mock.return_value = [blocks] * window_count
                    calls.clear()
                    encoding_calls.clear()
                    folder = Path(temp)/f'{topic_count}_{width}'
                    result = runner.run('meeting', 'transcript', artifact_dir=folder)
                    self.assertEqual(result['summary'], 'Final summary')
                    self.assertEqual(calls[0][0], 'topic_discovery_000')
                    self.assertIsNone(calls[0][1])
                    routes = json.loads((folder/'routing.json').read_text())
                    for i in range(topic_count):
                        expected = sorted([0]+blocks[i%4]['positions']+blocks[(i+2)%4]['positions'])
                        self.assertEqual(routes[f'topic_{i:03d}']['selected_positions'], expected)
                    merged = json.loads((folder/'topic_reports.json').read_text())
                    topics = json.loads((folder/'topics.json').read_text())
                    self.assertEqual([c['topic_id'] for c in merged], [t['id'] for t in topics])
                    self.assertEqual([min(t['source_block_ids']) for t in topics],
                                     sorted(i % 4 for i in range(topic_count)))
                    self.assertFalse((folder/'retrieval.json').exists())
                    self.assertFalse(result['evidence_review'])
                    self.assertEqual(result['topic_count'], topic_count)
                    self.assertEqual(result['discovery_window_count'], window_count)
                    self.assertTrue(all(positions is None for name, positions in calls
                                        if name.startswith('topic_discovery_')))
                    if window_count > 1:
                        prompt = (folder/'topic_discovery_001.input.txt').read_text()
                        self.assertIn('topic_000', prompt)
                    groups = json.loads((folder/'groups.json').read_text())
                    count = (topic_count + width - 1)//width if width else int(topic_count > 0)
                    self.assertEqual(len(groups), count)
                    self.assertEqual(result['group_count'], count)
                    for group in groups:
                        expected = sorted({p for t in group['topics']
                                           for p in routes[t['id']]['selected_positions']})
                        self.assertEqual(group['selected_positions'], expected)
                        self.assertIn(('specialist_'+group['id'], expected), calls)
                    self.assertEqual(len(calls), window_count+count+1)
                    self.assertEqual(result['total_decoder_input_tokens'], 17+topic_count+2*(window_count+count+1))
                    self.assertEqual(result['total_decoder_output_tokens'], 2*(window_count+count+1))
                    self.assertEqual(len(encoding_calls), 1)
                    self.assertEqual(result['embedding_input_tokens'], 4+topic_count)
            runner._decode = lambda *a, **k: 'invalid JSON'
            with self.assertRaises(ValueError):
                runner.run('meeting', 'transcript', artifact_dir=Path(temp)/'failure')
            self.assertEqual(json.loads((Path(temp)/'failure/status.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
