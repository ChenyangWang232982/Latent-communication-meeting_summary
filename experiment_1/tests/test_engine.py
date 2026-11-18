"""Exercise shared routing and both remaining engine branches without model weights."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class EngineTest(unittest.TestCase):
    def test_one_prefill_and_identical_selected_context_for_both_variants(self):
        import torch
        from receiver_kv_workflow.engine import ReceiverKVConfig, ReceiverKVRunner
        runner = ReceiverKVRunner.__new__(ReceiverKVRunner)
        runner.config = ReceiverKVConfig(device='cpu', role_budget_ratio=1, speaker_neighbor_blocks=0)
        runner.device = torch.device('cpu')
        runner.torch = torch
        runner.context_limit = 100
        runner.tokenizer = SimpleNamespace(encode=lambda *a, **k: [5])
        runner.model = object()
        runner.semantic_router = None
        runner.progress = lambda _: None
        source = object()
        calls, events = [], []

        def specialists(runner, transcript, prefix, protected, selected, incoming, blocks, **kwargs):
            calls.append((kwargs['communication'], list(selected), incoming))
            stage = dict(source_kv_positions=len(selected) if kwargs['communication'] == 'kv' else 0,
                source_kv_bytes=128 if kwargs['communication'] == 'kv' else 0, source_text_bytes=9,
                current_part_text_bytes=5, prompt_tokens=3, output_tokens=2, output_cap_hit=False,
                generation_counts={'answer_tokens': 1, 'terminal_tokens': 1}, seconds=.01)
            return 'Final summary', dict(reports=[{'report': 'Local notes'}], windows=[{'positions': [1,2]}],
                stages={'local': stage, 'final': stage}, pipeline_seconds=.02, text_payload_bytes=11)

        with patch('receiver_kv_workflow.engine.prompt_parts', return_value=([0,1,2,3], [8], [0])), \
             patch('receiver_kv_workflow.speaker_blocks.speaker_units', return_value=[[1,2,3]]), \
             patch('receiver_kv_workflow.cache.forward_tokens', return_value=SimpleNamespace(past_key_values=source)) as prefill, \
             patch('receiver_kv_workflow.cache.size_bytes', return_value=128), \
             patch('receiver_kv_workflow.scoring.query_scores', return_value=[1.,2.,3.,4.]), \
             patch('chunk_specialist_workflow.components.generation.run_specialists', side_effect=specialists):
            result = runner.run('one', 'Alice: source', variants=('chunk_kv', 'chunk_text'),
                                on_variant=lambda *event: events.append(event))
        self.assertEqual(prefill.call_count, 1)
        self.assertEqual(calls, [('kv', [0,1,2,3], source), ('text', [0,1,2,3], source)])
        self.assertEqual(list(result['answers']), ['chunk_kv', 'chunk_text'])
        self.assertEqual(events, [('chunk_kv','start'), ('chunk_kv','end'), ('chunk_text','start'), ('chunk_text','end')])
        for measurement in result['variants'].values():
            self.assertEqual(measurement['token_accounting']['specialist_prompt_tokens'], 3)
            self.assertEqual(measurement['token_accounting']['source_prefill_tokens'], 4)
        with self.assertRaisesRegex(ValueError, 'variants'):
            runner.run('one', 'Alice: source', variants=('routed_kv',))
