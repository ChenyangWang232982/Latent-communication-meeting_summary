import unittest
from types import SimpleNamespace
from unittest.mock import patch

from chunk_specialist_workflow.components.windows import selected_windows


class ChunkSpecialistTest(unittest.TestCase):
    def test_text_control_uses_identical_windows_requests_and_decoding_without_source_cache(self):
        import torch
        from receiver_kv_workflow.engine import ReceiverKVConfig
        from chunk_specialist_workflow.components.generation import run_specialists
        runner = SimpleNamespace(model=object(), tokenizer=object(), torch=torch,
            config=ReceiverKVConfig(device='cpu', summary_thinking=False, specialist_parts=2, specialist_overlap=0),
            context_limit=100, progress=lambda _: None, _sync=lambda: None,
            _tensor=lambda ids: torch.tensor([ids]))
        target = 'chunk_specialist_workflow.components.generation.'
        requests, calls, evidence, events = [], [], [], []
        def prompt(tokenizer, transcript, prefix, request, **kwargs):
            requests.append(request)
            return [9]
        def generate(model, tokenizer, cache, ids, **kwargs):
            calls.append((cache, kwargs['absolute_start'], kwargs['temperature'], kwargs['top_p']))
            kwargs['on_progress']({'phase': 'decode', 'tokens': 2, 'finished': True})
            return 'Local notes', 2, False, dict(answer_tokens=1, terminal_tokens=1)
        def excerpts(tokenizer, transcript, positions):
            evidence.append(positions)
            return 'Alice: original selected evidence', 4, 1
        with patch(target+'suffix', side_effect=prompt), patch(target+'generate', side_effect=generate), \
             patch(target+'select_cache', return_value='copied KV') as select, \
             patch(target+'size_bytes', return_value=100), \
             patch(target+'selected_evidence_excerpts', side_effect=excerpts):
            args = (runner, 'source', [0]*7, [0], [0,1,2,5,6], object(), [[1,2], [3,4], [5,6]])
            _, kv = run_specialists(*args)
            _, text = run_specialists(*args, communication='text', on_progress=events.append)
        self.assertEqual(kv['windows'], text['windows'])
        self.assertEqual(requests[:3], requests[3:])
        self.assertEqual(evidence, [[0,1,2], [0,5,6], [0,1,2,5,6]])
        self.assertEqual(kv['synthesis_selected_positions'], [0,1,2,5,6])
        self.assertNotIn('CURRENT MEETING PART', requests[0])
        self.assertIn('window 1 of 2', requests[0])
        self.assertIn('window 2 of 2', requests[1])
        self.assertEqual(select.call_count, 3)
        self.assertEqual(events[-1]['stage'], 'text_chunk_summary')
        self.assertTrue(all(cache is None and start == 0 for cache, start, _, _ in calls[3:]))
        self.assertEqual([c[2:] for c in calls[:3]], [c[2:] for c in calls[3:]])
        self.assertTrue(all(v['source_kv_bytes'] == 0 and v['source_text_bytes'] > 0
                            for v in text['stages'].values()))

    def test_only_selected_rows_covered_and_overlap_advances(self):
        blocks = [list(range(1+i*10, 11+i*10)) for i in range(64)]
        selected = [0]+[p for i, unit in enumerate(blocks) if i != 10 for p in unit]
        windows = list(selected_windows(blocks, selected, [0], parts=8, overlap=.25))
        self.assertEqual({p for window in windows for p in window['positions']}, set(selected))
        for window in windows:
            self.assertEqual(window['positions'], sorted(set(window['positions'])))
            self.assertNotIn(10, window['block_ids'])
        self.assertTrue(set(windows[0]['block_ids']) & set(windows[1]['block_ids']))
        long = list(selected_windows([list(range(1,1001)), [1001]], list(range(1002)), [0]))
        self.assertEqual(len(long), 2)
        self.assertEqual(len(long[0]['positions']), 1001)

    def test_independent_writable_caches_and_ordered_synthesis(self):
        import torch
        from transformers.cache_utils import DynamicCache
        from receiver_kv_workflow.cache import pairs
        from receiver_kv_workflow.engine import ReceiverKVConfig
        from chunk_specialist_workflow.components.generation import run_specialists
        source = DynamicCache()
        tensor = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7, 1)
        source.update(tensor.clone(), tensor.clone(), 0)
        runner = SimpleNamespace(model=object(), tokenizer=object(), torch=torch,
            config=ReceiverKVConfig(device='cpu', summary_thinking=False, specialist_parts=2, specialist_overlap=0),
            context_limit=100, progress=lambda _: None, _sync=lambda: None,
            _tensor=lambda ids: torch.tensor([ids]))
        requests, caches = [], []
        def prompt(tokenizer, transcript, prefix, request, **kwargs):
            requests.append(request)
            return [9]
        def generate(model, tokenizer, cache, ids, **kwargs):
            caches.append(pairs(cache)[0][0].flatten().tolist())
            pairs(cache)[0][0].fill_(-1)
            return f'local report {len(caches)}', 2, False, dict(answer_tokens=1, terminal_tokens=1)
        target = 'chunk_specialist_workflow.components.generation.'
        with patch(target+'suffix', side_effect=prompt), patch(target+'generate', side_effect=generate):
            answer, result = run_specialists(runner, 'source', [0]*7, [0], [0,1,2,5,6], source,
                                            [[1,2], [3,4], [5,6]])
        self.assertEqual(caches, [[0,1,2], [0,5,6], [0,1,2,5,6]])
        self.assertTrue(torch.equal(pairs(source)[0][0], tensor))
        self.assertEqual(answer, 'local report 3')
        self.assertEqual(len(result['reports']), 2)
        self.assertLess(requests[-1].index('local report 1'), requests[-1].index('local report 2'))
        self.assertEqual(sum(x['output_tokens'] for x in result['stages'].values()), 6)

    def test_thinking_only_at_synthesis_and_reasoning_excluded_from_answer(self):
        import torch
        from receiver_kv_workflow.engine import ReceiverKVConfig
        from chunk_specialist_workflow.components.generation import run_specialists
        pieces = {1: 'Reasoning', 2: '</think>', 3: 'Final answer', 4: '<eos>'}
        tokenizer = SimpleNamespace(encode=lambda text, **kw: [2],
            decode=lambda ids, **kw: ''.join(pieces[i] for i in ids if i not in (2,4)))
        runner = SimpleNamespace(model=object(), tokenizer=tokenizer, torch=torch,
            config=ReceiverKVConfig(device='cpu', summary_thinking=True), context_limit=100,
            progress=lambda _: None, _sync=lambda: None, _tensor=lambda ids: torch.tensor([ids]))
        flags = []
        def prompt(*args, enable_thinking=False):
            flags.append(enable_thinking)
            return [9]
        def generate(*args, **kwargs):
            if kwargs['on_token'] is not None:
                for index, token in enumerate([1,2,3,4]):
                    kwargs['on_token'](None, token, index)
                return 'Reasoning and final answer', 4, False, dict(answer_tokens=3, terminal_tokens=1)
            return 'Local notes', 2, False, dict(answer_tokens=1, terminal_tokens=1)
        target = 'chunk_specialist_workflow.components.generation.'
        with patch(target+'suffix', side_effect=prompt), patch(target+'generate', side_effect=generate), \
             patch(target+'select_cache', return_value=object()), patch(target+'size_bytes', return_value=100):
            answer, result = run_specialists(runner, 'source', [0]*2, [0], [0,1], object(), [[1]])
        self.assertEqual(flags, [False, True])
        self.assertEqual(answer, 'Final answer')
        self.assertEqual(result['thinking_outputs'], {'chunk_summary': 'Reasoning'})
        counts = result['stages']['chunk_summary']['generation_counts']
        self.assertEqual(counts, dict(thinking_tokens=1, delimiter_tokens=1, answer_tokens=1,
                                     terminal_tokens=1, thinking_complete=True))
        self.assertEqual(sum(counts[k] for k in ('thinking_tokens','delimiter_tokens','answer_tokens','terminal_tokens')),
                         result['stages']['chunk_summary']['output_tokens'])


if __name__ == '__main__':
    unittest.main()
