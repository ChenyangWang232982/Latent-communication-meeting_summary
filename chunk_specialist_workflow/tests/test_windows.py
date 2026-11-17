import unittest
from types import SimpleNamespace
from unittest.mock import patch

from chunk_specialist_workflow.components.windows import selected_windows


class ChunkSpecialistTest(unittest.TestCase):
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
            config=ReceiverKVConfig(device='cpu', specialist_parts=2, specialist_overlap=0),
            context_limit=100, progress=lambda _: None, _sync=lambda: None,
            _tensor=lambda ids: torch.tensor([ids]))
        requests, caches = [], []
        def prompt(tokenizer, transcript, prefix, request):
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


if __name__ == '__main__':
    unittest.main()
