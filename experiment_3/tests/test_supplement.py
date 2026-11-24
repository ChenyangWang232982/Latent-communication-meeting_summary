import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiment_3.components.retrieval import primary_windows, supplement_windows
from experiment_3.components.reporting import measurement
from experiment_3.components.runner import SupplementRunner
from receiver_kv_workflow.engine import ReceiverKVConfig


class Vec:
    def __init__(self, *values):
        self.values = values

    def __matmul__(self, other):
        return sum(a*b for a, b in zip(self.values, other.values))


class SupplementTest(unittest.TestCase):
    def test_retrieval_excludes_neighbors_dropped_blocks_and_deduplicates(self):
        blocks = [[i*2+1, i*2+2] for i in range(8)]
        selected = [0, *[p for i, block in enumerate(blocks) if i != 7 for p in block]]
        base = [dict(block_ids=[2, 3], positions=[0, *blocks[2], *blocks[3]])]
        vectors = {i: Vec(1, 0) for i in range(8)}
        out = supplement_windows(blocks, selected, base, vectors, top_k=2, neighbor_radius=1)[0]
        self.assertEqual(out['supplement_block_ids'], [0, 5])
        self.assertEqual(out['added_positions'], 4)
        self.assertEqual(out['positions'], sorted(set(out['positions'])))
        self.assertLessEqual(set(out['positions']), set(selected))
        self.assertEqual(base[0]['block_ids'], [2, 3])
        self.assertNotIn('supplement_block_ids', base[0])

    def test_similarity_not_source_order_controls_selection(self):
        blocks = [[i+1] for i in range(6)]
        base = [dict(block_ids=[0], positions=[0, 1])]
        vectors = {0: Vec(1, 0), 1: Vec(1, 0), 2: Vec(.1, .9),
                   3: Vec(.2, .8), 4: Vec(.9, .1), 5: Vec(.8, .2)}
        out = supplement_windows(blocks, range(7), base, vectors, top_k=1)[0]
        self.assertEqual(out['supplement_block_ids'], [4])
        self.assertEqual(out['matches'][0]['anchor_block_id'], 0)
        self.assertEqual(out['positions'], [0, 1, 5])
        self.assertEqual(supplement_windows(blocks, range(7), base, vectors, top_k=0)[0]['positions'], [0, 1])

    def test_partial_candidate_never_restores_unselected_positions(self):
        out = supplement_windows([[1], [2, 3]], [0, 1, 2],
            [dict(block_ids=[0], positions=[0, 1])], {0: Vec(1), 1: Vec(1)},
            top_k=5, neighbor_radius=0)[0]
        self.assertEqual(out['positions'], [0, 1, 2])

    def test_one_window_has_no_remote_candidates(self):
        base = [dict(block_ids=[0, 1], positions=[0, 1, 2])]
        out = supplement_windows([[1], [2]], [0, 1, 2], base, {0: Vec(1), 1: Vec(1)})
        self.assertEqual(out[0]['supplement_block_ids'], [])
        self.assertEqual(primary_windows(base)[0]['primary_block_ids'], [0, 1])

    def test_paired_generation_keeps_selection_seeds_and_final_input(self):
        runner = SupplementRunner.__new__(SupplementRunner)
        runner.config = ReceiverKVConfig(vector_routing=True)
        runner.post_router = SimpleNamespace(last_vectors={0: Vec(1), 1: Vec(1), 2: Vec(1)})
        runner.supplement_blocks = 1
        runner.neighbor_radius = 0
        runner.pair_index = 0
        runner.progress = lambda _: None
        selected = [0, 1, 2, 3]
        seen = []
        def fake(r, transcript, prefix, protected, chosen, source, blocks, **kwargs):
            self.assertIs(chosen, selected)
            self.assertTrue(kwargs['semantic_context'])
            self.assertEqual(r.config.seed, 12)
            seen.append(kwargs['windows'])
            return 'answer', dict(windows=kwargs['windows'])
        windows = [dict(block_ids=[0], positions=[0, 1]), dict(block_ids=[2], positions=[0, 3])]
        with patch('experiment_3.components.runner.selected_windows', return_value=windows), \
             patch('experiment_3.components.runner.run_specialists', side_effect=fake):
            runner._run_specialists('text', [0, 1, 2, 3], [0], selected, object(), [[1], [2], [3]],
                                    transcript_tokens=2048, communication='kv')
        self.assertEqual(runner.paired['execution_order'], ['baseline', 'semantic'])
        self.assertEqual(seen[0][0]['supplement_block_ids'], [])
        self.assertEqual(seen[1][0]['supplement_block_ids'], [1])
        self.assertEqual(runner.paired['retrieval']['embedding_input_tokens'], 0)

    def test_costs_count_shared_prefill_once_per_standalone_condition(self):
        stage = dict(prompt_tokens=10, output_tokens=5, source_kv_positions=20, source_kv_bytes=100)
        result = dict(source_positions=100, roles={'context': dict(query_tokens=4)},
                      prefill_seconds=1, selection_seconds=2, vector_routing=None)
        detail = dict(stages={'local': stage, 'final': stage}, pipeline_seconds=3)
        row = measurement(result, detail, extra_seconds=.5, extra_encoder_tokens=30)
        self.assertEqual(row['logical_total_tokens'], 134)
        self.assertEqual(row['total_embedding_input_tokens'], 30)
        self.assertEqual(row['standalone_pipeline_seconds'], 6.5)

    def test_generation_uses_augmented_local_cache_but_unchanged_final_cache(self):
        from chunk_specialist_workflow.components.generation import run_specialists
        runner = SimpleNamespace(config=ReceiverKVConfig(summary_thinking=False),
            tokenizer=object(), progress=lambda _: None, _sync=lambda: None,
            torch=SimpleNamespace(manual_seed=lambda _: None), model=object(),
            _tensor=lambda ids: ids, context_limit=100)
        windows = [dict(block_ids=[0, 2], primary_block_ids=[0], supplement_block_ids=[2],
                        positions=[0, 1, 3])]
        used = []
        requests = []
        def cache(source, positions):
            used.append(list(positions))
            return object()
        def prompt(tokenizer, transcript, prefix, request, **kwargs):
            requests.append(request)
            return [9]
        with patch('chunk_specialist_workflow.components.generation.select_cache', side_effect=cache), \
             patch('chunk_specialist_workflow.components.generation.size_bytes', return_value=10), \
             patch('chunk_specialist_workflow.components.generation.suffix', side_effect=prompt), \
             patch('chunk_specialist_workflow.components.generation.generate',
                   return_value=('notes', 2, False, dict(terminal_tokens=1))):
            answer, detail = run_specialists(runner, 'source', [0, 1, 2, 3], [0],
                [0, 1, 2, 3], object(), [[1], [2], [3]], transcript_tokens=1024,
                windows=windows, semantic_context=True)
        self.assertEqual(used, [[0, 1, 3], [0, 1, 2, 3]])
        self.assertIn('Additional related block IDs: [2]', requests[0])
        self.assertIn('not report order alone', requests[1])
        self.assertEqual(detail['reports'][0]['primary_block_ids'], [0])


if __name__ == '__main__':
    unittest.main()
