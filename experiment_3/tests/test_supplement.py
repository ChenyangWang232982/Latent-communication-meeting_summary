import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment_3.components.retrieval import content_terms, information_weights, primary_windows, supplement_windows
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
        out = supplement_windows(blocks, selected, base, vectors,
            texts={i: f'item{i}' for i in range(8)}, top_k=2, neighbor_radius=1)[0]
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
        texts = {i: f'item{i}' for i in range(6)}
        out = supplement_windows(blocks, range(7), base, vectors, texts=texts, top_k=1)[0]
        self.assertEqual(out['supplement_block_ids'], [4])
        self.assertEqual(out['matches'][0]['anchor_block_id'], 0)
        self.assertEqual(out['positions'], [0, 1, 5])
        self.assertEqual(supplement_windows(blocks, range(7), base, vectors, texts=texts, top_k=0)[0]['positions'], [0, 1])

    def test_partial_candidate_never_restores_unselected_positions(self):
        out = supplement_windows([[1], [2, 3]], [0, 1, 2],
            [dict(block_ids=[0], positions=[0, 1])], {0: Vec(1), 1: Vec(1)},
            texts={0: 'apples', 1: 'pears'}, top_k=5, neighbor_radius=0)[0]
        self.assertEqual(out['positions'], [0, 1, 2])

    def test_one_window_has_no_remote_candidates(self):
        base = [dict(block_ids=[0, 1], positions=[0, 1, 2])]
        out = supplement_windows([[1], [2]], [0, 1, 2], base, {0: Vec(1), 1: Vec(1)},
                                 texts={0: 'apples', 1: 'pears'})
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
        runner.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: f'item{ids[-1]}')
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

    def test_repeated_acknowledgments_do_not_beat_substantive_related_evidence(self):
        texts = {0: 'Alice: The loan batch will use disposable batteries without a dock.',
                 1: 'Alice: Yeah.', 2: 'Bob: Yeah.', 3: 'Carol: {annotation}',
                 4: 'Bob: The loan batch is confirmed with a charging dock instead.',
                 5: 'Carol: Yeah.', 6: 'Dan: Yeah.'}
        blocks = [[i+1] for i in range(7)]
        vectors = {i: Vec(1, 0) for i in texts}
        vectors[4] = Vec(.8, .6)
        base = [dict(block_ids=[0, 1], positions=[0, 1, 2])]
        row = supplement_windows(blocks, range(8), base, vectors, texts=texts,
                                 top_k=1, neighbor_radius=0)[0]
        self.assertEqual(row['supplement_block_ids'], [4])
        self.assertEqual(row['matches'][0]['anchor_block_id'], 0)
        _, weights = information_weights(texts)
        self.assertLess(weights[1], weights[0])
        self.assertEqual(content_terms('Someone: {nonverbal}\n'), ())
        self.assertNotEqual(content_terms('A: approve then cancel'), content_terms('B: cancel then approve'))

    def test_empty_or_identical_content_produces_no_supplement(self):
        rows = supplement_windows([[1], [2]], [0, 1, 2],
            [dict(block_ids=[0], positions=[0, 1])], {0: Vec(1), 1: Vec(1)},
            texts={0: 'Alice: Same wording.', 1: 'Bob: Same wording.'}, neighbor_radius=0)
        self.assertEqual(rows[0]['matches'], [])

    def test_retrieval_only_replays_saved_evidence_without_summary_model(self):
        from experiment_3.retrieve import run, saved_evidence
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'old'
            case = source / '00000'
            case.mkdir(parents=True)
            (source / 'manifest.json').write_text('{}', encoding='utf-8')
            audit = dict(sentence_coverage=[
                dict(unit=0, source=[dict(token_range=[1, 2], text='Alice: proposal batteries')]),
                dict(unit=1, source=[dict(token_range=[2, 3], text='Bob: revision charging dock')])])
            (case / 'selection.json').write_text(json.dumps(dict(routing_audit=audit,
                selected_positions=[0, 1, 2], selected_sha256='fixed')), encoding='utf-8')
            (case / 'baseline.specialists.json').write_text(json.dumps(dict(windows=[
                dict(block_ids=[0], positions=[0, 1])])), encoding='utf-8')
            self.assertEqual(saved_evidence(case)[0], [[1], [2]])
            router = SimpleNamespace(encode=lambda texts: ([Vec(1) for _ in texts],
                                     dict(embedding_input_tokens=6)))
            with patch('vector.routing.SemanticRouter', return_value=router):
                out = run(SimpleNamespace(run_dir=source, output_dir=root / 'new',
                    embedding_model='unused', device='cpu', supplement_blocks=1, exclude_neighbor_blocks=0))
            result = json.loads((out / '00000' / 'retrieval.json').read_text(encoding='utf-8'))
            self.assertEqual(result['matches'][0]['block_id'], 1)
            self.assertEqual(result['selected_sha256'], 'fixed')
            self.assertFalse(list(out.rglob('*.summary.txt')))


if __name__ == '__main__':
    unittest.main()
