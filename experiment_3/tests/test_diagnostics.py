import unittest

from experiment_3.components.diagnostics import diagnose, render_diagnostics
from experiment_3.components.retrieval import supplement_windows
from experiment_3.tests.test_supplement import Vec


class DiagnosticsTest(unittest.TestCase):
    def setup_case(self):
        texts = {0: 'A: alpha batteries loan', 1: 'B: gamma interface screen',
                 2: 'C: beta charger loan', 3: 'D: delta controls screen'}
        blocks = [[i+1] for i in range(4)]
        vectors = {0: Vec(1, 0), 1: Vec(0, 1), 2: Vec(.8, .6), 3: Vec(.1, .995)}
        units = [dict(unit=i, source=[dict(character_range=[i*10, i*10+10])]) for i in range(4)]
        key = dict(opening=dict(character_range=[0, 10]), closing=dict(character_range=[20, 30]))
        windows = [dict(block_ids=[0, 1], positions=[0, 1, 2]),
                   dict(block_ids=[2, 3], positions=[0, 3, 4])]
        return texts, blocks, vectors, units, key, windows

    def test_target_ranks_explain_window_competition_and_do_not_change_retrieval(self):
        texts, blocks, vectors, units, key, windows = self.setup_case()
        selected = [0, 1, 2, 3, 4]
        before = supplement_windows(blocks, selected, windows, vectors, texts=texts, top_k=1, neighbor_radius=0)
        data = diagnose(units, key, blocks, texts, windows, vectors, selected, before, neighbor_radius=0)
        opening = data['directions'][0]
        self.assertEqual(opening['candidate_block_id'], 2)
        self.assertEqual(opening['raw_anchor_rank'], 1)
        # V2 loses this association to a larger absolute score from another anchor.
        # V3 retains the association with the stronger within-anchor advantage.
        self.assertEqual(opening['windows'][0]['absolute_v2_rank'], 2)
        self.assertEqual(opening['windows'][0]['window_distinct_rank'], 1)
        self.assertTrue(opening['windows'][0]['actually_supplemented'])
        after = supplement_windows(blocks, selected, windows, vectors, texts=texts, top_k=1, neighbor_radius=0)
        self.assertEqual(before, after)
        self.assertIn('raw rank=1/', render_diagnostics(data))

    def test_neighbor_exclusion_and_missing_selected_target_are_explicit(self):
        texts, blocks, vectors, units, key, windows = self.setup_case()
        selected = [0, 1, 2, 3, 4]
        augmented = supplement_windows(blocks, selected, windows, vectors, texts=texts,
                                       top_k=1, neighbor_radius=1)
        data = diagnose(units, key, blocks, texts, windows, vectors, selected, augmented, neighbor_radius=1)
        self.assertIn('within_neighbor_radius', data['directions'][0]['windows'][0]['exclusion_reasons'])
        selected = [0, 1, 2, 4]
        windows = [dict(block_ids=[0, 1], positions=[0, 1, 2])]
        augmented = supplement_windows(blocks, selected, windows, vectors, texts=texts, neighbor_radius=0)
        data = diagnose(units, key, blocks, texts, windows, vectors, selected, augmented, neighbor_radius=0)
        self.assertEqual(data['directions'][0]['status'], 'target_evidence_not_selected')

    def test_multiple_blocks_in_one_annotation_are_not_forced_to_endpoints(self):
        texts, blocks, vectors, units, key, windows = self.setup_case()
        key['opening']['character_range'] = [5, 15]
        selected = [0, 1, 2, 3, 4]
        augmented = supplement_windows(blocks, selected, windows, vectors, texts=texts, neighbor_radius=0)
        data = diagnose(units, key, blocks, texts, windows, vectors, selected, augmented, neighbor_radius=0)
        self.assertEqual(data['target_blocks']['opening'], [0, 1])
        self.assertEqual(len(data['directions']), 4)


if __name__ == '__main__':
    unittest.main()
