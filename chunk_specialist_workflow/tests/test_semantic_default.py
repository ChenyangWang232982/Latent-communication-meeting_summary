import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from receiver_kv_workflow.semantic import SemanticSummaryRunner
from receiver_kv_workflow.engine import ReceiverKVConfig


class SemanticDefaultTests(unittest.TestCase):
    def run_window(self, vector_routing):
        runner = SemanticSummaryRunner.__new__(SemanticSummaryRunner)
        runner.config = ReceiverKVConfig(vector_routing=vector_routing)
        runner.progress = lambda _: None
        runner.tokenizer = SimpleNamespace(decode=lambda *a, **kw: 'evidence')
        router = Mock()
        router.last_vectors = ['cached']
        router.encode.return_value = (['fresh'], {'embedding_input_tokens': 12})
        runner.post_router = router
        windows = [dict(block_ids=[0], positions=[0, 1])]
        augmented = [dict(windows[0], added_positions=0)]
        with patch('receiver_kv_workflow.semantic.selected_windows', return_value=windows), \
             patch('receiver_kv_workflow.semantic.supplement_windows', return_value=augmented) as supplement, \
             patch('receiver_kv_workflow.semantic.time.perf_counter', side_effect=[10, 10.5]), \
             patch('receiver_kv_workflow.semantic.run_specialists',
                   return_value=('summary', {'pipeline_seconds': 2.0})) as generate:
            answer, detail = runner._run_specialists('source', [0, 1], [0], [0, 1],
                object(), [[1]], transcript_tokens=1024, communication='kv')
        generate.assert_called_once()
        self.assertEqual(answer, 'summary')
        self.assertEqual(detail['pipeline_seconds'], 2.5)
        self.assertEqual(detail['workflow'], 'semantic_supplement_v3')
        self.assertTrue(generate.call_args.kwargs['semantic_context'])
        self.assertEqual(generate.call_args.kwargs['windows'], augmented)
        self.assertEqual(supplement.call_args.kwargs['top_k'], 2)
        self.assertEqual(supplement.call_args.kwargs['neighbor_radius'], 1)
        return router, detail

    def test_reuses_embeddings_and_generates_only_once(self):
        router, detail = self.run_window(True)
        router.encode.assert_not_called()
        self.assertEqual(detail['retrieval']['embedding_input_tokens'], 0)

    def test_encodes_selected_blocks_without_initial_vector_routing(self):
        router, detail = self.run_window(False)
        router.encode.assert_called_once_with(['evidence'])
        self.assertEqual(detail['retrieval']['embedding_input_tokens'], 12)


if __name__ == '__main__':
    unittest.main()
