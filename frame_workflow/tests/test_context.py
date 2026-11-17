import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from frame_workflow.components.context import anchor_transcript, fact_context


class FrameContextTest(unittest.TestCase):
    def test_anchors_keep_speaker_grouping_and_original_words(self):
        from receiver_kv_workflow.speaker_blocks import speaker_ranges
        original = 'Alice: First.\nAlice: More.\nBob: Later.\n'
        anchored = anchor_transcript(original)
        self.assertEqual(len(speaker_ranges(anchored)), 2)
        self.assertEqual(anchored.replace('[B0]\n', '').replace('[B1]\n', ''), original)

    def test_references_deduplicate_and_invalid_pointers_preserve_context(self):
        blocks = {'B0': {'positions': [1, 2]}, 'B1': {'positions': [3, 4]}, 'B2': {'positions': [5, 6]}}
        raw = json.dumps({'facts': [dict(statement='A', context_refs=['B2', 'B0']),
                                   dict(statement='B', context_refs=['B2'])]})
        facts, positions, fallback = fact_context(raw, blocks, [0], list(range(7)))
        self.assertEqual(positions, [0, 1, 2, 5, 6])
        self.assertIsNone(fallback)
        self.assertEqual(facts[0]['context_refs'], ['B2', 'B0'])
        for broken in ['not JSON', '{"facts": []}',
                       '{"facts": [{"statement": "A", "context_refs": ["B99"]}]}']:
            records, positions, fallback = fact_context(broken, blocks, [0], list(range(7)))
            self.assertEqual(positions, list(range(7)))
            self.assertEqual(records[0]['raw_report'], broken)
            self.assertIsNotNone(fallback)

    def test_two_stages_use_original_kv_and_pass_statement_reference_map(self):
        import torch
        from frame_workflow.components.generation import run_frame
        runner = SimpleNamespace(tokenizer=object(), torch=torch, model=object(),
            context_limit=100, _sync=lambda: None, _tensor=lambda ids: torch.tensor([ids]), progress=lambda _: None,
            config=SimpleNamespace(seed=12, temperature=0, top_p=1, top_k=0,
                repetition_penalty=1, review_seed=42, review_temperature=.3, review_top_p=.9))
        blocks = {f'B{i}': dict(speaker=str(i), positions=[1+2*i, 2+2*i]) for i in range(3)}
        raw = json.dumps({'facts': [dict(statement='A qualified claim', context_refs=['B0', 'B2'])]})
        requests, selected = [], []
        def request_ids(tokenizer, transcript, prefix, request):
            requests.append(request)
            return [9, 10]
        def cache(source, positions):
            self.assertEqual(source, 'immutable-source')
            selected.append(positions)
            return positions
        target = 'frame_workflow.components.generation.'
        with patch(target+'context_blocks', return_value=blocks), patch(target+'suffix', side_effect=request_ids), \
             patch(target+'select_cache', side_effect=cache), patch(target+'size_bytes', return_value=100), \
             patch(target+'generate', side_effect=[(raw, 20, False, {}), ('Final summary', 10, False, {})]):
            answer, result = run_frame(runner, 'source', [0]*7, [0], list(range(7)), 'immutable-source')
        self.assertEqual(answer, 'Final summary')
        self.assertEqual(selected, [list(range(7)), [0, 1, 2, 5, 6]])
        self.assertIn('A qualified claim', requests[1])
        self.assertIn('"B2"', requests[1])
        self.assertNotIn('"B1"', requests[1])
        self.assertEqual(result['stages']['frame_facts']['output_tokens'], 20)
        self.assertEqual(result['stages']['frame_summary']['output_tokens'], 10)


if __name__ == '__main__':
    unittest.main()
