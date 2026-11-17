import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from grounded_specialist_workflow.components.evidence import retrieve
from grounded_specialist_workflow.components.structure import parse_claims, parse_reviews, ROLES


def item():
    return dict(topic='plan', content='Review the plan', owners=[], deadline=None, status='requested')


class GroundingTest(unittest.TestCase):
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

    def test_role_cache_mutation_does_not_modify_source_or_other_role(self):
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

    def test_recency_only_reranks_semantic_candidates_and_keeps_anchor(self):
        # Last block is unrelated; weak time preference must not introduce it.
        scores = [.95, .1, .90, .89, -.5]
        base = retrieve(scores, candidate_pool=3, top_k=2, neighbors=0, recency_weight=0)
        late = retrieve(scores, candidate_pool=3, top_k=2, neighbors=0, recency_weight=.1)
        self.assertEqual(base['seed_block_ids'], [0, 2])
        self.assertEqual(late['seed_block_ids'], [0, 3])
        self.assertNotIn(4, late['evidence_block_ids'])
        expanded = retrieve(scores, candidate_pool=3, top_k=3, neighbors=1, recency_weight=0)
        self.assertEqual(expanded['evidence_block_ids'], [0, 1, 2, 3, 4])

    def test_claim_ids_and_evidence_citations(self):
        claims = parse_claims(json.dumps({'items': [item(), item()]}), 'actions')
        self.assertEqual([c['id'] for c in claims], ['actions:0', 'actions:1'])
        retrievals = {c['id']: {'evidence_block_ids': [1, 4]} for c in claims}
        rows = [dict(c, verdict='supported', reason='An explicit request', evidence_block_ids=[4]) for c in claims]
        self.assertEqual(len(parse_reviews(json.dumps({'items': rows}), claims, retrievals)), 2)
        rows[0]['evidence_block_ids'] = [8]
        with self.assertRaisesRegex(ValueError, 'not supplied'):
            parse_reviews(json.dumps({'items': rows}), claims, retrievals)
        with self.assertRaisesRegex(ValueError, 'every input claim'):
            parse_reviews('{"items": []}', claims, retrievals)

    def test_pipeline_independent_routes_full_source_retrieval_and_failure_audit(self):
        import torch
        from receiver_kv_workflow.engine import ReceiverKVConfig
        from grounded_specialist_workflow.components.pipeline import GroundingConfig, GroundedSpecialistRunner
        runner = GroundedSpecialistRunner.__new__(GroundedSpecialistRunner)
        runner.config = ReceiverKVConfig(device='cpu', vector_routing=True, role_budget_ratio=.25,
                                         vector_weight=0, speaker_neighbor_blocks=0)
        runner.grounding = GroundingConfig(specialist_workers=2, evidence_top_k=1,
                                           evidence_candidate_pool=2, evidence_neighbors=0)
        runner.torch, runner.device, runner.context_limit = torch, torch.device('cpu'), 1000
        runner.progress = lambda _: None
        runner._log_lock = threading.Lock()
        runner._metrics_lock = threading.Lock()
        runner.tokenizer = SimpleNamespace(encode=lambda text, **k: [list(ROLES).index(
            next(role for role in ROLES if text == requests[role]))])
        from grounded_specialist_workflow.components.prompts import ROLE_REQUESTS
        requests = ROLE_REQUESTS
        blocks = [dict(block_id=i, positions=list(range(1+4*i, 5+4*i)), relative_position=i/3,
                       text=f'Speaker {i}: text', speaker=str(i)) for i in range(4)]
        calls = []
        def encode(texts):
            # Query 0 deliberately retrieves block 3, absent from context role KV.
            if texts[0].startswith('Instruct:'):
                return torch.eye(4)[[3, 1, 2, 3]], {'embedding_input_tokens': 4}
            return torch.eye(4), {'embedding_input_tokens': 4}
        runner.semantic_router = SimpleNamespace(encode=encode)
        def scores(model, source, query, **kwargs):
            values = [0.0] * 17
            for i in blocks[query.item()]['positions']:
                values[i] = 1.0
            return values
        def decode(stage, ids, **kwargs):
            calls.append((stage, kwargs.get('positions')))
            runner.stages[stage] = {'prompt_tokens': 2, 'output_tokens': 2}
            if stage.startswith('specialist_'):
                return json.dumps({'items': [item()]})
            if stage.startswith('grounding_'):
                role = stage[len('grounding_'):]
                i = ROLES.index(role)
                return json.dumps({'items': [dict(item(), id=role+':0', category=role,
                    verdict='supported', reason='Evidence supports the request', evidence_block_ids=[3 if i == 0 else i])]})
            return 'Final summary'
        runner._decode = decode
        runner.model = object()
        target = 'grounded_specialist_workflow.components.pipeline.'
        with tempfile.TemporaryDirectory() as temp, \
             patch(target+'prompt_parts', return_value=(list(range(17)), [], [0])), \
             patch(target+'extraction_suffix', return_value=[1]), \
             patch(target+'source_blocks', return_value=blocks), \
             patch(target+'chat', return_value=[1]), \
             patch('receiver_kv_workflow.cache.forward_tokens', return_value=SimpleNamespace(past_key_values=object())), \
             patch('receiver_kv_workflow.cache.size_bytes', return_value=100), \
             patch('receiver_kv_workflow.scoring.query_scores', side_effect=scores):
            result = runner.run('meeting', 'transcript', artifact_dir=Path(temp)/'success')
            self.assertEqual(result['summary'], 'Final summary')
            routes = json.loads((Path(temp)/'success/routing.json').read_text())
            for i, role in enumerate(ROLES):
                self.assertEqual(routes[role]['selected_positions'], [0]+blocks[i]['positions'])
            retrievals = json.loads((Path(temp)/'success/retrieval.json').read_text())
            self.assertEqual(retrievals['context:0']['evidence_block_ids'], [3])
            self.assertEqual(result['extracted_claims'], 4)
            self.assertEqual(len(calls), 9)  # four specialists, four checks, one synthesis
            runner._decode = lambda *a, **k: 'invalid JSON'
            with self.assertRaises(ValueError):
                runner.run('meeting', 'transcript', artifact_dir=Path(temp)/'failure')
            status = json.loads((Path(temp)/'failure/status.json').read_text())
            self.assertEqual(status['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
