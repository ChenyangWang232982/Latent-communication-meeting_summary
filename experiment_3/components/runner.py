"""One immutable source selection, two independent sets of specialist generations."""
import time

from receiver_kv_workflow.engine import ReceiverKVRunner
from chunk_specialist_workflow.components.generation import run_specialists
from chunk_specialist_workflow.components.windows import selected_windows, target_parts
from .retrieval import primary_windows, supplement_windows


class SupplementRunner(ReceiverKVRunner):
    def __init__(self, config, *, supplement_blocks=2, neighbor_radius=1, progress=None):
        super().__init__(config, progress)
        self.supplement_blocks = supplement_blocks
        self.neighbor_radius = neighbor_radius
        self.pair_index = 0
        if self.semantic_router is None:
            from vector.routing import SemanticRouter
            self.progress('loading embedding model for post-selection retrieval')
            self.semantic_router = SemanticRouter(config.embedding_model, config.device)
            # Parent selector must remain attention-only when --vector-routing is absent.
        self.post_router = self.semantic_router
        if not config.vector_routing:
            self.semantic_router = None

    def run(self, *args, **kwargs):
        if hasattr(self.post_router, 'last_vectors'):
            del self.post_router.last_vectors
        return super().run(*args, **kwargs)

    def _run_specialists(self, transcript, prefix, protected, selected, source, blocks, **kwargs):
        tokens = kwargs['transcript_tokens']
        base = primary_windows(list(selected_windows(blocks, selected, protected,
            target_parts(tokens), self.config.specialist_overlap)))
        self.progress('post-selection semantic matching started')
        start = time.perf_counter()
        if self.config.vector_routing and hasattr(self.post_router, 'last_vectors'):
            vectors = self.post_router.last_vectors
            encoder = dict(embedding_input_tokens=0, reused_selection_embeddings=True)
        else:
            # Encode only selected blocks. Zero rows for unused blocks are never candidates.
            ids = sorted({i for w in base for i in w['block_ids']})
            retained = set(selected)
            texts = [self.tokenizer.decode([prefix[p] for p in blocks[i] if p in retained], skip_special_tokens=True)
                     for i in ids]
            encoded, encoder = self.post_router.encode(texts)
            vectors = {i: encoded[j] for j, i in enumerate(ids)}
            encoder['reused_selection_embeddings'] = False
        augmented = supplement_windows(blocks, selected, base, vectors,
            top_k=self.supplement_blocks, neighbor_radius=self.neighbor_radius)
        retrieval_seconds = time.perf_counter() - start
        self.progress(f'post-selection matching complete; {sum(w["added_positions"] for w in augmented)} '
                      'additional local KV position events')
        order = ['baseline', 'semantic'] if self.pair_index % 2 == 0 else ['semantic', 'baseline']
        self.pair_index += 1
        conditions = {}
        # Both conditions receive identical instructions, seeds and global final evidence.
        for condition in order:
            self.progress(f'{condition}: independent selected-KV window run')
            callback = kwargs.get('on_generation')
            def snapshot(event, condition=condition):
                if callback:
                    callback(dict(event, stage=condition + '_' + event['stage']))
            answer, detail = run_specialists(self, transcript, prefix, protected, selected, source, blocks,
                **dict(kwargs, on_generation=snapshot),
                windows=base if condition == 'baseline' else augmented, semantic_context=True)
            conditions[condition] = dict(answer=answer, detail=detail)
        self.paired = dict(conditions=conditions, execution_order=order,
            retrieval=dict(**encoder, seconds=retrieval_seconds),
            supplement_blocks=self.supplement_blocks, excluded_neighbor_radius=self.neighbor_radius)
        return conditions['semantic']['answer'], conditions['semantic']['detail']
