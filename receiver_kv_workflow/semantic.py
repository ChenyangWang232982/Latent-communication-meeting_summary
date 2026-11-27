"""Production V3 workflow, independent of experiment orchestration."""
import time

from .engine import ReceiverKVRunner
from chunk_specialist_workflow.components.generation import run_specialists
from chunk_specialist_workflow.components.windows import selected_windows, target_parts
from vector.supplement import METHOD, primary_windows, supplement_windows


class SemanticSummaryRunner(ReceiverKVRunner):
    def __init__(self, config, progress=None):
        super().__init__(config, progress)
        if self.semantic_router is None:
            from vector.routing import SemanticRouter
            self.progress('loading embedding model for post-selection retrieval')
            self.post_router = SemanticRouter(config.embedding_model, config.device)
        else:
            self.post_router = self.semantic_router

    def run(self, *args, **kwargs):
        if hasattr(self.post_router, 'last_vectors'):
            del self.post_router.last_vectors
        return super().run(*args, **kwargs)

    def _run_specialists(self, transcript, prefix, protected, selected, source, blocks, **kwargs):
        base = primary_windows(list(selected_windows(blocks, selected, protected,
            target_parts(kwargs['transcript_tokens']), self.config.specialist_overlap)))
        self.progress('post-selection semantic matching started')
        started = time.perf_counter()
        retained = set(selected)
        ids = [i for i, block in enumerate(blocks) if retained.intersection(block)]
        texts = {i: self.tokenizer.decode([prefix[p] for p in blocks[i] if p in retained],
                                        skip_special_tokens=True) for i in ids}
        if self.config.vector_routing and hasattr(self.post_router, 'last_vectors'):
            vectors = self.post_router.last_vectors
            encoder = dict(embedding_input_tokens=0, reused_selection_embeddings=True)
        else:
            encoded, encoder = self.post_router.encode([texts[i] for i in ids])
            vectors = {i: encoded[j] for j, i in enumerate(ids)}
            encoder['reused_selection_embeddings'] = False
        windows = supplement_windows(blocks, selected, base, vectors, texts=texts,
                                     top_k=2, neighbor_radius=1)
        retrieval = dict(**encoder, seconds=time.perf_counter()-started, method=METHOD)
        self.progress(f'post-selection matching complete; {sum(w["added_positions"] for w in windows)} '
                      'additional local KV position events')
        answer, detail = run_specialists(self, transcript, prefix, protected, selected, source, blocks,
                                        **kwargs, windows=windows, semantic_context=True)
        detail.update(workflow='semantic_supplement_v3', retrieval=retrieval,
                      supplement_blocks=2, excluded_neighbor_radius=1)
        # The parent engine adds shared prefill/selection to this stage duration.
        detail['pipeline_seconds'] += retrieval['seconds']
        return answer, detail
