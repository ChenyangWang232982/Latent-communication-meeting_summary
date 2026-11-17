"""Independent role KV -> structured claims -> retrieved evidence -> synthesis."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time

from receiver_kv_workflow.engine import ReceiverKVConfig, ReceiverKVRunner
from receiver_kv_workflow.prompts import prompt_parts
from receiver_kv_workflow.selection import select_spans
from receiver_kv_workflow.speaker_blocks import select_complete_blocks, expand_neighbor_blocks
from vector.routing import combine_scores
from .evidence import source_blocks, retrieve, claim_query
from .prompts import ROLE_REQUESTS, chat, extraction_suffix, grounding_request, synthesis_request
from .structure import ROLES, parse_claims, parse_reviews


@dataclass(frozen=True)
class GroundingConfig:
    specialist_workers: int = 1
    evidence_top_k: int = 4
    evidence_candidate_pool: int = 12
    evidence_neighbors: int = 1
    evidence_recency_weight: float = 0.0
    routing_recency_weight: float = 0.0

    def __post_init__(self):
        if not 1 <= self.specialist_workers <= len(ROLES):
            raise ValueError('specialist_workers must be between 1 and 4')
        if self.evidence_candidate_pool < self.evidence_top_k or self.evidence_top_k < 1 or self.evidence_neighbors < 0:
            raise ValueError('Require candidate_pool >= top_k >= 1 and evidence_neighbors >= 0')
        for weight in (self.evidence_recency_weight, self.routing_recency_weight):
            if not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError('Recency weights must be finite and in [0, 1]')


class GroundedSpecialistRunner(ReceiverKVRunner):
    def __init__(self, config: ReceiverKVConfig, grounding: GroundingConfig, progress=None):
        if not config.vector_routing:
            raise ValueError('Grounded specialist workflow requires vector routing for evidence retrieval')
        self.grounding = grounding
        self._log_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        super().__init__(config, progress=progress)

    def _log(self, message):
        with self._log_lock:
            self.progress(message)

    def _save(self, name, data):
        path = self.artifacts / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def _decode(self, stage, ids, *, source=None, positions=None, absolute_start=0, seed_offset=0):
        from receiver_kv_workflow.cache import generate, select_cache, size_bytes
        self._log(f'{stage}: started')
        if absolute_start + len(ids) >= self.context_limit:
            raise ValueError(f'{stage}: The context requires {absolute_start + len(ids)} positions; '
                             f'model limit={self.context_limit}. No input was truncated.')
        rng = self.torch.Generator(device=self.device).manual_seed(self.config.seed + seed_offset)
        start = time.perf_counter()
        with self.torch.inference_mode():
            incoming = select_cache(source, positions) if source is not None else None
            cache_bytes = size_bytes(incoming) if incoming is not None else 0
            def snapshot(event):
                (self.artifacts / f'{stage}.partial.txt').write_text(event['text'], encoding='utf-8')
            def progress(event):
                count = event.get('tokens', 0)
                if event.get('finished') or (count and count % 256 == 0):
                    self._log(f'{stage}: {count} tokens' + ('; finished' if event.get('finished') else ''))
            answer, count, cap_hit, counts = generate(
                self.model, self.tokenizer, incoming, self._tensor(ids),
                absolute_start=absolute_start, context_limit=self.context_limit,
                temperature=self.config.temperature, top_p=self.config.top_p, top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty, generator=rng,
                on_progress=progress, on_snapshot=snapshot)
            del incoming
        self._sync()
        metrics = dict(prompt_tokens=len(ids), output_tokens=count, cap_hit=cap_hit,
                       source_kv_positions=len(positions or []), source_kv_bytes=cache_bytes,
                       seconds=time.perf_counter() - start, generation_counts=counts,
                       seed=self.config.seed + seed_offset)
        (self.artifacts / f'{stage}.txt').write_text(answer + '\n', encoding='utf-8')
        with self._metrics_lock:
            self.stages[stage] = metrics
            self._save('stages.json', self.stages)
        if cap_hit:
            raise RuntimeError(f'{stage}: model context exhausted before EOS; raw output saved')
        return answer

    def run(self, record_id, transcript, *, artifact_dir):
        from receiver_kv_workflow.cache import forward_tokens, size_bytes
        from receiver_kv_workflow.scoring import query_scores
        if not transcript.strip():
            raise ValueError('Transcript is empty')
        self.artifacts = Path(artifact_dir)
        self.artifacts.mkdir(parents=True, exist_ok=False)
        self.stages = {}
        self._save('config.json', dict(record_id=record_id, model=asdict(self.config), grounding=asdict(self.grounding)))
        started = time.perf_counter()
        try:
            result = self._run(transcript, forward_tokens, query_scores, size_bytes)
            result['pipeline_seconds_excluding_model_load'] = time.perf_counter() - started
            self._save('metrics.json', result)
            self._save('status.json', {'status': 'completed'})
            return result
        except Exception as exc:
            self._save('status.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            raise

    def _run(self, transcript, forward_tokens, query_scores, size_bytes):
        prefix, _, protected = prompt_parts(self.tokenizer, transcript)
        suffixes = {role: extraction_suffix(self.tokenizer, transcript, prefix, role) for role in ROLES}
        queries = {role: self.tokenizer.encode(ROLE_REQUESTS[role], add_special_tokens=False) for role in ROLES}
        required = len(prefix) + max(max(map(len, suffixes.values())) + 1, max(map(len, queries.values())))
        if required > self.context_limit:
            raise ValueError(f'The context requires {required} positions; model limit={self.context_limit}. No input was truncated.')
        blocks = source_blocks(self.tokenizer, transcript, prefix, protected)
        units = [b['positions'] for b in blocks]
        self._save('source_blocks.json', blocks)
        self._log(f'Prepared {len(blocks)} speaker blocks; full source prefill started')
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = forward_tokens(self.model, self._tensor(prefix), None, 0, logits_to_keep=1)
            source = output.past_key_values
            del output
        self._sync()
        prefill_seconds = time.perf_counter() - start
        source_bytes = size_bytes(source)
        self._log('Embedding all source blocks once')
        start = time.perf_counter()
        vectors, embedding_stats = self.semantic_router.encode([b['text'] for b in blocks])
        self._sync()
        embedding_seconds = time.perf_counter() - start
        query_embedding_tokens = 0
        routing, chosen = {}, {}
        budget = max(len(protected), math.ceil(len(prefix) * self.config.role_budget_ratio))
        start = time.perf_counter()
        for role in ROLES:
            self._log(f'Selector {role}: independent role KV selection')
            scores = query_scores(self.model, source, self._tensor(queries[role]), query_batch=self.config.query_batch_tokens)
            baseline = select_spans(scores, budget, span_size=self.config.span_tokens, protected=protected)
            initial = set(select_complete_blocks(units, baseline, [scores], protected))
            anchors = [i for i, unit in enumerate(units) if initial.intersection(unit)]
            attention = [sum(scores[j] for j in unit) / len(unit) for unit in units]
            semantic = [0.0] * len(units)
            if anchors:
                similarities = vectors @ vectors[anchors].T
                semantic = [max((float(similarities[i, j]) for j, a in enumerate(anchors) if a != i), default=0.0)
                            for i in range(len(units))]
            combined = combine_scores(attention, semantic, self.config.vector_weight)
            combined = [score + self.grounding.routing_recency_weight * blocks[i]['relative_position']
                        for i, score in enumerate(combined)]
            seeds = select_complete_blocks(units, baseline, [scores], protected, block_scores=combined)
            positions = expand_neighbor_blocks(seeds, units, protected, radius=self.config.speaker_neighbor_blocks)
            if not set(positions).difference(protected):
                raise ValueError(f'{role}: no complete speaker block fits the role budget; increase --role-budget-ratio')
            chosen[role] = positions
            pos_set = set(positions)
            routing[role] = dict(seed_positions=seeds, selected_positions=positions,
                                 block_ids=[i for i, unit in enumerate(units) if pos_set.intersection(unit)],
                                 block_scores=combined, semantic_anchor_ids=anchors)
        self._sync()
        selection_seconds = time.perf_counter() - start
        self._save('routing.json', routing)
        # Selectors have restored the attention backend before concurrent decoding.
        # Each task clones its own KV and uses its own RNG; the source is read-only.
        def extract(role):
            raw = self._decode(f'specialist_{role}', suffixes[role], source=source,
                               positions=chosen[role], absolute_start=len(prefix), seed_offset=ROLES.index(role))
            claims = parse_claims(raw, role)
            self._save(f'claims_{role}.json', claims)
            return claims
        extracted = {}
        with ThreadPoolExecutor(max_workers=self.grounding.specialist_workers) as executor:
            pending = {executor.submit(extract, role): role for role in ROLES}
            for future in as_completed(pending):
                role = pending[future]
                extracted[role] = future.result()
                self._log(f'Specialists complete: {len(extracted)}/{len(ROLES)}')
        del source
        claims = [claim for role in ROLES for claim in extracted[role]]
        retrievals, reviewed = {}, []
        retrieval_seconds = 0.0
        if claims:
            self._log(f'Retrieving full-source evidence for {len(claims)} claims')
            start = time.perf_counter()
            query_vectors, query_stats = self.semantic_router.encode([claim_query(c) for c in claims])
            similarities = query_vectors @ vectors.T
            query_embedding_tokens = query_stats['embedding_input_tokens']
            for i, claim in enumerate(claims):
                retrievals[claim['id']] = retrieve(similarities[i].tolist(),
                    candidate_pool=self.grounding.evidence_candidate_pool, top_k=self.grounding.evidence_top_k,
                    neighbors=self.grounding.evidence_neighbors, recency_weight=self.grounding.evidence_recency_weight)
            self._sync()
            retrieval_seconds = time.perf_counter() - start
            self._save('retrieval.json', retrievals)
            for index, role in enumerate(ROLES):
                items = extracted[role]
                if not items:
                    continue
                # Individual retrieval, one shared deduplicated evidence packet per category.
                request = grounding_request(items, retrievals, blocks)
                (self.artifacts / f'grounding_{role}.input.txt').write_text(request, encoding='utf-8')
                raw = self._decode(f'grounding_{role}', chat(self.tokenizer, request), seed_offset=100 + index)
                rows = parse_reviews(raw, items, retrievals)
                self._save(f'grounded_{role}.json', rows)
                for row in rows:
                    if row.get('citation_warnings'):
                        self._log(f"{row['id']}: invalid grounding citations; retained as uncertain (see audit)")
                reviewed.extend(rows)
        self._save('grounded_claims.json', reviewed)
        # Preserve all machine judgments in the audit, including unsupported ones.
        accepted = [r for r in reviewed if r['verdict'] != 'unsupported']
        # Keep the rejected model text in audit files only, outside synthesis.
        synthesis_items = [{k: v for k, v in row.items() if k not in ('reported_review', 'invalid_citations')}
                           for row in accepted]
        request = synthesis_request(synthesis_items)
        (self.artifacts / 'summary.input.txt').write_text(request, encoding='utf-8')
        summary = self._decode('summary', chat(self.tokenizer, request), seed_offset=200)
        return dict(summary=summary, source_prefill_tokens=len(prefix), source_cache_bytes=source_bytes,
                    selection_probe_tokens=sum(map(len, queries.values())), prefill_seconds=prefill_seconds,
                    selection_seconds=selection_seconds, embedding_seconds=embedding_seconds,
                    retrieval_seconds=retrieval_seconds, stages=self.stages,
                    embedding_input_tokens=embedding_stats['embedding_input_tokens'] + query_embedding_tokens,
                    total_decoder_input_tokens=len(prefix) + sum(map(len, queries.values())) +
                        sum(stage['prompt_tokens'] for stage in self.stages.values()),
                    total_decoder_output_tokens=sum(stage['output_tokens'] for stage in self.stages.values()),
                    extracted_claims=len(claims), grounded_claims=len(reviewed),
                    unsupported_claims=sum(r['verdict'] == 'unsupported' for r in reviewed),
                    citation_warning_claims=sum(bool(r.get('citation_warnings')) for r in reviewed),
                    note='Hybrid workflow: source KV to specialists; structured text and verbatim retrieved '
                         'evidence for grounding; structured text to final summary. Grounding is model judgment, '
                         'not an independent quality score. Concurrent stage times overlap; use pipeline wall time.')
