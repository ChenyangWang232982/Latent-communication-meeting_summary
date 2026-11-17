"""Full-source topics -> per-topic KV -> specialist reports -> synthesis."""
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
from .evidence import annotate_transcript, source_blocks
from .prompts import DISCOVERY_REQUEST, chat, source_suffix, topic_query, analysis_request, synthesis_request
from .structure import parse_topics, parse_object


@dataclass(frozen=True)
class SpecialistConfig:
    specialist_workers: int = 1
    routing_recency_weight: float = 0.0

    def __post_init__(self):
        if self.specialist_workers < 1:
            raise ValueError('specialist_workers must be positive')
        if not math.isfinite(self.routing_recency_weight) or not 0 <= self.routing_recency_weight <= 1:
            raise ValueError('routing_recency_weight must be finite and in [0, 1]')


class GroundedSpecialistRunner(ReceiverKVRunner):
    def __init__(self, config: ReceiverKVConfig, specialists: SpecialistConfig, progress=None):
        if not config.vector_routing:
            raise ValueError('Specialist workflow requires vector routing for topic block selection')
        self.specialists = specialists
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
        self._save('config.json', dict(record_id=record_id, model=asdict(self.config),
                                      specialists=asdict(self.specialists), selector_mode='topic', evidence_review=False))
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
        transcript = annotate_transcript(transcript)
        prefix, _, protected = prompt_parts(self.tokenizer, transcript)
        discovery = source_suffix(self.tokenizer, transcript, prefix, DISCOVERY_REQUEST)
        if len(prefix) + len(discovery) >= self.context_limit:
            raise ValueError('The context exceeds the model limit; no transcript was truncated.')
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
        raw = self._decode('topic_discovery', discovery, source=source,
                           positions=list(range(len(prefix))), absolute_start=len(prefix))
        topics = parse_topics(raw)
        self._save('topics.json', topics)
        self._log(f'Discovered {len(topics)} topics; embedding source blocks and topic queries')
        start = time.perf_counter()
        query_texts = [topic_query(t) for t in topics]
        embedding_queries = ['Instruct: Retrieve all meeting passages about this issue, including revisions.\nQuery: '
                             + t['title'] + '. ' + t['scope'] for t in topics]
        vectors, embedding_stats = self.semantic_router.encode([b['text'] for b in blocks] + embedding_queries)
        block_vectors, topic_vectors = vectors[:len(blocks)], vectors[len(blocks):]
        similarities = topic_vectors @ block_vectors.T
        self._sync()
        embedding_seconds = time.perf_counter() - start
        routing, suffixes = {}, {}
        probe_tokens = 0
        budget = max(len(protected), math.ceil(len(prefix) * self.config.role_budget_ratio))
        start = time.perf_counter()
        for index, topic in enumerate(topics):
            name = topic['id']
            self._log(f'Selector {index+1}/{len(topics)}: {topic["title"]}')
            query = self.tokenizer.encode(query_texts[index], add_special_tokens=False)
            if len(prefix) + len(query) > self.context_limit:
                raise ValueError(f'{name}: topic query exceeds model context')
            probe_tokens += len(query)
            scores = query_scores(self.model, source, self._tensor(query), query_batch=self.config.query_batch_tokens)
            baseline = select_spans(scores, budget, span_size=self.config.span_tokens, protected=protected)
            attention = [sum(scores[j] for j in unit) / len(unit) for unit in units]
            combined = combine_scores(attention, similarities[index].tolist(), self.config.vector_weight)
            combined = [score + self.specialists.routing_recency_weight * blocks[i]['relative_position']
                        for i, score in enumerate(combined)]
            seeds = select_complete_blocks(units, baseline, [scores], protected, block_scores=combined)
            positions = expand_neighbor_blocks(seeds, units, protected, radius=self.config.speaker_neighbor_blocks)
            if not set(positions).difference(protected):
                raise ValueError(f'{name}: no whole block fits; increase --topic-budget-ratio')
            pos_set = set(positions)
            block_ids = [i for i, unit in enumerate(units) if pos_set.intersection(unit)]
            routing[name] = dict(topic=topic, seed_positions=seeds, selected_positions=positions,
                                 block_ids=block_ids, block_scores=combined)
            suffixes[name] = source_suffix(self.tokenizer, transcript, prefix, analysis_request(topic, block_ids))
        self._sync()
        selection_seconds = time.perf_counter() - start
        self._save('routing.json', routing)
        # Queue in topic discovery order. Each task gets chronological source KV,
        # independent writable cache and RNG; selectors finish before any workers.
        def analyze(index, topic):
            name = topic['id']
            raw = self._decode(f'specialist_{name}', suffixes[name], source=source,
                               positions=routing[name]['selected_positions'], absolute_start=len(prefix),
                               seed_offset=index+1)
            report = dict(parse_object(raw), topic_id=name, topic=topic['title'],
                          source_block_ids=routing[name]['block_ids'])
            self._save(f'report_{name}.json', report)
            return report
        reports = {}
        with ThreadPoolExecutor(max_workers=self.specialists.specialist_workers) as executor:
            pending = {executor.submit(analyze, i, topic): topic['id'] for i, topic in enumerate(topics)}
            for future in as_completed(pending):
                reports[pending[future]] = future.result()
                self._log(f'Topic specialists complete: {len(reports)}/{len(topics)}')
        del source
        # Parallel completion order must not affect the synthesis input.
        ordered = [reports[t['id']] for t in topics]
        self._save('topic_reports.json', ordered)
        request = synthesis_request(ordered)
        (self.artifacts / 'summary.input.txt').write_text(request, encoding='utf-8')
        summary = self._decode('summary', chat(self.tokenizer, request), seed_offset=len(topics)+1)
        return dict(summary=summary, source_prefill_tokens=len(prefix), source_cache_bytes=source_bytes,
                    selection_probe_tokens=probe_tokens, prefill_seconds=prefill_seconds,
                    selection_seconds=selection_seconds, embedding_seconds=embedding_seconds,
                    stages=self.stages, evidence_review=False, topic_count=len(topics),
                    embedding_input_tokens=embedding_stats['embedding_input_tokens'],
                    total_decoder_input_tokens=len(prefix) + probe_tokens +
                        sum(stage['prompt_tokens'] for stage in self.stages.values()),
                    total_decoder_output_tokens=sum(stage['output_tokens'] for stage in self.stages.values()),
                    note='Hybrid: full source KV for topic discovery; topic-specific KV to specialists; '
                         'structured text to final summary. Source block IDs are selected context, not '
                         'verified claim citations. Concurrent stage times overlap; use pipeline wall time.')
