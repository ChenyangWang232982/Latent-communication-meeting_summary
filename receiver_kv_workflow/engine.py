"""Reusable inference engine for receiver-conditioned KV routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

from .prompts import ROLE_QUERIES, prompt_parts
from .selection import select_spans, union_positions


@dataclass(frozen=True)
class ReceiverKVConfig:
    model: str = "Qwen/Qwen3-8B"
    device: str = "cuda"
    role_budget_ratio: float = 0.25
    span_tokens: int = 32
    query_batch_tokens: int = 8
    seed: int = 12
    summary_seed: int = 42
    summary_temperature: float = 0.3
    summary_top_p: float = 0.9
    speaker_neighbor_blocks: int = 1
    vector_routing: bool = False
    embedding_model: str = "models/Qwen3-Embedding-0.6B"
    vector_weight: float = 0.3
    fp32_lm_head: bool = False
    top_k: int = 0
    repetition_penalty: float = 1.0
    specialist_overlap: float = 0.25
    specialist_temperature: float = 0.3
    specialist_top_p: float = 0.9
    specialist_repetition_penalty: float = 1.1
    summary_thinking: bool = True


class ReceiverKVRunner:
    """Load a homogeneous Qwen3 once, then run many meeting records."""

    def __init__(self, config: ReceiverKVConfig, progress: Callable[[str], None] | None = None) -> None:
        if not 0 < config.role_budget_ratio <= 1:
            raise ValueError("role_budget_ratio must be in (0, 1]")
        if min(config.span_tokens, config.query_batch_tokens) < 1:
            raise ValueError("Token budgets must be positive")
        if config.speaker_neighbor_blocks < 0:
            raise ValueError("speaker_neighbor_blocks must be non-negative")
        if not 0 <= config.vector_weight <= 1:
            raise ValueError("vector_weight must be in [0, 1]")
        if not math.isfinite(config.summary_temperature) or config.summary_temperature < 0:
            raise ValueError("summary_temperature must be finite and non-negative")
        if not math.isfinite(config.summary_top_p) or not 0 < config.summary_top_p <= 1:
            raise ValueError("summary_top_p must be in (0, 1]")
        if config.top_k < 0:
            raise ValueError('top_k must be non-negative')
        if not math.isfinite(config.repetition_penalty) or config.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be finite and >= 1")
        if not 0 <= config.specialist_overlap < 1:
            raise ValueError('specialist_overlap must be in [0, 1)')
        if not math.isfinite(config.specialist_temperature) or config.specialist_temperature < 0:
            raise ValueError('specialist_temperature must be finite and non-negative')
        if not math.isfinite(config.specialist_top_p) or not 0 < config.specialist_top_p <= 1:
            raise ValueError('specialist_top_p must be in (0, 1]')
        if not math.isfinite(config.specialist_repetition_penalty) or config.specialist_repetition_penalty < 1:
            raise ValueError('specialist_repetition_penalty must be finite and >= 1')
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.progress = progress or (lambda _: None)
        self.torch = torch
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        model_config = AutoConfig.from_pretrained(config.model)
        if model_config.model_type != "qwen3" or getattr(model_config, "use_sliding_window", False):
            raise ValueError("This experimental cache adapter supports dense full-attention Qwen3 only")
        if any(layer != "full_attention" for layer in (getattr(model_config, "layer_types", None) or [])):
            raise ValueError("Sliding/hybrid caches require a separate position-aware adapter")
        rope = getattr(model_config, "rope_parameters", None) or getattr(model_config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("Scaled/dynamic RoPE is not validated by this workflow")
        self.context_limit = model_config.max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device.type == "cuda" else torch.float32)
        self.progress("loading model")
        self.model = AutoModelForCausalLM.from_pretrained(config.model, dtype=dtype, attn_implementation="sdpa")
        self.model = self.model.to(self.device).eval()
        if config.fp32_lm_head:
            from .precision import enable_fp32_lm_head
            enable_fp32_lm_head(self.model)
            self.progress("FP32 LM head enabled; transformer and KV cache keep their loaded dtype")
        self.semantic_router = None
        if config.vector_routing:
            from vector.routing import SemanticRouter
            self.progress("loading embedding model")
            self.semantic_router = SemanticRouter(config.embedding_model, config.device)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _tensor(self, ids):
        return self.torch.tensor([ids], device=self.device, dtype=self.torch.long)

    def _run_specialists(self, *args, **kwargs):
        from chunk_specialist_workflow.components.generation import run_specialists
        return run_specialists(self, *args, **kwargs)

    def run(self, record_id: str, transcript: str, *, variants: tuple[str, ...],
            routing_audit: bool = False,
            on_state: Callable[[dict], None] | None = None,
            on_variant: Callable[[str, str], None] | None = None,
            on_generation: Callable[[dict], None] | None = None,
            ) -> dict:
        """Return transient answers plus auditable measurement fields for one record."""
        from .cache import forward_tokens, size_bytes
        from .scoring import query_scores

        if not transcript.strip():
            raise ValueError(f"Transcript is empty: {record_id}")
        transcript_tokens = len(self.tokenizer.encode(transcript, add_special_tokens=False))
        requested = tuple(dict.fromkeys(variants))
        chunk_result = None
        chunk_results = {}
        valid_variants = {'chunk_kv', 'chunk_text'}
        if not requested or any(name not in valid_variants for name in requested):
            raise ValueError('variants must be chunk_kv and/or chunk_text')
        def emit(stage: str, message: str, **fields) -> None:
            if on_state is not None:
                on_state({"stage": stage, "message": message, **fields})

        emit("queued", "Preparing transcript", reviewStatus="skipped")
        prefix, suffix, protected = prompt_parts(self.tokenizer, transcript)
        query_text = dict(ROLE_QUERIES)
        queries = {role: self.tokenizer.encode(query, add_special_tokens=False) for role, query in query_text.items()}
        required = len(prefix) + max(
            len(suffix) + 1,
            max(map(len, queries.values())),
        )
        if required > self.context_limit:
            raise ValueError(
                f"{record_id}: context requires {required} positions but model supports {self.context_limit}; no text was truncated."
            )
        budget = max(len(protected), math.ceil(len(prefix) * self.config.role_budget_ratio))
        from .speaker_blocks import speaker_units
        block_start = time.perf_counter()
        speaker_blocks = speaker_units(self.tokenizer, transcript, prefix, protected)
        block_seconds = time.perf_counter() - block_start
        self.progress(f"{record_id}: prepared {len(speaker_blocks)} consecutive speaker blocks")
        self._sync()
        prefill_start = time.perf_counter()
        self.progress(f"{record_id}: full source prefill started")
        emit("prefill", "Reading the meeting transcript", metrics={"sourcePositions": len(prefix)})
        with self.torch.inference_mode():
            output = forward_tokens(self.model, self._tensor(prefix), None, 0, logits_to_keep=1)
            source = output.past_key_values
            del output
            self._sync()
            prefill_seconds = time.perf_counter() - prefill_start
            source_bytes = size_bytes(source)

            messages, role_metrics = [], {}
            role_scores = []
            selection_seconds = 0.0
            selection_probe_tokens = 0
            if requested:
                emit("selection", "Selecting relevant meeting context")
                selection_start = time.perf_counter()
                for index, (role, query) in enumerate(queries.items(), start=1):
                    emit("selection", f"Selector: {role}", completed=index - 1, total=len(queries))
                    self.progress(f"{record_id}: selector {role} started ({index - 1}/{len(queries)} complete)")
                    scores = query_scores(self.model, source, self._tensor(query), query_batch=self.config.query_batch_tokens)
                    role_scores.append(scores)
                    selected = select_spans(scores, budget, span_size=self.config.span_tokens, protected=protected)
                    messages.append(selected)
                    role_metrics[role] = {"query_tokens": len(query), "selected_positions": len(selected)}
                    selection_probe_tokens += len(query)
                    emit("selection", f"Selector: {role}", completed=index, total=len(queries))
                self._sync()
                selection_seconds = time.perf_counter() - selection_start
            selected = union_positions(messages)
            baseline_selection = list(selected)
            baseline_positions = len(selected)
            seed_selection = list(selected)
            vector_metrics = None
            if selected:
                from .speaker_blocks import expand_neighbor_blocks, select_complete_blocks
                boundary_start = time.perf_counter()
                selected = select_complete_blocks(speaker_blocks, selected, role_scores, protected)
                attention_seed_selection = list(selected)
                if self.semantic_router is not None:
                    emit("embedding", f"Embedding/ranking {len(speaker_blocks)} speaker blocks")
                    self.progress(f"{record_id}: semantic speaker-block ranking started")
                    pooled = [0.0] * len(prefix)
                    for scores in role_scores:
                        scale = sum(scores)
                        if scale > 0:
                            pooled = [max(a, b / scale) for a, b in zip(pooled, scores)]
                    attention = [sum(pooled[i] for i in block) / len(block) for block in speaker_blocks]
                    anchors = [i for i, block in enumerate(speaker_blocks)
                               if set(block).intersection(selected)]
                    if anchors:
                        texts = [self.tokenizer.decode([prefix[i] for i in block], skip_special_tokens=True)
                                 for block in speaker_blocks]
                        scores, vector_metrics = self.semantic_router.rank(
                            texts, attention, anchors, self.config.vector_weight)
                        selected = select_complete_blocks(speaker_blocks, baseline_selection, role_scores,
                                                          protected, block_scores=scores)
                    else:
                        vector_metrics = {"skipped": "No complete attention seed block fits the budget",
                                          "embedding_input_tokens": 0, "seconds": 0.0}
                seed_selection = list(selected)
                if self.config.speaker_neighbor_blocks:
                    selected = expand_neighbor_blocks(
                        selected, speaker_blocks, protected, radius=self.config.speaker_neighbor_blocks
                    )
                if vector_metrics is not None:
                    original_expanded = expand_neighbor_blocks(attention_seed_selection, speaker_blocks, protected,
                                                              radius=self.config.speaker_neighbor_blocks)
                    vector_metrics.update(seed_budget_positions=baseline_positions,
                                          attention_only_final_positions=len(original_expanded),
                                          hybrid_final_positions=len(selected),
                                          selected_block_ids=[i for i, block in enumerate(speaker_blocks)
                                                              if set(block).intersection(selected)])
                selection_seconds += time.perf_counter() - boundary_start + block_seconds
                detail = f" seeds={len(seed_selection)}, +/-{self.config.speaker_neighbor_blocks} blocks"
                self.progress(f"{record_id}: speaker KV selection {len(selected)}/{baseline_positions} positions{detail}")
                emit("selected", f"Selected {len(selected)} KV positions; {len(prefix)} source positions")
            answers, measurements, handoffs = {}, {}, {}
            emit('chunk_specialists', 'Specialists reading independent windows of selected source context')
            for name in requested:
                if on_variant is not None:
                    on_variant(name, 'start')
                answer, detail = self._run_specialists(transcript, prefix, protected, selected, source, speaker_blocks,
                    on_generation=on_generation, on_progress=on_state,
                    communication='kv' if name == 'chunk_kv' else 'text', transcript_tokens=transcript_tokens)
                chunk_results[name] = detail
                if name == 'chunk_kv':
                    chunk_result = detail
                final_stage = list(detail['stages'].values())[-1]
                local_stages = list(detail['stages'].values())[:-1]
                answers[name] = answer
                handoffs[name] = '\n\n'.join(row['report'] for row in detail['reports'])
                measurements[name] = dict(
                    incoming_positions=final_stage['source_kv_positions'] if name == 'chunk_kv' else final_stage['prompt_tokens'],
                    selected_source_positions=len(selected), incoming_cache_bytes=final_stage['source_kv_bytes'],
                    logical_payload_bytes=sum(v['source_kv_bytes']+v['source_text_bytes'] for v in detail['stages'].values())+
                        detail['text_payload_bytes'],
                    positions_vs_full_percent=100*(len(selected)-len(prefix))/len(prefix) if name == 'chunk_kv' else None,
                    cache_bytes_vs_full_percent=100*(final_stage['source_kv_bytes']-source_bytes)/source_bytes if name == 'chunk_kv' else None,
                    receiver_seconds=final_stage['seconds'], specialist_seconds=sum(v['seconds'] for v in local_stages),
                    specialist_count=len(local_stages),
                    specialist_handoff_tokens=sum(v['output_tokens'] for v in local_stages),
                    specialist_handoff_cap_hit=any(v['output_cap_hit'] for v in local_stages),
                    pipeline_seconds_excluding_model_load=prefill_seconds+selection_seconds+detail['pipeline_seconds'],
                    generated_tokens=final_stage['output_tokens'], output_cap_hit=final_stage['output_cap_hit'],
                    generation_counts=final_stage['generation_counts'],
                    token_accounting=dict(source_prefill_tokens=len(prefix), selection_probe_tokens=selection_probe_tokens,
                        specialist_prompt_tokens=sum(v['prompt_tokens'] for v in local_stages),
                        specialist_handoff_tokens=sum(v['output_tokens'] for v in local_stages),
                        receiver_prompt_tokens=final_stage['prompt_tokens'], receiver_output_tokens=final_stage['output_tokens']))
                if on_variant is not None:
                    on_variant(name, 'end')
        emit("completed", "Summary complete", reviewStatus="skipped")
        audit = None
        if routing_audit:
            from .routing_audit import build_audit
            audit = build_audit(self.tokenizer, transcript, prefix, protected, selected, baseline_selection,
                                dict(zip(queries, messages)), "speaker",
                                seed_selection=seed_selection,
                                neighbor_blocks=self.config.speaker_neighbor_blocks)
            audit["vector_routing"] = vector_metrics
        return {
            "vector_routing": vector_metrics,
            "routing_audit": audit,
            "id": record_id, "answers": answers, "variants": measurements, "handoffs": handoffs,
            "transcript_tokens": transcript_tokens,
            "source_positions": len(prefix), "context_limit": self.context_limit,
            "source_cache_bytes": source_bytes, "prefill_seconds": prefill_seconds,
            "selection_seconds": selection_seconds, "roles": role_metrics,
            "selected_source_positions": selected,
            "selection_strategy": "speaker",
            "speaker_neighbor_blocks": self.config.speaker_neighbor_blocks,
            "seed_selected_positions": seed_selection,
            "baseline_selected_positions": baseline_positions,
            "chunk_specialists": chunk_result,
            "chunk_specialists_by_variant": chunk_results,
        }
