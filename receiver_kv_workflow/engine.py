"""Reusable inference engine for receiver-conditioned KV routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Callable

from .prompts import (ROLE_QUERIES, prompt_parts, organize_prompt, selected_evidence_excerpts,
                      specialist_sender_prompt, specialist_text_prompt)
from .selection import select_spans, union_positions


@dataclass(frozen=True)
class ReceiverKVConfig:
    model: str = "Qwen/Qwen3-4B"
    device: str = "cuda"
    role_budget_ratio: float = 0.15
    span_tokens: int = 32
    query_batch_tokens: int = 8
    seed: int = 7
    review_seed: int = 7
    review_temperature: float = 0.0
    review_top_p: float = 0.9
    speaker_neighbor_blocks: int = 1
    vector_routing: bool = False
    embedding_model: str = "models/Qwen3-Embedding-0.6B"
    vector_weight: float = 0.3
    fp32_lm_head: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    chunk_specialists: bool = False
    specialist_parts: int = 8
    specialist_overlap: float = 0.25
    specialist_temperature: float = 0.3
    specialist_top_p: float = 0.9
    specialist_repetition_penalty: float = 1.1
    summary_thinking: bool = False


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
        if not math.isfinite(config.review_temperature) or config.review_temperature < 0:
            raise ValueError("review_temperature must be finite and non-negative")
        if not math.isfinite(config.review_top_p) or not 0 < config.review_top_p <= 1:
            raise ValueError("review_top_p must be in (0, 1]")
        if not math.isfinite(config.temperature) or config.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(config.top_p) or not 0 < config.top_p <= 1 or config.top_k < 0:
            raise ValueError("top_p must be in (0, 1] and top_k must be non-negative")
        if not math.isfinite(config.repetition_penalty) or config.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be finite and >= 1")
        if config.specialist_parts < 1 or not 0 <= config.specialist_overlap < 1:
            raise ValueError('specialist_parts must be positive; specialist_overlap must be in [0, 1)')
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

    def run(self, record_id: str, transcript: str, *, variants: tuple[str, ...],
            routing_audit: bool = False, review: bool = False,
            on_state: Callable[[dict], None] | None = None,
            on_variant: Callable[[str, str], None] | None = None,
            on_generation: Callable[[dict], None] | None = None,
            on_token: Callable | None = None,
            diagnostic_full_source: bool = False,
            diagnostic_refill_selected: bool = False) -> dict:
        """Return transient answers plus auditable measurement fields for one record."""
        from .cache import forward_tokens, generate, select_cache, size_bytes
        from .scoring import query_scores

        if not transcript.strip():
            raise ValueError(f"Transcript is empty: {record_id}")
        requested = tuple(dict.fromkeys(variants))
        chunk_result = None
        chunk_results = {}
        if self.config.chunk_specialists:
            if diagnostic_full_source or diagnostic_refill_selected:
                raise ValueError('Chunk specialists do not support cache diagnostics')
        valid_variants = {"routed_kv", "specialist_text"}
        if not requested or any(name not in valid_variants for name in requested):
            raise ValueError("variants must be routed_kv and/or specialist_text")
        if review and 'routed_kv' not in requested:
            raise ValueError('Summary review requires the routed_kv variant')
        if diagnostic_full_source and diagnostic_refill_selected:
            raise ValueError("Full-source and selected-refill controls are mutually exclusive")
        if (diagnostic_full_source or diagnostic_refill_selected) and (requested != ("routed_kv",) or review):
            raise ValueError("Cache diagnostic requires only routed_kv and review=False")
        def emit(stage: str, message: str, **fields) -> None:
            if on_state is not None:
                on_state({"stage": stage, "message": message, **fields})

        def generation_progress(name):
            if on_state is None:
                return None
            return lambda event: emit(name, name, **event)

        def generation_snapshot(name):
            if on_generation is None:
                return None
            return lambda event: on_generation({"stage": name, **event})

        def token_observer(name):
            if on_token is None:
                return None
            return lambda logits, token, step: on_token(name, logits, token, step)

        def generation_input(name, prompt_ids, source_ids, positions, condition):
            if on_generation is not None:
                on_generation({
                    "stage": name, "event": "input", "sample_id": record_id,
                    "prefix_token_ids": source_ids, "suffix_token_ids": prompt_ids,
                    "selected_positions": positions, "protected_positions": protected if source_ids else [],
                    "cache_condition": condition, "config": asdict(self.config),
                    "model_revision": getattr(self.model.config, "_commit_hash", None),
                    "model_dtype": str(getattr(self.model, "dtype", None)),
                    "lm_head_weight_dtype": str(self.model.get_output_embeddings().weight.dtype),
                    "lm_head_output_dtype": getattr(self.model.get_output_embeddings(), "last_output_dtype", None),
                    "attention_backend": self.model.config._attn_implementation,
                    "eos_token_ids": sorted({value for value in (
                        list(self.model.generation_config.eos_token_id)
                        if isinstance(self.model.generation_config.eos_token_id, (list, tuple))
                        else [self.model.generation_config.eos_token_id]
                    ) + [self.tokenizer.eos_token_id] if value is not None}),
                    "temperature": self.config.temperature, "top_p": self.config.top_p,
                    "top_k": self.config.top_k, "seed": self.config.seed,
                    "repetition_penalty": self.config.repetition_penalty,
                    "repetition_history_scope": "generated_only",
                })

        emit("queued", "Preparing transcript", reviewStatus="pending" if review else "skipped")
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
            if diagnostic_full_source:
                # Diagnostic control only. Clone all rows of the same source
                # cache; keep the identical suffix and original continuation.
                selected = list(range(len(prefix)))
                emit("full_source_control", f"Diagnostic: retaining all {len(selected)} source positions")
            answers, measurements, handoffs = {}, {}, {}
            if self.config.chunk_specialists:
                from chunk_specialist_workflow.components.generation import run_specialists
                emit('chunk_specialists', 'Specialists reading shared global selected context plus current transcript parts')
                for name in requested:
                    if on_variant is not None:
                        on_variant(name, 'start')
                    answer, detail = run_specialists(self, transcript, prefix, protected, selected, source, speaker_blocks,
                        on_generation=on_generation, on_progress=on_state,
                        communication='kv' if name == 'routed_kv' else 'text')
                    chunk_results[name] = detail
                    if name == 'routed_kv':
                        chunk_result = detail
                    final_stage = list(detail['stages'].values())[-1]
                    local_stages = list(detail['stages'].values())[:-1]
                    answers[name] = answer
                    handoffs[name] = '\n\n'.join(row['report'] for row in detail['reports'])
                    measurements[name] = dict(
                        incoming_positions=final_stage['source_kv_positions'] if name == 'routed_kv' else final_stage['prompt_tokens'],
                        selected_source_positions=len(selected), incoming_cache_bytes=final_stage['source_kv_bytes'],
                        logical_payload_bytes=sum(v['source_kv_bytes']+v['source_text_bytes']+v['current_part_text_bytes'] for v in detail['stages'].values())+
                            detail['text_payload_bytes'],
                        positions_vs_full_percent=100*(len(selected)-len(prefix))/len(prefix) if name == 'routed_kv' else None,
                        cache_bytes_vs_full_percent=100*(final_stage['source_kv_bytes']-source_bytes)/source_bytes if name == 'routed_kv' else None,
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
            if "routed_kv" in requested and not self.config.chunk_specialists:
                emit("summary", "Generating the summary", metrics={"selectedPositions": len(selected)})
                if on_variant is not None:
                    on_variant("routed_kv", "start")
                self.torch.manual_seed(self.config.seed)
                receiver_start = time.perf_counter()
                ids = selected
                generation_input("routed_kv_receiver", suffix, prefix, ids,
                                 "full" if diagnostic_full_source else "refill" if diagnostic_refill_selected else "routed")
                if diagnostic_refill_selected:
                    from .cache import refill_selected_cache
                    emit("selected_refill", f"Recomputing {len(ids)} selected tokens at original positions")
                    incoming = refill_selected_cache(self.model, self._tensor(prefix), ids)
                else:
                    incoming = select_cache(source, ids)
                absolute_start, payload_bytes = len(prefix), size_bytes(incoming)
                incoming_bytes = size_bytes(incoming)
                answer, generated_tokens, output_cap_hit, generation_counts = generate(
                    self.model, self.tokenizer, incoming, self._tensor(suffix), absolute_start=absolute_start,
                    context_limit=self.context_limit,
                    temperature=self.config.temperature, top_p=self.config.top_p, top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    on_progress=generation_progress("routed_kv receiver"),
                    on_snapshot=generation_snapshot("routed_kv_receiver"),
                    on_token=token_observer("routed_kv_receiver"),
                )
                del incoming
                self._sync()
                receiver_seconds = time.perf_counter() - receiver_start
                answers["routed_kv"] = answer
                measurements["routed_kv"] = {
                    "incoming_positions": len(ids),
                    "selected_source_positions": None,
                    "incoming_cache_bytes": incoming_bytes,
                    "logical_payload_bytes": payload_bytes,
                    "positions_vs_full_percent": 100 * (len(ids) - len(prefix)) / len(prefix),
                    "cache_bytes_vs_full_percent": 100 * (incoming_bytes - source_bytes) / source_bytes,
                    "receiver_seconds": receiver_seconds,
                    "pipeline_seconds_excluding_model_load": prefill_seconds + selection_seconds + receiver_seconds,
                    "generated_tokens": generated_tokens, "output_cap_hit": output_cap_hit,
                    "generation_counts": generation_counts,
                    "token_accounting": {
                        "source_prefill_tokens": len(prefix),
                        "selection_probe_tokens": selection_probe_tokens,
                        "specialist_prompt_tokens": 0,
                        "specialist_handoff_tokens": 0,
                        "receiver_prompt_tokens": len(suffix),
                        "receiver_output_tokens": generated_tokens,
                    },
                }
                if on_variant is not None:
                    on_variant("routed_kv", "end")
            if "specialist_text" in requested and not self.config.chunk_specialists:
                if on_variant is not None:
                    on_variant("specialist_text", "start")
                self.progress(f"{record_id}: specialist_text sender started")
                emit("text_mapping", "Mapping selected positions to source evidence")
                excerpts, evidence_tokens, excerpt_count = selected_evidence_excerpts(self.tokenizer, transcript, selected)
                sender_prompt = self._tensor(specialist_sender_prompt(self.tokenizer, excerpts))
                if sender_prompt.shape[1] >= self.context_limit:
                    raise ValueError(f"{record_id}: specialist text sender prompt exhausts the model context limit.")
                specialist_start = time.perf_counter()
                self.torch.manual_seed(self.config.seed)
                generation_input("specialist_text_sender", sender_prompt[0].tolist(), [], [], "text")
                handoff, handoff_tokens, handoff_cap_hit, handoff_counts = generate(
                    self.model, self.tokenizer, None, sender_prompt,
                    absolute_start=0, context_limit=self.context_limit,
                    temperature=self.config.temperature, top_p=self.config.top_p, top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    on_progress=generation_progress("specialist_text sender"),
                    on_snapshot=generation_snapshot("specialist_text_sender"),
                    on_token=token_observer("specialist_text_sender"),
                )
                self._sync()
                specialist_seconds = time.perf_counter() - specialist_start
                handoffs["specialist_text"] = handoff
                receiver_prompt = self._tensor(specialist_text_prompt(self.tokenizer, handoff))
                if receiver_prompt.shape[1] >= self.context_limit:
                    raise ValueError(
                        f"{record_id}: specialist handoff leaves no room for output in the model context; no input was truncated."
                    )
                receiver_start = time.perf_counter()
                self.torch.manual_seed(self.config.seed)
                generation_input("specialist_text_receiver", receiver_prompt[0].tolist(), [], [], "text")
                answer, generated_tokens, output_cap_hit, generation_counts = generate(
                    self.model, self.tokenizer, None, receiver_prompt, absolute_start=0,
                    context_limit=self.context_limit,
                    temperature=self.config.temperature, top_p=self.config.top_p, top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    on_progress=generation_progress("specialist_text receiver"),
                    on_snapshot=generation_snapshot("specialist_text_receiver"),
                    on_token=token_observer("specialist_text_receiver"),
                )
                self._sync()
                receiver_seconds = time.perf_counter() - receiver_start
                answers["specialist_text"] = answer
                measurements["specialist_text"] = {
                    "incoming_positions": receiver_prompt.shape[1],
                    "selected_source_positions": len(selected),
                    "incoming_cache_bytes": 0,
                    "logical_payload_bytes": handoff_tokens * 4,
                    "positions_vs_full_percent": None,
                    "cache_bytes_vs_full_percent": None,
                    "specialist_handoff_tokens": handoff_tokens,
                    "specialist_handoff_cap_hit": handoff_cap_hit,
                    "specialist_handoff_generation_counts": handoff_counts,
                    "handoff_utf8_bytes": len(handoff.encode("utf-8")),
                    "specialist_count": 1,
                    "specialist_seconds": specialist_seconds,
                    "receiver_seconds": receiver_seconds,
                    "pipeline_seconds_excluding_model_load": prefill_seconds + selection_seconds + specialist_seconds + receiver_seconds,
                    "generated_tokens": generated_tokens,
                    "output_cap_hit": output_cap_hit,
                    "generation_counts": generation_counts,
                    "token_accounting": {
                        "source_prefill_tokens": len(prefix),
                        "selection_probe_tokens": selection_probe_tokens,
                        "specialist_prompt_tokens": sender_prompt.shape[1],
                        "specialist_handoff_tokens": handoff_tokens,
                        "receiver_prompt_tokens": receiver_prompt.shape[1],
                        "receiver_output_tokens": generated_tokens,
                    },
                    "retrieved_evidence_tokens": evidence_tokens,
                    "retrieved_excerpt_count": excerpt_count,
                }
                if on_variant is not None:
                    on_variant("specialist_text", "end")
            review_result = None
            if review and not self.config.chunk_specialists:
                from .review_output import change_log, split_review_response
                draft = answers['routed_kv']
                if not draft.strip():
                    review_result = {'status': 'skipped', 'reason': 'The draft summary is empty.'}
                else:
                    organize_ids = organize_prompt(self.tokenizer, draft, transcript)
                    if len(prefix) + len(organize_ids) >= self.context_limit:
                        review_result = {'status': 'skipped', 'reason': 'Source positions and review prompt leave no output room; draft preserved.'}
                    else:
                        self.progress(f"{record_id}: evidence-based summary review started ({len(selected)} source KV positions)")
                        emit("review", "Reviewing the summary against selected context", reviewStatus="running")
                        self._sync()
                        start = time.perf_counter()
                        self.torch.manual_seed(self.config.review_seed)
                        review_cache = select_cache(source, selected)
                        response, tokens, cap_hit, counts = generate(
                            self.model, self.tokenizer, review_cache, self._tensor(organize_ids),
                            absolute_start=len(prefix), context_limit=self.context_limit,
                            temperature=self.config.review_temperature, top_p=self.config.review_top_p)
                        del review_cache
                        self._sync()
                        seconds = time.perf_counter() - start
                        organized, reasons = split_review_response(response)
                        if cap_hit or not organized.strip():
                            review_result = {'status': 'skipped', 'reason':
                                f'Organization incomplete (cap-hit={cap_hit}); draft preserved.'}
                        else:
                            review_result = {
                                'status': 'completed', 'answer': organized,
                                'changes': change_log(draft, organized, reasons),
                                'prompt_tokens': len(organize_ids), 'output_tokens': tokens,
                                'generation_counts': counts, 'output_cap_hit': cap_hit,
                                'same_text_as_draft': organized == draft,
                                'seed': self.config.review_seed, 'seconds': seconds,
                                'temperature': self.config.review_temperature,
                                'top_p': self.config.review_top_p,
                                'source_kv_positions': len(selected), 'review_reasons_present': bool(reasons),
                                'draft_output_cap_hit': measurements['routed_kv']['output_cap_hit'],
                            }
                            self.progress(f"{record_id}: organization completed; same text as draft={organized == draft}")
        if review and not self.config.chunk_specialists:
            emit("completed", "Review complete" if review_result['status'] == 'completed' else "Review skipped; showing the draft",
                 reviewStatus=review_result['status'])
        else:
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
            "source_positions": len(prefix), "context_limit": self.context_limit,
            "diagnostic_full_source": diagnostic_full_source,
            "diagnostic_refill_selected": diagnostic_refill_selected,
            "source_cache_bytes": source_bytes, "prefill_seconds": prefill_seconds,
            "selection_seconds": selection_seconds, "roles": role_metrics,
            "selected_source_positions": selected,
            "selection_strategy": "speaker",
            "speaker_neighbor_blocks": self.config.speaker_neighbor_blocks,
            "seed_selected_positions": seed_selection,
            "baseline_selected_positions": baseline_positions,
            "review": review_result,
            "chunk_specialists": chunk_result,
            "chunk_specialists_by_variant": chunk_results,
        }
