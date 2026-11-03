"""Reusable inference engine for receiver-conditioned KV routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

from .prompts import (DECISION_EVOLUTION_REQUEST, EVENT_QUERY, ROLE_QUERIES, TIMELINE_REQUEST,
                      following_turn_suffix, prompt_parts, selected_evidence_excerpts,
                      specialist_sender_prompt, specialist_text_prompt, summary_grounded_request)
from .selection import select_spans, union_positions


@dataclass(frozen=True)
class ReceiverKVConfig:
    model: str = "Qwen/Qwen3-4B"
    device: str = "cuda"
    role_budget_ratio: float = 0.15
    span_tokens: int = 32
    query_batch_tokens: int = 8
    seed: int = 7
    speaker_neighbor_blocks: int = 1


class ReceiverKVRunner:
    """Load a homogeneous Qwen3 once, then run many meeting records."""

    def __init__(self, config: ReceiverKVConfig, progress: Callable[[str], None] | None = None) -> None:
        if not 0 < config.role_budget_ratio <= 1:
            raise ValueError("role_budget_ratio must be in (0, 1]")
        if min(config.span_tokens, config.query_batch_tokens) < 1:
            raise ValueError("Token budgets must be positive")
        if config.speaker_neighbor_blocks < 0:
            raise ValueError("speaker_neighbor_blocks must be non-negative")
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

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _tensor(self, ids):
        return self.torch.tensor([ids], device=self.device, dtype=self.torch.long)

    def run(self, record_id: str, transcript: str, *, variants: tuple[str, ...], timeline: bool = False,
            decision_evolution: bool = False, routing_audit: bool = False) -> dict:
        """Return transient answers plus auditable measurement fields for one record."""
        from .cache import forward_tokens, generate, select_cache, size_bytes
        from .scoring import query_scores

        if not transcript.strip():
            raise ValueError(f"Transcript is empty: {record_id}")
        requested = tuple(dict.fromkeys(variants))
        valid_variants = {"routed_kv", "specialist_text"}
        if not requested or any(name not in valid_variants for name in requested):
            raise ValueError("variants must be routed_kv and/or specialist_text")
        if (timeline or decision_evolution) and "routed_kv" not in requested:
            raise ValueError("timeline and decision evolution require the routed_kv summary")
        prefix, suffix, protected = prompt_parts(self.tokenizer, transcript)
        query_text = dict(ROLE_QUERIES)
        if timeline:
            query_text["events"] = EVENT_QUERY
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
            if timeline or requested:
                selection_start = time.perf_counter()
                for index, (role, query) in enumerate(queries.items(), start=1):
                    self.progress(f"{record_id}: selector {role} started ({index - 1}/{len(queries)} complete)")
                    scores = query_scores(self.model, source, self._tensor(query), query_batch=self.config.query_batch_tokens)
                    role_scores.append(scores)
                    selected = select_spans(scores, budget, span_size=self.config.span_tokens, protected=protected)
                    messages.append(selected)
                    role_metrics[role] = {"query_tokens": len(query), "selected_positions": len(selected)}
                    selection_probe_tokens += len(query)
                self._sync()
                selection_seconds = time.perf_counter() - selection_start
            selected = union_positions(messages)
            baseline_selection = list(selected)
            baseline_positions = len(selected)
            seed_selection = list(selected)
            if selected:
                from .speaker_blocks import expand_neighbor_blocks, select_complete_blocks
                boundary_start = time.perf_counter()
                selected = select_complete_blocks(speaker_blocks, selected, role_scores, protected)
                seed_selection = list(selected)
                if self.config.speaker_neighbor_blocks:
                    selected = expand_neighbor_blocks(
                        selected, speaker_blocks, protected, radius=self.config.speaker_neighbor_blocks
                    )
                selection_seconds += time.perf_counter() - boundary_start + block_seconds
                detail = f" seeds={len(seed_selection)}, +/-{self.config.speaker_neighbor_blocks} blocks"
                self.progress(f"{record_id}: speaker KV selection {len(selected)}/{baseline_positions} positions{detail}")
            answers, measurements = {}, {}
            if "routed_kv" in requested:
                receiver_start = time.perf_counter()
                ids = selected
                incoming = select_cache(source, ids)
                absolute_start, payload_bytes = len(prefix), size_bytes(incoming)
                incoming_bytes = size_bytes(incoming)
                answer, generated_tokens, output_cap_hit = generate(
                    self.model, self.tokenizer, incoming, self._tensor(suffix), absolute_start=absolute_start,
                    context_limit=self.context_limit,
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
                    "token_accounting": {
                        "source_prefill_tokens": len(prefix),
                        "selection_probe_tokens": selection_probe_tokens,
                        "specialist_prompt_tokens": 0,
                        "specialist_handoff_tokens": 0,
                        "receiver_prompt_tokens": len(suffix),
                        "receiver_output_tokens": generated_tokens,
                    },
                }
            if "specialist_text" in requested:
                self.progress(f"{record_id}: specialist_text sender started")
                excerpts, evidence_tokens, excerpt_count = selected_evidence_excerpts(self.tokenizer, transcript, selected)
                sender_prompt = self._tensor(specialist_sender_prompt(self.tokenizer, excerpts))
                if sender_prompt.shape[1] >= self.context_limit:
                    raise ValueError(f"{record_id}: specialist text sender prompt exhausts the model context limit.")
                specialist_start = time.perf_counter()
                handoff, handoff_tokens, _ = generate(
                    self.model, self.tokenizer, None, sender_prompt,
                    absolute_start=0, context_limit=self.context_limit,
                )
                self._sync()
                specialist_seconds = time.perf_counter() - specialist_start
                receiver_prompt = self._tensor(specialist_text_prompt(self.tokenizer, handoff))
                if receiver_prompt.shape[1] >= self.context_limit:
                    raise ValueError(
                        f"{record_id}: specialist handoff leaves no room for output in the model context; no input was truncated."
                    )
                receiver_start = time.perf_counter()
                answer, generated_tokens, output_cap_hit = generate(
                    self.model, self.tokenizer, None, receiver_prompt, absolute_start=0,
                    context_limit=self.context_limit,
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
                    "specialist_count": 1,
                    "specialist_seconds": specialist_seconds,
                    "receiver_seconds": receiver_seconds,
                    "pipeline_seconds_excluding_model_load": prefill_seconds + selection_seconds + specialist_seconds + receiver_seconds,
                    "generated_tokens": generated_tokens,
                    "output_cap_hit": output_cap_hit,
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
            derived_answers, derived_metrics = {}, {}
            for name, enabled, request in (
                ("timeline", timeline, TIMELINE_REQUEST),
                ("decision_evolution", decision_evolution, DECISION_EVOLUTION_REQUEST),
            ):
                if not enabled:
                    continue
                self.progress(f"{record_id}: {name} generation started using routed KV and summary")
                grounded_request = summary_grounded_request(answers["routed_kv"], request)
                derived_suffix = following_turn_suffix(self.tokenizer, transcript, grounded_request)
                if len(prefix) + len(derived_suffix) >= self.context_limit:
                    raise ValueError(
                        f"{record_id}: {name} context (source KV, summary and request) leaves no room "
                        f"for output in the model context; no text was truncated."
                    )
                self._sync()
                started = time.perf_counter()
                derived_answer, generated, cap_hit = generate(
                    self.model, self.tokenizer, select_cache(source, selected), self._tensor(derived_suffix),
                    absolute_start=len(prefix),
                    context_limit=self.context_limit,
                )
                self._sync()
                derived_answers[name] = derived_answer
                derived_metrics[name] = {"prompt_tokens": len(derived_suffix), "output_tokens": generated,
                                         "output_cap_hit": cap_hit, "seconds": time.perf_counter() - started}
        audit = None
        if routing_audit:
            from .routing_audit import build_audit
            audit = build_audit(self.tokenizer, transcript, prefix, protected, selected, baseline_selection,
                                dict(zip(queries, messages)), "speaker",
                                seed_selection=seed_selection,
                                neighbor_blocks=self.config.speaker_neighbor_blocks)
        return {
            "routing_audit": audit,
            "id": record_id, "answers": answers, "variants": measurements,
            "source_positions": len(prefix), "context_limit": self.context_limit,
            "source_cache_bytes": source_bytes, "prefill_seconds": prefill_seconds,
            "selection_seconds": selection_seconds, "roles": role_metrics,
            "selected_source_positions": selected,
            "selection_strategy": "speaker",
            "speaker_neighbor_blocks": self.config.speaker_neighbor_blocks,
            "seed_selected_positions": seed_selection,
            "baseline_selected_positions": baseline_positions,
            "timeline": derived_answers.get("timeline"),
            "decision_evolution": derived_answers.get("decision_evolution"),
            "derived_metrics": derived_metrics,
        }
