"""Reusable inference engine for receiver-conditioned KV routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

from .prompts import ROLE_HANDOFF_REQUESTS, ROLE_QUERIES, following_turn_suffix, prompt_parts, specialist_text_prompt
from .selection import select_spans, union_positions


@dataclass(frozen=True)
class ReceiverKVConfig:
    model: str = "Qwen/Qwen3-4B"
    device: str = "cuda"
    role_budget_ratio: float = 0.15
    span_tokens: int = 32
    query_batch_tokens: int = 8
    max_new_tokens: int = 1024
    specialist_text_max_new_tokens: int = 384
    seed: int = 7


class ReceiverKVRunner:
    """Load a homogeneous Qwen3 once, then run many meeting records."""

    def __init__(self, config: ReceiverKVConfig, progress: Callable[[str], None] | None = None) -> None:
        if not 0 < config.role_budget_ratio <= 1:
            raise ValueError("role_budget_ratio must be in (0, 1]")
        if min(config.span_tokens, config.query_batch_tokens, config.max_new_tokens, config.specialist_text_max_new_tokens) < 1:
            raise ValueError("Token budgets must be positive")
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

    def run(self, record_id: str, transcript: str, *, variants: tuple[str, ...]) -> dict:
        """Return transient answers plus auditable measurement fields for one record."""
        from .cache import forward_tokens, generate, select_cache, size_bytes
        from .scoring import query_scores

        if not transcript.strip():
            raise ValueError(f"Transcript is empty: {record_id}")
        requested = tuple(dict.fromkeys(variants))
        valid_variants = {"routed_kv", "full_kv", "selected_text", "specialist_text"}
        if not requested or any(name not in valid_variants for name in requested):
            raise ValueError("variants must be routed_kv, full_kv, selected_text and/or specialist_text")
        prefix, suffix, protected = prompt_parts(self.tokenizer, transcript)
        queries = {role: self.tokenizer.encode(query, add_special_tokens=False) for role, query in ROLE_QUERIES.items()}
        specialist_suffixes = (
            [following_turn_suffix(self.tokenizer, transcript, request) for request in ROLE_HANDOFF_REQUESTS.values()]
            if "specialist_text" in requested else []
        )
        required = len(prefix) + max(
            len(suffix) + self.config.max_new_tokens,
            max(map(len, queries.values())),
            *(len(role_suffix) + self.config.specialist_text_max_new_tokens for role_suffix in specialist_suffixes),
        )
        if required > self.context_limit:
            raise ValueError(
                f"{record_id}: context requires {required} positions but model supports {self.context_limit}; no text was truncated."
            )
        budget = max(len(protected), math.ceil(len(prefix) * self.config.role_budget_ratio))
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
            selection_seconds = 0.0
            selection_probe_tokens = 0
            if any(name in {"routed_kv", "selected_text"} for name in requested):
                selection_start = time.perf_counter()
                for index, (role, query) in enumerate(queries.items(), start=1):
                    self.progress(f"{record_id}: selector {role} started ({index - 1}/4 complete)")
                    scores = query_scores(self.model, source, self._tensor(query), query_batch=self.config.query_batch_tokens)
                    selected = select_spans(scores, budget, span_size=self.config.span_tokens, protected=protected)
                    messages.append(selected)
                    role_metrics[role] = {"query_tokens": len(query), "selected_positions": len(selected)}
                    selection_probe_tokens += len(query)
                self._sync()
                selection_seconds = time.perf_counter() - selection_start
            selected = union_positions(messages)
            answers, measurements = {}, {}
            for variant in (name for name in requested if name != "specialist_text"):
                receiver_start = time.perf_counter()
                ids = list(range(len(prefix))) if variant == "full_kv" else selected
                if variant == "selected_text":
                    output = forward_tokens(self.model, self._tensor([prefix[i] for i in ids]), None, 0, logits_to_keep=1)
                    incoming = output.past_key_values
                    del output
                    absolute_start, payload_bytes = len(ids), len(ids) * 4
                else:
                    incoming = select_cache(source, ids)
                    absolute_start, payload_bytes = len(prefix), size_bytes(incoming)
                incoming_bytes = size_bytes(incoming)
                answer, generated_tokens, output_cap_hit = generate(
                    self.model, self.tokenizer, incoming, self._tensor(suffix), absolute_start=absolute_start,
                    max_new_tokens=self.config.max_new_tokens, context_limit=self.context_limit,
                )
                del incoming
                self._sync()
                receiver_seconds = time.perf_counter() - receiver_start
                answers[variant] = answer
                measurements[variant] = {
                    "incoming_positions": len(ids), "incoming_cache_bytes": incoming_bytes,
                    "logical_payload_bytes": payload_bytes,
                    "positions_vs_full_percent": 100 * (len(ids) - len(prefix)) / len(prefix),
                    "cache_bytes_vs_full_percent": 100 * (incoming_bytes - source_bytes) / source_bytes,
                    "receiver_seconds": receiver_seconds,
                    "pipeline_seconds_excluding_model_load": prefill_seconds + receiver_seconds + (
                        0 if variant == "full_kv" else selection_seconds),
                    "generated_tokens": generated_tokens, "output_cap_hit": output_cap_hit,
                    "token_accounting": {
                        "source_prefill_tokens": len(prefix),
                        "selection_probe_tokens": selection_probe_tokens if variant != "full_kv" else 0,
                        "specialist_prompt_tokens": 0,
                        "specialist_handoff_tokens": 0,
                        "receiver_prompt_tokens": len(ids) + len(suffix) if variant == "selected_text" else len(suffix),
                        "receiver_output_tokens": generated_tokens,
                    },
                }
            if "specialist_text" in requested:
                handoffs, handoff_tokens, specialist_seconds = {}, 0, 0.0
                specialist_prompt_tokens = sum(len(role_suffix) for role_suffix in specialist_suffixes)
                for index, ((role, _), suffix_ids) in enumerate(zip(ROLE_HANDOFF_REQUESTS.items(), specialist_suffixes), start=1):
                    self.progress(f"{record_id}: specialist_text {role} started ({index - 1}/4 complete)")
                    suffix = self._tensor(suffix_ids)
                    specialist_start = time.perf_counter()
                    handoff, generated_tokens, _ = generate(
                        self.model, self.tokenizer, select_cache(source, range(len(prefix))), suffix,
                        absolute_start=len(prefix), max_new_tokens=self.config.specialist_text_max_new_tokens,
                        context_limit=self.context_limit,
                    )
                    self._sync()
                    specialist_seconds += time.perf_counter() - specialist_start
                    handoffs[role] = handoff
                    handoff_tokens += generated_tokens
                receiver_prompt = self._tensor(specialist_text_prompt(self.tokenizer, handoffs))
                if receiver_prompt.shape[1] + self.config.max_new_tokens > self.context_limit:
                    raise ValueError(
                        f"{record_id}: specialist handoffs plus output reserve exceed the model context limit; no input was truncated."
                    )
                receiver_start = time.perf_counter()
                answer, generated_tokens, output_cap_hit = generate(
                    self.model, self.tokenizer, None, receiver_prompt, absolute_start=0,
                    max_new_tokens=self.config.max_new_tokens, context_limit=self.context_limit,
                )
                self._sync()
                receiver_seconds = time.perf_counter() - receiver_start
                answers["specialist_text"] = answer
                measurements["specialist_text"] = {
                    "incoming_positions": receiver_prompt.shape[1],
                    "incoming_cache_bytes": 0,
                    "logical_payload_bytes": handoff_tokens * 4,
                    "positions_vs_full_percent": None,
                    "cache_bytes_vs_full_percent": None,
                    "specialist_handoff_tokens": handoff_tokens,
                    "specialist_count": len(handoffs),
                    "specialist_seconds": specialist_seconds,
                    "receiver_seconds": receiver_seconds,
                    "pipeline_seconds_excluding_model_load": prefill_seconds + specialist_seconds + receiver_seconds,
                    "generated_tokens": generated_tokens,
                    "output_cap_hit": output_cap_hit,
                    "token_accounting": {
                        "source_prefill_tokens": len(prefix),
                        "selection_probe_tokens": 0,
                        "specialist_prompt_tokens": specialist_prompt_tokens,
                        "specialist_handoff_tokens": handoff_tokens,
                        "receiver_prompt_tokens": receiver_prompt.shape[1],
                        "receiver_output_tokens": generated_tokens,
                    },
                }
        return {
            "id": record_id, "answers": answers, "variants": measurements,
            "source_positions": len(prefix), "context_limit": self.context_limit,
            "source_cache_bytes": source_bytes, "prefill_seconds": prefill_seconds,
            "selection_seconds": selection_seconds, "roles": role_metrics,
            "selected_source_positions": selected,
        }
