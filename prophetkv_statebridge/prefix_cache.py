"""Exact shared-prefix KV cache utilities for Hugging Face causal LMs.

Unlike the query-aware selector baseline, this module performs real cache
reuse: one immutable transcript prefix is prefetched once and cloned for each
specialist continuation. It intentionally does not concatenate unrelated KV
caches, which would be invalid without an engine-level cache-fusion method.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import torch


def clone_cache(cache):
    """Return a per-agent cache copy; DynamicCache grows in place during decoding."""
    return copy.deepcopy(cache)


def cache_nbytes(cache) -> int:
    """Best-effort cache-size accounting across legacy and DynamicCache layouts."""
    seen: set[int] = set()

    def tensors(value) -> Iterable[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
        elif hasattr(value, "keys") and hasattr(value, "values"):
            yield from tensors(value.keys)
            yield from tensors(value.values)
        elif hasattr(value, "layers"):
            yield from tensors(value.layers)

    total = 0
    for tensor in tensors(cache):
        storage_id = tensor.untyped_storage().data_ptr()
        if storage_id not in seen:
            seen.add(storage_id)
            total += tensor.numel() * tensor.element_size()
    return total


@torch.inference_mode()
def prefill_prefix(model, prefix_ids: torch.Tensor):
    """Prefill a shared prefix exactly once and return its reusable KV cache."""
    mask = torch.ones_like(prefix_ids)
    outputs = model(
        input_ids=prefix_ids,
        attention_mask=mask,
        use_cache=True,
        return_dict=True,
    )
    if outputs.past_key_values is None:
        raise RuntimeError("Model did not return past_key_values; KV reuse is unavailable.")
    return outputs.past_key_values


@torch.inference_mode()
def prefill_prefix_streaming(model, prefix_ids: torch.Tensor, *, chunk_tokens: int):
    """Build one exact prefix cache by appending chronological token chunks.

    ``chunk 2`` receives the cache produced by ``chunk 1``; no independent
    caches are concatenated.  When the complete prefix fits the model context,
    the result is the streaming counterpart of a single full-prefix prefill.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    if prefix_ids.ndim != 2 or prefix_ids.size(0) != 1:
        raise ValueError("prefix_ids must have shape [1, sequence_length]")

    cache = None
    total_tokens = prefix_ids.size(1)
    chunks = 0
    for start in range(0, total_tokens, chunk_tokens):
        end = min(start + chunk_tokens, total_tokens)
        current_ids = prefix_ids[:, start:end]
        outputs = model(
            input_ids=current_ids,
            attention_mask=torch.ones((1, end), device=prefix_ids.device, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        chunks += 1
    if cache is None:
        raise RuntimeError("Cannot prefill an empty prefix")
    return cache, chunks


@torch.inference_mode()
def generate_after_prefix_cache(
    model,
    tokenizer,
    prefix_cache,
    prefix_length: int,
    suffix_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily generate a specialist response after an exact cached prefix.

    The returned states correspond only to emitted answer tokens, so they can
    be passed directly to StateBridge without recomputing the long transcript.
    """
    cache = clone_cache(prefix_cache)
    device = suffix_ids.device
    suffix_length = suffix_ids.size(1)
    attention = torch.ones((1, prefix_length + suffix_length), device=device, dtype=torch.long)
    outputs = model(
        input_ids=suffix_ids,
        attention_mask=attention,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    cache = outputs.past_key_values
    next_token = outputs.logits[:, -1:].argmax(dim=-1)
    generated_ids: list[torch.Tensor] = []
    generated_states: list[torch.Tensor] = []
    eos_ids = {token_id for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token_id is not None}

    for step in range(max_new_tokens):
        current_id = next_token
        generated_ids.append(current_id)
        current_length = prefix_length + suffix_length + step + 1
        outputs = model(
            input_ids=current_id,
            attention_mask=torch.ones((1, current_length), device=device, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        generated_states.append(outputs.hidden_states[-1][:, -1, :])
        cache = outputs.past_key_values
        if int(current_id.item()) in eos_ids:
            break
        next_token = outputs.logits[:, -1:].argmax(dim=-1)

    return torch.cat(generated_ids, dim=1), torch.cat(generated_states, dim=0)
