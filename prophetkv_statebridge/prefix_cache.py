"""Exact shared-prefix KV cache utilities for Hugging Face causal LMs.

Unlike the query-aware selector baseline, this module performs real cache
reuse: one immutable transcript prefix is prefetched once and cloned for each
specialist continuation. It intentionally does not concatenate unrelated KV
caches, which would be invalid without an engine-level cache-fusion method.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from collections.abc import Callable

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
def generate_after_prefix_cache(
    model,
    tokenizer,
    prefix_cache,
    prefix_length: int,
    suffix_ids: torch.Tensor,
    *,
    max_context_tokens: int,
    max_new_tokens: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
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

    remaining_tokens = max_context_tokens - prefix_length - suffix_length
    if remaining_tokens < 1:
        raise RuntimeError("The specialist prompt leaves no generation space in the model context window.")
    if max_new_tokens is not None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        remaining_tokens = min(remaining_tokens, max_new_tokens)
    for step in range(remaining_tokens):
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
        if progress_callback is not None and (step + 1) % 128 == 0:
            progress_callback(step + 1)
        if int(current_id.item()) in eos_ids:
            break
        next_token = outputs.logits[:, -1:].argmax(dim=-1)

    return torch.cat(generated_ids, dim=1), torch.cat(generated_states, dim=0)


@torch.inference_mode()
def generate_after_embedding_prefix(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    embedding_prefix: torch.Tensor,
    *,
    max_context_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate after a continuous prefix and retain states for a later bridge.

    ``embedding_prefix`` is a StateBridge message rather than a token cache.  The
    emitted states are therefore captured during decoding, which lets a block
    receiver become the sender for the final document-level merger.
    """
    embedding_layer = model.get_input_embeddings()
    prompt_embeddings = embedding_layer(prompt_ids)
    inputs_embeds = torch.cat([embedding_prefix.unsqueeze(0), prompt_embeddings], dim=1)
    device = inputs_embeds.device
    prefix_length = inputs_embeds.size(1)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones((1, prefix_length), device=device, dtype=torch.long),
        use_cache=True,
        return_dict=True,
    )
    cache = outputs.past_key_values
    next_token = outputs.logits[:, -1:].argmax(dim=-1)
    generated_ids: list[torch.Tensor] = []
    generated_states: list[torch.Tensor] = []
    eos_ids = {token_id for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token_id is not None}

    remaining_tokens = max_context_tokens - prefix_length
    if remaining_tokens < 1:
        raise RuntimeError("The continuous prefix and prompt fill the model context window.")
    for step in range(remaining_tokens):
        current_id = next_token
        generated_ids.append(current_id)
        outputs = model(
            input_ids=current_id,
            attention_mask=torch.ones((1, prefix_length + step + 1), device=device, dtype=torch.long),
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
