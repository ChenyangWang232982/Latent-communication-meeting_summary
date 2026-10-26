"""Dense Qwen3 cache handling with original RoPE positions retained.

This does NOT splice independently computed agent caches. Every selected row
comes from one source prefill. Physical cache slots and absolute positions are
different after selection: continuation uses the original absolute position.
"""

import torch
from transformers.cache_utils import DynamicCache


def pairs(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if isinstance(cache, (tuple, list)):
        return list(cache)
    raise TypeError(f"Unsupported cache: {type(cache).__name__}")


def size_bytes(cache):
    return sum(t.numel() * t.element_size() for pair in pairs(cache) for t in pair)


def select_cache(cache, indices):
    result = DynamicCache()
    for layer, (key, value) in enumerate(pairs(cache)):
        ids = torch.as_tensor(indices, dtype=torch.long, device=key.device)
        result.update(key.index_select(-2, ids), value.index_select(-2, ids), layer)
    return result


def forward_tokens(model, ids, cache, absolute_start, **kwargs):
    physical = 0 if cache is None else cache.get_seq_length()
    positions = torch.arange(absolute_start, absolute_start + ids.shape[1], device=ids.device)
    return model(
        input_ids=ids, past_key_values=cache,
        attention_mask=torch.ones((1, physical + ids.shape[1]), device=ids.device, dtype=torch.long),
        position_ids=positions.unsqueeze(0), use_cache=True, return_dict=True,
        **kwargs,
    )


@torch.inference_mode()
def generate_from_prefill(model, tokenizer, output, *, next_position, max_new_tokens):
    """Decode after a caller has already prefetched a complete receiver prompt."""
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tokenizer.eos_token_id)
    generated = []
    stopped = False
    for step in range(max_new_tokens):
        token = output.logits[:, -1].argmax(dim=-1).view(1, 1)
        value = token.item()
        generated.append(value)
        if value in eos:
            stopped = True
            break
        if step + 1 < max_new_tokens:
            output = forward_tokens(model, token, output.past_key_values, next_position, logits_to_keep=1)
            next_position += 1
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated), not stopped


@torch.inference_mode()
def generate(model, tokenizer, cache, suffix, *, absolute_start, max_new_tokens, context_limit):
    """Caller supplies a disposable cache. Never mutates the source cache."""
    if absolute_start + suffix.shape[1] + max_new_tokens > context_limit:
        raise ValueError("The context plus output reserve exceeds the model context limit; no input was truncated.")
    output = forward_tokens(model, suffix, cache, absolute_start, logits_to_keep=1)
    return generate_from_prefill(
        model, tokenizer, output,
        next_position=absolute_start + suffix.shape[1], max_new_tokens=max_new_tokens,
    )
