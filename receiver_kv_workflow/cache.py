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
def generate_from_prefill(model, tokenizer, output, *, next_position, context_limit,
                          temperature=0.0, top_p=1.0):
    """Decode after a caller has already prefetched a complete receiver prompt."""
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tokenizer.eos_token_id)
    generated = []
    stopped = False
    remaining = context_limit - next_position
    if remaining < 1:
        raise ValueError("The prompt leaves no room for generated tokens in the model context.")
    for step in range(remaining):
        logits = output.logits[:, -1].float()
        if temperature == 0:
            token = logits.argmax(dim=-1).view(1, 1)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
            discard = (sorted_probs.cumsum(dim=-1) - sorted_probs) >= top_p
            sorted_probs = sorted_probs.masked_fill(discard, 0)
            token = sorted_ids.gather(-1, torch.multinomial(sorted_probs, 1))
        value = token.item()
        generated.append(value)
        if value in eos:
            stopped = True
            break
        if step + 1 < remaining:
            output = forward_tokens(model, token, output.past_key_values, next_position, logits_to_keep=1)
            next_position += 1
    # Count the generated sequence itself, not a re-tokenization of decoded text.
    content = generated[:-1] if stopped else generated
    terminal_tokens = len(generated) - len(content)
    counts = {'answer_tokens': len(content), 'terminal_tokens': terminal_tokens}
    return tokenizer.decode(content, skip_special_tokens=True), len(generated), not stopped, counts


@torch.inference_mode()
def generate(model, tokenizer, cache, suffix, *, absolute_start, context_limit,
             temperature=0.0, top_p=1.0):
    """Caller supplies a disposable cache. Never mutates the source cache."""
    if absolute_start + suffix.shape[1] >= context_limit:
        raise ValueError("The prompt leaves no room for generated tokens in the model context; no input was truncated.")
    output = forward_tokens(model, suffix, cache, absolute_start, logits_to_keep=1)
    answer, generated, cap_hit, counts = generate_from_prefill(
        model, tokenizer, output,
        next_position=absolute_start + suffix.shape[1], context_limit=context_limit,
        temperature=temperature, top_p=top_p,
    )
    return answer, generated, cap_hit, counts
