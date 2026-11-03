"""Dense Qwen3 cache handling with original RoPE positions retained.

This does NOT splice independently computed agent caches. Every selected row
comes from one source prefill. Physical cache slots and absolute positions are
different after selection: continuation uses the original absolute position.
"""

import hashlib
import torch
from transformers.cache_utils import DynamicCache


def token_fingerprint(token_ids):
    """Hash token IDs for comparing calls without recording meeting text."""
    data = ','.join(str(int(token)) for token in token_ids).encode('ascii')
    return hashlib.sha256(data).hexdigest()


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
def generate_from_prefill(model, tokenizer, output, *, next_position, context_limit, enable_thinking=False):
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
        token = output.logits[:, -1].argmax(dim=-1).view(1, 1)
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
    delimiter_tokens = 0
    reasoning_tokens = 0
    thinking_complete = True
    if enable_thinking:
        marker = tokenizer.encode('</think>', add_special_tokens=False)
        if not marker:
            raise ValueError('Tokenizer does not encode the thinking delimiter.')
        boundary = next((i for i in range(len(content) - len(marker) + 1)
                         if content[i:i + len(marker)] == marker), None)
        thinking_complete = boundary is not None
        if thinking_complete:
            reasoning_tokens = boundary
            delimiter_tokens = len(marker)
            content = content[boundary + len(marker):]
        else:
            # An unfinished thinking section is not a final summary.
            reasoning_tokens = len(content)
            content = []
    counts = {'thinking_tokens': reasoning_tokens, 'answer_tokens': len(content),
              'delimiter_tokens': delimiter_tokens, 'terminal_tokens': terminal_tokens,
              'thinking_complete': thinking_complete,
              'generated_token_sha256': token_fingerprint(generated)}
    return tokenizer.decode(content, skip_special_tokens=True), len(generated), not stopped, counts


@torch.inference_mode()
def generate(model, tokenizer, cache, suffix, *, absolute_start, context_limit, enable_thinking=False):
    """Caller supplies a disposable cache. Never mutates the source cache."""
    if absolute_start + suffix.shape[1] >= context_limit:
        raise ValueError("The prompt leaves no room for generated tokens in the model context; no input was truncated.")
    output = forward_tokens(model, suffix, cache, absolute_start, logits_to_keep=1)
    return generate_from_prefill(
        model, tokenizer, output,
        next_position=absolute_start + suffix.shape[1], context_limit=context_limit,
        enable_thinking=enable_thinking,
    )
