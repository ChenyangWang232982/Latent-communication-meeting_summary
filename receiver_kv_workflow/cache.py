"""Dense Qwen3 cache handling with original RoPE positions retained.

This does NOT splice independently computed agent caches. Every selected row
comes from one source prefill. Physical cache slots and absolute positions are
different after selection: continuation uses the original absolute position.
"""

import math
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


def generate_from_prefill(model, tokenizer, output, *, next_position, context_limit,
                          temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, on_progress=None, on_snapshot=None,
                          on_token=None):
    """Decode after a caller has already prefetched a complete receiver prompt."""
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError("repetition_penalty must be finite and >= 1")
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tokenizer.eos_token_id)
    generated = []
    stopped = False
    remaining = context_limit - next_position
    if remaining < 1:
        raise ValueError("The prompt leaves no room for generated tokens in the model context.")
    if on_progress is not None:
        on_progress({"phase": "decode", "tokens": 0})
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
    penalty = RepetitionPenaltyLogitsProcessor(float(repetition_penalty)) if repetition_penalty > 1 else None
    warpers = []
    if temperature > 0:
        from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
        warpers.append(TemperatureLogitsWarper(float(temperature)))
        if top_k > 0:
            warpers.append(TopKLogitsWarper(top_k))
        if top_p < 1:
            warpers.append(TopPLogitsWarper(float(top_p)))
    def snapshot(finished=False):
        if on_snapshot is None:
            return
        # Diagnostics only: never change logits, RNG state or stopping rules.
        tail = generated[-256:]
        repeated = any(len(tail) >= width * 3 and
                       tail[-width:] == tail[-2 * width:-width] == tail[-3 * width:-2 * width]
                       for width in range(1, min(64, len(tail) // 3) + 1))
        on_snapshot({
            "tokens": len(generated), "text": tokenizer.decode(generated, skip_special_tokens=True),
            "raw_tail": tokenizer.decode(tail, skip_special_tokens=False),
            "tail_token_ids": tail, "repeated_tail": repeated,
            "eos_token_ids": sorted(value for value in eos if value is not None),
            "finished": finished,
            "stop_reason": ("eos" if stopped else "context_limit") if finished else None,
        })
    for step in range(remaining):
        logits = output.logits[:, -1].float()
        # Penalize only this stage's generated history, never source/prompt IDs.
        # Clone to keep the raw-logit observer and model output unmodified.
        scores = logits
        if penalty is not None and generated:
            history = torch.tensor([generated], device=logits.device, dtype=torch.long)
            scores = penalty(history, logits.clone())
        if temperature == 0:
            token = scores.argmax(dim=-1).view(1, 1)
        else:
            for warper in warpers:
                scores = warper(None, scores)
            token = torch.multinomial(torch.softmax(scores, dim=-1), 1)
        value = token.item()
        if on_token is not None:
            on_token(logits, value, len(generated))
        generated.append(value)
        if on_progress is not None and len(generated) % 16 == 0:
            on_progress({"phase": "decode", "tokens": len(generated)})
        if value in eos:
            stopped = True
            break
        if on_snapshot is not None and (len(generated) == 16 or len(generated) % 256 == 0):
            snapshot()
        if step + 1 < remaining:
            output = forward_tokens(model, token, output.past_key_values, next_position, logits_to_keep=1)
            next_position += 1
    # Count the generated sequence itself, not a re-tokenization of decoded text.
    content = generated[:-1] if stopped else generated
    terminal_tokens = len(generated) - len(content)
    counts = {'answer_tokens': len(content), 'terminal_tokens': terminal_tokens}
    snapshot(finished=True)
    if on_progress is not None:
        on_progress({"phase": "decode", "tokens": len(generated), "finished": True,
                     "stop_reason": "eos" if stopped else "context_limit"})
    return tokenizer.decode(content, skip_special_tokens=True), len(generated), not stopped, counts


@torch.inference_mode()
def generate(model, tokenizer, cache, suffix, *, absolute_start, context_limit,
             temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, on_progress=None, on_snapshot=None, on_token=None):
    """Caller supplies a disposable cache. Never mutates the source cache."""
    if absolute_start + suffix.shape[1] >= context_limit:
        raise ValueError("The prompt leaves no room for generated tokens in the model context; no input was truncated.")
    if on_progress is not None:
        on_progress({"phase": "prompt_prefill", "prompt_tokens": suffix.shape[1]})
    output = forward_tokens(model, suffix, cache, absolute_start, logits_to_keep=1)
    answer, generated, cap_hit, counts = generate_from_prefill(
        model, tokenizer, output,
        next_position=absolute_start + suffix.shape[1], context_limit=context_limit,
        temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, on_progress=on_progress, on_snapshot=on_snapshot,
        on_token=on_token,
    )
    return answer, generated, cap_hit, counts
