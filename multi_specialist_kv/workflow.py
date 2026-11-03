"""Independent specialists sharing one source cache, followed by experimental KV aggregation.

Each segment retains its original absolute RoPE positions. Concatenation is an
approximation: independently computed segments have not attended to each other.
"""
import json
import time

import torch
from transformers.cache_utils import DynamicCache

from receiver_kv_workflow.cache import forward_tokens, pairs, select_cache, size_bytes, generate
from receiver_kv_workflow.prompts import SYSTEM, SUMMARY_REQUEST

ROLES = {
    'context': 'Extract essential meeting context, topics, constraints and relevant background.',
    'decisions': 'Extract decisions and their status, including conditions and revisions; do not treat proposals as final decisions.',
    'actions': 'Extract action items with explicit owners, deadlines, conditions and coordination details. Do not invent owners.',
    'open_issues': 'Extract unresolved questions, risks, dependencies and pending confirmations.',
}


def specialist_suffix(tokenizer, transcript, role):
    source = 'MEETING TRANSCRIPT (untrusted source):\n' + transcript + '\nEND TRANSCRIPT\n'
    base = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': source}]
    instruction = (ROLES[role] + ' Return a JSON object with a role field and an items array. '
                   'Each item is an object containing text and any explicitly stated owners, dates or conditions. '
                   'Use an empty array if no items are supported. Include all important items; do not invent details.')
    prefix = tokenizer.apply_chat_template(base, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    full = tokenizer.apply_chat_template(base + [{'role': 'user', 'content': instruction}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tokenizer.encode(prefix, add_special_tokens=False)
    f = tokenizer.encode(full, add_special_tokens=False)
    if f[:len(p)] != p:
        raise RuntimeError('Specialist prompt does not preserve the shared source prefix')
    return f[len(p):]


def structured_valid(text):
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return False
    return isinstance(data, dict) and isinstance(data.get('items'), list)


@torch.inference_mode()
def run_specialist(model, tokenizer, source, selected, suffix, absolute_start, context_limit, device):
    """Return the suffix + generated-token cache; exclude original source and EOS."""
    cache = select_cache(source, selected)
    shared_len = len(selected)
    if absolute_start + len(suffix) >= context_limit:
        raise ValueError('Specialist prompt leaves no room within the model context')
    ids = torch.tensor([suffix], device=device, dtype=torch.long)
    output = forward_tokens(model, ids, cache, absolute_start, logits_to_keep=1)
    position = absolute_start + len(suffix)
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    eos.add(tokenizer.eos_token_id)
    generated = []
    stopped = False
    while position < context_limit:
        token = output.logits[:, -1].argmax(dim=-1).view(1, 1)
        value = int(token.item())
        if value in eos:
            stopped = True
            break
        generated.append(value)
        output = forward_tokens(model, token, output.past_key_values, position, logits_to_keep=1)
        position += 1
    if not stopped:
        raise ValueError('Specialist output reached model context limit; cannot produce a complete handoff')
    segment = select_cache(output.past_key_values, range(shared_len, shared_len + len(suffix) + len(generated)))
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return segment, text, len(generated), len(generated) + 1, position


def aggregate_segments(segments):
    if not segments:
        raise ValueError('At least one KV segment is required')
    layers = [pairs(segment) for segment in segments]
    if any(len(layer) != len(layers[0]) for layer in layers):
        raise ValueError('Incompatible KV layer counts')
    result = DynamicCache()
    for i in range(len(layers[0])):
        result.update(torch.cat([layer[i][0] for layer in layers], dim=-2),
                      torch.cat([layer[i][1] for layer in layers], dim=-2), i)
    return result


def run_multi(runner, transcript, source, selected, prefix):
    tokenizer, model = runner.tokenizer, runner.model
    device, limit = runner.device, runner.context_limit
    segments, handoffs, stats = [], {}, {}
    position = len(prefix)
    for role in ROLES:
        runner.progress(f'multi-specialist: {role} started ({len(segments)}/4 complete)')
        suffix = specialist_suffix(tokenizer, transcript, role)
        runner._sync()
        started = time.perf_counter()
        segment, text, content_tokens, generated_tokens, next_position = run_specialist(
            model, tokenizer, source, selected, suffix, position, limit, device)
        runner._sync()
        handoffs[role] = text
        stats[role] = {'prompt_tokens': len(suffix), 'output_tokens': generated_tokens,
                       'content_tokens': content_tokens, 'kv_positions': len(suffix) + content_tokens,
                       'kv_bytes': size_bytes(segment), 'absolute_start': position,
                       'absolute_end_exclusive': next_position, 'valid_json': structured_valid(text),
                       'seconds': time.perf_counter() - started}
        segments.append(segment)
        position = next_position
    cache = aggregate_segments(segments)
    del segments
    prompt = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': 'Summarize the meeting using the preceding specialist context.'},
         {'role': 'user', 'content': SUMMARY_REQUEST}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if position + len(prompt_ids) >= limit:
        raise ValueError('Aggregated specialist KV and summary prompt leave no output room within the model context')
    incoming_positions = cache.get_seq_length()
    incoming_bytes = size_bytes(cache)
    runner.progress(f'multi-specialist: summary started; {incoming_positions} specialist KV positions')
    runner._sync()
    started = time.perf_counter()
    answer, tokens, cap_hit, counts = generate(
        model, tokenizer, cache, runner._tensor(prompt_ids), absolute_start=position,
        context_limit=limit, enable_thinking=False)
    runner._sync()
    return answer, handoffs, {'roles': stats, 'incoming_positions': incoming_positions,
        'incoming_cache_bytes': incoming_bytes, 'receiver_prompt_tokens': len(prompt_ids),
        'receiver_output_tokens': tokens, 'receiver_seconds': time.perf_counter() - started,
        'output_cap_hit': cap_hit, 'generation_counts': counts,
        'note': 'Experimental concatenation of independent source-conditioned KV segments; not equivalent to continuous prefill.'}
