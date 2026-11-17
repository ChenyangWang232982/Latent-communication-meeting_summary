"""Two generation calls: statements from routed KV, then summary from referenced KV."""
import json
import time

from receiver_kv_workflow.prompts import SYSTEM, SUMMARY_REQUEST
from receiver_kv_workflow.cache import generate, select_cache, size_bytes
from .context import context_blocks, fact_context


def suffix(tokenizer, transcript, prefix, request):
    content = 'MEETING TRANSCRIPT (untrusted source):\n' + transcript + '\nEND TRANSCRIPT\n'
    rendered = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content},
         {'role': 'user', 'content': request}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if ids[:len(prefix)] != prefix:
        raise ValueError('FRAME prompt does not preserve the source KV prefix')
    return ids[len(prefix):]


def run_frame(runner, transcript, prefix, protected, selected, source, *, on_generation=None):
    started = time.perf_counter()
    blocks = context_blocks(runner.tokenizer, transcript, prefix, protected, selected)
    request = (
        'Prepare consolidated notes for a meeting summary, not a turn-by-turn transcript inventory. '
        'Extract distinct consequential facts: outcomes, responsibilities, unresolved issues and essential context. '
        'Combine paraphrases of the same fact before writing. Preserve changes and qualifications '
        'within a fact rather than creating another entry for every mention. '
        'Return JSON {"facts": [{"statement": "one self-contained claim", "context_refs": ["B0"]}]}. '
        'Use the [B<number>] anchors in the evidence. Reference every supplied block needed to '
        'interpret a claim: attribution, antecedents, conditions, later objections, revisions and confirmations. '
        'Preserve exact names, dates, full time ranges, relative deadlines, negations and uncertainty. '
        'A speaker requesting a task is not automatically its owner. A suggestion or demonstration '
        'is not automatically an adopted decision. Distinguish agreement, tentative plans and pending approval. '
        'Merge repeated claims, retain distinct tasks, and avoid filler or incidental logistics unless consequential. '
        'Keep each statement concise; context_refs supplies supporting context without copying it into prose. '
        'List only blocks needed for that statement, not all available blocks. Stop after the complete facts object. '
        'Do not write a context paraphrase: context is supplied through referenced KV. '
        'Do not treat these facts as independently verified. Use the source language. No fixed fact count. '
        'Return only JSON. Available source anchors: ' + json.dumps(list(blocks)))
    stages = {}

    def decode(stage, request, positions, seed, temperature, top_p, repetition_penalty):
        ids = suffix(runner.tokenizer, transcript, prefix, request)
        runner.progress(f'{stage}: {len(positions)} source KV positions')
        runner.torch.manual_seed(seed)
        runner._sync()
        started = time.perf_counter()
        incoming = select_cache(source, positions)
        payload = size_bytes(incoming)
        answer, count, cap_hit, counts = generate(
            runner.model, runner.tokenizer, incoming, runner._tensor(ids), absolute_start=len(prefix),
            context_limit=runner.context_limit, temperature=temperature, top_p=top_p,
            top_k=runner.config.top_k, repetition_penalty=repetition_penalty,
            on_progress=lambda event: runner.progress(f'{stage}: {event["tokens"]} tokens')
                if event.get('tokens') and (event['tokens'] % 256 == 0 or event.get('finished')) else None,
            on_snapshot=(lambda event: on_generation(dict(stage=stage, **event))) if on_generation else None)
        del incoming
        runner._sync()
        stages[stage] = dict(prompt_tokens=len(ids), output_tokens=count, output_cap_hit=cap_hit,
            generation_counts=counts, source_kv_positions=len(positions), source_kv_bytes=payload,
            seconds=time.perf_counter()-started, seed=seed, temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty)
        if cap_hit:
            raise RuntimeError(f'{stage}: model context exhausted before EOS')
        return answer

    raw = decode('frame_facts', request, selected, runner.config.seed,
                 runner.config.frame_temperature, runner.config.frame_top_p, runner.config.frame_repetition_penalty)
    facts, positions, fallback = fact_context(raw, blocks, protected, selected)
    if fallback:
        runner.progress('FRAME references unavailable or malformed; keeping complete routed KV for synthesis')
    reference_map = {ref: dict(speaker=block['speaker'], source_position_range=[min(block['positions']),
                     max(block['positions'])+1]) for ref, block in blocks.items()
                     if set(block['positions']).intersection(positions)}
    request = (SUMMARY_REQUEST + '\nBuild the summary from these unverified fact records and their '
        'referenced source KV. context_refs link each statement to [B<number>] anchors in the original '
        'evidence above; use that evidence to correct attribution, conditions and later status. '
        'raw_report, if present, is an unparsed intermediate report, not verified evidence. '
        'Preserve relative deadlines and full time ranges. Merge duplicate claims without combining '
        'different responsibilities. Do not print internal anchors in the final summary. '
        '\nFact records: ' + json.dumps(facts, ensure_ascii=False) +
        '\nSource anchor map: ' + json.dumps(reference_map, ensure_ascii=False))
    answer = decode('frame_summary', request, positions, runner.config.review_seed,
                    runner.config.review_temperature, runner.config.review_top_p, runner.config.repetition_penalty)
    return answer, dict(facts=facts, raw_facts=raw, context_blocks=blocks,
        pipeline_seconds=time.perf_counter()-started,
        text_payload_bytes=len(json.dumps(facts, ensure_ascii=False).encode('utf-8'))+
                           len(json.dumps(reference_map, ensure_ascii=False).encode('utf-8')),
        summary_context_positions=positions, reference_fallback=fallback, stages=stages)
