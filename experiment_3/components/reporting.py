"""Stage costs, without interpreting logical tokens as attention compute or billing."""


def measurement(result, detail, *, extra_seconds=0, extra_encoder_tokens=0):
    stages = list(detail['stages'].values())
    local, final = stages[:-1], stages[-1]
    inputs = result['source_positions'] + sum(r['query_tokens'] for r in result['roles'].values())
    inputs += sum(s['prompt_tokens'] for s in stages)
    outputs = sum(s['output_tokens'] for s in stages)
    shared_encoder = (result.get('vector_routing') or {}).get('embedding_input_tokens', 0)
    return dict(logical_input_tokens=inputs, logical_output_tokens=outputs,
        logical_total_tokens=inputs + outputs,
        standalone_pipeline_seconds=result['prefill_seconds'] + result['selection_seconds'] +
            detail['pipeline_seconds'] + extra_seconds,
        additional_embedding_input_tokens=extra_encoder_tokens,
        total_embedding_input_tokens=shared_encoder + extra_encoder_tokens,
        local_kv_position_events=sum(s['source_kv_positions'] for s in local),
        final_kv_positions=final['source_kv_positions'],
        kv_payload_bytes=sum(s['source_kv_bytes'] for s in stages),
        final_output_tokens=final['output_tokens'])
