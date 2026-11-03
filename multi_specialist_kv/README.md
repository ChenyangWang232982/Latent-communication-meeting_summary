# Experimental multi-specialist KV workflow

Run `python -m multi_specialist_kv.run --transcript input/meeting1.txt --output output/multi_kv/meeting1.txt --model Qwen/Qwen3-4B --device cuda --role-budget-ratio 0.25 --speaker-neighbor-blocks 1 --vector-routing`.

The existing router produces one selected source KV cache. Four specialists independently reuse clones of that same cache and emit JSON-style handoffs. Their role-prompt and generated-token KV segments, including each generated end-of-sequence token, are concatenated in role order; the final receiver sees those segments, not the generated handoff text. Each segment keeps its absolute RoPE positions and is assigned a distinct position range. Source KV itself is not passed to the final receiver. This concatenation is experimental: each segment was computed independently with the source, without attending to preceding specialist segments, and is not equivalent to a continuous forward pass.

The raw handoff text is saved separately for inspection, and JSON validity is reported, not guaranteed. Metrics record actual generated tokens, KV positions, bytes and stage times separately. The original absolute positions and prompt reserve count against model context; the workflow reports an error instead of truncating input when there is no room. No end-to-end speedup or factuality improvement is assumed.
